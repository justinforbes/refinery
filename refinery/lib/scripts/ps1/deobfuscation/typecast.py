"""
PowerShell type cast simplification transforms.
"""
from __future__ import annotations

import string

from refinery.lib.scripts import Node, Transformer, canonical
from refinery.lib.scripts.ps1.analysis.cache import model_cache
from refinery.lib.scripts.ps1.analysis.dataflow import Ps1VariableFlow
from refinery.lib.scripts.ps1.analysis.separator import coerced_text_at
from refinery.lib.scripts.ps1.analysis.values import (
    Ps1Outcome,
    collect_integers,
    convert,
    make_string_literal,
    read,
    render,
    text_of,
)
from refinery.lib.scripts.ps1.data import named_type, resolve_type
from refinery.lib.scripts.ps1.deobfuscation.helpers import unwrap_single_paren
from refinery.lib.scripts.ps1.model import (
    Expression,
    Ps1BinaryExpression,
    Ps1CastExpression,
    Ps1Script,
    Ps1TypeExpression,
)

#: The three targets this pass treats as something other than a value conversion: one that names a
#: type rather than producing a value of one, and two whose answer the conversion grid does not
#: carry, because it was captured over scalar targets only. See the arm each belongs to.
_TYPE = named_type('System.Type')
_STRING = named_type('System.String')
_CHAR_ARRAY = named_type('char[]')


class Ps1TypeCasts(Transformer):
    """
    A cast written back as the value it produces, asked of
    `refinery.lib.scripts.ps1.analysis.values` rather than decided here.

    What stood here was a table of integer ranges and a dispatch on the accelerator's spelling, and
    it lost a type on every fold whose target the language spells no literal for: `[byte] 5` became
    `5`, which re-reads as an Int32. It read a String operand with Python's own `int(text, 0)`,
    which knows a digit separator and the `0b` and `0o` prefixes that 5.1 has never had, so
    `[int] '1_0'` answered ten for a script that stops. And it folded a `[char]` to a one-character
    String, which is a different value from a Char and not merely a different type.

    `-as` is **not** rewritten into a cast. The two are different expressions, measured:
    `'abc' -as [int]` is `$null` where `[int] 'abc'` throws, and `300 -as [byte]` is `$null` where
    `[byte] 300` throws. `Ps1Outcome` is what tells them apart, and a conversion that may throw
    folds neither of them — so an `-as` this cannot answer is left standing rather than turned into
    the cast that stops the script.

    One arm is still read syntactically and is a ledgered defect rather than a fold that could be
    asked of the domain: `[char[]]` of a list of numbers. The conversion grid was captured over
    scalar targets only, so there is no cell to read for an array target, and it retires with the
    capture that adds the column. `[string]` of a collection stood beside it until the separator
    became a question that could be asked — see `_joined_collection`.

    Every question here is asked of one step. The operand has already been visited, so `read` names
    whatever it came to and `convert` answers the cast over that; `evaluate` would walk the operand
    again at every node, which is quadratic over a tree a visitor is already descending.
    """

    def __init__(self):
        super().__init__()
        self._flow: Ps1VariableFlow | None = None
        self._entry = False

    def visit(self, node: Node):
        """
        Captured once rather than per cast: every fold below marks the pass changed, which drops the
        cache, so a per-site lookup would rebuild the control-flow graphs of the whole script once
        per folded cast. This pass replaces an expression with the value it produces and neither
        adds nor removes a statement, so the graphs it would rebuild are the graphs it already has,
        and the writes it would find are the same writes.

        Dropped again when the walk it was captured for ends, for the reason
        `refinery.lib.scripts.ps1.deobfuscation.typenames.VariableTypeAwareTransformer` states: a
        second walk over a tree the first one rewrote enters on the guarded arm and would otherwise
        be answered from the first walk's graphs.
        """
        if self._entry or not isinstance(node, Ps1Script):
            return super().visit(node)
        self._entry = True
        try:
            self._flow = model_cache(self, node).variable_flow
            return super().visit(node)
        finally:
            self._entry = False
            self._flow = None

    def visit_Ps1BinaryExpression(self, node: Ps1BinaryExpression):
        self.generic_visit(node)
        if node.operator.lower() != '-as' or node.left is None:
            return None
        if not isinstance(node.right, Ps1TypeExpression):
            return None
        target = resolve_type(node.right.name)
        if target is None:
            return None
        return _spelled(node, convert(read(node.left), target))

    def visit_Ps1CastExpression(self, node: Ps1CastExpression):
        self.generic_visit(node)
        target = resolve_type(node.type_name)
        if target is None:
            return None
        return (
            self._named_type(node, target)
            or _spelled(node, convert(read(node.operand), target))
            or self._joined_collection(node, target)
            or self._characters(node, target)
        )

    @staticmethod
    def _named_type(node: Ps1CastExpression, target) -> Expression | None:
        """
        `[type] 'X'`, the one arm here that names a type rather than producing a value of one. What
        the expression evaluates to is a `System.RuntimeType`, which the domain deliberately carries
        no element for, so what this reads out of the operand is a name and what it writes is the
        type literal naming the same thing.
        """
        if target != _TYPE:
            return None
        named = text_of(read(node.operand))
        return None if named is None else Ps1TypeExpression(offset=node.offset, name=named)

    def _joined_collection(self, node: Ps1CastExpression, target) -> Expression | None:
        """
        `[string]` of a collection, whose elements 5.1 separates with `$OFS`. The conversion grid
        was captured over scalar targets only, so the domain has no cell to read for this, and the
        separator is not a property of the value in any case:
        `refinery.lib.scripts.ps1.analysis.separator` is what answers it, at the point the cast
        stands, and refuses wherever a run could have written the name something else.
        """
        if target != _STRING or node.operand is None or self._flow is None:
            return None
        text = coerced_text_at(unwrap_single_paren(node.operand), node, self._flow)
        return None if text is None else make_string_literal(text)

    @staticmethod
    def _characters(node: Ps1CastExpression, target) -> Expression | None:
        """
        `[char[]]` of a list of numbers, the second ledgered defect standing here: 5.1 builds a
        `Char[]` and this writes a String, so the container type and the element type are both lost.
        The conversion grid was captured over scalar targets only, so the domain has no cell to read
        for an array one and cannot answer it at all; the capture that adds the column retires this
        with `collect_integers`.
        """
        if target != _CHAR_ARRAY or node.operand is None:
            return None
        numbers = collect_integers(unwrap_single_paren(node.operand))
        if numbers is None:
            return None
        try:
            text = bytes(numbers).decode('ascii')
        except (ValueError, UnicodeDecodeError, OverflowError):
            return None
        if not all(c in string.printable or c.isspace() for c in text):
            return None
        return make_string_literal(text)


def _spelled(node: Expression, outcome: Ps1Outcome) -> Expression | None:
    """
    The expression an outcome is written as, or `None` where nothing is written.

    An outcome that may throw is never folded, because the script's throw is part of what it does
    and replacing it with the value the operation would have had deletes that. A fact that names no
    value has no spelling. And a value the node *already* spells is left alone: replacing a tree
    with an equal one is a rewrite that never converges, and `refinery.lib.scripts.canonical` is the
    model's own answer to whether two trees spell the same program.
    """
    if outcome.may_throw:
        return None
    spelled = render(outcome.value)
    if spelled is None or canonical(spelled) == canonical(node):
        return None
    return spelled
