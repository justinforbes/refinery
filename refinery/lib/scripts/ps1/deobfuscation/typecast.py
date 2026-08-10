"""
PowerShell type cast simplification transforms.
"""
from __future__ import annotations

import string

from refinery.lib.scripts import Transformer, canonical
from refinery.lib.scripts.ps1.analysis.values import (
    Ps1Outcome,
    collect_int_arguments,
    convert,
    make_string_literal,
    read,
    render,
    text_of,
    unwrap_integer,
)
from refinery.lib.scripts.ps1.data import named_type, resolve_type
from refinery.lib.scripts.ps1.deobfuscation.helpers import (
    collect_string_arguments,
    unwrap_single_paren,
)
from refinery.lib.scripts.ps1.model import (
    Expression,
    Ps1BinaryExpression,
    Ps1CastExpression,
    Ps1TypeExpression,
)

#: The four targets this pass treats as something other than a value conversion: one that names a
#: type rather than producing a value of one, one whose value the rest of the unit cannot read
#: back, and two whose answer the conversion grid does not carry. See the arm each belongs to.
_CHAR = named_type('System.Char')
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

    Two arms are still read syntactically, and each is a ledgered defect that retires with the
    reader it uses rather than here: `[char[]]` of a list of numbers, and `[string]` of a
    collection. Both belong to the string half of the folding pass.

    Every question here is asked of one step. The operand has already been visited, so `read` names
    whatever it came to and `convert` answers the cast over that; `evaluate` would walk the operand
    again at every node, which is quadratic over a tree a visitor is already descending.
    """

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
        if target == _CHAR:
            return self._one_character_string(node)
        return (
            self._named_type(node, target)
            or _spelled(node, convert(read(node.operand), target))
            or self._joined_collection(node, target)
            or self._characters(node, target)
        )

    @staticmethod
    def _one_character_string(node: Ps1CastExpression) -> Expression | None:
        """
        `[char]` of a number, spelled as the one-character String that is not the value 5.1
        produces. The domain answers this exactly and this does not ask it, which is deliberate and
        is the one place the migration stops short.

        A Char is what the *rest* of the unit cannot read. `render` writes one as `[char]65`,
        because that is the only spelling the language has for it, and every pass that recognizes a
        constant by `refinery.lib.scripts.ps1.ast.string_value` goes blind the moment a Char stops
        being a String: a method call on one, a `-replace` argument, a `-match` operand and the
        constant inliner each stop folding, which is five real samples that stop resolving. So the
        erasure outlives this commit by exactly as long as those readers do, and it goes with them
        in the commit that puts the string and member half of the folding pass onto the domain —
        which is where the ledger has always said the Char rows retire.

        Half of it cannot go early. Asking the domain here for the cases it already answers — a
        `[char]` of a String, which it converts exactly — would spell the result `[char]65`, and
        this arm would erase *that* on the next pass. `[char]'A'` is left alone today and would
        become a wrong answer; the arm is all of one piece and moves whole.
        """
        number = unwrap_integer(node.operand)
        if number is None or not 0 <= number.value <= 0xFFFF:
            return None
        return make_string_literal(chr(number.value))

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

    @staticmethod
    def _joined_collection(node: Ps1CastExpression, target) -> Expression | None:
        """
        `[string]` of a collection, which is a ledgered defect rather than a fold: 5.1 separates the
        elements with `$OFS`, which lives in the session and not in the script, so the text written
        here is only right for a run that left the separator alone. The domain refuses it for that
        reason and this does not, which is why it is still here and why it goes with
        `collect_string_arguments`.
        """
        if target != _STRING or node.operand is None:
            return None
        parts = collect_string_arguments(unwrap_single_paren(node.operand))
        if parts is None or len(parts) < 2:
            return None
        return make_string_literal(' '.join(parts))

    @staticmethod
    def _characters(node: Ps1CastExpression, target) -> Expression | None:
        """
        `[char[]]` of a list of numbers, the second ledgered defect standing here: 5.1 builds a
        `Char[]` and this writes a String, so the container type and the element type are both lost.
        The conversion grid was captured over scalar targets only, so the domain has no cell to read
        for an array one and cannot answer it at all; the capture that adds the column retires this
        with `collect_int_arguments`.
        """
        if target != _CHAR_ARRAY or node.operand is None:
            return None
        numbers = collect_int_arguments(unwrap_single_paren(node.operand))
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
