"""
PowerShell constant folding transforms.
"""
from __future__ import annotations

import base64
import codecs
import re

from typing import Iterator, NamedTuple

from refinery.lib.scripts import Node, reattach
from refinery.lib.scripts.ps1.analysis.effects import is_fault_free, may_be_dropped
from refinery.lib.scripts.ps1.analysis.values import (
    apply,
    apply_unary,
    collect_byte_array,
    collect_integers,
    convert,
    integer_at,
    integer_of,
    is_truthy,
    make_string_literal,
    pattern_at,
    read,
    render,
    text_of,
    unwrap_to_array_literal,
)
from refinery.lib.scripts.ps1.ast import get_body, get_member_name, string_value, unwrap_parens
from refinery.lib.scripts.ps1.data import ENCODING_MAP, named_type
from refinery.lib.scripts.ps1.dotnet import Ps1TypeName
from refinery.lib.scripts.ps1.deobfuscation.constants import PS1_ENV_CONSTANTS
from refinery.lib.scripts.ps1.deobfuscation.helpers import (
    WorldAwareTransformer,
    StringMethodError,
    apply_format_string,
    apply_string_method,
    collect_format_arguments,
    collect_string_arguments,
    detect_encoding_chain,
    dotnet_regex_replace,
    extract_foreach_scriptblock,
    is_array_reverse_call,
    is_pipeline_item,
    is_static_type_call,
    unwrap_single_paren,
)
from refinery.lib.scripts.ps1.deobfuscation.substitution import (
    substitute_field,
    substitute_list,
    substituted,
)
from refinery.lib.scripts.ps1.analysis.values import resolve_expression_type
from refinery.lib.scripts.ps1.data import MemberLookup, member_record
from refinery.lib.scripts.ps1.model import (
    Expression,
    Ps1ArrayExpression,
    Ps1ArrayLiteral,
    Ps1AssignmentExpression,
    Ps1BinaryExpression,
    Ps1ExpandableString,
    Ps1ExpressionStatement,
    Ps1HashLiteral,
    Ps1IndexExpression,
    Ps1IntegerLiteral,
    Ps1InvokeMember,
    Ps1MemberAccess,
    Ps1Pipeline,
    Ps1RangeExpression,
    Ps1ScopeModifier,
    Ps1ScriptBlock,
    Ps1StringLiteral,
    Ps1UnaryExpression,
    Ps1Variable,
)

_REGEX_OPTION_FLAGS: dict[str, int] = {
    'ignorecase'              : re.IGNORECASE,
    'multiline'               : re.MULTILINE,
    'singleline'              : re.DOTALL,
    'ignorepatternwhitespace' : re.VERBOSE,
    'none'                    : 0,
}

_REGEX_OPTION_INT: dict[int, int] = {
    1  : re.IGNORECASE,
    2  : re.MULTILINE,
    16 : re.DOTALL,
    32 : re.VERBOSE,
}

_RIGHT_TO_LEFT = 64
_MAX_STRING_EXPAND = 0x1000
_MAX_RANGES_EXPAND = 15

#: The whitespace `[Convert]::To<integer>(string)` strips, measured as `' 5 '` converting to 5. Only
#: the one-argument form strips anything: `[Convert]::ToInt32(' 5 ', 16)` throws.
_CONVERT_TRIM = ' \t\r\n'

#: What `[Convert]::To<integer>(string)` reads, which is neither what a cast reads nor what Python's
#: `int` does: an optional sign and decimal digits. Measured — `'+7'` is 7, `'007'` is 7 and `'-5'`
#: is -5, while `'0x10'`, `'1_0'`, `'0b1010'`, `'7.5'`, `'1e3'`, `'1,000'` and `''` each throw.
_CONVERT_SIGNED = re.compile(r'[+-]?[0-9]+\Z')

#: What `[Convert]::To<integer>(string, base)` reads for a base that is not ten: the digits of that
#: base and nothing else, and only base sixteen takes a `0x` prefix — `'0b1010'` at base two throws.
#: A sign throws and so does whitespace, so neither is written here.
_CONVERT_PATTERN: dict[int, re.Pattern[str]] = {
    2: re.compile(r'[01]+\Z'),
    8: re.compile(r'[0-7]+\Z'),
    16: re.compile(r'(?:0[xX])?[0-9a-fA-F]+\Z'),
}


def _is_static_regex_call(node: Ps1InvokeMember) -> bool:
    return is_static_type_call(node, 'system.text.regularexpressions.regex')


def _parse_regex_options(node: Expression) -> tuple[int, bool] | None:
    """
    Parse a RegexOptions argument (string or integer) into Python re flags
    and a right_to_left boolean.
    """
    sv = string_value(node)
    if sv is not None:
        flags = 0
        right_to_left = False
        for part in sv.split(','):
            key = part.strip().lower()
            if not key:
                continue
            if key == 'righttoleft':
                right_to_left = True
                continue
            flag = _REGEX_OPTION_FLAGS.get(key)
            if flag is None:
                return None
            flags |= flag
        return flags, right_to_left
    if isinstance(node, Ps1IntegerLiteral):
        value = node.value
        right_to_left = bool(value & _RIGHT_TO_LEFT)
        flags = 0
        for bit, flag in _REGEX_OPTION_INT.items():
            if value & bit:
                flags |= flag
        return flags, right_to_left
    return None


def _iter_regex_matches(node: Ps1InvokeMember) -> Iterator[str] | None:
    """
    Yield matched strings from a call to

        [Regex]::Match/Matches(input, pattern[, options])

    Returns `None` if the arguments cannot be resolved.
    """
    if len(node.arguments) not in (2, 3):
        return None
    input = string_value(node.arguments[0])
    pattern = string_value(node.arguments[1])
    if input is None or pattern is None:
        return None
    if len(node.arguments) == 3:
        if (options := _parse_regex_options(node.arguments[2])) is None:
            return None
        flags, right_to_left = options
    else:
        flags, right_to_left = 0, False
    try:
        matches = [m[0] for m in re.finditer(pattern, input, flags)]
    except re.error:
        return None
    if right_to_left:
        matches.reverse()
    return iter(matches)


def _compute_regex_matches(node: Ps1InvokeMember) -> list[str] | None:
    if it := _iter_regex_matches(node):
        return list(it)


def _compute_regex_match(node: Ps1InvokeMember) -> str | None:
    if it := _iter_regex_matches(node):
        return next(it, '')


def _integer(value: int) -> Ps1IntegerLiteral:
    return Ps1IntegerLiteral(raw=str(value))


#: The members whose value is decided by the shape of the receiver rather than by anything the
#: receiver holds, and what each answers. Dispatch is on the member and never on the type its read
#: produces: `Rank`, `Length` and `Count` all produce an `Int32`, so a gate that asked only for an
#: integer result answered `Rank` with the element count — 5.1 says 1, because `Rank` is the number
#: of dimensions of the array and not the number of things in it.
#:
#: Each answer is measured on a 5.1 host; see `TYPE_TRANSCRIPTS` in
#: `test.lib.scripts.ps1.test_oracle`. The one that reads oddly is `Count` on a string, which is 1
#: and not the character count: `Count` comes from the object adapter, which counts the value as one
#: object, while `Length` is the string's own member.
_SHAPE_MEMBERS = frozenset({'length', 'count', 'rank'})


def _foreach_extracts_value(sb: Ps1ScriptBlock) -> bool:
    """
    Check whether a ForEach scriptblock body is of the form `$_.Value`,
    `$_.Groups.Value`, or `$_.Groups.Captures.Groups.Value` — i.e. it
    extracts the string value from Match objects.
    """
    if sb.body is None or len(sb.body) != 1:
        return False
    stmt = sb.body[0]
    if not isinstance(stmt, Ps1ExpressionStatement) or stmt.expression is None:
        return False
    node = stmt.expression
    if not isinstance(node, Ps1Pipeline):
        expr = node
    elif len(node.elements) == 1 and node.elements[0].expression is not None:
        expr = node.elements[0].expression
    else:
        return False
    if not isinstance(expr, Ps1MemberAccess):
        return False
    member = expr.member if isinstance(expr.member, str) else None
    if member is None or member.lower() != 'value':
        return False
    inner = expr.object
    while isinstance(inner, Ps1MemberAccess):
        prop = inner.member if isinstance(inner.member, str) else None
        if prop is None or prop.lower() not in ('groups', 'captures'):
            return False
        inner = inner.object
    return is_pipeline_item(inner)


def _escape_for_expandable(text: str) -> str:
    """
    Escape characters that are special inside double-quoted strings.
    """
    return text.replace('`', '``').replace('$', '`$')


def _variable_raw(var: Ps1Variable) -> str:
    """
    Produce the braced variable reference for use inside an expandable string.
    """
    prefix = '@' if var.splatted else '$'
    scope = var.scope.value
    if scope:
        return F'{prefix}{{{scope}:{var.name}}}'
    return F'{prefix}{{{var.name}}}'


def _is_string_typed_variable(node: Expression | None) -> bool:
    """
    Return `True` only for a variable whose value is provably a string, so that folding a `+`
    concatenation into an expandable string cannot change array/number `+` semantics. Environment
    variables are always strings in PowerShell.
    """
    return isinstance(node, Ps1Variable) and node.scope == Ps1ScopeModifier.ENV


def _variable_string_to_expandable(
    var: Ps1Variable,
    text: str,
    *,
    var_first: bool,
) -> Ps1ExpandableString:
    """
    Fold `$var + 'text'` or `'text' + $var` into a
    `refinery.lib.scripts.ps1.model.Ps1ExpandableString`.
    """
    escaped = _escape_for_expandable(text)
    var_raw = _variable_raw(var)
    text_part = Ps1StringLiteral(value=text, raw=F"'{text}'")
    if var_first:
        raw = F'"{var_raw}{escaped}"'
        parts = [var, text_part]
    else:
        raw = F'"{escaped}{var_raw}"'
        parts = [text_part, var]
    return Ps1ExpandableString(parts=parts, raw=raw)


def _resolve_index_values(index: Expression) -> int | list[int] | None:
    """
    The index or indices an expression names. A scalar and a collection are told apart because a
    read at one index yields the element and a read at several yields a collection of them, so the
    two are different values rather than one of length one.
    """
    single = integer_of(read(index))
    if single is not None:
        return single
    array = unwrap_to_array_literal(index)
    return None if array is None else collect_integers(array)


class _Selection(NamedTuple):
    """
    What reading a value out of a literal container yields, beside what building the container
    evaluated and the read then leaves behind.

    Indexing is one such read and `.Length` is another: the count carries nothing forward at all, so
    every element is dropped and every element has to be answered for.

    The two halves are answered together because they are one decision. A fold that reports only
    what it carries forward leaves its caller to reconstruct the rest, and the reconstruction is
    what went wrong: indexing an array literal was read as choosing among *values*, where the
    elements are also *work* — `@(1, (Start-Process calc))[0]` folded to `1` and the command ran in
    the original. It is the same rule the effect layer already states for `[Void](Start-Process x)`,
    which is an `EFFECT` because the wrapper discards a value and never the evaluation behind it.
    """
    carried: Expression
    dropped: list[Expression]


def _index_into_string(s: str, indices: int | list[int]) -> _Selection | None:
    """
    A string is a value and not a container of expressions, so a character selected out of one
    leaves no evaluation behind, whatever the index.
    """
    n = len(s)
    if isinstance(indices, int):
        if -n <= indices < n:
            return _Selection(make_string_literal(s[indices]), [])
        return None
    selected: list[Expression] = []
    for i in indices:
        if not (-n <= i < n):
            return None
        selected.append(make_string_literal(s[i]))
    return _Selection(Ps1ArrayLiteral(elements=selected), [])


def _index_into_array(
    array: Ps1ArrayLiteral, indices: int | list[int],
) -> _Selection | None:
    """
    The element or elements a literal array yields for `indices`, beside the elements the selection
    leaves behind.

    **An index that repeats is refused rather than folded.** The selected elements are the array's
    own nodes, so `@(1, 2, 3)[0, 0]` would put one object in two slots of the result: `Node.parent`
    holds one holder, so a later `refinery.lib.scripts._replace_in_parent` rewrites one occurrence
    of two, a transformer visits it twice, and a walk counts whatever it carries twice.

    Copying the node instead would answer a different question — whether the element may be
    *evaluated* twice — and the answer is no for anything with an effect: `@($a.B(), 2)[0, 0]`
    builds the array once and calls `B` once, where the copy calls it twice. That question has no
    caller, because nothing in the corpus or the suite selects a repeated index out of an array
    literal, so it is refused here rather than answered. `_index_into_string` is unaffected: it
    builds a fresh literal per index out of a value that was never a node.
    """
    n = len(array.elements)
    if isinstance(indices, int):
        if not (-n <= indices < n):
            return None
        selected = [array.elements[indices]]
        carried = selected[0]
    else:
        selected = []
        for i in indices:
            if not (-n <= i < n):
                return None
            selected.append(array.elements[i])
        if len({id(element) for element in selected}) != len(selected):
            return None
        carried = Ps1ArrayLiteral(elements=list(selected))
    kept = {id(element) for element in selected}
    return _Selection(
        carried, [element for element in array.elements if id(element) not in kept])


def _lookup_hashtable(ht: Ps1HashLiteral, index: Expression) -> _Selection | None:
    """
    The value a literal hash table holds for `index`, beside every other part of the literal.

    Both halves of each pair are reported as dropped, keys included. PowerShell 5.1 rejects a bare
    subexpression key outright, so the shape that runs is an expandable string holding one, and
    telling that spelling apart from a plain name here would be a second rule about which parts of
    a literal are evaluated — where the whole literal plainly is.
    """
    key = string_value(index)
    if key is None:
        return None
    lower = key.lower()
    for pair_key, pair_value in ht.pairs:
        k = string_value(pair_key)
        if k is not None and k.lower() == lower:
            return _Selection(pair_value, [
                part
                for other_key, other_value in ht.pairs
                for part in (other_key, other_value)
                if part is not pair_value
            ])
    return None


def _pipeline_output(value: Expression | None) -> Expression | None:
    """
    What a pipeline hands to whoever consumes it. A pipeline that emits exactly one object passes
    that object along; it does not wrap it in a collection. Folding a single match out of
    `[regex]::Matches(...) | %{ $_.Value }` therefore yields the string, and an array literal of one
    element here would assign an array where PowerShell assigns a string. Two or more values are a
    collection either way and are left alone.
    """
    if isinstance(value, Ps1ArrayLiteral) and len(value.elements) == 1:
        return value.elements[0]
    return value


def _folded(outcome) -> Expression | None:
    """
    The expression a domain outcome folds to, or `None` where it does not fold. An outcome that may
    throw is never folded: the script's throw is part of what it does, and replacing it with the
    value the operation would have had deletes that. Everything else is left to `render`, which is
    the one place that decides whether a value has a spelling — including `$null`, which is a value
    an operation can produce and not a sign that it produced nothing.
    """
    return None if outcome.may_throw else render(outcome.value)


class Ps1ConstantFolding(WorldAwareTransformer):

    def visit_Ps1Pipeline(self, node: Ps1Pipeline):
        if len(node.elements) == 2:
            result = substituted(node, _pipeline_output(self._try_fold_regex_pipeline(node)))
            if result is not None:
                return result
        self.generic_visit(node)
        return None

    @staticmethod
    def _fold_regex_call_result(
        invoke: Ps1InvokeMember, member_lower: str,
    ) -> Expression | None:
        if member_lower == 'matches':
            matches = _compute_regex_matches(invoke)
            if matches is not None:
                elements: list[Expression] = [make_string_literal(s) for s in matches]
                return Ps1ArrayLiteral(elements=elements)
        elif member_lower == 'match':
            result = _compute_regex_match(invoke)
            if result is not None:
                return make_string_literal(result)
        return None

    def _try_fold_regex_pipeline(self, node: Ps1Pipeline) -> Expression | None:
        first = node.elements[0].expression
        second_expr = node.elements[1].expression
        if not isinstance(first, Ps1InvokeMember) or not _is_static_regex_call(first):
            return None
        member = get_member_name(first.member)
        if member is None:
            return None
        sb = extract_foreach_scriptblock(second_expr) if second_expr else None
        if sb is None or not _foreach_extracts_value(sb):
            return None
        return self._fold_regex_call_result(first, member.lower())

    def visit_Ps1MemberAccess(self, node: Ps1MemberAccess):
        self.generic_visit(node)
        member = get_member_name(node.member)
        if member is None:
            return None
        obj = node.object
        if obj is None:
            return None
        shaped = self._fold_shape_member(node, obj, member)
        if shaped is not None:
            return shaped
        if string_value(obj) is not None or isinstance(obj, Ps1IntegerLiteral):
            obj_type = resolve_expression_type(obj)
            if obj_type is not None:
                record = member_record(obj_type, member)
                if record is MemberLookup.ABSENT:
                    return Ps1Variable(name='Null')
        result = self._try_fold_regex_member_access(node, member)
        if result is not None:
            return result
        return None

    def _fold_shape_member(
        self,
        node: Ps1MemberAccess,
        obj: Expression,
        member: str,
    ) -> Expression | None:
        """
        Fold a member whose value the receiver's shape decides. The array answers go through
        `_selected` because folding one discards the elements that built it, and whether that is
        safe is the selection's question rather than this one's.
        """
        name = member.lower()
        if name not in _SHAPE_MEMBERS:
            return None
        text = string_value(obj)
        if text is not None:
            return _integer(len(text) if name == 'length' else 1)
        array = unwrap_to_array_literal(obj)
        if array is not None:
            count = 1 if name == 'rank' else len(array.elements)
            return self._selected(node, _Selection(_integer(count), list(array.elements)))
        if isinstance(obj, Ps1IntegerLiteral) and name != 'rank':
            return _integer(1)
        return None

    def _try_fold_regex_member_access(
        self, node: Ps1MemberAccess, member: str,
    ) -> Expression | None:
        chain: list[str] = [member]
        inner = node.object
        while isinstance(inner, Ps1MemberAccess):
            prop = get_member_name(inner.member)
            if prop is None:
                return None
            chain.append(prop)
            inner = inner.object
        chain.reverse()
        if not isinstance(inner, Ps1InvokeMember) or not _is_static_regex_call(inner):
            return None
        normalized = [c.lower() for c in chain]
        if normalized[-1] != 'value':
            return None
        for c in normalized[:-1]:
            if c not in ('groups', 'captures'):
                return None
        call_member = inner.member if isinstance(inner.member, str) else None
        if call_member is None:
            return None
        return self._fold_regex_call_result(inner, call_member.lower())

    @staticmethod
    def _try_join_regex_matches(operand: Expression) -> Expression | None:
        unwrapped = unwrap_parens(operand)
        if not isinstance(unwrapped, Ps1InvokeMember) or not _is_static_regex_call(unwrapped):
            return None
        member = unwrapped.member if isinstance(unwrapped.member, str) else None
        if member is None or member.lower() != 'matches':
            return None
        matches = _compute_regex_matches(unwrapped)
        if matches is None:
            return None
        return make_string_literal(''.join(matches))

    def visit_Ps1UnaryExpression(self, node: Ps1UnaryExpression):
        self.generic_visit(node)
        if node.operand is None:
            return None
        op = node.operator.lower()
        if op == '-join':
            return self._handle_unary_join(node)
        if op == '-bnot':
            return _folded(apply_unary(op, read(node.operand)))
        if op in ('-not', '!'):
            truth = is_truthy(node.operand)
            if truth is not None:
                return Ps1Variable(name='False' if truth else 'True')
        return None

    def _handle_unary_join(self, node: Ps1UnaryExpression) -> Expression | None:
        operand = node.operand
        if operand is None:
            return None
        scalar = string_value(operand)
        if scalar is not None:
            return make_string_literal(scalar)
        result = self._try_join_regex_matches(operand)
        if result is not None:
            return result
        array = unwrap_to_array_literal(operand)
        if array is None:
            if isinstance(operand, Ps1ArrayExpression) and len(operand.body) == 1:
                stmt = operand.body[0]
                if isinstance(stmt, Ps1ExpressionStatement):
                    sv = string_value(stmt.expression) if stmt.expression else None
                    if sv is not None:
                        return make_string_literal(sv)
            return None
        args = collect_string_arguments(array)
        if args is None:
            return None
        return make_string_literal(''.join(args))

    def visit_Ps1RangeExpression(self, node: Ps1RangeExpression):
        self.generic_visit(node)
        if isinstance(node.parent, Ps1RangeExpression):
            return None
        a = integer_of(read(node.start))
        b = integer_of(read(node.end))
        if a is None or b is None:
            return None
        step = 1 if b >= a else -1
        count = abs(b - a) + 1
        if count > _MAX_RANGES_EXPAND:
            return None
        if not is_fault_free(node):
            return None
        return Ps1ArrayLiteral(elements=[
            Ps1IntegerLiteral(raw=str(v)) for v in range(a, b + step, step)])

    def _selected(self, node: Node, selection: _Selection | None) -> Expression | None:
        """
        The expression a selection out of `node` folds to, or `None` when what it leaves behind is
        work the script would no longer do; see
        `refinery.lib.scripts.ps1.analysis.effects.may_be_dropped` for what that means.

        The world is the one captured at the root by
        `refinery.lib.scripts.ps1.deobfuscation.helpers.WorldAwareTransformer`. This pass only
        folds, so a verdict taken before its own edits is the more open, and so the more
        conservative, of the two.

        A refused selection is released the way
        `refinery.lib.scripts.ps1.deobfuscation.substitution` releases one: a multi-index read has
        already built the array literal that carries the result, and building it adopted elements
        that are still standing under `node`.
        """
        if selection is None:
            return None
        if not all(may_be_dropped(part, self._world) for part in selection.dropped):
            reattach(node)
            return None
        return selection.carried

    def visit_Ps1IndexExpression(self, node: Ps1IndexExpression):
        self.generic_visit(node)
        if node.index is None or node.object is None:
            return None
        if isinstance(node.object, Ps1HashLiteral):
            return self._selected(node, _lookup_hashtable(node.object, node.index))
        indices = _resolve_index_values(node.index)
        if indices is None:
            return None
        obj_str = string_value(node.object)
        if obj_str is not None:
            return self._selected(node, _index_into_string(obj_str, indices))
        array = unwrap_to_array_literal(node.object)
        if array is not None:
            return self._selected(node, _index_into_array(array, indices))
        return None

    def visit_Ps1ExpressionStatement(self, node: Ps1ExpressionStatement):
        self.generic_visit(node)
        var = is_array_reverse_call(node)
        if var is not None and self._try_apply_array_reverse(node, var):
            return node
        return None

    def _try_apply_array_reverse(
        self, node: Ps1ExpressionStatement, var: Ps1Variable,
    ) -> bool:
        body = get_body(node.parent)
        if body is None:
            return False
        try:
            idx = body.index(node)
        except ValueError:
            return False
        var_name = var.name.lower()
        for i in range(idx - 1, -1, -1):
            stmt = body[i]
            if not isinstance(stmt, Ps1ExpressionStatement):
                continue
            expr = stmt.expression
            if not isinstance(expr, Ps1AssignmentExpression):
                continue
            if expr.operator != '=':
                continue
            target = expr.target
            if not isinstance(target, Ps1Variable):
                continue
            if target.name.lower() != var_name:
                continue
            value = expr.value
            if isinstance(value, Ps1ArrayLiteral):
                return self._reversed(node, substitute_list(
                    value, 'elements', value.elements[::-1]))
            if isinstance(value, Ps1ArrayExpression) and len(value.body) == 1:
                inner = value.body[0]
                if (
                    isinstance(inner, Ps1ExpressionStatement)
                    and isinstance(inner.expression, Ps1ArrayLiteral)
                ):
                    literal = inner.expression
                    return self._reversed(node, substitute_list(
                        literal, 'elements', literal.elements[::-1]))
            sv = string_value(value)
            if sv is not None:
                return self._reversed(node, substitute_field(
                    expr, 'value', make_string_literal(sv[::-1])))
            return False
        return False

    def _reversed(self, node: Ps1ExpressionStatement, applied: bool) -> bool:
        """
        Drop the `[Array]::Reverse` call `node` holds once the reversal it asks for has landed, and
        report whether the pair happened.

        The order is the whole of it. Clearing the call first and reversing second leaves a refused
        reversal beside a deleted call, so the emitted script reads the array in its original order
        with nothing left to say it should not — a silent change of values rather than a rewrite
        declined.
        """
        if not applied:
            return False
        if not substitute_field(node, 'expression', None):
            return False
        self.mark_changed()
        return True

    def visit_Ps1InvokeMember(self, node: Ps1InvokeMember):
        self.generic_visit(node)
        member_name = get_member_name(node.member)
        if member_name is None:
            return None
        lower = member_name.lower()
        return (
            self._try_fold_invoke_redirect(node, lower)
            or self._try_fold_instance_method(node, lower)
            or self._try_fold_static_method(node, lower)
        ) or None

    @staticmethod
    def _try_fold_invoke_redirect(
        node: Ps1InvokeMember, lower: str,
    ) -> Expression | None:
        if lower == 'invoke' and isinstance(node.object, Ps1MemberAccess):
            return Ps1InvokeMember(
                offset=node.offset,
                object=node.object.object,
                member=node.object.member,
                arguments=node.arguments,
                access=node.object.access,
            )
        return None

    @staticmethod
    def _try_fold_instance_method(
        node: Ps1InvokeMember, lower: str,
    ) -> Expression | None:
        obj_str = string_value(node.object) if node.object else None
        if obj_str is None:
            return None
        coerced: list[str | int] = []
        for arg in node.arguments:
            sv = string_value(arg)
            if sv is not None:
                coerced.append(sv)
                continue
            if isinstance(arg, Ps1IntegerLiteral):
                coerced.append(arg.value)
                continue
            return None
        try:
            result = apply_string_method(obj_str, lower, coerced)
        except StringMethodError:
            return None
        if isinstance(result, str):
            return make_string_literal(result)
        if isinstance(result, bool):
            return Ps1Variable(name='True' if result else 'False')
        if isinstance(result, int):
            return Ps1IntegerLiteral(raw=str(result))
        if isinstance(result, list):
            elements: list[Expression] = [make_string_literal(p) for p in result]
            return Ps1ArrayLiteral(elements=elements)
        return None

    def _try_fold_static_method(
        self, node: Ps1InvokeMember, lower: str,
    ) -> Expression | None:
        if is_static_type_call(node, 'system.convert'):
            return self._try_fold_convert(node, lower)
        encoding_name = detect_encoding_chain(node)
        if encoding_name is not None:
            if len(node.arguments) == 1:
                arg = unwrap_single_paren(node.arguments[0])
                if isinstance(arg, Ps1ArrayExpression) and len(arg.body) == 1:
                    stmt = arg.body[0]
                    if isinstance(stmt, Ps1ExpressionStatement) and stmt.expression:
                        arg = stmt.expression
                int_values = collect_integers(arg)
                if int_values is not None:
                    try:
                        raw_bytes = bytearray(int_values)
                    except (ValueError, OverflowError):
                        return None
                    encoding = ENCODING_MAP.get(
                        encoding_name.lower(), encoding_name)
                    try:
                        codecs.lookup(encoding)
                    except LookupError:
                        encoding = 'utf-8'
                    try:
                        decoded_str = raw_bytes.decode(encoding)
                    except Exception:
                        return None
                    return make_string_literal(decoded_str)
        if is_static_type_call(node, 'system.string'):
            if lower == 'concat' and len(node.arguments) >= 1:
                parts: list[str] = []
                for arg in node.arguments:
                    if (sv := string_value(arg)) is None:
                        break
                    parts.append(sv)
                else:
                    return make_string_literal(''.join(parts))
            if lower == 'join' and len(node.arguments) >= 2:
                separator = string_value(node.arguments[0])
                if separator is not None:
                    joined: list[str] = []
                    for arg in node.arguments[1:]:
                        if (sv := string_value(arg)) is None:
                            break
                        joined.append(sv)
                    else:
                        return make_string_literal(separator.join(joined))
                    if len(node.arguments) == 2:
                        array = unwrap_to_array_literal(node.arguments[1])
                        if array is not None:
                            args = collect_string_arguments(array)
                            if args is not None:
                                return make_string_literal(separator.join(args))
        if _is_static_regex_call(node) and lower == 'replace':
            return self._handle_regex_replace(node)
        if is_static_type_call(node, 'system.bitconverter') and lower == 'tostring':
            return self._try_fold_bitconverter_tostring(node)
        if (
            is_static_type_call(node, 'system.environment')
            and lower == 'getenvironmentvariable'
            and len(na := node.arguments) == 1
            and (_en := string_value(na[0])) is not None
            and (_ev := PS1_ENV_CONSTANTS.get(_en.lower())) is not None
        ):
            return make_string_literal(_ev)
        return None

    #: The integer type each `[Convert]::To<T>` produces, which is what the result is spelled at.
    #: Measured: `[Convert]::ToByte('FF', 16)` is a Byte and `[Convert]::ToInt64(5)` an Int64, so a
    #: fold that wrote a bare numeral for either reported an Int32 the call never produced.
    _CONVERT_INT_METHODS: dict[str, Ps1TypeName] = {
        'tobyte'  : named_type('System.Byte'),
        'toint16' : named_type('System.Int16'),
        'toint32' : named_type('System.Int32'),
        'toint64' : named_type('System.Int64'),
        'tosbyte' : named_type('System.SByte'),
        'touint16': named_type('System.UInt16'),
        'touint32': named_type('System.UInt32'),
        'touint64': named_type('System.UInt64'),
    }

    def _try_fold_convert(
        self, node: Ps1InvokeMember, lower: str,
    ) -> Expression | None:
        if lower == 'frombase64string' and len(node.arguments) == 1:
            b64_str = string_value(node.arguments[0])
            if b64_str is not None:
                try:
                    decoded = base64.b64decode(b64_str)
                except Exception:
                    return None
                elements: list[Expression] = [
                    Ps1IntegerLiteral(raw=F'0x{b:02X}') for b in decoded
                ]
                array = Ps1ArrayLiteral(elements=elements)
                return Ps1ArrayExpression(
                    body=[Ps1ExpressionStatement(expression=array)])
        target = self._CONVERT_INT_METHODS.get(lower)
        if target is not None:
            return self._fold_convert_int(node, target)
        if lower == 'tochar':
            n = integer_of(read(node.arguments[0])) if len(node.arguments) == 1 else None
            if n is not None and 0 <= n <= 0xFFFF:
                return make_string_literal(chr(n))
        return None

    @staticmethod
    def _fold_convert_int(node: Ps1InvokeMember, target: Ps1TypeName) -> Expression | None:
        """
        `[Convert]::To<integer>(...)`, whose String operand is read by an oracle that is neither the
        cast's nor Python's — a third one, measured, and this is where the difference is written.

        For every source but a String the call *is* the cast, so it asks `convert`:
        `[Convert]::ToInt32(1.5)` and `(2.5)` are both 2, which is the half-to-even a cast performs,
        and a Char, a `$true` and a `$null` each convert exactly as they do under one.

        A String is stricter than the cast in the one-argument form. `[Convert]::ToInt32('0x10')`
        throws where `[int]'0x10'` is 16, and so do `'7.5'`, `'1e3'`, `'1,000'`, `'1_0'`, `'0b1010'`
        and the empty String, each of which a cast or Python's own `int` reads as a number — which
        is what this used to do, so `[Convert]::ToInt32('0x10')` answered 16 for a script that stops.

        With an explicit base it is stricter still and it reads a *pattern*: `'FFFFFFFF'` at base
        sixteen is the Int32 -1 and `'80000000'` is -2147483648, where the digits read as a number
        are out of range and this refused to fold them at all.
        """
        arguments = node.arguments
        if len(arguments) == 1:
            fact = read(arguments[0])
            text = text_of(fact)
            if text is None:
                return _folded(convert(fact, target))
            digits = text.strip(_CONVERT_TRIM)
            if not _CONVERT_SIGNED.match(digits):
                return None
            return render(integer_at(target, int(digits)))
        if len(arguments) != 2:
            return None
        text = text_of(read(arguments[0]))
        base = integer_of(read(arguments[1]))
        if text is None or base is None:
            return None
        if base == 10:
            if not _CONVERT_SIGNED.match(text):
                return None
            return render(integer_at(target, int(text)))
        digits_of_base = _CONVERT_PATTERN.get(base)
        if digits_of_base is None or not digits_of_base.match(text):
            return None
        return render(pattern_at(target, int(text, base)))

    @staticmethod
    def _try_fold_bitconverter_tostring(node: Ps1InvokeMember) -> Expression | None:
        if not node.arguments:
            return None
        data = collect_byte_array(node.arguments[0])
        if data is None:
            return None
        offset = 0
        length = len(data)
        if len(node.arguments) >= 2:
            n = integer_of(read(node.arguments[1]))
            if n is None:
                return None
            offset = n
        if len(node.arguments) >= 3:
            n = integer_of(read(node.arguments[2]))
            if n is None:
                return None
            length = n
        if offset < 0 or length < 0 or offset + length > len(data):
            return None
        segment = data[offset:offset + length]
        return make_string_literal('-'.join(F'{b:02X}' for b in segment))

    def _handle_regex_replace(self, node: Ps1InvokeMember) -> Expression | None:
        if len(node.arguments) not in (3, 4):
            return None
        input_str = string_value(node.arguments[0])
        pattern_str = string_value(node.arguments[1])
        replacement_str = string_value(node.arguments[2])
        if input_str is None or pattern_str is None or replacement_str is None:
            return None
        flags = 0
        if len(node.arguments) == 4:
            opts = _parse_regex_options(node.arguments[3])
            if opts is None:
                return None
            flags, _ = opts
        try:
            result = dotnet_regex_replace(pattern_str, replacement_str, input_str, flags=flags)
        except re.error:
            return None
        return make_string_literal(result)

    def visit_Ps1BinaryExpression(self, node: Ps1BinaryExpression):
        self.generic_visit(node)
        op = node.operator.lower()
        if op == '-f':
            return self._handle_format(node)
        if op == '+':
            return self._handle_concat(node) or self._handle_arithmetic(node, op)
        if op == '*':
            return self._handle_string_multiply(node) or self._handle_arithmetic(node, op)
        if op == '-join':
            return self._handle_binary_join(node)
        if op in ('-replace', '-creplace', '-ireplace'):
            return self._handle_binary_replace(node, op)
        if op in ('-split', '-csplit', '-isplit'):
            return self._handle_binary_split(node, op)
        if op in ('-and', '-or', '-xor'):
            return self._handle_logical(node, op)
        return self._handle_comparison(node, op) or self._handle_arithmetic(node, op)

    @staticmethod
    def _handle_arithmetic(node: Ps1BinaryExpression, op: str) -> Expression | None:
        """
        Fold an arithmetic, bitwise or shift operator by asking the value domain what it produces.
        The result is spelled at the type the domain gives it, so a fold cannot quietly change one:
        `0xFFFFFFFF -bxor 0x5A` is Int32 -91 rather than the 4294967205 an operand read as an
        unsigned Python integer would give, and `2147483647 + 1` widens to a Double as a host does.
        """
        return _folded(apply(op, read(node.left), read(node.right)))

    @staticmethod
    def _handle_string_multiply(node: Ps1BinaryExpression) -> Expression | None:
        """
        Replication, which `*` performs when its *left* operand is a String and nothing else.

        A negative count is a throw and not an empty string. Measured: `'ab' * -1` terminates the
        script with an `ArgumentOutOfRangeException`, and so does `'ab' * 0xFFFFFFFF`, whose count
        is the Int32 -1. Clamping it to zero answered `''` for both, which is the direction that
        turns a script that stopped into one that carries on — and it only became reachable once
        the count was read as the number its spelling names.
        """
        s = string_value(node.left) if node.left else None
        count = integer_of(read(node.right))
        if s is None or count is None or count < 0:
            return None
        if len(s) * count > _MAX_STRING_EXPAND:
            return None
        return make_string_literal(s * count)

    @staticmethod
    def _bool_literal(result: bool) -> Ps1Variable:
        """
        Build the `$True`/`$False` variable node that represents a folded boolean value.
        """
        return Ps1Variable(name='True' if result else 'False')

    def _handle_comparison(self, node: Ps1BinaryExpression, op: str) -> Expression | None:
        compared = _folded(apply(op, read(node.left), read(node.right)))
        if compared is not None:
            return compared
        return self._handle_string_equality(node, op)

    def _handle_string_equality(self, node: Ps1BinaryExpression, op: str) -> Expression | None:
        """
        Fold an equality comparison between two constant strings. Only equality operators are folded
        (`-eq`/`-ne` and their case-sensitive `-ceq`/`-cne` and explicit case-insensitive `-ieq`/`-ine`
        variants); ordering comparisons follow culture-dependent rules and are left untouched.
        """
        base = op[2:] if op[:2] in ('-c', '-i') else op[1:]
        if base not in ('eq', 'ne'):
            return None
        left = string_value(node.left)
        right = string_value(node.right)
        if left is None or right is None:
            return None
        if op.startswith('-c'):
            equal = left == right
        else:
            equal = left.lower() == right.lower()
        return self._bool_literal(equal if base == 'eq' else not equal)

    def _handle_logical(self, node: Ps1BinaryExpression, op: str) -> Expression | None:
        """
        Fold the logical operators `-and`, `-or`, and `-xor` when both operands are constant.
        """
        left = is_truthy(node.left)
        right = is_truthy(node.right)
        if left is None or right is None:
            return None
        if op == '-and':
            result = left and right
        elif op == '-or':
            result = left or right
        else:
            result = left != right
        return self._bool_literal(result)

    def _handle_format(self, node: Ps1BinaryExpression) -> Expression | None:
        fmt_str = string_value(node.left) if node.left else None
        if fmt_str is None or node.right is None:
            return None
        args = collect_format_arguments(node.right)
        if args is None:
            return None
        result = apply_format_string(fmt_str, args)
        if result is None:
            return None
        return make_string_literal(result)

    def _handle_concat(self, node: Ps1BinaryExpression) -> Expression | None:
        left_str = string_value(node.left) if node.left else None
        right_str = string_value(node.right) if node.right else None
        if left_str is not None and right_str is not None:
            return make_string_literal(left_str + right_str)
        if right_str is not None and isinstance(node.left, Ps1BinaryExpression):
            if node.left.operator == '+':
                inner_right_str = string_value(node.left.right) if node.left.right else None
                if inner_right_str is not None:
                    nl = make_string_literal(inner_right_str + right_str)
                    nl.parent = node.left
                    node.left.right = nl
                    return node.left
        if right_str is not None and isinstance(node.left, Ps1ArrayLiteral):
            elements = list(node.left.elements)
            elements.append(make_string_literal(right_str))
            return Ps1ArrayLiteral(elements=elements)
        is_inner_concat = (
            isinstance(node.parent, Ps1BinaryExpression)
            and node.parent.operator == '+'
            and node.parent.left is node
        )
        if not is_inner_concat:
            # `'literal' + $var` is always string concatenation (the string-typed left operand
            # governs `+`), so it is safe to fold into an expandable string. `$var + 'literal'`
            # depends on $var's runtime type (array append / numeric add), so only fold it when the
            # variable is provably a string.
            if isinstance(node.right, Ps1Variable) and left_str is not None:
                return _variable_string_to_expandable(node.right, left_str, var_first=False)
            if _is_string_typed_variable(node.left) and right_str is not None:
                return _variable_string_to_expandable(node.left, right_str, var_first=True)
        return None

    def _handle_binary_join(self, node: Ps1BinaryExpression) -> Expression | None:
        separator = string_value(node.right) if node.right else None
        if separator is None or node.left is None:
            return None
        # Binary -Join on a scalar string is a no-op.
        scalar = string_value(node.left)
        if scalar is not None:
            return make_string_literal(scalar)
        array = unwrap_to_array_literal(node.left)
        if array is None:
            return None
        args = collect_string_arguments(array)
        if args is None:
            return None
        return make_string_literal(separator.join(args))

    def _handle_binary_replace(
        self, node: Ps1BinaryExpression, op: str,
    ) -> Expression | None:
        haystack = string_value(node.left) if node.left else None
        if haystack is None or node.right is None:
            return None
        if isinstance(node.right, Ps1ArrayLiteral) and len(node.right.elements) == 2:
            needle_str = string_value(node.right.elements[0])
            insert_str = string_value(node.right.elements[1])
        else:
            return None
        if needle_str is None or insert_str is None:
            return None
        flags = re.IGNORECASE if op != '-creplace' else 0
        try:
            result = dotnet_regex_replace(needle_str, insert_str, haystack, flags=flags)
        except re.error:
            return None
        return make_string_literal(result)

    def _handle_binary_split(
        self, node: Ps1BinaryExpression, op: str,
    ) -> Expression | None:
        if node.right is None or node.left is None:
            return None
        pattern_str = string_value(node.right)
        if pattern_str is None:
            return None
        flags = re.IGNORECASE if op != '-csplit' else 0
        left_str = string_value(node.left)
        if left_str is not None:
            inputs = [left_str]
        else:
            array = unwrap_to_array_literal(node.left)
            if array is None:
                return None
            inputs_opt = collect_string_arguments(array)
            if inputs_opt is None:
                return None
            inputs = inputs_opt
        try:
            parts: list[str] = []
            for s in inputs:
                parts.extend(re.split(pattern_str, s, flags=flags))
        except re.error:
            return None
        elements: list[Expression] = [make_string_literal(p) for p in parts]
        return Ps1ArrayLiteral(elements=elements)
