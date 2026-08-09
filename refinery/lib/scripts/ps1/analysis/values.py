"""
What a PowerShell expression evaluates to and what type that value carries: the value when the
source pins it, the .NET type when the static surface determines it, and *nothing known* when this
module cannot say. These are language semantics rather than deobfuscation policy — the truth value
of `''` and the type of `'abc'` are properties of PowerShell, not of any pass — so they sit in the
analysis layer where both the effect substrate and the transforms can read them without either
importing the other. This is the only module in `refinery.lib.scripts.ps1.analysis` that answers
either question.

**A value and its type are one fact, not two.** `Ps1Fact` is that fact, and it is what makes a Char
and a one-character String different things: both carry the Python string `'A'`, and only the type
they are stamped with tells them apart. Nothing here dispatches on the Python type of a payload —
the `Ps1TypeName` decides, always, because the Python type is the erasure this module exists to
undo. The four elements are *nothing is known*, *`$null`*, *a value of this type*, and *this exact
value of this type*. When an interval or a known-bits refinement is built it becomes a field on
`Ps1Typed`, not a fifth element.

Throwing is a separate axis, which is why an operation answers a `Ps1Outcome` rather than a fact:
`[int] $s` over a String is *an Int32, or it throws*, and a domain that had to fold that into one
element could only answer that it knows nothing. `may_throw` is set only where a measurement
recorded a throw; not knowing is `UNKNOWN`.

The type side has two views over one engine. `resolve_expression_type` is the single-type core: one
expression, one `refinery.lib.scripts.ps1.dotnet.Ps1TypeName` or `None`. `candidate_types` is the
set-valued view the effect layer reasons over, and it is a strict superset — it additionally
resolves a static method call and a cmdlet whose declared output is closed, either of which can name
several types. The set is the primitive and the single type the derived view, because a caller
reasoning about a value must have its conclusion hold for every type the value could carry.
"""
from __future__ import annotations

import dataclasses
import decimal
import math
import operator as operator_module
import re
import typing

from typing import Callable, TypeVar

from refinery.lib.scripts import Node
from refinery.lib.scripts.ps1.ast import (
    extract_first_positional_string,
    get_command_name,
    get_member_name,
    is_builtin_variable,
    unwrap_parens,
)
from refinery.lib.scripts.ps1.analysis.world import Ps1TypeWorld
from refinery.lib.scripts.ps1.data import (
    OBJ_COMMANDS,
    TYPE_ARG_COMMANDS,
    VARIABLE_TYPES,
    WMI_COMMANDS,
    binary_outcome,
    command_output_types,
    resolve_member_type,
    resolve_type,
    static_overloads,
)
from refinery.lib.scripts.ps1.dotnet import Ps1TypeName
from refinery.lib.scripts.ps1.model import (
    Expression,
    Ps1AccessKind,
    Ps1ArrayExpression,
    Ps1ArrayLiteral,
    Ps1CastExpression,
    Ps1CommandInvocation,
    Ps1ExpressionStatement,
    Ps1HereString,
    Ps1IntegerLiteral,
    Ps1InvokeMember,
    Ps1MemberAccess,
    Ps1ParenExpression,
    Ps1RealLiteral,
    Ps1StringLiteral,
    Ps1TypeExpression,
    Ps1UnaryExpression,
    Ps1Variable,
)

_T = TypeVar('_T')


def is_truthy(node: Node | None) -> bool | None:
    """
    Determine the boolean truth value of a constant expression using PowerShell semantics. Returns
    `None` for non-constant or unrecognized expressions.
    """
    node = unwrap_parens(node) if isinstance(node, Expression) else node
    if node is None:
        return None
    if is_builtin_variable(node):
        lower = node.name.lower()
        if lower == 'true':
            return True
        if lower in ('false', 'null'):
            return False
        return None
    if isinstance(node, (Ps1IntegerLiteral, Ps1RealLiteral, Ps1StringLiteral)):
        return bool(node.value)
    if isinstance(node, Ps1UnaryExpression) and node.operator == '-':
        return is_truthy(node.operand)
    return None


def unwrap_integer(node: Node | None) -> Ps1IntegerLiteral | None:
    """
    Peel parentheses and unary negation to extract an integer literal, or return `None`.
    """
    node = unwrap_parens(node) if isinstance(node, Expression) else node
    if isinstance(node, Ps1IntegerLiteral):
        return node
    if is_builtin_variable(node, {'null'}):
        return Ps1IntegerLiteral(value=0, raw='0')
    if isinstance(node, Ps1UnaryExpression) and node.operator == '-':
        inner = unwrap_parens(node.operand) if isinstance(node.operand, Expression) else node.operand
        if isinstance(inner, Ps1IntegerLiteral):
            return Ps1IntegerLiteral(value=-inner.value, raw=str(-inner.value))
    return None


def unwrap_to_array_literal(node: Node) -> Ps1ArrayLiteral | None:
    """
    Unwrap parentheses and array expressions to find an inner
    `refinery.lib.scripts.ps1.model.Ps1ArrayLiteral`.
    """
    node = unwrap_parens(node)
    if isinstance(node, Ps1ArrayLiteral):
        return node
    if isinstance(node, Ps1ArrayExpression) and len(node.body) == 1:
        stmt = node.body[0]
        if isinstance(stmt, Ps1ExpressionStatement) and isinstance(stmt.expression, Ps1ArrayLiteral):
            return stmt.expression
    return None


def collect_typed_arguments(
    node: Expression, extract: Callable[[Expression], _T | None],
) -> list[_T] | None:
    if isinstance(node, Ps1ArrayLiteral):
        result: list[_T] = []
        for elem in node.elements:
            value = extract(elem)
            if value is None:
                return None
            result.append(value)
        return result
    value = extract(node)
    if value is not None:
        return [value]
    return None


def extract_int(node: Expression) -> int | None:
    return node.value if isinstance(node, Ps1IntegerLiteral) else None


def collect_int_arguments(node: Expression) -> list[int] | None:
    if isinstance(node, Ps1ParenExpression) and node.expression is not None:
        return collect_int_arguments(node.expression)
    return collect_typed_arguments(node, extract_int)


def collect_byte_array(node: Expression) -> bytes | None:
    """
    Extract an integer array from `node` and convert to `bytes`. Handles
    `refinery.lib.scripts.ps1.model.Ps1ArrayLiteral`,
    `refinery.lib.scripts.ps1.model.Ps1ArrayExpression`, and parenthesized wrappers.
    """
    array = unwrap_to_array_literal(node)
    if array is not None:
        node = array
    elif isinstance(node, Ps1ArrayExpression):
        items: list[int] = []
        for stmt in node.body:
            if not isinstance(stmt, Ps1ExpressionStatement) or stmt.expression is None:
                return None
            value = extract_int(stmt.expression)
            if value is None:
                return None
            items.append(value)
        try:
            return bytes(items)
        except (ValueError, OverflowError):
            return None
    values = collect_int_arguments(node)
    if values is None:
        return None
    try:
        return bytes(values)
    except (ValueError, OverflowError):
        return None


def _type(name: str) -> Ps1TypeName:
    """
    A type this module names, resolved through the one resolver rather than spelled here — a name
    written out as text would be a second vocabulary inside the module whose purpose is to have one.
    A name the table does not resolve raises at import: every answer below is keyed by the result,
    so a missing row would not move an answer, it would make every comparison silently false.
    """
    resolved = resolve_type(name)
    if resolved is None:
        raise ValueError(F'the collected type table does not resolve {name}')
    return resolved


#: The types a literal has.
_STRING = _type('System.String')
_INT32 = _type('System.Int32')

#: What an array literal builds. PowerShell collects the elements into an `Object[]` whatever they
#: are, which is measured rather than assumed: an array of integers and an array of strings both
#: report `System.Object[]`. The rank is what makes a member read on one resolve against
#: `System.Array`, which is where an array's members actually live.
_OBJECT_ARRAY = _type('System.Object[]')


def resolve_expression_type(
    expr: Expression,
    variable_types: dict[str, Ps1TypeName] | None = None,
) -> Ps1TypeName | None:
    """
    Trace the .NET type of a PowerShell expression by walking member access chains. Returns the one
    canonical `Ps1TypeName`, or `None` if the type cannot be determined.
    """
    unwrapped = unwrap_parens(expr)
    if not isinstance(unwrapped, Expression):
        return None
    expr = unwrapped
    if isinstance(expr, (Ps1StringLiteral, Ps1HereString)):
        return _STRING
    if isinstance(expr, Ps1IntegerLiteral):
        return _INT32
    if isinstance(expr, Ps1ArrayLiteral):
        return _OBJECT_ARRAY
    if isinstance(expr, Ps1ArrayExpression):
        if (
            len(expr.body) == 1
            and isinstance(expr.body[0], Ps1ExpressionStatement)
            and isinstance(expr.body[0].expression, Ps1ArrayLiteral)
        ):
            return _OBJECT_ARRAY
    if isinstance(expr, Ps1Variable):
        key = expr.name.lower()
        if variable_types and key in variable_types:
            return variable_types[key]
        declared = VARIABLE_TYPES.get(key)
        return None if declared is None else resolve_type(declared)
    if isinstance(expr, Ps1TypeExpression):
        return resolve_type(expr.name)
    if isinstance(expr, Ps1CastExpression):
        return resolve_type(expr.type_name)
    if isinstance(expr, Ps1CommandInvocation):
        cmd_name = get_command_name(expr)
        if cmd_name is not None:
            cmd_lower = cmd_name.lower()
            if cmd_lower in OBJ_COMMANDS:
                type_str = extract_first_positional_string(expr)
                if type_str is not None:
                    return resolve_type(type_str)
            elif cmd_lower in WMI_COMMANDS:
                class_str = extract_first_positional_string(expr)
                if class_str is not None:
                    return resolve_type(class_str)
    if isinstance(expr, Ps1MemberAccess):
        if expr.object is None:
            return None
        obj_type = resolve_expression_type(expr.object, variable_types)
        if obj_type is None:
            return None
        member_name = get_member_name(expr.member)
        if member_name is None:
            return None
        return resolve_member_type(obj_type, member_name)
    return None


#: Commands whose declared `[OutputType]` is a trustworthy *superset* of what they emit at runtime,
#: not merely a lower bound. Most commands under-declare: one that forwards its input emits the
#: input's type, which it never lists — `Get-Random -InputObject $procs` returns a `Process`,
#: `Get-Content` on a non-filesystem provider returns whatever that provider yields — so trusting
#: the declaration lets the member gate prove `(...).Path` pure over an incomplete candidate set and
#: delete a live effect. Only commands that emit their own output and cannot pass input through
#: belong here; a read on any other command's result stays unresolved, and therefore kept.
_CLOSED_OUTPUT_CMDLETS = frozenset({
    'get-date',
})


def candidate_types(
    expr: Expression,
    world: Ps1TypeWorld,
    variable_types: dict[str, Ps1TypeName] | None = None,
) -> frozenset[Ps1TypeName]:
    """
    The set of canonical .NET type names the expression's value could have, or the empty set when
    the type cannot be determined. A static method call contributes the return its overloads agree
    on, and a cmdlet call the output types it declares; either can be several, so a caller reasoning
    about the value must have its conclusion hold for every candidate. The single-type forms —
    literals, variables, casts, `New-Object`, WMI, and property chains — are delegated to
    `resolve_expression_type` rather than re-derived here.

    `world` is what decides whether a command name still denotes what the metadata says, so a cmdlet
    whose name the script has taken over contributes nothing. An open world trusts no name.
    """
    unwrapped = unwrap_parens(expr)
    if not isinstance(unwrapped, Expression):
        return frozenset()
    expr = unwrapped
    if isinstance(expr, Ps1InvokeMember):
        return _static_method_candidates(expr)
    if isinstance(expr, Ps1CommandInvocation):
        return _command_candidates(expr, world, variable_types)
    single = resolve_expression_type(expr, variable_types)
    return frozenset() if single is None else frozenset({single})


def _static_method_candidates(node: Ps1InvokeMember) -> frozenset[Ps1TypeName]:
    """
    The return type of a `[Type]::Method(...)` call, taken only when every matching static overload
    agrees on it; disagreement or an unrecognized call is the empty set. An instance method call is
    not resolved — its receiver type would have to be traced and its overloads selected by argument
    type — so it contributes nothing rather than a guess.
    """
    if node.access is not Ps1AccessKind.STATIC:
        return frozenset()
    obj = node.object
    member = node.member
    if not isinstance(obj, Ps1TypeExpression) or not isinstance(member, str):
        return frozenset()
    returns = {
        resolve_type(overload['returns'])
        for overload in static_overloads(obj.name, member)
        if overload.get('returns')
    }
    if len(returns) != 1:
        return frozenset()
    single = next(iter(returns))
    return frozenset() if single is None else frozenset({single})


def _command_candidates(
    cmd: Ps1CommandInvocation,
    world: Ps1TypeWorld,
    variable_types: dict[str, Ps1TypeName] | None,
) -> frozenset[Ps1TypeName]:
    """
    The types a command's result could have: the constructed or queried type for the `New-Object`
    and WMI forms the single-type ladder already knows, otherwise the output types a command
    declares through `[OutputType]` — but only for a command whose declaration is a trustworthy
    *superset* of what it emits (`_CLOSED_OUTPUT_CMDLETS`). `[OutputType]` is a lower bound in
    general: a command that forwards its input emits types it never declares, and trusting the
    declaration there would let the member gate prove an effectful read pure over an incomplete
    candidate set. Every other command contributes nothing, so a read on its result stays
    unresolved and is kept.
    """
    name = get_command_name(cmd)
    if name is None:
        return frozenset()
    lower = name.lower()
    if not world.may_trust_command_name(lower, cmd):
        return frozenset()
    if lower in TYPE_ARG_COMMANDS:
        single = resolve_expression_type(cmd, variable_types)
        return frozenset() if single is None else frozenset({single})
    if lower not in _CLOSED_OUTPUT_CMDLETS:
        return frozenset()
    declared = command_output_types(name)
    if declared is None:
        return frozenset()
    resolved = {resolve_type(one) for one in declared}
    return frozenset(one for one in resolved if one is not None)


#: The types the domain names. Resolved once through the one resolver, so that a fact carries the
#: same `Ps1TypeName` a member lookup or a grid cell is keyed by and the two can be compared.
_BYTE = _type('System.Byte')
_BOOLEAN = _type('System.Boolean')
_CHAR = _type('System.Char')
_DECIMAL = _type('System.Decimal')
_DOUBLE = _type('System.Double')
_INT16 = _type('System.Int16')
_INT64 = _type('System.Int64')
_SBYTE = _type('System.SByte')
_UINT16 = _type('System.UInt16')
_UINT32 = _type('System.UInt32')
_UINT64 = _type('System.UInt64')

#: The widths the numeric ladder is written in terms of. `System.Decimal`'s bound is its documented
#: maximum rather than a power of two, because its range is not a bit width.
_INT32_RANGE = (-0x80000000, 0x7FFFFFFF)
_INT64_RANGE = (-0x8000000000000000, 0x7FFFFFFFFFFFFFFF)
_DECIMAL_MAX = decimal.Decimal('79228162514264337593543950335')


class Ps1Fact:
    """
    What is known about one PowerShell value: nothing (`UNKNOWN`), that it is `$null` (`NULL`), that
    it has a type (`Ps1Typed`), or that it is a particular value of a particular type
    (`Ps1Constant`). These are the four elements of the lattice every question in this module is
    answered in, ordered `Ps1Constant` below `Ps1Typed` below `UNKNOWN`, with `NULL` beside the
    typed ones rather than under them: `$null.GetType()` throws, so there is no type it could carry.

    The base is a marker and carries no accessor: `type_of` is where a fact's type is read, so that
    the one place a caller asks the question is a function it can be pointed at, and an element that
    has no type does not have to pretend to answer.
    """

    __slots__ = ()


@dataclasses.dataclass(frozen=True)
class _Ps1Unknown(Ps1Fact):
    """
    Nothing is known about the value. This is the answer to every question this module declines,
    and it never means *the value is absent* — that is `NULL`.
    """

    def __repr__(self) -> str:
        return 'UNKNOWN'


@dataclasses.dataclass(frozen=True)
class _Ps1Null(Ps1Fact):
    """
    The value is `$null`. Its own element rather than a `Ps1Constant` of some type, because it has
    no type to be constant *of*: reading `GetType()` off it throws.
    """

    def __repr__(self) -> str:
        return 'NULL'


UNKNOWN: Ps1Fact = _Ps1Unknown()
NULL: Ps1Fact = _Ps1Null()


@dataclasses.dataclass(frozen=True)
class Ps1Typed(Ps1Fact):
    """
    The value has this type and no more is known about it. A refinement — an interval, a known-bits
    mask — becomes a field here when one is built, so that narrowing what a typed value can be does
    not add an element to the lattice or a case to any caller.
    """

    type: Ps1TypeName

    def __repr__(self) -> str:
        return F'Typed({self.type})'


@dataclasses.dataclass(frozen=True)
class Ps1Constant(Ps1Fact):
    """
    The value is exactly `payload`, and its type is `type`. The payload's Python type is an
    implementation of the .NET one and never a substitute for it: `Ps1Constant(System.Char, 'A')`
    and `Ps1Constant(System.String, 'A')` hold equal payloads and are different values, which is the
    distinction this whole layer exists to keep. A caller deciding what a payload means reads
    `type`.

    An array's payload is a tuple of facts rather than of payloads, so that an `Object[]` whose
    elements are Chars is a different value from one whose elements are Strings — the fact a
    pipeline builds and the erasure that made `foreach` iterate once over a joined string.
    """

    type: Ps1TypeName
    payload: int | float | decimal.Decimal | str | bool | tuple[Ps1Fact, ...]

    def __repr__(self) -> str:
        return F'Constant({self.type}, {self.payload!r})'


class Ps1Outcome(typing.NamedTuple):
    """
    What an operation does: the fact it produces, and whether it may instead throw. The two are
    separate because they are not alternatives — an operation that yields an Int32 *or* throws is
    both, and a domain that had to choose could only answer `UNKNOWN` and lose the type it knows.

    `may_throw` is asserted only where a measurement recorded a throw. An operation this module
    cannot decide answers `Ps1Outcome(False, UNKNOWN)`: not knowing what happens is not a claim that
    nothing is thrown, and a caller must read `UNKNOWN` as the refusal it is.
    """

    may_throw: bool
    value: Ps1Fact


#: The refusal, named once so that the several places that decline read alike.
NOTHING = Ps1Outcome(False, UNKNOWN)


def type_of(fact: Ps1Fact) -> Ps1TypeName | None:
    """
    The .NET type a fact carries, or `None` for `UNKNOWN` and `NULL`. `None` is *no type is named
    here* in both cases, and a caller that needs to tell them apart compares against `NULL`.
    """
    if isinstance(fact, (Ps1Typed, Ps1Constant)):
        return fact.type
    return None


#: What a numeric multiplier suffix multiplies by, and the whole set of them. Measured, the
#: multiplier applies to whatever the numeral is and the *result* is then typed by the rule that
#: numeral's form uses: `1kb` is an Int32 1024, `4gb` an Int64 4294967296, `1lkb` an Int64 1024 and
#: `1.5kb` a Double 1536.
_MULTIPLIERS = {
    'kb': 1 << 10,
    'mb': 1 << 20,
    'gb': 1 << 30,
    'tb': 1 << 40,
    'pb': 1 << 50,
}

_DECIMAL_DIGITS = re.compile(r'[0-9]+\Z')
_REAL_DIGITS = re.compile(r'(?:[0-9]*\.[0-9]+|[0-9]+\.?)(?:e[+-]?[0-9]+)?\Z', re.IGNORECASE)
_HEX_DIGITS = re.compile(r'[0-9a-f]+\Z', re.IGNORECASE)


def read(node: Node | None) -> Ps1Fact:
    """
    What the source pins this expression to, as a fact, or `UNKNOWN` when it pins nothing. This is
    the floor the rest of the domain stands on and it **refuses rather than invents**: an expression
    it cannot decide, a literal spelled in a way no measurement covers, and a number too wide for
    any type all answer `UNKNOWN`, never a value that happens to be close.

    Only literal structure is read — literals, the array and parenthesis forms that wrap them,
    `$true`, `$false` and `$null`. A cast is `convert` and an operator is not read at all, so that a
    caller asking what the *source* says never receives an answer that came from evaluating
    something.

    The one exception looks like an operator and is not: a `-` written *directly against* a numeral
    is part of the numeral, which is measurable rather than stylistic and turns on the whitespace.
    `-2147483648` is one literal and fits Int32; `- 2147483648` and `-(2147483648)` are both unary
    minus over the Int64 literal `2147483648` and stay Int64. Only the glued spelling is read here,
    and it is the source offsets that say which one it is. The other two are an operator over a
    value and belong to `apply`, so they are refused rather than answered — reaching past the space
    or the parenthesis to the numeral would report the Int32 that only the third spelling has.
    """
    if node is None:
        return UNKNOWN
    if isinstance(node, Ps1ParenExpression):
        return UNKNOWN if node.expression is None else read(node.expression)
    if isinstance(node, (Ps1StringLiteral, Ps1HereString)):
        return Ps1Constant(_STRING, node.value)
    if isinstance(node, (Ps1IntegerLiteral, Ps1RealLiteral)):
        return _numeral(node.raw)
    if is_builtin_variable(node, {'true'}):
        return Ps1Constant(_BOOLEAN, True)
    if is_builtin_variable(node, {'false'}):
        return Ps1Constant(_BOOLEAN, False)
    if is_builtin_variable(node, {'null'}):
        return NULL
    if isinstance(node, Ps1UnaryExpression) and node.operator == '-':
        if _is_signed_numeral(node):
            return _numeral(F'-{node.operand.raw}')  # type: ignore[union-attr]
        return UNKNOWN
    if isinstance(node, (Ps1ArrayLiteral, Ps1ArrayExpression)):
        return _array(node)
    return UNKNOWN


def _is_signed_numeral(node: Ps1UnaryExpression) -> bool:
    """
    Whether this `-` is the sign of the numeral it stands before rather than an operator applied to
    it. It is the sign exactly when nothing separates the two in the source, which the offsets say:
    `-2147483648` is one Int32 literal and `- 2147483648` is unary minus over an Int64.

    A node the source did not place — one a fold built — has no offsets to compare and is not read
    as glued. That refuses rather than misreads, and nothing builds this shape anyway: a rendered
    negative number carries its sign in its own spelling.
    """
    operand = node.operand
    if not isinstance(operand, (Ps1IntegerLiteral, Ps1RealLiteral)):
        return False
    return operand.offset == node.offset + len(node.operator)


def _array(node: Ps1ArrayLiteral | Ps1ArrayExpression) -> Ps1Fact:
    """
    An array literal as a fact whose payload is the facts of its elements. One element this module
    cannot read makes the whole array unknown: a caller reasoning about the array would otherwise be
    handed a shorter one than the script builds.
    """
    elements: list[Expression] = []
    if isinstance(node, Ps1ArrayLiteral):
        elements.extend(node.elements)
    else:
        for statement in node.body:
            if not isinstance(statement, Ps1ExpressionStatement) or statement.expression is None:
                return UNKNOWN
            inner = statement.expression
            if isinstance(inner, Ps1ArrayLiteral):
                elements.extend(inner.elements)
            else:
                elements.append(inner)
    facts = tuple(read(element) for element in elements)
    if any(fact is UNKNOWN for fact in facts):
        return UNKNOWN
    return Ps1Constant(_OBJECT_ARRAY, facts)


def _numeral(raw: str) -> Ps1Fact:
    """
    The fact a numeric literal's spelling denotes, measured rather than derived from the digits
    alone: the same digits are an Int32, an Int64, a Decimal or a Double depending on how wide they
    are and what is written after them.

    A spelling no measurement covers is refused. `_` is one: PowerShell 5.1 has no digit separator
    and reads `1_0` as a command name, so a lexer that accepts it must not be allowed to hand the
    domain the number ten.
    """
    if '_' in raw:
        return UNKNOWN
    text = raw
    sign = 1
    if text[:1] in ('-', '+'):
        sign = -1 if text[0] == '-' else 1
        text = text[1:]
    multiplier = 1
    lowered = text.lower()
    for suffix, factor in _MULTIPLIERS.items():
        if lowered.endswith(suffix):
            multiplier = factor
            text = text[:-len(suffix)]
            break
    if text[:2].lower() == '0x':
        return _hex_numeral(text[2:], sign, multiplier)
    return _decimal_numeral(text, sign, multiplier)


def _hex_numeral(digits: str, sign: int, multiplier: int) -> Ps1Fact:
    """
    A hexadecimal literal, which names a *bit pattern* rather than a magnitude: measured, `0xFF` is
    255, `0xFFFFFFFF` is Int32 -1 because eight digits fill an Int32, `0x100000000` is Int64
    4294967296 and seventeen digits fit nothing, which 5.1 reports as a parse error.

    A `L` suffix changes the question from *which width holds this pattern* to *read these digits as
    an Int64*, so `0xFFFFFFFFL` is 4294967295 rather than -1.

    The width the pattern fills is a *floor* on the result type and not merely a step on the way to
    it. `0xFFFFFFFFFFFFFFFF` is Int64 -1: the value -1 would fit an Int32, but the sixteen digits
    said which width was being filled, and narrowing back to what the number needs would report a
    type no value in the script has.

    A multiplier over a pattern that had to be reinterpreted as negative is refused: composing the
    two rules would answer where nothing was measured, and nothing here answers from a composition.
    """
    long_suffix = digits[-1:].lower() == 'l'
    if long_suffix:
        digits = digits[:-1]
    if not _HEX_DIGITS.match(digits):
        return UNKNOWN
    magnitude = int(digits, 16)
    if long_suffix:
        if magnitude > _INT64_RANGE[1]:
            return UNKNOWN
        return _long(sign * magnitude * multiplier)
    if magnitude <= 0xFFFFFFFF:
        width = _INT32
        value = magnitude - 0x100000000 if magnitude > _INT32_RANGE[1] else magnitude
    elif magnitude <= 0xFFFFFFFFFFFFFFFF:
        width = _INT64
        value = magnitude - 0x10000000000000000 if magnitude > _INT64_RANGE[1] else magnitude
    else:
        return UNKNOWN
    if value < 0 and multiplier != 1:
        return UNKNOWN
    return _no_narrower_than(width, sign * value * multiplier)


def _decimal_numeral(text: str, sign: int, multiplier: int) -> Ps1Fact:
    """
    A decimal literal. Without a suffix it takes the narrowest of Int32, Int64, Decimal and Double
    that holds it — measured all the way up, `2147483648` being Int64, `9223372036854775808` Decimal
    and `10^32` Double. A `L` or `D` suffix names the type instead, and over a real that is a
    conversion rather than a refusal: `1.5L` is Int64 2 and `2.5L` is Int64 2, which is the
    half-to-even rounding a cast performs.
    """
    suffix = text[-1:].lower()
    if suffix in ('l', 'd'):
        text = text[:-1]
    else:
        suffix = ''
    if _DECIMAL_DIGITS.match(text):
        magnitude = int(text) * multiplier
        if suffix == 'l':
            return _long(sign * magnitude)
        if suffix == 'd':
            return Ps1Constant(_DECIMAL, decimal.Decimal(sign * magnitude))
        return _widest_needed(sign * magnitude)
    if not _REAL_DIGITS.match(text):
        return UNKNOWN
    if suffix and multiplier != 1:
        return UNKNOWN
    if suffix == 'd':
        return Ps1Constant(_DECIMAL, sign * decimal.Decimal(text))
    if suffix == 'l':
        return _long(sign * round(decimal.Decimal(text)))
    return _double(sign * float(text) * multiplier)


def _widest_needed(value: int) -> Ps1Fact:
    """
    The narrowest type that holds `value`, which is what an unsuffixed decimal literal takes.
    """
    if _INT32_RANGE[0] <= value <= _INT32_RANGE[1]:
        return Ps1Constant(_INT32, value)
    if _INT64_RANGE[0] <= value <= _INT64_RANGE[1]:
        return Ps1Constant(_INT64, value)
    if -_DECIMAL_MAX <= value <= _DECIMAL_MAX:
        return Ps1Constant(_DECIMAL, decimal.Decimal(value))
    return _double(value)


def _no_narrower_than(floor: Ps1TypeName, value: int) -> Ps1Fact:
    """
    The narrowest type that holds `value`, but never narrower than `floor`. What a hexadecimal
    literal fills is a width, so the width is what it has however small the number it denotes is.
    """
    fact = _widest_needed(value)
    if floor == _INT64 and isinstance(fact, Ps1Constant) and fact.type == _INT32:
        return Ps1Constant(_INT64, value)
    return fact


def _long(value: int) -> Ps1Fact:
    return Ps1Constant(_INT64, value) if _INT64_RANGE[0] <= value <= _INT64_RANGE[1] else UNKNOWN


def _double(value: int | float) -> Ps1Fact:
    try:
        return Ps1Constant(_DOUBLE, float(value))
    except OverflowError:
        return UNKNOWN


#: The integer types the domain computes in, narrowest first, each with the range it holds. The
#: order is what `_stamped` walks to decide which of a cell's candidate types a computed value has,
#: so it is the widening order and not merely a listing.
_INTEGER_WIDTHS: tuple[tuple[Ps1TypeName, int, int], ...] = (
    (_BYTE, 0, 0xFF),
    (_SBYTE, -0x80, 0x7F),
    (_INT16, -0x8000, 0x7FFF),
    (_UINT16, 0, 0xFFFF),
    (_INT32, *_INT32_RANGE),
    (_UINT32, 0, 0xFFFFFFFF),
    (_INT64, *_INT64_RANGE),
    (_UINT64, 0, 0xFFFFFFFFFFFFFFFF),
)

#: The widths a shift masks its count by, from the *left operand's type* rather than from how large
#: its value happens to be. Only the two the mask is documented for are here; a shift over a narrower
#: left operand keeps that operand's type in the grid and is not computed, because what the count is
#: masked by there was never measured.
_SHIFT_WIDTHS = {_INT32: 32, _INT64: 64}

#: The grid's row for `$null`, which the capture collected under the one name that is not a type a
#: value can have.
_VOID = _type('System.Void')


#: What a kernel computes in. A `Decimal` is here because PowerShell has one and Python's is the
#: only faithful carrier for it; a `bool` because a comparison is an operation like any other.
_Number = int | float | bool | decimal.Decimal


class _Throws(Exception):
    """
    Raised by a kernel for an application that PowerShell answers by throwing, so that a throw is
    reported as one rather than as a refusal. The two are different answers: a throw is knowledge.
    """


def apply(operator: str, left: Ps1Fact, right: Ps1Fact) -> Ps1Outcome:
    """
    What `left <operator> right` produces. The *type* comes from the measured grid in
    `refinery.lib.scripts.ps1.data`, never from a rule written here, and the *value* from a kernel
    that is checked against it: a computed value whose type is not one the grid recorded for that
    cell is refused rather than reported, because the grid is what a host did and the kernel is only
    what we believe.

    A cell that recorded a throw stops the kernel being consulted, unless every way that cell can
    throw is one the kernel checks for itself — see `_throws_are_modelled`. Without that exception a
    single throwing pair anywhere in a cell would cost every other pair in it its fold; with it, a
    throw the kernel cannot see is still never folded past.
    """
    cell = binary_outcome(operator, *(_grid_type(left), _grid_type(right)))  # type: ignore[misc]
    if cell is None:
        return NOTHING
    if not cell.may_throw or _throws_are_modelled(operator, left, right):
        try:
            computed = _kernel(operator, left, right)
        except _Throws:
            return Ps1Outcome(True, UNKNOWN)
        if computed is not None:
            stamped = _stamped(computed, cell.types)
            if stamped is not UNKNOWN:
                return Ps1Outcome(False, stamped)
    return _from_cell(cell)


def _grid_type(fact: Ps1Fact) -> Ps1TypeName | None:
    """
    The type a fact is looked up under in the grid. `$null` has none, and the capture recorded its
    row under `System.Void` — the one name in the grid that is not a type any value carries.
    """
    if fact is NULL:
        return _VOID
    return type_of(fact)


def _from_cell(cell) -> Ps1Outcome:
    """
    What a cell says on its own, with nothing computed from the values. One type and no `$null`
    among the outcomes is a typed value; `$null` and nothing else is `$null`, which is a value and
    not an absence — `$null * 5` really is `$null`, and reading that cell as *unknown* would leave a
    caller to guess where a measurement had already answered. Anything wider is a refusal, because a
    caller cannot act on a value that might be either of two types.
    """
    if not cell.types and cell.may_be_null and not cell.may_throw:
        return Ps1Outcome(False, NULL)
    if len(cell.types) == 1 and not cell.may_be_null:
        return Ps1Outcome(cell.may_throw, Ps1Typed(next(iter(cell.types))))
    return Ps1Outcome(cell.may_throw, UNKNOWN)


def _stamped(value: _Number, candidates: frozenset[Ps1TypeName]) -> Ps1Fact:
    """
    The fact a computed value has, given the types the grid recorded for the cell it came out of.

    A `Decimal` and a `Double` are stamped with themselves, and refused where the cell did not
    record that type: a computed `Decimal` reported as a `Double` would be a value the operation
    never had.

    An integer is stamped with the one integer candidate that holds it. *One* is the whole rule:
    where two of them do — a cell such as `SByte + UInt32`, whose outcome set holds both `Int64` and
    `UInt32` because which one a pair takes depends on the signs — nothing here can say which, and
    the value is refused rather than guessed. An integer no candidate holds takes `Decimal` or
    `Double` when the cell recorded one, which is the widening a host performs on overflow and the
    reason `2147483647 + 1` is a Double.
    """
    if isinstance(value, bool):
        return Ps1Constant(_BOOLEAN, value) if _BOOLEAN in candidates else UNKNOWN
    if isinstance(value, int):
        holders = [
            name for name, low, high in _INTEGER_WIDTHS
            if name in candidates and low <= value <= high
        ]
        if len(holders) == 1:
            return Ps1Constant(holders[0], value)
        if holders:
            return UNKNOWN
        if _DECIMAL in candidates and -_DECIMAL_MAX <= value <= _DECIMAL_MAX:
            return Ps1Constant(_DECIMAL, decimal.Decimal(value))
        return _double(value) if _DOUBLE in candidates else UNKNOWN
    if isinstance(value, decimal.Decimal):
        return Ps1Constant(_DECIMAL, value) if _DECIMAL in candidates else UNKNOWN
    return Ps1Constant(_DOUBLE, value) if _DOUBLE in candidates else UNKNOWN


def _kernel(operator: str, left: Ps1Fact, right: Ps1Fact) -> _Number | None:
    """
    The value an application produces, or `None` where this module computes nothing for it. Only
    operands that are integers of the domain's own widths are computed: a Decimal, a Double, a
    String or a collection reaches the grid for its type and stops there, so that no arithmetic here
    is performed in a Python type that is not what PowerShell was using.

    `$null` computes as the integer zero, which is what a host converts it to in an arithmetic
    context: `10 - $null` is 10, `$null - 5` is -5 and `$null -band 1` is 0, all measured. It is the
    grid that decides whether the context is arithmetic at all, so a `$null` reaching an operator
    that does something else with it never gets here.

    A shift is computed only over an `Int32` or `Int64` left operand, because the count is masked by
    the *left operand's width* and only those two widths are documented. That the width comes from
    the type and not from how large the value happens to be is the point: a small value in a wide
    variable is still shifted at the wide mask.

    A bitwise operator is computed only over integers. PowerShell will bitwise a Double by rounding
    it first, which is a conversion, and a kernel that reached for Python's operators there would be
    performing a different one.
    """
    if operator in ('-shl', '-shr'):
        if not _is_domain_integer(left) or not _is_domain_integer(right):
            return None
        left_type = type_of(left)
        width = None if left_type is None else _SHIFT_WIDTHS.get(left_type)
        if width is None:
            return None
        count = _integer_payload(right) & (width - 1)
        return _shifted(_integer_payload(left), count, width, operator == '-shl')
    if operator in _BITWISE:
        if not _is_domain_integer(left) or not _is_domain_integer(right):
            return None
        return _BITWISE[operator](_integer_payload(left), _integer_payload(right))
    operands = _numeric_pair(left, right)
    if operands is None:
        return None
    a, b = operands
    if operator in _COMPARISONS:
        return _COMPARISONS[operator](a, b)
    if operator == '/':
        if b == 0:
            raise _Throws
        if isinstance(a, int) and isinstance(b, int) and a % b == 0:
            return a // b
        return _decimal_result(operator_module.truediv(a, b))
    if operator == '%':
        if b == 0:
            raise _Throws
        if isinstance(a, int) and isinstance(b, int):
            remainder = abs(a) % abs(b)
            return -remainder if a < 0 else remainder
        if isinstance(a, decimal.Decimal) and isinstance(b, decimal.Decimal):
            return _decimal_result(a % b)
        return _finite(math.fmod(a, b))
    arithmetic = _ARITHMETIC.get(operator)
    return None if arithmetic is None else _decimal_result(arithmetic(a, b))


def _numeric_pair(left: Ps1Fact, right: Ps1Fact):
    """
    The two payloads as Python numbers that compute in the same way PowerShell's promoted pair does,
    or `None` where they do not. An integer beside a Decimal computes as a Decimal and an integer
    beside a Double as a Double, which is what the promotion does; a Decimal beside a Double is
    refused, because Python will not mix them and choosing one to convert would be performing the
    promotion rather than reading it.
    """
    kinds = []
    values: list[int | float | decimal.Decimal] = []
    for fact in (left, right):
        if _is_domain_integer(fact):
            kinds.append('i')
            values.append(_integer_payload(fact))
            continue
        if not isinstance(fact, Ps1Constant):
            return None
        if fact.type == _DECIMAL and isinstance(fact.payload, decimal.Decimal):
            kinds.append('m')
        elif fact.type == _DOUBLE and isinstance(fact.payload, float):
            kinds.append('f')
        else:
            return None
        values.append(fact.payload)
    if 'm' in kinds and 'f' in kinds:
        return None
    if 'm' in kinds:
        return decimal.Decimal(values[0]), decimal.Decimal(values[1])
    if 'f' in kinds:
        return float(values[0]), float(values[1])
    return values[0], values[1]


def _decimal_result(value: _Number) -> _Number | None:
    """
    A computed number, with a `Decimal` that has left the range of a `Decimal` reported as the throw
    it is. Python carries such a value without complaint; .NET does not have it, so neither does the
    domain, and calling it a throw is what the host does rather than a refusal.
    """
    if isinstance(value, decimal.Decimal) and not -_DECIMAL_MAX <= value <= _DECIMAL_MAX:
        raise _Throws
    return _finite(value)


def _finite(value: _Number) -> _Number | None:
    """
    A computed number, unless it is one no literal spells. An overflow to infinity is a value
    PowerShell has and this domain deliberately does not carry, because every use of it downstream
    would have to refuse anyway and a fact that cannot be spelled is worse than no fact.
    """
    if isinstance(value, float) and (value != value or value in (INFINITY, -INFINITY)):
        return None
    return value


def _is_domain_integer(fact: Ps1Fact) -> bool:
    if fact is NULL:
        return True
    return isinstance(fact, Ps1Constant) and isinstance(fact.payload, int) and not isinstance(
        fact.payload, bool) and any(name == fact.type for name, _, _ in _INTEGER_WIDTHS)


def _integer_payload(fact: Ps1Fact) -> int:
    return 0 if fact is NULL else typing.cast(int, typing.cast(Ps1Constant, fact).payload)


def _shifted(value: int, count: int, width: int, left: bool) -> int:
    """
    A shift performed in a `width`-bit two's complement register, which is where PowerShell performs
    it: shifting left out of the register discards the bits rather than growing the number, and
    shifting right preserves the sign.
    """
    if not left:
        return value >> count
    span = 1 << width
    result = (value << count) & (span - 1)
    return result - span if result >= span >> 1 else result


#: The value a computed Double reaches on overflow, which the domain does not carry.
INFINITY = float('inf')


def _throws_are_modelled(operator: str, left: Ps1Fact, right: Ps1Fact) -> bool:
    """
    Whether the kernel can see, for these operands, every way the operator throws — so that a cell
    which recorded a throw somewhere may still be computed here.

    Division and remainder throw for a zero divisor and nothing else, and the divisor is in hand.
    Addition, subtraction and multiplication throw only where a `Decimal` result leaves the range of
    a `Decimal`, which `_decimal_result` raises for; over the other numeric types they do not throw
    at all, and a cell of theirs that recorded one is recording something this does not model.
    """
    if operator in ('/', '%'):
        return True
    if operator in _ARITHMETIC:
        return _DECIMAL in (type_of(left), type_of(right))
    return False


_ARITHMETIC = {
    '+': operator_module.add,
    '-': operator_module.sub,
    '*': operator_module.mul,
}

_BITWISE: dict[str, Callable[[int, int], int]] = {
    '-band': int.__and__,
    '-bor': int.__or__,
    '-bxor': int.__xor__,
}

_COMPARISONS = {
    '-eq': operator_module.eq,
    '-ne': operator_module.ne,
    '-lt': operator_module.lt,
    '-le': operator_module.le,
    '-gt': operator_module.gt,
    '-ge': operator_module.ge,
}


#: The literal suffix that pins a spelled number to its type, for the types that have one. A type
#: absent here has no literal spelling at all, and `render` refuses it rather than spelling a number
#: that would re-read as something else: `[byte] 1` written back as `1` is an Int32.
_LITERAL_SUFFIX = {_INT32: '', _INT64: 'L', _DECIMAL: 'd'}


def render(fact: Ps1Fact) -> Expression | None:
    """
    The expression that spells this value, or `None` when nothing spells it. `None` is a **refusal**
    and never *the value is not constant*: a caller holding a `Ps1Constant` it cannot render has to
    leave the source alone, because every alternative it might reach for spells a different value.

    What is refused, and why each one is a value and not a gap:

    - A number of a type with no literal suffix. `[uint16] 7` spelled `7` is an Int32, and there is
      no spelling of a UInt16 literal for it to take instead.
    - A `Double` that is not finite, and negative zero. `Infinity` and `NaN` have no literal at all
      — written out they read as command names — and `-0.0` spelled `-0` re-reads as an Int32.
    - A `Char`, a `String` and a collection, which are not this commit's to spell: a `Char` needs
      the question of whether a `String` spelling is observationally equivalent where it stands, and
      a collection needs the empty and one-element forms that change type when written naively.

    A number is spelled with its sign attached to the digits, which is the spelling that keeps its
    type: `-2147483648` is one literal that fits Int32, and a caller putting the result somewhere a
    parenthesis would separate the two has changed an Int32 into an Int64.
    """
    if fact is NULL:
        return Ps1Variable(name='Null')
    if not isinstance(fact, Ps1Constant):
        return None
    if fact.type == _BOOLEAN:
        return Ps1Variable(name='True' if fact.payload else 'False')
    if fact.type == _DOUBLE:
        return _rendered_double(fact.payload)
    suffix = _LITERAL_SUFFIX.get(fact.type)
    if suffix is None or not isinstance(fact.payload, (int, decimal.Decimal)):
        return None
    if fact.type == _DECIMAL:
        return Ps1RealLiteral(value=float(fact.payload), raw=F'{fact.payload}{suffix}')
    return Ps1IntegerLiteral(value=int(fact.payload), raw=F'{fact.payload}{suffix}')


def _rendered_double(payload) -> Expression | None:
    if not isinstance(payload, float) or payload != payload or payload in (
        float('inf'), float('-inf')
    ):
        return None
    if payload == 0.0 and math.copysign(1.0, payload) < 0:
        return None
    return Ps1RealLiteral(value=payload, raw=repr(payload))
