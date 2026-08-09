"""
What a PowerShell expression evaluates to and what type that value carries: the value when the
source pins it, the .NET type when the static surface determines it, and `None` when nothing here
can say. These are language semantics rather than deobfuscation policy — the truth value of `''` and
the type of `'abc'` are properties of PowerShell, not of any pass — so they sit in the analysis
layer where both the effect substrate and the transforms can read them without either importing the
other. This is the only module in `refinery.lib.scripts.ps1.analysis` that answers either question.

The type side has two views over one engine. `resolve_expression_type` is the single-type core: one
expression, one `refinery.lib.scripts.ps1.dotnet.Ps1TypeName` or `None`. `candidate_types` is the
set-valued view the effect layer reasons over, and it is a strict superset — it additionally
resolves a static method call and a cmdlet whose declared output is closed, either of which can name
several types. The set is the primitive and the single type the derived view, because a caller
reasoning about a value must have its conclusion hold for every type the value could carry.
"""
from __future__ import annotations

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


#: The types a literal has, resolved once through the one resolver rather than spelled here. A name
#: written out as text would be a second vocabulary inside the module whose purpose is to have one.
_STRING = resolve_type('System.String')
_INT32 = resolve_type('System.Int32')

#: What an array literal builds. PowerShell collects the elements into an `Object[]` whatever they
#: are, which is measured rather than assumed: an array of integers and an array of strings both
#: report `System.Object[]`. The rank is what makes a member read on one resolve against
#: `System.Array`, which is where an array's members actually live.
_OBJECT_ARRAY = resolve_type('System.Object[]')


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
