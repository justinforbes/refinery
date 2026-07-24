"""
The expression-type oracle of the PowerShell analysis substrate: the .NET type a PowerShell
expression evaluates to, traced through member-access chains, or `None` when it cannot be
determined. It reads the collected type surface in `refinery.lib.scripts.ps1.data` and the syntactic
accessors in `refinery.lib.scripts.ps1.ast`, and nothing else, so it sits at the analysis level
where both the effect layer (`refinery.lib.scripts.ps1.analysis.effects`) and the deobfuscation
transforms can reach it without either importing the other.

Like `effects`, this is a free function that decides a fact about one expression from the shared
database rather than from anywhere else in the tree.
"""
from __future__ import annotations

from refinery.lib.scripts.ps1.ast import (
    extract_first_positional_string,
    get_command_name,
    get_member_name,
    unwrap_parens,
)
from refinery.lib.scripts.ps1.data import (
    OBJ_COMMANDS,
    PROPERTY_TYPES,
    VARIABLE_TYPES,
    WMI_CLASS_NAMES,
    WMI_COMMANDS,
    _resolve_type_name,
)
from refinery.lib.scripts.ps1.model import (
    Expression,
    Ps1ArrayExpression,
    Ps1ArrayLiteral,
    Ps1CastExpression,
    Ps1CommandInvocation,
    Ps1ExpressionStatement,
    Ps1HereString,
    Ps1IntegerLiteral,
    Ps1MemberAccess,
    Ps1StringLiteral,
    Ps1TypeExpression,
    Ps1Variable,
)


def resolve_expression_type(
    expr: Expression,
    variable_types: dict[str, str] | None = None,
) -> str | None:
    """
    Trace the .NET type of a PowerShell expression by walking member access chains. Returns the
    lowercase full .NET type name, or `None` if the type cannot be determined.
    """
    unwrapped = unwrap_parens(expr)
    if not isinstance(unwrapped, Expression):
        return None
    expr = unwrapped
    if isinstance(expr, (Ps1StringLiteral, Ps1HereString)):
        return 'system.string'
    if isinstance(expr, Ps1IntegerLiteral):
        return 'system.int32'
    if isinstance(expr, Ps1ArrayLiteral):
        return 'system.array'
    if isinstance(expr, Ps1ArrayExpression):
        if (
            len(expr.body) == 1
            and isinstance(expr.body[0], Ps1ExpressionStatement)
            and isinstance(expr.body[0].expression, Ps1ArrayLiteral)
        ):
            return 'system.array'
    if isinstance(expr, Ps1Variable):
        key = expr.name.lower()
        if variable_types and key in variable_types:
            return variable_types[key]
        return VARIABLE_TYPES.get(key)
    if isinstance(expr, Ps1TypeExpression):
        return _resolve_type_name(expr.name)
    if isinstance(expr, Ps1CastExpression):
        return _resolve_type_name(expr.type_name)
    if isinstance(expr, Ps1CommandInvocation):
        cmd_name = get_command_name(expr)
        if cmd_name is not None:
            cmd_lower = cmd_name.lower()
            if cmd_lower in OBJ_COMMANDS:
                type_str = extract_first_positional_string(expr)
                if type_str is not None:
                    return _resolve_type_name(type_str)
            elif cmd_lower in WMI_COMMANDS:
                class_str = extract_first_positional_string(expr)
                if class_str is not None:
                    wmi_lower = class_str.lower()
                    if wmi_lower in WMI_CLASS_NAMES:
                        return wmi_lower
    if isinstance(expr, Ps1MemberAccess):
        if expr.object is None:
            return None
        obj_type = resolve_expression_type(expr.object, variable_types)
        if obj_type is None:
            return None
        member_name = get_member_name(expr.member)
        if member_name is None:
            return None
        return PROPERTY_TYPES.get((obj_type, member_name.lower()))
    return None
