"""
Constant evaluation of PowerShell expressions: what a node's value is when the source pins it, and
`None` when nothing here can say. These are language semantics rather than deobfuscation policy —
the truth value of `''` is a property of PowerShell, not of any pass — so they sit in the analysis
layer where both the effect substrate and the transforms can read them without either importing the
other.
"""
from __future__ import annotations

from refinery.lib.scripts import Node
from refinery.lib.scripts.ps1.ast import is_builtin_variable, unwrap_parens
from refinery.lib.scripts.ps1.model import (
    Expression,
    Ps1IntegerLiteral,
    Ps1RealLiteral,
    Ps1StringLiteral,
    Ps1UnaryExpression,
)


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
