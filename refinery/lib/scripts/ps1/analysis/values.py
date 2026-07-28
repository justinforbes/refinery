"""
Constant evaluation of PowerShell expressions: what a node's value is when the source pins it, and
`None` when nothing here can say. These are language semantics rather than deobfuscation policy —
the truth value of `''` is a property of PowerShell, not of any pass — so they sit in the analysis
layer where both the effect substrate and the transforms can read them without either importing the
other.
"""
from __future__ import annotations

from typing import Callable, TypeVar

from refinery.lib.scripts import Node
from refinery.lib.scripts.ps1.ast import is_builtin_variable, unwrap_parens
from refinery.lib.scripts.ps1.model import (
    Expression,
    Ps1ArrayExpression,
    Ps1ArrayLiteral,
    Ps1ExpressionStatement,
    Ps1IntegerLiteral,
    Ps1ParenExpression,
    Ps1RealLiteral,
    Ps1StringLiteral,
    Ps1UnaryExpression,
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
