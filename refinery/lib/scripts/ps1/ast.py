"""
Accessors over the PowerShell node model: small, total functions that read a shape out of a
`refinery.lib.scripts.ps1.model` node without interpreting it. They live at the language level
because both the analysis substrate (`refinery.lib.scripts.ps1.analysis`) and the deobfuscation
transforms (`refinery.lib.scripts.ps1.deobfuscation`) need them, and neither subsystem may import
from the other.

Nothing here decides anything. A function that answers a semantic question — whether a write is
dead, whether an expression is pure, whether a body's value is observed — belongs to the analysis
layer instead.
"""
from __future__ import annotations

from typing import TypeGuard

from refinery.lib.scripts import Block, Node
from refinery.lib.scripts.ps1.data import BUILTIN_VARIABLES
from refinery.lib.scripts.ps1.model import (
    Expression,
    Ps1ArrayLiteral,
    Ps1AssignmentExpression,
    Ps1CastExpression,
    Ps1Code,
    Ps1CommandArgument,
    Ps1CommandArgumentKind,
    Ps1CommandInvocation,
    Ps1ParenExpression,
    Ps1ScopeModifier,
    Ps1StringLiteral,
    Ps1SubExpression,
    Ps1Variable,
)


def get_body(node) -> list | None:
    """
    The statement list that `node` owns, or `None` when it owns none.

    A `refinery.lib.scripts.ps1.model.Ps1ArrayExpression` also has a `body` and is deliberately
    excluded: the cleanup passes recognize a prunable body only through this accessor, so returning
    it here would drop the contents of `@( ... )` — a captured value — into their pruning walks.
    """
    if isinstance(node, (Ps1Code, Block, Ps1SubExpression)):
        return node.body
    return None


def get_named_blocks(node) -> list[Block]:
    """
    The `begin`, `process`, `end` and `dynamicparam` blocks a
    `refinery.lib.scripts.ps1.model.Ps1Code` node owns.

    The parser fills either these or `body`, never both, so `get_body` reports an empty list for an
    advanced function whose whole implementation sits in a named block. `get_body` deliberately does
    not merge them — they are not the single statement list a pass can rebuild with
    `refinery.lib.scripts.set_body` — so any caller that reads "no statements" as "nothing happens
    here" has to ask this as well.
    """
    if not isinstance(node, Ps1Code):
        return []
    blocks = (node.begin_block, node.process_block, node.end_block, node.dynamicparam_block)
    return [block for block in blocks if block is not None]


def get_command_name(cmd: Ps1CommandInvocation) -> str | None:
    """
    The literal name a command is invoked under, or `None` when the name is computed (`& $cmd`, an
    expandable string) and therefore not statically known.
    """
    if isinstance(cmd.name, Ps1StringLiteral):
        return cmd.name.value
    return None


def extract_new_object(cmd: Ps1CommandInvocation) -> tuple[str, list[Expression]] | None:
    """
    Extract the type name and constructor arguments from a `New-Object` invocation. Returns
    `(type_name, [arg_expressions])`, or `None` when `cmd` is not a resolvable `New-Object` call.

    `New-Object` binds only two positional parameters, the type name and the argument list, so a
    third positional argument does not resolve. Reporting the first two and dropping the rest would
    hand every caller a shape that leaves part of the call unexamined — that is how a purity check
    came to clear a `New-Object` whose trailing argument runs a command.
    """
    if not isinstance(cmd.name, Ps1StringLiteral):
        return None
    if cmd.name.value.lower() != 'new-object':
        return None
    positional: list[Expression] = []
    for arg in cmd.arguments:
        if isinstance(arg, Ps1CommandArgument):
            if arg.kind != Ps1CommandArgumentKind.POSITIONAL or arg.value is None:
                return None
            positional.append(arg.value)
        elif isinstance(arg, Expression):
            positional.append(arg)
        else:
            return None
    if not positional or len(positional) > 2:
        return None
    type_name_expr = positional[0]
    if not isinstance(type_name_expr, Ps1StringLiteral):
        return None
    type_name = type_name_expr.value
    ctor_args: list[Expression] = []
    if len(positional) == 2:
        second = positional[1]
        if isinstance(second, Ps1ParenExpression) and second.expression is not None:
            inner = second.expression
            if isinstance(inner, Ps1ArrayLiteral):
                ctor_args = list(inner.elements)
            else:
                ctor_args = [inner]
        else:
            ctor_args = [second]
    return type_name, ctor_args


def normalize_type_expression(name: str) -> str:
    """
    Fold a type name as written in PowerShell source into its lookup form: lower-cased, with the
    whitespace a type literal may carry between namespace parts removed.
    """
    return name.lower().replace(' ', '')


def normalize_dotnet_type_name(name: str) -> str:
    """
    Fold a type name into its lookup form as `normalize_type_expression` does, additionally dropping
    the redundant `System.` prefix so `[System.Convert]` and `[Convert]` reduce to the same key.
    """
    result = normalize_type_expression(name)
    if result.startswith('system.'):
        result = result[7:]
    return result


def is_builtin_variable(
    node: Node | None,
    names: set[str] | frozenset[str] = BUILTIN_VARIABLES,
) -> TypeGuard[Ps1Variable]:
    """
    Return `True` when `node` is an unscoped `refinery.lib.scripts.ps1.model.Ps1Variable` whose
    lowered name is in `names` (defaults to `$Null`, `$True`, `$False`).
    """
    return (
        isinstance(node, Ps1Variable)
        and node.scope == Ps1ScopeModifier.NONE
        and node.name.lower() in names
    )


def unwrap_assignment_target(target: Node | None) -> Node | None:
    """
    Peel type-constraint casts and parentheses from an assignment target, so `[Type]$x` and `($x)`
    both resolve to the variable `$x` the assignment writes.
    """
    while isinstance(target, (Ps1ParenExpression, Ps1CastExpression)):
        target = target.expression if isinstance(target, Ps1ParenExpression) else target.operand
    return target


def assignment_target_variables(target: Node | None) -> list[Ps1Variable]:
    """
    The variables written by an assignment target. A plain variable target yields a single entry, a
    `refinery.lib.scripts.ps1.model.Ps1ArrayLiteral` target (the PowerShell multi-assignment
    `$a, $b = 1, 2`) yields one entry per element that unwraps to a variable, and any other target
    (index, member access, literal) yields an empty list.
    """
    target = unwrap_assignment_target(target)
    if isinstance(target, Ps1Variable):
        return [target]
    if isinstance(target, Ps1ArrayLiteral):
        variables: list[Ps1Variable] = []
        for element in target.elements:
            unwrapped = unwrap_assignment_target(element)
            if isinstance(unwrapped, Ps1Variable):
                variables.append(unwrapped)
        return variables
    return []


def assignment_target_is_all_variables(target: Node | None) -> bool:
    """
    Whether every slot of an assignment target unwraps to a plain variable. `False` when any slot is
    an index or member-access expression (e.g. `$arr[0]`), which means the assignment writes to
    memory other than a named variable and cannot be removed on variable-liveness information alone.
    """
    target = unwrap_assignment_target(target)
    if isinstance(target, Ps1Variable):
        return True
    if isinstance(target, Ps1ArrayLiteral):
        return all(isinstance(unwrap_assignment_target(e), Ps1Variable) for e in target.elements)
    return False


def assignment_of(var: Ps1Variable) -> Ps1AssignmentExpression | None:
    """
    The `refinery.lib.scripts.ps1.model.Ps1AssignmentExpression` that writes `var` when `var`
    occupies its target position — directly, or as an element of a multi-assignment
    `refinery.lib.scripts.ps1.model.Ps1ArrayLiteral` target — else `None`. Enclosing
    type-constraint casts and parentheses are transparent.
    """
    cursor: Node = var
    parent = cursor.parent
    while isinstance(parent, (Ps1CastExpression, Ps1ParenExpression, Ps1ArrayLiteral)):
        cursor = parent
        parent = cursor.parent
    if isinstance(parent, Ps1AssignmentExpression) and parent.target is cursor:
        return parent
    return None
