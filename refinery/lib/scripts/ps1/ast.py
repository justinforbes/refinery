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

import io

from typing import TypeGuard

from refinery.lib.scripts import Block, Node
from refinery.lib.scripts.ps1.data import BUILTIN_VARIABLES, KNOWN_ALIAS
from refinery.lib.scripts.ps1.model import (
    Expression,
    Ps1AccessKind,
    Ps1ArrayLiteral,
    Ps1AssignmentExpression,
    Ps1CastExpression,
    Ps1Code,
    Ps1CommandArgument,
    Ps1CommandArgumentKind,
    Ps1CommandInvocation,
    Ps1ExpandableString,
    Ps1ExpressionStatement,
    Ps1HereString,
    Ps1InvokeMember,
    Ps1MemberAccess,
    Ps1ParamBlock,
    Ps1ParenExpression,
    Ps1ScopeModifier,
    Ps1ScriptBlock,
    Ps1StringLiteral,
    Ps1SubExpression,
    Ps1TypeExpression,
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


def get_param_block(node) -> Ps1ParamBlock | None:
    """
    The `param( ... )` block a `refinery.lib.scripts.ps1.model.Ps1Code` node owns, or `None`.

    Like `get_named_blocks` this is code that `get_body` does not report: a parameter default is an
    expression the engine evaluates on every call that omits the argument, and a validation
    attribute carries arguments of its own. Any caller that reads "no statements" as "nothing
    happens here" has to ask this as well.
    """
    if not isinstance(node, Ps1Code):
        return None
    return node.param_block


def get_command_name(cmd: Ps1CommandInvocation) -> str | None:
    """
    The literal name a command is invoked under, or `None` when the name is computed (`& $cmd`, an
    expandable string) and therefore not statically known.
    """
    if isinstance(cmd.name, Ps1StringLiteral):
        return cmd.name.value
    return None


#: The scope qualifiers a command name may carry. Each selects which scope table the name is written
#: to or read from, and none of them is part of the name. Spelled through the enum so the set cannot
#: drift from the scopes the parser produces.
_COMMAND_SCOPES = frozenset({
    Ps1ScopeModifier.GLOBAL.value,
    Ps1ScopeModifier.LOCAL.value,
    Ps1ScopeModifier.PRIVATE.value,
    Ps1ScopeModifier.SCRIPT.value,
})


def normalize_command_name(name: str) -> str:
    """
    The lowercased command name with every leading scope qualifier stripped: the key under which a
    definition and the calls that reach it agree. `function global:Get-Date` defines what an
    unqualified `Get-Date` then resolves to, so both spellings must key as `get-date`. Qualifiers are
    stripped in a loop because they stack — `global:script:Get-Date` parses as one name.

    Only a caller that keys a *definition* should normalize: a shadow set, a callgraph. A caller
    deciding whether to *trust* a name against an allow-list must not. An unqualified spelling that
    fails a lookup is kept, which is the safe answer, whereas normalizing there would turn a
    scope-qualified spelling into a purity grant.
    """
    name = name.lower()
    while True:
        scope, colon, rest = name.partition(':')
        if not colon or scope not in _COMMAND_SCOPES:
            return name
        name = rest


def resolve_command_name(cmd: Ps1CommandInvocation) -> str | None:
    """
    The lowercased command name a call resolves to, following one level of known alias
    (`ipmo` → `import-module`), or `None` when the name is not a static literal. A module qualifier
    is dropped first and a scope qualifier after it, so that
    `& 'Microsoft.PowerShell.Utility\\Invoke-Expression'` and `& 'global:iex'` each run what the
    bare spelling runs.

    This is the *deny-list* reading of a name, and it is the exact opposite of what
    `normalize_command_name` advises for an allow-list. Resolving toward a bare name can only match
    more entries, so on a table whose hits withhold an action — a world opener, a command that emits
    nothing — every extra match is the conservative answer, and a spelling that dodges the table is
    the dangerous one. A table whose hits *grant* something must not read a name this way.

    **The stripping reaches only a quoted name.** Written bare, `Microsoft.PowerShell.Utility\\iex`
    and `global:iex` never arrive here as one token: the lexer splits at the backslash and at the
    scope colon, so `get_command_name` answers `'Microsoft.PowerShell.Utility'` and `'global'`, and
    every table keyed on the bare spelling is dodged. That is a hole in the lexer rather than here,
    and it is the dangerous direction on every caller — a world opener that reads as closed, a
    silent command that reads as emitting. Until the lexer joins a qualified name, do not read this
    function as evidence that a qualified call has been seen.
    """
    name = get_command_name(cmd)
    if name is None:
        return None
    name = normalize_command_name(name.rpartition('\\')[2])
    return KNOWN_ALIAS.get(name, name).lower()


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


def string_value(node: Node | None) -> str | None:
    if isinstance(node, Ps1StringLiteral):
        return node.value
    if isinstance(node, Ps1HereString):
        return node.value
    if isinstance(node, Ps1ExpandableString):
        out = io.StringIO()
        for p in node.parts:
            if not isinstance(p, Ps1StringLiteral):
                break
            out.write(p.value)
        else:
            return out.getvalue()
    if isinstance(node, Ps1SubExpression) and len(node.body) == 1:
        stmt = node.body[0]
        if isinstance(stmt, Ps1ExpressionStatement) and stmt.expression is not None:
            return string_value(stmt.expression)
    return None


def unwrap_parens(node: Node) -> Node:
    """
    Unwrap nested `refinery.lib.scripts.ps1.model.Ps1ParenExpression` wrappers and single-statement
    `refinery.lib.scripts.ps1.model.Ps1SubExpression` wrappers, stopping at an empty wrapper.
    """
    while True:
        if isinstance(node, Ps1ParenExpression) and node.expression is not None:
            node = node.expression
            continue
        if isinstance(node, Ps1SubExpression) and len(node.body) == 1:
            stmt = node.body[0]
            if isinstance(stmt, Ps1ExpressionStatement) and stmt.expression is not None:
                node = stmt.expression
                continue
        break
    return node


def get_member_name(member: str | Expression) -> str | None:
    """
    Extract a plain member name string from a member that may be a string
    or a string literal expression.
    """
    if isinstance(member, str):
        return member
    if isinstance(member, Ps1StringLiteral):
        return member.value
    return None


def extract_positional_values(
    cmd: Ps1CommandInvocation,
) -> list[Expression]:
    """
    Collect all positional argument values from a command invocation.
    """
    result: list[Expression] = []
    for arg in cmd.arguments:
        if isinstance(arg, Ps1CommandArgument):
            if arg.kind == Ps1CommandArgumentKind.POSITIONAL and arg.value is not None:
                result.append(arg.value)
        elif isinstance(arg, Expression):
            result.append(arg)
    return result


def extract_first_positional_string(
    cmd: Ps1CommandInvocation,
) -> str | None:
    values = extract_positional_values(cmd)
    if values:
        return string_value(values[0])
    return None


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


def is_opaque_dispatch(cmd: Ps1CommandInvocation) -> bool:
    """
    Whether an invocation may resolve to an arbitrary command at runtime: its name is neither a
    string literal (a statically known command) nor an inline scriptblock (`&{ ... }`, whose body is
    visible). `& $f`, `. $f`, and a call through an expandable string or subexpression all dispatch
    to whatever the expression yields, so nothing static bounds what they run. The
    inline-scriptblock exclusion is why this is not `get_command_name(cmd) is None`, which would
    also flag `&{ ... }`.
    """
    return not isinstance(cmd.name, (Ps1StringLiteral, Ps1ScriptBlock))


_SCRIPTBLOCK_TYPE_NAMES = frozenset({
    'scriptblock',
    'management.automation.scriptblock',
})


def is_scriptblock_create(expr: Expression) -> bool:
    """
    Whether `expr` is a `[scriptblock]::Create(...)` call, which compiles an arbitrary string into a
    runnable scriptblock. The argument count is not checked — any such call is recognized — so a
    caller that needs the single argument checks the arity itself.
    """
    return (
        isinstance(expr, Ps1InvokeMember)
        and expr.access is Ps1AccessKind.STATIC
        and isinstance(expr.object, Ps1TypeExpression)
        and normalize_dotnet_type_name(expr.object.name) in _SCRIPTBLOCK_TYPE_NAMES
        and isinstance(expr.member, str)
        and expr.member.lower() == 'create'
    )


_SCRIPTBLOCK_INVOKE_METHODS = frozenset({
    'invoke',
    'invokereturnasis',
    'invokewithcontext',
    'foreach',
    'where',
})


def is_scriptblock_invoke(expr: Expression) -> bool:
    """
    Whether `expr` runs code a receiver carries rather than a fixed .NET method: `$sb.Invoke(...)`,
    `.InvokeReturnAsIs`, `.InvokeWithContext`, or the intrinsic `.ForEach`/`.Where`, which each take
    and run a scriptblock. The receiver is not typed, so any instance call by one of these names
    counts; a false positive on an unrelated `.Where` only keeps a statement, never deletes one.
    """
    return (
        isinstance(expr, Ps1InvokeMember)
        and expr.access is Ps1AccessKind.INSTANCE
        and isinstance(expr.member, str)
        and expr.member.lower() in _SCRIPTBLOCK_INVOKE_METHODS
    )


def is_execution_context_invoke(expr: Expression) -> bool:
    """
    Whether `expr` invokes a member of `$ExecutionContext.InvokeCommand` — `.InvokeScript(...)`,
    `.NewScriptBlock(...)`, `.ExpandString(...)` — each of which runs or compiles code from a
    string. Matched on the `.InvokeCommand` receiver chain rather than the member name, so the whole
    command surface is covered.

    The chain is followed to whatever it is rooted in rather than being pinned to one depth:
    `$ExecutionContext.SessionState.InvokeCommand` reaches the same
    `CommandInvocationIntrinsics` object as `$ExecutionContext.InvokeCommand`, so accepting only the
    shorter spelling would leave the longer one reading as an ordinary member call.
    """
    if not (isinstance(expr, Ps1InvokeMember) and expr.access is Ps1AccessKind.INSTANCE):
        return False
    middle = expr.object
    if not isinstance(middle, Ps1MemberAccess):
        return False
    inner = get_member_name(middle.member)
    if inner is None or inner.lower() != 'invokecommand':
        return False
    receiver = middle.object
    while isinstance(receiver, Ps1MemberAccess):
        receiver = receiver.object
    return isinstance(receiver, Ps1Variable) and receiver.name.lower() == 'executioncontext'


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
