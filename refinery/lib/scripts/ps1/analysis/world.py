"""
The closed-world model of the PowerShell analysis substrate: whether the script leaves the .NET type
system and the command table intact, so that a present-member purity grant in
`refinery.lib.scripts.ps1.analysis.effects` can be trusted. A member read the metadata proves inert
(`String.Length` is a plain property) still runs code when the script has re-pointed that member
through the Extended Type System (`Update-TypeData -Force`), and a resolved type name still
constructs a different type when the script has remapped its accelerator — both confirmed possible
on Windows PowerShell. The gate therefore grants a present-member read only where the world is
*closed*: no code the script runs can have performed such a mutation.

The predicate opens the world on two axes with opposite defaults. **Dispatch** is an allow-list: a
command is inert only when its name is statically known, so any `& $x` / `. $x` / computed-name
call opens the world, closing the open-ended escape of runtime-constructed code without a list to
forget.
**Mutation** is a curated, documented deny-list — a pure allow-list would be vacuous, since the
collected metadata omits hundreds of host cmdlets and every one would then read as a possible
mutator. The deny-list is enumerated here rather than left silent, and its residual — an exotic
aliasing spelling, a `using module` statement, a computed provider path — is a *soundness* gap, not a
recall gap: a mutator the list misses leaves the world reading closed, which fires the grants and
deletes the reads that mutator makes effectful. Every name added to it buys correctness, not recall.

This is a leaf model in Phase 5c — a one-shot whole-script verdict — cached in
`refinery.lib.scripts.ps1.analysis.cache.Ps1ModelCache` and queried through
`refinery.lib.scripts.ps1.analysis.types.TypeOracle`. `Ps1TypeWorld.world_closed_at` takes the read
node so a flow-sensitive successor can make the answer depend on where the read sits relative to the
leaks that reach it; in this phase the node is not yet consulted.
"""
from __future__ import annotations

import enum

from typing import NamedTuple

from refinery.lib.scripts.ps1.ast import (
    assignment_target_variables,
    get_command_name,
    get_member_name,
    is_execution_context_invoke,
    is_opaque_dispatch,
    is_scriptblock_create,
    is_scriptblock_invoke,
    normalize_command_name,
    normalize_dotnet_type_name,
    string_value,
    unwrap_assignment_target,
    unwrap_parens,
)
from refinery.lib.scripts.ps1.data import KNOWN_ALIAS
from refinery.lib.scripts.ps1.model import (
    Ps1AccessKind,
    Ps1ArrayLiteral,
    Ps1AssignmentExpression,
    Ps1CommandArgument,
    Ps1CommandInvocation,
    Ps1FunctionDefinition,
    Ps1InvokeMember,
    Ps1MemberAccess,
    Ps1ScopeModifier,
    Ps1Script,
    Ps1ScriptBlock,
    Ps1StringLiteral,
    Ps1TypeExpression,
    Ps1Variable,
)

#: Commands that execute arbitrary code supplied as data. `Invoke-Expression` is the canonical one;
#: the opaque dispatch and scriptblock-execution forms are recognized syntactically instead.
_LEAK_CMDLETS = frozenset({
    'invoke-expression',
})

#: Commands that mutate the .NET type system, so reflection can no longer be trusted to describe a
#: type's members. Curated and documented rather than derived — the module docstring says why a
#: mutation allow-list would be vacuous. Names are compared after alias resolution.
_MUTATION_CMDLETS = frozenset({
    'add-member',
    'import-module',
    'update-typedata',
})

#: Commands that redefine command identity, after which a later bareword can no longer be trusted to
#: name what the metadata says — including a mutator hidden behind the new name. A static
#: single-definition alias is inlined away before this runs, so a *surviving* one is an alias the
#: inliner could not resolve.
_ALIAS_CMDLETS = frozenset({
    'new-alias',
    'remove-alias',
    'set-alias',
})

#: The variable namespaces that name a command rather than a value: assigning into either redefines
#: command identity the way `_ALIAS_CMDLETS` do.
_IDENTITY_SCOPES = frozenset({
    Ps1ScopeModifier.ALIAS,
    Ps1ScopeModifier.FUNCTION,
})

#: The provider path prefixes that address command identity, written as a string argument to an item
#: cmdlet (`Set-Item alias:x ...`). Matched by prefix rather than by enumerating every aliasing
#: cmdlet, which is the family the mutation deny-list cannot close by name.
_IDENTITY_PROVIDERS = ('alias:', 'function:')


class Ps1TypeWorld:
    """
    The verdict of `build_closed_world`: whether the running script leaves the type system and
    command table intact. It carries both command-table facts the purity gate needs — the
    whole-world verdict (`world_closed_at`) and the set of command names the script redefines
    (`command_shadowed`) — so the two cannot drift apart. Held in a
    `refinery.lib.scripts.ps1.analysis.cache.Ps1ModelCache` slot and consulted through
    `refinery.lib.scripts.ps1.analysis.types.TypeOracle`.
    """

    def __init__(self, closed: bool, shadowed: frozenset[str]):
        self._closed = closed
        self._shadowed = shadowed

    def world_closed_at(self, node) -> bool:
        """
        Whether the type world is closed at `node`. In Phase 5c this is the whole-script verdict and
        `node` is not consulted — it is a forward-declared seam for the flow-sensitive successor,
        which will make the answer depend on whether a leak reaches this read rather than whether
        the script contains one anywhere.
        """
        return self._closed

    def command_shadowed(self, name: str) -> bool:
        """
        Whether `name` is a command the script redefines with a script-local `function`/`filter`
        or a `function:`/`alias:`-scope assignment, so the collected metadata no longer describes
        what the name runs. The analysis must not trust such a name for typing or purity. The set is
        whole-script and conservative — an inner-scope redefinition distrusts the name everywhere,
        which only keeps more — mirroring `world_closed_at`'s whole-script granularity.
        """
        return name.lower() in self._shadowed


def build_closed_world(root: Ps1Script) -> Ps1TypeWorld:
    """
    Walk the whole tree once, computing both command-table facts: whether any node opens the world
    (a single opener anywhere is global and retroactive, so it closes off the verdict) and the set
    of command names the script redefines. The walk cannot short-circuit on the first opener
    because the shadow set needs every redefinition, wherever it sits.
    """
    closed = True
    shadowed: set[str] = set()
    for node in root.walk():
        redefined = _identity_redefinitions(node)
        shadowed.update(record.name for record in redefined)
        if _opens_world(node, redefined):
            closed = False
    return Ps1TypeWorld(closed, frozenset(shadowed))


class _IdentityBody(enum.Enum):
    """
    What a command redefinition binds the name to, which is what decides whether the redefinition
    also opens the type world. An enum rather than a boolean because the two ways of *not* being a
    visible block are unrelated — one hides a body inside a value, the other names another command
    entirely — and collapsing them would make the world rule read as if it had one reason.
    """
    #: A scriptblock literal standing in the tree, so the whole-tree walk reads its statements and
    #: catches a mutation inside it by presence. The only kind that leaves the world closed.
    VISIBLE_BLOCK = enum.auto()
    #: A value the walk cannot see through — a variable, a call's result, or a compound assignment
    #: folding in whatever the name held before.
    OPAQUE_VALUE = enum.auto()
    #: Another command, named rather than defined. Its body is not in this script at all.
    EXTERNAL_COMMAND = enum.auto()


class _IdentityRedefinition(NamedTuple):
    name: str
    body: _IdentityBody


def _identity_redefinitions(node) -> tuple[_IdentityRedefinition, ...]:
    """
    The command names `node` redefines, each normalized to the key a call resolves under and paired
    with what it binds, or an empty tuple. A `function`/`filter` definition names one command
    directly; an assignment into the `function:`/`alias:` variable namespace names one per slot it
    writes, so the multi-assignment `${function:Get-Date}, $y = { ... }, 2` records `get-date` where
    matching one target shape against one variable would miss it.

    Normalizing is what makes the name usable: `function global:Get-Date` defines exactly what a
    later unqualified `Get-Date` runs, and a shadow set holding the qualified spelling answers `False`
    to every consumer that asks about the unqualified one.

    Only a plain `=` onto a single target reports `VISIBLE_BLOCK`. A multi-assignment could be
    paired up with the values on its right, but the target list drops non-variable slots, so the
    position a variable came from is already lost — and the shape is rare enough that reading
    every slot of one as opaque costs nothing.
    """
    if isinstance(node, Ps1FunctionDefinition):
        return (_IdentityRedefinition(
            normalize_command_name(node.name), _IdentityBody.VISIBLE_BLOCK),)
    if not isinstance(node, Ps1AssignmentExpression):
        return ()
    targets = assignment_target_variables(node.target)
    single = len(targets) == 1 and not isinstance(
        unwrap_assignment_target(node.target), Ps1ArrayLiteral)
    return tuple(
        _IdentityRedefinition(
            normalize_command_name(variable.name),
            _assigned_identity_body(node, variable, single),
        )
        for variable in targets
        if variable.scope in _IDENTITY_SCOPES
    )


def _assigned_identity_body(
    node: Ps1AssignmentExpression, variable: Ps1Variable, single: bool,
) -> _IdentityBody:
    """
    What an identity-scope assignment binds its name to. The `alias:` namespace always names another
    command; the `function:` namespace binds a scriptblock, which is readable only when written out
    as a literal and this assignment plainly rebinds one target.
    """
    if variable.scope is Ps1ScopeModifier.ALIAS:
        return _IdentityBody.EXTERNAL_COMMAND
    if not single or node.operator != '=' or node.value is None:
        return _IdentityBody.OPAQUE_VALUE
    if isinstance(unwrap_parens(node.value), Ps1ScriptBlock):
        return _IdentityBody.VISIBLE_BLOCK
    return _IdentityBody.OPAQUE_VALUE


def _opens_world(node, redefined: tuple[_IdentityRedefinition, ...]) -> bool:
    """
    Whether `node` leaves the type system or the command table in a state the collected metadata no
    longer describes. `redefined` is the identity classification of the same node, so an assignment
    into the identity namespaces is recognized once, not by two functions that can drift apart.

    A redefinition binding a visible scriptblock does *not* open the world. Its body stands in the
    tree, so a mutation inside it is caught by presence like any other statement, and the same
    construct spelled `function X { }` has always left the world closed. Opening on it would kill
    every member grant in the script over a name the shadow set already distrusts.
    """
    if isinstance(node, Ps1CommandInvocation):
        return _command_opens_world(node)
    if isinstance(node, Ps1InvokeMember):
        return (
            is_scriptblock_create(node)
            or is_scriptblock_invoke(node)
            or is_execution_context_invoke(node)
            or _is_type_accelerator_mutation(node)
            or _is_psobject_member_mutation(node)
        )
    if isinstance(node, Ps1AssignmentExpression):
        return any(record.body is not _IdentityBody.VISIBLE_BLOCK for record in redefined)
    return False


def _command_opens_world(cmd: Ps1CommandInvocation) -> bool:
    if is_opaque_dispatch(cmd):
        return True
    if cmd.invocation_operator == '.' and isinstance(cmd.name, Ps1StringLiteral):
        # Dot-sourcing a file runs its definitions into the current scope, adding types and
        # redefining commands. A dot-sourced inline block (`.{ ... }`) runs only its visible body,
        # which the walk covers, and `. $x` is already opaque dispatch.
        return True
    name = _resolved_command_name(cmd)
    if name is None:
        return False
    if name in _LEAK_CMDLETS or name in _MUTATION_CMDLETS or name in _ALIAS_CMDLETS:
        return True
    return _touches_identity_provider(cmd)


def _resolved_command_name(cmd: Ps1CommandInvocation) -> str | None:
    """
    The lowercased command name a call resolves to, following one level of known alias
    (`ipmo` → `import-module`), or `None` when the name is not a static literal.
    """
    name = get_command_name(cmd)
    if name is None:
        return None
    return KNOWN_ALIAS.get(name.lower(), name).lower()


def _touches_identity_provider(cmd: Ps1CommandInvocation) -> bool:
    """
    Whether any argument is a literal path into the `alias:` or `function:` provider, the vector
    that escapes a name-keyed deny-list because `Set-Item alias:x Update-TypeData` mutates identity
    without `Set-Item` being an aliasing cmdlet. Recognized by the path prefix, not by resolving the
    aliased target, so an obfuscated or dynamic target cannot slip through.
    """
    for arg in cmd.arguments:
        value = arg.value if isinstance(arg, Ps1CommandArgument) else arg
        text = string_value(value)
        if text is not None and text.lower().startswith(_IDENTITY_PROVIDERS):
            return True
    return False


def _is_type_accelerator_mutation(node: Ps1InvokeMember) -> bool:
    """
    Whether `node` adds or removes a type accelerator through
    `[…PSObject+TypeAccelerators]::Add/Remove`, which remaps what a type name resolves to.
    """
    obj = node.object
    return (
        node.access is Ps1AccessKind.STATIC
        and isinstance(obj, Ps1TypeExpression)
        and 'typeaccelerators' in normalize_dotnet_type_name(obj.name)
        and isinstance(node.member, str)
        and node.member.lower() in ('add', 'remove')
    )


def _is_psobject_member_mutation(node: Ps1InvokeMember) -> bool:
    """
    Whether `node` adds or removes a member through the reflective
    `$obj.PSObject.Members.Add/Remove` chain, the Extended Type System mutation that is not a cmdlet
    call and so escapes the name-keyed deny-list.
    """
    if node.access is not Ps1AccessKind.INSTANCE:
        return False
    if not (isinstance(node.member, str) and node.member.lower() in ('add', 'remove')):
        return False
    members = node.object
    if not isinstance(members, Ps1MemberAccess):
        return False
    members_name = get_member_name(members.member)
    if members_name is None or members_name.lower() != 'members':
        return False
    psobject = members.object
    if not isinstance(psobject, Ps1MemberAccess):
        return False
    psobject_name = get_member_name(psobject.member)
    return psobject_name is not None and psobject_name.lower() == 'psobject'
