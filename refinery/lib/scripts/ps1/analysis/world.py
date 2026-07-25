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
mutator. The deny-list is enumerated here rather than left silent; its residual (an exotic aliasing
spelling, a `using module` statement, a computed provider path) is a recall gap that keeps a read,
never deletes one.

This is a leaf model in Phase 5c — a one-shot whole-script verdict — cached in
`refinery.lib.scripts.ps1.analysis.cache.Ps1ModelCache` and queried through
`refinery.lib.scripts.ps1.analysis.types.TypeOracle`. `Ps1TypeWorld.world_closed_at` takes the read
node so a flow-sensitive successor can make the answer depend on where the read sits relative to the
leaks that reach it; in this phase the node is not yet consulted.
"""
from __future__ import annotations

from refinery.lib.scripts.ps1.ast import (
    get_command_name,
    get_member_name,
    is_execution_context_invoke,
    is_opaque_dispatch,
    is_scriptblock_create,
    is_scriptblock_invoke,
    normalize_dotnet_type_name,
    string_value,
    unwrap_assignment_target,
)
from refinery.lib.scripts.ps1.data import KNOWN_ALIAS
from refinery.lib.scripts.ps1.model import (
    Ps1AccessKind,
    Ps1AssignmentExpression,
    Ps1CommandArgument,
    Ps1CommandInvocation,
    Ps1InvokeMember,
    Ps1MemberAccess,
    Ps1ScopeModifier,
    Ps1Script,
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
    command table intact. Held in a `refinery.lib.scripts.ps1.analysis.cache.Ps1ModelCache` slot and
    consulted through `refinery.lib.scripts.ps1.analysis.types.TypeOracle.world_closed_at`.
    """

    def __init__(self, closed: bool):
        self._closed = closed

    def world_closed_at(self, node) -> bool:
        """
        Whether the type world is closed at `node`. In Phase 5c this is the whole-script verdict and
        `node` is not consulted — it is a forward-declared seam for the flow-sensitive successor,
        which will make the answer depend on whether a leak reaches this read rather than whether
        the script contains one anywhere.
        """
        return self._closed


def build_closed_world(root: Ps1Script) -> Ps1TypeWorld:
    """
    Walk the whole tree once and report whether any node opens the world. A single opener anywhere
    closes off the verdict, because a type-system mutation is global and retroactive: it changes
    what a read means regardless of where in the script it sits.
    """
    for node in root.walk():
        if _opens_world(node):
            return Ps1TypeWorld(False)
    return Ps1TypeWorld(True)


def _opens_world(node) -> bool:
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
        target = unwrap_assignment_target(node.target)
        return isinstance(target, Ps1Variable) and target.scope in _IDENTITY_SCOPES
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
