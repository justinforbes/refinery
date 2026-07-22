"""
A semantic model for PowerShell: a tree of scopes with resolved variable bindings and def/use sets,
computed once over an AST and then queried by the deobfuscation transforms instead of each transform
re-deriving scope, binding, and liveness facts on its own. This is the foundation layer of the ps1
analysis substrate, mirroring `refinery.lib.scripts.js.analysis.model` — later layers (effect and
control-flow models) attach behind the same representation-agnostic surface.

Only three constructs introduce a scope: the script itself and every
`refinery.lib.scripts.ps1.model.Ps1ScriptBlock` (a function or method body, a stored closure, or a
bare `&{ ... }`). PowerShell has no block scoping — a variable assigned in an `if`/loop/`try` body is
visible after it — so those bodies share the scope of their enclosing script or scriptblock.

The two PowerShell scoping rules the model encodes are the point the two hand-rolled liveness passes
used to disagree on, now made authoritative:

- **Write-local.** A bare (unqualified) assignment inside a scriptblock creates a scriptblock-local
  binding; it does not write the enclosing binding of that name.
- **Read fall-through.** A bare read inside a scriptblock references the nearest enclosing binding of
  that name. Because PowerShell creates the local only at the first assignment and a read before it
  falls through at runtime, the model resolves a bare read *conservatively*: it records the read on
  every enclosing scope that binds the name, so a read that might observe an outer value keeps that
  outer binding live. Distinguishing which definition actually reaches a use needs a control-flow
  graph and is left to a later layer.

Where PowerShell scoping is genuinely dynamic — a scope qualifier (`$script:`, `$global:`, …), a
name reachable through `Invoke-Expression`, `&`/`.` dispatch, or a function invoked elsewhere reading
a caller's variables — the model errs toward keeping a binding live rather than risk treating a live
reference as free. A qualified read marks the script-scope binding of that name reachable, so it is
never reported dead.
"""
from __future__ import annotations

import enum

from dataclasses import dataclass, field
from typing import Iterator

from refinery.lib.scripts import Node
from refinery.lib.scripts.ps1.model import (
    Ps1ArrayLiteral,
    Ps1AssignmentExpression,
    Ps1CastExpression,
    Ps1ForEachLoop,
    Ps1FunctionDefinition,
    Ps1ParameterDeclaration,
    Ps1ParenExpression,
    Ps1ScopeModifier,
    Ps1Script,
    Ps1ScriptBlock,
    Ps1UnaryExpression,
    Ps1Variable,
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
    an index or member-access expression (e.g. `$arr[0]`), which means the assignment writes to memory
    other than a named variable and cannot be removed on variable-liveness information alone.
    """
    target = unwrap_assignment_target(target)
    if isinstance(target, Ps1Variable):
        return True
    if isinstance(target, Ps1ArrayLiteral):
        return all(isinstance(unwrap_assignment_target(e), Ps1Variable) for e in target.elements)
    return False


def assignment_of(var: Ps1Variable) -> Ps1AssignmentExpression | None:
    """
    The `refinery.lib.scripts.ps1.model.Ps1AssignmentExpression` that writes `var` when `var` occupies
    its target position — directly, or as an element of a multi-assignment
    `refinery.lib.scripts.ps1.model.Ps1ArrayLiteral` target — else `None`. Enclosing type-constraint
    casts and parentheses are transparent.
    """
    cursor: Node = var
    parent = cursor.parent
    while isinstance(parent, (Ps1CastExpression, Ps1ParenExpression, Ps1ArrayLiteral)):
        cursor = parent
        parent = cursor.parent
    if isinstance(parent, Ps1AssignmentExpression) and parent.target is cursor:
        return parent
    return None


def is_assignment_write_target(var: Ps1Variable) -> bool:
    """
    Whether `var` occupies the target position of an enclosing
    `refinery.lib.scripts.ps1.model.Ps1AssignmentExpression`, including as an element of a
    multi-assignment `refinery.lib.scripts.ps1.model.Ps1ArrayLiteral` target. Enclosing casts and
    parentheses are transparent.
    """
    return assignment_of(var) is not None


def is_write_occurrence(var: Ps1Variable) -> bool:
    """
    Whether `var` occurs in a position that writes it: the target of an assignment (including a
    multi-assignment slot), the operand of a `++`/`--` update, the loop variable of a `foreach`, or a
    parameter declaration. Every other occurrence reads the variable.
    """
    if is_assignment_write_target(var):
        return True
    parent = var.parent
    if isinstance(parent, Ps1UnaryExpression) and parent.operator in ('++', '--'):
        return parent.operand is var
    if isinstance(parent, Ps1ForEachLoop):
        return parent.variable is var
    if isinstance(parent, Ps1ParameterDeclaration):
        return parent.variable is var
    return False


def _binding_key(var: Ps1Variable) -> str:
    """
    The key a variable binds under within a scope's binding table: its lowercased name, prefixed with
    `env:` for an environment variable so the process-global `$env:X` namespace stays distinct from a
    script variable `$X` of the same name.
    """
    if var.scope is Ps1ScopeModifier.ENV:
        return F'env:{var.name.lower()}'
    return var.name.lower()


class ScopeKind(enum.Enum):
    SCRIPT      = 'script'       # noqa
    FUNCTION    = 'function'     # noqa
    SCRIPTBLOCK = 'scriptblock'  # noqa


#: Scope qualifiers that reach a binding beyond the lexical scope of the reference, so a read through
#: one keeps the script-scope binding of that name live rather than resolving it locally. `$env:` is
#: excluded — it names an operating-system environment variable, a namespace distinct from script
#: variables — as is the bare (unqualified) case, which resolves by fall-through.
_QUALIFIED_SCOPES = frozenset({
    Ps1ScopeModifier.GLOBAL,
    Ps1ScopeModifier.LOCAL,
    Ps1ScopeModifier.SCRIPT,
    Ps1ScopeModifier.PRIVATE,
    Ps1ScopeModifier.USING,
    Ps1ScopeModifier.VARIABLE,
})


@dataclass(eq=False)
class Binding:
    """
    A single variable name bound within one scope. `writes` holds every occurrence that writes it (an
    assignment target, a `++`/`--` operand, a `foreach` variable, a parameter); `reads` holds every
    occurrence that reads it, including a bare read that fell through from a nested scriptblock.
    `dynamic_or_qualified` marks a binding a scope qualifier or dynamic scope could reach with no
    occurrence in `reads` — conservatively kept live.
    """
    name: str
    scope: Scope
    reads: list[Ps1Variable] = field(default_factory=list)
    writes: list[Ps1Variable] = field(default_factory=list)
    dynamic_or_qualified: bool = False

    @property
    def is_read(self) -> bool:
        """
        Whether any occurrence reads the binding's value.
        """
        return bool(self.reads)

    @property
    def is_dead(self) -> bool:
        """
        Whether no use observes the binding's value: it is read through no occurrence and reached by
        no qualifier or dynamic scope. The write occurrences of a dead binding can be removed when they
        carry no other side effect (which the caller decides).
        """
        return not self.reads and not self.dynamic_or_qualified


@dataclass(eq=False)
class Scope:
    """
    A lexical scope introduced by the script or a `refinery.lib.scripts.ps1.model.Ps1ScriptBlock`.
    `node` is the introducing AST node, `bindings` maps a lowercased variable name to its `Binding`.
    """
    kind: ScopeKind
    node: Node
    parent: Scope | None = None
    children: list[Scope] = field(default_factory=list)
    bindings: dict[str, Binding] = field(default_factory=dict)


def _scope_local_nodes(scope_node: Node) -> Iterator[Node]:
    """
    Yield every descendant of *scope_node* that belongs to its scope, yielding but not descending into
    a nested `refinery.lib.scripts.ps1.model.Ps1ScriptBlock` — each introduces its own scope, so its
    contents are attributed there instead.
    """
    stack: list[Node] = list(scope_node.children())
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, Ps1ScriptBlock):
            continue
        stack.extend(node.children())


class Ps1SemanticModel:
    """
    The resolved scope/binding/def-use model for one PowerShell script. Build it with
    `build_semantic_model` and query it through `scope_of`, `binding_of`, `bindings_in`, `is_dead`,
    and — for the flow-sensitive dead-store sweep — `reads_in_scope` and `variables_in_scope`.
    """

    def __init__(self, root: Ps1Script):
        self.root = root
        self._node_scope: dict[int, Scope] = {}
        self._binding_of: dict[int, Binding] = {}
        self.root_scope = Scope(kind=ScopeKind.SCRIPT, node=root)
        self._node_scope[id(root)] = self.root_scope
        self._populate(self.root_scope)
        self._build_def_use()

    @property
    def script_scope(self) -> Scope:
        """
        The scope the script itself introduces — the outermost scope, whose bindings are the
        script-level variables.
        """
        return self.root_scope

    def scope_of(self, node: Node) -> Scope | None:
        """
        The innermost scope that contains *node*, or `None` if the node was not part of the script the
        model was built from. A node in an `if`/loop/`try` body resolves to the enclosing script or
        scriptblock scope, since those bodies introduce no scope of their own.
        """
        return self._node_scope.get(id(node))

    def binding_of(self, var: Ps1Variable) -> Binding | None:
        """
        The binding a variable occurrence resolves to — for a write, the binding in its defining scope;
        for a bare read, the nearest enclosing binding of the name — or `None` when the occurrence is
        free (an automatic or external variable the model never binds) or names a non-script namespace.
        """
        return self._binding_of.get(id(var))

    @staticmethod
    def bindings_in(scope: Scope) -> Iterator[Binding]:
        """
        Every binding declared directly in *scope*.
        """
        return iter(scope.bindings.values())

    @staticmethod
    def is_dead(binding: Binding) -> bool:
        """
        Whether *binding* is never read and reachable by no qualifier or dynamic scope — see
        `Binding.is_dead`.
        """
        return binding.is_dead

    def reads_in_scope(self, node: Node, scope: Scope) -> set[str]:
        """
        The names of *scope*'s bindings read anywhere within *node*'s subtree — every bare read of a
        name *scope* binds, including one nested in a scriptblock, but not an assignment target. This
        is the read set the dead-store sweep flushes pending stores against: unlike the walk it
        replaces, it does not stop at a nested scriptblock, so a store read only through a captured
        block is correctly seen as live.
        """
        names: set[str] = set()
        for descendant in node.walk():
            if not isinstance(descendant, Ps1Variable):
                continue
            if descendant.scope is not Ps1ScopeModifier.NONE:
                continue
            name = descendant.name.lower()
            if name in scope.bindings and not is_assignment_write_target(descendant):
                names.add(name)
        return names

    def variables_in_scope(self, node: Node, scope: Scope) -> set[str]:
        """
        The names of *scope*'s bindings referenced in any way — read or written — within *node*'s
        subtree. The conservative flush set for a control-flow statement whose internal effect on a
        variable the linear sweep does not model: any mention of a bound name defers its pending store.
        """
        names: set[str] = set()
        for descendant in node.walk():
            if isinstance(descendant, Ps1Variable) and descendant.scope is Ps1ScopeModifier.NONE:
                name = descendant.name.lower()
                if name in scope.bindings:
                    names.add(name)
        return names

    def _populate(self, scope: Scope):
        for node in _scope_local_nodes(scope.node):
            if isinstance(node, Ps1ScriptBlock):
                child = Scope(kind=self._scriptblock_kind(node), node=node, parent=scope)
                scope.children.append(child)
                self._node_scope[id(node)] = child
                self._populate(child)
                continue
            self._node_scope[id(node)] = scope
            if isinstance(node, Ps1Variable) and is_write_occurrence(node):
                self._declare(node, scope)

    @staticmethod
    def _scriptblock_kind(node: Ps1ScriptBlock) -> ScopeKind:
        if isinstance(node.parent, Ps1FunctionDefinition) and node.parent.body is node:
            return ScopeKind.FUNCTION
        return ScopeKind.SCRIPTBLOCK

    def _declare(self, var: Ps1Variable, current: Scope):
        scope = self._defining_scope(var, current)
        if scope is None:
            return
        key = _binding_key(var)
        if key not in scope.bindings:
            scope.bindings[key] = Binding(name=key, scope=scope)

    def _defining_scope(self, var: Ps1Variable, current: Scope) -> Scope | None:
        """
        The scope a write to *var* binds. A bare, `$local:`, or `$private:` assignment binds in the
        current scope (write-local); a `$script:`, `$global:`, or `$using:` assignment, and an `$env:`
        assignment (a process-global environment variable, bound under an `env:`-prefixed key), bind at
        the script scope. The provider namespaces (`variable:`, `function:`, `alias:`, `drive:`) name a
        namespace distinct from script variables and bind nothing here.
        """
        modifier = var.scope
        if modifier in (Ps1ScopeModifier.NONE, Ps1ScopeModifier.LOCAL, Ps1ScopeModifier.PRIVATE):
            return current
        if modifier in (
            Ps1ScopeModifier.SCRIPT,
            Ps1ScopeModifier.GLOBAL,
            Ps1ScopeModifier.USING,
            Ps1ScopeModifier.ENV,
        ):
            return self.root_scope
        return None

    def _build_def_use(self):
        for node in self.root.walk():
            if not isinstance(node, Ps1Variable):
                continue
            scope = self._node_scope.get(id(node))
            if scope is None:
                continue
            if is_write_occurrence(node):
                self._attribute_write(node, scope)
            else:
                self._attribute_read(node, scope)

    def _attribute_write(self, var: Ps1Variable, scope: Scope):
        binding = self._lookup_write_binding(var, scope)
        if binding is not None:
            binding.writes.append(var)
            self._binding_of[id(var)] = binding

    def _lookup_write_binding(self, var: Ps1Variable, scope: Scope) -> Binding | None:
        defining = self._defining_scope(var, scope)
        if defining is None:
            return None
        return defining.bindings.get(_binding_key(var))

    def _attribute_read(self, var: Ps1Variable, scope: Scope):
        if var.scope is Ps1ScopeModifier.NONE:
            self._attribute_bare_read(var, scope)
        elif var.scope is Ps1ScopeModifier.ENV:
            binding = self.root_scope.bindings.get(_binding_key(var))
            if binding is not None:
                binding.reads.append(var)
                self._binding_of[id(var)] = binding
        elif var.scope in _QUALIFIED_SCOPES:
            binding = self.root_scope.bindings.get(var.name.lower())
            if binding is not None:
                binding.dynamic_or_qualified = True
                self._binding_of[id(var)] = binding

    def _attribute_bare_read(self, var: Ps1Variable, scope: Scope):
        name = var.name.lower()
        primary: Binding | None = None
        cursor: Scope | None = scope
        while cursor is not None:
            binding = cursor.bindings.get(name)
            if binding is not None:
                binding.reads.append(var)
                if primary is None:
                    primary = binding
            cursor = cursor.parent
        if primary is not None:
            self._binding_of[id(var)] = primary


def build_semantic_model(root: Ps1Script) -> Ps1SemanticModel:
    """
    Build the `Ps1SemanticModel` for a parsed PowerShell script.
    """
    return Ps1SemanticModel(root)
