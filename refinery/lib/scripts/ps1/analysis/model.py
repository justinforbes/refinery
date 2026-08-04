"""
A semantic model for PowerShell: a tree of scopes with resolved variable bindings and def/use sets,
computed once over an AST and then queried by the deobfuscation transforms instead of each
transform re-deriving scope, binding, and liveness facts on its own. This is the foundation layer of
the ps1 analysis substrate, mirroring `refinery.lib.scripts.js.analysis.model` — later layers
(effect and control-flow models) attach behind the same representation-agnostic surface.

Only three constructs introduce a scope: the script itself and every
`refinery.lib.scripts.ps1.model.Ps1ScriptBlock` (a function or method body, a stored closure, or a
bare `&{ ... }`). PowerShell has no block scoping — a variable assigned in an `if`/loop/`try` body
is visible after it — so those bodies share the scope of their enclosing script or scriptblock.

The two PowerShell scoping rules the model encodes are the point the two hand-rolled liveness passes
used to disagree on, now made authoritative:

- **Write-local.** A bare (unqualified) assignment inside a scriptblock creates a scriptblock-local
  binding; it does not write the enclosing binding of that name.
- **Read fall-through.** A bare read inside a scriptblock references the nearest enclosing binding
  of that name. Because PowerShell creates the local only at the first assignment and a read before
  it falls through at runtime, the model resolves a bare read *conservatively*: it records the read
  on every enclosing scope that binds the name, so a read that might observe an outer value keeps
  that outer binding live. Distinguishing which definition actually reaches a use needs a
  control-flow graph and is left to a later layer.

Where PowerShell scoping is genuinely dynamic — a scope qualifier (`$script:`, `$global:`, …), a
name reachable through `Invoke-Expression`, `&`/`.` dispatch, or a function invoked elsewhere
reading a caller's variables — the model errs toward keeping a binding live rather than risk
treating a live reference as free. A qualified read marks the binding of that name reachable, so it
is never reported dead.
"""
from __future__ import annotations

import enum

from dataclasses import dataclass, field
from typing import Iterator

from refinery.lib.scripts import Node
from refinery.lib.scripts.ps1.ast import assignment_of, binding_key, is_reference_cast
from refinery.lib.scripts.ps1.model import (
    Ps1ArrayLiteral,
    Ps1AssignmentExpression,
    Ps1CastExpression,
    Ps1ForEachLoop,
    Ps1FunctionDefinition,
    Ps1IndexExpression,
    Ps1MemberAccess,
    Ps1ParenExpression,
    Ps1ParameterDeclaration,
    Ps1PropertyMember,
    Ps1ScopeModifier,
    Ps1Script,
    Ps1ScriptBlock,
    Ps1UnaryExpression,
    Ps1Variable,
)


class Ps1OccurrenceRole(enum.Enum):
    """
    What one occurrence of a variable does to the value the name holds. Every occurrence has exactly
    one role, and the transforms ask for it rather than each assembling an answer from a handful of
    positional predicates — which is how `[ref]$n` came to be read as a plain read by every one of
    them at once.

    `NOT_A_REFERENCE` — the occurrence does not reference a variable at all: a class property member
    declaration names a member of the class, a namespace of its own.
    `READ` — observes the value and does not change it.
    `WRITE_REPLACING` — stores without observing what was there: `$x = v`, a `foreach` variable, a
    parameter.
    `WRITE_OBSERVING` — stores *and* observes: `$x += v`, `$x++`, and `[ref]$x`, whose callee may
    store back through the wrapper it is handed.
    `WRITE_THROUGH` — reads the variable to reach a place inside it that is written: the `$x` of
    `$x[0] = 'z'` or `$x.Length = 5`. The name still holds whatever it held, so this observes the
    value like a read, but no value may be installed in its place.
    """
    NOT_A_REFERENCE = enum.auto()
    READ            = enum.auto()  # noqa
    WRITE_REPLACING = enum.auto()
    WRITE_OBSERVING = enum.auto()
    WRITE_THROUGH   = enum.auto()  # noqa


def occurrence_role(var: Ps1Variable) -> Ps1OccurrenceRole:
    """
    The `Ps1OccurrenceRole` of `var`.

    The order the cases are tried in is the order they nest. An occurrence an assignment stores
    through is a target position as much as a plain target is, and is decided first because
    `assignment_of` deliberately answers `None` for it; a reference cast is decided last among the
    writes because everything above it is a syntactic position and a cast is a value form.
    """
    if _is_member_declaration(var):
        return Ps1OccurrenceRole.NOT_A_REFERENCE
    if _stores_through(var):
        return Ps1OccurrenceRole.WRITE_THROUGH
    assignment = assignment_of(var)
    if assignment is not None:
        if assignment.operator == '=':
            return Ps1OccurrenceRole.WRITE_REPLACING
        return Ps1OccurrenceRole.WRITE_OBSERVING
    parent = var.parent
    if isinstance(parent, Ps1UnaryExpression) and parent.operator in ('++', '--'):
        if parent.operand is var:
            return Ps1OccurrenceRole.WRITE_OBSERVING
    if isinstance(parent, Ps1ForEachLoop) and parent.variable is var:
        return Ps1OccurrenceRole.WRITE_REPLACING
    if isinstance(parent, Ps1ParameterDeclaration) and parent.variable is var:
        return Ps1OccurrenceRole.WRITE_REPLACING
    if is_reference_cast(parent) and parent.operand is var:
        return Ps1OccurrenceRole.WRITE_OBSERVING
    return Ps1OccurrenceRole.READ


def is_substitutable_position(var: Ps1Variable) -> bool:
    """
    Whether a value may be installed where `var` stands, replacing the occurrence.

    This is not the complement of writing, and reading it off the role alone is what let two
    corruptions through. A splatted `@p` observes the value like any read, but it spreads an array
    over a command's parameters, and the array written in its place is one argument rather than
    several. A `[ref]$n` observes the value too, and the literal put in its place is a reference to
    nothing that the callee's store is silently lost through.
    """
    return occurrence_role(var) is Ps1OccurrenceRole.READ and not var.splatted


def declares_binding(var: Ps1Variable) -> bool:
    """
    Whether the occurrence brings the binding into existence in the scope it resolves to.

    Every write does except a reference: PowerShell resolves `[ref]$n` by ordinary lookup and
    creates nothing, so filing one as a declaration invents a local binding in whatever body the
    reference is written in and hides the outer one the callee actually stores through.
    """
    if is_reference_cast(var.parent):
        return False
    return occurrence_role(var) in (
        Ps1OccurrenceRole.WRITE_REPLACING,
        Ps1OccurrenceRole.WRITE_OBSERVING,
    )


def is_assignment_write_target(var: Ps1Variable) -> bool:
    """
    Whether `var` occupies the target position of an enclosing
    `refinery.lib.scripts.ps1.model.Ps1AssignmentExpression`, including as an element of a
    multi-assignment `refinery.lib.scripts.ps1.model.Ps1ArrayLiteral` target. Enclosing casts and
    parentheses are transparent.

    A question about syntax rather than about role, which is why it is not derived from
    `occurrence_role`: a `foreach` variable and a parameter replace the value exactly as a plain
    assignment target does and occupy no assignment at all.
    """
    return assignment_of(var) is not None


def replaces_value(var: Ps1Variable) -> bool:
    """
    Whether `var` occupies the target position of a plain `=` assignment, which overwrites the
    variable without observing its previous value. The target of a compound assignment (`+=`, `-=`,
    `.=`, …) is excluded: it reads the variable as well as writing it.
    """
    assignment = assignment_of(var)
    return assignment is not None and assignment.operator == '='


def observes_previous_value(var: Ps1Variable) -> bool:
    """
    Whether `var` occupies a position that reads the variable as part of writing it: the target of a
    compound assignment (`+=`, `.=`, …), the operand of `++`/`--`, or a `[ref]` cast the callee may
    store back through. Such a write is also a use, so a binding that has one is not dead however
    many of its `Binding.reads` a caller has accounted for.
    """
    return occurrence_role(var) is Ps1OccurrenceRole.WRITE_OBSERVING


def is_mutated_in_place(var: Ps1Variable) -> bool:
    """
    Whether an assignment stores *through* `var` rather than into it — the `$x` of `$x[0] = 'z'`, of
    `$x.Length = 5`, of `$x[0][1] = 'z'` and of the multi-assignment `$x[0], $x[1] = 'p', 'q'`.

    Such an occurrence reads the variable in order to reach the part that is written, so
    `is_write_occurrence` calls it a read and no occurrence of the name records the change.
    """
    return occurrence_role(var) is Ps1OccurrenceRole.WRITE_THROUGH


def _stores_through(var: Ps1Variable) -> bool:
    """
    The receiver-chain climb behind `Ps1OccurrenceRole.WRITE_THROUGH`.

    The whole chain counts, not just its innermost step. A target is only a target once the index
    and member accesses, the parentheses, the casts and the multi-assignment slots between it and
    the assignment have been climbed, and stopping at the first of them answers `$x[0] = 'z'` while
    missing `$x[0][1] = 'z'`.
    """
    cursor: Node = var
    parent = cursor.parent
    through = False
    while parent is not None:
        if isinstance(parent, (Ps1IndexExpression, Ps1MemberAccess)):
            if parent.object is not cursor:
                return False
            through = True
        elif isinstance(parent, Ps1CastExpression):
            if parent.operand is not cursor:
                return False
        elif isinstance(parent, (Ps1ParenExpression, Ps1ArrayLiteral)):
            pass
        elif isinstance(parent, Ps1AssignmentExpression):
            return through and parent.target is cursor
        else:
            return False
        cursor = parent
        parent = cursor.parent
    return False


def is_write_occurrence(var: Ps1Variable) -> bool:
    """
    Whether `var` occurs in a position that writes it: the target of an assignment (including a
    multi-assignment slot), the operand of a `++`/`--` update, the loop variable of a `foreach`, a
    parameter declaration, or the operand of a `[ref]` cast. Every other occurrence reads the
    variable.

    An occurrence an assignment stores *through* is not one of these: `$x[0] = 'z'` leaves `$x`
    holding what it held, so the name records no change and the occurrence counts as a read.
    """
    return occurrence_role(var) in (
        Ps1OccurrenceRole.WRITE_REPLACING,
        Ps1OccurrenceRole.WRITE_OBSERVING,
    )


def _is_member_declaration(var: Ps1Variable) -> bool:
    """
    Whether `var` names a class property member (`class C { [int]$x }`) rather than referencing a
    variable. A property declares a member of the class, a namespace distinct from the script's
    variables, so the model binds nothing for it and attributes neither a read nor a write.
    """
    parent = var.parent
    return isinstance(parent, Ps1PropertyMember) and parent.variable is var


class ScopeKind(enum.Enum):
    SCRIPT      = 'script'       # noqa
    FUNCTION    = 'function'     # noqa
    SCRIPTBLOCK = 'scriptblock'  # noqa


#: Scope qualifiers that reach a binding beyond the lexical fall-through a bare reference resolves
#: by, so a read through one keeps the binding it names live rather than resolving it locally. Which
#: scope each names is decided by `Ps1SemanticModel._qualified_read_scopes`. `$env:` is excluded —
#: it names an operating-system environment variable, a namespace distinct from script variables —
#: as is the bare (unqualified) case.
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
    A single variable name bound within one scope. `writes` holds every occurrence that writes it
    (an assignment target, a `++`/`--` operand, a `foreach` variable, a parameter); `reads` holds
    every occurrence that reads it, including a bare read that fell through from a nested block.
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
    def uses(self) -> list[Ps1Variable]:
        """
        Every occurrence that observes the binding's value: its `reads`, and those of its `writes`
        that read what was there in order to write it.

        The two lists are buckets, not roles, and an occurrence that both reads and writes has no
        bucket of its own — `$x += 1` and `[ref]$x` are filed under `writes` and observe the value
        as surely as anything in `reads`. Every consumer deciding whether a value is still wanted
        asks this rather than `reads`, because asking `reads` is exactly how a store whose only
        reader is a compound assignment came to be deletable.
        """
        return [*self.reads, *(w for w in self.writes if observes_previous_value(w))]

    @property
    def is_dead(self) -> bool:
        """
        Whether no use observes the binding's value: no occurrence observes it and no qualifier or
        dynamic scope reaches it. The write occurrences of a dead binding can be removed when they
        carry no other side effect (which the caller decides).
        """
        return not self.uses and not self.dynamic_or_qualified


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
    Yield every descendant of *scope_node* that belongs to its scope, yielding but not descending
    into a nested `refinery.lib.scripts.ps1.model.Ps1ScriptBlock` — each introduces its own scope,
    so its contents are attributed there instead.
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
    `build_semantic_model` and query it through `scope_of` and `binding_of`, through the `bindings`
    of a `Scope`, and — for the flow-sensitive dead-store sweep — through `reads_in_scope` and
    `variables_in_scope`.
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
        The innermost scope that contains *node*, or `None` if the node was not part of the script
        the model was built from. A node in an `if`/loop/`try` body resolves to the enclosing script
        or scriptblock scope, since those bodies introduce no scope of their own.
        """
        return self._node_scope.get(id(node))

    def binding_of(self, var: Ps1Variable) -> Binding | None:
        """
        The binding a variable occurrence resolves to — for a write, the binding in its defining
        scope; for a bare read, the nearest enclosing binding of the name — or `None` when the
        occurrence is free (an automatic or external variable the model never binds) or names a
        namespace outside the script's variables.
        """
        return self._binding_of.get(id(var))

    def reads_in_scope(self, node: Node, scope: Scope) -> set[str]:
        """
        The names of *scope*'s bindings read anywhere within *node*'s subtree — every bare read of a
        name *scope* binds, including one nested in a scriptblock, but not the target of a plain `=`
        assignment, which replaces the value without observing it. A compound-assignment target
        (`$x += 1`) does observe it and counts as a read. This is the read set the dead-store sweep
        flushes pending stores against: unlike the walk it replaces, it does not stop at a nested
        scriptblock, so a store read only through a captured block is correctly seen as live.
        """
        names: set[str] = set()
        for descendant in node.walk():
            if not isinstance(descendant, Ps1Variable):
                continue
            if descendant.scope is not Ps1ScopeModifier.NONE:
                continue
            name = descendant.name.lower()
            if name in scope.bindings and not replaces_value(descendant):
                names.add(name)
        return names

    def variables_in_scope(self, node: Node, scope: Scope) -> set[str]:
        """
        The names of *scope*'s bindings referenced in any way — read or written — within *node*'s
        subtree. The conservative flush set for a control-flow statement whose internal effect on a
        variable the linear sweep does not model: any mention of a bound name defers its pending
        store.
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
            if isinstance(node, Ps1Variable) and declares_binding(node):
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
        key = binding_key(var)
        if key not in scope.bindings:
            scope.bindings[key] = Binding(name=key, scope=scope)

    def _defining_scope(self, var: Ps1Variable, current: Scope) -> Scope | None:
        """
        The scope a write to *var* binds. A bare, `$local:`, or `$private:` assignment binds in the
        current scope (write-local); a `$script:`, `$global:`, or `$using:` assignment, and an
        `$env:` assignment (a process-global environment variable, bound under an `env:`-prefixed
        key), bind at the script scope. The provider namespaces (`variable:`, `function:`,
        `alias:`, `drive:`) name a namespace distinct from script variables and bind nothing here.
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
            if not isinstance(node, Ps1Variable) or _is_member_declaration(node):
                continue
            scope = self._node_scope.get(id(node))
            if scope is None:
                continue
            if is_reference_cast(node.parent):
                self._attribute_reference(node, scope)
            elif is_write_occurrence(node):
                self._attribute_write(node, scope)
            else:
                self._attribute_read(node, scope)

    def _attribute_write(self, var: Ps1Variable, scope: Scope):
        binding = self._lookup_write_binding(var, scope)
        if binding is not None:
            binding.writes.append(var)
            self._binding_of[id(var)] = binding

    def _attribute_reference(self, var: Ps1Variable, scope: Scope):
        """
        Attribute a `[ref]$x` occurrence: resolved the way a read is, recorded the way a write is.

        The two halves are not the same question. PowerShell resolves the name by ordinary lookup,
        so a reference written inside a body reaches the enclosing binding and declares nothing —
        resolving it the way a write is resolved would look for a local binding that was never
        created and attribute the occurrence to nothing at all, losing the very use this exists to
        keep. What it then does to that binding is store into it, so it is recorded among the
        writes, where it both keeps the binding alive through `Binding.uses` and stops an earlier
        value reaching a later read.
        """
        if var.scope is not Ps1ScopeModifier.NONE:
            self._attribute_read(var, scope)
            return
        name = binding_key(var)
        primary: Binding | None = None
        cursor: Scope | None = scope
        while cursor is not None:
            binding = cursor.bindings.get(name)
            if binding is not None:
                binding.writes.append(var)
                if primary is None:
                    primary = binding
            cursor = cursor.parent
        if primary is not None:
            self._binding_of[id(var)] = primary

    def _lookup_write_binding(self, var: Ps1Variable, scope: Scope) -> Binding | None:
        defining = self._defining_scope(var, scope)
        if defining is None:
            return None
        return defining.bindings.get(binding_key(var))

    def _attribute_read(self, var: Ps1Variable, scope: Scope):
        if var.scope is Ps1ScopeModifier.NONE:
            self._attribute_bare_read(var, scope)
        elif var.scope is Ps1ScopeModifier.ENV:
            binding = self.root_scope.bindings.get(binding_key(var))
            if binding is not None:
                binding.reads.append(var)
                self._binding_of[id(var)] = binding
        elif var.scope in _QUALIFIED_SCOPES:
            self._attribute_qualified_read(var, scope)

    def _attribute_qualified_read(self, var: Ps1Variable, scope: Scope):
        """
        Mark every binding a scope-qualified read can reach as `Binding.dynamic_or_qualified`, so it
        is never reported dead even though no occurrence in `Binding.reads` names it.
        """
        name = var.name.lower()
        primary: Binding | None = None
        for target in self._qualified_read_scopes(var, scope):
            binding = target.bindings.get(name)
            if binding is None:
                continue
            binding.dynamic_or_qualified = True
            if primary is None:
                primary = binding
        if primary is not None:
            self._binding_of[id(var)] = primary

    def _qualified_read_scopes(self, var: Ps1Variable, scope: Scope) -> Iterator[Scope]:
        """
        The scopes a scope-qualified read of *var* can reach. `$variable:` addresses the Variable
        provider drive, which resolves like a bare reference, so it reaches every enclosing scope;
        every other qualifier names the one scope `_defining_scope` binds a write through it in —
        the scope of the reference itself for `$local:` and `$private:`, the script scope for
        `$script:`, `$global:`, and `$using:`.
        """
        if var.scope is Ps1ScopeModifier.VARIABLE:
            cursor: Scope | None = scope
            while cursor is not None:
                yield cursor
                cursor = cursor.parent
            return
        defining = self._defining_scope(var, scope)
        if defining is not None:
            yield defining

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
