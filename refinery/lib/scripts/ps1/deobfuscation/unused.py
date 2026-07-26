"""
Remove unused variable assignments and junk expression statements.
"""
from __future__ import annotations

from refinery.lib.scripts import Node, Transformer, _remove_from_parent, _replace_in_parent
from refinery.lib.scripts.ps1.analysis.cache import model_cache
from refinery.lib.scripts.ps1.analysis.effects import (
    BodyRole,
    StatementEffect,
    body_is_inert,
    body_role,
    is_side_effect_free,
    output_is_covered,
    output_observed,
    pruning_erases_body,
    statement_effect,
)
from refinery.lib.scripts.ps1.analysis.model import Binding, Ps1SemanticModel, Scope
from refinery.lib.scripts.ps1.analysis.types import TypeOracle
from refinery.lib.scripts.ps1.ast import (
    assignment_of,
    assignment_target_is_all_variables,
    assignment_target_variables,
    get_body,
    get_command_name,
    is_opaque_dispatch,
    normalize_command_name,
)
from refinery.lib.scripts.ps1.deobfuscation.constants import (
    _PS1_SKIP_VARIABLES,
    _find_removable_statement,
    _walk_outer_scope,
)
from refinery.lib.scripts.ps1.model import (
    Expression,
    Ps1AssignmentExpression,
    Ps1CommandInvocation,
    Ps1ExpressionStatement,
    Ps1ForLoop,
    Ps1FunctionDefinition,
    Ps1Pipeline,
    Ps1PipelineElement,
    Ps1ScopeModifier,
    Ps1Script,
    Ps1UnaryExpression,
    Ps1Variable,
)

#: The variable namespaces that name a command rather than a value. Kept in step with
#: `refinery.lib.scripts.ps1.analysis.world._IDENTITY_SCOPES`, which this pass cannot reuse because
#: the world reports one verdict for the whole script and this pass needs the individual node.
_IDENTITY_SCOPES = frozenset({
    Ps1ScopeModifier.ALIAS,
    Ps1ScopeModifier.FUNCTION,
})


def _binds_command_identity(node: Node) -> bool:
    """
    Whether `node` writes the `function:` or `alias:` namespace, binding a command name to something
    the definition scan does not read as a definition and the call scan does not read as a call.
    Both of this module's name-keyed removals have to treat such a node as an unknown: it can bind
    the name they are about to delete, and it can be the only thing that reaches it.
    """
    if not isinstance(node, Ps1AssignmentExpression):
        return False
    return any(
        variable.scope in _IDENTITY_SCOPES
        for variable in assignment_target_variables(node.target)
    )


def _drop_store_keep_value(stmt: Node, rhs: Node) -> bool:
    """
    Replace `stmt` — a dead store — with a discard of `rhs`, so the store goes but whatever
    evaluating the value does survives. Returns whether the tree changed.

    The discard wrapper is not decoration. A bare expression statement *emits* its value, where the
    assignment being removed swallowed it, so rewriting `$unused = [ordered]@{ a = 1 }` to the
    hashtable alone makes the deobfuscated script print something the original never printed — and
    inside a function body, return it. `$Null = ...` keeps the work and emits nothing, which is what
    the dead store did.

    A right-hand side that is a *statement* rather than an expression — `$x = if ($c) { ... }`, and
    the `switch`/`foreach`/`while`/`try` forms beside it, all legal assignment sources — has no
    expression-statement spelling to be moved into, so the whole assignment stays and this reports
    no change. Removing it instead would take the branch bodies with it.

    Both passes that drop a dead store share this, rather than each spelling the rule: the two
    outcomes are keep-the-value and keep-the-statement, and a pass that learned only one of them
    would delete what the other keeps.
    """
    if not isinstance(rhs, Expression):
        return False
    target = Ps1Variable(name='Null')
    discard = Ps1AssignmentExpression(target=target, operator='=', value=rhs)
    target.parent = discard
    rhs.parent = discard
    replacement = Ps1ExpressionStatement(expression=discard)
    discard.parent = replacement
    _replace_in_parent(stmt, replacement)
    return True


class Ps1UnusedVariableRemoval(Transformer):
    """
    Remove assignments to variables that are never read anywhere in the outer scope. Liveness comes
    from the shared `refinery.lib.scripts.ps1.analysis.model.Ps1SemanticModel`, so a read that
    reaches the assignment through a nested function, a captured scriptblock, or a scope qualifier
    keeps it alive. When the right-hand side of a removable assignment has side effects, the
    assignment wrapper is stripped but the expression is preserved as a standalone statement.
    """

    def visit(self, node: Node):
        cache = model_cache(self, node)
        model = cache.model
        oracle = cache.oracle
        candidates: dict[Binding, list[Node]] = {}
        for binding in model.script_scope.bindings.values():
            if binding.dynamic_or_qualified or binding.name in _PS1_SKIP_VARIABLES:
                continue
            mutations = self._removable_mutations(binding)
            if mutations:
                candidates[binding] = mutations
        if not candidates:
            return None
        dead = self._dead_bindings(candidates)
        if not dead:
            return None
        removable = self._removable_statements(candidates, dead, model)
        if not removable:
            return None
        body = get_body(node)
        if body is not None:
            dead_stmts: set[Node] = set()
            for mutation in removable:
                stmt = _find_removable_statement(mutation)
                if stmt is not None:
                    dead_stmts.add(stmt)
            surviving = [
                s for s in body
                if s not in dead_stmts
                and not isinstance(s, Ps1FunctionDefinition)
            ]
            if not surviving:
                return None
        for mutation in removable:
            self._remove_mutation(mutation, oracle)

    @staticmethod
    def _removable_mutations(binding: Binding) -> list[Node]:
        """
        The removable mutation nodes that write `binding`: a bare (unqualified) or `$env:`
        assignment or a `++`/`--` update. A parameter or `foreach` loop variable writes the binding
        but is not a removable mutation, and a scope-qualified write (`$script:x = ...`) is never
        removed, so both are excluded.
        """
        mutations: list[Node] = []
        seen: set[int] = set()
        for var in binding.writes:
            if var.scope not in (Ps1ScopeModifier.NONE, Ps1ScopeModifier.ENV):
                continue
            mutation = Ps1UnusedVariableRemoval._mutation_of(var)
            if mutation is not None and id(mutation) not in seen:
                seen.add(id(mutation))
                mutations.append(mutation)
        return mutations

    @staticmethod
    def _mutation_of(var: Ps1Variable) -> Node | None:
        assignment = assignment_of(var)
        if assignment is not None:
            return assignment
        parent = var.parent
        if isinstance(parent, Ps1UnaryExpression) and parent.operator in ('++', '--'):
            if parent.operand is var:
                return parent
        return None

    def _dead_bindings(self, candidates: dict[Binding, list[Node]]) -> list[Binding]:
        """
        From candidate bindings mapped to their removable mutations, return those that are dead. A
        binding is live if it has a read not contained in the right-hand side of any candidate
        assignment — a use in live code, in a live function, or a captured scriptblock. Liveness
        propagates back along right-hand sides: if a live binding's assignment reads another
        candidate, that candidate is live too. The rest, whose every read sits inside the right-hand
        side of an assignment that is itself dead, are dead — removing those assignments removes the
        reads, so nothing observes the value.
        """
        bindings = list(candidates)
        rhs_owner: dict[int, Binding] = {}
        for binding, mutations in candidates.items():
            for mutation in mutations:
                if not isinstance(mutation, Ps1AssignmentExpression) or mutation.value is None:
                    continue
                if len(assignment_target_variables(mutation.target)) == 1:
                    rhs_owner[id(mutation.value)] = binding
        readers: dict[Binding, set[Binding]] = {binding: set() for binding in bindings}
        live: set[Binding] = set()
        for binding in bindings:
            for read in binding.reads:
                owner = self._covering_owner(read, rhs_owner)
                if owner is None or owner is binding:
                    live.add(binding)
                else:
                    readers[binding].add(owner)
        changed = True
        while changed:
            changed = False
            for binding in bindings:
                if binding not in live and readers[binding] & live:
                    live.add(binding)
                    changed = True
        return [binding for binding in bindings if binding not in live]

    @staticmethod
    def _covering_owner(node: Node, rhs_owner: dict[int, Binding]) -> Binding | None:
        """
        The candidate binding whose assignment right-hand side encloses *node*, taken at the
        outermost such right-hand side, or `None` when *node* lies outside every candidate
        right-hand side.
        """
        owner: Binding | None = None
        cursor: Node | None = node
        while cursor is not None:
            found = rhs_owner.get(id(cursor))
            if found is not None:
                owner = found
            cursor = cursor.parent
        return owner

    def _removable_statements(
        self, candidates: dict[Binding, list[Node]], dead: list[Binding], model: Ps1SemanticModel,
    ) -> list[Node]:
        dead_set = set(dead)
        removable: list[Node] = []
        seen: set[int] = set()
        for binding in dead:
            for mutation in candidates[binding]:
                if id(mutation) in seen:
                    continue
                if (
                    isinstance(mutation, Ps1AssignmentExpression)
                    and not self._all_targets_dead(mutation, model, dead_set)
                ):
                    continue
                seen.add(id(mutation))
                removable.append(mutation)
        return removable

    @staticmethod
    def _all_targets_dead(
        assign: Ps1AssignmentExpression, model: Ps1SemanticModel, dead: set[Binding],
    ) -> bool:
        """
        Whether every variable written by `assign` is dead. A multi-assignment such as
        `$a, $b = 1, 2` is only removable when all of its targets are dead; removing it while a
        co-target is still live would destroy that live write. A target that contains a non-variable
        slot (e.g. `$arr[0], $b`) writes to memory beyond the named variables and is never
        considered fully dead.
        """
        if not assignment_target_is_all_variables(assign.target):
            return False
        variables = assignment_target_variables(assign.target)
        if not variables:
            return False
        for var in variables:
            binding = model.binding_of(var)
            if binding is None or binding not in dead:
                return False
        return True

    def _remove_mutation(self, mutation: Node, oracle: TypeOracle):
        if isinstance(mutation, Ps1AssignmentExpression):
            rhs = mutation.value
            if rhs is not None and not is_side_effect_free(rhs, oracle):
                self._drop_store_keep_value(mutation, rhs)
                return
        elif not isinstance(mutation, Ps1UnaryExpression):
            return
        stmt = _find_removable_statement(mutation)
        if stmt is not None and _remove_from_parent(stmt):
            self.mark_changed()

    def _drop_store_keep_value(self, mutation: Ps1AssignmentExpression, rhs: Node):
        """
        Replace the statement holding `mutation` with its right-hand side alone: the store is dead
        but evaluating the value is not, so what it does has to survive.
        """
        stmt = _find_removable_statement(mutation)
        if stmt is not None and _drop_store_keep_value(stmt, rhs):
            self.mark_changed()


class Ps1JunkStatementRemoval(Transformer):
    """
    Remove standalone expression statements that produce no observable side effects (junk/noise
    injected for anti-analysis) and function definitions that are never called.
    """

    def visit(self, node: Node):
        oracle = model_cache(self, node).oracle
        called = self._reachable_functions(node)
        for parent in list(node.walk()):
            role = body_role(parent)
            if role is None or role is BodyRole.OPAQUE:
                continue
            self._prune_body(get_body(parent), role, called, oracle)
        self._remove_inert_functions(node, oracle)

    @staticmethod
    def _reachable_functions(node: Node) -> set[str]:
        """
        Collect all function names transitively reachable from top-level call sites. First gather
        direct calls from the outer scope, then expand through function bodies until stable. When an
        invocation may dispatch dynamically (see `refinery.lib.scripts.ps1.ast.is_opaque_dispatch`),
        every defined function is treated as reachable so a function called only through `& $f` is
        never removed.

        Definitions and calls are keyed through `refinery.lib.scripts.ps1.ast.normalize_command_name`,
        so `function global:F` is reached by a call to `F`. Every definition sharing a key is walked,
        not just the last one: a redefinition shadows the earlier body only from the point it runs,
        which is order and scope information this pass does not have.
        """
        directly_called: set[str] = set()
        functions: dict[str, list[Ps1FunctionDefinition]] = {}
        dynamic_call = False
        for n in _walk_outer_scope(node):
            if isinstance(n, Ps1CommandInvocation):
                name = get_command_name(n)
                if name is not None:
                    directly_called.add(normalize_command_name(name))
                elif is_opaque_dispatch(n):
                    dynamic_call = True
            elif isinstance(n, Ps1FunctionDefinition):
                functions.setdefault(normalize_command_name(n.name), []).append(n)
            elif _binds_command_identity(n):
                # `${alias:q} = 'j'` reaches `j` without naming it in command position, so a
                # definition with no call site is not evidence of anything once one of these is in
                # the tree. Treated like an opaque dispatch, which is the same unknown.
                dynamic_call = True
        if dynamic_call:
            return set(functions.keys()) | directly_called
        reachable = set(directly_called)
        frontier = list(reachable & functions.keys())
        while frontier:
            for fdef in functions[frontier.pop()]:
                if fdef.body is None:
                    continue
                for n in fdef.body.walk():
                    if isinstance(n, Ps1CommandInvocation):
                        name = get_command_name(n)
                        if name is None:
                            if is_opaque_dispatch(n):
                                return set(functions.keys()) | reachable
                            continue
                        key = normalize_command_name(name)
                        if key not in reachable:
                            reachable.add(key)
                            if key in functions:
                                frontier.append(key)
        return reachable

    def _remove_inert_functions(self, node: Node, oracle: TypeOracle):
        """
        Remove top-level functions whose body carries no observable output or side effect together
        with the bare call statements that invoke them. After body pruning, an injected junk function
        such as `function j { $Null = 915 }` has an empty body; calling it is a no-op, so the
        definition and its call sites drop out as a unit. Only whole-statement, argument-free
        invocations of the function count as call sites — if the name is referenced any other way, or
        anything in the script dispatches dynamically (`& $f`), the function is kept, because its
        result might then be observed or its identity might not be provable.

        A name is inert only when *every* definition of it is, since the calls are attributed to the
        name rather than to one of its definitions: removing them because one definition is empty
        would silence a payload-bearing other one. Definitions are looked for over the whole tree,
        not only at the top level, because a `function` inside an `if` or a loop body is written
        into the enclosing scope just the same — while only a top-level definition is itself
        removable, since a nested one is not this pass's to reason about.

        "Every definition" can only mean every definition standing in this tree, so the tree has to
        be the whole story: an open world binds names from a file the walk never read, and an
        identity-namespace assignment binds one by a spelling this scan does not read as a
        definition. In either case the empty body standing here is not the body the call reaches.

        Both walks run in source order, because removal is by identity scan over the containing
        list: taking the reverse order `Node.walk` yields would delete from the back and make every
        scan traverse the whole body.
        """
        if self._any_dynamic_dispatch(node) or not oracle.world_closed_at(node):
            return
        inert: dict[str, list[Ps1FunctionDefinition]] = {}
        acting: set[str] = set()
        for definition in node.walk_in_order():
            if _binds_command_identity(definition):
                return
            if not isinstance(definition, Ps1FunctionDefinition):
                continue
            key = normalize_command_name(definition.name)
            if not body_is_inert(definition.body, oracle):
                acting.add(key)
            elif definition.parent is node:
                inert.setdefault(key, []).append(definition)
        for key in acting:
            inert.pop(key, None)
        if not inert:
            return
        call_sites: dict[str, list[Node]] = {name: [] for name in inert}
        other_reference: set[str] = set()
        for ref in node.walk_in_order():
            if not isinstance(ref, Ps1CommandInvocation):
                continue
            name = get_command_name(ref)
            if name is None:
                continue
            key = normalize_command_name(name)
            if key not in inert:
                continue
            statement = self._bare_call_statement(ref)
            if statement is not None and not ref.arguments:
                call_sites[key].append(statement)
            else:
                other_reference.add(key)
        for key, definitions in inert.items():
            if key in other_reference:
                continue
            for statement in call_sites[key]:
                if _remove_from_parent(statement):
                    self.mark_changed()
            for definition in definitions:
                if _remove_from_parent(definition):
                    self.mark_changed()

    @staticmethod
    def _bare_call_statement(cmd: Ps1CommandInvocation) -> Node | None:
        """
        Return the enclosing statement when `cmd` is invoked as a whole expression statement (`f`
        alone, or `f` as the sole element of a statement-level pipeline), or `None` when its result
        flows into a larger expression where the call's value could be observed.
        """
        parent = cmd.parent
        if isinstance(parent, Ps1ExpressionStatement):
            return parent
        if isinstance(parent, Ps1PipelineElement):
            pipeline = parent.parent
            if (
                isinstance(pipeline, Ps1Pipeline)
                and len(pipeline.elements) == 1
                and isinstance(pipeline.parent, Ps1ExpressionStatement)
            ):
                return pipeline.parent
        return None

    @staticmethod
    def _any_dynamic_dispatch(node: Node) -> bool:
        for n in node.walk():
            if isinstance(n, Ps1CommandInvocation):
                if get_command_name(n) is None and is_opaque_dispatch(n):
                    return True
        return False

    def _prune_body(self, body: list, role: BodyRole, called: set[str], oracle: TypeOracle):
        discard: set[Node] = set()
        output: set[Node] = set()
        dead_functions: set[Node] = set()
        for stmt in body:
            if isinstance(stmt, Ps1FunctionDefinition):
                # `called` is every call site standing in this tree, which is only the whole story
                # while the tree is the whole script. When the world is open, a dot-sourced file, an
                # imported module or an `iex` holds call sites the walk never read, so a definition
                # with none here is not unreachable — it is reachable from somewhere unreadable.
                if (
                    role is BodyRole.SCRIPT
                    and oracle.world_closed_at(stmt)
                    and normalize_command_name(stmt.name) not in called
                ):
                    dead_functions.add(stmt)
                continue
            effect = statement_effect(stmt, oracle)
            if effect is StatementEffect.DISCARD:
                discard.add(stmt)
            elif effect is StatementEffect.OUTPUT:
                output.add(stmt)
        # A body whose value is observed may exist only to produce it, so a pure output statement
        # is given up only when another survivor still carries the output. A `DISCARD` emits nothing
        # and is always safe to drop, even when it empties the body — that is what turns a junk
        # function inert.
        if output and output_observed(role):
            if not output_is_covered(self._survivors(body, discard | output | dead_functions)):
                output.clear()
        removable = discard | output | dead_functions
        if not removable:
            return
        if pruning_erases_body(role, self._survivors(body, removable)):
            return
        for stmt in list(body):
            if stmt in removable:
                if _remove_from_parent(stmt):
                    self.mark_changed()

    @staticmethod
    def _survivors(body: list, removable: set[Node]) -> list[Node]:
        return [stmt for stmt in body if stmt not in removable]


class Ps1DeadStoreElimination(Transformer):
    """
    Remove assignments to a variable that are provably overwritten before their next read within the
    same linear scope. This targets the pattern left behind by tier-2 empty-for rewriting: dozens of
    `$i = N` statements followed by a for-loop whose initializer `$i = 0` overwrites them all. Reads
    come from the shared `refinery.lib.scripts.ps1.analysis.model.Ps1SemanticModel`, so a store read
    only through a nested scriptblock is correctly seen as live rather than skipped.
    """

    def __init__(self):
        super().__init__()
        self._oracle: TypeOracle | None = None

    def visit(self, node: Node):
        cache = model_cache(self, node)
        model = cache.model
        # The model is re-read per body, because removing a store changes what the next body's scope
        # says; the world behind the oracle is not, because it is a whole-script fact and this pass
        # only removes stores. Re-reading it would rebuild the whole-tree walk after every removal
        # to reach a verdict that can only have become *more* closed, and the captured one — taken
        # before those removals — is the more open, and so more conservative, of the two.
        if self._oracle is None:
            self._oracle = cache.oracle
        oracle = self._oracle
        body = get_body(node)
        scope = model.scope_of(node)
        if body is None or scope is None:
            self.generic_visit(node)
            return None
        pending: dict[str, list[Ps1ExpressionStatement]] = {}
        dead: list[Ps1ExpressionStatement] = []
        has_read: set[str] = set()
        for stmt in body:
            reads, kills, writes = self._classify_statement(stmt, scope, model)
            for var in kills:
                if var in pending:
                    dead.extend(pending.pop(var))
                has_read.add(var)
            for var in reads:
                pending.pop(var, None)
                has_read.add(var)
            for var, store in writes:
                if var in pending:
                    dead.extend(pending[var])
                pending[var] = [store]
        if isinstance(node, Ps1Script):
            for var, stores in pending.items():
                if var in has_read:
                    dead.extend(stores)
        if not dead:
            self.generic_visit(node)
            return None
        for stmt in dead:
            if not isinstance(stmt, Ps1ExpressionStatement):
                continue
            rhs = stmt.expression
            if isinstance(rhs, Ps1AssignmentExpression):
                rhs = rhs.value
            if rhs is not None and not is_side_effect_free(rhs, oracle):
                if not _drop_store_keep_value(stmt, rhs):
                    continue
            else:
                _remove_from_parent(stmt)
            self.mark_changed()
        self.generic_visit(node)

    def _classify_statement(
        self, stmt, scope: Scope, model: Ps1SemanticModel,
    ) -> tuple[set[str], set[str], list[tuple[str, Ps1ExpressionStatement]]]:
        """
        Return `(reads, kills, writes)` for a top-level statement:
        - `reads`: variable keys read by the statement (flush: pending writes are needed).
        - `kills`: variable keys whose previous pending writes are dead (overwritten before read).
        - `writes`: list of `(key, statement)` pairs for pure assignments to register as pending.

        A for-loop's initializer kills previous stores without itself becoming a pending write
        (the for-loop is not a removable expression statement). Other control-flow constructs
        contribute all internal variable references as reads and produce no kills/writes.
        """
        reads: set[str] = set()
        kills: set[str] = set()
        writes: list[tuple[str, Ps1ExpressionStatement]] = []
        if isinstance(stmt, Ps1ExpressionStatement) and isinstance(
            stmt.expression, Ps1AssignmentExpression
        ):
            assign = stmt.expression
            if (
                assign.operator == '='
                and isinstance(assign.target, Ps1Variable)
                and assign.target.scope is Ps1ScopeModifier.NONE
            ):
                key = assign.target.name.lower()
                if key not in _PS1_SKIP_VARIABLES:
                    if assign.value is not None:
                        reads |= model.reads_in_scope(assign.value, scope)
                    writes.append((key, stmt))
                    return reads, kills, writes
        if isinstance(stmt, Ps1ForLoop):
            if (
                isinstance(stmt.initializer, Ps1AssignmentExpression)
                and stmt.initializer.operator == '='
                and isinstance(stmt.initializer.target, Ps1Variable)
                and stmt.initializer.target.scope is Ps1ScopeModifier.NONE
            ):
                key = stmt.initializer.target.name.lower()
                if key not in _PS1_SKIP_VARIABLES:
                    init_rhs_reads: set[str] = set()
                    if stmt.initializer.value is not None:
                        init_rhs_reads = model.reads_in_scope(stmt.initializer.value, scope)
                    if key in init_rhs_reads:
                        reads.update(init_rhs_reads)
                    else:
                        kills.add(key)
        reads |= model.variables_in_scope(stmt, scope)
        return reads, kills, writes
