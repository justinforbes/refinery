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
from refinery.lib.scripts.ps1.ast import (
    assignment_of,
    assignment_target_is_all_variables,
    assignment_target_variables,
    get_body,
    get_command_name,
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
    Ps1ScriptBlock,
    Ps1StringLiteral,
    Ps1UnaryExpression,
    Ps1Variable,
)


class Ps1UnusedVariableRemoval(Transformer):
    """
    Remove assignments to variables that are never read anywhere in the outer scope. Liveness comes
    from the shared `refinery.lib.scripts.ps1.analysis.model.Ps1SemanticModel`, so a read that
    reaches the assignment through a nested function, a captured scriptblock, or a scope qualifier
    keeps it alive. When the right-hand side of a removable assignment has side effects, the
    assignment wrapper is stripped but the expression is preserved as a standalone statement.
    """

    def visit(self, node: Node):
        model = model_cache(self, node).model
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
            self._remove_mutation(mutation)

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

    def _remove_mutation(self, mutation: Node):
        if isinstance(mutation, Ps1AssignmentExpression):
            rhs = mutation.value
            if rhs is not None and not is_side_effect_free(rhs) and isinstance(rhs, Expression):
                stmt = _find_removable_statement(mutation)
                if stmt is None:
                    return
                replacement = Ps1ExpressionStatement(expression=rhs)
                _replace_in_parent(stmt, replacement)
                self.mark_changed()
            else:
                stmt = _find_removable_statement(mutation)
                if stmt is not None and _remove_from_parent(stmt):
                    self.mark_changed()
        elif isinstance(mutation, Ps1UnaryExpression):
            stmt = _find_removable_statement(mutation)
            if stmt is not None and _remove_from_parent(stmt):
                self.mark_changed()


class Ps1JunkStatementRemoval(Transformer):
    """
    Remove standalone expression statements that produce no observable side effects (junk/noise
    injected for anti-analysis) and function definitions that are never called.
    """

    def visit(self, node: Node):
        called = self._reachable_functions(node)
        for parent in list(node.walk()):
            role = body_role(parent)
            if role is None or role is BodyRole.OPAQUE:
                continue
            self._prune_body(get_body(parent), role, called)
        self._remove_inert_functions(node)

    @staticmethod
    def _is_dynamic_dispatch(cmd: Ps1CommandInvocation) -> bool:
        """
        Return `True` when an invocation may resolve to an arbitrary function name at runtime. A
        literal command name resolves statically, and a literal scriptblock body (`&{ ... }`) runs
        inline; any other command expression (a variable like `& $f`, an expandable string, or a
        subexpression) could dispatch to any defined function.
        """
        return not isinstance(cmd.name, (Ps1StringLiteral, Ps1ScriptBlock))

    @staticmethod
    def _reachable_functions(node: Node) -> set[str]:
        """
        Collect all function names transitively reachable from top-level call sites. First gather
        direct calls from the outer scope, then expand through function bodies until stable. When an
        invocation may dispatch dynamically (see `_is_dynamic_dispatch`), every defined function is
        treated as reachable so that a function called only through `& $f` is never removed.
        """
        directly_called: set[str] = set()
        functions: dict[str, Ps1FunctionDefinition] = {}
        dynamic_call = False
        for n in _walk_outer_scope(node):
            if isinstance(n, Ps1CommandInvocation):
                name = get_command_name(n)
                if name is not None:
                    directly_called.add(name.lower())
                elif Ps1JunkStatementRemoval._is_dynamic_dispatch(n):
                    dynamic_call = True
            elif isinstance(n, Ps1FunctionDefinition):
                functions[n.name.lower()] = n
        if dynamic_call:
            return set(functions.keys()) | directly_called
        reachable = set(directly_called)
        frontier = list(reachable & functions.keys())
        while frontier:
            fname = frontier.pop()
            fdef = functions[fname]
            if fdef.body is None:
                continue
            for n in fdef.body.walk():
                if isinstance(n, Ps1CommandInvocation):
                    name = get_command_name(n)
                    if name is None:
                        if Ps1JunkStatementRemoval._is_dynamic_dispatch(n):
                            return set(functions.keys()) | reachable
                        continue
                    key = name.lower()
                    if key not in reachable:
                        reachable.add(key)
                        if key in functions:
                            frontier.append(key)
        return reachable

    def _remove_inert_functions(self, node: Node):
        """
        Remove top-level functions whose body carries no observable output or side effect together
        with the bare call statements that invoke them. After body pruning, an injected junk function
        such as `function j { $Null = 915 }` has an empty body; calling it is a no-op, so the
        definition and its call sites drop out as a unit. Only whole-statement, argument-free
        invocations of the function count as call sites — if the name is referenced any other way, or
        anything in the script dispatches dynamically (`& $f`), the function is kept, because its
        result might then be observed or its identity might not be provable.
        """
        if self._any_dynamic_dispatch(node):
            return
        inert: dict[str, Ps1FunctionDefinition] = {}
        for stmt in node.body:
            if isinstance(stmt, Ps1FunctionDefinition) and body_is_inert(stmt.body):
                inert[stmt.name.lower()] = stmt
        if not inert:
            return
        call_sites: dict[str, list[Node]] = {name: [] for name in inert}
        other_reference: set[str] = set()
        for ref in node.walk():
            if not isinstance(ref, Ps1CommandInvocation):
                continue
            name = get_command_name(ref)
            if name is None:
                continue
            key = name.lower()
            if key not in inert:
                continue
            statement = self._bare_call_statement(ref)
            if statement is not None and not ref.arguments:
                call_sites[key].append(statement)
            else:
                other_reference.add(key)
        for key, definition in inert.items():
            if key in other_reference:
                continue
            for statement in call_sites[key]:
                if _remove_from_parent(statement):
                    self.mark_changed()
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
                if get_command_name(n) is None and Ps1JunkStatementRemoval._is_dynamic_dispatch(n):
                    return True
        return False

    def _prune_body(self, body: list, role: BodyRole, called: set[str]):
        discard: set[Node] = set()
        output: set[Node] = set()
        dead_functions: set[Node] = set()
        for stmt in body:
            if isinstance(stmt, Ps1FunctionDefinition):
                if role is BodyRole.SCRIPT and stmt.name.lower() not in called:
                    dead_functions.add(stmt)
                continue
            effect = statement_effect(stmt)
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
        self._root: Node | None = None

    def visit(self, node: Node):
        # The root is remembered rather than the model: `visit` recurses into every nested body,
        # and a cache built over a nested node would both rebuild the model per body and detach
        # this transform from the run's shared cache. Reading the model back through the cache on
        # every visit keeps it consistent with the tree after this pass removes a statement.
        if self._root is None:
            self._root = node
        model = model_cache(self, self._root).model
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
            if rhs is not None and isinstance(rhs, Expression) and not is_side_effect_free(rhs):
                replacement = Ps1ExpressionStatement(expression=rhs)
                _replace_in_parent(stmt, replacement)
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
