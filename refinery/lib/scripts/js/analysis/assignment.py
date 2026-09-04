"""
Definite assignment for implicit globals: a forward must-analysis over the per-function control-flow
graphs answering whether a write to a name the program never declares has certainly completed before
a given reference runs. The consumers are the removal sweep and the alias collapse, both of which may
treat a read of such a name as unable to throw only where this model vouches for it.

The transfer function reasons about normal completion alone: a statement's gen set holds the bindings
whose write is performed on *every* evaluation of the statement that completes normally, computed by
an evaluation-order walk of the statement's own expressions, so a write guarded by a short-circuit, a
ternary arm, an optional chain, a logical assignment's right-hand side, or a loop target contributes
nothing. A write whose statement throws first is discarded structurally — the throw leaves along the
raising edge, which carries no gens, and the meet at the join forgets the fact — except where the
statement provably cannot throw at all, whose raising edge is never taken and may carry them
vacuously. No throw analysis decides what the normal edge means.

A bare write in sloppy script code creates the property; in strict code, or anywhere under the module
reading, it throws instead, so it gens nothing there. A member write through this realm's global
object creates the property in either mode; `top` and `frames` name another document's global and
never qualify.

Facts are monotone: a name is tracked only while nothing the analysis cannot see can unbind it. A
binding a reflection surface can reach is not tracked, and neither is one any `delete` in the program
addresses — through its bare name, through a global-object member, or possibly through a computed key
no scan can read, which untracks everything. What remains can only ever gain its property, so there
are no kill sets, and a fact held at a call site still holds whenever a body invoked there runs —
including a getter, an iterator, or an async continuation the text spells no call for.

Interprocedurally, a call completing normally has run its callee to normal exit, so a call whose
callee is one known non-async, non-generator function — resolved through its name, or spelled inline
as the callee — adds that function's exit summary to the gen set; and a function whose every
invocation site is a direct call of its one pinned name, or an immediately-invoked function
expression, takes the meet of the facts at those sites as its entry fact. Both start empty and grow
to a least fixpoint, so no fact ever rests on itself.
"""
from __future__ import annotations

from refinery.lib.scripts import Node, Statement
from refinery.lib.scripts.js.analysis.cfg import (
    CfgNode,
    ControlFlowGraph,
    ControlFlowModel,
    build_control_flow_model,
)
from refinery.lib.scripts.js.analysis.model import (
    FUNCTION_NODES,
    GLOBAL_OBJECT_ALIASES,
    SAME_REALM_GLOBAL_OBJECT_ALIASES,
    Binding,
    BindingKind,
    SemanticModel,
    may_be_global_object_base,
)
from refinery.lib.scripts.js.model import (
    JsArrayExpression,
    JsAssignmentExpression,
    JsBinaryExpression,
    JsBooleanLiteral,
    JsCallExpression,
    JsConditionalExpression,
    JsExpressionStatement,
    JsForInStatement,
    JsForOfStatement,
    JsIdentifier,
    JsLogicalExpression,
    JsMemberExpression,
    JsNewExpression,
    JsNullLiteral,
    JsNumericLiteral,
    JsObjectExpression,
    JsParenthesizedExpression,
    JsProperty,
    JsSequenceExpression,
    JsSpreadElement,
    JsStringLiteral,
    JsTaggedTemplateExpression,
    JsTemplateLiteral,
    JsUnaryExpression,
    JsVariableDeclaration,
    strip_parens,
)
from refinery.lib.scripts.js.strict import strict_mode_at

_OTHER_REALM_ALIASES = GLOBAL_OBJECT_ALIASES - SAME_REALM_GLOBAL_OBJECT_ALIASES

_LOGICAL_ASSIGNMENT = frozenset({
    '&&=',
    '||=',
    '??=',
})


class DefiniteAssignmentModel:
    """
    Build over a `SemanticModel` and its `ControlFlowModel`; query with `definitely_assigned_at`.
    *module_scope* is the run's execution model: under the module reading every region is strict and
    a top-level `this` does not denote the global object, so no bare write and no `this` member
    write establishes anything.
    """

    def __init__(
        self,
        model: SemanticModel,
        control_flow: ControlFlowModel | None = None,
        *,
        module_scope: bool = False,
    ):
        self.model = model
        self._flow = control_flow if control_flow is not None else build_control_flow_model(model.root)
        self._module = module_scope
        self._tracked: dict[int, Binding] = {}
        self._by_name: dict[str, Binding] = {}
        for binding in model.root_scope.bindings.values():
            if binding.kind is not BindingKind.IMPLICIT_GLOBAL:
                continue
            if not binding.writes:
                continue
            if model.reflection_can_reach(binding):
                continue
            self._tracked[id(binding)] = binding
            self._by_name[binding.name] = binding
        self._in_facts: dict[int, frozenset[Binding]] = {}
        self._effects: dict[int, tuple[frozenset[Binding], bool]] = {}
        self._summaries: dict[int, frozenset[Binding]] = {}
        if self._tracked:
            self._untrack_deleted()
        if self._tracked:
            self._solve()

    def definitely_assigned_at(self, binding: Binding | None, reference: Node) -> bool:
        """
        Whether a write to *binding* has certainly completed before *reference* is evaluated. Facts
        hold at statement entry, so a write in *reference*'s own statement never vouches for it.
        """
        if binding is None or id(binding) not in self._tracked:
            return False
        located = self._flow.locate(reference)
        if located is None:
            return False
        _, node = located
        return binding in self._in_facts.get(id(node), frozenset())

    def _untrack_deleted(self):
        """
        Remove from tracking every binding a `delete` in the program could unbind, wherever that
        delete stands — a called function, a getter a member read fires, an iterator a spread drives.
        Untracking is what makes the remaining facts monotone without enumerating who runs the code.
        """
        deleted: set[int] = set()
        for node in self.model.root.walk():
            if not isinstance(node, JsUnaryExpression) or node.operator != 'delete':
                continue
            target = strip_parens(node.operand) if node.operand is not None else None
            if isinstance(target, JsIdentifier):
                resolved = self.model.resolve(target)
                if resolved is not None:
                    deleted.add(id(resolved))
                    continue
                self._tracked = {}
                return
            if isinstance(target, JsMemberExpression):
                base = target.object
                if base is not None and not (
                    may_be_global_object_base(base)
                    or self.model.names_the_global_object(base)
                ):
                    continue
                name = self.model.global_alias_member_name(target, module_scope=self._module)
                if name is not None:
                    named = self._by_name.get(name)
                    if named is not None:
                        deleted.add(id(named))
                    continue
                self._tracked = {}
                return
            self._tracked = {}
            return
        if deleted:
            self._tracked = {
                key: binding for key, binding in self._tracked.items() if key not in deleted
            }
            self._by_name = {binding.name: binding for binding in self._tracked.values()}

    def _solve(self):
        graphs = list(self._flow.graphs.values())
        summaries: dict[int, frozenset[Binding]] = {}
        entries: dict[int, frozenset[Binding]] = {}
        while True:
            self._summaries = summaries
            self._effects = {}
            for graph in graphs:
                for node in graph.nodes:
                    if node.element is not None:
                        self._effects[id(node)] = (
                            self._parts_gen(graph, node.element),
                            self._parts_cannot_throw(graph, node.element),
                        )
            facts: dict[int, frozenset[Binding]] = {}
            for graph in graphs:
                self._solve_graph(graph, entries.get(id(graph.owner), frozenset()), facts)
            new_summaries = {id(graph.owner): self._exit_summary(graph, facts) for graph in graphs}
            new_entries = {id(graph.owner): self._entry_fact(graph, facts) for graph in graphs}
            if new_summaries == summaries and new_entries == entries:
                self._in_facts = facts
                return
            summaries = new_summaries
            entries = new_entries

    def _edge_out(
        self,
        graph: ControlFlowGraph,
        pred: CfgNode,
        target: CfgNode,
        state: dict[int, frozenset[Binding]],
    ) -> frozenset[Binding]:
        gen, cannot_throw = self._effects.get(id(pred), (frozenset(), False))
        out = state[id(pred)]
        if cannot_throw or not graph.raise_taken(pred, target):
            out = out | gen
        return out

    def _solve_graph(
        self, graph: ControlFlowGraph, entry: frozenset[Binding], facts: dict[int, frozenset[Binding]],
    ):
        top = frozenset(self._tracked.values())
        state: dict[int, frozenset[Binding]] = {id(node): top for node in graph.nodes}
        state[id(graph.entry)] = entry
        reachable: set[int] = set()
        frontier = [graph.entry]
        while frontier:
            node = frontier.pop()
            if id(node) in reachable:
                continue
            reachable.add(id(node))
            frontier.extend(node.successors)
        changed = True
        while changed:
            changed = False
            for node in graph.nodes:
                if node is graph.entry:
                    continue
                met: frozenset[Binding] | None = None
                for pred in node.predecessors:
                    out = self._edge_out(graph, pred, node, state)
                    met = out if met is None else met & out
                if met is None:
                    met = frozenset()
                if met != state[id(node)]:
                    state[id(node)] = met
                    changed = True
        for node in graph.nodes:
            facts[id(node)] = state[id(node)] if id(node) in reachable else frozenset()

    def _exit_summary(
        self, graph: ControlFlowGraph, facts: dict[int, frozenset[Binding]],
    ) -> frozenset[Binding]:
        met: frozenset[Binding] | None = None
        for pred in graph.exit.predecessors:
            if graph.raise_taken(pred, graph.exit):
                continue
            gen, _ = self._effects.get(id(pred), (frozenset(), False))
            out = facts.get(id(pred), frozenset()) | gen
            met = out if met is None else met & out
        entry = facts.get(id(graph.entry), frozenset())
        return frozenset() if met is None else frozenset(met - entry)

    def _entry_fact(
        self, graph: ControlFlowGraph, facts: dict[int, frozenset[Binding]],
    ) -> frozenset[Binding]:
        owner = graph.owner
        if not isinstance(owner, FUNCTION_NODES):
            return frozenset()
        if getattr(owner, 'generator', False):
            return frozenset()
        binding = self.model.invocation_binding(owner)
        if binding is None:
            immediate = self._immediate_call_of(owner)
            if immediate is None:
                return frozenset()
            return self._fact_at(immediate, facts)
        if binding.dynamic_refs or not self.model.binding_pinned_to(binding, owner):
            return frozenset()
        met: frozenset[Binding] | None = None
        for read in binding.reads:
            if not self._is_direct_callee(read):
                return frozenset()
            met_here = self._fact_at(read, facts)
            met = met_here if met is None else met & met_here
        return met or frozenset()

    def _fact_at(self, site: Node, facts: dict[int, frozenset[Binding]]) -> frozenset[Binding]:
        located = self._flow.locate(site)
        if located is None:
            return frozenset()
        return facts.get(id(located[1]), frozenset())

    @staticmethod
    def _immediate_call_of(function: Node) -> JsCallExpression | None:
        cursor: Node = function
        parent = cursor.parent
        while isinstance(parent, JsParenthesizedExpression):
            cursor = parent
            parent = parent.parent
        if (
            isinstance(parent, JsCallExpression)
            and parent.callee is not None
            and strip_parens(parent.callee) is function
            and not parent.optional
        ):
            return parent
        return None

    @staticmethod
    def _is_direct_callee(read: Node) -> bool:
        parent = read.parent
        while isinstance(parent, JsParenthesizedExpression):
            read = parent
            parent = parent.parent
        return (
            isinstance(parent, JsCallExpression)
            and parent.callee is not None
            and strip_parens(parent.callee) is read
            and not parent.optional
        )

    def _evaluated_parts(self, graph: ControlFlowGraph, element: Node) -> list[Node]:
        """
        The expressions the control-flow node for *element* evaluates itself: the element where it is
        an expression handed its own node (a loop header's init, test, or update), a declaration's
        initializers, a `for-in`/`for-of` head's subject, and otherwise every child that is neither a
        function body nor a statement owned by another node.
        """
        if not isinstance(element, (
            JsVariableDeclaration,
            JsForInStatement,
            JsForOfStatement,
        )) and not self._is_statement(element):
            return [element]
        if isinstance(element, JsVariableDeclaration):
            inits: list[Node] = []
            for declarator in element.declarations:
                init = getattr(declarator, 'init', None)
                if init is not None:
                    inits.append(init)
            return inits
        if isinstance(element, (JsForInStatement, JsForOfStatement)):
            subject = getattr(element, 'right', None)
            return [subject] if subject is not None else []
        parts: list[Node] = []
        for child in element.children():
            if isinstance(child, FUNCTION_NODES):
                continue
            if graph.node_of(child) is not None:
                continue
            parts.append(child)
        return parts

    @staticmethod
    def _is_statement(element: Node) -> bool:
        return isinstance(element, Statement)

    def _parts_gen(self, graph: ControlFlowGraph, element: Node) -> frozenset[Binding]:
        gens: set[Binding] = set()
        for part in self._evaluated_parts(graph, element):
            gens |= self._gens(part)
        return frozenset(gens)

    def _parts_cannot_throw(self, graph: ControlFlowGraph, element: Node) -> bool:
        if not isinstance(element, (JsExpressionStatement, JsVariableDeclaration)):
            return False
        return all(self._cannot_throw(part) for part in self._evaluated_parts(graph, element))

    def _cannot_throw(self, node: Node | None) -> bool:
        """
        Whether evaluating *node* is guaranteed not to throw, so its statement's raising edge is never
        taken and may vacuously carry its gens. Deliberately minimal: literals, function values, a
        sloppy bare store of such a value to a tracked name, and a member store of one through this
        realm's global object — the shapes a `try`-wrapped establishment scaffold is written in.
        """
        if node is None:
            return False
        if isinstance(node, (
            JsStringLiteral,
            JsNumericLiteral,
            JsBooleanLiteral,
            JsNullLiteral,
        )):
            return True
        if isinstance(node, FUNCTION_NODES):
            return True
        if isinstance(node, JsParenthesizedExpression):
            return node.expression is not None and self._cannot_throw(node.expression)
        if isinstance(node, JsSequenceExpression):
            return all(self._cannot_throw(part) for part in node.expressions)
        if isinstance(node, JsAssignmentExpression):
            if node.operator != '=':
                return False
            if not self._target_gen(node):
                return False
            return self._cannot_throw(node.right)
        return False

    def _gens(self, node: Node | None) -> frozenset[Binding]:
        if node is None:
            return frozenset()
        if isinstance(node, JsParenthesizedExpression):
            return self._gens(node.expression)
        if isinstance(node, JsAssignmentExpression):
            if node.operator in _LOGICAL_ASSIGNMENT:
                return self._target_gen(node)
            return self._gens(node.right) | self._target_gen(node)
        if isinstance(node, JsLogicalExpression):
            return self._gens(node.left)
        if isinstance(node, JsConditionalExpression):
            return self._gens(node.test) | (
                self._gens(node.consequent) & self._gens(node.alternate))
        if isinstance(node, JsBinaryExpression):
            return self._gens(node.left) | self._gens(node.right)
        if isinstance(node, JsSequenceExpression):
            result: frozenset[Binding] = frozenset()
            for expression in node.expressions:
                result = result | self._gens(expression)
            return result
        if isinstance(node, JsUnaryExpression):
            if node.operator == 'delete':
                return frozenset()
            return self._gens(node.operand)
        if isinstance(node, JsMemberExpression):
            result = self._gens(node.object)
            if node.optional or self._chain_short_circuits(node.object):
                return result
            if node.computed:
                result = result | self._gens(node.property)
            return result
        if isinstance(node, (JsCallExpression, JsNewExpression)):
            result = self._gens(node.callee)
            if getattr(node, 'optional', False) or self._chain_short_circuits(node.callee):
                return result
            for argument in node.arguments:
                result = result | self._gens(argument)
            if isinstance(node, JsCallExpression):
                result = result | self._callee_summary(node)
            return result
        if isinstance(node, JsSpreadElement):
            return self._gens(node.argument)
        if isinstance(node, JsArrayExpression):
            result = frozenset()
            for element in node.elements:
                if element is not None:
                    result = result | self._gens(element)
            return result
        if isinstance(node, JsObjectExpression):
            result = frozenset()
            for prop in node.properties:
                if isinstance(prop, JsProperty):
                    if prop.computed and prop.key is not None:
                        result = result | self._gens(prop.key)
                    if prop.value is not None and not isinstance(prop.value, FUNCTION_NODES):
                        result = result | self._gens(prop.value)
                elif isinstance(prop, JsSpreadElement):
                    result = result | self._gens(prop.argument)
            return result
        if isinstance(node, JsTemplateLiteral):
            result = frozenset()
            for expression in node.expressions:
                result = result | self._gens(expression)
            return result
        if isinstance(node, JsTaggedTemplateExpression):
            return self._gens(node.tag) | self._gens(node.quasi)
        return frozenset()

    def _target_gen(self, assignment: JsAssignmentExpression) -> frozenset[Binding]:
        target = strip_parens(assignment.left)
        if isinstance(target, JsIdentifier):
            binding = self.model.resolve(target)
            if binding is None or id(binding) not in self._tracked:
                return frozenset()
            if assignment.operator != '=':
                return frozenset({binding})
            if self._module or strict_mode_at(target):
                return frozenset()
            return frozenset({binding})
        if isinstance(target, JsMemberExpression):
            if assignment.operator in _LOGICAL_ASSIGNMENT:
                return frozenset()
            base = strip_parens(target.object) if target.object is not None else None
            if isinstance(base, JsIdentifier) and base.name in _OTHER_REALM_ALIASES:
                return frozenset()
            name = self.model.global_alias_member_name(target, module_scope=self._module)
            if name is None:
                return frozenset()
            binding = self._by_name.get(name)
            if binding is None:
                return frozenset()
            return frozenset({binding})
        return frozenset()

    @staticmethod
    def _chain_short_circuits(callee: Node | None) -> bool:
        cursor = callee
        while cursor is not None:
            if isinstance(cursor, JsParenthesizedExpression):
                cursor = cursor.expression
                continue
            if isinstance(cursor, (JsMemberExpression, JsCallExpression)):
                if getattr(cursor, 'optional', False):
                    return True
                cursor = cursor.object if isinstance(cursor, JsMemberExpression) else cursor.callee
                continue
            return False
        return False

    def _callee_summary(self, call: JsCallExpression) -> frozenset[Binding]:
        callee = strip_parens(call.callee) if call.callee is not None else None
        function: Node | None = None
        if isinstance(callee, FUNCTION_NODES):
            function = callee
        elif isinstance(callee, JsIdentifier):
            binding = self.model.resolve(callee)
            if binding is None or binding.dynamic_refs:
                return frozenset()
            function = self.model.singular_value(binding)
        if function is None or not isinstance(function, FUNCTION_NODES):
            return frozenset()
        if getattr(function, 'generator', False) or getattr(function, 'is_async', False):
            return frozenset()
        return self._summaries.get(id(function), frozenset())


def build_definite_assignment(
    model: SemanticModel,
    control_flow: ControlFlowModel | None = None,
    *,
    module_scope: bool = False,
) -> DefiniteAssignmentModel:
    """
    Build the `DefiniteAssignmentModel` for *model*'s script.
    """
    return DefiniteAssignmentModel(model, control_flow, module_scope=module_scope)
