"""
Per-body control-flow graphs, derived from an AST and independent of the language it came from.

Each function — and the script itself — gets one `ControlFlowGraph`: a graph whose nodes wrap the
statements and loop-head expressions the body evaluates, connected by the order in which control may
pass between them. Sequential flow, the branches of a conditional, loop back-edges, the non-local
jumps a `break`/`continue`/`return` performs, and *exceptional* edges from any point inside a guarded
block to the handler that would catch a throw.

The graph is keyed to AST node identity (`node_of`) and is a disposable, per-body view — the tree
stays the spine. It is *conservative by construction*: where modelling control flow precisely would
be intricate (the order of evaluation inside an expression, the exact point a statement throws, a
finalizer on an exceptional path) the graph adds edges rather than omits them, so an analysis reading
it sees at least every path the program can take. Nested function bodies are not descended into; each
has its own graph.

**What a language supplies is the dispatch, not the shapes.** `CfgBuilder` holds the frontier
threading, the jump-target stack, the handler stack and one method per control-flow *shape* —
`branch_on`, `loop_head_tested`, `loop_tail_tested`, `loop_counted`, `dispatch`, `guarded`,
`labelled`, `jump_out`, `jump_back`, `terminate`. A subclass implements `statement`, recognises its
own node types, pulls the parts out of them, and calls the shape that matches. No method here reads a
field a language declares, which is what lets two languages whose `for` loops share nothing but their
meaning share this code.

The shapes are parameterised where languages genuinely differ rather than being duplicated: what an
arm of a `dispatch` reaches when control runs off its end is an `ArmFlow`, and each of its members
is some language's answer to the same construct.
"""
from __future__ import annotations

import enum

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from refinery.lib.scripts import Node


class CfgEdge(enum.Flag):
    """
    What an edge says about the run that takes it, beyond the two nodes it joins.

    `NORMAL` — control simply passed on, the source having done whatever it does.

    `ERROR_CARRYING` — the source threw and the error travels here: the edge into the handler that
    is offered it, and onward from a handler set that may decline. A definition the source makes is
    not guaranteed to have happened along one of these.

    `RESUMPTION_HUB`, `RESUMPTION_FORWARD` — the two projections of a handler that *resumes* the
    block it guards rather than ending it. The engine carries on at the statement after the one that
    threw, and no graph knows which statement that was, so the shape is carried twice: the hub is
    the over-approximation, joining every resumption point to every statement of the block, and the
    forward edges are the precise half, joining each statement to the one it would resume at. Both
    stand for the same runs. An analysis asking whether *any* resumption path exists reads the hub;
    one asking what a point reaches *going forward* reads the forward edges and skips the hub — see
    `reachable_forward_from_any`.

    The distinction the resumption kinds force is that `ERROR_CARRYING` conflates two bits which
    coincide nowhere else: *taken only when the source threw*, and *the error object travels here*.
    A resumption edge is the first that is the former and not the latter — the handler swallowed the
    error — so a consumer reasoning about whether the source completed must read `RAISE_TAKEN`, and
    only one reasoning about where an error went may read `ERROR_CARRYING` alone.
    """
    NORMAL = 0
    ERROR_CARRYING = enum.auto()
    RESUMPTION_HUB = enum.auto()
    RESUMPTION_FORWARD = enum.auto()


#: The kinds an edge is taken along only on runs where its source threw rather than completing. A
#: store the source makes may not have happened at the other end of one, whichever of the three it
#: is, which is the one question every flow-sensitive consumer asks of the kind.
RAISE_TAKEN = CfgEdge.ERROR_CARRYING | CfgEdge.RESUMPTION_HUB | CfgEdge.RESUMPTION_FORWARD


class ArmFlow(enum.Enum):
    """
    What an arm of a multi-way branch may reach when control runs off its end.

    The three members are three languages' answers to the same construct, and reading one as another
    either invents paths that cannot be taken or drops paths that can — the second of which is the
    direction that lets an analysis call live code unreachable.

    `EXCLUSIVE` — at most one arm ever runs, so an arm's exits leave the construct.
    `SEQUENTIAL` — an arm that runs off its end enters the *next* arm's body unconditionally, which
    is C-style fallthrough and what JavaScript's `switch` does.
    `CUMULATIVE` — every arm is tested in turn and every matching one runs, so an arm's exits may
    reach *any* later arm, not only the next. PowerShell's `switch` is this: it does not fall
    through, it keeps matching, and a `break` is what stops it.
    """
    EXCLUSIVE  = enum.auto()  # noqa
    SEQUENTIAL = enum.auto()  # noqa
    CUMULATIVE = enum.auto()  # noqa


@dataclass(eq=False, repr=False)
class CfgNode:
    """
    One vertex of a control-flow graph. `element` is the AST node it stands for — a statement, or a
    loop-head expression whose reads and writes occur at this point — or `None` for the synthetic
    entry and exit. `successors` lists the nodes control may pass to next.

    `eq=False` is load bearing. Every map in every layer above is keyed by `id(node)`, and two
    structurally equal statements are two distinct points in the program.

    `repr=False` is too. A generated repr expands `successors` and `predecessors`, and the guard
    against recursion only covers the node currently being formatted, so a graph re-expands at every
    join: a chain of five hundred nodes — a script of five hundred statements — exhausts the
    interpreter's stack, and a handful of branches produces megabytes. A debugger, a failing
    assertion, or `pytest --showlocals` would print one.
    """
    element: Node | None
    successors: list[CfgNode] = field(default_factory=list)
    predecessors: list[CfgNode] = field(default_factory=list)


def reachable_from_any(sources: Iterable[CfgNode]) -> frozenset[int]:
    """
    The ids of the nodes forward-reachable from *any* of *sources*, each source included — the
    multi-source flood a taint analysis wants from a set of origins. A raw walk over successor
    edges, which is why it lives with the graph rather than on any model built over it.

    Walked as one depth-first sweep seeded with every source rather than a union of per-source
    walks: the union grows the same answer at the cost of one full walk per source, where a single
    sweep visits each node once however many sources reach it. The result is a fresh set the caller
    owns, so it may intersect or discard it in place.
    """
    seen: set[int] = set()
    stack: list[CfgNode] = []
    for source in sources:
        if id(source) not in seen:
            seen.add(id(source))
            stack.append(source)
    while stack:
        node = stack.pop()
        for successor in node.successors:
            if id(successor) not in seen:
                seen.add(id(successor))
                stack.append(successor)
    return frozenset(seen)


class ControlFlowGraph:
    """
    The control-flow graph of one function or script body. `entry` and `exit` are synthetic; every
    other node wraps an AST element reachable through `node_of`.
    """

    def __init__(self, owner: Node):
        self.owner = owner
        self.entry = CfgNode(None)
        self.exit = CfgNode(None)
        self.nodes: list[CfgNode] = [self.entry, self.exit]
        self._node_of: dict[int, CfgNode] = {}
        self._edge_kinds: dict[tuple[int, int], CfgEdge] = {}
        self._hub_bound: set[int] = set()
        self._fallback: dict[int, CfgNode] = {}

    def node_of(self, element: Node) -> CfgNode | None:
        """
        The graph node standing for *element*, or `None` if *element* is not part of this body, or is
        a node the graph does not represent on its own such as a plain expression inside a statement.
        """
        return self._node_of.get(id(element))

    def fallback_of(self, handler: CfgNode) -> CfgNode | None:
        """
        Where a throw offered to *handler* goes if *handler* does not take it, or `None` when
        *handler* is not a handler entry of this graph.

        This is recorded rather than read off the edges because the two are not the same claim. The
        edge is drawn only where some run may decline — a clause with a type filter, a `trap` set
        that may fail to match — and a handler certain to take the throw has none, which is what
        makes it shield whatever guards the construct. The fact holds either way, and it is what
        answers the *counterfactual*: not where the throw goes, but where it would go if this
        handler were not written at all, which is the question asked before one is deleted.
        """
        return self._fallback.get(id(handler))

    def edge_kind(self, source: CfgNode, target: CfgNode) -> CfgEdge:
        """
        What the edge from *source* to *target* says about the run that takes it — see `CfgEdge`.
        An edge the builder recorded nothing for is `CfgEdge.NORMAL`, which is what an unrelated
        pair of nodes reads as too.
        """
        return self._edge_kinds.get((id(source), id(target)), CfgEdge.NORMAL)

    def is_exceptional(self, source: CfgNode, target: CfgNode) -> bool:
        """
        Whether the error travels along the edge from *source* to *target*: the edge into a handler
        offered the throw, or outward from a set that declined it. This is the question `faults`
        asks — where an error goes — and it is *narrower* than `raise_taken`, because a handler that
        resumes swallows the error and carries on along an edge no error travels.
        """
        return bool(self.edge_kind(source, target) & CfgEdge.ERROR_CARRYING)

    def raise_taken(self, source: CfgNode, target: CfgNode) -> bool:
        """
        Whether the edge from *source* to *target* is taken only when *source* throws rather than
        completing normally. A definition *source* makes is not guaranteed to have happened along
        such an edge, so a flow-sensitive analysis must not treat it as a kill there.

        This is the question about *completion*, and it is the one nearly every consumer means. A
        resumption edge answers it although no error travels along it: the statement that resumed
        the block is precisely the one that did not finish.
        """
        return bool(self.edge_kind(source, target) & RAISE_TAKEN)

    def is_hub_bound(self, node: CfgNode) -> bool:
        """
        Whether *node* sits where the precise, forward-only projection of resumption does not reach
        it — inside a handler body that resumes the block around it. The block's forward edges join
        each *guarded* statement to the one it resumes at; a statement of the handler itself is on
        neither end of one, and everything it may reach afterwards hangs off the hub.

        A forward-only walk seeded at such a node would therefore stop dead where the real run
        carries on, which for a flood is the unsound direction. `reachable_forward_from_any` reads
        this and falls back to the hub for those sources.
        """
        return id(node) in self._hub_bound


def reachable_forward_from_any(
    graph: ControlFlowGraph,
    sources: Iterable[CfgNode],
) -> frozenset[int]:
    """
    `reachable_from_any` restricted to the paths a run takes *going forward* from each source: the
    resumption hub is not followed, so a source does not reach the statements standing before it in
    its own block merely because some later statement of that block may throw and resume.

    The hub claims every resumption point reaches every statement of the guarded block, earlier ones
    included. That is the reading a *may* query wants and the ruin of a flood: one leak anywhere in
    a trap-guarded script poisons the whole of it. The `CfgEdge.RESUMPTION_FORWARD` edges carry the
    same runs precisely — each statement joined to the one control would resume at — so declining
    the hub drops no path that goes forward.

    A source the graph reports `is_hub_bound` for is flooded with the hub followed instead, because
    the forward edges carry nothing for such a source and the precise walk would stop at the
    resumption point: an opener written inside a `trap` body would then vouch for the very
    statements the handler resumes into. Its flood is the over-approximate one, which is the
    fail-closed direction.
    """
    seen: set[int] = set()
    precise: list[CfgNode] = []
    coarse: list[CfgNode] = []
    for source in sources:
        (coarse if graph.is_hub_bound(source) else precise).append(source)
    if coarse:
        seen.update(reachable_from_any(coarse))
    stack: list[CfgNode] = []
    for source in precise:
        if id(source) not in seen:
            seen.add(id(source))
            stack.append(source)
    while stack:
        node = stack.pop()
        for successor in node.successors:
            if id(successor) in seen:
                continue
            if graph.edge_kind(node, successor) is CfgEdge.RESUMPTION_HUB:
                continue
            seen.add(id(successor))
            stack.append(successor)
    return frozenset(seen)


class ElementLocator:
    """
    Locates an AST node among the per-body control-flow graphs of one script. Built once from the
    graph set, it maps an element to the graph and node that evaluate it — directly for an element a
    graph node stands for (`node_of`), or by climbing to the enclosing statement for one nested
    inside an expression (`locate`). Every flow-sensitive layer built on the graphs shares it, so the
    AST-to-graph mapping and its parent-climb live in one place.
    """

    def __init__(self, graphs: dict[int, ControlFlowGraph]):
        self._element_graph: dict[int, ControlFlowGraph] = {}
        self._owners = {id(graph.owner) for graph in graphs.values()}
        for graph in graphs.values():
            for node in graph.nodes:
                if node.element is not None:
                    self._element_graph[id(node.element)] = graph

    def node_of(self, element: Node) -> CfgNode | None:
        """
        The control-flow node standing for *element* in whichever graph owns it, or `None` when
        *element* is not itself a node the graphs represent.
        """
        graph = self._element_graph.get(id(element))
        return graph.node_of(element) if graph is not None else None

    def locate(self, element: Node) -> tuple[ControlFlowGraph, CfgNode] | None:
        """
        The graph and node that evaluate *element*, climbing out of any expression it is nested in to
        the enclosing statement or loop head, or `None` when it has no enclosing graph node.

        The climb stops at the body *element* is written in rather than continuing into the body
        around it. Something inside a body that no node of that body's graph stands for — the default
        of a parameter, an attribute on the body itself — is evaluated when that body is invoked, and
        the enclosing body's statement that mentions it is not that point. Answering with that
        statement orders the element against code the invocation may never run beside, which is the
        false claim the per-body split exists to avoid; `None` says the graphs do not place it, and a
        caller reads that as unknown.

        The body *element* is itself is not its own boundary: a block is a value written at a point
        in the body around it, so locating one climbs out to the statement that mentions it.
        """
        cursor: Node | None = element
        while cursor is not None:
            graph = self._element_graph.get(id(cursor))
            if graph is not None:
                node = graph.node_of(cursor)
                if node is not None:
                    return graph, node
            if cursor is not element and id(cursor) in self._owners:
                return None
            cursor = cursor.parent
        return None


@dataclass
class _Target:
    """
    A jump destination active while a breakable or continuable construct is being built. `breaks`
    collects the nodes that leave it early, wired to whatever follows once that is known;
    `continue_to` is the node a back-jump reaches, or `None` for a construct only a break can leave.

    `continues` collects the back-jumps of a loop whose `continue_to` is not yet known — a counted
    loop with neither a test nor an update jumps back to its body's own entry, and that entry only
    exists once the body has been built. Wiring them to the exit instead would drop the back-edge
    the loop is made of.
    """
    label: str | None
    breaks: list[CfgNode]
    continue_to: CfgNode | None
    is_continuable: bool
    is_breakable: bool
    continues: list[CfgNode] = field(default_factory=list)


def distinct(nodes: Iterable[CfgNode]) -> list[CfgNode]:
    """
    *nodes* with every repetition dropped, compared by identity and in first-seen order.

    A frontier carried from one arm of a construct into the next must not accumulate duplicates. An
    arm that creates no node of its own hands its incoming frontier straight back, so concatenating
    the two doubles the list on every such arm: a `switch` with thirty empty clause bodies would
    build 2**30 edges, every one of them a repetition of an edge already there.
    """
    seen: set[int] = set()
    result: list[CfgNode] = []
    for node in nodes:
        if id(node) in seen:
            continue
        seen.add(id(node))
        result.append(node)
    return result


class CfgBuilder:
    """
    Single-pass construction of one `ControlFlowGraph` by structural recursion over a body, threading
    a *frontier* — the set of nodes from which normal control currently falls through — into each
    statement and out the other side.

    A language subclasses this and implements `statement` and `body_blocks`. Everything else is
    shared, and the shape methods below take the parts of a construct rather than the construct, so
    that no code here has to know what a language calls the pieces of its `for` loop.
    """

    def __init__(self, owner: Node):
        self.cfg = ControlFlowGraph(owner)
        self._handlers: list[CfgNode] = []
        self._targets: list[_Target] = []
        self._pending_label: str | None = None

    def build(self) -> ControlFlowGraph:
        frontier = [self.cfg.entry]
        for statements in self.body_blocks(self.cfg.owner):
            frontier = self.block(statements, frontier)
        self.link(frontier, self.cfg.exit)
        return self.cfg

    def body_blocks(self, owner: Node) -> Sequence[Sequence[Node]]:
        """
        The statement blocks *owner* runs, in the order it runs them. A language whose body may be a
        single unbraced statement resolves that here, and one that splits a body across several
        named blocks reports them apart rather than joined, because a construct scoped to a block
        cannot be modelled from a body that has forgotten where its blocks ended.
        """
        raise NotImplementedError

    def block(self, statements: Sequence[Node], frontier: list[CfgNode]) -> list[CfgNode]:
        """
        One statement block. The statements in order, which is all a block is to a language whose
        blocks carry nothing of their own; a language whose blocks scope a handler installs it here
        and calls this for every block it builds.
        """
        return self.sequence(statements, frontier)

    def statement(self, statement: Node, frontier: list[CfgNode]) -> list[CfgNode]:
        """
        Add *statement* to the graph and report the frontier that follows it. A language recognises
        its own node types here and calls the shape method that matches; anything it does not
        recognise goes to `opaque`, which is the conservative answer.
        """
        raise NotImplementedError

    def node(self, element: Node) -> CfgNode:
        """
        A graph node for *element*, joined to the innermost active handler if one is open.

        The handler edge is added here rather than at each site that creates a node, because *any*
        statement inside a guarded block may throw and the whole point of the edge is that it does
        not depend on which statement it is.
        """
        node = self.detached_node(element)
        if self._handlers:
            self.exceptional_edge(node, self._handlers[-1])
        return node

    def synthetic_node(self) -> CfgNode:
        """
        A graph node standing for no element: a join or fan-out point the graph needs and the
        program does not name. It locates into no query and generates no dataflow fact, which is
        what lets one stand in for a set of edges between real nodes without an analysis reading it
        as a statement.
        """
        node = CfgNode(None)
        self.cfg.nodes.append(node)
        return node

    def detached_node(self, element: Node) -> CfgNode:
        """
        A graph node for *element* that the enclosing handler does not guard.

        A handler's own entry is one: it stands for the point at which a throw is *offered* to a
        clause, which is not a point inside the region that clause guards. The edge outward from
        there says the clause did not take the throw, and whether any run says that is a property of
        the whole handler set, which `guarded` reads once.
        """
        node = CfgNode(element)
        self.cfg.nodes.append(node)
        self.cfg._node_of[id(element)] = node
        return node

    @staticmethod
    def add_edge(source: CfgNode, target: CfgNode) -> None:
        source.successors.append(target)
        target.predecessors.append(source)

    def kind_edge(self, source: CfgNode, target: CfgNode, kind: CfgEdge) -> None:
        """
        An edge carrying *kind*, which every edge but a plain one is drawn through so that the
        adjacency lists and the kind map are never written apart.
        """
        self.add_edge(source, target)
        key = (id(source), id(target))
        self.cfg._edge_kinds[key] = self.cfg._edge_kinds.get(key, CfgEdge.NORMAL) | kind

    def exceptional_edge(self, source: CfgNode, target: CfgNode) -> None:
        self.kind_edge(source, target, CfgEdge.ERROR_CARRYING)

    def resumption_hub_edge(self, source: CfgNode, target: CfgNode) -> None:
        """
        An edge of the over-approximate half of a resuming handler — see `CfgEdge`.
        """
        self.kind_edge(source, target, CfgEdge.RESUMPTION_HUB)

    def resumption_forward_edge(self, source: CfgNode, target: CfgNode) -> None:
        """
        An edge of the precise, forward-only half of a resuming handler — see `CfgEdge`.
        """
        self.kind_edge(source, target, CfgEdge.RESUMPTION_FORWARD)

    def mark_hub_bound(self, nodes: Iterable[CfgNode]) -> None:
        """
        Record that the forward-only projection of resumption does not reach *nodes* — see
        `ControlFlowGraph.is_hub_bound`.
        """
        self.cfg._hub_bound.update(id(node) for node in nodes)

    def close_handler_set(self, entries: Sequence[CfgNode], *, escapes: bool) -> None:
        """
        Finish a chain of handler entries: record where a throw goes when none of them takes it,
        and draw the edge there when some run may leave one of them to.

        Called once the set's bodies are built and the set itself is off the handler stack, so that
        `unwinding` names what guards the *construct* rather than the set being closed.

        A member that is not the last one falls back to the member after it and not to what guards
        the construct, because the engine consults the set in order: a throw the first clause
        declines is offered to the second, which is also where it would go if the first were not
        written. Recording the construct's fallback for every member reads one clause's
        counterfactual off another's, and a set whose later member acts then looks deletable.
        """
        outward = self.unwinding()
        for index, entry in enumerate(entries):
            following = entries[index + 1] if index + 1 < len(entries) else outward
            self.cfg._fallback[id(entry)] = following
        if escapes and entries:
            self.exceptional_edge(entries[-1], outward)

    def unwinding(self) -> CfgNode:
        """
        Where a throw goes when nothing open here takes it: the innermost active handler, or the
        body's exit, which is how the graph says the throw leaves the body.
        """
        return self._handlers[-1] if self._handlers else self.cfg.exit

    def link(self, frontier: Iterable[CfgNode], target: CfgNode) -> None:
        for node in frontier:
            self.add_edge(node, target)

    def sequence(self, statements: Sequence[Node], frontier: list[CfgNode]) -> list[CfgNode]:
        for statement in statements:
            frontier = self.statement(statement, frontier)
        return frontier

    def opaque(self, element: Node, frontier: list[CfgNode]) -> list[CfgNode]:
        """
        A statement whose internal control flow this does not model: one node, entered from the
        frontier and falling through. The default for anything a language does not recognise.
        """
        node = self.node(element)
        self.link(frontier, node)
        return [node]

    def _body(self, body: Node | None, frontier: list[CfgNode]) -> list[CfgNode]:
        return self.statement(body, list(frontier)) if body is not None else list(frontier)

    def _branch(self, body: Node | None, head: CfgNode) -> list[CfgNode]:
        return self._body(body, [head])

    def _capture_body(
        self, body: Node | None, frontier: list[CfgNode],
    ) -> tuple[CfgNode | None, list[CfgNode]]:
        """
        Build *body* and return its entry node — the node control reaches first — alongside its exit
        frontier. Used where a back-edge must target the body's own entry, which the plain frontier
        threading does not expose.

        The entry is the first successor the incoming *frontier* gains while *body* is built, not the
        first node created. A body that opens with a guarded block builds its handler or finalizer
        node before any guarded statement, so creation order would return that handler — a node with
        no edge back into the body — and the loop's back-edge would be wired to it, hiding the real
        body head from a backward reachability walk. The frontier instead links to the first guarded
        statement, which is the node control actually enters.
        """
        before = [(node, len(node.successors)) for node in frontier]
        exits = self._body(body, frontier)
        for node, count in before:
            if len(node.successors) > count:
                return node.successors[count], exits
        return None, exits

    def park_label(self, label: str | None) -> None:
        """
        Park *label* for the construct about to be built, for `take_label` to consume.

        `labelled` calls this for a language whose label is a statement wrapping the construct. A
        language whose label is a field *of* the construct calls it directly, because there is no
        wrapping statement to recognise — and a construct built without it carries no label, so
        every jump naming it misses and leaves the body instead.
        """
        self._pending_label = label

    def take_label(self) -> str | None:
        """
        The label parked for the construct about to be built, consumed once.
        """
        label = self._pending_label
        self._pending_label = None
        return label

    def branch_on(
        self,
        element: Node,
        arms: Sequence[Node | None],
        frontier: list[CfgNode],
        *,
        exhaustive: bool = False,
    ) -> list[CfgNode]:
        """
        A conditional: one head node the frontier enters, and one arm per branch. When the arms do
        not cover every case — an `if` with no `else` — the head itself is an exit, because control
        may pass the whole construct without entering any arm. `exhaustive` says the arms do cover
        it, which is what an `if`/`else` pair reports.
        """
        head = self.node(element)
        self.link(frontier, head)
        exits: list[CfgNode] = []
        for arm in arms:
            exits += self._branch(arm, head)
        if not exhaustive:
            exits.append(head)
        return exits

    def loop_head_tested(
        self, element: Node, body: Node | None, frontier: list[CfgNode],
    ) -> list[CfgNode]:
        """
        A loop whose condition is evaluated before the body, so the head is both the entry and an
        exit: `while`, and every `foreach` whose iteration may run zero times.
        """
        head = self.node(element)
        self.link(frontier, head)
        target = _Target(self.take_label(), [], head, is_continuable=True, is_breakable=True)
        self._targets.append(target)
        body_exits = self._branch(body, head)
        self._targets.pop()
        self.link(body_exits, head)
        return [head] + target.breaks

    def loop_tail_tested(
        self, element: Node, body: Node | None, frontier: list[CfgNode],
    ) -> list[CfgNode]:
        """
        A loop whose condition is evaluated after the body, so the body always runs once and the
        back-edge targets the body's own entry rather than the test.
        """
        test = self.node(element)
        target = _Target(self.take_label(), [], test, is_continuable=True, is_breakable=True)
        self._targets.append(target)
        entry, body_exits = self._capture_body(body, frontier)
        self._targets.pop()
        self.link(body_exits, test)
        self.add_edge(test, entry if entry is not None else test)
        return [test] + target.breaks

    def loop_counted(
        self,
        initializer: Node | None,
        test: Node | None,
        update: Node | None,
        body: Node | None,
        frontier: list[CfgNode],
    ) -> list[CfgNode]:
        """
        A loop with a separate initializer, test and update, each evaluated at its own point and so
        each given its own node. Any of the three may be absent; a loop with no test has no exit
        other than the jumps out of it, and its back-edge targets the body entry.
        """
        label = self.take_label()
        if initializer is not None:
            start = self.node(initializer)
            self.link(frontier, start)
            frontier = [start]
        head = self.node(test) if test is not None else None
        if head is not None:
            self.link(frontier, head)
            body_frontier: list[CfgNode] = [head]
        else:
            body_frontier = list(frontier)
        step = self.node(update) if update is not None else None
        target = _Target(label, [], step or head, is_continuable=True, is_breakable=True)
        self._targets.append(target)
        entry, body_exits = self._capture_body(body, body_frontier)
        self._targets.pop()
        latch = body_exits
        if step is not None:
            self.link(body_exits, step)
            latch = [step]
        back_to = head if head is not None else entry
        if back_to is not None:
            self.link(latch, back_to)
        self.link(target.continues, back_to if back_to is not None else self.cfg.exit)
        exits = list(target.breaks)
        if head is not None:
            exits.append(head)
        return exits

    def branch_chain(
        self,
        clauses: Sequence[tuple[Node, Node | None]],
        otherwise: Node | None,
        frontier: list[CfgNode],
    ) -> list[CfgNode]:
        """
        A chain of guarded arms, each tested only when every earlier test failed: `if`/`elseif`/`else`
        where the whole chain is one node rather than a nest of two-armed conditionals.

        Each test gets its own node, because the tests run at different points and an analysis that
        collapsed them could not order two of them. A test's node flows into its own arm and on to
        the next test; the last test flows into `otherwise` when there is one, and out of the
        construct when there is not.
        """
        exits: list[CfgNode] = []
        current = frontier
        for test, body in clauses:
            head = self.node(test)
            self.link(current, head)
            exits += self._branch(body, head)
            current = [head]
        if otherwise is not None:
            return distinct(exits + self._body(otherwise, current))
        return distinct(exits + current)

    def dispatch(
        self,
        element: Node,
        arms: Sequence[Sequence[Node]],
        frontier: list[CfgNode],
        *,
        arm_flow: ArmFlow,
        exhaustive: bool,
        iterated: bool = False,
    ) -> list[CfgNode]:
        """
        A multi-way branch: one head the frontier enters and one arm per clause, each arm a statement
        sequence the head may jump into. `arm_flow` says what an arm reaches when it runs off its
        end; see `ArmFlow`, which is where the languages differ.

        `exhaustive` says some arm always runs — a default clause — so the head is not itself an
        exit.

        `iterated` says the construct runs its arms once per element of an input, so it is a loop:
        a back-jump inside an arm re-enters the head rather than resolving to an enclosing loop, and
        an arm that simply runs off its end re-enters it too, for the next element. Without that
        second edge the arms lie on no cycle and an analysis reads a store inside one as happening
        once, which is what lets a self-referential assignment be folded to its first value.
        PowerShell's `switch` is this; JavaScript's is not, and reading one as the other both invents
        a back-edge to a loop the jump never reaches and drops the one it does.

        A jump *out* takes no back-edge: it leaves the construct rather than advancing it, which is
        the whole difference between the two spellings.
        """
        head = self.node(element)
        self.link(frontier, head)
        target = _Target(
            self.take_label(),
            [],
            head if iterated else None,
            is_continuable=iterated,
            is_breakable=True,
        )
        self._targets.append(target)
        carried: list[CfgNode] = []
        completed: list[CfgNode] = []
        for arm in arms:
            reached = self.sequence(list(arm), distinct([head, *carried]))
            if arm_flow is ArmFlow.SEQUENTIAL:
                carried = reached
            elif arm_flow is ArmFlow.CUMULATIVE:
                carried = distinct([*carried, *reached])
            else:
                completed += reached
        self._targets.pop()
        completed = distinct(completed + carried)
        if iterated:
            self.link(completed, head)
        exits = completed + target.breaks
        if not exhaustive:
            exits.append(head)
        return distinct(exits)

    def guarded(
        self,
        block: Node | None,
        handlers: Sequence[tuple[Node, Node | None]],
        finalizer: Node | None,
        finalizer_body: Sequence[Node],
        frontier: list[CfgNode],
        *,
        escapes: bool,
    ) -> list[CfgNode]:
        """
        A guarded block with any number of handlers and an optional finalizer.

        The handler nodes are created *before* the guarded block is built and the first is pushed on
        the handler stack, so that `node` joins every statement created inside the block to it. That
        ordering is the whole mechanism: it is why no statement inside the block has to know it is
        guarded.

        Several handlers are chained from the first, because which one runs depends on the type of
        the exception and none of them is guaranteed — a language with one handler passes a
        one-element sequence and the chain degenerates. The chain edge is *exceptional*: control
        takes it exactly when the earlier handler did not match, so nothing that handler's node
        stands for has run and its kill must not apply along it.

        `escapes` says the set of handlers may fail to take a throw the block makes, which is what a
        language whose clauses carry a type filter reports and what one whose single clause takes
        everything denies. It decides one edge — from the last handler to whatever guards this
        construct, or to the body exit when nothing does, which is how the graph says a throw leaves
        the body with no handler having taken it. Denying the edge is the difference between a
        clause that swallows and one that passes the throw on, so it is asked of every language
        rather than defaulted: the safe answer invents a path the program cannot take, and the
        other drops one it can.

        The finalizer is entered from the block's normal exits and from every handler's, and itself
        carries an exceptional edge outward, because a finalizer runs on the exceptional path too and
        control leaves the construct from it either way.
        """
        handlers = list(handlers)
        entries = [self.detached_node(handler) for handler, _ in handlers]
        finalizer_entry = self.detached_node(finalizer) if finalizer is not None else None
        guard = entries[0] if entries else finalizer_entry
        if guard is not None:
            self._handlers.append(guard)
        block_exits = self.statement(block, frontier) if block is not None else list(frontier)
        if guard is not None:
            self._handlers.pop()
        normal_exits = list(block_exits)
        for index, ((_, body), entry) in enumerate(zip(handlers, entries)):
            if index:
                self.exceptional_edge(entries[index - 1], entry)
            normal_exits += (
                self.statement(body, [entry]) if body is not None else [entry])
        self.close_handler_set(entries, escapes=escapes)
        if finalizer_entry is not None and finalizer is not None:
            self.link(normal_exits, finalizer_entry)
            final_exits = self.sequence(list(finalizer_body), [finalizer_entry])
            self.exceptional_edge(finalizer_entry, self.unwinding())
            return final_exits
        return normal_exits

    def labelled(
        self,
        label: str | None,
        body: Node | None,
        frontier: list[CfgNode],
        *,
        binds_to_body: bool,
    ) -> list[CfgNode]:
        """
        A labelled statement. When the label names a construct a jump can target directly —
        `binds_to_body` — it is parked for that construct to consume through `take_label`; otherwise
        the label names this statement itself and only a break can leave it.
        """
        if binds_to_body:
            self.park_label(label)
            return self.statement(body, frontier) if body is not None else list(frontier)
        target = _Target(label, [], None, is_continuable=False, is_breakable=False)
        self._targets.append(target)
        exits = self.statement(body, frontier) if body is not None else list(frontier)
        self._targets.pop()
        return exits + target.breaks

    def jump_out(
        self, element: Node, label: str | None, frontier: list[CfgNode],
    ) -> list[CfgNode]:
        """
        A jump that leaves the construct it names, or the innermost breakable one when unlabelled. A
        jump naming nothing this body holds leaves the body, which is the conservative reading.
        """
        node = self.node(element)
        self.link(frontier, node)
        target = self._break_target(label)
        if target is not None:
            target.breaks.append(node)
        else:
            self.add_edge(node, self.cfg.exit)
        return []

    def jump_back(
        self, element: Node, label: str | None, frontier: list[CfgNode],
    ) -> list[CfgNode]:
        """
        A jump to the next iteration of the loop it names, or of the innermost loop when unlabelled.
        """
        node = self.node(element)
        self.link(frontier, node)
        target = self._continue_target(label)
        if target is None:
            self.add_edge(node, self.cfg.exit)
        elif target.continue_to is not None:
            self.add_edge(node, target.continue_to)
        else:
            target.continues.append(node)
        return []

    def terminate(
        self, element: Node, frontier: list[CfgNode], *, exceptional: bool,
    ) -> list[CfgNode]:
        """
        A statement after which control does not continue in this body: a return, or a throw.

        A throw is `exceptional`, so it reaches the innermost open handler rather than the exit, and
        only reaches the exit when no handler is open. A return leaves the body outright — the
        finalizer question a return inside a guarded block raises is one this graph deliberately
        answers by the conservative edge rather than by modelling the unwind.
        """
        node = self.node(element)
        self.link(frontier, node)
        if exceptional:
            self.exceptional_edge(node, self.unwinding())
        else:
            self.add_edge(node, self.cfg.exit)
        return []

    def has_continue_target(self, label: str | None) -> bool:
        """
        Whether a back-jump naming *label* — or an unlabelled one — resolves to a construct
        currently being built. A language in which `continue` means something other than a back-jump
        when no such construct is open asks this to tell the two spellings apart.
        """
        return self._continue_target(label) is not None

    def has_break_target(self, label: str | None) -> bool:
        """
        Whether a jump out naming *label* — or an unlabelled one — resolves to a construct currently
        being built, the counterpart of `has_continue_target` for the language in which `break`
        means something other than leaving a construct when no construct is open to leave.
        """
        return self._break_target(label) is not None

    def _break_target(self, label: str | None) -> _Target | None:
        for target in reversed(self._targets):
            if label is None:
                if target.is_breakable:
                    return target
            elif target.label == label:
                return target
        return None

    def _continue_target(self, label: str | None) -> _Target | None:
        for target in reversed(self._targets):
            if not target.is_continuable:
                continue
            if label is None or target.label == label:
                return target
        return None


class ControlFlowModel:
    """
    The per-body control-flow graphs of one script, paired with the `ElementLocator` that maps any
    AST node to the graph node evaluating it. Built once over the script root — the graphs are purely
    syntactic and need no semantic model — and shared by every solver layered on it, which would
    otherwise each rebuild the whole set.
    """

    def __init__(self, graphs: dict[int, ControlFlowGraph]):
        self.graphs = graphs
        self._locator = ElementLocator(graphs)

    def graph_of(self, owner: Node) -> ControlFlowGraph | None:
        """
        The control-flow graph owned by *owner* — a function node or the script root — or `None` when
        it owns none.
        """
        return self.graphs.get(id(owner))

    def node_of(self, element: Node) -> CfgNode | None:
        """
        The control-flow node standing for *element*, or `None` when the graphs do not represent it
        directly. Delegates to the shared `ElementLocator`.
        """
        return self._locator.node_of(element)

    def locate(self, element: Node) -> tuple[ControlFlowGraph, CfgNode] | None:
        """
        The graph and node that evaluate *element*, climbing out of any enclosing expression, or
        `None` when it has no enclosing graph node. Delegates to the shared `ElementLocator`.
        """
        return self._locator.locate(element)


def build_control_flow(
    root: Node,
    builder: type[CfgBuilder],
    function_nodes: tuple[type, ...],
) -> dict[int, ControlFlowGraph]:
    """
    Build one control-flow graph per function and one for the script itself, keyed by the owner
    node's identity. The graphs are independent: a nested function appears in its parent's graph only
    as the statement that defines it, never as descended-into control flow.
    """
    graphs: dict[int, ControlFlowGraph] = {id(root): builder(root).build()}
    for node in root.walk():
        if isinstance(node, function_nodes):
            graphs[id(node)] = builder(node).build()
    return graphs
