"""
The shared control-flow substrate: one graph per body, built by structural recursion over a
language's own node types. A language contributes only a dispatch — see
`refinery.lib.scripts.ps1.analysis.cfg`.
"""
from __future__ import annotations

import enum

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import AbstractSet as Set, Iterable, Iterator, Sequence

from refinery.lib.scripts import Node


class CfgEdge(enum.Flag):
    """
    What an edge says about the run that takes it, beyond the two nodes it joins.

    `NORMAL` — control passed on. `ERROR_CARRYING` — the source threw and the error travels here.
    `RESUMPTION_HUB` and `RESUMPTION_FORWARD` — the two projections of a handler that *resumes* the
    block it guards: no graph knows which statement threw, so the hub over-approximates by joining
    every resumption point to every statement of the block, and the forward edges join each
    statement to the one it would resume at. See `Projection`.

    A consumer asking whether the source *completed* reads `RAISE_TAKEN`; only one asking where an
    error went may read `ERROR_CARRYING` alone. The edges *out* of a hub carry the kind and the edge
    *into* one does not: its source is a handler body that ran to its end.
    """
    NORMAL = 0
    ERROR_CARRYING = enum.auto()
    RESUMPTION_HUB = enum.auto()
    RESUMPTION_FORWARD = enum.auto()


#: The kinds an edge is taken along only on runs where its source threw rather than completing, so a
#: store the source makes may not have happened at the other end of one.
RAISE_TAKEN = CfgEdge.ERROR_CARRYING | CfgEdge.RESUMPTION_HUB | CfgEdge.RESUMPTION_FORWARD


class Projection(enum.Enum):
    """
    Which reading of a resuming handler a walk is asked for.

    `MAY` — every path the graph draws, the hub among them, so a resumption reaches the statements
    written above it too. `FORWARD` — the hub is declined, so a resumption reaches only the
    statement control resumes at and what follows.

    Neither is the safe one, which is why every walk names one instead of taking a default: `MAY`
    claims runs that cannot happen, and `FORWARD` drops the fact that a handler body carries on
    somewhere the forward edges do not name. The hub is declined as a *node*, in both directions.
    """
    MAY = enum.auto()
    FORWARD = enum.auto()

    def successors(self, node: CfgNode) -> list[CfgNode]:
        if self is Projection.MAY:
            return node.successors
        if node.is_resumption_hub:
            return []
        return [target for target in node.successors if not target.is_resumption_hub]

    def predecessors(self, node: CfgNode) -> list[CfgNode]:
        if self is Projection.MAY:
            return node.predecessors
        if node.is_resumption_hub:
            return []
        return [source for source in node.predecessors if not source.is_resumption_hub]

    @property
    def declines_the_hub(self) -> bool:
        return self is Projection.FORWARD


class ArmFlow(enum.Enum):
    """
    What an arm of a multi-way branch may reach when control runs off its end.

    `EXCLUSIVE` — at most one arm runs, so an arm's exits leave the construct. `SEQUENTIAL` —
    C-style fallthrough into the next arm. `CUMULATIVE` — every arm is tested and every matching one
    runs, so an arm's exits may reach any *later* arm, not only the next.
    """
    EXCLUSIVE  = enum.auto()  # noqa
    SEQUENTIAL = enum.auto()  # noqa
    CUMULATIVE = enum.auto()  # noqa


@dataclass(eq=False, repr=False)
class CfgNode:
    """
    One vertex of a control-flow graph. `element` is the AST node it stands for, or `None` for a
    synthetic node. `graph` is the body it belongs to, which is what makes the two flags answerable
    of a node alone: `is_resumption_hub` marks the synthetic fan-out of a resuming handler, and
    `is_hub_bound` marks a node inside such a handler's body, which the forward projection does not
    reach and `reachable_forward_from_any` therefore walks under `MAY`.

    `eq=False` because every map above is keyed by `id(node)`. `repr=False` because a generated one
    re-expands the adjacency lists at every join and exhausts the stack on a few hundred statements.
    """
    graph: ControlFlowGraph
    element: Node | None
    successors: list[CfgNode] = field(default_factory=list)
    predecessors: list[CfgNode] = field(default_factory=list)
    is_resumption_hub: bool = False
    is_hub_bound: bool = False


def flood(
    sources: Iterable[CfgNode], *, forward: bool, projection: Projection,
) -> set[int]:
    """
    The nodes reachable from *sources*, in the direction *forward* names and under *projection*.

    A source the projection cannot walk from — a statement inside a resuming handler body — is swept
    under `MAY` instead, because a forward walk seeded there would stop where the real run carries
    on.
    """
    seen: set[int] = set()
    stack: list[CfgNode] = []
    for source in sources:
        if id(source) not in seen:
            seen.add(id(source))
            stack.append(source)
    declines = projection.declines_the_hub and any(
        node.graph.carries_resumption for node in stack)
    if declines and forward:
        coarse = [node for node in stack if node.is_hub_bound]
        if coarse:
            _sweep(coarse, seen, forward=True, declines=False)
            stack = [node for node in stack if not node.is_hub_bound]
    _sweep(stack, seen, forward=forward, declines=declines)
    return seen


def _sweep(
    stack: list[CfgNode], seen: set[int], *, forward: bool, declines: bool,
) -> None:
    while stack:
        node = stack.pop()
        if declines and node.is_resumption_hub:
            continue
        for neighbour in (node.successors if forward else node.predecessors):
            if id(neighbour) in seen:
                continue
            if declines and neighbour.is_resumption_hub:
                continue
            seen.add(id(neighbour))
            stack.append(neighbour)


def reachable_from_any(sources: Iterable[CfgNode]) -> frozenset[int]:
    """
    Every node reachable from *sources*, following every edge the graph draws. A walk over a graph
    that may carry resumption wants `reachable_forward_from_any` instead.
    """
    return frozenset(flood(sources, forward=True, projection=Projection.MAY))


class ControlFlowGraph:
    def __init__(self, owner: Node):
        self.owner = owner
        self._node_of: dict[int, CfgNode] = {}
        self._edge_kinds: dict[tuple[int, int], CfgEdge] = {}
        self._hub_bound: set[int] = set()
        self._resuming = False
        self._fallback: dict[int, CfgNode] = {}
        self.entry = CfgNode(self, None)
        self.exit = CfgNode(self, None)
        self.nodes: list[CfgNode] = [self.entry, self.exit]

    def node_of(self, element: Node) -> CfgNode | None:
        return self._node_of.get(id(element))

    def fallback_of(self, handler: CfgNode) -> CfgNode | None:
        return self._fallback.get(id(handler))

    def edge_kind(self, source: CfgNode, target: CfgNode) -> CfgEdge:
        return self._edge_kinds.get((id(source), id(target)), CfgEdge.NORMAL)

    def is_exceptional(self, source: CfgNode, target: CfgNode) -> bool:
        return bool(self.edge_kind(source, target) & CfgEdge.ERROR_CARRYING)

    def raise_taken(self, source: CfgNode, target: CfgNode) -> bool:
        return bool(self.edge_kind(source, target) & RAISE_TAKEN)

    @property
    def hub_bound(self) -> Set[int]:
        return self._hub_bound

    @property
    def carries_resumption(self) -> bool:
        return self._resuming


def reachable_forward_from_any(
    graph: ControlFlowGraph,
    sources: Iterable[CfgNode],
) -> frozenset[int]:
    """
    The nodes *sources* reach going forward, declining the resumption hub. Every source must belong
    to *graph*; one that does not is a caller error rather than an empty answer, because the walk
    would otherwise report a node unreachable that a run reaches.
    """
    sources = list(sources)
    for source in sources:
        if source.graph is not graph:
            raise ValueError(
                'a source of a forward flood belongs to another body: the two facts this walk '
                'reads are recorded per graph, so such a source reads as neither a hub nor '
                'hub-bound and takes the precise walk over what this graph calls plain flow'
            )
    return frozenset(flood(sources, forward=True, projection=Projection.FORWARD))


class ElementLocator:
    def __init__(self, graphs: dict[int, ControlFlowGraph]):
        self._element_graph: dict[int, ControlFlowGraph] = {}
        self._owners = {id(graph.owner) for graph in graphs.values()}
        for graph in graphs.values():
            for node in graph.nodes:
                if node.element is not None:
                    self._element_graph[id(node.element)] = graph

    def node_of(self, element: Node) -> CfgNode | None:
        graph = self._element_graph.get(id(element))
        return graph.node_of(element) if graph is not None else None

    def locate(self, element: Node) -> tuple[ControlFlowGraph, CfgNode] | None:
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
    One construct a jump may name. `breaks` collects the nodes leaving it early, wired once what
    follows is known; `continues` collects back-jumps whose `continue_to` is not yet built.
    """
    label: str | None
    breaks: list[CfgNode]
    continue_to: CfgNode | None
    is_continuable: bool
    is_breakable: bool
    continues: list[CfgNode] = field(default_factory=list)


@dataclass(eq=False)
class Resumption:
    """
    One statement block being built whose handler set resumes it. `hub` carries the over-approximate
    projection; `slot` is the precise one for the statement being built right now, created on the
    first node that needs it because a statement building no node resumes where the one before it
    does.
    """
    hub: CfgNode
    slot: CfgNode | None = None


def distinct(nodes: Iterable[CfgNode]) -> list[CfgNode]:
    seen: set[int] = set()
    result: list[CfgNode] = []
    for node in nodes:
        if id(node) in seen:
            continue
        seen.add(id(node))
        result.append(node)
    return result


class CfgBuilder:
    def __init__(self, owner: Node):
        self.cfg = ControlFlowGraph(owner)
        self._handlers: list[CfgNode] = []
        self._targets: list[_Target] = []
        self._resumptions: list[Resumption] = []
        self._pending_label: str | None = None

    def build(self) -> ControlFlowGraph:
        frontier = [self.cfg.entry]
        for statements in self.body_blocks(self.cfg.owner):
            frontier = self.block(statements, frontier)
        self.link(frontier, self.cfg.exit)
        return self.cfg

    def body_blocks(self, owner: Node) -> Sequence[Sequence[Node]]:
        """
        The statement blocks *owner* runs, in order. Reported apart rather than joined, because a
        construct scoped to a block cannot be modelled from a body that has forgotten where its
        blocks ended.
        """
        raise NotImplementedError

    def block(self, statements: Sequence[Node], frontier: list[CfgNode]) -> list[CfgNode]:
        return self.sequence(statements, frontier)

    def statement(self, statement: Node, frontier: list[CfgNode]) -> list[CfgNode]:
        raise NotImplementedError

    def node(self, element: Node) -> CfgNode:
        node = self.detached_node(element)
        if self._handlers:
            self.exceptional_edge(node, self._handlers[-1])
        return node

    def synthetic_node(self) -> CfgNode:
        node = CfgNode(self.cfg, None)
        self.cfg.nodes.append(node)
        return node

    def detached_node(self, element: Node) -> CfgNode:
        """
        A graph node for *element* that the enclosing handler does not guard — a handler's own entry
        is one, since it stands for the point at which a throw is offered to a clause.
        """
        node = CfgNode(self.cfg, element)
        self.cfg.nodes.append(node)
        self.cfg._node_of[id(element)] = node
        if self._resumptions:
            self.join_resumption(node)
        return node

    @staticmethod
    def add_edge(source: CfgNode, target: CfgNode) -> None:
        source.successors.append(target)
        target.predecessors.append(source)

    def kind_edge(self, source: CfgNode, target: CfgNode, kind: CfgEdge) -> None:
        self.add_edge(source, target)
        kinds = self.cfg._edge_kinds
        key = (id(source), id(target))
        carried = kinds.get(key)
        kinds[key] = kind if carried is None else carried | kind

    def exceptional_edge(self, source: CfgNode, target: CfgNode) -> None:
        self.kind_edge(source, target, CfgEdge.ERROR_CARRYING)

    def resumption_hub(self) -> CfgNode:
        node = self.synthetic_node()
        node.is_resumption_hub = True
        self.cfg._resuming = True
        return node

    def resumption_hub_edge(self, source: CfgNode, target: CfgNode) -> None:
        self.cfg._resuming = True
        self.kind_edge(source, target, CfgEdge.RESUMPTION_HUB)

    def resumption_forward_edge(self, source: CfgNode, target: CfgNode) -> None:
        self.cfg._resuming = True
        self.kind_edge(source, target, CfgEdge.RESUMPTION_FORWARD)

    def mark_hub_bound(self, nodes: Iterable[CfgNode]) -> None:
        self.cfg._resuming = True
        for node in nodes:
            if node.is_resumption_hub:
                continue
            node.is_hub_bound = True
            self.cfg._hub_bound.add(id(node))

    @contextmanager
    def resumption(self, resumes: Iterable[CfgNode]) -> Iterator[None]:
        """
        Build the enclosed block as one whose handler set resumes it from *resumes*, taking
        ownership of every node built inside it. Ownership ends on every way out, which is why this
        is a block and not a pair of calls.
        """
        hub = self.resumption_hub()
        self.resumption_hub_edge(hub, self.cfg.exit)
        if self._resumptions:
            self.resumption_hub_edge(self._resumptions[-1].hub, hub)
        for resume in resumes:
            self.add_edge(resume, hub)
        self._resumptions.append(Resumption(hub))
        try:
            yield
        finally:
            self._resumptions.pop()

    def resume_past(self, frontier: list[CfgNode]) -> list[CfgNode]:
        """
        *frontier* extended by the point control resumes at when the statement just built throws,
        with the innermost open block left ready for the next one. There is a slot exactly when that
        statement built a node of its own; a statement that built none resumes where the one before
        it does.
        """
        frame = self._resumptions[-1]
        slot, frame.slot = frame.slot, None
        if slot is None:
            return frontier
        return distinct([*frontier, slot])

    def join_resumption(self, node: CfgNode) -> None:
        frame = self._resumptions[-1]
        if frame.slot is None:
            frame.slot = self.synthetic_node()
        self.resumption_forward_edge(node, frame.slot)
        self.resumption_hub_edge(frame.hub, node)

    def close_handler_set(self, entries: Sequence[CfgNode], *, escapes: bool) -> None:
        outward = self.unwinding()
        for index, entry in enumerate(entries):
            following = entries[index + 1] if index + 1 < len(entries) else outward
            self.cfg._fallback[id(entry)] = following
        if escapes and entries:
            self.exceptional_edge(entries[-1], outward)

    def unwinding(self) -> CfgNode:
        return self._handlers[-1] if self._handlers else self.cfg.exit

    def link(self, frontier: Iterable[CfgNode], target: CfgNode) -> None:
        for node in frontier:
            self.add_edge(node, target)

    def sequence(self, statements: Sequence[Node], frontier: list[CfgNode]) -> list[CfgNode]:
        for statement in statements:
            frontier = self.statement(statement, frontier)
        return frontier

    def opaque(self, element: Node, frontier: list[CfgNode]) -> list[CfgNode]:
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
        before = [(node, len(node.successors)) for node in frontier]
        exits = self._body(body, frontier)
        for node, count in before:
            if len(node.successors) > count:
                return node.successors[count], exits
        return None, exits

    def park_label(self, label: str | None) -> None:
        self._pending_label = label

    def take_label(self) -> str | None:
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
        A multi-way branch: *arms* in source order, threaded according to *arm_flow*. An *iterated*
        construct enumerates its input, so `continue` inside it advances the input rather than
        jumping to an enclosing loop, and one that is not *exhaustive* may leave without entering
        any arm.
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
        A construct whose block is guarded by a chain of handler clauses and an optional finalizer.
        The clauses are chained rather than joined because which one runs depends on the error, and
        *escapes* says whether the set may decline it.
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
        node = self.node(element)
        self.link(frontier, node)
        if exceptional:
            self.exceptional_edge(node, self.unwinding())
        else:
            self.add_edge(node, self.cfg.exit)
        return []

    def has_continue_target(self, label: str | None) -> bool:
        return self._continue_target(label) is not None

    def has_break_target(self, label: str | None) -> bool:
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
    def __init__(self, graphs: dict[int, ControlFlowGraph]):
        self.graphs = graphs
        self._locator = ElementLocator(graphs)

    def graph_of(self, owner: Node) -> ControlFlowGraph | None:
        return self.graphs.get(id(owner))

    def node_of(self, element: Node) -> CfgNode | None:
        return self._locator.node_of(element)

    def locate(self, element: Node) -> tuple[ControlFlowGraph, CfgNode] | None:
        return self._locator.locate(element)


def build_control_flow(
    root: Node,
    builder: type[CfgBuilder],
    function_nodes: tuple[type, ...],
) -> dict[int, ControlFlowGraph]:
    graphs: dict[int, ControlFlowGraph] = {id(root): builder(root).build()}
    for node in root.walk():
        if isinstance(node, function_nodes):
            graphs[id(node)] = builder(node).build()
    return graphs
