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

The shapes are parameterised where languages genuinely differ rather than being duplicated: a
`dispatch` either falls through from one arm to the next or does not, and both spellings exist in
languages this substrate serves.
"""
from __future__ import annotations

import enum

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from refinery.lib.scripts import Node


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


@dataclass(eq=False)
class CfgNode:
    """
    One vertex of a control-flow graph. `element` is the AST node it stands for — a statement, or a
    loop-head expression whose reads and writes occur at this point — or `None` for the synthetic
    entry and exit. `successors` lists the nodes control may pass to next.

    `eq=False` is load bearing. Every map in every layer above is keyed by `id(node)`, and two
    structurally equal statements are two distinct points in the program.
    """
    element: Node | None
    successors: list[CfgNode] = field(default_factory=list)
    predecessors: list[CfgNode] = field(default_factory=list)


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
        self.exceptional_edges: set[tuple[int, int]] = set()

    def node_of(self, element: Node) -> CfgNode | None:
        """
        The graph node standing for *element*, or `None` if *element* is not part of this body, or is
        a node the graph does not represent on its own such as a plain expression inside a statement.
        """
        return self._node_of.get(id(element))

    def is_exceptional(self, source: CfgNode, target: CfgNode) -> bool:
        """
        Whether the edge from *source* to *target* is taken only when *source* throws rather than
        completing normally. A definition *source* makes is not guaranteed to have happened along
        such an edge, so a flow-sensitive analysis must not treat it as a kill there.
        """
        return (id(source), id(target)) in self.exceptional_edges


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
        """
        cursor: Node | None = element
        while cursor is not None:
            graph = self._element_graph.get(id(cursor))
            if graph is not None:
                node = graph.node_of(cursor)
                if node is not None:
                    return graph, node
            cursor = cursor.parent
        return None


@dataclass
class _Target:
    """
    A jump destination active while a breakable or continuable construct is being built. `breaks`
    collects the nodes that leave it early, wired to whatever follows once that is known;
    `continue_to` is the node a back-jump reaches, or `None` for a construct only a break can leave.
    """
    label: str | None
    breaks: list[CfgNode]
    continue_to: CfgNode | None
    is_loop: bool
    is_breakable: bool


class CfgBuilder:
    """
    Single-pass construction of one `ControlFlowGraph` by structural recursion over a body, threading
    a *frontier* — the set of nodes from which normal control currently falls through — into each
    statement and out the other side.

    A language subclasses this and implements `statement` and `body_statements`. Everything else is
    shared, and the shape methods below take the parts of a construct rather than the construct, so
    that no code here has to know what a language calls the pieces of its `for` loop.
    """

    def __init__(self, owner: Node):
        self.cfg = ControlFlowGraph(owner)
        self._handlers: list[CfgNode] = []
        self._targets: list[_Target] = []
        self._pending_label: str | None = None

    def build(self) -> ControlFlowGraph:
        frontier = self.sequence(self.body_statements(self.cfg.owner), [self.cfg.entry])
        self.link(frontier, self.cfg.exit)
        return self.cfg

    def body_statements(self, owner: Node) -> list[Node]:
        """
        The statements *owner* runs, in order. A language whose body may be a single unbraced
        statement, or which splits one body across several named blocks, resolves that here.
        """
        raise NotImplementedError

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
        node = CfgNode(element)
        self.cfg.nodes.append(node)
        self.cfg._node_of[id(element)] = node
        if self._handlers:
            self.exceptional_edge(node, self._handlers[-1])
        return node

    @staticmethod
    def add_edge(source: CfgNode, target: CfgNode) -> None:
        source.successors.append(target)
        target.predecessors.append(source)

    def exceptional_edge(self, source: CfgNode, target: CfgNode) -> None:
        self.add_edge(source, target)
        self.cfg.exceptional_edges.add((id(source), id(target)))

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

    def take_label(self) -> str | None:
        """
        The label a `labelled` shape parked for the construct about to be built, consumed once.
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
        target = _Target(self.take_label(), [], head, is_loop=True, is_breakable=True)
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
        target = _Target(self.take_label(), [], test, is_loop=True, is_breakable=True)
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
        target = _Target(label, [], step or head, is_loop=True, is_breakable=True)
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
            return exits + self._body(otherwise, current)
        return exits + current

    def dispatch(
        self,
        element: Node,
        arms: Sequence[Sequence[Node]],
        frontier: list[CfgNode],
        *,
        arm_flow: ArmFlow,
        exhaustive: bool,
    ) -> list[CfgNode]:
        """
        A multi-way branch: one head the frontier enters and one arm per clause, each arm a statement
        sequence the head may jump into. `arm_flow` says what an arm reaches when it runs off its
        end; see `ArmFlow`, which is where the languages differ.

        `exhaustive` says some arm always runs — a default clause — so the head is not itself an
        exit.
        """
        head = self.node(element)
        self.link(frontier, head)
        target = _Target(self.take_label(), [], None, is_loop=False, is_breakable=True)
        self._targets.append(target)
        carried: list[CfgNode] = []
        exits: list[CfgNode] = []
        for arm in arms:
            reached = self.sequence(list(arm), [head] + carried)
            if arm_flow is ArmFlow.SEQUENTIAL:
                carried = reached
            elif arm_flow is ArmFlow.CUMULATIVE:
                carried = carried + reached
            else:
                exits += reached
        self._targets.pop()
        exits += list(carried) + target.breaks
        if not exhaustive:
            exits.append(head)
        return exits

    def guarded(
        self,
        block: Node | None,
        handlers: Sequence[tuple[Node, Node | None]],
        finalizer: Node | None,
        finalizer_body: Sequence[Node],
        frontier: list[CfgNode],
    ) -> list[CfgNode]:
        """
        A guarded block with any number of handlers and an optional finalizer.

        The handler nodes are created *before* the guarded block is built and the first is pushed on
        the handler stack, so that `node` joins every statement created inside the block to it. That
        ordering is the whole mechanism: it is why no statement inside the block has to know it is
        guarded.

        Several handlers are chained from the first, because which one runs depends on the type of
        the exception and none of them is guaranteed — a language with one handler passes a
        one-element sequence and the chain degenerates.

        The finalizer is entered from the block's normal exits and from every handler's, and itself
        carries an exceptional edge outward, because a finalizer runs on the exceptional path too and
        control leaves the construct from it either way.
        """
        entries = [self.node(handler) for handler, _ in handlers]
        finalizer_entry: CfgNode | None = None
        if finalizer is not None:
            finalizer_entry = CfgNode(finalizer)
            self.cfg.nodes.append(finalizer_entry)
        guard = entries[0] if entries else finalizer_entry
        if guard is not None:
            self._handlers.append(guard)
        block_exits = self.statement(block, frontier) if block is not None else list(frontier)
        if guard is not None:
            self._handlers.pop()
        normal_exits = list(block_exits)
        for index, ((_, body), entry) in enumerate(zip(handlers, entries)):
            if index:
                self.add_edge(entries[index - 1], entry)
            normal_exits += (
                self.statement(body, [entry]) if body is not None else [entry])
        if finalizer_entry is not None and finalizer is not None:
            self.link(normal_exits, finalizer_entry)
            self.cfg._node_of[id(finalizer)] = finalizer_entry
            final_exits = self.sequence(list(finalizer_body), [finalizer_entry])
            self.exceptional_edge(
                finalizer_entry, self._handlers[-1] if self._handlers else self.cfg.exit)
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
            self._pending_label = label
            return self.statement(body, frontier) if body is not None else list(frontier)
        target = _Target(label, [], None, is_loop=False, is_breakable=False)
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
        if target is not None and target.continue_to is not None:
            self.add_edge(node, target.continue_to)
        else:
            self.add_edge(node, self.cfg.exit)
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
            self.exceptional_edge(node, self._handlers[-1] if self._handlers else self.cfg.exit)
        else:
            self.add_edge(node, self.cfg.exit)
        return []

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
            if not target.is_loop:
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
