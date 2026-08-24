"""
Dominance over the per-body control-flow graphs of one script, in the graph-theoretic sense only:
one node dominates another when every path from the entry to the second passes through the first.

The queries are keyed to AST nodes, because that is what callers hold; an expression resolves to the
statement that evaluates it. Two elements in different bodies are not ordered here at all.

A node no path from the entry reaches is dominated by nothing and dominates nothing, itself included
— every answer about one is vacuous, and the tail after a `return` or a `throw` makes that the
ordinary case rather than the exotic one.

Dominance orders, it does not certify: that a definition dominates a use says the statement ran, not
that its store completed. A caller reading it as evidence a value was established needs the refusal
`refinery.lib.scripts.analysis.reaching` makes.
"""
from __future__ import annotations

from typing import AbstractSet as Set, Iterator

from refinery.lib.scripts import Node
from refinery.lib.scripts.analysis.cfg import (
    CfgNode,
    ControlFlowGraph,
    ControlFlowModel,
    Projection,
    flood,
)


def _reverse_postorder(graph: ControlFlowGraph, projection: Projection) -> list[CfgNode]:
    order: list[CfgNode] = []
    seen: set[int] = {id(graph.entry)}
    stack: list[tuple[CfgNode, Iterator[CfgNode]]] = [
        (graph.entry, iter(projection.successors(graph.entry)))
    ]
    while stack:
        node, successors = stack[-1]
        for successor in successors:
            if id(successor) in seen:
                continue
            seen.add(id(successor))
            stack.append((successor, iter(projection.successors(successor))))
            break
        else:
            order.append(node)
            stack.pop()
    order.reverse()
    return order


def _immediate_dominators(
    graph: ControlFlowGraph, order: list[CfgNode], projection: Projection,
) -> dict[int, int]:
    """
    Cooper-Harvey-Kennedy: walk the graph in reverse postorder until no immediate dominator changes,
    intersecting each node's already-placed predecessors by climbing the tree built so far.

    The predecessors are folded deepest-first, which changes only how many iterations it takes and
    not the answer, the intersect being commutative and associative at the fixpoint. Folding in list
    order instead costs one full climb per guarded statement on a handler entry, which is quadratic
    in the size of a guarded block.
    """
    rank = {id(node): index for index, node in enumerate(order)}
    idom: dict[int, int] = {id(graph.entry): id(graph.entry)}

    def common(a: int, b: int) -> int:
        while a != b:
            while rank[a] > rank[b]:
                a = idom[a]
            while rank[b] > rank[a]:
                b = idom[b]
        return a

    ranked = [
        (
            id(node),
            sorted(
                (known for known in map(id, projection.predecessors(node)) if known in rank),
                key=rank.__getitem__,
                reverse=True,
            ),
        )
        for node in order
        if node is not graph.entry
    ]
    changed = True
    while changed:
        changed = False
        for key, predecessors in ranked:
            candidate: int | None = None
            for known in predecessors:
                if known not in idom:
                    continue
                candidate = known if candidate is None else common(known, candidate)
            if candidate is not None and idom.get(key) != candidate:
                idom[key] = candidate
                changed = True
    return idom


def _dominance_intervals(
    idom: dict[int, int], graph: ControlFlowGraph, order: list[CfgNode],
) -> tuple[dict[int, int], dict[int, int]]:
    root = id(graph.entry)
    children: dict[int, list[int]] = {id(node): [] for node in order}
    for node in order:
        key = id(node)
        if key != root:
            children[idom[key]].append(key)
    entered: dict[int, int] = {root: 0}
    left: dict[int, int] = {}
    clock = 1
    stack: list[tuple[int, Iterator[int]]] = [(root, iter(children[root]))]
    while stack:
        key, remaining = stack[-1]
        for child in remaining:
            entered[child] = clock
            clock += 1
            stack.append((child, iter(children[child])))
            break
        else:
            left[key] = clock
            stack.pop()
    return entered, left


class DominanceTree:
    """
    The immediate-dominator tree of one graph under one projection, with the entry depth of each
    node, so that `dominates` is a climb rather than a walk.
    """

    def __init__(self, graph: ControlFlowGraph, projection: Projection):
        order = _reverse_postorder(graph, projection)
        self.idom = _immediate_dominators(graph, order, projection)
        self._entered, self._left = _dominance_intervals(self.idom, graph, order)

    @property
    def reached(self) -> Set[int]:
        return self._entered.keys()

    def dominates(self, a: CfgNode, b: CfgNode) -> bool:
        opened = self._entered.get(id(a))
        if opened is None:
            return False
        reached = self._entered.get(id(b))
        return reached is not None and opened <= reached < self._left[id(a)]


class DominatorModel:
    """
    The dominance trees of one script's graphs, built on demand and memoized per graph and
    projection.
    """

    def __init__(self, flow: ControlFlowModel):
        self._flow = flow
        self._trees: dict[tuple[int, Projection], DominanceTree] = {}

    def locate(self, element: Node) -> tuple[ControlFlowGraph, CfgNode] | None:
        return self._flow.locate(element)

    def cfg_node_of(self, element: Node) -> CfgNode | None:
        located = self._flow.locate(element)
        return located[1] if located is not None else None

    def locate_pair(self, a: Node, b: Node) -> tuple[ControlFlowGraph, CfgNode, CfgNode] | None:
        located_a = self._flow.locate(a)
        located_b = self._flow.locate(b)
        if located_a is None or located_b is None:
            return None
        graph_a, node_a = located_a
        graph_b, node_b = located_b
        if graph_a is not graph_b:
            return None
        return graph_a, node_a, node_b

    def dominates(self, a: Node, b: Node) -> bool:
        located = self.locate_pair(a, b)
        return located is not None and self.dominates_node(*located, Projection.MAY)

    def strictly_dominates(self, a: Node, b: Node) -> bool:
        located = self.locate_pair(a, b)
        if located is None or located[1] is located[2]:
            return False
        return self.dominates_node(*located, Projection.MAY)

    def dominates_node(
        self, graph: ControlFlowGraph, a: CfgNode, b: CfgNode, projection: Projection,
    ) -> bool:
        """
        Whether *a* dominates *b*, both being nodes of *graph*. A node the entry does not reach
        answers `False` in either position.
        """
        return self.tree_of(graph, projection).dominates(a, b)

    def tree_of(self, graph: ControlFlowGraph, projection: Projection) -> DominanceTree:
        key = (id(graph), projection if graph.carries_resumption else Projection.MAY)
        found = self._trees.get(key)
        if found is None:
            found = self._trees[key] = DominanceTree(graph, key[1])
        return found

    def reached_from_entry(self, graph: ControlFlowGraph, projection: Projection) -> Set[int]:
        return self.tree_of(graph, projection).reached

    def reachable(
        self, start: CfgNode, *, forward: bool, projection: Projection,
    ) -> set[int]:
        """
        The nodes reachable from *start* in the direction *forward* names, under *projection*.
        """
        return flood([start], forward=forward, projection=projection)
