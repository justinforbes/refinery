"""
Dominance over the per-body control-flow graphs of one script, in the graph-theoretic sense only.

One node *dominates* another when every path from the body's entry to the second passes through the
first, so the first is guaranteed to have executed by the time the second runs. That is a statement
about a graph and nothing else, which is why the whole of it lives here: a language contributes the
graph, and the relation over it is the same relation for every language.

The queries are keyed to AST nodes rather than to graph nodes, because that is what every caller
holds. An element the graph does not represent on its own — an expression inside a statement —
resolves to the node of the statement that evaluates it, which is the granularity the graph reasons
at. Two elements in different bodies' graphs are not ordered here at all: whether one body runs
before another is a question about calls, and a language answers it with the layer it builds on this
one.

Exceptional edges take part in the computation like any other edge. A definition is therefore
reported as dominating a use only when it runs first on *every* path, including the ones that leave a
guarded block by throwing, which is the conservative direction and the one a caller may rely on.
"""
from __future__ import annotations

from refinery.lib.scripts import Node
from refinery.lib.scripts.analysis.cfg import CfgNode, ControlFlowGraph, ControlFlowModel


class DominatorModel:
    """
    Dominator relations for the per-body control-flow graphs of one script.

    Every graph's dominator sets are computed once, on construction, because the caller that asks one
    ordering question almost always asks many, and the sets are fixed for as long as the tree is.
    """

    def __init__(self, flow: ControlFlowModel):
        self._flow = flow
        self._dominators: dict[int, frozenset[int]] = {}
        for graph in flow.graphs.values():
            self._compute_dominators(graph)

    def locate(self, element: Node) -> tuple[ControlFlowGraph, CfgNode] | None:
        """
        The control-flow graph and node that evaluate *element*, climbing out of any expression it is
        nested in, or `None` when it has no enclosing graph node. The graph identifies the body whose
        invocation runs *element*, which a caller needs to keep a query within one graph.
        """
        return self._flow.locate(element)

    def cfg_node_of(self, element: Node) -> CfgNode | None:
        """
        The control-flow node of the statement or loop head that evaluates *element*, or `None` when
        *element* has no enclosing graph node.
        """
        located = self._flow.locate(element)
        return located[1] if located is not None else None

    def locate_pair(self, a: Node, b: Node) -> tuple[CfgNode, CfgNode] | None:
        """
        The control-flow nodes that evaluate *a* and *b* when both lie in the same body's graph, or
        `None` when either is unlocatable or the two lie in different graphs, where intraprocedural
        dominance does not apply.
        """
        located_a = self._flow.locate(a)
        located_b = self._flow.locate(b)
        if located_a is None or located_b is None:
            return None
        graph_a, node_a = located_a
        graph_b, node_b = located_b
        if graph_a is not graph_b:
            return None
        return node_a, node_b

    def dominates(self, a: Node, b: Node) -> bool:
        """
        Whether the statement evaluating *a* is guaranteed to have executed by the time the statement
        evaluating *b* runs. Reflexive — *a* and *b* in the same statement share one control-flow
        node, so a node dominates itself. `False` when either element is unlocatable or the two lie
        in different graphs.
        """
        pair = self.locate_pair(a, b)
        return pair is not None and self.dominates_node(*pair)

    def strictly_dominates(self, a: Node, b: Node) -> bool:
        """
        Like `dominates`, but not reflexive: `False` when *a* and *b* share one control-flow node.

        A caller that must reject a same-statement occurrence needs this. Two occurrences inside one
        statement may be evaluated in either order as far as this granularity can tell, so the
        reflexive answer would accept a reference that is in fact evaluated first.
        """
        pair = self.locate_pair(a, b)
        if pair is None or pair[0] is pair[1]:
            return False
        return self.dominates_node(*pair)

    def dominates_node(self, a: CfgNode, b: CfgNode) -> bool:
        """
        Whether control-flow node *a* dominates *b*. Reflexive. The node-level counterpart of
        `dominates`, for a caller that has already located the two.
        """
        return id(a) in self._dominators.get(id(b), frozenset())

    def reachable(self, start: CfgNode, *, forward: bool) -> set[int]:
        """
        The ids of the control-flow nodes reachable from *start* — over successor edges when
        *forward*, over predecessor edges otherwise — including *start* itself. Exceptional edges are
        followed like any other, since they live in the same successor and predecessor lists.

        The two directions are kept separate rather than offered as one path-between query because a
        caller intersecting them memoizes each side independently: one definition is asked against
        many uses, and the forward set from the definition is the same every time.
        """
        seen: set[int] = {id(start)}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbour in (node.successors if forward else node.predecessors):
                if id(neighbour) not in seen:
                    seen.add(id(neighbour))
                    stack.append(neighbour)
        return seen

    def _compute_dominators(self, graph: ControlFlowGraph) -> None:
        """
        The classic iterative intersection: every node starts dominated by everything, the entry by
        itself alone, and each round replaces a node's set with the intersection over its
        predecessors plus itself, until nothing moves.

        A node with no predecessors that is not the entry is unreachable, and takes the empty set
        before adding itself — so it dominates only itself and is dominated by nothing, which keeps
        an unreachable region from claiming to dominate anything downstream of it.
        """
        nodes = graph.nodes
        all_ids = {id(node) for node in nodes}
        dom: dict[int, set[int]] = {
            id(node): {id(node)} if node is graph.entry else set(all_ids) for node in nodes
        }
        changed = True
        while changed:
            changed = False
            for node in nodes:
                if node is graph.entry:
                    continue
                incoming = node.predecessors
                if incoming:
                    new = set(all_ids)
                    for predecessor in incoming:
                        new &= dom[id(predecessor)]
                else:
                    new = set()
                new.add(id(node))
                if new != dom[id(node)]:
                    dom[id(node)] = new
                    changed = True
        for node in nodes:
            self._dominators[id(node)] = frozenset(dom[id(node)])
