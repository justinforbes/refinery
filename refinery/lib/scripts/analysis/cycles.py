"""
Whether a point in a script can be reached more than once, over the control-flow graphs of one
script.

This is the question every loop-awareness check in a transform is really asking. A value established
at a point holds until something changes it — but if control can come back to that point, the value
it establishes on the second visit is not the one an earlier reader saw, and a fact derived from a
single visit is not a fact about the program. Loops are the obvious way that happens and the reason
the check is usually written as a walk up the ancestors looking for loop node types, which is a
guess at the graph rather than a reading of it: it misses a construct that iterates without being
spelled as a loop, and it claims one for a body that runs once.

Read off the graph, the question is exactly whether the node lies on a cycle, which is whether it
belongs to a strongly connected component with more than one member or carries an edge to itself.
Nothing else in the shape of the source matters.

**One body's graph only answers for one invocation of that body.** A block written inside a loop is
invoked once per iteration, so everything in it repeats even though its own graph is acyclic — the
repetition is the *owner's*, and `CycleModel.repeats` asks the owner in turn. That walk is lexical:
it follows a body to where its value is written, not to where it is called, so a block stored in a
variable and invoked from a loop elsewhere is reported as running once. Answering that soundly is a
call-graph question, and the graphs here do not hold it.
"""
from __future__ import annotations

from refinery.lib.scripts import Node
from refinery.lib.scripts.analysis.cfg import CfgNode, ControlFlowGraph, ControlFlowModel


def strongly_connected_components(graph: ControlFlowGraph) -> list[list[CfgNode]]:
    """
    The strongly connected components of *graph*, each a list of the nodes that can all reach one
    another. Tarjan's algorithm, driven by an explicit stack rather than by recursion: the graphs
    are as deep as the script is long and a deeply nested body would otherwise exhaust the
    interpreter's stack on a script that parses fine.
    """
    index: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    stacked: set[int] = set()
    pending: list[CfgNode] = []
    components: list[list[CfgNode]] = []
    counter = 0
    for start in graph.nodes:
        if id(start) in index:
            continue
        work: list[tuple[CfgNode, int]] = [(start, 0)]
        while work:
            node, position = work[-1]
            if position == 0:
                index[id(node)] = lowlink[id(node)] = counter
                counter += 1
                pending.append(node)
                stacked.add(id(node))
            descended = False
            for offset in range(position, len(node.successors)):
                successor = node.successors[offset]
                if id(successor) not in index:
                    work[-1] = (node, offset + 1)
                    work.append((successor, 0))
                    descended = True
                    break
                if id(successor) in stacked:
                    lowlink[id(node)] = min(lowlink[id(node)], index[id(successor)])
            if descended:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[id(parent)] = min(lowlink[id(parent)], lowlink[id(node)])
            if lowlink[id(node)] == index[id(node)]:
                component: list[CfgNode] = []
                while True:
                    member = pending.pop()
                    stacked.discard(id(member))
                    component.append(member)
                    if member is node:
                        break
                components.append(component)
    return components


def nodes_on_a_cycle(graph: ControlFlowGraph) -> frozenset[int]:
    """
    The ids of the nodes of *graph* that control can return to: those in a strongly connected
    component of more than one node, and those carrying an edge to themselves.

    A component of one node is not a cycle unless that edge is there, which is the case a plain
    component-size test gets wrong — a construct that iterates over nothing but its own head, such as
    a dispatch whose every arm is empty, is one node reaching only itself.
    """
    found: set[int] = set()
    for component in strongly_connected_components(graph):
        if len(component) > 1:
            found.update(id(node) for node in component)
            continue
        node = component[0]
        if any(successor is node for successor in node.successors):
            found.add(id(node))
    return frozenset(found)


class CycleModel:
    """
    Which points of one script can be reached more than once. Every graph's cycle set is computed
    once, on construction, because a caller that asks about one node almost always asks about many
    and the sets are fixed for as long as the tree is.
    """

    def __init__(self, flow: ControlFlowModel):
        self._flow = flow
        self._on_a_cycle: set[int] = set()
        for graph in flow.graphs.values():
            self._on_a_cycle |= nodes_on_a_cycle(graph)

    def on_a_cycle(self, node: CfgNode) -> bool:
        """
        Whether control can return to *node* within one invocation of the body it belongs to. The
        node-level counterpart of `repeats`, for a caller that has already located it.
        """
        return id(node) in self._on_a_cycle

    def repeats(self, element: Node) -> bool:
        """
        Whether *element* can be evaluated more than once by one run of the script — because the
        statement evaluating it lies on a cycle, or because the body it is written in does.

        `False` for an element the graphs do not place at all, which is the answer a caller wanting
        to know whether a single-visit fact holds must not act on as if it were proven. See the
        module docstring for what the walk out through the enclosing bodies does and does not see.
        """
        located = self._flow.locate(element)
        while located is not None:
            graph, node = located
            if self.on_a_cycle(node):
                return True
            located = self._flow.locate(graph.owner)
        return False
