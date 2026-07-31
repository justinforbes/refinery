from __future__ import annotations

from refinery.lib.scripts import Script
from refinery.lib.scripts.analysis.cfg import CfgBuilder, CfgNode, ControlFlowGraph


def graph_from_edges(edges: dict[str, list[str]]) -> tuple[ControlFlowGraph, dict[str, CfgNode]]:
    """
    A graph over the names in *edges*, each mapped to the names it reaches, with the **first** name
    as the entry.

    The synthetic entry and exit a real graph carries are dropped, so that a component count is a
    count of the named nodes and a dominator set holds nothing a test did not put there. Writing the
    graph out rather than parsing a script keeps these tests on the shapes the algorithms are stated
    over, which no one language's syntax can produce all of.
    """
    graph = ControlFlowGraph(Script())
    graph.nodes.clear()
    named = {name: CfgNode(None) for name in edges}
    graph.entry = named[next(iter(edges))]
    graph.nodes.extend(named.values())
    for name, successors in edges.items():
        for successor in successors:
            CfgBuilder.add_edge(named[name], named[successor])
    return graph, named
