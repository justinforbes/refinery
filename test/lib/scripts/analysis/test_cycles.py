from __future__ import annotations

from test import TestBase

from refinery.lib.scripts import Script
from refinery.lib.scripts.analysis.cfg import CfgBuilder, CfgNode, ControlFlowGraph
from refinery.lib.scripts.analysis.cycles import nodes_on_a_cycle, strongly_connected_components


def _graph(edges: dict[str, list[str]]) -> tuple[ControlFlowGraph, dict[str, CfgNode]]:
    """
    A graph over the names in *edges*, each mapped to the names it reaches. The synthetic entry and
    exit a real graph carries are dropped, so that a component count is a count of the named nodes.
    """
    graph = ControlFlowGraph(Script())
    graph.nodes.clear()
    named = {name: CfgNode(None) for name in edges}
    graph.nodes.extend(named.values())
    for name, successors in edges.items():
        for successor in successors:
            CfgBuilder.add_edge(named[name], named[successor])
    return graph, named


class TestCycles(TestBase):

    def _on_a_cycle(self, edges: dict[str, list[str]]) -> set[str]:
        graph, named = _graph(edges)
        found = nodes_on_a_cycle(graph)
        return {name for name, node in named.items() if id(node) in found}

    def test_an_isolated_node_is_not_on_a_cycle(self):
        self.assertEqual(self._on_a_cycle({'a': []}), set())

    def test_a_chain_puts_no_node_on_a_cycle(self):
        self.assertEqual(self._on_a_cycle({'a': ['b'], 'b': ['c'], 'c': []}), set())

    def test_a_node_that_reaches_itself_directly_is_on_a_cycle(self):
        """
        A self-edge is a cycle, and its component has one member like every acyclic node's does.
        """
        self.assertEqual(self._on_a_cycle({'a': ['a']}), {'a'})

    def test_every_node_of_a_ring_is_on_a_cycle(self):
        edges = {'a': ['b'], 'b': ['c'], 'c': ['a']}
        self.assertEqual(self._on_a_cycle(edges), {'a', 'b', 'c'})

    def test_a_node_on_the_path_between_two_cycles_is_not_on_either(self):
        """
        The case that separates cycle membership from merely having a predecessor and a successor.
        """
        edges = {
            'a': ['b'],
            'b': ['a', 'm'],
            'm': ['c'],
            'c': ['d'],
            'd': ['c'],
        }
        self.assertEqual(self._on_a_cycle(edges), {'a', 'b', 'c', 'd'})

    def test_a_node_that_only_reaches_a_cycle_is_not_on_it(self):
        edges = {'s': ['a'], 'a': ['b'], 'b': ['a']}
        self.assertEqual(self._on_a_cycle(edges), {'a', 'b'})

    def test_a_node_only_reachable_from_a_cycle_is_not_on_it(self):
        edges = {'a': ['b'], 'b': ['a', 't'], 't': []}
        self.assertEqual(self._on_a_cycle(edges), {'a', 'b'})

    def test_two_cycles_sharing_a_node_form_one_component(self):
        edges = {
            'h': ['a', 'x'],
            'a': ['h'],
            'x': ['h'],
        }
        components = strongly_connected_components(_graph(edges)[0])
        self.assertEqual(sorted(len(c) for c in components), [3])

    def test_disjoint_cycles_are_separate_components(self):
        edges = {
            'a': ['b'],
            'b': ['a'],
            'c': ['d'],
            'd': ['c'],
        }
        components = strongly_connected_components(_graph(edges)[0])
        self.assertEqual(sorted(len(c) for c in components), [2, 2])

    def test_every_node_belongs_to_exactly_one_component(self):
        edges = {
            'a': ['b'],
            'b': ['c', 'd'],
            'c': ['a'],
            'd': ['d'],
            'e': [],
        }
        graph, named = _graph(edges)
        members = [node for component in strongly_connected_components(graph) for node in component]
        self.assertEqual(len(members), len(named))
        self.assertEqual(len({id(node) for node in members}), len(named))

    def test_a_chain_longer_than_the_interpreter_recursion_limit_is_handled(self):
        """
        The graphs are as deep as the script is long, so the traversal may not recurse.
        """
        length = 20000
        names = [F'n{index}' for index in range(length)]
        edges = {name: [] for name in names}
        for source, target in zip(names, names[1:]):
            edges[source] = [target]
        edges[names[-1]] = [names[0]]
        self.assertEqual(len(self._on_a_cycle(edges)), length)
