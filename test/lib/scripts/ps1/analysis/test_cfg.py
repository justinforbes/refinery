from __future__ import annotations

from test import TestBase

from refinery.lib.scripts.analysis.cfg import CfgNode, ControlFlowGraph
from refinery.lib.scripts.ps1.analysis.cfg import build_ps1_control_flow
from refinery.lib.scripts.ps1.model import (
    Ps1BreakStatement,
    Ps1ContinueStatement,
    Ps1FunctionDefinition,
    Ps1Script,
    Ps1TrapStatement,
)
from refinery.lib.scripts.ps1.parser import Ps1Parser

#: One script per construct the builder recognises. The invariants below are asserted over every
#: entry, so a construct added to the dispatch without an entry here is covered by nothing.
_CORPUS = [
    "Write-Host 'a'",
    "if ($x) { 'a' }",
    "if ($x) { 'a' } else { 'b' }",
    "if ($x) { 'a' } elseif ($y) { 'b' } elseif ($z) { 'c' } else { 'd' }",
    "while ($x) { 'a' }",
    "while ($x) { break }",
    "while ($x) { continue }",
    "do { 'a' } while ($x)",
    "do { 'a' } until ($x)",
    "for ($i = 0; $i -lt 3; $i++) { 'a' }",
    "for (;;) { 'a' }",
    "foreach ($i in $x) { 'a' }",
    "switch ($x) { 1 { 'a' } 2 { 'b' } }",
    "switch ($x) { 1 { 'a' } default { 'b' } }",
    "switch ($x) { 1 { break } 2 { 'b' } }",
    "try { 'a' } catch { 'b' }",
    "try { 'a' } finally { 'c' }",
    "try { 'a' } catch { 'b' } finally { 'c' }",
    "trap { continue }\n'a'",
    "function f { 'a' }\nf",
    "function f { return 1 }",
    "'a'\nexit\n'b'",
    "'a'\nthrow 'x'\n'b'",
    ":outer while ($x) { while ($y) { break outer } }",
    "'a'",
]


class TestPs1ControlFlowGraph(TestBase):

    @staticmethod
    def _graphs(source: str) -> dict[int, ControlFlowGraph]:
        return build_ps1_control_flow(Ps1Parser(source).parse())

    def _script_graph(self, source: str) -> ControlFlowGraph:
        tree = Ps1Parser(source).parse()
        return build_ps1_control_flow(tree)[id(tree)]

    @staticmethod
    def _forward(start: CfgNode) -> set[int]:
        seen = {id(start)}
        stack = [start]
        while stack:
            for successor in stack.pop().successors:
                if id(successor) not in seen:
                    seen.add(id(successor))
                    stack.append(successor)
        return seen

    def _node_for(self, graph: ControlFlowGraph, kind: type) -> CfgNode:
        for node in graph.nodes:
            if isinstance(node.element, kind):
                return node
        self.fail(F'no control-flow node stands for a {kind.__name__}')

    def test_every_graph_is_internally_consistent(self):
        for source in _CORPUS:
            with self.subTest(source):
                for graph in self._graphs(source).values():
                    held = {id(node) for node in graph.nodes}
                    for node in graph.nodes:
                        for successor in node.successors:
                            self.assertIn(id(successor), held)
                            self.assertIn(node, successor.predecessors)
                        for predecessor in node.predecessors:
                            self.assertIn(id(predecessor), held)
                            self.assertIn(node, predecessor.successors)

    def test_entry_has_no_predecessors_and_exit_no_successors(self):
        for source in _CORPUS:
            with self.subTest(source):
                for graph in self._graphs(source).values():
                    self.assertEqual(graph.entry.predecessors, [])
                    self.assertEqual(graph.exit.successors, [])

    def test_every_statement_node_is_reachable_from_entry_unless_written_after_a_terminator(self):
        for source in _CORPUS:
            if 'exit' in source or 'throw' in source:
                continue
            with self.subTest(source):
                for graph in self._graphs(source).values():
                    reached = self._forward(graph.entry)
                    for node in graph.nodes:
                        if node.element is not None:
                            self.assertIn(id(node), reached)

    def test_a_function_definition_owns_its_own_graph(self):
        graphs = self._graphs("function f { 'a' }\nf")
        self.assertEqual(len(graphs), 2)

    def test_a_nested_function_body_is_not_descended_into(self):
        tree = Ps1Parser("function f { 'a' }\nf").parse()
        graphs = build_ps1_control_flow(tree)
        definition = next(n for n in tree.walk() if isinstance(n, Ps1FunctionDefinition))
        script = graphs[id(tree)]
        self.assertIsNotNone(script.node_of(definition))
        self.assertIsNot(graphs[id(definition)], script)

    def test_a_while_loop_has_a_back_edge(self):
        tree = Ps1Parser("while ($x) { 'a' }").parse()
        graph = build_ps1_control_flow(tree)[id(tree)]
        head = graph.node_of(tree.body[0])
        self.assertIsNotNone(head)
        body = next(n for n in graph.nodes if n.element is not None and n is not head)
        self.assertIn(id(head), self._forward(body))

    def test_break_leaves_the_loop_and_does_not_return_to_its_head(self):
        graph = self._script_graph("while ($x) { break }")
        jump = self._node_for(graph, Ps1BreakStatement)
        self.assertIn(graph.exit, jump.successors)

    def test_continue_returns_to_the_loop_head(self):
        graph = self._script_graph("while ($x) { continue }")
        jump = self._node_for(graph, Ps1ContinueStatement)
        self.assertNotIn(graph.exit, jump.successors)
        self.assertTrue(jump.successors)

    def test_a_labelled_break_leaves_the_named_loop(self):
        graph = self._script_graph(":outer while ($x) { while ($y) { break outer } }")
        jump = self._node_for(graph, Ps1BreakStatement)
        self.assertTrue(jump.successors)

    def test_code_after_exit_is_unreachable(self):
        tree = Ps1Parser("'a'\nexit\n'b'").parse()
        graph = build_ps1_control_flow(tree)[id(tree)]
        reached = self._forward(graph.entry)
        unreachable = [
            node for node in graph.nodes
            if node.element is not None and id(node) not in reached
        ]
        self.assertEqual(len(unreachable), 1)

    def test_code_after_throw_is_unreachable(self):
        tree = Ps1Parser("'a'\nthrow 'x'\n'b'").parse()
        graph = build_ps1_control_flow(tree)[id(tree)]
        reached = self._forward(graph.entry)
        unreachable = [
            node for node in graph.nodes
            if node.element is not None and id(node) not in reached
        ]
        self.assertEqual(len(unreachable), 1)

    def test_a_guarded_statement_has_an_exceptional_edge_to_the_handler(self):
        graph = self._script_graph("try { 'a' } catch { 'b' }")
        exceptional = [
            (source, target)
            for source in graph.nodes
            for target in source.successors
            if graph.is_exceptional(source, target)
        ]
        self.assertTrue(exceptional)

    def test_an_exceptional_edge_is_distinguished_from_normal_flow(self):
        graph = self._script_graph("try { 'a' } catch { 'b' }")
        normal = [
            (source, target)
            for source in graph.nodes
            for target in source.successors
            if not graph.is_exceptional(source, target)
        ]
        self.assertTrue(normal)

    def test_a_trap_is_reachable_from_a_statement_written_above_it(self):
        # A trap catches for the whole body it is declared in, statements before it included, so the
        # edge is not the one a guarded block would produce.
        graph = self._script_graph("'a'\ntrap { continue }\n'b'")
        trap = self._node_for(graph, Ps1TrapStatement)
        self.assertTrue(trap.predecessors)

    def test_a_switch_clause_may_be_entered_after_an_earlier_one_ran(self):
        # PowerShell tests every clause and runs every match, so the first clause's body reaches the
        # second's. Reading it as C-style fallthrough would give the same edge; reading it as
        # exclusive would not.
        tree = Ps1Parser("switch ($x) { 1 { 'a' } 2 { 'b' } }").parse()
        graph = build_ps1_control_flow(tree)[id(tree)]
        head = graph.node_of(tree.body[0])
        self.assertIsNotNone(head)
        self.assertGreaterEqual(len(head.successors), 2)

    def test_a_switch_clause_may_be_skipped_when_a_later_one_matches(self):
        """
        The path in which the first and third clauses match and the second does not.
        """
        # Asserted as a *direct* edge rather than as reachability. Under C-style fallthrough the
        # third clause is reachable from the first anyway — through the second — so a reachability
        # assertion passes whichever answer the builder gives and measures nothing. The edge that
        # only PowerShell's semantics produce is the one that skips the clause in between.
        tree = Ps1Parser("switch ($x) { 1 { 'a' } 2 { 'b' } 3 { 'c' } }").parse()
        graph = build_ps1_control_flow(tree)[id(tree)]
        clauses = tree.body[0].clauses
        first = graph.node_of(clauses[0][1].body[0])
        third = graph.node_of(clauses[2][1].body[0])
        self.assertIsNotNone(first)
        self.assertIsNotNone(third)
        self.assertIn(third, first.successors)

    def test_an_advanced_function_body_is_not_empty(self):
        # `Ps1Code` fills begin/process/end instead of `body`, and reading only `body` would report
        # a graph with no statements for a function that runs a great deal.
        tree = Ps1Parser("function f { begin { 'a' } process { 'b' } end { 'c' } }").parse()
        definition = next(n for n in tree.walk() if isinstance(n, Ps1FunctionDefinition))
        graph = build_ps1_control_flow(tree)[id(definition)]
        self.assertGreater(len([n for n in graph.nodes if n.element is not None]), 2)

    def test_each_elseif_condition_is_its_own_point(self):
        tree = Ps1Parser("if ($x) { 'a' } elseif ($y) { 'b' } else { 'c' }").parse()
        graph = build_ps1_control_flow(tree)[id(tree)]
        self.assertIsNotNone(graph.node_of(tree.body[0]))
        self.assertIsNotNone(graph.node_of(tree.body[0].clauses[1][0]))

    def test_the_script_root_owns_a_graph(self):
        tree = Ps1Parser("'a'").parse()
        graphs = build_ps1_control_flow(tree)
        self.assertIsInstance(tree, Ps1Script)
        self.assertIn(id(tree), graphs)
