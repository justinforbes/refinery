from __future__ import annotations

from test import TestBase

from refinery.lib.scripts import Statement
from refinery.lib.scripts.analysis.cfg import CfgNode, ControlFlowGraph
from refinery.lib.scripts.analysis.cycles import CycleModel
from refinery.lib.scripts.analysis.dominance import DominatorModel
from refinery.lib.scripts.ps1.analysis.cfg import build_control_flow_model, build_ps1_control_flow
from refinery.lib.scripts.ps1.analysis.model import build_semantic_model
from refinery.lib.scripts.ps1.model import (
    Ps1BreakStatement,
    Ps1ContinueStatement,
    Ps1FunctionDefinition,
    Ps1Script,
    Ps1ScriptBlock,
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
    "switch ($x) { 1 { } 2 { } 3 { } }",
    "while ($x) { switch ($y) { 1 { continue } } }",
    "try { 'a' } catch [System.IO.IOException] { 'b' } catch { 'c' }",
    "trap { continue }\n'a'",
    "trap { 'h' }\ntrap [System.IO.IOException] { 'i' }\n'a'",
    "if ($c) { trap { 'h' }\n'a' }\n'b'",
    "function f { 'a' }\nf",
    "function f { return 1 }",
    "function f { dynamicparam { 'd' } begin { 'b' } process { 'p' } end { 'e' } }",
    "'a'\nexit\n'b'",
    "'a'\nthrow 'x'\n'b'",
    ":outer while ($x) { while ($y) { break outer } }",
    ":outer while ($x) { while ($y) { continue outer } }",
    "while ($x) { while ($y) { break } }\n'after'",
    "do { try { 'a' } catch { 'b' } } while ($x)",
    "for (;;) { 'a'\ncontinue }",
    "'a'",
    "1..3 | ForEach-Object { 'a' }",
    "1..3 | %{ 'a'\n'b' }",
    "& { while ($x) { 'a' } }",
    "$f = { 'a' }\n& $f",
    "1..3 | ForEach-Object { 1..3 | ForEach-Object { 'a' } }",
    "function f { 1..3 | ForEach-Object { 'a' } }",
    "1..3 | ForEach-Object { trap { 'h' }\n'a' }",
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

    def _tree_and_graph(self, source: str) -> tuple[Ps1Script, ControlFlowGraph]:
        tree = Ps1Parser(source).parse()
        return tree, build_ps1_control_flow(tree)[id(tree)]

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
        self.assertIsNot(graphs[id(definition.body)], script)

    def test_a_while_loop_has_a_back_edge(self):
        tree = Ps1Parser("while ($x) { 'a' }").parse()
        graph = build_ps1_control_flow(tree)[id(tree)]
        head = graph.node_of(tree.body[0])
        self.assertIsNotNone(head)
        body = next(n for n in graph.nodes if n.element is not None and n is not head)
        self.assertIn(id(head), self._forward(body))

    def test_break_leaves_the_loop_and_does_not_return_to_its_head(self):
        # Asserted against the statement *after* the loop rather than against the graph exit. A
        # break that never resolved to its target is wired to the exit as a fallback, and with the
        # loop written last the two answers coincide, so the exit assertion cannot discriminate.
        tree, graph = self._tree_and_graph("while ($x) { break }\n'after'")
        jump = self._node_for(graph, Ps1BreakStatement)
        self.assertIn(graph.node_of(tree.body[1]), jump.successors)
        self.assertNotIn(graph.node_of(tree.body[0]), jump.successors)
        self.assertNotIn(graph.exit, jump.successors)

    def test_an_unlabelled_break_leaves_only_the_innermost_loop(self):
        tree, graph = self._tree_and_graph("while ($x) { while ($y) { break } }\n'after'")
        jump = self._node_for(graph, Ps1BreakStatement)
        inner = tree.body[0].body.body[0]
        self.assertNotIn(graph.node_of(tree.body[1]), jump.successors)
        self.assertIn(graph.node_of(tree.body[0]), jump.successors)
        self.assertNotIn(graph.node_of(inner), jump.successors)

    def test_continue_returns_to_the_loop_head(self):
        tree, graph = self._tree_and_graph("while ($x) { continue }")
        jump = self._node_for(graph, Ps1ContinueStatement)
        self.assertEqual(jump.successors, [graph.node_of(tree.body[0])])

    def test_a_labelled_break_leaves_the_named_loop(self):
        tree, graph = self._tree_and_graph(
            ":outer while ($x) { while ($y) { break outer } }\n'after'")
        jump = self._node_for(graph, Ps1BreakStatement)
        self.assertIn(graph.node_of(tree.body[1]), jump.successors)
        self.assertNotIn(graph.exit, jump.successors)

    def test_a_labelled_continue_returns_to_the_named_loop_head(self):
        tree, graph = self._tree_and_graph(":outer while ($x) { while ($y) { continue outer } }")
        jump = self._node_for(graph, Ps1ContinueStatement)
        self.assertEqual(jump.successors, [graph.node_of(tree.body[0])])

    def test_a_labelled_break_leaves_the_named_switch(self):
        tree, graph = self._tree_and_graph(":sw switch ($x) { 1 { break sw } }\n'after'")
        jump = self._node_for(graph, Ps1BreakStatement)
        self.assertIn(graph.node_of(tree.body[1]), jump.successors)
        self.assertNotIn(graph.exit, jump.successors)

    def test_a_counted_loop_evaluates_its_initializer_condition_and_iterator_at_their_own_points(
        self,
    ):
        """
        The three parts run at three different points, and none of them is the loop statement.
        """
        # Asserted per part. The parts are expressions rather than statements, so the invariant that
        # every statement is represented does not reach them: a builder that dropped the initializer
        # would still produce a graph with a node for the loop body.
        tree, graph = self._tree_and_graph("for ($i = 0; $i -lt 3; $i++) { 'a' }")
        loop = tree.body[0]
        for part in (loop.initializer, loop.condition, loop.iterator):
            self.assertIsNotNone(graph.node_of(part))

    def test_a_counted_loop_without_test_or_update_still_has_a_back_edge_from_continue(self):
        tree, graph = self._tree_and_graph("for (;;) { 'a'\ncontinue }")
        jump = self._node_for(graph, Ps1ContinueStatement)
        body = tree.body[0].body.body[0]
        self.assertEqual(jump.successors, [graph.node_of(body)])

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
        """
        A trap catches for the whole body it is declared in, statements before it included.
        """
        # Asserted as an edge from that specific statement. The resumption fan-out gives the trap
        # node a predecessor whatever the handler's scope is, so a non-empty predecessor list holds
        # even when the handler is installed only from the point the trap is written.
        tree, graph = self._tree_and_graph("'a'\ntrap { continue }\n'b'")
        trap = self._node_for(graph, Ps1TrapStatement)
        self.assertIn(graph.node_of(tree.body[0]), trap.predecessors)

    def test_a_trap_declared_inside_a_nested_block_still_guards_the_whole_body(self):
        tree, graph = self._tree_and_graph("if ($c) { trap { 'h' }\n'a' }\n'b'")
        trap = self._node_for(graph, Ps1TrapStatement)
        self.assertIn(graph.node_of(tree.body[1]), trap.predecessors)
        self.assertIsNotNone(graph.node_of(trap.element.body.body[0]))

    def test_a_trap_that_continues_resumes_in_the_body_it_guards(self):
        """
        `continue` inside a trap resumes at the statement after the one that threw, so the guarded
        body is reachable from it — it is not a loop back-jump and does not leave the body.
        """
        tree, graph = self._tree_and_graph("trap { continue }\n'a'\n'b'")
        jump = self._node_for(graph, Ps1ContinueStatement)
        reached = self._forward(jump)
        self.assertIn(id(graph.node_of(tree.body[1])), reached)
        self.assertIn(id(graph.node_of(tree.body[2])), reached)

    def test_a_trap_does_not_guard_its_own_body(self):
        """
        An error raised inside a trap body leaves the scope; it does not re-enter the handler it was
        raised in. An exceptional edge back to that handler would close a cycle through it that no
        run can take, and every statement of every trap body would read as repeating.
        """
        tree, graph = self._tree_and_graph("trap { return }\n'a'")
        trap = self._node_for(graph, Ps1TrapStatement)
        inside = graph.node_of(tree.body[0].body.body[0])
        self.assertNotIn(trap, inside.successors)
        self.assertFalse(CycleModel(build_control_flow_model(tree)).repeats(tree.body[0].body.body[0]))

    def test_a_trap_still_guards_the_statements_it_is_declared_beside(self):
        # The floor under the test above: building the trap bodies outside the handler must not take
        # the guarded body out with them.
        tree, graph = self._tree_and_graph("trap { return }\n'a'")
        trap = self._node_for(graph, Ps1TrapStatement)
        self.assertIn(trap, graph.node_of(tree.body[1]).successors)

    def test_a_switch_clause_may_be_entered_after_the_previous_one_ran(self):
        # Asserted as a direct edge from the first clause's body to the second's. Under EXCLUSIVE
        # the head still reaches every arm, so counting the head's successors measures nothing;
        # only an arm-to-arm edge distinguishes the answers.
        tree, graph = self._tree_and_graph("switch ($x) { 1 { 'a' } 2 { 'b' } }")
        clauses = tree.body[0].clauses
        first = graph.node_of(clauses[0][1].body[0])
        second = graph.node_of(clauses[1][1].body[0])
        self.assertIn(second, first.successors)

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

    def test_continue_in_a_switch_clause_advances_to_the_next_input_value(self):
        """
        A PowerShell `switch` enumerates its input, so `continue` inside one re-enters the switch
        rather than the loop around it.
        """
        tree, graph = self._tree_and_graph("while ($x) { switch ($y) { 1 { continue } } }")
        jump = self._node_for(graph, Ps1ContinueStatement)
        switch = tree.body[0].body.body[0]
        self.assertEqual(jump.successors, [graph.node_of(switch)])

    def test_a_switch_with_a_default_clause_may_still_be_left_without_entering_an_arm(self):
        """
        Over an empty input collection no clause of a PowerShell `switch` runs, `default` included.
        """
        tree, graph = self._tree_and_graph("switch ($x) { 1 { 'a' } default { 'b' } }\n'c'")
        head = graph.node_of(tree.body[0])
        self.assertIn(graph.node_of(tree.body[1]), head.successors)

    def test_a_second_catch_clause_is_entered_only_on_the_exceptional_path(self):
        tree, graph = self._tree_and_graph("try { 'a' } catch [System.IO.IOException] { 'b' } "
                                           "catch { 'c' }")
        clauses = tree.body[0].catch_clauses
        first = graph.node_of(clauses[0])
        second = graph.node_of(clauses[1])
        self.assertIn(second, first.successors)
        self.assertTrue(graph.is_exceptional(first, second))

    def test_a_switch_of_empty_clauses_stays_linear_in_the_number_of_clauses(self):
        """
        The frontier a clause carries into the next one must not accumulate duplicates.
        """
        # An empty clause hands its incoming frontier back unchanged, so concatenating the two
        # doubles it per clause. The last clause is deliberately not empty: that is where the
        # accumulated frontier turns into edges, and without one the doubling stays invisible in a
        # list nobody counts. At twenty clauses an unguarded builder wires 2**20 of them.
        clauses = ' '.join(F'{index} {{ }}' for index in range(20))
        graph = self._script_graph(F"switch ($x) {{ {clauses} 20 {{ 'z' }} }}")
        edges = sum(len(node.successors) for node in graph.nodes)
        self.assertLess(edges, 200)

    def test_a_trap_that_continues_stays_linear_in_the_size_of_the_body_it_guards(self):
        """
        A `trap` that resumes reaches every statement of the body it guards, which is a relation of
        size `resumes * guarded` and must not be spelled as that many edges.
        """
        # Written as one edge per pair, a body of eighty statements costs thousands, and every
        # reachability walk over the graph pays it — one real sample reached half a million edges
        # and three minutes of deobfuscation. The bound sits far above the linear count and far
        # below the quadratic one, so it moves only if the shape regresses.
        body = ' '.join(F"'s{index}';" for index in range(80))
        graph = self._script_graph(F'trap {{ continue }}; {body}')
        edges = sum(len(node.successors) for node in graph.nodes)
        self.assertLess(edges, 500)

    def test_dynamicparam_runs_before_the_begin_block(self):
        tree = Ps1Parser("function f { dynamicparam { 'd' } begin { 'b' } end { 'e' } }").parse()
        definition = next(n for n in tree.walk() if isinstance(n, Ps1FunctionDefinition))
        graph = build_ps1_control_flow(tree)[id(definition.body)]
        code = definition.body
        dynamic = graph.node_of(code.dynamicparam_block.body[0])
        begin = graph.node_of(code.begin_block.body[0])
        self.assertIn(id(begin), self._forward(dynamic))
        self.assertNotIn(id(dynamic), self._forward(begin))

    def test_every_statement_is_represented_by_a_node_or_by_one_of_its_parts(self):
        """
        A construct whose body the builder drops leaves no node behind, and a graph that never
        created a node is not the same as one whose node is unreachable — no edge invariant sees it.
        """
        for source in _CORPUS:
            with self.subTest(source):
                tree = Ps1Parser(source).parse()
                graphs = build_ps1_control_flow(tree)
                held = {
                    id(node.element)
                    for graph in graphs.values()
                    for node in graph.nodes
                    if node.element is not None
                }
                for statement in tree.walk():
                    if statement is tree or not isinstance(statement, Statement):
                        continue
                    self.assertTrue(
                        any(id(part) in held for part in statement.walk()),
                        F'no node stands for {type(statement).__name__} or any part of it',
                    )

    def test_an_advanced_function_body_is_not_empty(self):
        # `Ps1Code` fills begin/process/end instead of `body`, and reading only `body` would report
        # a graph with no statements for a function that runs a great deal.
        tree = Ps1Parser("function f { begin { 'a' } process { 'b' } end { 'c' } }").parse()
        definition = next(n for n in tree.walk() if isinstance(n, Ps1FunctionDefinition))
        graph = build_ps1_control_flow(tree)[id(definition.body)]
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

    def test_the_bodies_that_own_a_graph_are_the_bodies_that_own_a_scope(self):
        """
        A body is one unit for both layers: the same braces introduce a scope and a graph.
        """
        for source in _CORPUS:
            with self.subTest(source):
                tree = Ps1Parser(source).parse()
                graphs = build_ps1_control_flow(tree)
                model = build_semantic_model(tree)
                scopes = [model.root_scope]
                owners: set[int] = set()
                while scopes:
                    scope = scopes.pop()
                    owners.add(id(scope.node))
                    scopes.extend(scope.children)
                self.assertEqual(set(graphs), owners)

    def test_a_script_block_owns_its_own_graph(self):
        tree = Ps1Parser("1..3 | ForEach-Object { 'a' }").parse()
        graphs = build_ps1_control_flow(tree)
        block = next(n for n in tree.walk_in_order() if isinstance(n, Ps1ScriptBlock))
        self.assertIn(id(block), graphs)
        self.assertIsNotNone(graphs[id(block)].node_of(block.body[0]))

    def test_a_statement_in_a_script_block_is_not_ordered_against_one_outside_it(self):
        """
        A block is a value where it is written and runs when it is invoked, so nothing in it is
        guaranteed to have run by the time the statement after the block does. Asserted in both
        directions and against a statement *before* the block as well: a model that placed the
        block's statements on the enclosing graph would order them against the code after it,
        which is the claim that licenses carrying a value across the block.
        """
        tree = Ps1Parser("$a = 1\n1..3 | ForEach-Object { $b = 2 }\n$c = 3").parse()
        dominators = DominatorModel(build_control_flow_model(tree))
        block = next(n for n in tree.walk_in_order() if isinstance(n, Ps1ScriptBlock))
        inside = block.body[0]
        self.assertFalse(dominators.dominates(inside, tree.body[2]))
        self.assertFalse(dominators.dominates(tree.body[0], inside))

    def test_two_statements_in_one_script_block_are_ordered_against_each_other(self):
        """
        The counterpart of the test above: refusing to place the block's statements at all would
        satisfy that one while losing every ordering the block does have.
        """
        tree = Ps1Parser("1..3 | ForEach-Object { $b = 2\n$c = 3 }").parse()
        dominators = DominatorModel(build_control_flow_model(tree))
        block = next(n for n in tree.walk_in_order() if isinstance(n, Ps1ScriptBlock))
        self.assertTrue(dominators.dominates(block.body[0], block.body[1]))
        self.assertFalse(dominators.dominates(block.body[1], block.body[0]))

    def test_a_switch_arm_can_be_reached_again_because_the_switch_enumerates_its_input(self):
        """
        `switch ($array)` runs its clauses once per element, so a store in an arm is not a store
        that happens once.
        """
        tree = Ps1Parser("switch ($x) { 1 { 'a' } }").parse()
        cycles = CycleModel(build_control_flow_model(tree))
        self.assertTrue(cycles.repeats(tree.body[0].clauses[0][1].body[0]))

    def test_a_branch_of_an_if_is_not_reached_again(self):
        """
        The floor under the switch test above: a model that called every arm of every multi-way
        construct repeated would satisfy it while saying nothing about iteration.
        """
        tree = Ps1Parser("if ($x) { 'a' } else { 'b' }").parse()
        cycles = CycleModel(build_control_flow_model(tree))
        self.assertFalse(cycles.repeats(tree.body[0].clauses[0][1].body[0]))
        self.assertFalse(cycles.repeats(tree.body[0].else_block.body[0]))

    def test_a_for_loop_initializer_does_not_repeat_with_the_body_it_precedes(self):
        """
        Being written inside the `for` is not being on its cycle: the initializer runs once, before
        the head, and only the condition, iterator and body are reached again.
        """
        tree = Ps1Parser("for ($q = 0; $q -lt 3; $q++) { 'a' }").parse()
        cycles = CycleModel(build_control_flow_model(tree))
        loop = tree.body[0]
        self.assertFalse(cycles.repeats(loop.initializer))
        self.assertTrue(cycles.repeats(loop.condition))
        self.assertTrue(cycles.repeats(loop.iterator))
        self.assertTrue(cycles.repeats(loop.body.body[0]))

    def test_a_script_block_written_inside_a_loop_repeats_with_the_loop(self):
        """
        The block's own graph is acyclic; the repetition belongs to the body it is written in.
        """
        tree = Ps1Parser("while ($x) { 1..3 | ForEach-Object { $a = 1 } }").parse()
        cycles = CycleModel(build_control_flow_model(tree))
        block = next(n for n in tree.walk_in_order() if isinstance(n, Ps1ScriptBlock))
        self.assertTrue(cycles.repeats(block.body[0]))

    def test_a_script_block_written_outside_a_loop_does_not_repeat(self):
        tree = Ps1Parser("& { $a = 1 }").parse()
        cycles = CycleModel(build_control_flow_model(tree))
        block = next(n for n in tree.walk_in_order() if isinstance(n, Ps1ScriptBlock))
        self.assertFalse(cycles.repeats(block.body[0]))

    def test_a_parameter_default_is_not_placed_in_the_body_around_its_block(self):
        """
        It is evaluated when the block is invoked, which is not where the block is written.
        """
        tree = Ps1Parser("$a = 1\n$f = { param($p = 2) $p }\n$c = 3").parse()
        flow = build_control_flow_model(tree)
        block = next(n for n in tree.walk_in_order() if isinstance(n, Ps1ScriptBlock))
        default = block.param_block.parameters[0].default_value
        self.assertIsNotNone(default)
        self.assertIsNone(flow.locate(default))

    def test_an_element_the_graphs_do_not_place_is_reported_as_repeating(self):
        """
        A parameter default is evaluated once per invocation of its block, and the graphs hold no
        invocation. Reporting it as running once would be a claim these graphs cannot support, and
        a caller asking whether a single-visit fact still holds would act on it.
        """
        tree = Ps1Parser("while ($c) { & { param($p = ($x = 1)) $p } }").parse()
        flow = build_control_flow_model(tree)
        cycles = CycleModel(flow)
        block = next(n for n in tree.walk_in_order() if isinstance(n, Ps1ScriptBlock))
        default = block.param_block.parameters[0].default_value
        self.assertIsNone(flow.locate(default))
        self.assertTrue(cycles.repeats(default))

    def test_a_function_body_repeats_with_the_loop_its_definition_is_written_in(self):
        """
        The other spelling of the owner walk: a function body's graph is reached from its
        definition, which is a statement of the enclosing graph, where a bare block is reached from
        the statement that merely mentions it.
        """
        tree = Ps1Parser("while ($x) { function f { $a = 1 } }").parse()
        cycles = CycleModel(build_control_flow_model(tree))
        definition = next(n for n in tree.walk() if isinstance(n, Ps1FunctionDefinition))
        self.assertTrue(cycles.repeats(definition.body.body[0]))

    def test_a_function_body_outside_a_loop_does_not_repeat(self):
        tree = Ps1Parser("function f { $a = 1 }").parse()
        cycles = CycleModel(build_control_flow_model(tree))
        definition = next(n for n in tree.walk() if isinstance(n, Ps1FunctionDefinition))
        self.assertFalse(cycles.repeats(definition.body.body[0]))

    def test_a_switch_whose_only_arm_is_empty_still_returns_to_its_head(self):
        """
        The head is then one node reaching only itself, which a component-size test reads as
        acyclic.
        """
        tree = Ps1Parser("switch ($x) { 1 { } }").parse()
        cycles = CycleModel(build_control_flow_model(tree))
        self.assertTrue(cycles.repeats(tree.body[0]))
