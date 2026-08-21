from __future__ import annotations

from inspect import cleandoc

from test import TestBase

from refinery.lib.scripts import Statement
from refinery.lib.scripts.ps1.analysis.cfg import build_control_flow_model
from refinery.lib.scripts.ps1.analysis.faults import Ps1FaultReach, build_fault_reach, handler_acts
from refinery.lib.scripts.ps1.ast import get_body
from refinery.lib.scripts.ps1.model import Ps1Script, Ps1TrapStatement
from refinery.lib.scripts.ps1.parser import Ps1Parser


def _parse(source: str) -> Ps1Script:
    return Ps1Parser(cleandoc(source)).parse()


def _model(source: str) -> tuple[Ps1Script, Ps1FaultReach]:
    tree = _parse(source)
    return tree, build_fault_reach(build_control_flow_model(tree))


class TestPs1FaultRoutingOffersTheRaiseToTheHandlersThatGuardIt(TestBase):

    def test_a_raise_in_a_try_block_is_offered_to_that_constructs_catch_clause(self):
        tree, reach = _model("try { 'a' } catch { 'b' }")
        construct = tree.body[0]
        clause, = construct.catch_clauses
        self.assertEqual(reach.routing_at(construct.try_block.body[0]).handlers, (clause,))

    def test_a_raise_after_the_construct_is_offered_to_no_catch_clause(self):
        tree, reach = _model("""
            try { 'a' } catch { 'b' }
            'c'
        """)
        self.assertEqual(reach.routing_at(tree.body[1]).handlers, ())

    def test_every_catch_clause_of_the_construct_is_offered_the_raise(self):
        tree, reach = _model("try { 'a' } catch [System.IO.IOException] { 'x' } catch { 'y' }")
        construct = tree.body[0]
        self.assertEqual(
            set(reach.routing_at(construct.try_block.body[0]).handlers),
            set(construct.catch_clauses),
        )

    def test_a_trap_is_offered_a_raise_written_above_it(self):
        tree, reach = _model("""
            'a'
            trap { 'h' }
        """)
        self.assertEqual(reach.routing_at(tree.body[0]).handlers, (tree.body[1],))

    def test_a_trap_is_offered_a_raise_from_a_block_nested_in_the_one_it_guards(self):
        tree, reach = _model("""
            trap { 'h' }
            if ($c) { 'a' }
        """)
        nested = tree.body[1].clauses[0][1].body[0]
        self.assertEqual(reach.routing_at(nested).handlers, (tree.body[0],))

    def test_a_trap_is_not_offered_a_raise_written_outside_the_block_it_belongs_to(self):
        tree, reach = _model("""
            if ($c) {
                trap { 'h' }
                'a'
            }
            'b'
        """)
        self.assertEqual(reach.routing_at(tree.body[1]).handlers, ())

    def test_only_the_innermost_untyped_trap_is_offered_the_raise(self):
        tree, reach = _model("""
            trap { 'o' }
            if ($c) {
                trap { 'i' }
                'a'
            }
        """)
        guarded = tree.body[1].clauses[0][1]
        self.assertEqual(reach.routing_at(guarded.body[1]).handlers, (guarded.body[0],))

    def test_a_typed_innermost_trap_passes_the_raise_on_to_the_trap_around_it(self):
        tree, reach = _model("""
            trap [System.IO.IOException] { 'o' }
            if ($c) {
                trap [System.IO.IOException] { 'i' }
                'a'
            }
        """)
        guarded = tree.body[1].clauses[0][1]
        self.assertEqual(
            set(reach.routing_at(guarded.body[1]).handlers),
            {guarded.body[0], tree.body[0]},
        )

    def test_an_untyped_catch_clause_passes_the_raise_on_to_nothing(self):
        tree, reach = _model("""
            try {
                try { 'a' } catch { 'i' }
            } catch { 'o' }
        """)
        inner = tree.body[0].try_block.body[0]
        self.assertEqual(
            reach.routing_at(inner.try_block.body[0]).handlers,
            (inner.catch_clauses[0],),
        )

    def test_a_narrow_catch_clause_passes_the_raise_on_to_what_guards_the_construct(self):
        tree, reach = _model("""
            try {
                try { 'a' } catch [System.IO.IOException] { 'i' }
            } catch { 'o' }
        """)
        outer = tree.body[0]
        inner = outer.try_block.body[0]
        self.assertEqual(
            set(reach.routing_at(inner.try_block.body[0]).handlers),
            {inner.catch_clauses[0], outer.catch_clauses[0]},
        )


class TestPs1FaultLeavesTheBody(TestBase):

    def test_a_raise_that_nothing_guards_stays_in_the_body(self):
        tree, reach = _model("'a'")
        self.assertFalse(reach.leaves_the_body(tree.body[0]))

    def test_a_catch_clause_lets_the_raise_past_exactly_when_its_filter_is_narrower_than_exception(
        self,
    ):
        broad, broad_reach = _model("try { 'a' } catch [System.Exception] { 'b' }")
        narrow, narrow_reach = _model("try { 'a' } catch [System.IO.IOException] { 'b' }")
        self.assertFalse(broad_reach.leaves_the_body(broad.body[0].try_block.body[0]))
        self.assertTrue(narrow_reach.leaves_the_body(narrow.body[0].try_block.body[0]))

    def test_a_trap_lets_the_raise_past_exactly_when_it_carries_a_type_filter(self):
        untyped, untyped_reach = _model("""
            trap { 'h' }
            'a'
        """)
        typed, typed_reach = _model("""
            trap [System.IO.IOException] { 'h' }
            'a'
        """)
        self.assertFalse(untyped_reach.leaves_the_body(untyped.body[1]))
        self.assertTrue(typed_reach.leaves_the_body(typed.body[1]))

    def test_a_trap_lets_the_raise_past_when_it_breaks_but_not_when_it_continues(self):
        rethrowing, rethrowing_reach = _model("""
            trap { break }
            'a'
        """)
        resuming, resuming_reach = _model("""
            trap { continue }
            'a'
        """)
        self.assertTrue(rethrowing_reach.leaves_the_body(rethrowing.body[1]))
        self.assertFalse(resuming_reach.leaves_the_body(resuming.body[1]))

    def test_a_point_the_graphs_cannot_place_is_answered_as_leaving_the_body(self):
        _, reach = _model("'a'")
        elsewhere = _parse("'z'").body[0]
        self.assertIsNone(reach.routing_at(elsewhere))
        self.assertTrue(reach.leaves_the_body(elsewhere))


class TestPs1ATrapDisposesOfTheErrorWhateverConstructEnclosesItsBlock(TestBase):
    """
    A `trap` body is a scope of its own for `break` and `continue`, so what the set does with an
    error is decided by the body alone and not by the loop or `switch` the guarded block happens to
    be written inside. Measured on 5.1: a `trap { break }` written in a `while` body stops the
    script where the raise is, exactly as one written at script scope does, and a
    `trap { continue }` written there resumes at the statement after the raise on every iteration.
    """

    _SPELLINGS = [
        "trap { %s }\n'a'",
        "while ($c) {\n  trap { %s }\n  'a'\n}",
        "do {\n  trap { %s }\n  'a'\n} while ($c)",
        "for ($i = 0; $i -lt 3; $i++) {\n  trap { %s }\n  'a'\n}",
        "foreach ($i in $x) {\n  trap { %s }\n  'a'\n}",
        "switch ($x) {\n  1 {\n    trap { %s }\n    'a'\n  }\n}",
    ]

    def _trap_and_the_raise_it_guards(
        self, source: str
    ) -> tuple[Ps1FaultReach, Ps1TrapStatement, Statement]:
        tree, reach = _model(source)
        for node in tree.walk_in_order():
            if isinstance(node, Ps1TrapStatement):
                block = get_body(node.parent)
                if block is not None:
                    return reach, node, block[1]
        self.fail('no trap of this script is written in a statement block')

    def test_a_trap_that_breaks_lets_the_raise_past_wherever_its_block_is_written(self):
        for spelling in self._SPELLINGS:
            source = spelling % 'break'
            with self.subTest(source):
                reach, trap, raised = self._trap_and_the_raise_it_guards(source)
                self.assertEqual(reach.routing_at(raised).handlers, (trap,))
                self.assertTrue(reach.leaves_the_body(raised))
                self.assertTrue(reach.observed_at(raised))

    def test_a_trap_that_continues_keeps_the_raise_wherever_its_block_is_written(self):
        for spelling in self._SPELLINGS:
            source = spelling % 'continue'
            with self.subTest(source):
                reach, trap, raised = self._trap_and_the_raise_it_guards(source)
                self.assertEqual(reach.routing_at(raised).handlers, (trap,))
                self.assertFalse(reach.leaves_the_body(raised))
                self.assertFalse(reach.observed_at(raised))


class TestPs1HandlerActs(TestBase):

    def test_a_catch_clause_acts_exactly_when_its_body_is_not_empty(self):
        acting, = _parse("try { 'a' } catch { 'b' }").body[0].catch_clauses
        swallowing, = _parse("try { 'a' } catch { }").body[0].catch_clauses
        self.assertTrue(handler_acts(acting))
        self.assertFalse(handler_acts(swallowing))

    def test_a_trap_holding_only_an_unlabelled_jump_does_not_act(self):
        self.assertFalse(handler_acts(_parse('trap { break }').body[0]))
        self.assertFalse(handler_acts(_parse('trap { continue }').body[0]))

    def test_a_trap_holding_a_labelled_jump_acts(self):
        self.assertTrue(handler_acts(_parse('trap { break outer }').body[0]))
        self.assertTrue(handler_acts(_parse('trap { continue outer }').body[0]))

    def test_a_trap_whose_body_is_a_bare_value_acts(self):
        self.assertTrue(handler_acts(_parse('trap { 5 }').body[0]))


class TestPs1FaultIsObservedWhereAHandlerActsOrATrapMayDecline(TestBase):

    def test_a_raise_that_nothing_guards_is_not_observed(self):
        tree, reach = _model("'a'")
        self.assertFalse(reach.observed_at(tree.body[0]))

    def test_a_raise_is_observed_by_a_catch_clause_with_a_body_but_not_by_an_empty_one(self):
        acting, acting_reach = _model("try { 'a' } catch { 'b' }")
        swallowing, swallowing_reach = _model("try { 'a' } catch { }")
        self.assertTrue(acting_reach.observed_at(acting.body[0].try_block.body[0]))
        self.assertFalse(swallowing_reach.observed_at(swallowing.body[0].try_block.body[0]))

    def test_a_raise_guarded_by_a_trap_that_only_continues_is_not_observed(self):
        tree, reach = _model("""
            trap { continue }
            'a'
        """)
        self.assertFalse(reach.observed_at(tree.body[1]))

    def test_a_raise_guarded_by_a_trap_that_only_breaks_is_observed_although_it_does_not_act(self):
        tree, reach = _model("""
            trap { break }
            'a'
        """)
        self.assertFalse(handler_acts(tree.body[0]))
        self.assertTrue(reach.observed_at(tree.body[1]))

    def test_a_raise_guarded_by_a_typed_trap_is_observed_because_the_set_may_decline_it(self):
        tree, reach = _model("""
            trap [System.IO.IOException] { continue }
            'a'
        """)
        self.assertFalse(handler_acts(tree.body[0]))
        self.assertTrue(reach.observed_at(tree.body[1]))

    def test_a_raise_guarded_by_a_trap_whose_body_is_a_bare_value_is_observed(self):
        tree, reach = _model("""
            trap { 5 }
            'a'
        """)
        self.assertTrue(reach.observed_at(tree.body[1]))

    def test_a_point_the_graphs_cannot_place_is_answered_as_observed(self):
        _, reach = _model("'a'")
        elsewhere = _parse("'z'").body[0]
        self.assertIsNone(reach.routing_at(elsewhere))
        self.assertTrue(reach.observed_at(elsewhere))


class TestPs1FaultPointsIn(TestBase):

    def test_a_simple_statement_is_the_only_point_it_holds(self):
        tree, reach = _model("'a'")
        self.assertEqual(list(reach.points_in(tree.body[0])), [tree.body[0]])

    def test_a_while_loop_is_a_point_of_its_own_beside_each_statement_of_its_body(self):
        tree, reach = _model("""
            while ($c) {
                'a'
                'b'
            }
        """)
        loop = tree.body[0]
        self.assertEqual(set(reach.points_in(loop)), {loop, *loop.body.body})

    def test_a_counted_loop_holds_its_three_parts_and_its_body_rather_than_itself(self):
        tree, reach = _model("for ($i = 0; $i -lt 3; $i++) { 'a' }")
        loop = tree.body[0]
        self.assertIsNone(reach.routing_at(loop))
        self.assertEqual(set(reach.points_in(loop)), {
            loop.initializer,
            loop.condition,
            loop.iterator,
            loop.body.body[0],
        })

    def test_an_if_chain_holds_a_point_per_test_beside_the_statements_of_its_branches(self):
        tree, reach = _model("if ($c) { 'a' } elseif ($d) { 'b' } else { 'e' }")
        branch = tree.body[0]
        (_, first), (second_test, second) = branch.clauses
        self.assertEqual(set(reach.points_in(branch)), {
            branch,
            second_test,
            first.body[0],
            second.body[0],
            branch.else_block.body[0],
        })

    def test_a_try_construct_holds_its_clauses_and_blocks_rather_than_itself(self):
        tree, reach = _model("try { 'a' } catch { 'b' } finally { 'c' }")
        construct = tree.body[0]
        clause, = construct.catch_clauses
        self.assertIsNone(reach.routing_at(construct))
        self.assertEqual(set(reach.points_in(construct)), {
            construct.try_block.body[0],
            clause,
            clause.body.body[0],
            construct.finally_block,
            construct.finally_block.body[0],
        })

    def test_a_construct_holds_the_points_of_a_construct_nested_in_it(self):
        tree, reach = _model("""
            try {
                if ($c) { 'a' }
            } catch { 'b' }
        """)
        construct = tree.body[0]
        clause, = construct.catch_clauses
        branch = construct.try_block.body[0]
        self.assertEqual(set(reach.points_in(construct)), {
            branch,
            branch.clauses[0][1].body[0],
            clause,
            clause.body.body[0],
        })

    def test_a_subtree_the_graphs_model_nowhere_holds_no_points(self):
        _, reach = _model("'a'")
        elsewhere = _parse("'z'").body[0]
        self.assertEqual(list(reach.points_in(elsewhere)), [])


class TestPs1RemovingATrapIsJudgedByWhereItsErrorsWouldGoInstead(TestBase):

    def test_a_trap_matters_only_while_its_block_still_holds_something_that_may_raise(self):
        guarding, guarding_reach = _model("""
            trap { 'o' }
            if ($c) {
                trap { 'i' }
                'a'
            }
        """)
        alone, alone_reach = _model("""
            trap { 'o' }
            if ($c) {
                trap { 'i' }
            }
        """)
        self.assertTrue(guarding_reach.removing_a_handler_is_observed(
            guarding.body[1].clauses[0][1].body[0]))
        self.assertFalse(alone_reach.removing_a_handler_is_observed(
            alone.body[1].clauses[0][1].body[0]))

    def test_a_trap_at_script_scope_may_go_although_the_raise_it_guards_is_observed(self):
        tree, reach = _model("""
            trap { 'h' }
            'a'
        """)
        self.assertTrue(reach.observed_at(tree.body[1]))
        self.assertFalse(reach.removing_a_handler_is_observed(tree.body[0]))

    def test_a_trap_in_a_try_block_may_go_only_where_the_catch_clause_swallows(self):
        acting, acting_reach = _model("""
            try {
                trap { 'h' }
                'a'
            } catch { 'c' }
        """)
        swallowing, swallowing_reach = _model("""
            try {
                trap { 'h' }
                'a'
            } catch { }
        """)
        self.assertTrue(acting_reach.removing_a_handler_is_observed(
            acting.body[0].try_block.body[0]))
        self.assertFalse(swallowing_reach.removing_a_handler_is_observed(
            swallowing.body[0].try_block.body[0]))
