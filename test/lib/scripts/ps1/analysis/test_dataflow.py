from __future__ import annotations

from test import TestBase

from refinery.lib.scripts.analysis.cycles import CycleModel
from refinery.lib.scripts.ps1.analysis.blocks import build_block_model
from refinery.lib.scripts.ps1.analysis.cfg import build_control_flow_model
from refinery.lib.scripts.ps1.analysis.dataflow import Ps1FlowUnknown, build_variable_flow
from refinery.lib.scripts.ps1.analysis.model import (
    Ps1SemanticModel,
    build_semantic_model,
    is_write_occurrence,
)
from refinery.lib.scripts.ps1.model import Ps1Variable
from refinery.lib.scripts.ps1.parser import Ps1Parser


def _models(source: str):
    tree = Ps1Parser(source).parse()
    semantic = build_semantic_model(tree)
    control = build_control_flow_model(tree)
    blocks = build_block_model(tree)
    flow = build_variable_flow(semantic, control, blocks, CycleModel(control, blocks.body_site))
    return tree, semantic, flow


def _in_source_order(node):
    """
    Pre-order over `children()`, which is source order. `Node.walk` is driven by a stack and comes
    back reversed, and `Ps1Variable.offset` is never set by the parser, so neither orders occurrences.
    """
    yield node
    for child in node.children():
        yield from _in_source_order(child)


class TestPs1VariableFlow(TestBase):
    """
    Which write each read observes, named by its position among the writes of that variable rather
    than by its value — two writes may carry the same text, and the question is which occurrence.
    Expectations are what PowerShell does, not what the ordering rules imply.
    """

    def _flow(self, source: str, name: str = 'x'):
        tree, _, flow = _models(source)
        occurrences = [
            node for node in _in_source_order(tree)
            if isinstance(node, Ps1Variable) and node.name.lower() == name
        ]
        writes = [node for node in occurrences if is_write_occurrence(node)]
        reads = [node for node in occurrences if not is_write_occurrence(node)]
        return flow, writes, reads

    def _observed(self, source: str, name: str = 'x', read: int = -1) -> int | None:
        """
        The index among the writes of *name* of the one the *read*-th read observes, or `None`.
        """
        flow, writes, reads = self._flow(source, name)
        found = flow.reaching_definition(reads[read])
        if found is None:
            return None
        return next(index for index, write in enumerate(writes) if write is found)

    def test_the_only_write_before_a_read_is_the_one_observed(self):
        self.assertEqual(self._observed("$x = 'a'; Write-Host $x"), 0)

    def test_the_nearest_of_two_writes_is_the_one_observed(self):
        self.assertEqual(self._observed("$x = 'a'; $x = 'b'; Write-Host $x"), 1)

    def test_a_write_in_a_branch_leaves_the_read_after_it_with_no_single_write(self):
        """
        The reproduction the two-table split produced: the pass this replaces kept `'a'` here while
        correctly refusing the identical shape with a non-constant value. Both must refuse.
        """
        self.assertIsNone(self._observed("$x = 'a'; if ($c) { $x = 'b' }; Write-Host $x"))
        self.assertIsNone(self._observed("$x = 'a'; if ($c) { $x = $y }; Write-Host $x"))

    def test_a_branch_that_does_not_write_leaves_the_write_before_it_standing(self):
        """
        The floor under the branch rule: refusing on any branch at all would pass the test above and
        inline nothing anywhere.
        """
        self.assertEqual(self._observed("$x = 'a'; if ($c) { Write-Host 1 }; Write-Host $x"), 0)

    def test_a_dotted_block_writing_the_name_kills_the_write_before_it(self):
        self.assertIsNone(self._observed("$x = 'a'; . { $x = 'b' }; Write-Host $x"))

    def test_an_ampersand_block_writing_the_name_does_not_kill_it(self):
        """
        The floor under the block work, and the case a scope model must not over-approximate: `&`
        opens a child scope, so the caller's `$x` is still `'a'` and PowerShell prints `a`.
        """
        self.assertEqual(self._observed("$x = 'a'; & { $x = 'b' }; Write-Host $x"), 0)

    def test_a_read_inside_a_stored_block_observes_nothing(self):
        """
        `$b` may be invoked at any time, including after `$x = 'b'`, so the read inside it is not
        ordered against the script that wrote it.
        """
        self.assertIsNone(self._observed("$x = 'a'; $b = { Write-Host $x }; $x = 'b'; & $b"))

    def test_a_write_a_loop_returns_to_leaves_the_read_with_no_single_write(self):
        self.assertIsNone(self._observed("$x = 'a'; while ($c) { Write-Host $x; $x = 'b' }"))

    def test_a_loop_that_only_reads_leaves_the_write_before_it_standing(self):
        self.assertEqual(self._observed("$x = 'a'; while ($c) { Write-Host $x }"), 0)

    def test_a_store_that_may_not_have_completed_is_not_observed_in_the_handler(self):
        """
        Dominance says the statement ran, not that its store landed: the cast raises before the
        assignment takes effect, so the handler must not be told `$x` holds `'abc'`.
        """
        self.assertIsNone(self._observed("try { [int]$x = 'abc' } catch { Write-Host $x }"))

    def test_a_write_inside_a_trap_is_never_what_a_later_read_observes(self):
        """
        5.1 runs a trap body in a child scope, so the enclosing `$x` is untouched by it. The semantic
        model binds the two together, and the exceptional-entry rule is what stops the write escaping.
        """
        self.assertIsNone(
            self._observed("$x = 'a'; trap { $x = 'b'; continue }; throw 'e'; Write-Host $x"))

    def test_a_foreach_object_body_writing_the_name_kills_it(self):
        self.assertIsNone(
            self._observed("$x = 'a'; 1..3 | ForEach-Object { $x = 'b' }; Write-Host $x"))

    def test_a_handler_observes_its_own_completed_store(self):
        """
        The floor under the partial-store rule, and the case a handler-keyed rule gets wrong: the
        store here is in the handler and completes, so its own read does see it.
        """
        self.assertEqual(self._observed("try { f } catch { $x = 'b'; Write-Host $x }"), 0)

    def test_a_read_inside_a_body_is_answered_at_the_statement_that_runs_the_body(self):
        """
        The read is one graph in and every write is in the script's, and each of these bodies runs
        exactly where it is written, so the statement running it is a point both can be ordered
        against. Refusing them costs most of what obfuscated PowerShell reads through.
        """
        for source in [
            "$x = 'a'; 1..3 | ForEach-Object { Write-Host $x }",
            "$x = 'a'; . { Write-Host $x }",
            "$x = 'a'; & { Write-Host $x }",
            "$x = 'a'; & { & { Write-Host $x } }",
        ]:
            with self.subTest(source):
                self.assertEqual(self._observed(source), 0)

    def test_a_read_inside_a_body_observes_the_writes_the_site_is_ordered_against(self):
        for source, expected in [
            ("$x = 'a'; & { Write-Host $x }; $x = 'b'", 0),
            ("$x = 'a'; $x = 'b'; & { Write-Host $x }", 1),
            ("$x = 'a'; if ($c) { $x = 'b' }; & { Write-Host $x }", None),
        ]:
            with self.subTest(source):
                self.assertEqual(self._observed(source), expected)

    def test_a_read_inside_a_body_a_call_site_reaches_is_not_placed_where_it_is_written(self):
        """
        The floor under the projection, and what separates a body that runs where it stands from one
        that does not: `f` and `$b` may run at any time, including after the second write.
        """
        for source in [
            "$x = 'a'; function f { Write-Host $x }; $x = 'b'; f",
            "$x = 'a'; $b = { Write-Host $x }; $x = 'b'; & $b",
            "$x = 'a'; Invoke-Command { Write-Host $x }; $x = 'b'",
        ]:
            with self.subTest(source):
                self.assertIsNone(self._observed(source, read=0))

    def test_a_body_running_before_a_write_in_its_own_statement_observes_the_earlier_one(self):
        """
        Evaluation order reaches into the body: the block is the first element, so it runs before the
        assignment beside it and reads what stood before the statement.
        """
        self.assertEqual(self._observed("$x = 'a'; (& { $x }), ($x = 'b')"), 0)

    def test_a_read_beside_the_writes_is_answered_though_a_function_body_reads_the_name_too(self):
        """
        The floor under the cross-body refusal: `f` may run at any time, so the read inside it is
        unanswerable, but the one beside the write is not and refusing it inlines nothing in any
        script that defines a function.
        """
        self.assertEqual(
            self._observed("$x = 'a'; function f { Write-Host $x }; Write-Host $x"), 0)

    def test_a_read_and_its_writes_inside_one_block_are_ordered_normally(self):
        """
        The floor under the cross-body rule. Refusing every read inside a block would pass the stored
        block test above while answering nothing anywhere.
        """
        self.assertEqual(self._observed("$b = { $x = 'a'; Write-Host $x }"), 0)

    def test_a_write_the_graphs_do_not_place_is_refused_rather_than_ignored(self):
        """
        A `param` declaration is evaluated when the function is invoked, which is not a point these
        graphs hold. Ignoring it would leave the default as the only write in sight and publish it
        for every call that passed an argument.
        """
        self.assertIsNone(
            self._observed("function f { param($x = 'a') Write-Host $x }"))

    def test_a_write_the_same_statement_stores_after_the_read_is_not_what_it_observes(self):
        """
        `$x + 'b'` is produced before it is stored, so the read is of the value from before the
        statement. The graphs place both occurrences at one node and can order neither.
        """
        self.assertEqual(self._observed("$x = 'a'; $x = $x + 'b'; Write-Host $x", read=0), 0)

    def test_a_target_written_left_of_the_read_still_stores_after_it(self):
        """
        The case source position answers backwards: the target of `$x = [char]($x)` is written first
        and stored last, so a rule reading the statement left to right calls the read stale.
        """
        self.assertEqual(self._observed('$x = 39; $x = [char]($x)', read=0), 0)

    def test_an_argument_evaluated_before_a_later_assignment_observes_the_earlier_write(self):
        self.assertEqual(self._observed("$x = 'old'; Write-Host $x ($x = 'new')", read=0), 0)

    def test_a_write_the_same_statement_stores_before_the_read_leaves_it_unanswered(self):
        """
        The floor under the rule above, and what stops it becoming *any* write sharing a node is
        ignored: here `'new'` is stored first, so the read sees neither write for certain.
        """
        self.assertIsNone(self._observed("$x = 'old'; Write-Host ($x = 'new') $x"))

    def test_a_self_referencing_write_control_returns_to_observes_no_single_value(self):
        """
        The floor under evaluation order: it orders one visit to a statement, and a second visit
        reads what the first stored, so the write is no longer merely later than the read.
        """
        self.assertIsNone(self._observed("$x = 'a'; while ($c) { $x = $x + 'b' }", read=0))

    def test_a_multi_assignment_target_is_a_write_like_any_other(self):
        """
        That `$x, $y = 1, 2` writes `$x` is this layer's question; what the write is *worth* is the
        caller's, and conflating the two is how `$x` came to inline to the whole list.
        """
        self.assertEqual(self._observed("$x, $y = 1, 2; Write-Host $x"), 0)


class TestPs1FlowUnknowns(TestBase):

    def _unknowns(self, source: str, name: str = 'x') -> Ps1FlowUnknown:
        _, semantic, flow = _models(source)
        return flow.unknowns(semantic.script_scope.bindings[name])

    def test_a_binding_written_and_read_in_one_body_has_no_unknowns(self):
        self.assertIs(self._unknowns("$x = 'a'; Write-Host $x"), Ps1FlowUnknown.NONE)

    def test_writes_in_two_bodies_leave_the_binding_unorderable(self):
        """
        A qualified write reaches the script scope from inside a block, so this binding is written
        at two points no graph orders against each other.
        """
        self.assertIn(
            Ps1FlowUnknown.WRITES_IN_SEVERAL_BODIES,
            self._unknowns("$x = 'a'; & { $script:x = 'b' }"))

    def test_a_read_in_another_body_leaves_the_binding_itself_answerable(self):
        """
        The floor under the rule above, and the difference between refusing a read and refusing a
        name: a function body mentioning `$x` says nothing about the writes of `$x`, and treating it
        as an unknown of the binding refuses every read of the name anywhere.
        """
        self.assertIs(
            self._unknowns("$x = 'a'; function f { Write-Host $x }; Write-Host $x"),
            Ps1FlowUnknown.NONE)

    def test_a_qualified_reader_is_reported_separately_from_an_unorderable_write(self):
        """
        The reason this is a flag: a caller may be able to live with one of these and not the other,
        and a single boolean would make them indistinguishable.
        """
        found = self._unknowns("$x = 'a'; Write-Host $script:x")
        self.assertIn(Ps1FlowUnknown.REACHED_BY_QUALIFIER, found)
        self.assertNotIn(Ps1FlowUnknown.WRITES_IN_SEVERAL_BODIES, found)

    def test_a_stored_block_writing_the_name_defers_the_binding(self):
        self.assertIn(
            Ps1FlowUnknown.WRITTEN_BY_DEFERRED_BODY,
            self._unknowns("$x = 'a'; $b = { $x = 'b' }"))

    def test_an_index_assignment_changes_the_value_without_writing_the_name(self):
        """
        `$x[0] = 'z'` writes no occurrence of `$x` — the name is read to reach the slot — so every
        occurrence lands in `reads` and the value the binding holds changes with nothing to order.
        """
        self.assertIn(
            Ps1FlowUnknown.MUTATED_IN_PLACE,
            self._unknowns("$x = @('a', 'b'); $x[0] = 'z'"))

    def test_a_member_assignment_changes_the_value_without_writing_the_name(self):
        self.assertIn(
            Ps1FlowUnknown.MUTATED_IN_PLACE,
            self._unknowns("$x = 'hello'; $x.Length = 5"))

    def test_reading_through_an_index_leaves_the_binding_trackable(self):
        """
        The floor: refusing every name an index expression mentions refuses every array this layer
        exists to resolve.
        """
        self.assertIs(
            self._unknowns("$x = @('a', 'b'); Write-Host $x[0]"), Ps1FlowUnknown.NONE)


class TestPs1VariableFlowRegressions(TestPs1VariableFlow):

    def test_a_use_the_throwing_path_also_reaches_observes_no_store(self):
        """
        The statement after a `try` is reached both by the body completing and by the handler it
        threw into, so dominance and a completed-exit walk on their own both accept it. The run that
        enters the handler is the run in which the cast raised and `$x` was never stored.
        """
        self.assertIsNone(self._observed("try { [int]$x = 'abc' } catch { }; Write-Host $x"))

    def test_a_statement_a_trap_resumes_into_observes_no_store(self):
        """
        The same asymmetry spelled as error handling: `continue` resumes at the statement after the
        one that threw, so that statement is reached with the store never performed.
        """
        self.assertIsNone(self._observed("trap { continue }; [int]$x = 'abc'; Write-Host $x"))

    def test_a_use_reached_only_by_completing_still_observes_the_store(self):
        """
        The floor under both tests above: refusing whenever the definition has an exceptional edge
        at all refuses every write inside a `try`.
        """
        self.assertEqual(self._observed("try { $x = 'a'; Write-Host $x } catch { }"), 0)


class TestPs1FlowUnknownRegressions(TestPs1FlowUnknowns):

    def test_a_stored_block_outside_the_binding_body_is_still_a_deferred_writer(self):
        """
        `. $b` performs the block's bare writes on whoever dot-sources it, so a block written at the
        root reaches a binding local to a body it is not written inside.
        """
        _, semantic, flow = _models(
            "$b = { $x = 'INNER' }\n. {\n  $x = 'OUTER'\n  . $b\n  Write-Host $x\n}")
        inner = next(
            scope for scope in semantic.script_scope.children if 'x' in scope.bindings)
        self.assertIn(
            Ps1FlowUnknown.WRITTEN_BY_DEFERRED_BODY,
            flow.unknowns(inner.bindings['x']))

    def test_a_receiver_chain_an_assignment_stores_through_mutates_the_binding_in_place(self):
        for source in (
            "$x = @(@('a','b'))\n$x[0][1] = 'z'",
            "$x = 'abc'\n$x.A.B = 5",
            "$x = @('a','b')\n($x)[0] = 'z'",
            "$x = @('a','b')\n$x[0], $x[1] = 'p', 'q'",
        ):
            with self.subTest(source):
                self.assertIn(Ps1FlowUnknown.MUTATED_IN_PLACE, self._unknowns(source))
