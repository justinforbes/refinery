from __future__ import annotations

from test import TestBase

from refinery.lib.scripts.ps1.analysis.blocks import build_block_model
from refinery.lib.scripts.ps1.analysis.cfg import build_control_flow_model
from refinery.lib.scripts.ps1.analysis.dataflow import Ps1FlowUnknown, build_variable_flow
from refinery.lib.scripts.ps1.analysis.model import build_semantic_model, is_write_occurrence
from refinery.lib.scripts.ps1.model import Ps1Variable
from refinery.lib.scripts.ps1.parser import Ps1Parser


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
        tree = Ps1Parser(source).parse()
        flow = build_variable_flow(
            build_semantic_model(tree),
            build_control_flow_model(tree),
            build_block_model(tree),
        )
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

    def test_a_multi_assignment_target_is_a_write_like_any_other(self):
        """
        That `$x, $y = 1, 2` writes `$x` is this layer's question; what the write is *worth* is the
        caller's, and conflating the two is how `$x` came to inline to the whole list.
        """
        self.assertEqual(self._observed("$x, $y = 1, 2; Write-Host $x"), 0)


class TestPs1FlowUnknowns(TestBase):

    def _unknowns(self, source: str, name: str = 'x') -> Ps1FlowUnknown:
        tree = Ps1Parser(source).parse()
        semantic = build_semantic_model(tree)
        flow = build_variable_flow(semantic, build_control_flow_model(tree), build_block_model(tree))
        binding = semantic.script_scope.bindings[name]
        return flow.unknowns(binding)

    def test_a_binding_written_and_read_in_one_body_has_no_unknowns(self):
        self.assertIs(self._unknowns("$x = 'a'; Write-Host $x"), Ps1FlowUnknown.NONE)

    def test_a_read_in_another_body_makes_the_binding_unorderable(self):
        self.assertIn(
            Ps1FlowUnknown.WRITE_IN_ANOTHER_BODY,
            self._unknowns("$x = 'a'; $b = { Write-Host $x }"))

    def test_a_qualified_reader_is_reported_separately_from_an_unorderable_write(self):
        """
        The reason this is a flag: a caller may be able to live with one of these and not the other,
        and a single boolean would make them indistinguishable.
        """
        found = self._unknowns("$x = 'a'; Write-Host $script:x")
        self.assertIn(Ps1FlowUnknown.REACHED_BY_QUALIFIER, found)
        self.assertNotIn(Ps1FlowUnknown.WRITE_IN_ANOTHER_BODY, found)

    def test_a_stored_block_writing_the_name_defers_the_binding(self):
        self.assertIn(
            Ps1FlowUnknown.WRITTEN_BY_DEFERRED_BODY,
            self._unknowns("$x = 'a'; $b = { $x = 'b' }"))
