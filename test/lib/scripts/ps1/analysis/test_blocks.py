from __future__ import annotations

from test import TestBase

from refinery.lib.scripts.analysis.cycles import CycleModel
from refinery.lib.scripts.ps1.analysis.blocks import (
    Ps1BlockIteration,
    Ps1BlockReach,
    Ps1BlockScope,
    build_block_model,
)
from refinery.lib.scripts.ps1.analysis.cfg import build_control_flow_model
from refinery.lib.scripts.ps1.model import Ps1ScriptBlock
from refinery.lib.scripts.ps1.parser import Ps1Parser


class TestPs1BlockFacts(TestBase):
    """
    Every expectation here is what Windows PowerShell 5.1 does, measured on a real host — never what
    the classifier's own rules imply.
    """

    def _facts(self, source: str, index: int = 0):
        tree = Ps1Parser(source).parse()
        blocks = build_block_model(tree)
        found = [node for node in tree.walk() if isinstance(node, Ps1ScriptBlock)]
        return blocks, found[index]

    def _scope(self, source: str, index: int = 0) -> Ps1BlockScope:
        blocks, block = self._facts(source, index)
        return blocks.facts(block).scope

    def _written_names(self, source: str, index: int = 0) -> list[str]:
        blocks, block = self._facts(source, index)
        return sorted(var.name for var in blocks.writes_reaching_caller(block))

    def test_a_dotted_block_writes_the_caller_and_an_ampersand_block_does_not(self):
        """
        The pair the whole question turns on: identical syntax, opposite answers, so neither can be
        read off the block itself.
        """
        self.assertIs(self._scope(". { $x = 'b' }"), Ps1BlockScope.CALLER)
        self.assertIs(self._scope("& { $x = 'b' }"), Ps1BlockScope.CHILD)
        self.assertEqual(self._written_names(". { $x = 'b' }"), ['x'])
        self.assertEqual(self._written_names("& { $x = 'b' }"), [])

    def test_a_function_body_opens_a_child_scope(self):
        self.assertIs(self._scope("function f { $x = 'b' }"), Ps1BlockScope.CHILD)
        self.assertEqual(self._written_names("function f { $x = 'b' }"), [])

    def test_a_child_scope_contains_a_dotted_block_nested_in_it(self):
        """
        The dot runs the inner body in the `&` block's scope, and that scope ends with the `&`, so
        the write never reaches the script.
        """
        self.assertEqual(self._written_names("& { . { $x = 'b' } }"), [])
        self.assertEqual(self._written_names(". { & { $x = 'b' } }"), [])
        self.assertEqual(self._written_names(". { . { $x = 'b' } }"), ['x'])

    def test_a_qualified_write_is_not_a_fact_about_where_the_block_runs(self):
        self.assertEqual(self._written_names(". { $script:x = 'b' }"), [])

    def test_a_foreach_object_body_writes_the_caller_and_runs_per_input(self):
        for source in (
            "1 | ForEach-Object { $x = 'b' }",
            "1 | % { $x = 'b' }",
            "1 | ForEach-Object -Process { $x = 'b' }",
        ):
            with self.subTest(source):
                blocks, block = self._facts(source)
                facts = blocks.facts(block)
                self.assertIs(facts.scope, Ps1BlockScope.CALLER)
                self.assertIs(facts.iteration, Ps1BlockIteration.REPEATED)
                self.assertEqual(self._written_names(source), ['x'])

    def test_a_where_object_body_writes_the_caller_and_runs_per_input(self):
        for source in (
            "1 | Where-Object { $x = 'b' }",
            "1 | ? { $x = 'b' }",
        ):
            with self.subTest(source):
                blocks, block = self._facts(source)
                facts = blocks.facts(block)
                self.assertIs(facts.scope, Ps1BlockScope.CALLER)
                self.assertIs(facts.iteration, Ps1BlockIteration.REPEATED)

    def test_a_block_a_command_merely_receives_is_not_claimed_to_run(self):
        """
        `f { }`, `Invoke-Command -ScriptBlock { }` and `ForEach-Object { }` are one AST shape, and
        only the last is known to invoke what it is handed.
        """
        for source in ("f { $x = 'b' }", "Invoke-Command -ScriptBlock { $x = 'b' }"):
            with self.subTest(source):
                blocks, block = self._facts(source)
                self.assertIs(blocks.facts(block).reach, Ps1BlockReach.UNKNOWN)

    def test_a_block_invoked_through_a_computed_name_is_not_claimed_to_run(self):
        blocks, block = self._facts("& $c { $x = 'b' }")
        self.assertIs(blocks.facts(block).reach, Ps1BlockReach.UNKNOWN)

    def test_a_block_that_is_only_a_value_is_never_invoked_where_it_is_written(self):
        for source in ("$b = { $x = 'b' }", "{ $x = 'b' } | Out-Null"):
            with self.subTest(source):
                blocks, block = self._facts(source)
                self.assertIs(blocks.facts(block).reach, Ps1BlockReach.STORED)

    def test_an_unknown_block_is_projected_as_writing_the_caller(self):
        """
        The asymmetry the module exists to hold: a kill nobody performs loses an inlining, a kill
        somebody performs that nobody recorded keeps a stale value.
        """
        blocks, block = self._facts("f { $x = 'b' }")
        self.assertIs(blocks.facts(block).scope, Ps1BlockScope.UNKNOWN)
        self.assertTrue(blocks.may_write_caller_scope(block))
        self.assertEqual(self._written_names("f { $x = 'b' }"), ['x'])


class TestPs1BlockFactsReachCycleModel(TestBase):

    def _repeats(self, source: str) -> bool:
        tree = Ps1Parser(source).parse()
        blocks = build_block_model(tree)
        cycles = CycleModel(build_control_flow_model(tree), blocks.body_site)
        block = next(
            node for node in tree.walk() if isinstance(node, Ps1ScriptBlock) and node.body)
        return cycles.repeats(block.body[0])

    def test_a_foreach_object_body_repeats_though_the_statement_handing_it_over_does_not(self):
        self.assertTrue(self._repeats("1..3 | ForEach-Object { $x = 'b' }"))

    def test_a_body_invoked_once_beside_it_does_not_repeat(self):
        """
        The floor under the iteration work: without it, classifying every block as repeating would
        pass the test above and read as caution.
        """
        self.assertFalse(self._repeats("& { $x = 'b' }"))
        self.assertFalse(self._repeats(". { $x = 'b' }"))

    def test_a_block_written_inside_a_loop_repeats_with_the_loop_around_it(self):
        self.assertTrue(self._repeats("while ($c) { & { $x = 'b' } }"))
