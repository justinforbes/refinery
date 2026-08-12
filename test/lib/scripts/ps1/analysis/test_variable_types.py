from __future__ import annotations

from inspect import cleandoc

from test import TestBase

from refinery.lib.scripts.analysis.cycles import CycleModel
from refinery.lib.scripts.ps1.analysis.blocks import build_block_model
from refinery.lib.scripts.ps1.analysis.cfg import build_control_flow_model
from refinery.lib.scripts.ps1.analysis.dataflow import build_variable_flow
from refinery.lib.scripts.ps1.analysis.dominance import build_dominance
from refinery.lib.scripts.ps1.analysis.model import build_semantic_model, is_write_occurrence
from refinery.lib.scripts.ps1.analysis.variable_types import type_at
from refinery.lib.scripts.ps1.model import Ps1Variable
from refinery.lib.scripts.ps1.parser import Ps1Parser


def _in_source_order(node):
    """
    Pre-order over `children()`, which is source order. `Node.walk` is driven by a stack and comes
    back reversed, and `Ps1Variable.offset` is never set by the parser, so neither of them orders
    occurrences.
    """
    yield node
    for child in node.children():
        yield from _in_source_order(child)


class TestPs1TypeAt(TestBase):
    """
    What .NET type a variable carries at one read of it, expressed as the name PowerShell reports
    for that type. Reads are addressed by their position in the source rather than by name, since
    the whole point is that two reads of one name can answer differently.
    """

    def _type_at(self, source: str, read: int = -1, name: str = 'q') -> str | None:
        tree = Ps1Parser(source).parse()
        semantic = build_semantic_model(tree)
        control = build_control_flow_model(tree)
        blocks = build_block_model(tree)
        flow = build_variable_flow(
            semantic,
            control,
            build_dominance(control),
            blocks,
            CycleModel(control, blocks.body_site),
        )
        reads = [
            node for node in _in_source_order(tree)
            if isinstance(node, Ps1Variable)
            and node.name.lower() == name
            and not is_write_occurrence(node)
        ]
        found = type_at(reads[read], flow)
        return None if found is None else str(found)

    def test_a_name_the_script_never_writes_has_no_type(self):
        self.assertIsNone(self._type_at("$q.downloadstring('u')"))

    def test_a_read_after_the_only_write_carries_what_that_write_established(self):
        self.assertEqual(self._type_at(cleandoc("""
            $q = New-Object Net.WebClient
            $q.downloadstring('u')
        """)), 'System.Net.WebClient')

    def test_a_write_inside_a_function_body_does_not_type_a_read_outside_it(self):
        """
        The top-level `$q` holds `$null` however the body writes it, since nothing says the function
        was ever called. Answering `System.Net.WebClient` here folded `($q | Get-Member)[0].Name` to
        a member of a type that name never holds.
        """
        self.assertIsNone(self._type_at(cleandoc("""
            function f {
              $q = New-Object Net.WebClient
            }
            ($q | Get-Member)[0].Name
        """)))

    def test_a_write_inside_a_function_body_types_a_read_inside_that_body(self):
        self.assertEqual(self._type_at(cleandoc("""
            function f {
              $q = New-Object Net.WebClient
              $q.downloadstring('u')
            }
        """)), 'System.Net.WebClient')

    def test_a_write_inside_one_function_body_does_not_type_a_read_inside_another(self):
        self.assertIsNone(self._type_at(cleandoc("""
            function f {
              $q = New-Object Net.WebClient
            }
            function g {
              $q.downloadstring('u')
            }
        """)))

    def test_two_function_bodies_writing_one_name_are_typed_from_their_own_write(self):
        source = cleandoc("""
            function f {
              $q = New-Object Net.WebClient
              $q.downloadstring('u')
            }
            function g {
              $q = New-Object Text.StringBuilder
              $q.tostring()
            }
        """)
        self.assertEqual(self._type_at(source, read=0), 'System.Net.WebClient')
        self.assertEqual(self._type_at(source, read=1), 'System.Text.StringBuilder')

    def test_a_store_through_a_member_leaves_the_name_holding_what_it_held(self):
        self.assertEqual(self._type_at(cleandoc("""
            $q = New-Object Net.WebClient
            $q.Proxy = $null
            $q.downloadstring('u')
        """)), 'System.Net.WebClient')

    def test_a_write_and_a_read_inside_one_subexpression_are_typed(self):
        self.assertEqual(
            self._type_at("$($q = New-Object Net.WebClient; $q.downloadstring('u'))"),
            'System.Net.WebClient',
        )

    def test_a_read_before_the_write_it_shares_a_subexpression_with_is_refused(self):
        self.assertIsNone(
            self._type_at("$(($q | Get-Member)[0].Name; $q = New-Object Net.WebClient)"))

    def test_each_read_is_typed_by_the_write_it_observes_where_the_writes_disagree(self):
        source = cleandoc("""
            $q = New-Object Net.WebClient
            $q.downloadstring('u')
            $q = New-Object Text.StringBuilder
            $q.tostring()
        """)
        self.assertEqual(self._type_at(source, read=0), 'System.Net.WebClient')
        self.assertEqual(self._type_at(source, read=1), 'System.Text.StringBuilder')

    def test_a_read_no_single_write_reaches_is_refused_where_the_writes_disagree(self):
        self.assertIsNone(self._type_at(cleandoc("""
            $q = New-Object Net.WebClient
            if ($c) {
              $q = New-Object Text.StringBuilder
            }
            $q.tostring()
        """)))

    def test_a_read_no_single_write_reaches_is_typed_where_the_writes_agree(self):
        self.assertEqual(self._type_at(cleandoc("""
            $q = New-Object Net.WebClient
            if ($c) {
              $q = New-Object Net.WebClient
            }
            $q.downloadstring('u')
        """)), 'System.Net.WebClient')

    def test_a_read_preceding_every_write_of_the_name_is_refused(self):
        """
        Writes that agree about a type say nothing here: what stands at the read is whatever stood
        before the script ran, and `Get-Member` over that `$null` is an error, not a member list.
        """
        self.assertIsNone(self._type_at(cleandoc("""
            ($q | Get-Member)[0].Name
            $q = New-Object Net.WebClient
        """)))

    def test_a_read_at_the_top_of_a_loop_body_precedes_the_write_below_it(self):
        self.assertIsNone(self._type_at(cleandoc("""
            while ($c) {
              ($q | Get-Member)[0].Name
              $q = New-Object Net.WebClient
            }
        """)))

    def test_a_loop_body_read_is_typed_once_a_write_before_the_loop_reaches_it(self):
        self.assertEqual(self._type_at(cleandoc("""
            $q = New-Object Net.WebClient
            while ($c) {
              $q.downloadstring('u')
              $q = New-Object Net.WebClient
            }
        """)), 'System.Net.WebClient')

    def test_a_write_on_one_branch_only_does_not_type_the_read_after_the_branch(self):
        self.assertIsNone(self._type_at(cleandoc("""
            if ($c) {
              $q = New-Object Net.WebClient
            }
            $q.downloadstring('u')
        """)))

    def test_a_write_inside_a_body_reached_by_a_qualifier_does_not_type_a_read_outside(self):
        """
        `$script:q` written in a function body names the same variable the top-level read does, but
        nothing says the function was ever called.
        """
        self.assertIsNone(self._type_at(cleandoc("""
            function f {
              $script:q = New-Object Net.WebClient
            }
            ($q | Get-Member)[0].Name
        """)))

    def test_a_write_through_a_qualifier_types_the_reads_that_follow_it_in_its_body(self):
        self.assertEqual(self._type_at(cleandoc("""
            function f {
              $script:q = New-Object Net.WebClient
              $q.downloadstring('u')
            }
        """)), 'System.Net.WebClient')

    def test_a_write_carrying_no_type_of_its_own_is_not_looked_past(self):
        self.assertIsNone(self._type_at(cleandoc("""
            $q = New-Object Net.WebClient
            $q += 1
            $q.downloadstring('u')
        """)))

    def test_a_read_an_unattributable_write_may_precede_is_refused(self):
        self.assertIsNone(self._type_at(cleandoc("""
            $q = New-Object Net.WebClient
            Invoke-Expression $code
            $q.downloadstring('u')
        """)))

    def test_an_unattributable_write_after_the_read_leaves_the_type_standing(self):
        self.assertEqual(self._type_at(cleandoc("""
            $q = New-Object Net.WebClient
            $q.downloadstring('u')
            Invoke-Expression $code
        """)), 'System.Net.WebClient')

    def test_a_read_through_a_scope_qualifier_is_refused(self):
        for source in [
            cleandoc("""
                $global:q = New-Object Net.WebClient
                $global:q.downloadstring('u')
            """),
            cleandoc("""
                $q = New-Object Net.WebClient
                $script:q.downloadstring('u')
            """),
        ]:
            with self.subTest(source):
                self.assertIsNone(self._type_at(source))

    def test_a_foreach_over_a_string_binds_the_string_and_not_its_characters(self):
        self.assertEqual(
            self._type_at("foreach ($s in 'abc') { $s.substring(0, 1) }", name='s'),
            'System.String',
        )

    def test_a_foreach_binding_carries_the_type_the_iterated_elements_share(self):
        for source, expected in [
            ("foreach ($s in 'a', 'b') { $s.tostring() }", 'System.String'),
            ('foreach ($s in 1, 2) { $s.tostring() }', 'System.Int32'),
            ("foreach ($s in 'a', 1) { $s.tostring() }", None),
            ('foreach ($s in $items) { $s.tostring() }', None),
        ]:
            with self.subTest(source):
                self.assertEqual(self._type_at(source, name='s'), expected)


if __name__ == '__main__':
    import unittest
    unittest.main()
