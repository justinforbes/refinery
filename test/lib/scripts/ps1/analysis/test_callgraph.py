from __future__ import annotations

from test import TestBase

from refinery.lib.scripts.ps1.analysis.callgraph import build_call_graph
from refinery.lib.scripts.ps1.analysis.types import TypeOracle
from refinery.lib.scripts.ps1.analysis.world import Ps1TypeWorld
from refinery.lib.scripts.ps1.parser import Ps1Parser

#: A world with nothing shadowed and nothing imported, so that `is_readable` answers on the rows
#: this file is about rather than on a world that was open to begin with.
_CLOSED_WORLD = TypeOracle(world=Ps1TypeWorld(True, frozenset()))


class TestPs1CallGraphReadability(TestBase):

    @staticmethod
    def _graph(source: str):
        return build_call_graph(Ps1Parser(source).parse(), _CLOSED_WORLD)

    def test_a_script_whose_calls_and_definitions_meet_is_readable(self):
        self.assertTrue(self._graph("function Qzmr { 'P' }\nQzmr").is_readable)

    def test_a_module_qualified_call_onto_a_defined_name_is_not(self):
        self.assertFalse(self._graph("function Qzmr { 'P' }\n& 'MyModule\\Qzmr'").is_readable)

    def test_the_bare_spelling_of_that_call_is_not_either(self):
        self.assertFalse(self._graph("function Qzmr { 'P' }\n& MyModule\\Qzmr").is_readable)

    def test_the_definition_may_be_written_below_the_call(self):
        self.assertFalse(self._graph("& 'MyModule\\Qzmr'\nfunction Qzmr { 'P' }").is_readable)

    def test_an_aliased_call_onto_a_defined_name_is_not_readable(self):
        self.assertFalse(
            self._graph("function Invoke-Expression { 'P' }\niex 'x'").is_readable)

    def test_a_scope_qualified_call_is_readable_because_the_key_already_matches(self):
        self.assertTrue(self._graph("function Qzmr { 'P' }\n& 'global:Qzmr'").is_readable)

    def test_an_executable_invoked_by_path_is_readable(self):
        self.assertTrue(self._graph("function j { 1 }\nj\n& 'C:\\tools\\stage2.exe'").is_readable)

    def test_an_executable_invoked_by_unc_path_is_readable(self):
        self.assertTrue(self._graph("function j { 1 }\nj\n& '\\\\host\\share\\p.exe'").is_readable)

    def test_an_executable_whose_stem_names_a_definition_is_not(self):
        self.assertFalse(
            self._graph("function stage2.exe { 'P' }\n& 'C:\\tools\\stage2.exe'").is_readable)

    def test_a_qualified_call_onto_a_name_nothing_defines_is_readable(self):
        self.assertTrue(self._graph("function Qzmr { 'P' }\n& 'MyModule\\Other'").is_readable)

    def test_a_module_qualified_export_is_still_read_as_an_export(self):
        graph = self._graph(
            "& 'Microsoft.PowerShell.Core\\Export-ModuleMember' -Function f\nfunction f { 'P' }")
        self.assertTrue(graph.exports_a_name)
        self.assertFalse(graph.is_readable)
