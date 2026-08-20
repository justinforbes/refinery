from __future__ import annotations

from test import TestBase

from refinery.lib.scripts import set_body
from refinery.lib.scripts.ps1.analysis.cache import Ps1ModelCache
from refinery.lib.scripts.ps1.parser import Ps1Parser


class TestPs1WorldReachIsBoundToTheTreeItMeasured(TestBase):

    def test_a_positional_grant_is_withdrawn_once_the_tree_changes(self):
        script = Ps1Parser('$Null = [Math]::Sqrt(144)\nInvoke-Expression $c').parse()
        reach = Ps1ModelCache(script).world_reach
        read = script.body[0]
        self.assertFalse(reach.closed_for_the_whole_run)
        self.assertTrue(reach.closed_at(read))
        set_body(script, [*script.body, Ps1Parser('Write-Host done').parse().body[0]])
        self.assertFalse(reach.closed_at(read))

    def test_every_grant_is_withdrawn_once_the_tree_gains_a_world_opener(self):
        script = Ps1Parser('$x = 4\nWrite-Host $x').parse()
        reach = Ps1ModelCache(script).world_reach
        read = script.body[0]
        self.assertTrue(reach.closed_for_the_whole_run)
        self.assertTrue(reach.closed_at(read))
        self.assertTrue(reach.may_trust_command_name('write-host'))
        opener = Ps1Parser('Invoke-Expression $env:PAYLOAD').parse().body[0]
        set_body(script, [*script.body, opener])
        self.assertFalse(reach.closed_for_the_whole_run)
        self.assertFalse(reach.closed_at(read))
        self.assertFalse(reach.may_trust_command_name('write-host'))


class TestPs1CommandTrustIsPositionalWhereTheWorldStaysClosed(TestBase):

    def test_whole_run_and_positional_trust_disagree_across_a_function_redefinition(self):
        script = Ps1Parser(
            '$Null = Get-Random -Maximum 88175\n'
            'function Get-Random { Start-Process calc }\n'
            'Get-Random').parse()
        reach = Ps1ModelCache(script).world_reach
        call_before, _, call_after = script.body
        self.assertTrue(reach.closed_for_the_whole_run)
        self.assertTrue(reach.may_trust_command_name('Start-Process'))
        self.assertFalse(reach.may_trust_command_name('Get-Random'))
        self.assertTrue(reach.may_trust_command_name_at('Get-Random', call_before))
        self.assertFalse(reach.may_trust_command_name_at('Get-Random', call_after))
