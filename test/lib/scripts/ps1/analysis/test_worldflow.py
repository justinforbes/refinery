from __future__ import annotations

from test import TestBase

from refinery.lib.scripts import set_body
from refinery.lib.scripts.ps1.analysis.cache import Ps1ModelCache
from refinery.lib.scripts.ps1.analysis.worldflow import Ps1WorldReach
from refinery.lib.scripts.ps1.model import Ps1Script
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


class TestPs1FloodsGoForwardThroughAResumingTrap(TestBase):
    """
    A `trap { continue }` resumes the block it guards at the statement after the one that threw, so
    a leak written late in such a block cannot have run at a read written above it. Measured on 5.1:
    `trap { continue }; Write-Host 'one'; throw 'e'; Write-Host 'three'` writes `one` once and then
    `three`.

    Both floods answer this way or neither is worth having: an obfuscated script that wraps its
    whole body in a resuming trap is exactly the one whose every read the over-approximate reading
    refuses.
    """

    def _reach(self, source: str) -> tuple[Ps1Script, Ps1WorldReach]:
        script = Ps1Parser(source).parse()
        return script, Ps1ModelCache(script).world_reach

    def test_an_opener_late_in_a_guarded_block_leaves_the_reads_above_it_closed(self):
        script, reach = self._reach(
            'trap { continue }\n'
            '$Null = [Math]::Sqrt(144)\n'
            'Invoke-Expression $env:PAYLOAD\n'
            '$Null = [Math]::Sqrt(169)')
        self.assertFalse(reach.closed_for_the_whole_run)
        self.assertTrue(reach.closed_at(script.body[1]))
        self.assertFalse(reach.closed_at(script.body[3]))

    def test_an_opener_poisons_the_reads_resumption_reaches_across_a_terminator(self):
        """
        Nothing but resumption joins the two: the `throw` between them ends the statement list, and
        a flood that stopped there would call the world closed at a read the leak precedes.
        """
        script, reach = self._reach(
            'trap { continue }\n'
            'Invoke-Expression $env:PAYLOAD\n'
            "throw 'x'\n"
            '$Null = [Math]::Sqrt(144)')
        self.assertFalse(reach.closed_at(script.body[3]))

    def test_an_opener_poisons_the_body_of_a_try_resumption_reaches(self):
        script, reach = self._reach(
            'trap { continue }\n'
            'Invoke-Expression $env:PAYLOAD\n'
            "throw 'x'\n"
            'try { $Null = [Math]::Sqrt(144) } catch { }')
        self.assertFalse(reach.closed_at(script.body[3].try_block.body[0]))

    def test_an_opener_last_in_a_nested_guarded_block_poisons_what_follows_the_block(self):
        """
        Measured on 5.1: `if ($true) { trap { continue }; Write-Host 'in'; throw 'e' };
        Write-Host 'after'` writes `in` and then `after`, so a leak in the block has run by the time
        the statement after the block does.
        """
        script, reach = self._reach(
            'if ($c) { trap { continue }\n'
            'Invoke-Expression $env:PAYLOAD\n'
            "throw 'x' }\n"
            '$Null = [Math]::Sqrt(144)')
        self.assertFalse(reach.closed_at(script.body[1]))

    def test_an_opener_inside_the_handler_poisons_the_block_it_resumes_into(self):
        """
        A handler runs only after some statement of the block threw and may resume at any of them,
        so an opener written in one has run wherever the block goes on. No forward edge leaves a
        handler statement, and answering from the forward edges alone would vouch for exactly the
        statements the handler resumes into.
        """
        script, reach = self._reach(
            'trap { Invoke-Expression $env:PAYLOAD\ncontinue }\n'
            '$Null = [Math]::Sqrt(144)\n'
            '$Null = [Math]::Sqrt(169)')
        self.assertFalse(reach.closed_at(script.body[1]))
        self.assertFalse(reach.closed_at(script.body[2]))

    def test_a_redefinition_late_in_a_guarded_block_leaves_the_calls_above_it_trusted(self):
        script, reach = self._reach(
            'trap { continue }\n'
            '$Null = Get-Random -Maximum 88175\n'
            'function Get-Random { Start-Process calc }\n'
            'Get-Random')
        self.assertFalse(reach.may_trust_command_name('Get-Random'))
        self.assertTrue(reach.may_trust_command_name_at('Get-Random', script.body[1]))
        self.assertFalse(reach.may_trust_command_name_at('Get-Random', script.body[3]))

    def test_a_redefinition_inside_the_handler_is_distrusted_across_the_block(self):
        script, reach = self._reach(
            'trap { function Get-Random { Start-Process calc }\ncontinue }\n'
            '$Null = Get-Random -Maximum 88175\n'
            '$Null = Get-Random -Maximum 88176')
        self.assertFalse(reach.may_trust_command_name_at('Get-Random', script.body[1]))
        self.assertFalse(reach.may_trust_command_name_at('Get-Random', script.body[2]))
