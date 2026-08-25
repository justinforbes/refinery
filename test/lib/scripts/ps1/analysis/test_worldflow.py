from __future__ import annotations

import unittest

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
        self.assertTrue(reach.may_trust_command_name_at('write-host', read))
        opener = Ps1Parser('Invoke-Expression $env:PAYLOAD').parse().body[0]
        set_body(script, [*script.body, opener])
        self.assertFalse(reach.closed_for_the_whole_run)
        self.assertFalse(reach.closed_at(read))
        self.assertFalse(reach.may_trust_command_name_at('write-host', read))


class TestPs1CommandTrustIsPositionalWhereTheWorldStaysClosed(TestBase):

    def test_whole_run_and_positional_trust_disagree_across_a_function_redefinition(self):
        script = Ps1Parser(
            '$Null = Get-Random -Maximum 88175\n'
            'function Get-Random { Start-Process calc }\n'
            'Get-Random').parse()
        cache = Ps1ModelCache(script)
        reach = cache.world_reach
        call_before, _, call_after = script.body
        self.assertTrue(reach.closed_for_the_whole_run)
        self.assertTrue(cache.closed_world.may_trust_command_name('Start-Process'))
        self.assertFalse(cache.closed_world.may_trust_command_name('Get-Random'))
        self.assertTrue(reach.may_trust_command_name_at('Get-Random', call_before))
        self.assertFalse(reach.may_trust_command_name_at('Get-Random', call_after))

    def test_a_name_the_whole_run_trusts_is_trusted_at_every_position(self):
        """
        The positional query is a widening of the whole-run one and never a second opinion: it
        short-circuits on the whole-run verdict, so it grants wherever that grants and the flood is
        consulted only where it refuses. A shadow site sits below to keep the flood in play for the
        name it spells, and the untouched name has to stay trusted on both sides of it.
        """
        script = Ps1Parser(
            '$Null = Get-Random -Maximum 88175\n'
            'function Get-Random { Start-Process calc }\n'
            'Write-Host done').parse()
        cache = Ps1ModelCache(script)
        self.assertTrue(cache.closed_world.may_trust_command_name('Start-Process'))
        for index, statement in enumerate(script.body):
            with self.subTest(index):
                self.assertTrue(
                    cache.world_reach.may_trust_command_name_at('Start-Process', statement))


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
        self.assertTrue(reach.may_trust_command_name_at('Get-Random', script.body[1]))
        self.assertFalse(reach.may_trust_command_name_at('Get-Random', script.body[3]))

    def test_a_redefinition_inside_the_handler_is_distrusted_across_the_block(self):
        script, reach = self._reach(
            'trap { function Get-Random { Start-Process calc }\ncontinue }\n'
            '$Null = Get-Random -Maximum 88175\n'
            '$Null = Get-Random -Maximum 88176')
        self.assertFalse(reach.may_trust_command_name_at('Get-Random', script.body[1]))
        self.assertFalse(reach.may_trust_command_name_at('Get-Random', script.body[2]))


class TestPs1TheStatementAResumptionLandsOnIsTheOneControlEnters(TestBase):
    """
    Where the forward half finds the statement control resumes at, and the one shape it finds the
    wrong one for. A slot is put in front of each guarded statement and joined to whatever that
    statement is entered by; a statement that enters nothing leaves its slot unclaimed and the slot
    carries on to the next one, which is right for a `trap` declaration and wrong for a construct
    that builds nodes without being entered at any of them.

    `try { }` with an empty guarded block is that construct: it builds its `catch` clause and links
    no frontier, so the slot rolls past the whole `try` and the flood never reaches inside it. The
    grant costs nothing today — an empty `try` cannot throw, so the clause is dead — which is
    exactly why it needs pinning rather than trusting: what is wrong is the reading, not the shape,
    and the next construct built this way need not be dead.
    """

    def _reach(self, source: str):
        script = Ps1Parser(source).parse()
        return script, Ps1ModelCache(script).world_reach

    @staticmethod
    def _guarded(clause: str) -> str:
        return (
            'trap { continue }\n'
            'Invoke-Expression $env:PAYLOAD\n'
            F'{clause}\n'
            '$Null = [Math]::Sqrt(169)'
        )

    def test_a_read_in_the_handler_of_a_guarded_construct_is_poisoned(self):
        script, reach = self._reach(self._guarded("try { 'a' } catch { $Null = [Math]::Sqrt(144) }"))
        read = script.body[2].catch_clauses[0].body.body[0]
        self.assertFalse(reach.closed_at(read))

    @unittest.expectedFailure
    def test_the_same_read_is_poisoned_where_the_construct_guards_nothing(self):
        script, reach = self._reach(self._guarded('try { } catch { $Null = [Math]::Sqrt(144) }'))
        read = script.body[2].catch_clauses[0].body.body[0]
        self.assertFalse(reach.closed_at(read))
