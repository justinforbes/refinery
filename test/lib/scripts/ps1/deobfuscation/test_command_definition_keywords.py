from __future__ import annotations

import unittest

from test.lib.scripts.ps1.deobfuscation import TestPs1

#: An acting statement that is never a removal candidate, so its survival says only that the pass
#: ran and did not empty the script wholesale. A `Write-Host` writes to the host and never drops.
_ANCHOR = "Write-Host 'ANCHOR_SURVIVES'"
_ANCHOR_TOKEN = 'ANCHOR_SURVIVES'


class _Ps1CommandDefinitionKeyword(TestPs1):

    def _assertDiscardedCallRemoved(self, source: str, token: str) -> None:
        result = self._deobfuscate(source)
        self.assertNotIn(token, result)
        self.assertIn(_ANCHOR_TOKEN, result)

    def _assertDiscardedCallKept(self, source: str, token: str, acting: str) -> None:
        result = self._deobfuscate(source)
        self.assertIn(token, result)
        self.assertIn(acting, result)
        self.assertIn(_ANCHOR_TOKEN, result)


class TestPs1ADiscardedPureCommandWithNoDefinitionOfItsNameIsRemoved(_Ps1CommandDefinitionKeyword):
    """
    The controls that keep the pins below from being vacuous. A discarded side-effect-free command
    is removable when nothing redefines its name — whether the script names no command-defining
    keyword at all, or a `workflow`/`configuration` that defines some *other* command. In each case
    the call still denotes the pure built-in the metadata describes and is shed, across the varied
    discard idioms.
    """

    def test_a_null_assigned_pure_command_with_no_definition_is_removed(self):
        self._assertDiscardedCallRemoved(
            F'$Null = Get-Random -Maximum 88175\n{_ANCHOR}', '88175')

    def test_a_void_cast_pure_command_with_no_definition_is_removed(self):
        self._assertDiscardedCallRemoved(
            F"[Void](Get-Date -Format 'Zqxwmp')\n{_ANCHOR}", 'Zqxwmp')

    def test_an_out_null_piped_pure_command_with_no_definition_is_removed(self):
        self._assertDiscardedCallRemoved(
            F"Get-ChildItem 'Vbnmqz' | Out-Null\n{_ANCHOR}", 'Vbnmqz')

    def test_a_workflow_defining_a_different_name_does_not_keep_the_discarded_call(self):
        self._assertDiscardedCallRemoved(
            F'workflow Do-Thing {{ Start-Process calc }}\n'
            F'$Null = Get-Random -Maximum 44011\n{_ANCHOR}', '44011')

    def test_a_configuration_defining_a_different_name_does_not_keep_the_discarded_call(self):
        self._assertDiscardedCallRemoved(
            F'configuration Set-Thing {{ Start-Process notepad }}\n'
            F"[Void](Get-Date -Format 'Wmzqxp')\n{_ANCHOR}", 'Wmzqxp')


class TestPs1ACallAfterAWorkflowDefiningItsNameIsKept(_Ps1CommandDefinitionKeyword):
    """
    A `workflow NAME { ... }` statement introduces a command named NAME that shadows a same-named
    built-in cmdlet under PowerShell's command resolution, so a bare NAME after it resolves to the
    workflow body rather than the pure built-in the metadata describes. A discarded side-effect-free
    call to NAME may therefore no longer be trusted as pure and must be kept, just as it is after a
    `function` redefining the name. The tool does not yet recognize the `workflow` keyword as a
    command definition, so it still trusts the name and unsoundly deletes the call.
    """

    @unittest.expectedFailure
    def test_a_null_assigned_call_after_a_workflow_of_its_name_is_kept(self):
        # Correct behavior: the workflow redefines Get-Random ahead of this discarded call, so the
        # call may run the workflow body and must be kept. Not yet implemented; it is dropped.
        self._assertDiscardedCallKept(
            F'workflow Get-Random {{ Start-Process calc }}\n'
            F'$Null = Get-Random -Maximum 88175\n{_ANCHOR}', '88175', 'Start-Process calc')

    @unittest.expectedFailure
    def test_an_out_null_piped_call_after_a_workflow_of_its_name_is_kept(self):
        # Correct behavior: the workflow shadows Get-Date, so this discarded pipeline now runs the
        # workflow body and must be kept. Not yet implemented — the pipeline is dropped.
        self._assertDiscardedCallKept(
            F'workflow Get-Date {{ Start-Process notepad }}\n'
            F"Get-Date -Format 'Zqxwmp' | Out-Null\n{_ANCHOR}", 'Zqxwmp', 'Start-Process notepad')


class TestPs1ACallAfterAConfigurationDefiningItsNameIsKept(_Ps1CommandDefinitionKeyword):
    """
    A `configuration NAME { ... }` (DSC) statement likewise introduces a command named NAME that
    shadows a same-named built-in cmdlet, so a discarded side-effect-free call to NAME placed after
    it resolves to the configuration body and must be kept rather than deleted. The tool does not
    yet recognize the `configuration` keyword as a command definition and unsoundly removes it.
    """

    @unittest.expectedFailure
    def test_a_void_cast_call_after_a_configuration_of_its_name_is_kept(self):
        # Correct behavior: the configuration redefines Get-Random ahead of this discarded call, so
        # the call may run the configuration body and must be kept. Not yet implemented — dropped.
        self._assertDiscardedCallKept(
            F'configuration Get-Random {{ Start-Process calc }}\n'
            F'[Void](Get-Random -Maximum 44011)\n{_ANCHOR}', '44011', 'Start-Process calc')

    @unittest.expectedFailure
    def test_an_out_null_piped_call_after_a_configuration_of_its_name_is_kept(self):
        # Correct behavior: the configuration shadows Get-ChildItem, so this discarded pipeline now
        # runs the configuration body and must be kept. Not yet implemented; it is dropped.
        self._assertDiscardedCallKept(
            F'configuration Get-ChildItem {{ Start-Process notepad }}\n'
            F"Get-ChildItem 'Vbnmqz' | Out-Null\n{_ANCHOR}", 'Vbnmqz', 'Start-Process notepad')
