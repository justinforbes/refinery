from __future__ import annotations

from test.lib.scripts.ps1.deobfuscation import TestPs1

#: An acting statement that is never a removal candidate, so its survival says only that the pass
#: did not empty the script wholesale. A `Write-Host` writes to the host and can never be dropped.
_ANCHOR = "Write-Host 'ANCHOR_SURVIVES'"
_ANCHOR_TOKEN = 'ANCHOR_SURVIVES'

#: Discarded side-effect-free commands, each carrying a distinctive literal token that is present in
#: the source and absent once the statement is removed. The discard idiom is varied on purpose —
#: `$Null =`, `[Void](...)`, `| Out-Null` all sink the value — so the trust decision is what the
#: tests turn on rather than one spelling of the discard.
_RANDOM = ('$Null = Get-Random -Maximum 88175', '88175')
_DATE = ("$Null = Get-Date -Format 'Zqxwmp'", 'Zqxwmp')
_MEASURE = ('$Null = Measure-Object -InputObject 90210', '90210')
_SELECT = ("[Void](Select-Object -First 3 -InputObject 'Kdwlrp')", 'Kdwlrp')
_CHILDITEM = ("$Null = Get-ChildItem 'Vbnmqz'", 'Vbnmqz')
_SORT = ("Sort-Object -InputObject 'Rtyplo' | Out-Null", 'Rtyplo')

_PURE_COMMANDS = (_RANDOM, _DATE, _MEASURE, _SELECT, _CHILDITEM, _SORT)

#: The leaks below all rebind the SAME command the discarded statement calls, because only such a
#: statement can make that command run something other than the built-in the metadata describes. An
#: opaque leak (`Invoke-Expression`, an opaque call `& $var`, `Import-Module`) can rebind any name,
#: so it stands in for whichever command sits beside it; a `Set-Alias`/`function` leak must name the
#: command itself. A `Set-Alias` onto an existing name uses a wildcard target so the aliaser is not
#: inlined away, and it survives even unreferenced because it is an identity leak, not a dead
#: binding. An alias to some *other* name would leave the command resolving to its built-in and so
#: would not justify keeping it — those cases are deliberately absent.


class _Ps1CommandTrustFlow(TestPs1):

    def _assertPureCommandRemoved(self, source: str, token: str, acting: str) -> None:
        result = self._deobfuscate(source)
        self.assertNotIn(token, result)
        self.assertIn(acting, result)
        self.assertIn(_ANCHOR_TOKEN, result)

    def _assertPureCommandKept(self, source: str, token: str, acting: str) -> None:
        result = self._deobfuscate(source)
        self.assertIn(token, result)
        self.assertIn(acting, result)
        self.assertIn(_ANCHOR_TOKEN, result)


class TestPs1APureCommandBeforeTheOnlyLeakIsRemoved(_Ps1CommandTrustFlow):
    """
    A discarded side-effect-free command placed where no statement that could rebind its own name
    can have run before it observes the command table the metadata describes, so it is junk like
    any other and is removed. Command-name trust is positional — `may_trust_command_name_at`
    floods forward from the openers and from the name's own definition sites — so a call neither
    flood reaches still names the built-in and goes.
    """

    def test_a_pure_command_before_the_only_invoke_expression_is_removed(self):
        source, token = _RANDOM
        self._assertPureCommandRemoved(
            F'{source}\nInvoke-Expression $enc\n{_ANCHOR}', token, 'Invoke-Expression')

    def test_a_pure_command_before_the_only_opaque_call_is_removed(self):
        source, token = _CHILDITEM
        self._assertPureCommandRemoved(
            F'{source}\n& $dispatch\n{_ANCHOR}', token, '$dispatch')

    def test_a_pure_command_before_the_only_import_module_is_removed(self):
        source, token = _SELECT
        self._assertPureCommandRemoved(
            F'{source}\nImport-Module .\\mod.psm1\n{_ANCHOR}', token, 'Import-Module')

    def test_a_pure_command_before_its_own_function_redefinition_is_removed(self):
        # The bare command runs before the `function` statement that takes its name over, so it
        # names the built-in and not the effectful redefinition below it; the redefinition and the
        # later call that reaches it both stay.
        self._assertPureCommandRemoved(
            F'$Null = Get-Random -Maximum 88175\n'
            F'function Get-Random {{ Start-Process calc }}\n'
            F'Get-Random\n{_ANCHOR}',
            '88175', 'Start-Process calc')

    def test_a_pure_command_before_its_own_set_alias_is_removed(self):
        # The discarded call precedes the `Set-Alias` that rebinds its name, so it still resolves to
        # the built-in `Get-Date` and is removable; the alias itself is an identity leak and stays.
        source, token = _DATE
        self._assertPureCommandRemoved(
            F'{source}\nSet-Alias Get-Date i*x\n{_ANCHOR}', token, 'Set-Alias')


class TestPs1APureCommandALeakCanPrecedeIsKept(_Ps1CommandTrustFlow):
    """
    Where a statement that could rebind the discarded command's own name can have run before it, the
    command may have been rebound to something effectful, so deleting it could drop an effect and it
    must be kept. This is the soundness guardrail, covering the leak ahead of the command in
    straight-line code and the leak sharing a loop body with it, whose back edge places an earlier
    iteration's leak before a later iteration's command.
    """

    def test_a_pure_command_after_an_invoke_expression_is_kept(self):
        source, token = _RANDOM
        self._assertPureCommandKept(
            F'Invoke-Expression $enc\n{source}\n{_ANCHOR}', token, 'Invoke-Expression')

    def test_a_pure_command_after_an_opaque_call_is_kept(self):
        source, token = _MEASURE
        self._assertPureCommandKept(
            F'& $dispatch\n{source}\n{_ANCHOR}', token, '$dispatch')

    def test_a_pure_command_after_an_import_module_is_kept(self):
        source, token = _SELECT
        self._assertPureCommandKept(
            F'Import-Module .\\mod.psm1\n{source}\n{_ANCHOR}', token, 'Import-Module')

    def test_a_pure_command_after_its_own_function_redefinition_is_kept(self):
        # The name now resolves to the effectful redefinition, so the discarded call is the one that
        # could run `Start-Process` and may not be deleted.
        self._assertPureCommandKept(
            F'function Get-Random {{ Start-Process calc }}\n'
            F'$Null = Get-Random -Maximum 88175\n{_ANCHOR}',
            '88175', 'Start-Process calc')

    def test_a_pure_command_after_its_own_set_alias_is_kept(self):
        # The alias rebinds `Get-Date` before the discarded call, so the call may run the alias
        # target rather than the built-in and must be kept.
        source, token = _DATE
        self._assertPureCommandKept(
            F'Set-Alias Get-Date i*x\n{source}\n{_ANCHOR}', token, 'Set-Alias')

    def test_a_pure_command_looping_with_an_invoke_expression_is_kept(self):
        source, token = _MEASURE
        result = self._deobfuscate(F'while ($True) {{ {source}; Invoke-Expression $enc }}')
        self.assertIn(token, result)
        self.assertIn('Invoke-Expression', result)

    def test_a_pure_command_looping_with_an_opaque_call_is_kept(self):
        source, token = _RANDOM
        self._assertPureCommandKept(
            F'foreach ($i in 1..4) {{ {source}; & $dispatch }}\n{_ANCHOR}', token, '$dispatch')

    def test_a_pure_command_looping_with_an_import_module_is_kept(self):
        source, token = _CHILDITEM
        self._assertPureCommandKept(
            F'while ($True) {{ {source}; Import-Module .\\mod.psm1 }}\n{_ANCHOR}',
            token, 'Import-Module')


class TestPs1CommandTrustPolesWithAndWithoutALeak(_Ps1CommandTrustFlow):
    """
    The two poles the flow-sensitive verdict swings between, so that neither the removal nor the
    guardrail can pass by doing nothing. With no name-rebinding statement anywhere the discarded
    command is removed; with one reachable before it the same command is kept, and that statement is
    the only thing that changed.
    """

    def test_a_pure_command_with_no_identity_leak_anywhere_is_removed(self):
        for source, token in _PURE_COMMANDS:
            with self.subTest(source):
                self._assertPureCommandRemoved(F'{source}\n{_ANCHOR}', token, _ANCHOR_TOKEN)

    def test_a_pure_command_with_an_identity_leak_reachable_before_it_is_kept(self):
        source, token = _RANDOM
        for leak, acting in (
            ('Invoke-Expression $enc', 'Invoke-Expression'),
            ('& $dispatch', '$dispatch'),
            ('Import-Module .\\mod.psm1', 'Import-Module'),
            ('function Get-Random { Start-Process calc }', 'Start-Process calc'),
            ('Set-Alias Get-Random i*x', 'Set-Alias'),
        ):
            with self.subTest(leak):
                self._assertPureCommandKept(F'{leak}\n{source}\n{_ANCHOR}', token, acting)
