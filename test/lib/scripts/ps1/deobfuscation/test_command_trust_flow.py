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


class TestPs1AFunctionRedefinitionSplitsItsNameIntoTwoEras(_Ps1CommandTrustFlow):
    """
    One script, one name, two eras: the discarded call ahead of the `function` statement still
    names the built-in and goes, while the bare call behind it runs the effectful redefinition and
    stays, in the same pass over the same tree.
    """

    def test_the_discarded_call_before_the_definition_goes_and_the_bare_call_after_it_stays(self):
        result = self._deobfuscate(
            F'$Null = Get-Random -Maximum 88175\n'
            F'function Get-Random {{ Start-Process calc }}\n'
            F'Get-Random -Maximum 70223\n{_ANCHOR}')
        self.assertNotIn('88175', result)
        self.assertIn('70223', result)
        self.assertIn('Start-Process calc', result)
        self.assertIn(_ANCHOR_TOKEN, result)


class TestPs1APureCommandLoopingWithItsOwnRedefinitionIsKept(_Ps1CommandTrustFlow):
    """
    A redefinition of the discarded command's own name sharing a loop body with it precedes a
    later iteration's call along the back edge, so the call stays although the redefinition follows
    it in source. The same call in the same loop without the redefinition goes: a loop body is not
    itself a sanctuary.
    """

    def test_a_pure_command_alone_in_a_loop_body_is_removed(self):
        source, token = _RANDOM
        self._assertPureCommandRemoved(
            F'while ($True) {{ {source}; {_ANCHOR} }}', token, _ANCHOR_TOKEN)

    def test_a_pure_command_looping_with_its_own_function_redefinition_is_kept(self):
        source, token = _RANDOM
        self._assertPureCommandKept(
            F'while ($True) {{ {source}; function Get-Random {{ Start-Process calc }} }}\n{_ANCHOR}',
            token, 'Start-Process calc')

    def test_a_pure_command_looping_with_its_own_set_alias_is_kept(self):
        source, token = _DATE
        self._assertPureCommandKept(
            F'foreach ($i in 1..4) {{ {source}; Set-Alias Get-Date i*x }}\n{_ANCHOR}',
            token, 'Set-Alias')


class TestPs1ARedefinitionSiteTheGraphCannotPlaceIsAlwaysAhead(_Ps1CommandTrustFlow):
    """
    A `function global:` statement inside a parameter default rebinds the name when the enclosing
    function is called, but no control-flow graph places a default's evaluation, so no position can
    prove itself ahead of it and every discarded call of the name stays — whether that site is the
    name's only one or stands beside an ordinary redefinition the flood could place.
    """

    def test_a_pure_command_after_a_param_default_redefinition_of_its_own_name_is_kept(self):
        self._assertPureCommandKept(
            F'function f {{ param($p = $(function global:Get-Random {{ Start-Process calc }})) }}\n'
            F'f\n'
            F'$Null = Get-Random -Maximum 88175\n{_ANCHOR}',
            '88175', 'Start-Process calc')

    def test_an_unplaceable_site_keeps_a_call_an_ordinary_later_redefinition_alone_would_shed(self):
        self._assertPureCommandKept(
            F'function f {{ param($p = $(function global:Get-Random {{ Start-Process calc }})) }}\n'
            F'f\n'
            F'$Null = Get-Random -Maximum 88175\n'
            F'function Get-Random {{ Start-Process iexplore }}\n{_ANCHOR}',
            '88175', 'iexplore')


class TestPs1BothPipedDiscardSinksFollowTheTrustVerdict(_Ps1CommandTrustFlow):
    """
    The verdict turns on the trust of the piped command's own name, not on which sink swallows the
    pipeline: `| Out-Null` and `| ForEach-Object { [Void]$_ }` are removed and kept together.
    """

    _SINKS = ('| Out-Null', '| ForEach-Object { [Void]$_ }')

    def test_a_piped_discard_with_no_rebinder_of_its_name_anywhere_is_removed(self):
        for sink in self._SINKS:
            with self.subTest(sink):
                self._assertPureCommandRemoved(
                    F'Get-Random -Maximum 88175 {sink}\n{_ANCHOR}', '88175', _ANCHOR_TOKEN)

    def test_a_piped_discard_after_a_rebinder_of_its_own_name_is_kept(self):
        for sink in self._SINKS:
            for rebinder, acting in (
                ('Set-Alias Get-Random i*x', 'Set-Alias'),
                ('function Get-Random { Start-Process calc }', 'Start-Process calc'),
                ('Invoke-Expression $enc', 'Invoke-Expression'),
            ):
                with self.subTest(F'{rebinder} {sink}'):
                    self._assertPureCommandKept(
                        F'{rebinder}\nGet-Random -Maximum 88175 {sink}\n{_ANCHOR}', '88175', acting)


class TestPs1AWholeRunRefusalOverridesThePositionalGrant(_Ps1CommandTrustFlow):
    """
    A script that spells `$PSCommandPath` can re-run its own statements after its leaks, and a root
    `process` block re-runs per pipeline input, so either refuses the whole run and the positional
    grant that removes a call ahead of its own redefinition yields to that refusal.
    """

    _REMOVABLE = (
        F'$Null = Get-Random -Maximum 88175\n'
        F'function Get-Random {{ Start-Process calc }}\n'
        F'Get-Random -Maximum 70223\n{_ANCHOR}')

    def test_naming_the_own_script_path_keeps_a_call_its_position_alone_would_shed(self):
        self._assertPureCommandRemoved(self._REMOVABLE, '88175', 'Start-Process calc')
        self._assertPureCommandKept(
            F'{self._REMOVABLE}\nWrite-Host $PSCommandPath', '88175', 'Start-Process calc')

    def test_a_root_process_block_keeps_a_call_its_position_alone_would_shed(self):
        self._assertPureCommandRemoved(self._REMOVABLE, '88175', 'Start-Process calc')
        self._assertPureCommandKept(
            F'process {{\n{self._REMOVABLE}\n}}', '88175', 'Start-Process calc')


class TestPs1AnOpenerRefusesACallItsOwnDefinitionSitesWouldGrant(_Ps1CommandTrustFlow):
    """
    The call sits where the flood from its own name's definition sites does not reach, because the
    `function` statement only follows it, so the own-name axis alone would remove it; the
    `Invoke-Expression` ahead of it can rebind any name and must keep it on its own.
    """

    def test_a_call_between_an_opener_and_its_own_later_redefinition_is_kept(self):
        self._assertPureCommandKept(
            F'Invoke-Expression $enc\n'
            F'$Null = Get-Random -Maximum 88175\n'
            F'function Get-Random {{ Start-Process calc }}\n{_ANCHOR}',
            '88175', 'Invoke-Expression')
