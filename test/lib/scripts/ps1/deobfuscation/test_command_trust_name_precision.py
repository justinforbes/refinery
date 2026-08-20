from __future__ import annotations

import unittest

from test.lib.scripts.ps1.deobfuscation import TestPs1

#: An acting statement that is never a removal candidate, so its survival only says the pass did not
#: empty the script wholesale. A `Write-Host` writes to the host and can never be dropped.
_ANCHOR = "Write-Host 'ANCHOR_SURVIVES'"
_ANCHOR_TOKEN = 'ANCHOR_SURVIVES'

#: The discarded side-effect-free command whose survival every test turns on. Its distinctive
#: literal `88175` is in the source and gone once the statement is removed, and the three spellings
#: sink the value the same way (`$Null =`, `[Void](...)`, `| Out-Null`), so a verdict depends on the
#: trust decision rather than on one discard idiom.
_DISCARDED_TOKEN = '88175'
_DISCARDS = (
    '$Null = Get-Random -Maximum 88175',
    '[Void](Get-Random -Maximum 88175)',
    'Get-Random -Maximum 88175 | Out-Null',
)

#: A surviving statement that rebinds a command name OTHER than the discarded call's. `Set-Alias`
#: takes the command table over for the whole run, and the wildcard target keeps the binding from
#: being inlined away, so `Some-Other-Name` genuinely stays in the output. The name it rebinds is
#: `Some-Other-Name`, never `Get-Random`, so the discarded `Get-Random` call still runs the built-in
#: the metadata describes and is safe to drop.
_ALIAS_OF_ANOTHER_NAME = 'Set-Alias Some-Other-Name i*x'
_OTHER_NAME_TOKEN = 'Some-Other-Name'

#: The same aliaser pointed at the discarded call's OWN name, which genuinely rebinds `Get-Random`,
#: so keeping the call is correct here and both the coarse and a name-precise verdict agree.
_ALIAS_OF_OWN_NAME = 'Set-Alias Get-Random i*x'

#: A `function` redefinition of another name, called so it is a real surviving leak and not dead
#: code the pass sheds. Its `Start-Process calc` body cannot be dropped, so that token in the output
#: proves the redefinition stayed. A `function` is a per-name redefinition, so it leaves
#: `Get-Random` trustworthy — the behavior the aliaser above ought to share.
_FUNCTION_OF_ANOTHER_NAME = 'function Some-Other-Name { Start-Process calc }'
_CALL_OF_ANOTHER_NAME = 'Some-Other-Name'
_FUNCTION_LEAK_TOKEN = 'Start-Process calc'


class TestPs1DiscardedPureCallNamePrecisionControls(TestPs1):
    """
    The discarded pure call is dropped only while its bareword still names the built-in the metadata
    describes. These controls pin the verdict at both poles and for the rebinder families the tool
    already reads name-precisely, so the expected failure beside them is not vacuous: with a clean
    command table the call goes; with a rebinder of the call's own name reaching it the call stays;
    and a `function` redefinition of a different name — a per-name redefinition the model trusts
    around — leaves the call removable although the redefinition itself survives.
    """

    def test_a_discarded_pure_call_with_no_rebinder_anywhere_is_removed(self):
        for discard in _DISCARDS:
            with self.subTest(discard):
                result = self._deobfuscate(F'{discard}\n{_ANCHOR}')
                self.assertNotIn(_DISCARDED_TOKEN, result)
                self.assertIn(_ANCHOR_TOKEN, result)

    def test_a_discarded_pure_call_after_a_set_alias_of_its_own_name_is_kept(self):
        for discard in _DISCARDS:
            with self.subTest(discard):
                result = self._deobfuscate(F'{_ALIAS_OF_OWN_NAME}\n{discard}\n{_ANCHOR}')
                self.assertIn(_DISCARDED_TOKEN, result)
                self.assertIn('Set-Alias', result)
                self.assertIn(_ANCHOR_TOKEN, result)

    def test_a_wildcard_set_alias_of_another_name_survives_deobfuscation(self):
        result = self._deobfuscate(F'{_ALIAS_OF_ANOTHER_NAME}\n{_ANCHOR}')
        self.assertIn(_OTHER_NAME_TOKEN, result)
        self.assertIn('Set-Alias', result)
        self.assertIn(_ANCHOR_TOKEN, result)

    def test_a_discarded_pure_call_after_a_function_redefinition_of_another_name_is_removed(self):
        result = self._deobfuscate(
            F'{_FUNCTION_OF_ANOTHER_NAME}\n{_CALL_OF_ANOTHER_NAME}\n'
            F'$Null = Get-Random -Maximum 88175\n{_ANCHOR}')
        self.assertNotIn(_DISCARDED_TOKEN, result)
        self.assertIn(_FUNCTION_LEAK_TOKEN, result)
        self.assertIn(_OTHER_NAME_TOKEN, result)
        self.assertIn(_ANCHOR_TOKEN, result)


class TestPs1DiscardedPureCallAfterASetAliasOfAnotherNameShouldBeRemoved(TestPs1):
    """
    A `Set-Alias` that rebinds `Some-Other-Name` does not touch `Get-Random`, so the discarded
    `Get-Random` call still runs the built-in and is as removable as it is with no rebinder at
    all — which is exactly what happens when the rebinder is a `function` redefinition of the other
    name (see the control beside this). The aliaser survives, proving the world is not emptied.

    The tool's command-name trust is coarse across the command table: any surviving statement that
    opens it distrusts every name rather than only the one it rebinds, so the discarded call is
    kept. This expected failure encodes the correct behavior — the call is removed — and marks the
    open recall gap; it turns into an unexpected success the day trust becomes name-precise for
    aliasers, distrusting only the name the aliaser actually rebinds.
    """

    @unittest.expectedFailure
    def test_a_discarded_pure_call_after_a_set_alias_of_another_name_is_removed(self):
        for discard in _DISCARDS:
            with self.subTest(discard):
                result = self._deobfuscate(F'{_ALIAS_OF_ANOTHER_NAME}\n{discard}\n{_ANCHOR}')
                self.assertIn(_OTHER_NAME_TOKEN, result)
                self.assertIn(_ANCHOR_TOKEN, result)
                self.assertNotIn(_DISCARDED_TOKEN, result)
