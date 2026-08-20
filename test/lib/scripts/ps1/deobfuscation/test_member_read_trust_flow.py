from __future__ import annotations

import unittest

from test.lib.scripts.ps1.deobfuscation import TestPs1

#: An acting statement that is never a removal candidate; a `Write-Host` writes to the host and can
#: never be dropped, so its survival says only that the pass did not empty the script wholesale.
_ANCHOR = "Write-Host 'ANCHOR_SURVIVES'"

#: A discarded pure member read on the result of a pure typed cmdlet: `(Get-Date)` yields a `DateTime`
#: and `.Ticks` is a plain property read with no side effect, so `$Null =` sinks a value nothing
#: observes. Neither `Get-Date` nor `Ticks` occurs in the anchor or the leak, so the rendered read is
#: a distinctive marker whose absence from the output means the whole statement was removed.
_MEMBER_READ = '$Null = (Get-Date).Ticks'

#: An opaque rebinder of any command name. Placed after the read, it cannot have run at the read's
#: position and so cannot change what `(Get-Date)` resolves to before the read executes.
_LEAK = 'Invoke-Expression $enc'


class TestPs1APureMemberReadFollowsTheSamePositionalTrust(TestPs1):
    """
    A discarded pure member read on a pure typed cmdlet result is junk when nothing that could
    change the command world has run before it, exactly as a discarded pure command is. Trust is
    positional: a leak that only follows the read cannot have run at the read's position, so it
    cannot make `(Get-Date)` resolve to anything other than the built-in whose result carries the
    side-effect-free `.Ticks` property. With the leak ahead of the read instead, the read may observe
    a rebound world and is soundly kept.
    """

    def test_a_pure_member_read_with_no_leak_anywhere_is_removed(self):
        self.assertEqual(self._deobfuscate(F'{_MEMBER_READ}\n{_ANCHOR}'), _ANCHOR)

    def test_a_pure_member_read_after_a_leak_reachable_before_it_is_kept(self):
        source = F'{_LEAK}\n{_MEMBER_READ}\n{_ANCHOR}'
        self.assertEqual(self._deobfuscate(source), source)


class TestPs1APureMemberReadBeforeTheOnlyLeakShouldBeRemoved(TestPs1):
    """
    The command-name trust pass already removes a discarded pure command placed before the only
    later leak, because the leak cannot have run at that position. A member read's purity depends in
    addition on resolving the receiver's type, and that type resolution is currently decided over the
    whole run rather than positionally, so the later leak makes the result type unresolvable and the
    read is conservatively kept. The correct, safe verdict is removal — nothing that changes the
    world has run before the read — so this pin tracks the recall gap and starts passing the day
    type-resolution trust becomes positional too.
    """

    @unittest.expectedFailure
    def test_a_pure_member_read_before_the_only_later_leak_is_removed(self):
        self.assertEqual(
            self._deobfuscate(F'{_MEMBER_READ}\n{_LEAK}\n{_ANCHOR}'),
            F'{_LEAK}\n{_ANCHOR}')
