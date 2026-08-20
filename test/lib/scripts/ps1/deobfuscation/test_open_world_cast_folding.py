from __future__ import annotations

import unittest

from test.lib.scripts.ps1.deobfuscation import TestPs1

#: An opaque rebinder of the type world. `Invoke-Expression` runs text this analysis cannot read, and
#: that text may re-point a type accelerator through `TypeAccelerators::Remove` and `::Add`, so after
#: it no spelling of a type name is known to denote the type the metadata describes.
_LEAK = 'Invoke-Expression $payload'

#: An acting statement that is never a removal candidate, so its survival says only that the pass did
#: not empty the script wholesale.
_ANCHOR = "Write-Host 'ANCHOR_SURVIVES'"


class TestPs1AnOpenWorldCastIsFoldedThoughTheEffectModelRefusesIt(TestPs1):
    """
    A cast is a conversion the engine performs by calling into the target type, so a re-pointed
    accelerator makes `[Int]'45'` run code of the payload's choosing. The effect model says so: it
    answers that the expression is not side-effect-free at a position an `Invoke-Expression` reaches.
    The constant folder evaluates it anyway — the value domain carries no world — and the resulting
    literal turns a statement the effect model would have kept into a dead store nothing preserves.

    These pin the inconsistency, not a preference between the two answers: the same script asked of
    the two layers gets opposite verdicts, and the folder's is the one that survives into the output.
    """

    @unittest.expectedFailure
    def test_a_cast_after_a_leak_is_not_folded(self):
        source = F'{_LEAK}\n$q = [Int]"45" + 5\nWrite-Host $q'
        self.assertEqual(self._deobfuscate(source), source)

    @unittest.expectedFailure
    def test_a_discarded_cast_after_a_leak_is_kept(self):
        source = F'{_LEAK}\n$Null = [Int]"45" + 5\n{_ANCHOR}'
        self.assertEqual(self._deobfuscate(source), source)


class TestPs1TheWorldGateHoldsForEverythingButTheCast(TestPs1):
    """
    Controls for the pins above. The folder is not simply world-blind: a static method call and a
    conversion that parses are both held after the same leak. Only the primitive cast crosses.
    """

    def test_a_static_call_after_a_leak_is_kept(self):
        source = F'{_LEAK}\n$Null = [Math]::Sqrt(36)\n{_ANCHOR}'
        self.assertEqual(self._deobfuscate(source), source)

    def test_a_parsing_cast_after_a_leak_is_kept(self):
        source = F'{_LEAK}\n$q = [Xml]"<r/>"\nWrite-Host $q'
        self.assertEqual(self._deobfuscate(source), source)

    def test_a_cast_with_no_leak_anywhere_is_folded(self):
        self.assertEqual(
            self._deobfuscate(F'$q = [Int]"45" + 5\nWrite-Host $q'),
            'Write-Host 50')
