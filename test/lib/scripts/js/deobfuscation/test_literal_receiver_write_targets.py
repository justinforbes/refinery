from __future__ import annotations

import unittest

from test import TestBase
from test.lib.scripts.js.analysis.differential import (
    behavior,
    deobfuscate_source,
    node_executable,
)


@unittest.skipUnless(node_executable() is not None, 'node.js is required')
class TestLiteralReceiverWriteTargets(TestBase):
    """
    A member or index access whose object is an array literal or a string literal is a value the tool is
    otherwise free to fold: `[1, 2, 3][0]` denotes `1`. In a write position it may not, because the access
    is not read but assigned, incremented, decremented, or deleted, and replacing it with the value it
    would read turns an assignment target into a constant. Each case below places such an access in a write
    position and asserts that deobfuscation preserves what the program does, with Node — not our own reading
    of the semantics — deciding what that is for the original and the deobfuscated form alike.
    """

    def _preserves(self, source: str):
        deobfuscated = deobfuscate_source(source)
        self.assertEqual(
            behavior(source),
            behavior(deobfuscated),
            F'deobfuscation changed observable behavior; result was:\n{deobfuscated}',
        )

    def test_assignment_targets_preserve_behavior(self):
        for source in [
            'console.log([1, 2, 3][0] = 9);',
            'console.log("abc"[0] = "x");',
            'console.log([1, 2, 3].length = 1);',
            'console.log([10, 20, 30][1] += 5);',
            'console.log("abc"[1] += "z");',
            'console.log([1, 2, 3].length += 2);',
        ]:
            with self.subTest(source=source):
                self._preserves(source)

    def test_increment_and_decrement_targets_preserve_behavior(self):
        for source in [
            'console.log([1, 2, 3][2]++);',
            'console.log(--[9, 8, 7][1]);',
            'console.log([1, 2, 3].length++);',
            'console.log("abc"[0]++);',
            'console.log(--"abc"[2]);',
        ]:
            with self.subTest(source=source):
                self._preserves(source)

    def test_delete_targets_preserve_behavior(self):
        for source in [
            'console.log(delete [1, 2, 3][0]);',
            'console.log(delete "abc"[0]);',
            'console.log(delete [1, 2, 3].length);',
        ]:
            with self.subTest(source=source):
                self._preserves(source)
