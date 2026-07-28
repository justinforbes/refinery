from __future__ import annotations

from test import TestBase

from refinery.lib.scripts.ps1.deobfuscation import deobfuscate
from refinery.lib.scripts.ps1.model import Ps1Script
from refinery.lib.scripts.ps1.parser import Ps1Parser
from refinery.lib.scripts.ps1.synth import Ps1Synthesizer


class TestPs1(TestBase):

    def _deobfuscate(self, source: str, remove_junk: bool = True) -> str:
        ast = Ps1Parser(source).parse()
        deobfuscate(ast, remove_junk=remove_junk)
        return Ps1Synthesizer().convert(ast)

    def _deobfuscate_iterative(self, source: str, iterations: int = 100, remove_junk: bool = True) -> str:
        ast = Ps1Parser(source).parse()
        for _ in range(iterations):
            if not deobfuscate(ast, remove_junk=remove_junk):
                break
        return Ps1Synthesizer().convert(ast)

    def _transform(self, source: str, *transforms) -> Ps1Script:
        ast = Ps1Parser(source).parse()
        for transform in transforms:
            transform().visit(ast)
        return ast

    def _apply(self, source: str, *transforms) -> str:
        return Ps1Synthesizer().convert(self._transform(source, *transforms))

    def _assertUnchanged(self, source: str, *transforms) -> None:
        """
        `source` has to be given in the synthesizer's own rendering, since it stands in for the
        expected output as well.
        """
        self.assertEqual(self._apply(source, *transforms), source)

    def _assertTreeIsIntact(self, source: str, expected: str, *transforms) -> None:
        """
        `expected` is required because a stale parent pointer is left by a pass *building* a
        replacement, so a pass that stops building one satisfies the pointer assertion by doing
        nothing at all.
        """
        ast = self._transform(source, *transforms)
        self.assertEqual(Ps1Synthesizer().convert(ast), expected)
        for node in ast.walk():
            for child in node.children():
                self.assertIs(child.parent, node, F'{child!r} in {node!r} names {child.parent!r}')
