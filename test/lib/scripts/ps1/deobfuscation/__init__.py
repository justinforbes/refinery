from __future__ import annotations

import inspect

from test import TestBase

from refinery.lib.scripts.ps1.deobfuscation import deobfuscate
from refinery.lib.scripts.ps1.model import Ps1Script
from refinery.lib.scripts.ps1.parser import Ps1Parser
from refinery.lib.scripts.ps1.synth import Ps1Synthesizer


class TestPs1(TestBase):

    def _deobfuscate(
        self,
        source: str,
        remove_junk: bool = True,
        preserve_bare_output: bool = False,
    ) -> str:
        ast = Ps1Parser(source).parse()
        deobfuscate(ast, remove_junk=remove_junk, preserve_bare_output=preserve_bare_output)
        return Ps1Synthesizer().convert(ast)

    def _deobfuscate_iterative(
        self,
        source: str,
        iterations: int = 100,
        remove_junk: bool = True,
        preserve_bare_output: bool = False,
    ) -> str:
        ast = Ps1Parser(source).parse()
        for _ in range(iterations):
            if not deobfuscate(
                ast, remove_junk=remove_junk, preserve_bare_output=preserve_bare_output
            ):
                break
        return Ps1Synthesizer().convert(ast)

    def _assertDeobfuscatesTo(self, source: str, expected: str) -> None:
        """
        Both arguments are written as ordinary indented PowerShell, and `expected` is rendered
        through the synthesizer before the comparison, so that brace layout cannot be mistaken for a
        statement having been removed.
        """
        self.assertEqual(
            self._deobfuscate(inspect.cleandoc(source)),
            self._apply(inspect.cleandoc(expected)),
        )

    def _assertKept(self, source: str) -> None:
        self._assertDeobfuscatesTo(source, source)

    def _assertRemoved(self, source: str, statement: str) -> None:
        """
        The expected output is `source` with `statement` gone and nothing else touched, so the pair
        of arguments spells out one removal rather than a whole rewritten script. Naming a statement
        that `source` does not contain would leave the expectation saying nothing, so it is refused.
        """
        self.assertIn(statement, source)
        self._assertDeobfuscatesTo(source, source.replace(statement, ''))

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
