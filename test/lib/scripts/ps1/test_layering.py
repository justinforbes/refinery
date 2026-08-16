"""
Which way the ps1 packages are allowed to depend on each other.
"""
from __future__ import annotations

import ast
import pathlib
import unittest

_ANALYSIS = pathlib.Path('refinery/lib/scripts/ps1/analysis')
_FORBIDDEN = 'refinery.lib.scripts.ps1.deobfuscation'
_PACKAGE = 'refinery.lib.scripts.ps1.analysis'

#: The analysis modules that answer a question out of the shipped metadata and the AST alone, so
#: that every other module in the package may consult them and none of them can be caught in a cycle
#: with one. Declared rather than derived: a module that acquires a dependency on the package has to
#: say so here by leaving, which is the change worth noticing.
_RANK_ZERO = frozenset({
    'arguments.py',
    'naming.py',
})


def _imported_modules(source: str) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module)
    return imported


class TestPs1AnalysisDoesNotDependOnDeobfuscation(unittest.TestCase):
    """
    The analysis layer answers what PowerShell does and the deobfuscation layer decides what to
    rewrite, so every question a pass asks is a call downwards and never one back up. Nothing
    enforced that before: it held only because nobody had needed to reach the other way yet, and a
    single import would have made the domain depend on the policy that consults it.
    """

    def _modules(self) -> list[pathlib.Path]:
        return sorted(_ANALYSIS.glob('*.py'))

    def test_the_analysis_package_is_where_it_is_expected_to_be(self):
        self.assertNotEqual(self._modules(), [])

    def test_no_analysis_module_imports_from_the_deobfuscation_package(self):
        offenders = sorted(
            F'{path.name} imports {name}'
            for path in self._modules()
            for name in _imported_modules(path.read_text(encoding='utf8'))
            if name == _FORBIDDEN or name.startswith(F'{_FORBIDDEN}.')
        )
        self.assertEqual(offenders, [])

    def test_every_rank_zero_module_is_where_it_is_expected_to_be(self):
        self.assertEqual(sorted(_RANK_ZERO - {path.name for path in self._modules()}), [])

    def test_no_rank_zero_module_imports_another_analysis_module(self):
        """
        The package has a bottom, and it is what keeps a module every other one consults out of a
        cycle with any of them. `arguments` answers which slots of a call the callee writes through,
        which the semantic model asks while it is being built, so a dependency of it on the model
        would be a circular import rather than a layering opinion.
        """
        offenders = sorted(
            F'{path.name} imports {name}'
            for path in self._modules()
            if path.name in _RANK_ZERO
            for name in _imported_modules(path.read_text(encoding='utf8'))
            if name == _PACKAGE or name.startswith(F'{_PACKAGE}.')
        )
        self.assertEqual(offenders, [])
