import ast
import functools
import os.path
import pytest
import re

from glob import glob

from . import TestBase, TestUnitBase


#: The module that owns statement removal, and so the one place a removal may be spelled out. Path
#: anchored rather than matched by base name, so a `removal.py` elsewhere in the tree is still read.
_RATCHET_OWNER = os.path.join('deobfuscation', 'removal.py')

#: A `try` block, as opposed to the letters that end a word such as `registry`.
_PROTECTED_BODY = re.compile(R'\btry\b\s*\{')

#: A `catch` whose body says something. An empty handler is deliberately not one that acts — see
#: `refinery.lib.scripts.ps1.analysis.effects.fault_is_observed` — so a removal that empties the
#: body beside it is never refused, and a test written that way reaches none of the veto below.
_ACTING_HANDLER = re.compile(R'\bcatch\b[^{]*\{\s*[^\s}]')


def _project_root() -> str:
    """
    The repository root, found by walking up from this file. Every check below globs source trees
    relative to it, and a check whose glob comes back empty passes on nothing at all, so the walk
    is written once rather than repeated at each call site.
    """
    root = os.path.abspath(__file__)
    for _ in range(3):
        root = os.path.dirname(root)
    return root


@functools.lru_cache(maxsize=None)
def _parse(path: str) -> ast.Module:
    """
    The parsed source at `path`. Cached because the checks below overlap: the removal ratchet reads
    every `ps1` source and the protected-body ratchet reads the deobfuscation package again.
    """
    with open(path, 'r', encoding='utf8') as stream:
        return ast.parse(stream.read(), path)


def _sources(*parts: str) -> list[tuple[str, ast.Module]]:
    """
    Every Python source under the given path, relative to the repository root, beside its parse. A
    glob that comes back empty is an error rather than a result, for the reason `_project_root`
    gives: a check with nothing to read passes on nothing at all.
    """
    directory = os.path.join(*parts)
    paths = glob(os.path.join(_project_root(), directory, '**', '*.py'), recursive=True)
    if not paths:
        raise FileNotFoundError(F'no sources found under {directory}')
    return [(path, _parse(path)) for path in sorted(paths)]


class TestRemovalDiscipline(TestUnitBase):
    """
    A statement is removed from a PowerShell body in exactly one place. Every removal has to pass
    the vetoes that keep a payload from being deleted, and a pass that deletes a statement by any
    other route reaches none of them — the guard set it skips is invisible at the call site and
    stays invisible in review, which is how the two removal sites in `unused.py` came to answer the
    fault question differently. This is a ratchet: the exempt list only shrinks.

    Both spellings of a bypass are looked for, because both have occurred. `_remove_from_parent` is
    searched for by *name*, anywhere in the module, so that reaching it through the module object
    rather than through a `from ... import` is caught too. `BodyEdit` is searched for by *use*: a
    splice whose replacement is the empty list is a deletion however it is spelled, and that is the
    form in which `emulator.py` and `unflatten.py` deleted statements past every veto while the
    import check reported the package clean.
    """

    #: Packages whose deobfuscation passes still remove statements without a removal plan. Every
    #: name here is work not yet done, not a language that is exempt on principle.
    UNRATCHETED = ('js',)

    @classmethod
    def _removes_statements(cls, tree: ast.Module) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == '_remove_from_parent':
                return True
            if isinstance(node, ast.Attribute) and node.attr == '_remove_from_parent':
                return True
            if isinstance(node, ast.alias) and node.name == '_remove_from_parent':
                return True
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != 'splice':
                continue
            given = [*node.args, *(keyword.value for keyword in node.keywords)]
            if any(isinstance(item, ast.List) and not item.elts for item in given):
                return True
        return False

    @classmethod
    def _offenders(cls, language: str) -> list[str]:
        root = _project_root()
        return [
            os.path.relpath(path, root)
            for path, tree in _sources('refinery', 'lib', 'scripts', language)
            if not path.endswith(_RATCHET_OWNER)
            and cls._removes_statements(tree)
        ]

    def test_no_ps1_pass_removes_a_statement_outside_a_removal_plan(self):
        self.assertListEqual(
            self._offenders('ps1'), [], 'use refinery.lib.scripts.ps1.deobfuscation.removal')

    def test_the_unratcheted_languages_are_still_unratcheted(self):
        for language in self.UNRATCHETED:
            if not self._offenders(language):
                self.fail(F'{language} no longer needs the exemption; drop it from UNRATCHETED')


class TestRemovalIsTriedInsideAProtectedBody(TestBase):
    """
    A removal that empties a `try` body beside an acting handler makes the handler unreachable, and
    every pass answers that question through the same veto. Both bugs this ratchet was written for
    lived in exactly that shape and both survived suites that were otherwise well formed, so the
    shape is required rather than hoped for: what a suite never puts inside a `try` it never checks.

    A pass is credited by the test *method* that exercises it, not by the single call, because the
    script a method feeds to a pass is as often bound to a local first — a call-scoped scan reads
    such a method as evidence of nothing and demands a duplicate of the test that already exists.
    """

    PASSES = os.path.join('refinery', 'lib', 'scripts', 'ps1', 'deobfuscation')
    TESTS = os.path.join('test', 'lib', 'scripts', 'ps1', 'deobfuscation')
    PLANS = ('Ps1RemovalPlan', 'Ps1RemovalPlans')
    HELPERS = ('_apply', '_assertUnchanged', '_assertTreeIsIntact')

    @classmethod
    def _passes_that_remove(cls) -> set[str]:
        found = set()
        for path, tree in _sources(cls.PASSES):
            if path.endswith(_RATCHET_OWNER):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for call in ast.walk(node):
                    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                        continue
                    if call.func.id in cls.PLANS:
                        found.add(node.name)
        return found

    @staticmethod
    def _scripts(node: ast.AST) -> list[str]:
        return [
            item.value for item in ast.walk(node)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        ]

    @classmethod
    def _passes_tried_under_a_handler(cls) -> set[str]:
        found = set()
        for _, tree in _sources(cls.TESTS):
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                if not any(
                    _PROTECTED_BODY.search(script) and _ACTING_HANDLER.search(script)
                    for script in cls._scripts(node)
                ):
                    continue
                for call in ast.walk(node):
                    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                        continue
                    if call.func.attr not in cls.HELPERS:
                        continue
                    found.update(arg.id for arg in call.args if isinstance(arg, ast.Name))
        return found

    def test_every_pass_that_removes_is_tried_inside_a_protected_body(self):
        missing = sorted(self._passes_that_remove() - self._passes_tried_under_a_handler())
        self.assertListEqual(
            missing, [], 'apply each of these to a removal inside try/catch and pin what survives')


class TestStyleGuides(TestUnitBase):

    @pytest.mark.cosmetics
    def test_happy_flakes(self):
        import pyflakes.api
        import pyflakes.reporter
        import io

        python_files = [path for path in glob(
            os.path.join(_project_root(), 'refinery', '**', '*.py'), recursive=True)
            if 'thirdparty' not in path]

        alerts = io.StringIO()
        errors = io.StringIO()

        for path in python_files:
            with open(path, 'r', encoding='utf8') as stream:
                code = stream.read()
            pyflakes.api.check(code, path, pyflakes.reporter.Reporter(alerts, errors))

        error_log = errors.getvalue().strip().splitlines(False)
        alert_log = alerts.getvalue().strip().splitlines(False)

        error_log.extend(line for line in alert_log if not any(
            ignore in line for ignore in [
                ': undefined name',
                ': syntax error in forward annotation',
            ]
        ))

        if error_log:
            print()
        for error in error_log:
            print(error)

        self.assertListEqual(error_log, [])

    @pytest.mark.cosmetics
    def test_style_guide(self):
        import pycodestyle

        class RespectFlake8NoQA(pycodestyle.StandardReport):
            def error(self, lno, offset, text, check):
                for line in self.lines[:5]:
                    _, _, noqa = line.partition('flake8:')
                    if noqa.lstrip().startswith('noqa'):
                        return
                line: str = self.lines[lno - 1]
                _, _, comment = line.partition('#')
                if comment.lower().strip().startswith('noqa'):
                    return
                super().error(lno, offset, text, check)

        stylez = pycodestyle.StyleGuide(
            ignore=[
                'E128',  # A continuation line is under-indented for a visual indentation.
                'E203',  # Colons should not have any space before them.
                'W503',  # Line break occurred before a binary operator
                'E261',  # at least two spaces before inline comment
            ],
            max_line_length=140,
            reporter=RespectFlake8NoQA,
        )

        python_files = [path for path in glob(
            os.path.join(_project_root(), 'refinery', '**', '*.py'), recursive=True)
            if 'thirdparty' not in path]

        for file in python_files:
            with open(file, 'rb') as code_lines:
                for k, line in enumerate(code_lines):
                    self.assertFalse(line.endswith(b'\r\n'), F'CRLF sequence on line {k} in {file}')

        report = stylez.check_files(python_files)
        self.assertEqual(report.total_errors, 0, 'PEP8 formatting errors were found.')
