from __future__ import annotations

import inspect

from typing import NamedTuple

from test import TestBase

from refinery.lib.scripts import Expression, UnspellableNode, canonical, is_well_formed
from refinery.lib.scripts.guess import guess_language
from refinery.lib.scripts.js.model import (
    JsBlockStatement,
    JsFunctionDeclaration,
    JsScript,
    JsStringLiteral,
    JsTemplateElement,
    JsTemplateLiteral,
    JsVariableDeclaration,
)
from refinery.lib.scripts.js.parser import JsParser
from refinery.lib.scripts.js.synth import JsSynthesizer


_TOP_LEVEL = inspect.cleandoc("""
    const registry = {};

    function register(name, handler) {
      registry[name] = handler;
      return handler;
    }

    register('alpha', function (input) {
      return input.split(',').map(function (part) {
        return part.trim();
      });
    });
""") + '\n'

_FUNCTION_BODY = _TOP_LEVEL + inspect.cleandoc("""
    function describe(name) {
      const handler = registry[name];
""") + '\n'

_STATEMENTS_BEFORE_THE_CUT = 3


class Truncation(NamedTuple):
    """
    A file that stops in the middle of one construct, together with the text that would have
    finished it. `cut` is what an analyst carves out of memory and `whole` is the file it was cut
    from, so a difference between the two is caused by the cut and nothing else.
    """
    head: str
    opened: str
    closing: str

    @property
    def cut(self) -> str:
        return F'{self.head}{self.opened}'

    @property
    def whole(self) -> str:
        return F'{self.head}{self.opened}{self.closing}'


_TRUNCATIONS = {
    'string_at_top_level': Truncation(
        _TOP_LEVEL,
        "const banner = 'loading the alpha module",
        "';\n",
    ),
    'template_at_top_level': Truncation(
        _TOP_LEVEL,
        'const banner = `loading ${registry.alpha} and everything after it',
        '`;\n',
    ),
    'template_hole_at_top_level': Truncation(
        _TOP_LEVEL,
        'const banner = `loading ${Object.keys(registry',
        ').length} modules`;\n',
    ),
    'regexp_at_top_level': Truncation(
        _TOP_LEVEL,
        'const pattern = /^alpha-[0-9]+',
        '$/;\n',
    ),
    'comment_at_top_level': Truncation(
        _TOP_LEVEL,
        '/* the registry maps each name to the handler that reads it',
        " */\nregister('beta', registry.alpha);\n",
    ),
    'string_in_function_body': Truncation(
        _FUNCTION_BODY,
        "  const label = 'describing the handler for ",
        "';\n}\n",
    ),
    'template_in_function_body': Truncation(
        _FUNCTION_BODY,
        '  const label = `handler ${handler} registered for ',
        '`;\n}\n',
    ),
    'template_hole_in_function_body': Truncation(
        _FUNCTION_BODY,
        '  const label = `handler ${handler.toString(',
        ')} for ${name}`;\n}\n',
    ),
    'regexp_in_function_body': Truncation(
        _FUNCTION_BODY,
        '  const clean = name.replace(/[^a-z0-9',
        "]+/g, '-');\n}\n",
    ),
    'comment_in_function_body': Truncation(
        _FUNCTION_BODY,
        '  /* the handler is the function that was registered under this name',
        ' */\n  return handler;\n}\n',
    ),
}


def _string_continued_over(line_ending: str) -> str:
    return F"{_TOP_LEVEL}const banner = 'loading \\{line_ending}the alpha module';\n"


def _string_holding(separator: str) -> str:
    return F"{_TOP_LEVEL}const banner = 'loading{separator}the module';\n"


_INTACT = {
    'line_continuation_lf': _string_continued_over('\n'),
    'line_continuation_crlf': _string_continued_over('\r\n'),
    'line_continuation_cr': _string_continued_over('\r'),
    'line_separator_in_string': _string_holding(chr(0x2028)),
    'paragraph_separator_in_string': _string_holding(chr(0x2029)),
    'template_across_lines': _TOP_LEVEL + inspect.cleandoc("""
        const banner = `loading
        the alpha
        module`;
    """) + '\n',
    'slash_inside_character_class': F'{_TOP_LEVEL}const pattern = /^[/a-z]+$/;\n',
}


def _last_initializer(script: JsScript) -> Expression:
    """
    The value the last declaration of *script* is given, reached through the function body where the
    corpus put that declaration inside one.
    """
    statement = script.body[-1]
    if isinstance(statement, JsFunctionDeclaration):
        assert isinstance(statement.body, JsBlockStatement)
        statement = statement.body.body[-1]
    assert isinstance(statement, JsVariableDeclaration)
    initializer = statement.declarations[0].init
    assert initializer is not None
    return initializer


class TestTruncatedSource(TestBase):

    def _print(self, script: JsScript, unescape_strings: bool = False) -> str:
        return JsSynthesizer(unescape_strings=unescape_strings).convert(script)

    def _string_denoted_by(self, source: str) -> tuple[str, bool]:
        literal = _last_initializer(JsParser(source).parse())
        assert isinstance(literal, JsStringLiteral)
        return literal.value, literal.terminated

    def _template_runs_of(self, source: str) -> list[tuple[str | None, bool, bool]]:
        literal = _last_initializer(JsParser(source).parse())
        assert isinstance(literal, JsTemplateLiteral)
        return [(run.value, run.tail, run.terminated) for run in literal.quasis]

    def test_everything_before_the_cut_parses_to_what_the_whole_file_parses_to(self):
        for name, truncation in _TRUNCATIONS.items():
            with self.subTest(name):
                cut = JsParser(truncation.cut).parse()
                whole = JsParser(truncation.whole).parse()
                self.assertEqual(
                    [canonical(node) for node in cut.body[:_STATEMENTS_BEFORE_THE_CUT]],
                    [canonical(node) for node in whole.body[:_STATEMENTS_BEFORE_THE_CUT]],
                )

    def test_the_construct_the_cut_broke_still_becomes_a_statement_unless_it_was_a_comment(self):
        expected = {
            'string_at_top_level': 4,
            'template_at_top_level': 4,
            'template_hole_at_top_level': 4,
            'regexp_at_top_level': 4,
            'comment_at_top_level': 3,
            'string_in_function_body': 4,
            'template_in_function_body': 4,
            'template_hole_in_function_body': 4,
            'regexp_in_function_body': 4,
            'comment_in_function_body': 4,
        }
        for name, count in expected.items():
            with self.subTest(name):
                self.assertEqual(len(JsParser(_TRUNCATIONS[name].cut).parse().body), count)

    def test_a_string_the_cut_left_open_keeps_its_text_and_records_the_missing_quote(self):
        self.assertEqual(
            self._string_denoted_by(_TRUNCATIONS['string_at_top_level'].cut),
            ('loading the alpha module', False),
        )
        self.assertEqual(
            self._string_denoted_by(_TRUNCATIONS['string_at_top_level'].whole),
            ('loading the alpha module', True),
        )
        self.assertEqual(
            self._string_denoted_by(_TRUNCATIONS['string_in_function_body'].cut),
            ('describing the handler for ', False),
        )
        self.assertEqual(
            self._string_denoted_by(_TRUNCATIONS['string_in_function_body'].whole),
            ('describing the handler for ', True),
        )

    def test_a_template_the_cut_left_open_keeps_its_runs_and_records_the_missing_delimiter(self):
        expected = {
            'template_at_top_level': [
                ('loading ', False, True),
                (' and everything after it', True, False),
            ],
            'template_hole_at_top_level': [
                ('loading ', False, True),
                ('', True, False),
            ],
            'template_in_function_body': [
                ('handler ', False, True),
                (' registered for ', True, False),
            ],
            'template_hole_in_function_body': [
                ('handler ', False, True),
                ('', True, False),
            ],
        }
        for name, runs in expected.items():
            with self.subTest(name):
                self.assertEqual(self._template_runs_of(_TRUNCATIONS[name].cut), runs)

    def test_a_template_the_cut_completed_is_terminated(self):
        expected = {
            'template_at_top_level': [
                ('loading ', False, True),
                (' and everything after it', True, True),
            ],
            'template_hole_at_top_level': [
                ('loading ', False, True),
                (' modules', True, True),
            ],
            'template_in_function_body': [
                ('handler ', False, True),
                (' registered for ', True, True),
            ],
            'template_hole_in_function_body': [
                ('handler ', False, True),
                (' for ', False, True),
                ('', True, True),
            ],
        }
        for name, runs in expected.items():
            with self.subTest(name):
                self.assertEqual(self._template_runs_of(_TRUNCATIONS[name].whole), runs)

    def test_a_tree_is_well_formed_after_the_cut_only_where_no_literal_stayed_open(self):
        expected = {
            'string_at_top_level': False,
            'template_at_top_level': False,
            'template_hole_at_top_level': False,
            'regexp_at_top_level': False,
            'comment_at_top_level': True,
            'string_in_function_body': False,
            'template_in_function_body': False,
            'template_hole_in_function_body': False,
            'regexp_in_function_body': False,
            'comment_in_function_body': True,
        }
        for name, well_formed in expected.items():
            with self.subTest(name):
                script = JsParser(_TRUNCATIONS[name].cut).parse()
                self.assertEqual(is_well_formed(script), well_formed)

    def test_every_whole_file_and_every_intact_file_is_well_formed(self):
        sources = {name: case.whole for name, case in _TRUNCATIONS.items()}
        sources.update(_INTACT)
        for name, source in sources.items():
            with self.subTest(name):
                self.assertEqual(is_well_formed(JsParser(source).parse()), True)

    def test_the_synthesizer_refuses_the_literal_the_cut_left_open(self):
        expected = {
            'string_at_top_level': JsStringLiteral,
            'template_at_top_level': JsTemplateElement,
            'template_hole_at_top_level': JsTemplateElement,
            'string_in_function_body': JsStringLiteral,
            'template_in_function_body': JsTemplateElement,
            'template_hole_in_function_body': JsTemplateElement,
        }
        for name, node_type in expected.items():
            for unescape_strings in (False, True):
                with self.subTest(name, unescape_strings=unescape_strings):
                    script = JsParser(_TRUNCATIONS[name].cut).parse()
                    with self.assertRaises(UnspellableNode) as refusal:
                        self._print(script, unescape_strings)
                    self.assertEqual(type(refusal.exception.node), node_type)

    def test_the_synthesizer_prints_the_cut_regexp_as_the_arithmetic_it_was_reread_as(self):
        expected = {
            'regexp_at_top_level': ['const pattern = / ^ alpha - [0 - 9] + ;'],
            'regexp_in_function_body': [
                '  const clean = name.replace(/[^] - z0 - 9);',
                '}',
            ],
        }
        for name, tail in expected.items():
            for unescape_strings in (False, True):
                with self.subTest(name, unescape_strings=unescape_strings):
                    script = JsParser(_TRUNCATIONS[name].cut).parse()
                    printed = self._print(script, unescape_strings)
                    self.assertEqual(printed.splitlines()[-len(tail):], tail)

    def test_printing_the_cut_regexp_again_is_a_fixed_point_only_in_the_function_body(self):
        expected = {
            'regexp_at_top_level': ['const pattern = / ^ alpha - [0 - 9] + ;;'],
            'regexp_in_function_body': [
                '  const clean = name.replace(/[^] - z0 - 9);',
                '}',
            ],
        }
        for name, tail in expected.items():
            for unescape_strings in (False, True):
                with self.subTest(name, unescape_strings=unescape_strings):
                    once = self._print(JsParser(_TRUNCATIONS[name].cut).parse(), unescape_strings)
                    twice = self._print(JsParser(once).parse(), unescape_strings)
                    self.assertEqual(twice.splitlines()[-len(tail):], tail)

    def test_the_comment_the_cut_left_open_leaves_no_trace_in_the_output(self):
        for name in ('comment_at_top_level', 'comment_in_function_body'):
            for unescape_strings in (False, True):
                with self.subTest(name, unescape_strings=unescape_strings):
                    truncation = _TRUNCATIONS[name]
                    without = self._print(JsParser(truncation.head).parse(), unescape_strings)
                    printed = self._print(JsParser(truncation.cut).parse(), unescape_strings)
                    self.assertEqual(printed, without)

    def test_a_cut_that_prints_round_trips_in_text_and_in_tree_or_in_neither(self):
        expected = {
            'regexp_at_top_level': False,
            'regexp_in_function_body': True,
            'comment_at_top_level': True,
            'comment_in_function_body': True,
        }
        for name, stable in expected.items():
            for unescape_strings in (False, True):
                with self.subTest(name, unescape_strings=unescape_strings):
                    script = JsParser(_TRUNCATIONS[name].cut).parse()
                    printed = self._print(script, unescape_strings)
                    again = JsParser(printed).parse()
                    self.assertEqual(self._print(again, unescape_strings) == printed, stable)
                    self.assertEqual(canonical(again) == canonical(script), stable)

    def test_a_whole_file_prints_to_text_that_prints_and_parses_to_itself(self):
        for name, truncation in _TRUNCATIONS.items():
            for unescape_strings in (False, True):
                with self.subTest(name, unescape_strings=unescape_strings):
                    script = JsParser(truncation.whole).parse()
                    printed = self._print(script, unescape_strings)
                    again = JsParser(printed).parse()
                    self.assertEqual(self._print(again, unescape_strings), printed)
                    self.assertEqual(canonical(again), canonical(script))

    def test_an_intact_file_prints_to_text_that_prints_and_parses_to_itself(self):
        for name, source in _INTACT.items():
            for unescape_strings in (False, True):
                with self.subTest(name, unescape_strings=unescape_strings):
                    script = JsParser(source).parse()
                    printed = self._print(script, unescape_strings)
                    again = JsParser(printed).parse()
                    self.assertEqual(self._print(again, unescape_strings), printed)
                    self.assertEqual(canonical(again), canonical(script))

    def test_a_line_continuation_contributes_nothing_to_the_string_it_breaks(self):
        for name in ('line_continuation_lf', 'line_continuation_crlf', 'line_continuation_cr'):
            with self.subTest(name):
                self.assertEqual(
                    self._string_denoted_by(_INTACT[name]),
                    ('loading the alpha module', True),
                )

    def test_a_line_separator_inside_a_string_is_one_more_character_of_it(self):
        self.assertEqual(
            self._string_denoted_by(_INTACT['line_separator_in_string']),
            (F'loading{chr(0x2028)}the module', True),
        )
        self.assertEqual(
            self._string_denoted_by(_INTACT['paragraph_separator_in_string']),
            (F'loading{chr(0x2029)}the module', True),
        )

    def test_a_template_may_span_lines_without_being_cut(self):
        self.assertEqual(
            self._template_runs_of(_INTACT['template_across_lines']),
            [('loading\nthe alpha\nmodule', True, True)],
        )

    def test_a_slash_inside_a_character_class_does_not_end_the_regexp(self):
        self.assertEqual(
            canonical(_last_initializer(JsParser(_INTACT['slash_inside_character_class']).parse())),
            ('JsRegExpLiteral', '^[/a-z]+$', ''),
        )

    def test_the_language_is_still_recognized_after_the_cut(self):
        sources = {F'{name}::cut': case.cut for name, case in _TRUNCATIONS.items()}
        sources.update({F'{name}::whole': case.whole for name, case in _TRUNCATIONS.items()})
        sources.update(_INTACT)
        for name, source in sources.items():
            with self.subTest(name):
                self.assertEqual(guess_language(source), 'js')
