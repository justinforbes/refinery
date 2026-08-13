from __future__ import annotations

import inspect

from test import TestBase

from refinery.lib.scripts.js.model import (
    JsAssignmentExpression,
    JsBinaryExpression,
    JsErrorNode,
    JsRegExpLiteral,
)
from refinery.lib.scripts.js.parser import JsParser

DIVISION = '/'
DIVISION_ASSIGNMENT = '/='
PARSE_ERROR = 'parse error'

AFTER_A_COMPLETE_EXPRESSION = [
    'a/ zzz',
    '1/ zzz',
    '1n/ zzz',
    "'a'/ zzz",
    '`a`/ zzz',
    'true/ zzz',
    'null/ zzz',
    'this/ zzz',
    'f(a)/ zzz',
    '(a)/ zzz',
    'a[0]/ zzz',
    'a?.b/ zzz',
    'a++/ zzz',
    '++a/ zzz',
    'new a/ zzz',
    'tag`a`/ zzz',
    'a.return/ zzz',
    'a.in/ zzz',
    '({ if: 1 })/ zzz',
    '(() => {})/ zzz',
    'async/ zzz',
    'async (x)/ zzz',
    'let/ zzz',
    'of/ zzz',
]

AT_THE_START_OF_A_STATEMENT = [
    '/zzz/.test(a)',
    'a;/zzz/.test(b)',
    'lbl:/zzz/.test(a)',
    '{/zzz/.test(a) }',
    '{ }/zzz/.test(a)',
    'if (a) {}/zzz/.test(b)',
    'for (;;) {}/zzz/.test(a)',
    'switch (a) {}/zzz/.test(b)',
    'try {} finally {}/zzz/.test(a)',
    'x => {/zzz/.test(a) }',
]

WHERE_AN_OPERAND_IS_EXPECTED = [
    'f(/zzz/)',
    '[a,/zzz/]',
    'var x =/zzz/;',
    'a +/zzz/',
    '!/zzz/.test(a)',
    'typeof/zzz/',
    'void/zzz/',
    'a instanceof/zzz/',
    'a in/zzz/',
    'a ?/zzz/ : b',
    'a ? b :/zzz/',
    '({ a:/zzz/ })',
    'a[/zzz/.source]',
    'x =>/zzz/',
    'async x =>/zzz/',
    'throw/zzz/;',
    'function f(){ return/zzz/; }',
    'function f(a =/zzz/){}',
    'function* g(){ yield*/zzz/; }',
    'class C { m(){ return/zzz/; } }',
    'switch (a) { case/zzz/: break; }',
    'if (/zzz/.test(a)) ;',
    'for (a;/zzz/.test(b); ) ;',
    'for (a of/zzz/) ;',
]

AFTER_A_CONTROL_STRUCTURE_HEADER = [
    'if (a)/zzz/.test(b);',
    'if (a) ; else/zzz/.test(b);',
    'while (a)/zzz/.test(b);',
    'do/zzz/.test(a); while (b);',
]

AROUND_A_CLOSING_BRACE = [
    ('function f(){}/zzz/.test(a)', ['/zzz/']),
    ('var f = function (){}/ zzz;', [DIVISION]),
    ('class C{}/zzz/.test(a)', ['/zzz/']),
    ('var C = class {}/ zzz;', [DIVISION]),
    ('{ }/zzz/.test(a)', ['/zzz/']),
    ('var o = {}/ zzz;', [DIVISION]),
    ('var o = { m(){} }/ zzz;', [DIVISION]),
]

AROUND_YIELD_AND_AWAIT = [
    ('function* g(){ yield/zzz/; }', ['/zzz/']),
    ('function f(){ yield/ zzz }', [DIVISION]),
    ('async function f(){ await/zzz/; }', ['/zzz/']),
    ('function f(){ await/ zzz }', [DIVISION]),
]

INSIDE_A_TEMPLATE_INTERPOLATION = [
    ('`${/zzz/.source}`', ['/zzz/']),
    ('`${a/ zzz}`', [DIVISION]),
    ('`${/=zzz/.source}`', ['/=zzz/']),
    ('`${/`/.source}`', ['/`/']),
    ('`${`${/zzz/.source}`}`', ['/zzz/']),
    ('`${`${a/ zzz}`}`', [DIVISION]),
]

SLASH_EQUALS_WHERE_A_REGEXP_MAY_BEGIN = [
    '/=zzz/.test(a)',
    'if (a)/=zzz/.test(b);',
    'function f(){ return/=zzz/; }',
]

SLASH_EQUALS_AFTER_AN_ASSIGNMENT_TARGET = [
    ('a/= zzz', [DIVISION_ASSIGNMENT]),
    ('a.b/= zzz', [DIVISION_ASSIGNMENT]),
    ('a /=/zzz/', [DIVISION_ASSIGNMENT, '/zzz/']),
]

DIVISION_ACROSS_A_LINE_BREAK = inspect.cleandoc("""
    a = b
    / zzz / g
""")

RETURN_ACROSS_A_LINE_BREAK = inspect.cleandoc("""
    function f() {
      return
      /zzz/.source
    }
""")

CONTINUE_ACROSS_A_LINE_BREAK = inspect.cleandoc("""
    while (a) {
      continue
      /zzz/.test(b)
    }
""")

A_LINE_COMMENT_BEFORE_A_STATEMENT = inspect.cleandoc("""
    a;
    // c
    /zzz/.test(b)
""")

ACROSS_A_LINE_BREAK = [
    (DIVISION_ACROSS_A_LINE_BREAK, [DIVISION, DIVISION]),
    (RETURN_ACROSS_A_LINE_BREAK, ['/zzz/']),
    (CONTINUE_ACROSS_A_LINE_BREAK, ['/zzz/']),
]

A_PATTERN_THAT_CONTAINS_A_SLASH = [
    (r'/a\/b/.test(c)', [r'/a\/b/']),
    ('/[/]/.test(a)', ['/[/]/']),
    ('/a/gi.test(b)', ['/a/gi']),
]

MORE_THAN_ONE_SLASH = [
    ('a/ b/ c', [DIVISION, DIVISION]),
    ('a / /zzz/', [DIVISION, '/zzz/']),
    ('/a// zzz', ['/a/', DIVISION]),
    ('/a/g/ zzz', ['/a/g', DIVISION]),
    ('/a/ / /b/', ['/a/', DIVISION, '/b/']),
    ('/a/g / /b/i', ['/a/g', DIVISION, '/b/i']),
]

AROUND_A_COMMENT = [
    ('if (a) /* c */ /zzz/.test(b);', ['/zzz/']),
    ('a /* c */ / zzz', [DIVISION]),
    (A_LINE_COMMENT_BEFORE_A_STATEMENT, ['/zzz/']),
    ('var a = 8; a //zzz/', []),
]


class TestJsRegExpOrDivision(TestBase):

    def _slash_readings(self, source: str) -> list[str]:
        """
        What the parser made of every slash in the source, in the order the slashes are written.
        A regular expression is named by the literal it spells, a quotient by the operator that
        divides, and a place where the parser gave up by `PARSE_ERROR`; no two of the three
        readings can compare equal, so a swapped reading cannot pass for the one that was meant.
        """
        readings: list[tuple[int, str]] = []
        for node in JsParser(source).parse().walk():
            if isinstance(node, JsRegExpLiteral):
                readings.append((node.offset, node.raw))
            elif isinstance(node, JsErrorNode):
                readings.append((node.offset, PARSE_ERROR))
            elif isinstance(node, (JsBinaryExpression, JsAssignmentExpression)):
                right = node.right
                if node.operator in (DIVISION, DIVISION_ASSIGNMENT) and right is not None:
                    readings.append((source.rindex(DIVISION, 0, right.offset), node.operator))
        return [reading for _, reading in sorted(readings)]

    def test_slash_after_a_complete_expression_is_division(self):
        for source in AFTER_A_COMPLETE_EXPRESSION:
            with self.subTest(source=source):
                self.assertEqual(self._slash_readings(source), [DIVISION])

    def test_slash_at_the_start_of_a_statement_is_a_regexp(self):
        for source in AT_THE_START_OF_A_STATEMENT:
            with self.subTest(source=source):
                self.assertEqual(self._slash_readings(source), ['/zzz/'])

    def test_slash_where_an_operand_is_expected_is_a_regexp(self):
        for source in WHERE_AN_OPERAND_IS_EXPECTED:
            with self.subTest(source=source):
                self.assertEqual(self._slash_readings(source), ['/zzz/'])

    def test_slash_after_a_control_structure_header_is_a_regexp(self):
        for source in AFTER_A_CONTROL_STRUCTURE_HEADER:
            with self.subTest(source=source):
                self.assertEqual(self._slash_readings(source), ['/zzz/'])

    def test_slash_after_a_closing_brace_follows_what_the_brace_closed(self):
        for source, expected in AROUND_A_CLOSING_BRACE:
            with self.subTest(source=source):
                self.assertEqual(self._slash_readings(source), expected)

    def test_yield_and_await_take_an_operand_only_in_the_function_kind_that_defines_them(self):
        for source, expected in AROUND_YIELD_AND_AWAIT:
            with self.subTest(source=source):
                self.assertEqual(self._slash_readings(source), expected)

    def test_slash_in_a_template_interpolation_is_decided_by_the_hole_it_stands_in(self):
        for source, expected in INSIDE_A_TEMPLATE_INTERPOLATION:
            with self.subTest(source=source):
                self.assertEqual(self._slash_readings(source), expected)

    def test_slash_equals_is_a_regexp_where_a_regexp_may_begin(self):
        for source in SLASH_EQUALS_WHERE_A_REGEXP_MAY_BEGIN:
            with self.subTest(source=source):
                self.assertEqual(self._slash_readings(source), ['/=zzz/'])

    def test_slash_equals_is_compound_assignment_after_an_assignment_target(self):
        for source, expected in SLASH_EQUALS_AFTER_AN_ASSIGNMENT_TARGET:
            with self.subTest(source=source):
                self.assertEqual(self._slash_readings(source), expected)

    def test_a_line_break_starts_a_regexp_only_where_a_semicolon_is_inserted(self):
        for source, expected in ACROSS_A_LINE_BREAK:
            with self.subTest(source=source):
                self.assertEqual(self._slash_readings(source), expected)

    def test_a_regexp_pattern_may_contain_a_slash_that_does_not_end_it(self):
        for source, expected in A_PATTERN_THAT_CONTAINS_A_SLASH:
            with self.subTest(source=source):
                self.assertEqual(self._slash_readings(source), expected)

    def test_every_slash_in_an_expression_is_decided_on_its_own(self):
        for source, expected in MORE_THAN_ONE_SLASH:
            with self.subTest(source=source):
                self.assertEqual(self._slash_readings(source), expected)

    def test_a_comment_before_a_slash_does_not_change_its_reading(self):
        for source, expected in AROUND_A_COMMENT:
            with self.subTest(source=source):
                self.assertEqual(self._slash_readings(source), expected)
