from __future__ import annotations

import itertools
import unicodedata

from collections.abc import Iterable

from test import TestBase

from refinery.lib.scripts.js.lexer import JsLexer
from refinery.lib.scripts.js.parser import JsParser
from refinery.lib.scripts.js.token import (
    ASCII_WHITESPACE,
    LINE_TERMINATORS,
    WHITESPACE,
    JsTokenKind,
)


NAMED_WHITESPACE = [0x0009, 0x000B, 0x000C, 0xFEFF]
"""
The code points the ECMA-262 WhiteSpace production names one at a time: the tab, the vertical tab,
the form feed and the zero width no-break space. Its one remaining alternative is `<USP>`, which is
every code point of the Unicode general category `Zs`.
"""

NAMED_LINE_TERMINATORS = [0x000A, 0x000D]
"""
The code points the ECMA-262 LineTerminator production names one at a time: the line feed and the
carriage return. The two it has left are the line separator and the paragraph separator, which are
the whole of the Unicode general categories `Zl` and `Zp`.
"""

WHATWG_ASCII_WHITESPACE = [0x0009, 0x000A, 0x000C, 0x000D, 0x0020]
"""
The code points the WHATWG Infra Standard calls ASCII whitespace: `U+0009` TAB, `U+000A` LF,
`U+000C` FF, `U+000D` CR and `U+0020` SPACE. The vertical tab is not among them.
"""


def _spelled(code_points: Iterable[int]) -> list[str]:
    """
    The code points, sorted and named. Most of these characters have no width and the rest are
    indistinguishable from a space, so a failure reporting the characters themselves reports
    nothing at all.
    """
    return sorted(F'U+{code_point:04X}' for code_point in code_points)


def _code_points_of_category(*categories: str) -> set[int]:
    wanted = frozenset(categories)
    return {cp for cp in range(0x110000) if unicodedata.category(chr(cp)) in wanted}


class TestJsLexer(TestBase):

    def _tokens(self, source: str) -> list[tuple[JsTokenKind, str]]:
        lexer = JsLexer(source)
        result = []
        for tok in lexer.tokenize():
            if tok.kind == JsTokenKind.EOF:
                break
            result.append((tok.kind, tok.value))
        return result

    def _token_kinds(self, source: str) -> list[JsTokenKind]:
        return [k for k, _ in self._tokens(source)]

    def test_integer_decimal(self):
        tokens = self._tokens('42')
        self.assertEqual(tokens, [(JsTokenKind.INTEGER, '42')])

    def test_integer_hex(self):
        tokens = self._tokens('0xFF')
        self.assertEqual(tokens, [(JsTokenKind.INTEGER, '0xFF')])

    def test_integer_octal(self):
        tokens = self._tokens('0o77')
        self.assertEqual(tokens, [(JsTokenKind.INTEGER, '0o77')])

    def test_integer_binary(self):
        tokens = self._tokens('0b1010')
        self.assertEqual(tokens, [(JsTokenKind.INTEGER, '0b1010')])

    def test_float_decimal(self):
        tokens = self._tokens('3.14')
        self.assertEqual(tokens, [(JsTokenKind.FLOAT, '3.14')])

    def test_float_leading_dot(self):
        tokens = self._tokens('.5')
        self.assertEqual(tokens, [(JsTokenKind.FLOAT, '.5')])

    def test_float_exponent(self):
        tokens = self._tokens('1e10')
        self.assertEqual(tokens, [(JsTokenKind.FLOAT, '1e10')])

    def test_float_exponent_negative(self):
        tokens = self._tokens('2.5e-3')
        self.assertEqual(tokens, [(JsTokenKind.FLOAT, '2.5e-3')])

    def test_bigint(self):
        tokens = self._tokens('100n')
        self.assertEqual(tokens, [(JsTokenKind.BIGINT, '100n')])

    def test_bigint_hex(self):
        tokens = self._tokens('0xFFn')
        self.assertEqual(tokens, [(JsTokenKind.BIGINT, '0xFFn')])

    def test_numeric_separators(self):
        tokens = self._tokens('1_000_000')
        self.assertEqual(tokens, [(JsTokenKind.INTEGER, '1_000_000')])

    def test_string_single(self):
        tokens = self._tokens("'hello'")
        self.assertEqual(tokens, [(JsTokenKind.STRING_SINGLE, "'hello'")])

    def test_string_double(self):
        tokens = self._tokens('"world"')
        self.assertEqual(tokens, [(JsTokenKind.STRING_DOUBLE, '"world"')])

    def test_string_escape(self):
        tokens = self._tokens(r"'he\'llo'")
        self.assertEqual(tokens, [(JsTokenKind.STRING_SINGLE, r"'he\'llo'")])

    def test_string_unicode_escape(self):
        tokens = self._tokens(r"'\u0041'")
        self.assertEqual(tokens, [(JsTokenKind.STRING_SINGLE, r"'\u0041'")])

    def test_template_full(self):
        tokens = self._tokens('`hello`')
        self.assertEqual(tokens, [(JsTokenKind.TEMPLATE_FULL, '`hello`')])

    def test_template_with_expression(self):
        kinds = self._token_kinds('`a${x}b`')
        self.assertEqual(kinds, [
            JsTokenKind.TEMPLATE_HEAD,
            JsTokenKind.IDENTIFIER,
            JsTokenKind.TEMPLATE_TAIL,
        ])

    def test_template_multiple_expressions(self):
        kinds = self._token_kinds('`${a}mid${b}end`')
        self.assertEqual(kinds, [
            JsTokenKind.TEMPLATE_HEAD,
            JsTokenKind.IDENTIFIER,
            JsTokenKind.TEMPLATE_MIDDLE,
            JsTokenKind.IDENTIFIER,
            JsTokenKind.TEMPLATE_TAIL,
        ])

    def test_template_nested(self):
        kinds = self._token_kinds('`${`inner`}`')
        self.assertEqual(kinds, [
            JsTokenKind.TEMPLATE_HEAD,
            JsTokenKind.TEMPLATE_FULL,
            JsTokenKind.TEMPLATE_TAIL,
        ])

    def test_regexp(self):
        tokens = self._tokens('/abc/gi')
        self.assertEqual(tokens, [(JsTokenKind.REGEXP, '/abc/gi')])

    def test_regexp_with_class(self):
        tokens = self._tokens('/[a-z]+/i')
        self.assertEqual(tokens, [(JsTokenKind.REGEXP, '/[a-z]+/i')])

    def test_regexp_vs_division(self):
        kinds = self._token_kinds('x / y')
        self.assertEqual(kinds, [
            JsTokenKind.IDENTIFIER,
            JsTokenKind.SLASH,
            JsTokenKind.IDENTIFIER,
        ])

    def test_regexp_after_equals(self):
        kinds = self._token_kinds('x = /re/')
        self.assertEqual(kinds, [
            JsTokenKind.IDENTIFIER,
            JsTokenKind.EQUALS,
            JsTokenKind.REGEXP,
        ])

    def test_regexp_after_return(self):
        kinds = self._token_kinds('return /re/')
        self.assertEqual(kinds, [
            JsTokenKind.RETURN,
            JsTokenKind.REGEXP,
        ])

    def test_all_keywords(self):
        for kw in (
            'var', 'let', 'const', 'function', 'class', 'if', 'else',
            'for', 'while', 'do', 'switch', 'case', 'default', 'break',
            'continue', 'return', 'throw', 'try', 'catch', 'finally',
            'new', 'delete', 'typeof', 'void', 'instanceof', 'in', 'of',
            'import', 'export', 'from', 'as', 'yield', 'await', 'async',
            'extends', 'super', 'this', 'null', 'true', 'false',
            'debugger', 'with',
        ):
            tokens = self._tokens(kw)
            self.assertEqual(len(tokens), 1, F'keyword {kw!r} not recognized')
            self.assertTrue(
                tokens[0][0].is_keyword or tokens[0][0] in (
                    JsTokenKind.TRUE, JsTokenKind.FALSE,
                    JsTokenKind.NULL, JsTokenKind.THIS,
                    JsTokenKind.SUPER,
                ),
                F'{kw!r} not classified as keyword',
            )

    def test_identifier(self):
        tokens = self._tokens('myVar')
        self.assertEqual(tokens, [(JsTokenKind.IDENTIFIER, 'myVar')])

    def test_identifier_dollar(self):
        tokens = self._tokens('$el')
        self.assertEqual(tokens, [(JsTokenKind.IDENTIFIER, '$el')])

    def test_identifier_underscore(self):
        tokens = self._tokens('_private')
        self.assertEqual(tokens, [(JsTokenKind.IDENTIFIER, '_private')])

    def test_line_comment(self):
        tokens = self._tokens('x // comment\ny')
        kinds = [k for k, _ in tokens]
        self.assertIn(JsTokenKind.COMMENT, kinds)
        self.assertIn(JsTokenKind.IDENTIFIER, kinds)

    def test_block_comment(self):
        tokens = self._tokens('x /* comment */ y')
        kinds = [k for k, _ in tokens]
        self.assertIn(JsTokenKind.COMMENT, kinds)
        id_count = sum(1 for k in kinds if k == JsTokenKind.IDENTIFIER)
        self.assertEqual(id_count, 2)

    def test_block_comment_with_newline(self):
        tokens = self._tokens('x /*\n*/ y')
        kinds = [k for k, _ in tokens]
        self.assertIn(JsTokenKind.NEWLINE, kinds)

    def test_newlines(self):
        tokens = self._tokens('x\ny')
        kinds = [k for k, _ in tokens]
        self.assertEqual(kinds, [
            JsTokenKind.IDENTIFIER,
            JsTokenKind.NEWLINE,
            JsTokenKind.IDENTIFIER,
        ])

    def test_operators_basic(self):
        for src, expected in [
            ('+', JsTokenKind.PLUS),
            ('-', JsTokenKind.MINUS),
            ('*', JsTokenKind.STAR),
            ('%', JsTokenKind.PERCENT),
            ('!', JsTokenKind.BANG),
            ('~', JsTokenKind.TILDE),
        ]:
            tokens = self._tokens(src)
            self.assertEqual(tokens, [(expected, src)], F'failed for {src!r}')

    def test_operators_multi_char(self):
        for src, expected in [
            ('===', JsTokenKind.EQ3),
            ('!==', JsTokenKind.BANG_EQ2),
            ('>>>', JsTokenKind.GT3),
            ('>>=', JsTokenKind.GT2_ASSIGN),
            ('>>>=', JsTokenKind.GT3_ASSIGN),
            ('**', JsTokenKind.STAR2),
            ('=>', JsTokenKind.ARROW),
            ('&&', JsTokenKind.AND),
            ('||', JsTokenKind.OR),
            ('??', JsTokenKind.QQ),
            ('?.', JsTokenKind.QUESTION_DOT),
            ('...', JsTokenKind.ELLIPSIS),
            ('&&=', JsTokenKind.AND_ASSIGN),
            ('||=', JsTokenKind.OR_ASSIGN),
            ('??=', JsTokenKind.NULLISH_ASSIGN),
        ]:
            tokens = self._tokens(src)
            self.assertEqual(len(tokens), 1, F'expected 1 token for {src!r}, got {tokens}')
            self.assertEqual(tokens[0][0], expected, F'wrong kind for {src!r}')

    def test_punctuation(self):
        for src, expected in [
            ('(', JsTokenKind.LPAREN),
            (')', JsTokenKind.RPAREN),
            ('{', JsTokenKind.LBRACE),
            ('}', JsTokenKind.RBRACE),
            ('[', JsTokenKind.LBRACKET),
            (']', JsTokenKind.RBRACKET),
            (';', JsTokenKind.SEMICOLON),
            (',', JsTokenKind.COMMA),
        ]:
            tokens = self._tokens(src)
            self.assertEqual(tokens, [(expected, src)], F'failed for {src!r}')

    def test_arrow_function_tokens(self):
        kinds = self._token_kinds('(x) => x')
        self.assertEqual(kinds, [
            JsTokenKind.LPAREN,
            JsTokenKind.IDENTIFIER,
            JsTokenKind.RPAREN,
            JsTokenKind.ARROW,
            JsTokenKind.IDENTIFIER,
        ])

    def test_complex_expression_tokens(self):
        kinds = self._token_kinds('a.b(c, d)')
        self.assertEqual(kinds, [
            JsTokenKind.IDENTIFIER,
            JsTokenKind.DOT,
            JsTokenKind.IDENTIFIER,
            JsTokenKind.LPAREN,
            JsTokenKind.IDENTIFIER,
            JsTokenKind.COMMA,
            JsTokenKind.IDENTIFIER,
            JsTokenKind.RPAREN,
        ])

    def test_optional_chaining(self):
        kinds = self._token_kinds('a?.b')
        self.assertEqual(kinds, [
            JsTokenKind.IDENTIFIER,
            JsTokenKind.QUESTION_DOT,
            JsTokenKind.IDENTIFIER,
        ])

    def test_shebang_skipped_at_start(self):
        tokens = self._tokens('#!/usr/bin/env node\nvar x = 1;')
        self.assertEqual(tokens[0], (JsTokenKind.NEWLINE, '\n'))
        self.assertEqual(tokens[1], (JsTokenKind.VAR, 'var'))

    def test_shebang_only_at_position_zero(self):
        tokens = self._tokens('var x;\n#!/usr/bin/env node')
        kinds = [k for k, _ in tokens]
        self.assertIn(JsTokenKind.ERROR, kinds)

    def test_private_identifier(self):
        self.assertEqual(
            self._tokens('#field'), [(JsTokenKind.PRIVATE_IDENTIFIER, '#field')])

    def test_private_identifier_after_dot(self):
        self.assertEqual(self._token_kinds('this.#x'), [
            JsTokenKind.THIS,
            JsTokenKind.DOT,
            JsTokenKind.PRIVATE_IDENTIFIER,
        ])

    def test_hash_without_name_is_error(self):
        self.assertEqual(self._tokens('#'), [(JsTokenKind.ERROR, '#')])

    def test_at_token(self):
        self.assertEqual(self._tokens('@'), [(JsTokenKind.AT, '@')])

    def _bounded_tokens(self, source: str, limit: int = 16) -> list[tuple[JsTokenKind, str]]:
        """
        The first *limit* tokens of *source*, the end of file among them. A scan that consumes no
        input yields tokens for as long as anything reads them, and `list` cannot report that as a
        failure because it never returns at all; reading a bounded prefix turns the same defect into
        an assertion about a sequence whose end is missing.
        """
        return [
            (token.kind, token.value)
            for token in itertools.islice(JsLexer(source).tokenize(), limit)
        ]

    def test_a_backslash_that_begins_no_escape_is_a_token_that_consumes_it(self):
        """
        Node refuses every program written with one, so the character names nothing; what these
        pin is that the scan moves past it and reaches the end of the input.
        """
        backslash = '\\'
        self.assertEqual(self._bounded_tokens(backslash), [
            (JsTokenKind.ERROR, backslash),
            (JsTokenKind.EOF, ''),
        ])
        self.assertEqual(self._bounded_tokens(F'a{backslash}b'), [
            (JsTokenKind.IDENTIFIER, 'a'),
            (JsTokenKind.ERROR, backslash),
            (JsTokenKind.IDENTIFIER, 'b'),
            (JsTokenKind.EOF, ''),
        ])
        self.assertEqual(self._bounded_tokens(F'a{backslash}'), [
            (JsTokenKind.IDENTIFIER, 'a'),
            (JsTokenKind.ERROR, backslash),
            (JsTokenKind.EOF, ''),
        ])
        self.assertEqual(self._bounded_tokens(F'a{backslash}{backslash}b'), [
            (JsTokenKind.IDENTIFIER, 'a'),
            (JsTokenKind.ERROR, backslash),
            (JsTokenKind.ERROR, backslash),
            (JsTokenKind.IDENTIFIER, 'b'),
            (JsTokenKind.EOF, ''),
        ])

    def test_a_unicode_escape_in_an_identifier_is_still_one_identifier(self):
        """
        Node reads a declaration whose name is written with a unicode escape as a declaration of the
        name that escape denotes, and prints the value for a program that goes on to read the plain
        spelling. The escape is therefore part of the name, whether it opens the name or sits inside
        it, and never a token standing beside it.
        """
        self.assertEqual(
            self._bounded_tokens(R'\u0061bc'),
            [(JsTokenKind.IDENTIFIER, R'\u0061bc'), (JsTokenKind.EOF, '')])
        self.assertEqual(
            self._bounded_tokens(R'a\u0062c'),
            [(JsTokenKind.IDENTIFIER, R'a\u0062c'), (JsTokenKind.EOF, '')])

    def _statements(self, source: str) -> int:
        return len(JsParser(source).parse().body)

    def test_a_code_point_escape_naming_no_code_point_ends_with_the_literal_it_is_in(self):
        """
        Node refuses the program: `U+110000` is past the last code point there is, so the escape
        names nothing. The literal is over at its closing quote all the same, and the statement
        written behind it is still a statement — a scan that went looking for the character the
        escape promised would take the rest of the file with it.
        """
        source = R"x = '\u{110000}'; y;"
        self.assertEqual(self._bounded_tokens(source), [
            (JsTokenKind.IDENTIFIER, 'x'),
            (JsTokenKind.EQUALS, '='),
            (JsTokenKind.STRING_SINGLE, R"'\u{110000}'"),
            (JsTokenKind.SEMICOLON, ';'),
            (JsTokenKind.IDENTIFIER, 'y'),
            (JsTokenKind.SEMICOLON, ';'),
            (JsTokenKind.EOF, ''),
        ])
        self.assertEqual(self._statements(source), 2)

    def test_a_code_point_escape_whose_brace_is_never_closed_ends_with_its_literal(self):
        """
        Node refuses the program: nothing closes the brace the escape opens, so what stands behind
        it is no hexadecimal number. The quote ends the literal all the same, and the statement
        behind it survives an escape that named nothing.
        """
        source = R"x = '\u{41'; y;"
        self.assertEqual(self._bounded_tokens(source), [
            (JsTokenKind.IDENTIFIER, 'x'),
            (JsTokenKind.EQUALS, '='),
            (JsTokenKind.STRING_SINGLE, R"'\u{41'"),
            (JsTokenKind.SEMICOLON, ';'),
            (JsTokenKind.IDENTIFIER, 'y'),
            (JsTokenKind.SEMICOLON, ';'),
            (JsTokenKind.EOF, ''),
        ])
        self.assertEqual(self._statements(source), 2)

    def test_a_code_point_escape_naming_nothing_is_read_the_same_between_double_quotes(self):
        source = R'x = "\u{110000}"; y;'
        self.assertEqual(self._bounded_tokens(source), [
            (JsTokenKind.IDENTIFIER, 'x'),
            (JsTokenKind.EQUALS, '='),
            (JsTokenKind.STRING_DOUBLE, R'"\u{110000}"'),
            (JsTokenKind.SEMICOLON, ';'),
            (JsTokenKind.IDENTIFIER, 'y'),
            (JsTokenKind.SEMICOLON, ';'),
            (JsTokenKind.EOF, ''),
        ])
        self.assertEqual(self._statements(source), 2)

    def test_a_digit_of_another_script_is_a_digit_of_no_numeral(self):
        """
        Node refuses both programs. The decimal digits of the language are `0` through `9` and no
        others, so an Arabic-Indic digit and a superscript digit each stand for themselves rather
        than open a number.
        """
        self.assertEqual(self._bounded_tokens('x = \u0661\u0662\u0663; y;'), [
            (JsTokenKind.IDENTIFIER, 'x'),
            (JsTokenKind.EQUALS, '='),
            (JsTokenKind.ERROR, '\u0661'),
            (JsTokenKind.ERROR, '\u0662'),
            (JsTokenKind.ERROR, '\u0663'),
            (JsTokenKind.SEMICOLON, ';'),
            (JsTokenKind.IDENTIFIER, 'y'),
            (JsTokenKind.SEMICOLON, ';'),
            (JsTokenKind.EOF, ''),
        ])
        self.assertEqual(self._bounded_tokens('x = \u00B2; y;'), [
            (JsTokenKind.IDENTIFIER, 'x'),
            (JsTokenKind.EQUALS, '='),
            (JsTokenKind.ERROR, '\u00B2'),
            (JsTokenKind.SEMICOLON, ';'),
            (JsTokenKind.IDENTIFIER, 'y'),
            (JsTokenKind.SEMICOLON, ';'),
            (JsTokenKind.EOF, ''),
        ])

    def test_a_digit_of_another_script_behind_a_numeral_is_no_digit_of_it(self):
        """
        Node refuses each program. A number ends at the last of its own digits, so what follows is a
        character standing beside the number rather than one more digit of it.
        """
        for digit in ['\u0661', '\u00B2', '\uFF11']:
            with self.subTest(digit=F'U+{ord(digit):04X}'):
                self.assertEqual(self._bounded_tokens(F'x = 1{digit}; y;'), [
                    (JsTokenKind.IDENTIFIER, 'x'),
                    (JsTokenKind.EQUALS, '='),
                    (JsTokenKind.INTEGER, '1'),
                    (JsTokenKind.ERROR, digit),
                    (JsTokenKind.SEMICOLON, ';'),
                    (JsTokenKind.IDENTIFIER, 'y'),
                    (JsTokenKind.SEMICOLON, ';'),
                    (JsTokenKind.EOF, ''),
                ])

    def test_a_digit_of_another_script_behind_a_name_is_read_as_part_of_that_name(self):
        """
        Node runs the program written with the Arabic-Indic digit, whose general category `Nd` is
        IdentifierPart, and refuses the one written with the superscript digit, whose category `No`
        is not. Both are one name here, so the name the lexer reads is the wider of the two.
        """
        for digit in ['\u0661', '\u00B2']:
            with self.subTest(digit=F'U+{ord(digit):04X}'):
                self.assertEqual(self._bounded_tokens(F'x{digit} = 1; y;'), [
                    (JsTokenKind.IDENTIFIER, F'x{digit}'),
                    (JsTokenKind.EQUALS, '='),
                    (JsTokenKind.INTEGER, '1'),
                    (JsTokenKind.SEMICOLON, ';'),
                    (JsTokenKind.IDENTIFIER, 'y'),
                    (JsTokenKind.SEMICOLON, ';'),
                    (JsTokenKind.EOF, ''),
                ])

    def test_a_hash_bang_line_ends_at_a_line_terminator_and_that_ending_ends_a_line(self):
        """
        Node runs each of these programs. The first line is a comment of which no token survives,
        and the terminator that ends it is a line ending exactly as the one that ends any other
        comment is. A carriage return and a line feed together are one ending and not two.
        """
        endings = [chr(0x000A), chr(0x000D), chr(0x000D) + chr(0x000A), chr(0x2028), chr(0x2029)]
        for ending in endings:
            with self.subTest(ending=' '.join(_spelled(map(ord, ending)))):
                source = F'#!x{ending}y;'
                self.assertEqual(self._bounded_tokens(source), [
                    (JsTokenKind.NEWLINE, ending),
                    (JsTokenKind.IDENTIFIER, 'y'),
                    (JsTokenKind.SEMICOLON, ';'),
                    (JsTokenKind.EOF, ''),
                ])
                self.assertEqual(self._statements(source), 1)

    def test_a_file_that_is_only_a_hash_bang_line_holds_no_token(self):
        """
        Node runs this file, and it prints nothing. The line is over without a terminator to end it.
        """
        self.assertEqual(self._bounded_tokens('#!/usr/bin/env node'), [(JsTokenKind.EOF, '')])
        self.assertEqual(self._statements('#!/usr/bin/env node'), 0)

    def test_a_joiner_may_stand_inside_a_name_and_may_not_open_one(self):
        """
        Node runs the first program and refuses the second. The zero width joiner and non-joiner are
        IdentifierPart and not IdentifierStart, so a name holds one anywhere but at its beginning,
        where the character stands for itself and the name behind it is a name of its own.
        """
        for joiner in [chr(0x200C), chr(0x200D)]:
            with self.subTest(joiner=F'U+{ord(joiner):04X}'):
                self.assertEqual(self._bounded_tokens(F'x = a{joiner}b; y;'), [
                    (JsTokenKind.IDENTIFIER, 'x'),
                    (JsTokenKind.EQUALS, '='),
                    (JsTokenKind.IDENTIFIER, F'a{joiner}b'),
                    (JsTokenKind.SEMICOLON, ';'),
                    (JsTokenKind.IDENTIFIER, 'y'),
                    (JsTokenKind.SEMICOLON, ';'),
                    (JsTokenKind.EOF, ''),
                ])
                self.assertEqual(self._bounded_tokens(F'x = {joiner}ab; y;'), [
                    (JsTokenKind.IDENTIFIER, 'x'),
                    (JsTokenKind.EQUALS, '='),
                    (JsTokenKind.ERROR, joiner),
                    (JsTokenKind.IDENTIFIER, 'ab'),
                    (JsTokenKind.SEMICOLON, ';'),
                    (JsTokenKind.IDENTIFIER, 'y'),
                    (JsTokenKind.SEMICOLON, ';'),
                    (JsTokenKind.EOF, ''),
                ])


class TestTheWhitespaceProductionsAreThreeSetsAndNotOne(TestBase):
    """
    `WHITESPACE`, `LINE_TERMINATORS` and `ASCII_WHITESPACE` answer three different questions: what
    may stand between two tokens, where a line ends and a semicolon may therefore be inserted, and
    what a forgiving base64 decode removes from the argument of `atob` before it reads it. Their
    union is what pads a number, and pinning that union alone is blind to a character that moved
    from one of them into another — which is a change of where statements end, or of which arguments
    `atob` refuses.

    Each set is therefore pinned on its own, against the Unicode database or the standard that names
    it rather than against a list copied out of the code under test.
    """

    def test_the_whitespace_production_is_the_four_named_characters_and_the_space_separators(self):
        self.assertEqual(
            _spelled({*NAMED_WHITESPACE, *_code_points_of_category('Zs')}),
            _spelled(map(ord, WHITESPACE)))

    def test_the_line_terminator_production_is_the_two_controls_and_the_two_separators(self):
        self.assertEqual(
            _spelled({*NAMED_LINE_TERMINATORS, *_code_points_of_category('Zl', 'Zp')}),
            _spelled(map(ord, LINE_TERMINATORS)))

    def test_no_character_both_separates_two_tokens_and_ends_the_line_they_stand_on(self):
        self.assertEqual([], _spelled(map(ord, set(WHITESPACE) & set(LINE_TERMINATORS))))

    def test_ascii_whitespace_is_the_five_code_points_the_whatwg_definition_names(self):
        self.assertEqual(_spelled(WHATWG_ASCII_WHITESPACE), _spelled(map(ord, ASCII_WHITESPACE)))

    def test_ascii_whitespace_is_the_ascii_of_the_language_without_the_vertical_tab(self):
        """
        The vertical tab is the one ASCII character the language calls whitespace that a forgiving
        decode refuses rather than skips, and the two line terminators of ASCII are characters it
        skips although they end a line. The set is therefore neither of the other two sets, nor
        their union cut down to ASCII.
        """
        language = {ord(c) for c in WHITESPACE + LINE_TERMINATORS if ord(c) < 0x80}
        self.assertEqual(_spelled(language - {0x000B}), _spelled(map(ord, ASCII_WHITESPACE)))
        self.assertEqual(
            _spelled([0x000A, 0x000D]),
            _spelled(map(ord, set(ASCII_WHITESPACE) & set(LINE_TERMINATORS))))
        self.assertEqual(
            _spelled([0x0020]),
            _spelled({ord(c) for c in ASCII_WHITESPACE} & _code_points_of_category('Zs')))
