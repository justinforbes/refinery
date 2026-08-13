from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Generator

from refinery.lib.scripts.js.token import (
    KEYWORDS,
    LINE_TERMINATORS,
    WHITESPACE,
    JsToken,
    JsTokenKind,
)

_ESCAPE_MAP: dict[str, str] = {
    'b'  : '\b',
    'f'  : '\f',
    'n'  : '\n',
    'r'  : '\r',
    't'  : '\t',
    'v'  : '\v',
    '\\' : '\\',
    "'"  : "'",
    '"'  : '"',
    '`'  : '`',
}

_HEX = frozenset('0123456789abcdefABCDEF')
_OCTAL = frozenset('01234567')
_DECIMAL = frozenset('0123456789')

_WHITESPACE = frozenset(WHITESPACE)
_LINE_TERMINATOR = re.compile(F'[{re.escape(LINE_TERMINATORS)}]')
"""
The two productions `refinery.lib.scripts.js.token` spells, in the shapes this scan reads them. The
spelling stays a string there because `refinery.lib.scripts.js.numbers.TRIMMABLE_WHITESPACE` is the
concatenation of the two, and neither shape here is the right one for the other: whitespace is asked
about one character at a time, where a comment asks only where the next terminator is.
"""

_MAX_CODE_POINT = 0x10FFFF
"""
The largest code point a `\\u{...}` escape may name. A larger one is no escape at all, and it is
asked about here rather than left to `chr`, which raises: the scan runs inside a generator that the
parser reads statement by statement, so an exception raised in it is not a diagnostic but the end
of the token stream, and every statement behind the escape disappears with it.
"""

_IDENTIFIER_PUNCTUATION = frozenset('_$')
_IDENTIFIER_JOINERS = frozenset('\u200c\u200d')
"""
The zero width non-joiner and the zero width joiner, which are IdentifierPart and nothing else: they
may stand inside a name but not open one. They carry no width, so a name written with one reads as
the same name, which is exactly what makes them worth writing in an obfuscated file.
"""


def _begins_unicode_escape(src: str, pos: int) -> bool:
    return src[pos:pos + 2] == '\\u'


def _at_identifier_start(src: str, pos: int) -> bool:
    """
    Whether an IdentifierName begins at *pos*. A backslash opens one only where it opens a
    unicode escape; one that begins no escape is a character no name may hold, and reading it as
    the start of a name is how a scan that consumed nothing yielded empty identifiers for as long
    as anything read them.
    """
    c = src[pos:pos + 1]
    if not c:
        return False
    return c.isalpha() or c in _IDENTIFIER_PUNCTUATION or _begins_unicode_escape(src, pos)


def _decode_one_escape(src: str, pos: int, length: int) -> tuple[str, int]:
    if pos >= length:
        return '', pos
    c = src[pos]
    pos += 1
    mapped = _ESCAPE_MAP.get(c)
    if mapped is not None:
        return mapped, pos
    if c in _OCTAL:
        value = int(c, 8)
        remaining = 2 if c in '0123' else 1
        while remaining > 0 and pos < length and src[pos] in _OCTAL:
            value = value * 8 + int(src[pos], 8)
            pos += 1
            remaining -= 1
        return chr(value), pos
    if c in '89':
        return c, pos
    if c == 'x' and pos + 1 < length:
        hexstr = src[pos:pos + 2]
        if len(hexstr) == 2 and _HEX.issuperset(hexstr):
            return chr(int(hexstr, 16)), pos + 2
        return 'x', pos
    if c == 'u':
        if pos < length and src[pos] == '{':
            end = pos + 1
            while end < length and src[end] in _HEX:
                end += 1
            if end > pos + 1 and end < length and src[end] == '}':
                value = int(src[pos + 1:end], 16)
                return chr(value) if value <= _MAX_CODE_POINT else 'u', end + 1
        elif pos + 3 < length:
            hexstr = src[pos:pos + 4]
            if len(hexstr) == 4 and _HEX.issuperset(hexstr):
                return chr(int(hexstr, 16)), pos + 4
        return 'u', pos
    if c in LINE_TERMINATORS:
        if c == '\r' and pos < length and src[pos] == '\n':
            pos += 1
        return '', pos
    return c, pos


_FOUR_CHAR_OPS: dict[str, JsTokenKind] = {
    '>>>=' : JsTokenKind.GT3_ASSIGN,
}

_THREE_CHAR_OPS: dict[str, JsTokenKind] = {
    '===' : JsTokenKind.EQ3,
    '!==' : JsTokenKind.BANG_EQ2,
    '>>>' : JsTokenKind.GT3,
    '**=' : JsTokenKind.STAR2_ASSIGN,
    '<<=' : JsTokenKind.LT2_ASSIGN,
    '>>=' : JsTokenKind.GT2_ASSIGN,
    '&&=' : JsTokenKind.AND_ASSIGN,
    '||=' : JsTokenKind.OR_ASSIGN,
    '??=' : JsTokenKind.NULLISH_ASSIGN,
    '...' : JsTokenKind.ELLIPSIS,
}

_TWO_CHAR_OPS: dict[str, JsTokenKind] = {
    '==' : JsTokenKind.EQ2,
    '!=' : JsTokenKind.BANG_EQ,
    '<=' : JsTokenKind.LT_EQ,
    '>=' : JsTokenKind.GT_EQ,
    '+=' : JsTokenKind.PLUS_ASSIGN,
    '-=' : JsTokenKind.MINUS_ASSIGN,
    '*=' : JsTokenKind.STAR_ASSIGN,
    '/=' : JsTokenKind.SLASH_ASSIGN,
    '%=' : JsTokenKind.PERCENT_ASSIGN,
    '&=' : JsTokenKind.AMP_ASSIGN,
    '|=' : JsTokenKind.PIPE_ASSIGN,
    '^=' : JsTokenKind.CARET_ASSIGN,
    '**' : JsTokenKind.STAR2,
    '++' : JsTokenKind.INC,
    '--' : JsTokenKind.DEC,
    '&&' : JsTokenKind.AND,
    '||' : JsTokenKind.OR,
    '??' : JsTokenKind.QQ,
    '?.' : JsTokenKind.QUESTION_DOT,
    '=>' : JsTokenKind.ARROW,
    '<<' : JsTokenKind.LT2,
    '>>' : JsTokenKind.GT2,
}

_ONE_CHAR_OPS: dict[str, JsTokenKind] = {
    '+' : JsTokenKind.PLUS,
    '-' : JsTokenKind.MINUS,
    '*' : JsTokenKind.STAR,
    '/' : JsTokenKind.SLASH,
    '%' : JsTokenKind.PERCENT,
    '=' : JsTokenKind.EQUALS,
    '!' : JsTokenKind.BANG,
    '<' : JsTokenKind.LT,
    '>' : JsTokenKind.GT,
    '&' : JsTokenKind.AMP,
    '|' : JsTokenKind.PIPE,
    '^' : JsTokenKind.CARET,
    '~' : JsTokenKind.TILDE,
    '.' : JsTokenKind.DOT,
    '?' : JsTokenKind.QUESTION,
    ':' : JsTokenKind.COLON,
    '(' : JsTokenKind.LPAREN,
    ')' : JsTokenKind.RPAREN,
    '{' : JsTokenKind.LBRACE,
    '}' : JsTokenKind.RBRACE,
    '[' : JsTokenKind.LBRACKET,
    ']' : JsTokenKind.RBRACKET,
    ';' : JsTokenKind.SEMICOLON,
    ',' : JsTokenKind.COMMA,
    '@' : JsTokenKind.AT,
}


@dataclass(frozen=True)
class JsLexerState:
    """
    What a rewind has to put back. The position is not part of it, because a rewind always goes to
    the start of a token the parser is already holding: it is the template nesting alone that no
    longer follows from that offset once scanning has moved past it.
    """
    template_depth: int
    brace_stack: tuple[int, ...]


@dataclass
class JsLexer:
    source: str
    pos: int = 0
    _template_depth: int = 0
    _brace_stack: list[int] = field(default_factory=list)

    def capture(self) -> JsLexerState:
        return JsLexerState(self._template_depth, tuple(self._brace_stack))

    def rewind(self, pos: int, state: JsLexerState) -> None:
        self.pos = pos
        self._template_depth = state.template_depth
        self._brace_stack = list(state.brace_stack)

    def scan_regexp(self) -> JsToken:
        """
        Read a regular expression literal where scanning currently stands. ECMA-262 clause 12 picks
        the lexical goal symbol from the syntactic grammar context, which only the parser knows, so
        no path through `tokenize` reaches this scan: a slash is spelled as the operator it looks
        like until someone who is expecting an expression asks for it again.
        """
        start = self.pos
        return JsToken(JsTokenKind.REGEXP, self._read_regexp(), start)

    def _peek(self, count: int = 1) -> str:
        return self.source[self.pos:self.pos + count]

    def _at_end(self) -> bool:
        return self.pos >= len(self.source)

    def _skip_whitespace(self) -> bool:
        """
        Consume the ECMA-262 WhiteSpace between two tokens. It is the whole production and not the
        space and the tab alone, because every other character in it separates tokens just as they
        do: a file that opens with a byte order mark is ordinary, and reading that mark as a token
        of its own splits one program into two statements.

        Line terminators are deliberately not consumed here. They separate tokens too, but they also
        end a line, and the parser reads the end of a line as a place a semicolon may be inserted.
        """
        start = self.pos
        src = self.source
        length = len(src)
        while self.pos < length and src[self.pos] in _WHITESPACE:
            self.pos += 1
        return self.pos > start

    def _read_line_comment(self) -> str:
        """
        Consume a comment that runs to the end of its line, and the `#!` line, which is one. The end
        is looked for at once rather than one character at a time, because a comment is the longest
        run of characters this scan ever walks and the run is over as soon as a terminator is
        anywhere in it.
        """
        start = self.pos
        src = self.source
        end = _LINE_TERMINATOR.search(src, self.pos + 2)
        self.pos = end.start() if end else len(src)
        return src[start:self.pos]

    def _read_block_comment(self) -> tuple[str, bool]:
        start = self.pos
        src = self.source
        length = len(src)
        self.pos += 2
        has_newline = False
        while self.pos < length - 1:
            if src[self.pos] == '*' and src[self.pos + 1] == '/':
                self.pos += 2
                return src[start:self.pos], has_newline
            if src[self.pos] in LINE_TERMINATORS:
                has_newline = True
            self.pos += 1
        self.pos = length
        return src[start:self.pos], has_newline

    def _read_string_escape(self) -> str:
        self.pos += 1
        result, self.pos = _decode_one_escape(
            self.source, self.pos, len(self.source))
        return result

    def _read_string(self, quote: str) -> str:
        start = self.pos
        src = self.source
        length = len(src)
        self.pos += 1
        while self.pos < length:
            c = src[self.pos]
            if c == '\\':
                self._read_string_escape()
                continue
            self.pos += 1
            if c == quote:
                return src[start:self.pos]
            if c in '\r\n':
                return src[start:self.pos]
        return src[start:self.pos]

    def _scan_template_content(
        self,
        start: int,
        close_kind: JsTokenKind,
        interp_kind: JsTokenKind,
        depth_delta: int,
    ) -> JsToken:
        src = self.source
        length = len(src)
        while self.pos < length:
            c = src[self.pos]
            if c == '\\':
                self._read_string_escape()
                continue
            if c == '`':
                self.pos += 1
                self._template_depth += depth_delta
                return JsToken(close_kind, src[start:self.pos], start)
            if c == '$' and self.pos + 1 < length and src[self.pos + 1] == '{':
                self.pos += 2
                if depth_delta == 0:
                    self._template_depth += 1
                self._brace_stack.append(0)
                return JsToken(interp_kind, src[start:self.pos], start)
            self.pos += 1
        self._template_depth += depth_delta
        return JsToken(close_kind, src[start:self.pos], start)

    def _read_template(self) -> JsToken:
        start = self.pos
        self.pos += 1
        return self._scan_template_content(
            start, JsTokenKind.TEMPLATE_FULL, JsTokenKind.TEMPLATE_HEAD, 0)

    def _resume_template(self) -> JsToken:
        start = self.pos
        self.pos += 1
        return self._scan_template_content(
            start, JsTokenKind.TEMPLATE_TAIL, JsTokenKind.TEMPLATE_MIDDLE, -1)

    def _read_regexp(self) -> str:
        start = self.pos
        src = self.source
        length = len(src)
        self.pos += 1
        in_class = False
        while self.pos < length:
            c = src[self.pos]
            if c == '\\' and self.pos + 1 < length:
                self.pos += 2
                continue
            if c == '[':
                in_class = True
                self.pos += 1
                continue
            if c == ']' and in_class:
                in_class = False
                self.pos += 1
                continue
            if c == '/' and not in_class:
                self.pos += 1
                while self.pos < length and src[self.pos].isalpha():
                    self.pos += 1
                return src[start:self.pos]
            if c in LINE_TERMINATORS:
                break
            self.pos += 1
        return src[start:self.pos]

    def _read_prefixed_int(self, start: int, valid_digits: str) -> JsToken:
        src = self.source
        length = len(src)
        while self.pos < length and src[self.pos] in valid_digits:
            self.pos += 1
        if self.pos < length and src[self.pos] == 'n':
            self.pos += 1
            return JsToken(JsTokenKind.BIGINT, src[start:self.pos], start)
        return JsToken(JsTokenKind.INTEGER, src[start:self.pos], start)

    def _read_number(self) -> JsToken:
        start = self.pos
        src = self.source
        length = len(src)

        if src[self.pos] == '0' and self.pos + 1 < length:
            nc = src[self.pos + 1]
            if nc in 'xX':
                self.pos += 2
                return self._read_prefixed_int(start, '0123456789abcdefABCDEF_')
            if nc in 'oO':
                self.pos += 2
                return self._read_prefixed_int(start, '01234567_')
            if nc in 'bB':
                self.pos += 2
                return self._read_prefixed_int(start, '01_')

        while self.pos < length and (src[self.pos] in _DECIMAL or src[self.pos] == '_'):
            self.pos += 1

        is_float = False
        if self.pos < length and src[self.pos] == '.':
            next_pos = self.pos + 1
            if next_pos < length and src[next_pos] in _DECIMAL:
                is_float = True
                self.pos += 1
                while self.pos < length and (
                    src[self.pos] in _DECIMAL or src[self.pos] == '_'
                ):
                    self.pos += 1

        if self.pos < length and src[self.pos] in 'eE':
            is_float = True
            self.pos += 1
            if self.pos < length and src[self.pos] in '+-':
                self.pos += 1
            while self.pos < length and (
                src[self.pos] in _DECIMAL or src[self.pos] == '_'
            ):
                self.pos += 1

        if not is_float and self.pos < length and src[self.pos] == 'n':
            self.pos += 1
            return JsToken(JsTokenKind.BIGINT, src[start:self.pos], start)

        kind = JsTokenKind.FLOAT if is_float else JsTokenKind.INTEGER
        return JsToken(kind, src[start:self.pos], start)

    def _read_identifier_or_keyword(self) -> JsToken:
        start = self.pos
        src = self.source
        length = len(src)
        while self.pos < length:
            c = src[self.pos]
            if c.isalnum() or c in _IDENTIFIER_PUNCTUATION or c in _IDENTIFIER_JOINERS:
                self.pos += 1
            elif _begins_unicode_escape(src, self.pos):
                self._read_string_escape()
            else:
                break
        word = src[start:self.pos]
        kw = KEYWORDS.get(word)
        if kw is not None:
            return JsToken(kw, word, start)
        return JsToken(JsTokenKind.IDENTIFIER, word, start)

    def tokenize(self) -> Generator[JsToken, None, None]:
        src = self.source
        length = len(src)

        if self.pos == 0 and src.startswith('#!'):
            self._read_line_comment()

        while True:
            self._skip_whitespace()
            if self._at_end():
                yield JsToken(JsTokenKind.EOF, '', self.pos)
                return

            start = self.pos
            c = src[self.pos]
            c2 = src[self.pos:self.pos + 2]

            if c == '\r' and self.pos + 1 < length and src[self.pos + 1] == '\n':
                self.pos += 2
                yield JsToken(JsTokenKind.NEWLINE, '\r\n', start)
                continue
            if c in LINE_TERMINATORS:
                self.pos += 1
                yield JsToken(JsTokenKind.NEWLINE, c, start)
                continue

            if c2 == '//':
                text = self._read_line_comment()
                yield JsToken(JsTokenKind.COMMENT, text, start)
                continue
            if c2 == '/*':
                text, has_newline = self._read_block_comment()
                yield JsToken(JsTokenKind.COMMENT, text, start)
                if has_newline:
                    yield JsToken(JsTokenKind.NEWLINE, '', self.pos)
                continue

            if c == "'":
                text = self._read_string("'")
                yield JsToken(JsTokenKind.STRING_SINGLE, text, start)
                continue
            if c == '"':
                text = self._read_string('"')
                yield JsToken(JsTokenKind.STRING_DOUBLE, text, start)
                continue
            if c == '`':
                yield self._read_template()
                continue

            if c == '}' and self._template_depth > 0 and self._brace_stack:
                if self._brace_stack[-1] == 0:
                    self._brace_stack.pop()
                    yield self._resume_template()
                    continue
                else:
                    self._brace_stack[-1] -= 1

            if c in _DECIMAL or (
                c == '.' and src[self.pos + 1:self.pos + 2] in _DECIMAL
            ):
                yield self._read_number()
                continue

            if _at_identifier_start(src, self.pos):
                yield self._read_identifier_or_keyword()
                continue

            if c == '#':
                if _at_identifier_start(src, self.pos + 1):
                    self.pos += 1
                    name = self._read_identifier_or_keyword()
                    yield JsToken(JsTokenKind.PRIVATE_IDENTIFIER, '#' + name.value, start)
                    continue

            c4 = src[self.pos:self.pos + 4]
            if c4 in _FOUR_CHAR_OPS:
                self.pos += 4
                yield JsToken(_FOUR_CHAR_OPS[c4], c4, start)
                continue

            c3 = src[self.pos:self.pos + 3]
            if c3 in _THREE_CHAR_OPS:
                self.pos += 3
                yield JsToken(_THREE_CHAR_OPS[c3], c3, start)
                continue

            if c2 in _TWO_CHAR_OPS:
                self.pos += 2
                yield JsToken(_TWO_CHAR_OPS[c2], c2, start)
                continue

            if c in _ONE_CHAR_OPS:
                self.pos += 1
                kind = _ONE_CHAR_OPS[c]
                if kind == JsTokenKind.LBRACE and self._brace_stack:
                    self._brace_stack[-1] += 1
                yield JsToken(kind, c, start)
                continue

            self.pos += 1
            yield JsToken(JsTokenKind.ERROR, c, start)


def decode_js_string_body(text: str) -> str:
    if '\\' not in text:
        return text
    parts: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        c = text[i]
        if c != '\\' or i + 1 >= length:
            parts.append(c)
            i += 1
            continue
        decoded, i = _decode_one_escape(text, i + 1, length)
        if decoded:
            parts.append(decoded)
    return ''.join(parts)
