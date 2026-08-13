"""
A ledger of JavaScript defects that are known, understood, and not yet fixed.

Every test states what a correct implementation would do, never what the code does today, and is
marked `unittest.expectedFailure`. An entry that starts passing is therefore reported as an
unexpected success, which fails the suite: an entry leaves this file when its defect is fixed and
its marker is removed, and never by quietly ceasing to be true.

Where the question is one about JavaScript rather than about this project, the answer was
established with Node.js and is quoted in the docstring of the test that pins it.
"""
from __future__ import annotations

import unittest

from collections import Counter

from test import TestBase

from refinery.lib.scripts import is_well_formed
from refinery.lib.scripts.js.model import JsIdentifier, JsPropertyDefinition
from refinery.lib.scripts.js.parser import JsParser
from refinery.lib.scripts.js.synth import JsSynthesizer
from refinery.units.scripting.js import js


_ASTRAL = chr(0x1F600)

_SPANNING = F"'a{_ASTRAL}b'"
"""
A JavaScript string literal whose middle character is above the basic multilingual plane. It is
four code units long and JavaScript indexes it as `a`, a high surrogate, a low surrogate, `b`,
while Python sees three code points, so every offset past the middle character differs by one.
"""


def _printed(source: str) -> str:
    return JsSynthesizer().convert(JsParser(source).parse())


def _well_formed(source: str) -> bool:
    return is_well_formed(JsParser(source).parse())


def _folded(source: str) -> str:
    return source.encode('utf8') | js() | str


def _folded_value(expression: str) -> str:
    """
    The text `refinery.js` prints for *expression* once it has folded it. The expression is placed
    in the argument of a `console.log` call, which survives because it is a side effect, so nothing
    but the fold decides what comes back.
    """
    printed = _folded(F'console.log({expression});')
    return printed.removeprefix('console.log(').removesuffix(');')


def _sole_property_definition(source: str) -> JsPropertyDefinition:
    return [
        node for node in JsParser(source).parse().walk()
        if isinstance(node, JsPropertyDefinition)
    ][0]


def _dropped_source_characters(source: str, printed: str) -> str:
    """
    The characters of `source` that `printed` does not account for, whitespace aside. Layout is the
    printer's to choose and a recovery may add brackets, so only a character that went missing is
    reported.
    """
    available = Counter(character for character in printed if not character.isspace())
    missing: list[str] = []
    for character in source:
        if character.isspace():
            continue
        if available[character] > 0:
            available[character] -= 1
        else:
            missing.append(character)
    return ''.join(missing)


class TestClassBodyIsItsOwnFunctionContext(TestBase):
    """
    A class body does not belong to the function that encloses it. Each field initializer is a
    function context of its own and so is each static block, and both are strict code, so `await`
    and `yield` mean there what they mean in a fresh body that is neither async nor a generator.
    """

    @unittest.expectedFailure
    def test_await_in_a_field_initializer_is_a_name(self):
        """
        Node accepts `async function f() { class C { p = await; } }`, and with `var await = 41` in
        scope `new C().p` is `41`. The initializer is not an async context, so `await` there is an
        ordinary identifier reference and not the operator the enclosing function would give it.
        """
        source = 'async function f() { class C { p = await; } }'
        self.assertEqual(type(_sole_property_definition(source).value), JsIdentifier)
        self.assertEqual(
            _printed(source), 'async function f() {\n  class C {\n    p = await;\n  }\n}'
        )

    @unittest.expectedFailure
    def test_the_await_operator_is_not_available_in_a_field_initializer(self):
        """
        Node refuses `async function f() { class C { p = await x; } }` with `SyntaxError:
        Unexpected identifier 'x'`, because `await` there is a name and no name may be followed by
        another one. Reading an operator the grammar does not offer in that position turns a file
        an engine rejects into a tree that claims to be a program.
        """
        self.assertEqual(_well_formed('async function f() { class C { p = await x; } }'), False)

    @unittest.expectedFailure
    def test_await_is_banned_in_a_class_static_block(self):
        """
        Node refuses `async function f() { class C { static { var await = 1; } } }` with
        `SyntaxError: Unexpected reserved word`. A static block bans `await` in every position, as
        a name no less than as an operator, and the enclosing async function does not change that.
        """
        source = 'async function f() { class C { static { var await = 1; } } }'
        self.assertEqual(_well_formed(source), False)

    @unittest.expectedFailure
    def test_yield_in_a_field_initializer_is_refused(self):
        """
        Node refuses `function* f() { class C { p = yield; } }` with `SyntaxError: Unexpected
        strict mode reserved word`. The initializer is not the generator's body, so `yield` is not
        the operator, and a class body is strict code, where `yield` is not a usable name either.
        """
        self.assertEqual(_well_formed('function* f() { class C { p = yield; } }'), False)

    @unittest.expectedFailure
    def test_yield_is_banned_in_a_class_static_block(self):
        """
        Node refuses `function* f() { class C { static { yield; } } }` with `SyntaxError:
        Unexpected strict mode reserved word`, for the reason the field initializer is refused for:
        the block is its own context and it is strict.
        """
        self.assertEqual(_well_formed('function* f() { class C { static { yield; } } }'), False)


class TestLexicalDeclarationIsNotAStatement(TestBase):
    """
    A `let` declaration is a Declaration, and the body of an `if` is a Statement, so a declaration
    can never be the single-statement body of one.
    """

    @unittest.expectedFailure
    def test_a_lexical_declaration_cannot_be_a_single_statement_body(self):
        """
        `if (0) let` followed by `x = 1;` on the next line is two statements. Since `let` cannot
        open a declaration in that position it is the name it spells, and automatic semicolon
        insertion ends the branch before `x`. Node runs the program and leaves `x` at `1`, which a
        parser that draws the assignment into the never-taken branch cannot reproduce.
        """
        script = JsParser('if (0) let\nx = 1;').parse()
        self.assertEqual(len(script.body), 2)
        self.assertEqual(JsSynthesizer().convert(script.body[1]), 'x = 1;')


class TestStringIsASequenceOfUtf16CodeUnits(TestBase):
    """
    A JavaScript string is a sequence of UTF-16 code units, so one character above the basic
    multilingual plane occupies two of them and every operation that counts or indexes sees two.
    """

    @unittest.expectedFailure
    def test_length_index_char_code_and_split_all_read_code_units(self):
        """
        Node answers `2`, `55357`, the lone high surrogate, and `2` for these four questions about
        one astral character. The model spells the rule as
        `refinery.lib.scripts.js.deobfuscation.helpers.utf16_code_units` and applies it at two call
        sites, so `split` is right while length, indexing, and `charCodeAt` are answered from
        Python code points, where the same character is one unit.
        """
        source = (
            F"console.log('{_ASTRAL}'.length, '{_ASTRAL}'.charCodeAt(0), "
            F"'{_ASTRAL}'[0], '{_ASTRAL}'.split('').length);"
        )
        self.assertEqual(_folded(source), R"console.log(2, 55357, '\uD83D', 2);")


class TestStringMethodsIndexInUtf16CodeUnits(TestBase):
    """
    A `String.prototype` method that takes or returns an index or a length counts in UTF-16 code
    units, so over `_SPANNING` every offset past the middle character is one greater than the count
    of Python code points. Each call pinned here is folded today, which is what makes this worse
    than a read that is left alone: a wrong constant is written into the deobfuscated file, where
    nothing recalls what the source said.
    """

    @unittest.expectedFailure
    def test_char_at_returns_one_code_unit(self):
        """
        Node answers with the high surrogate 0xD83D at index 1 and the low surrogate 0xDE00 at
        index 2. `charAt` names a single code unit, so it can never answer with a whole character
        that spans two of them, and it can never run out of string before index 3.
        """
        self.assertEqual(
            (
                _folded_value(F'{_SPANNING}.charAt(1)'),
                _folded_value(F'{_SPANNING}.charAt(2)'),
            ),
            (R"'\uD83D'", R"'\uDE00'"),
        )

    @unittest.expectedFailure
    def test_at_indexes_code_units_from_either_end(self):
        """
        Node answers with the high surrogate 0xD83D at index 1 and the low surrogate 0xDE00 at
        index -2. A negative index counts back from a length that is also measured in code units,
        so the two ends of the string have to agree about where the middle character sits.
        """
        self.assertEqual(
            (
                _folded_value(F'{_SPANNING}.at(1)'),
                _folded_value(F'{_SPANNING}.at(-2)'),
            ),
            (R"'\uD83D'", R"'\uDE00'"),
        )

    @unittest.expectedFailure
    def test_slice_cuts_between_code_units(self):
        """
        Node answers with the astral character for `slice(1, 3)` and with the low surrogate 0xDE00
        followed by `b` for both `slice(2)` and `slice(-2)`. A cut may fall between the two halves
        of a character, and a slice that starts on the second half begins with a lone surrogate.
        """
        self.assertEqual(
            (
                _folded_value(F'{_SPANNING}.slice(1, 3)'),
                _folded_value(F'{_SPANNING}.slice(2)'),
                _folded_value(F'{_SPANNING}.slice(-2)'),
            ),
            (F"'{_ASTRAL}'", R"'\uDE00b'", R"'\uDE00b'"),
        )

    @unittest.expectedFailure
    def test_substring_cuts_between_code_units(self):
        """
        Node answers with the high surrogate 0xD83D for `substring(1, 2)` and with the low
        surrogate 0xDE00 followed by `b` for `substring(2)`.
        """
        self.assertEqual(
            (
                _folded_value(F'{_SPANNING}.substring(1, 2)'),
                _folded_value(F'{_SPANNING}.substring(2)'),
            ),
            (R"'\uD83D'", R"'\uDE00b'"),
        )

    @unittest.expectedFailure
    def test_substr_counts_its_length_in_code_units(self):
        """
        Node answers with the astral character for `substr(1, 2)` and with the low surrogate 0xDE00
        for `substr(2, 1)`. The second argument is a count of code units and not of characters, so
        two of them are what it takes to name the middle character.
        """
        self.assertEqual(
            (
                _folded_value(F'{_SPANNING}.substr(1, 2)'),
                _folded_value(F'{_SPANNING}.substr(2, 1)'),
            ),
            (F"'{_ASTRAL}'", R"'\uDE00'"),
        )

    @unittest.expectedFailure
    def test_index_of_reports_a_code_unit_offset(self):
        """
        Node answers `3` for both, since `b` follows two surrogates. The returned offset is what a
        caller feeds back into an index, so reporting it one short misplaces every later cut, and
        searching from position 3 must still find the character that sits there.
        """
        self.assertEqual(
            (
                _folded_value(F"{_SPANNING}.indexOf('b')"),
                _folded_value(F"{_SPANNING}.indexOf('b', 3)"),
            ),
            ('3', '3'),
        )

    @unittest.expectedFailure
    def test_last_index_of_reports_a_code_unit_offset(self):
        """
        Node answers `3`, the same offset the forward search reports for the only `b` in the
        string.
        """
        self.assertEqual(_folded_value(F"{_SPANNING}.lastIndexOf('b')"), '3')

    @unittest.expectedFailure
    def test_pad_start_measures_the_target_length_in_code_units(self):
        """
        Node answers with two dashes before the string. The target length is compared against a
        length of four, so exactly two units are wanted; counting the astral character once asks
        for one dash too many and changes what the padded string says.
        """
        self.assertEqual(
            _folded_value(F"{_SPANNING}.padStart(6, '-')"),
            F"'--a{_ASTRAL}b'",
        )

    @unittest.expectedFailure
    def test_pad_end_measures_the_target_length_in_code_units(self):
        """
        Node answers with two dashes after the string, for the reason `padStart` gets two before
        it.
        """
        self.assertEqual(
            _folded_value(F"{_SPANNING}.padEnd(6, '-')"),
            F"'a{_ASTRAL}b--'",
        )

    @unittest.expectedFailure
    def test_includes_takes_a_code_unit_position(self):
        """
        Node answers `true`: the search starts at code unit 3, which is where `b` is. A position
        measured in code points starts the search past the end of the string and finds nothing.
        """
        self.assertEqual(_folded_value(F"{_SPANNING}.includes('b', 3)"), 'true')

    @unittest.expectedFailure
    def test_starts_with_takes_a_code_unit_position(self):
        """
        Node answers `true`, because the string does start with `b` when read from code unit 3.
        """
        self.assertEqual(_folded_value(F"{_SPANNING}.startsWith('b', 3)"), 'true')

    @unittest.expectedFailure
    def test_ends_with_takes_a_code_unit_end_position(self):
        """
        Node answers `true`: the first three code units end with the two that spell the astral
        character. An end position read as a code point cuts one unit short of it.
        """
        self.assertEqual(
            _folded_value(F"{_SPANNING}.endsWith('{_ASTRAL}', 3)"),
            'true',
        )


class TestRecoveryKeepsTheSourceText(TestBase):
    """
    `refinery.lib.scripts.js.model.JsErrorNode` promises that a span no parser could read is kept
    verbatim, so that what an analyst gets back still contains what was written.
    """

    @unittest.expectedFailure
    def test_no_source_character_is_dropped_by_recovery(self):
        """
        Node refuses both of these, with `Unexpected identifier 'b'` and `Invalid regular
        expression: missing /`, which is precisely when that promise carries the analysis. Whatever
        the recovery makes of the text, no character of it may be missing from what is printed:
        a dropped character is a payload the analyst never sees.
        """
        sources = ['x = y[a b]', 'x = y.replace(/[^a-z']
        dropped = tuple(_dropped_source_characters(s, _printed(s)) for s in sources)
        self.assertEqual(dropped, ('', ''))


class TestPrintingIsIdempotent(TestBase):
    """
    Printing a parse and parsing that print has to reach a fixed point, including for a source no
    engine accepts, because a tool that reads its own output otherwise changes a file every pass.
    """

    @unittest.expectedFailure
    def test_printing_an_unterminated_regular_expression_twice_is_stable(self):
        """
        Node refuses `x = /ab+` with `SyntaxError: Invalid regular expression: missing /`, so what
        the parser makes of it is a recovery and its shape is the project's to choose. Whichever
        shape that is, printing the parse of the print must give the print back unchanged.
        """
        once = _printed('x = /ab+')
        self.assertEqual(_printed(once), once)


class TestCommentWithNoFollowingStatement(TestBase):
    """
    A comment is carried by the statement it precedes, which leaves a comment that precedes nothing
    with no carrier.
    """

    @unittest.expectedFailure
    def test_a_comment_that_no_statement_follows_is_kept(self):
        """
        A trailing note, marker, or half-written annotation is text the file contains, and a
        deobfuscator that drops it loses source it was handed. Each of these three programs is
        already in the form the printer emits, so each has to print back exactly as written.
        """
        sources = ['x = 1;\n/* note */', 'x = 1;\n// note', 'x = 1;\n/* note']
        self.assertEqual(tuple(_printed(source) for source in sources), tuple(sources))


class TestWellFormednessRefusesANonProgram(TestBase):
    """
    `refinery.lib.scripts.is_well_formed` is the domain over which fidelity is stated: it holds
    when every node in the tree spells something a parser agreed to read. A source no engine
    accepts is not such a tree, and a caller that is told otherwise compares a fabrication against
    the file it came from.
    """

    @unittest.expectedFailure
    def test_an_unclosed_parenthesis_is_not_a_well_formed_program(self):
        """
        Node refuses `var x = (1 + 2; g(x);` with `SyntaxError: Unexpected token ';'`. Supplying
        the bracket nobody wrote makes a truncated file read as though it had been written closed.
        """
        self.assertEqual(_well_formed('var x = (1 + 2; g(x);'), False)

    @unittest.expectedFailure
    def test_an_arrow_function_is_not_an_update_target(self):
        """
        Node refuses `f = a => {}++` with `SyntaxError: Unexpected token '++'`. An update operator
        needs an operand it can write back to, and a function made on the spot is not a reference.
        """
        self.assertEqual(_well_formed('f = a => {}++'), False)

    @unittest.expectedFailure
    def test_a_function_expression_is_not_an_update_target(self):
        """
        Node refuses `f = function () {}++` with `SyntaxError: Invalid left-hand side expression in
        postfix operation`, naming the same missing target the arrow form lacks.
        """
        self.assertEqual(_well_formed('f = function () {}++'), False)
