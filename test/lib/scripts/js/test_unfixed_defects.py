"""
A ledger of JavaScript defects that are known, understood, and not yet fixed.

Every test states what a correct implementation would do, never what the code does today, and is
marked `unittest.expectedFailure`. An entry that starts passing is therefore reported as an
unexpected success, which fails the suite: an entry leaves this file when its defect is fixed and
its marker is removed, and never by quietly ceasing to be true.

Where the question is one about JavaScript rather than about this project, the answer was
established with Node.js and is quoted in the docstring of the test that pins it.

An entry is quantified over the rows its defect is about, and where those rows belong to a corpus
some law elsewhere is stated over, they stay in that corpus and are imported here. The module
holding them is named in the docstring of the entry that pins them, and a row rejoins the law the
day the marker comes off.
"""
from __future__ import annotations

import unittest

from collections import Counter
from collections.abc import Callable

from test import TestBase
from test.lib.scripts.js.analysis.differential import (
    behavior,
    deobfuscate_source,
    host_behavior,
    node_executable,
)
from test.lib.scripts.js.analysis.test_differential import (
    SPELLINGS_A_FOLD_WRITES_AS_A_PLAIN_STRING,
    SPELLINGS_A_FOLD_WRITES_AS_THE_DIRECTIVE,
    a_file_holding_an_octal_literal_opening_with,
    a_function_body_opening_with,
    a_script_opening_with,
    a_script_whose_directive_stands_below,
)
from test.lib.scripts.js.deobfuscation.test_array_length_reads import (
    A_COUNT_THE_FOLD_DOES_NOT_REACH,
)
from test.lib.scripts.js.test_for_statement_head import (
    HEADS_THE_TOOL_MISREADS,
    node_says,
    printed_from_the_parse,
    printed_without_the_brackets,
)
from test.lib.scripts.js.test_template_literal import AN_ESCAPE_NEITHER_LITERAL_HAS
from test.lib.scripts.js.test_truncated_source import FOLDS_ANSWERED_WITH_A_PROGRAM

from refinery.lib.scripts import UnspellableNode, is_well_formed
from refinery.lib.scripts.js.model import JsIdentifier, JsPropertyDefinition, JsStringLiteral
from refinery.lib.scripts.js.parser import JsParser
from refinery.lib.scripts.js.synth import JsSynthesizer
from refinery.units.scripting.js import js


_ASTRAL = chr(0x1F600)


_ASTRAL_LETTER = chr(0x1D465)


def _printed(source: str) -> str:
    return JsSynthesizer().convert(JsParser(source).parse())


def _well_formed(source: str) -> bool:
    return is_well_formed(JsParser(source).parse())


def _folded(source: str) -> str:
    return source.encode('utf8') | js() | str


def _sole_property_definition(source: str) -> JsPropertyDefinition:
    return [
        node for node in JsParser(source).parse().walk()
        if isinstance(node, JsPropertyDefinition)
    ][0]


def _sole_string_literal(source: str) -> JsStringLiteral:
    return [
        node for node in JsParser(source).parse().walk()
        if isinstance(node, JsStringLiteral)
    ][0]


def _refuses_to_print(source: str) -> bool:
    """
    Whether `refinery.js` declines to write anything for *source*, which is the only answer that can
    be given for a buffer holding a literal no text spells.
    """
    try:
        _folded(source)
    except UnspellableNode:
        return True
    else:
        return False


def _what_the_engine_says_after(printing: Callable[[str], str]) -> list[str]:
    """
    What Node makes of the text *printing* produces for each head of `HEADS_THE_TOOL_MISREADS`, with
    a refusal to print reported in place of an output: declining to write anything is an answer a
    route can give, and it is not the one the corpus records either.
    """
    answers: list[str] = []
    for head in HEADS_THE_TOOL_MISREADS:
        try:
            printed = printing(head.source)
        except UnspellableNode as refusal:
            answers.append(F'{type(refusal.node).__name__} has no spelling')
        else:
            answers.append(node_says(printed))
    return answers


def _before_and_after(source: str) -> tuple[tuple[str, str | None], tuple[str, str | None]]:
    """
    What Node makes of *source* and what it makes of the text `refinery.js` deobfuscates it to,
    reported together because the law is that the two agree.
    """
    return behavior(source), behavior(deobfuscate_source(source))


def _before_and_after_as_a_script(
    source: str,
) -> tuple[tuple[str, str | None], tuple[str, str | None]]:
    """
    The same pair as `_before_and_after`, with both programs run as classic global scripts.

    `behavior` runs a file as a CommonJS module, which wraps the whole of it in a function, so the
    top of the file is the top of a function body there and never the top of a script. A law about
    what the first statement of a script is has to be witnessed where the file has one.
    """
    return host_behavior(source), host_behavior(deobfuscate_source(source))


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


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestTheForHeadBanOnInIsLiftedByWhatOpensAnExpressionOfItsOwn(TestBase):
    """
    The head of a `for` loop bans the `in` operator so that the word is free to open a for-in, and
    every construct that starts an expression of its own lifts the ban again: the arguments of
    `new`, the block body of a function or an arrow, a template substitution, the brackets of a
    computed member, and the consequent of a conditional. The heads of
    `test.lib.scripts.js.test_for_statement_head.HEADS_THE_TOOL_MISREADS` are read with the ban
    still in force, so the word is not read as the operator where the grammar offers it and what the
    parser makes of the text is a tree the file does not spell. Each entry below is one of the three
    routes from a source to an output that the law in that module states, quantified over the heads
    the law had to leave out.
    """

    @unittest.expectedFailure
    def test_printing_a_parsed_head_keeps_what_it_does(self):
        """
        Node prints what the corpus records for each of these loops, `[1]` among them for
        `for (r = new Array("k" in b); ...)`, whose single argument is one boolean. Read with the
        ban in force, that argument becomes two, and what comes back is a program printing `[2]`:
        the same file, a different answer, and nothing to say it changed. The rest come back as a
        `SyntaxError` or not at all, the template head reaching the printer as a run no text spells.
        """
        self.assertEqual(
            _what_the_engine_says_after(printed_from_the_parse),
            [head.prints for head in HEADS_THE_TOOL_MISREADS],
        )

    @unittest.expectedFailure
    def test_printing_a_head_whose_brackets_left_the_tree_keeps_what_it_does(self):
        """
        The same heads, printed after every bracket in the tree is replaced by what it holds, which
        is the tree a pass folding into a bracketed slot leaves behind. Node prints what the corpus
        records: no bracket in any of these heads carries meaning, so removing them all may not
        change what the loop does.
        """
        self.assertEqual(
            _what_the_engine_says_after(printed_without_the_brackets),
            [head.prints for head in HEADS_THE_TOOL_MISREADS],
        )

    @unittest.expectedFailure
    def test_deobfuscating_a_head_keeps_what_it_does(self):
        """
        The same heads again, this time as the deobfuscation passes leave them. Node prints what the
        corpus records, which is the whole promise: what an analyst is handed back runs the way the
        file they handed over ran.
        """
        self.assertEqual(
            _what_the_engine_says_after(deobfuscate_source),
            [head.prints for head in HEADS_THE_TOOL_MISREADS],
        )


class TestAStringLiteralReadsOnlyTheEscapesTheGrammarHas(TestBase):
    """
    A string literal and a template part company over the escapes each of them reads, but neither
    reads an escape the grammar has no rule for at all.
    """

    @unittest.expectedFailure
    def test_a_string_holding_an_escape_the_grammar_lacks_denotes_nothing(self):
        """
        Node refuses every one of the twelve files in
        `test.lib.scripts.js.test_template_literal.AN_ESCAPE_NEITHER_LITERAL_HAS` with a SyntaxError
        naming the escape: a hexadecimal escape short of its digits or holding a character that is
        not one, and a braced code point that is empty, out of range, unterminated, or written with
        anything besides hexadecimal digits. None of these files is a program in any mode, so no
        text is what the literal carries. The reader drops the backslash and answers with the
        letters that followed it, which reports a value for a file that has none, and the template
        beside each string is already refused.
        """
        self.assertEqual(
            [_sole_string_literal(string).value for string, _ in AN_ESCAPE_NEITHER_LITERAL_HAS],
            [None] * len(AN_ESCAPE_NEITHER_LITERAL_HAS),
        )


class TestAnIdentifierNamedKeyReadsAsUtf16CodeUnits(TestBase):
    """
    An object property key written as a bare identifier is a JavaScript string when it is read
    back, and it is the same sequence of UTF-16 code units as the identical key written as a string
    literal. One character above the basic multilingual plane occupies two code units, so a key
    named by a bare identifier spelled with U+1D465 and the same key named by a string literal are
    read alike, and each is read as Node reads it. The identifier-named key is read as one code
    point instead, so the two spellings of the one key disagree.
    """

    @unittest.expectedFailure
    def test_a_bare_identifier_key_reads_the_same_code_units_as_its_literal(self):
        """
        Node answers `55349` and `56421`, the high and low surrogate of U+1D465, for
        `Object.keys({K: 1})[0].charCodeAt(0)` and `.charCodeAt(1)` when `K` is that character
        written as a bare identifier, and answers the same two when `K` is written as the string
        literal `'K'`: the property key read back is a string of two code units however the key was
        spelled. The literal-named key is folded to those two, so the identifier-named key has to
        fold to the same two.
        """
        def program(key: str) -> str:
            return F'console.log({key}.charCodeAt(0), {key}.charCodeAt(1));'
        identifier_key = F'Object.keys({{{_ASTRAL_LETTER}: 1}})[0]'
        literal_key = F"Object.keys({{'{_ASTRAL_LETTER}': 1}})[0]"
        reads_as_the_two_surrogates = 'console.log(55349, 56421);'
        self.assertEqual(
            (_folded(program(identifier_key)), _folded(program(literal_key))),
            (reads_as_the_two_surrogates, reads_as_the_two_surrogates),
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

    @unittest.expectedFailure
    def test_printing_a_name_the_source_never_wrote_twice_is_stable(self):
        """
        Node refuses each of these six with `SyntaxError: Unexpected end of input`. Where a binding
        name, a property name, or a method name was expected and none was written, the recovery
        leaves a `refinery.lib.scripts.js.model.JsErrorNode` holding no text, and the printer writes
        nothing for it: whatever stood after the gap comes to rest against the word in front of it,
        and the next read takes that for the name the source never wrote. `var ;` binds a name
        spelled `;`, `x = y.;` reads the `;` as the property, and `x = { get () {} }` reads the
        accessor keyword as the method name.
        """
        sources = ['var', 'var a = 1,', 'x = y.', 'x = a?.', 'delete a.', 'x = { get']
        once = [_printed(source) for source in sources]
        self.assertEqual([_printed(text) for text in once], once)

    @unittest.expectedFailure
    def test_printing_a_parameter_list_with_no_arrow_behind_it_twice_is_stable(self):
        """
        Node refuses `x = ()` and `x = (a,)` with `SyntaxError: Unexpected token ')'`, `x = (...a)`
        with `SyntaxError: Unexpected token '...'`, and the two `new` forms with `SyntaxError:
        Unexpected end of input`. A bracket holding nothing, a trailing comma, or a rest element is
        spelled by an arrow head and by no other expression, so the recovery builds an arrow
        function whose body is an error node reading `a parameter list with no arrow behind it`.
        The printer writes the `=>` the file never had and nothing for the body, and the terminator
        that comes to rest behind the arrow is what the next read gives the body. In the two `new`
        forms the bracketed list is one the printer itself wrote for a callee the source left
        empty, so the arrow arrives on the second print and the text is still growing on the third.
        """
        sources = ['x = ()', 'x = (a,)', 'x = (...a)', 'x = new', 'throw new']
        once = [_printed(source) for source in sources]
        self.assertEqual([_printed(text) for text in once], once)

    @unittest.expectedFailure
    def test_printing_a_statement_the_source_never_wrote_twice_is_stable(self):
        """
        Node refuses each of these five with `SyntaxError: Unexpected end of input`. Each is a
        statement whose body the file stops short of, and the recovery stands an error node holding
        no text where that body belongs. The printer gives a single statement a block of its own,
        so it writes a line for a statement that spells nothing and the block comes out as a blank
        line between braces; reading that back finds a block with no statement in it at all, which
        prints tight. The first print is longer than every print after it.
        """
        sources = [
            'if (a)',
            'while (a)',
            'with (o)',
            'for (const v of a)',
            'if (a) { f(); } else',
        ]
        once = [_printed(source) for source in sources]
        self.assertEqual([_printed(text) for text in once], once)

    @unittest.expectedFailure
    def test_printing_a_heritage_clause_the_source_never_wrote_twice_is_stable(self):
        """
        Node refuses `class D extends` with `SyntaxError: Unexpected end of input` and refuses what
        printing it gives back with the same message. The recovery leaves an error node holding no
        text where the superclass belongs and the printer writes nothing for it, so
        `class D extends  {}` offers the class body to the slot an expression is read from. The next
        parse takes those braces for an object literal superclass, leaves the class with a body
        nobody wrote, and prints `class D extends {} {}`, which Node accepts: two passes turn a file
        that was cut into a program saying something the file never said.
        """
        once = _printed('class D extends')
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


#: Files that stop in the middle of a construct, grouped by the construct each one stops inside of.
#: No engine reads any of them, and this parser reads every one of them as a program, so the law
#: below is quantified over all of them at once: the group a row is in is only what the parser had
#: to finish writing in order to answer with a program at all.
SOURCES_THE_RECOVERY_SILENTLY_COMPLETES = {
    'a function': (
        'function',
        'function f',
        'function f(',
        'function f(a, b',
        'function f() {',
        'function f() { g();',
        'function* g() {',
        'async function h() {',
        'x = function (',
        'x = () => {',
    ),
    'a class': (
        'class',
        'class Foo',
        'class Foo {',
        'class Foo extends Bar {',
        'class Foo { m(',
        'class Foo { m() {',
        'class Foo { m() { g();',
        'class Foo { static {',
        'x = class {',
    ),
    'a statement': (
        'if (a) {',
        'if (a) { f();',
        'while (a) {',
        'for (;;) {',
        'for (const v of a) {',
        'with (o) {',
        'label: {',
        'try {',
        'try { f();',
        'try {} catch',
        'try {} catch (e',
        'try {} catch (e) {',
        'try { f(); } catch (e) { g();',
        'try {} finally',
        'switch (x',
        'switch (x) {',
        'switch (x) { case 1:',
        'switch (x) { case 1: f();',
        'switch (x) { default:',
    ),
    'a bracketed expression': (
        'x = (1 + 2',
        'x = { a',
        'x = { a: 1',
        'x = { a: 1, b: 2',
        'x = [1, 2',
        'x = [1, 2, 3',
        'x = f(1, 2',
        'x = f(g(1), 2',
        'x = a.b(',
        'x = new C(',
    ),
    'a binding pattern': (
        'const {',
        'const { a',
        'const { a, b',
        'const [',
        'const [a, b',
        'const { a: { b',
        'const [a, [b',
        'try {} catch ({ a',
    ),
    'an export declaration': (
        'export {',
        'export { a',
        'export function f() {',
        'export default function () {',
    ),
}


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

    @unittest.expectedFailure
    def test_a_file_that_stops_inside_a_construct_is_not_a_well_formed_program(self):
        """
        The parser answers a file that was cut off inside a construct by writing the token it was
        waiting for — a name, a bracket, a brace, a body — and records nowhere that it did so: not
        one of the sixty files in `SOURCES_THE_RECOVERY_SILENTLY_COMPLETES` leaves a
        `refinery.lib.scripts.js.model.JsErrorNode` anywhere in the tree, so every one of them is
        reported as a program and the cut is gone. `x = f(1, 2` is printed as `x = f(1, 2);`, which
        is a program that runs; `try {} catch` is printed as `try {} catch {}`, a handler no file
        wrote; `export {` is printed as `export {  };`.

        No engine reads any of them. `new vm.Script` refuses every row but the export declarations,
        each with `SyntaxError: Unexpected end of input` except `x = f(1, 2` and `x = f(g(1), 2`,
        which it refuses with `missing ) after argument list`. The export declarations are put to
        `new vm.SourceTextModule` under `node --experimental-vm-modules`, because `vm.Script`
        refuses them for a reason of its own, `Unexpected token 'export'`; that host refuses all
        four with `Unexpected end of input` and accepts `export {};`, `var a; export { a };`,
        `export function f() {}` and `export default function () {}`, so what it refuses is the cut
        and not the keyword.
        """
        sources = [
            source
            for group in SOURCES_THE_RECOVERY_SILENTLY_COMPLETES.values()
            for source in group
        ]
        self.assertEqual(
            {source: _well_formed(source) for source in sources},
            {source: False for source in sources},
        )


class TestACarvedFileIsNotAnsweredWithAProgram(TestBase):
    """
    A buffer carved out of memory can stop in the middle of a literal, and the literal it stopped
    inside is then spelled by no text at all. Refusing to print is the only answer that keeps that
    visible, because an analyst holding a clean program has no way left to tell that the file they
    handed over was cut.
    """

    @unittest.expectedFailure
    def test_a_fold_that_reaches_a_literal_the_cut_left_open_is_refused(self):
        """
        Node refuses every carved file in
        `test.lib.scripts.js.test_truncated_source.FOLDS_ANSWERED_WITH_A_PROGRAM` and accepts each
        of them with its delimiter restored, so the missing quote is the whole of the difference
        between a program and a buffer that is not one. In each of these the declaration the cut
        left open is read by nothing before the cut, so it is dropped as dead code and the literal
        no text spells never reaches the printer: what comes back is the head of the file, whole,
        and it says nothing about what was lost.
        """
        carved = FOLDS_ANSWERED_WITH_A_PROGRAM
        self.assertEqual(
            {name: _refuses_to_print(fold.cut) for name, fold in carved.items()},
            {name: True for name in carved},
        )


#: A parameter list that holds a default value, a rest element, or a destructuring pattern, mapped
#: to a call that reaches a body reading it, the expression such a body returns, and what Node
#: prints for that call. A parameter list written any of those ways is not a simple one, and a
#: function with such a list may hold no Use Strict Directive at all.
_A_PARAMETER_LIST_NO_DIRECTIVE_MAY_STAND_UNDER = {
    'a = 1': ('f()', 'a', '1\n'),
    '...a': ('f(1, 2)', 'a.length', '2\n'),
    '{a}': ('f({a: 5})', 'a', '5\n'),
}


def _a_function_whose_body_opens_with(head: str, parameters: str, call: str, read: str) -> str:
    """
    A program calling a function written with *parameters* as *call*, whose body opens with *head*
    and reads *read* below it.
    """
    return (
        F'function f({parameters}) {{ {head} return {read}; }}\n'
        F'console.log({call});\n'
    )


#: A function whose parameter list forbids a directive, whose body holds a `'use strict'` one
#: statement in, mapped to what Node prints for it. The statement above the directive is the `atob`
#: call of
#: `test.lib.scripts.js.analysis.test_differential.SPELLINGS_A_FOLD_WRITES_AS_A_PLAIN_STRING`, which
#: is not a string literal and therefore ends the prologue in front of the directive.
A_FUNCTION_WHOSE_PARAMETER_LIST_FORBIDS_A_DIRECTIVE = {
    _a_function_whose_body_opens_with(
        "atob('YQ=='); 'use strict';", parameters, call, read): prints
    for parameters, (call, read, prints) in _A_PARAMETER_LIST_NO_DIRECTIVE_MAY_STAND_UNDER.items()
}


#: A function whose parameter list forbids a directive, whose body opens with a statement that is
#: none but that a fold writes as the plain spelling of one, mapped to what Node prints for it.
A_FUNCTION_WHOSE_BODY_OPENS_WITH_A_FOLD_TO_THE_DIRECTIVE = {
    _a_function_whose_body_opens_with(head, parameters, call, read): prints
    for parameters, (call, read, prints) in _A_PARAMETER_LIST_NO_DIRECTIVE_MAY_STAND_UNDER.items()
    for head in SPELLINGS_A_FOLD_WRITES_AS_THE_DIRECTIVE
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAFoldDoesNotWriteADirectiveWhereNoneWasWritten(TestBase):
    """
    A directive is a string literal written plainly at the top of a script or of a function body,
    and nothing else is one: a bracket around the literal, an operator beside it, or a call that
    computes the same text each leaves a statement that is merely evaluated. Folding one of those to
    the text it denotes writes the plain spelling into directive position, and the whole script or
    body around it turns strict.
    """

    @unittest.expectedFailure
    def test_a_statement_that_only_denotes_the_text_leaves_the_script_sloppy(self):
        """
        Node prints `false` for each of these programs: the probe is called with no receiver, so
        `this` in its body is the global object, which is what sloppy code gives. The fold arrives
        at `'use strict';` as the first statement of the file, and Node prints `true` for what comes
        back, having read a directive nobody wrote.
        """
        sloppy = ('false\n', None)
        spellings = SPELLINGS_A_FOLD_WRITES_AS_THE_DIRECTIVE
        self.assertEqual(
            [_before_and_after(a_script_opening_with(head)) for head in spellings],
            [(sloppy, sloppy)] * len(spellings),
        )

    @unittest.expectedFailure
    def test_a_statement_that_only_denotes_the_text_leaves_the_function_body_sloppy(self):
        """
        Node prints `false` for each of these too, the statement standing at the top of the probe's
        own body rather than of the file. A directive there governs the body it opens, so the fold
        turns that one function strict while the file around it stays as it was.
        """
        sloppy = ('false\n', None)
        spellings = SPELLINGS_A_FOLD_WRITES_AS_THE_DIRECTIVE
        self.assertEqual(
            [_before_and_after(a_function_body_opening_with(head)) for head in spellings],
            [(sloppy, sloppy)] * len(spellings),
        )

    @unittest.expectedFailure
    def test_a_file_that_holds_an_octal_literal_still_parses(self):
        """
        Node prints `8` for each of these, an octal literal being a number in sloppy code and one of
        the spellings strict mode forbids outright. A directive that appears where none was written
        costs the file its ability to parse at all, so what comes back is not a program: Node reads
        it as a SyntaxError and the analysis of the file stops there.
        """
        eight = ('8\n', None)
        spellings = SPELLINGS_A_FOLD_WRITES_AS_THE_DIRECTIVE
        self.assertEqual(
            [
                _before_and_after(a_file_holding_an_octal_literal_opening_with(head))
                for head in spellings
            ],
            [(eight, eight)] * len(spellings),
        )

    @unittest.expectedFailure
    def test_a_statement_folded_to_a_plain_string_does_not_extend_the_prologue(self):
        """
        Node prints `false` for each of the five files
        `test.lib.scripts.js.analysis.test_differential.SPELLINGS_A_FOLD_WRITES_AS_A_PLAIN_STRING`
        builds: none of the heads is a string literal, so the prologue ends at it and the
        `'use strict'` below it governs nothing. Each fold writes a string literal there, and Node
        prints `true` for what comes back, having read a directive two statements after the file
        stopped offering one. The read among them is one a fold already declines in this position,
        written inside a bracket the printer removes.
        """
        sloppy = ('false\n', None)
        spellings = SPELLINGS_A_FOLD_WRITES_AS_A_PLAIN_STRING
        self.assertEqual(
            [_before_and_after(a_script_whose_directive_stands_below(head)) for head in spellings],
            [(sloppy, sloppy)] * len(spellings),
        )

    @unittest.expectedFailure
    def test_a_fold_writes_no_prologue_into_a_function_that_can_hold_none(self):
        """
        Node prints `1`, `2`, and `5` for the three programs of
        `A_FUNCTION_WHOSE_PARAMETER_LIST_FORBIDS_A_DIRECTIVE`, each of which runs a body holding a
        `'use strict'` that governs nothing, one statement below a call. The fold writes a string
        literal where that call stood, which extends the prologue over the line below it, and a
        function whose parameter list is not simple may not open with that directive under any
        circumstances. Node refuses all three of these with a SyntaxError, so the cost here is not a
        body that reports the wrong mode but a file that is no longer a program at all.
        """
        rows = A_FUNCTION_WHOSE_PARAMETER_LIST_FORBIDS_A_DIRECTIVE
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            _each_program_still_prints(rows),
        )

    @unittest.expectedFailure
    def test_a_fold_writes_no_directive_into_a_function_that_can_hold_none(self):
        """
        Node prints `1`, `2`, and `5` for the programs of
        `A_FUNCTION_WHOSE_BODY_OPENS_WITH_A_FOLD_TO_THE_DIRECTIVE`, one for each way of writing a
        parameter list that is not simple, crossed with every spelling that denotes the text of the
        directive without being one. Each body opens with a statement that is evaluated and
        discarded, so each function is sloppy code that a parameter list of that shape is welcome
        in. The fold writes the plain `'use strict'` in that statement's place, which a function
        whose parameter list is not simple may not hold under any circumstances: Node refuses every
        one of these with a SyntaxError. What the whole file is worth is lost here, and not merely
        the mode of the one body — the same fold in a function whose parameters are simple is
        pinned by `test_a_statement_that_only_denotes_the_text_leaves_the_function_body_sloppy`,
        where the file that comes back still runs and reports the wrong mode.
        """
        rows = A_FUNCTION_WHOSE_BODY_OPENS_WITH_A_FOLD_TO_THE_DIRECTIVE
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            _each_program_still_prints(rows),
        )


class TestADecodeReadsBackTheCharactersNoEscapeIntroduced(TestBase):
    """
    `decodeURIComponent` copies every character of its argument that no escape introduced straight
    into its answer, and a character above the basic multilingual plane is two code units there as
    it is anywhere else. For `C` the one character U+1F600, Node answers `decodeURIComponent(C)`
    with `C` and `decodeURIComponent(C + '%41')` with `C` followed by an `A`; and for an argument
    that is the lone high surrogate D800 it answers with that surrogate, an argument holding no
    escape being one a decode has nothing to refuse about.

    The fold refuses all three. A value is held as the code units a JavaScript string is made of, so
    the surrogates the argument itself spells are read as though the decoded bytes had produced
    them, and a call over an ordinary string is left standing for want of an answer.
    """

    @unittest.expectedFailure
    def test_a_character_the_argument_already_holds_survives_the_decode(self):
        backslash = chr(92)
        lone_surrogate = F'{backslash}uD800'
        programs = [
            F"console.log(decodeURIComponent('{_ASTRAL}'));",
            F"console.log(decodeURIComponent('{_ASTRAL}%41'));",
            F"console.log(decodeURIComponent('{lone_surrogate}'));",
        ]
        self.assertEqual(
            [_folded(program) for program in programs],
            [
                F"console.log('{_ASTRAL}');",
                F"console.log('{_ASTRAL}A');",
                F"console.log('{lone_surrogate}');",
            ],
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAnAllocationThatCannotBeBuiltAnswersNothing(TestBase):
    """
    A question the shape of an allocation answers — what `typeof` calls it, whether it is truthy —
    is answered without the allocation, which is then discarded along with every element in it. It
    may be discarded only if building it does nothing that can be observed, and resolving a name is
    such a thing: a reference no binding resolves throws a `ReferenceError`. The gate that asks
    whether an allocation is free of effects answers yes all the same, so the answer is printed
    where the file threw. Counting an array literal asked that same gate and no longer does; those
    reads are law in `test.lib.scripts.js.deobfuscation.test_array_length_reads`.
    """

    @unittest.expectedFailure
    def test_a_literal_holding_a_name_that_does_not_resolve_still_throws(self):
        """
        Node refuses both of these programs with a `ReferenceError` and prints nothing. Each
        deobfuscation prints instead: `object` for the `typeof`, and `1` for the branch a truthy
        object picks.
        """
        refused = ('', 'ReferenceError')
        sources = [
            'console.log(typeof [zzz]);',
            'console.log({p: zzz} ? 1 : 2);',
        ]
        self.assertEqual(
            [_before_and_after(source) for source in sources],
            [(refused, refused)] * len(sources),
        )


class TestALiteralNoElementOfWhichRunsIsCounted(TestBase):
    """
    An array literal's `length` is the number of positions it was written with, and reading it
    discards the array, so the count may replace the read whenever evaluating every element does
    nothing that can be observed. What the fold asks instead is whether every element is written as
    a literal or is an elision, which is narrower: the four reads of
    `test.lib.scripts.js.deobfuscation.test_array_length_reads.A_COUNT_THE_FOLD_DOES_NOT_REACH` hold
    a global value name, a function expression, an object and an array literal, and a getter that is
    defined rather than called, and not one of them runs while the array is built.
    """

    @unittest.expectedFailure
    def test_a_literal_whose_elements_are_not_written_as_literals_is_counted(self):
        """
        Node answers those four reads with `3`, `2`, `1`, and `2`, which the law in that module pins
        against the engine; each is the number of commas the literal is written with.
        """
        counts = A_COUNT_THE_FOLD_DOES_NOT_REACH
        self.assertEqual(
            [_folded(F'console.log({read});') for read in counts],
            [F'console.log({count});' for count in counts.values()],
        )


def _evaluated_in_a_body(receiver: str, read: str, installs: str = '') -> str:
    """
    A script that runs *installs*, then prints what *read* answers for a local holding *receiver*.

    The local inside a function is what puts the read where the tool answers it at all. Written at
    the top of the file the same read is left standing, so a program that only writes it there
    reports nothing about how it would have been answered.
    """
    body = F'function f() {{ var v = {receiver}; return {read}; }}\nconsole.log(f());\n'
    return F'{installs}\n{body}' if installs else body


def _an_accessor_at(prototype: str, key: str) -> str:
    """
    A statement installing a getter at *key* on *prototype* that answers `'G'`. A read the prototype
    chain decides is not merely a read of some other value: where the chain holds an accessor, the
    read runs the program's own code, and answering it off the receiver drops that code unrun.
    """
    return (
        F"Object.defineProperty({prototype}, '{key}', "
        F"{{get: function () {{ return 'G'; }}}});"
    )


def _each_program_still_prints(
    programs: dict[str, str],
) -> dict[str, tuple[tuple[str, str | None], tuple[str, str | None]]]:
    """
    The pair `_before_and_after` has to give for each program in *programs*: the text the program
    prints, printed by the deobfuscation too, with neither of the two throwing.
    """
    return {
        source: ((prints, None), (prints, None))
        for source, prints in programs.items()
    }


#: A program that installs a property at an index no receiver of the length written holds, and reads
#: that index, mapped to what Node prints for it. A string and an array own the slots `0` through
#: their length less one and nothing above, so every one of these reads is the chain's to answer.
A_READ_OF_AN_INDEX_THE_RECEIVER_DOES_NOT_HOLD = {
    _evaluated_in_a_body("'abc'", 'v[5]', "String.prototype[5] = 'X';"): 'X\n',
    _evaluated_in_a_body("''", 'v[0]', "String.prototype[0] = 'X';"): 'X\n',
    _evaluated_in_a_body("atob('YWJj')", 'v[5]', "String.prototype[5] = 'X';"): 'X\n',
    _evaluated_in_a_body('[1, 2]', 'v[5]', "Array.prototype[5] = 'X';"): 'X\n',
    _evaluated_in_a_body('[1, 2]', 'v[5]', "Object.prototype[5] = 'X';"): 'X\n',
    _evaluated_in_a_body('[1, 2]', 'v[5]', _an_accessor_at('Array.prototype', '5')): 'G\n',
    _evaluated_in_a_body("'abc'", 'v[5]', _an_accessor_at('String.prototype', '5')): 'G\n',
}


#: A program that installs a property at an index no array of the length written holds, and asks
#: whether that index is `in` the array, mapped to what Node prints for it.
A_MEMBERSHIP_TEST_OVER_AN_INDEX_THE_ARRAY_DOES_NOT_HOLD = {
    _evaluated_in_a_body('[1, 2]', '5 in v', "Array.prototype[5] = 'X';"): 'true\n',
    _evaluated_in_a_body('[1, 2]', "'5' in v", "Array.prototype[5] = 'X';"): 'true\n',
    _evaluated_in_a_body('[1, 2]', '5 in v', "Object.prototype[5] = 'X';"): 'true\n',
    _evaluated_in_a_body(
        '[1, 2]',
        '5 in v',
        "Object.defineProperty(Array.prototype, '5', {value: 'Q'});",
    ): 'true\n',
}


#: A program that writes one name onto a prototype in the receiver's chain and then asks whether a
#: different name is `in` the receiver, mapped to what Node prints for it. Nothing here installs the
#: name being asked about, so each answer is the one an engine nothing has touched would give.
A_MEMBERSHIP_TEST_FOR_A_KEY_NOBODY_INSTALLED = {
    _evaluated_in_a_body('{a: 1}', "'zz' in v", "Object.prototype.qq = 'X';"): 'false\n',
    _evaluated_in_a_body('[1, 2]', "'zz' in v", "Object.prototype.qq = 'X';"): 'false\n',
    _evaluated_in_a_body('[1, 2]', "'zz' in v", "Array.prototype.qq = 'X';"): 'false\n',
    _evaluated_in_a_body(
        '{a: 1}',
        "'zz' in v",
        _an_accessor_at('Object.prototype', 'qq'),
    ): 'false\n',
}


#: A program that installs a property at the index an elision left empty, and reads it, mapped to
#: what Node prints for it. Each receiver is written with a length the elision counts towards and
#: with no element at that index, so the read finds nothing on the array and walks the chain.
A_READ_OF_A_SLOT_AN_ELISION_LEFT_EMPTY = {
    _evaluated_in_a_body('[1, , 3]', 'v[1]', "Array.prototype[1] = 'X';"): 'X\n',
    _evaluated_in_a_body('[, 2]', 'v[0]', "Array.prototype[0] = 'X';"): 'X\n',
    _evaluated_in_a_body('[1, ,]', 'v[1]', "Array.prototype[1] = 'X';"): 'X\n',
    _evaluated_in_a_body('[1, , 3]', 'v[1]', "Object.prototype[1] = 'X';"): 'X\n',
    _evaluated_in_a_body(
        '[0, 1, 2, 3, 4, , 6]',
        'v[5]',
        _an_accessor_at('Array.prototype', '5'),
    ): 'G\n',
}


#: An array literal written with an elision, and a membership test over the index it left empty,
#: mapped to what Node prints for it in an engine nothing has touched. No prototype is written by
#: any of these: the slot is simply not in the array, which is the whole of what an elision is.
A_MEMBERSHIP_TEST_OVER_A_SLOT_AN_ELISION_LEFT_EMPTY = {
    _evaluated_in_a_body('[1, , 3]', '1 in v'): 'false\n',
    _evaluated_in_a_body('[, 2]', '0 in v'): 'false\n',
    _evaluated_in_a_body('[1, ,]', '1 in v'): 'false\n',
}


#: A program that installs a property on `Object.prototype` and reads it off a variable holding an
#: object literal that does not write it, mapped to what Node prints for it. `Object.prototype`
#: roots the chain of every object, so a key the literal lacks is always the chain's to answer.
A_READ_OF_A_KEY_AN_OBJECT_LITERAL_LACKS = {
    _evaluated_in_a_body('{a: 1}', 'v.zz', "Object.prototype.zz = 'X';"): 'X\n',
    _evaluated_in_a_body('{a: 1}', "v['zz']", "Object.prototype.zz = 'X';"): 'X\n',
    _evaluated_in_a_body('{a: 1}', 'v.length', 'Object.prototype.length = 9;'): '9\n',
    _evaluated_in_a_body('{a: 1}', 'v.zz', _an_accessor_at('Object.prototype', 'zz')): 'G\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAnIndexTheReceiverDoesNotHoldIsTheChainsToAnswer(TestBase):
    """
    A string and an array own the slots `0` through their length less one and no others, so an index
    above that is a name the receiver does not carry, no different in that from `zz`: the prototype
    chain answers it, and a program is free to put something there before the read runs. Those two
    reads are told apart all the same. A read of a name the receiver lacks is left to the engine,
    while a read of an index it lacks is answered `undefined` off the receiver's length alone, with
    nothing asked about what the program did to `String.prototype`, `Array.prototype`, or the
    `Object.prototype` below them.

    The `in` operator asks the same question of the same receiver and is answered the same way, by
    comparing the index against the length, so it reports an index the chain holds to be absent.
    """

    @unittest.expectedFailure
    def test_a_read_of_an_index_past_the_end_finds_what_the_chain_holds(self):
        """
        Node prints `X` for the five programs of
        `A_READ_OF_AN_INDEX_THE_RECEIVER_DOES_NOT_HOLD` that store a value on a prototype and `G`
        for the two that install an accessor there, the receiver being a string literal, the empty
        string, a string a call produced, or an array literal, and the prototype written being the
        one that owns the receiver's methods or the `Object.prototype` that roots every chain. Each
        deobfuscation prints `undefined` instead, and the two accessors never run at all.
        """
        rows = A_READ_OF_AN_INDEX_THE_RECEIVER_DOES_NOT_HOLD
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            _each_program_still_prints(rows),
        )

    @unittest.expectedFailure
    def test_membership_of_an_index_past_the_end_finds_what_the_chain_holds(self):
        """
        Node prints `true` for all four programs of
        `A_MEMBERSHIP_TEST_OVER_AN_INDEX_THE_ARRAY_DOES_NOT_HOLD`: `in` asks whether a property is
        reachable at all, which the whole chain answers and not the receiver's own slots. Each
        deobfuscation prints `false`, so the read and the membership test agree with each other and
        both disagree with the engine.
        """
        rows = A_MEMBERSHIP_TEST_OVER_AN_INDEX_THE_ARRAY_DOES_NOT_HOLD
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            _each_program_still_prints(rows),
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAnUnansweredMembershipTestIsNotAYes(TestBase):
    """
    `in` reports whether a property is reachable on the receiver or anywhere up its prototype chain.
    Once a program has written to that chain, what it put there is not a thing this tool can
    enumerate, so no name can be shown to be missing any more. A read in that position declines and
    is left for the engine, which is the right answer to a question one cannot answer.

    The membership test instead takes its own failure to prove the key absent for a proof that the
    key is there, and so answers `true` for every name at once — including all the names the program
    never installed, which are exactly the names the engine answers `false` for. A single write to
    one prototype in a receiver's chain, under any name at all, is enough: from then on every
    membership test over a name that receiver does not own is answered `true`.
    """

    @unittest.expectedFailure
    def test_a_key_nobody_installed_is_in_nothing(self):
        """
        Node prints `false` for all four programs of
        `A_MEMBERSHIP_TEST_FOR_A_KEY_NOBODY_INSTALLED`: each writes the one name `qq` onto a
        prototype the receiver inherits from and then asks about `zz`, which nothing in the program
        ever defines. Each deobfuscation prints `true`, and the same file with its one write removed
        prints `false` on both sides, so it is the write and not the question that moved the answer.
        """
        rows = A_MEMBERSHIP_TEST_FOR_A_KEY_NOBODY_INSTALLED
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            _each_program_still_prints(rows),
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAnElisionWritesNoSlot(TestBase):
    """
    An elision writes no element: `[1, , 3]` is three long and holds the slots `0` and `2` only,
    which is the whole of what tells it apart from `[1, undefined, 3]`, whose middle slot is present
    and holds `undefined`. The two are held alike here, as three elements the middle of which is
    `undefined`, and every question that parts them is then answered for the wrong one.

    Two such questions are asked below. Reading the empty slot is the prototype chain's to answer,
    since the array carries nothing there, and it is answered `undefined` off the element instead.
    Asking whether that index is `in` the array is answered from the length, so the array reports a
    slot it does not have — and this one is wrong in an engine no program has touched at all.
    """

    @unittest.expectedFailure
    def test_a_read_of_a_slot_an_elision_left_empty_finds_what_the_chain_holds(self):
        """
        Node prints `X` for the four programs of `A_READ_OF_A_SLOT_AN_ELISION_LEFT_EMPTY` that store
        a value on a prototype and `G` for the one that installs an accessor there, the elision
        standing first, in the middle, or before the closing bracket. Each deobfuscation prints
        `undefined`, which is the answer the same read has when the element is written `undefined`.
        """
        rows = A_READ_OF_A_SLOT_AN_ELISION_LEFT_EMPTY
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            _each_program_still_prints(rows),
        )

    @unittest.expectedFailure
    def test_a_slot_an_elision_left_empty_is_in_no_array(self):
        """
        Node prints `false` for all three programs of
        `A_MEMBERSHIP_TEST_OVER_A_SLOT_AN_ELISION_LEFT_EMPTY`, none of which touches a prototype:
        `1 in [1, , 3]` is `false` and `1 in [1, undefined, 3]` is `true`, and that is the one
        expression a program has for telling a hole from an element. Each deobfuscation prints
        `true`, which is the answer to the other array.
        """
        rows = A_MEMBERSHIP_TEST_OVER_A_SLOT_AN_ELISION_LEFT_EMPTY
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            _each_program_still_prints(rows),
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAKeyAnObjectLiteralLacksIsTheChainsToAnswer(TestBase):
    """
    `Object.prototype` roots the chain of every object, so a key an object literal does not write is
    never the literal's to answer: whatever the program installed there is what the read finds. A
    variable holding such a literal is read as though the literal were the whole of it, and each key
    it lacks is answered `undefined` unless the name is one of the handful an untouched
    `Object.prototype` carries. Which names this program put on `Object.prototype` is not asked, so
    a value stored under any other name is lost and an accessor stored there never runs.
    """

    @unittest.expectedFailure
    def test_a_read_of_a_key_the_literal_lacks_finds_what_the_chain_holds(self):
        """
        Node prints `X`, `X`, `9`, and `G` for the four programs of
        `A_READ_OF_A_KEY_AN_OBJECT_LITERAL_LACKS`, which read the installed key by name, by a string
        key, under the name `length` that no plain object owns, and through an accessor. Each
        deobfuscation prints `undefined`.
        """
        rows = A_READ_OF_A_KEY_AN_OBJECT_LITERAL_LACKS
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            _each_program_still_prints(rows),
        )


_REPORTS_THE_MODE_IT_STANDS_IN = '(function () { return this; })() === undefined'
"""
An expression that is `true` where it stands in strict code and `false` where it stands in sloppy
code. A plain call passes no receiver, so `this` in the callee is `undefined` under strict and the
global object under sloppy, and a function written inside a body runs in the mode that body runs in.
"""


def _a_strict_body_holding(statements: str, installs: str = '') -> str:
    """
    A program that runs *installs*, then prints whether the body of `f` runs strict, with
    *statements* standing between the directive that opens that body and the report.
    """
    body = (
        F"function f(a) {{ 'use strict'; {statements}"
        F' return {_REPORTS_THE_MODE_IT_STANDS_IN}; }}\n'
        'console.log(f(1));\n'
    )
    return F'{installs}\n{body}' if installs else body


def _a_strict_script_holding(statements: str) -> str:
    """
    A program printing whether the script runs strict, with *statements* standing between the
    directive that opens the file and the report.
    """
    return F"'use strict';\n{statements}\nconsole.log({_REPORTS_THE_MODE_IT_STANDS_IN});\n"


def _an_accessor_returning_a_strict_function(body: str, run: str) -> str:
    """
    A program building an accessor with an immediately invoked function that holds one local, whose
    returned function opens with the directive and closes with *body*, and that then runs *run*.
    Inlining the accessor is what writes the local of the outer function into the body the directive
    opens.
    """
    return (
        'var acc = (function () {\n'
        "  var t = ['a', 'b'];\n"
        F"  return function (i) {{ 'use strict'; {body} }};\n"
        '})();\n'
        F'{run}\n'
    )


#: An accessor whose returned function opens with a directive, mapped to what Node prints for it.
#: The first reports the mode that function runs in and the second assigns to a name nothing
#: declares, which strict code refuses and sloppy code answers with a new global.
AN_ACCESSOR_WHOSE_RETURNED_BODY_OPENS_WITH_A_DIRECTIVE = {
    _an_accessor_returning_a_strict_function(
        F"return t[i] + ({_REPORTS_THE_MODE_IT_STANDS_IN} ? 'S' : 'L');",
        'console.log(acc(1));',
    ): 'bS\n',
    _an_accessor_returning_a_strict_function(
        'undeclared = i; return t[i] + undeclared;',
        'try { console.log(acc(1)); } catch (e) { console.log(e.constructor.name); }',
    ): 'ReferenceError\n',
}


#: A body that opens with a directive and holds a binding nothing reads back, mapped to what Node
#: prints for it. Two of them are function bodies and two are whole scripts, and the binding is
#: written two ways: as a variable a single assignment stores into, and as a namespace object whose
#: one property is read straight back.
A_BODY_WHOSE_DIRECTIVE_STANDS_BESIDE_A_BINDING_NOTHING_READS = {
    _a_strict_body_holding('q = a;', 'var q;'): 'true\n',
    _a_strict_script_holding('var q;\nq = 1;'): 'true\n',
    _a_strict_body_holding('var NS = {}; NS.p = 1; console.log(NS.p);'): '1\ntrue\n',
    _a_strict_script_holding('var NS = {};\nNS.p = 1;\nconsole.log(NS.p);'): '1\ntrue\n',
}


#: A strict body assigning to a name no binding declares, mapped to the behavior Node gives it: the
#: pair of what it prints and what it throws. The write is the same in both and only what the body
#: does about the throw differs, one handling it and one letting it end the program.
A_STRICT_BODY_ASSIGNING_TO_AN_UNDECLARED_NAME = {
    (
        "function f() { 'use strict'; try { und = 1; return 'ok'; }"
        " catch (e) { return 'threw'; } }\n"
        'console.log(f());\n'
    ): ('threw\n', None),
    "(function () { 'use strict'; und = 1; console.log(1); })();\n": ('', 'ReferenceError'),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestMovingAStatementDoesNotChangeWhichStatementsAreDirectives(TestBase):
    """
    Whether a statement is a directive is decided by where it stands. A string literal written
    plainly in the run of statements a script or a function body opens with is one; the same
    statement a single position lower is an expression that computes a string and discards it. A
    pass that lifts the contents of a block into the body around it, one that drops a statement it
    found dead, and one that writes a declaration into a body each change which statements a body
    opens with, and none of them asks what the change did to that run.

    The mode of a whole script or body follows, in both directions: below a statement that is
    dropped, a string nobody wrote as a directive becomes one and turns sloppy code strict, and
    below a declaration that is written in, a directive that was written stops being one and turns
    strict code sloppy. Neither is reported anywhere, and the file that comes back is a program that
    runs by different rules than the one that was handed over. What a fold does to the same run is
    pinned in `TestAFoldDoesNotWriteADirectiveWhereNoneWasWritten`; these move no statement's text
    at all.
    """

    @unittest.expectedFailure
    def test_lifting_a_block_into_the_body_around_it_writes_no_directive(self):
        """
        Node prints `false` for both of these. A block is not a prologue and a statement inside one
        is never a directive, so the script and the probe's body are sloppy code however certainly
        the branch around that block is taken. The branch is replaced by the statements it holds,
        which is a rewrite of the block and nothing more, and it leaves that string opening the
        script in the first and the body in the second: Node prints `true` for both of the files
        that come back.
        """
        sloppy = ('false\n', None)
        sources = [
            a_script_opening_with("if (1) { 'use strict'; }"),
            a_function_body_opening_with("if (1) { 'use strict'; }"),
        ]
        self.assertEqual(
            [_before_and_after_as_a_script(source) for source in sources],
            [(sloppy, sloppy)] * len(sources),
        )

    @unittest.expectedFailure
    def test_dropping_a_statement_writes_no_directive_below_it(self):
        """
        Node prints `false` for both of these. The prologue ends at the declaration, which is no
        string literal, so the `'use strict'` standing below it computes a string and discards it.
        Nothing reads `dead`, so the declaration is dropped, which moves every statement below it up
        one place: the string that stood second now opens the script in the first file and the
        probe's body in the second, and Node prints `true` for both of the files that come back. A
        statement removed from a list is one the statements below it move up past.
        """
        sloppy = ('false\n', None)
        sources = [
            a_script_opening_with("var dead = 1; 'use strict';"),
            a_function_body_opening_with("var dead = 1; 'use strict';"),
        ]
        self.assertEqual(
            [_before_and_after_as_a_script(source) for source in sources],
            [(sloppy, sloppy)] * len(sources),
        )

    @unittest.expectedFailure
    def test_writing_a_declaration_into_a_body_leaves_its_directive_first(self):
        """
        Node prints `bS` and `ReferenceError` for the two programs of
        `AN_ACCESSOR_WHOSE_RETURNED_BODY_OPENS_WITH_A_DIRECTIVE`. Each returns a function that opens
        with the directive, so that function is strict: it reports the strict mode in the first, and
        in the second its assignment to a name nothing declares throws, which the file catches and
        names. Inlining the accessor writes the local of the outer function into that body, above
        the directive, which is a position no directive survives. Node prints `bL` for the first
        file that comes back, a body reporting a mode it no longer has, and `b1` for the second,
        which creates the global the file threw over and returns a value where it stopped.
        """
        rows = AN_ACCESSOR_WHOSE_RETURNED_BODY_OPENS_WITH_A_DIRECTIVE
        self.assertEqual(
            {source: _before_and_after_as_a_script(source) for source in rows},
            _each_program_still_prints(rows),
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestADirectiveIsNotAStatementThatCanBeDiscarded(TestBase):
    """
    A directive is written in the shape of an expression statement and is not one. Evaluating the
    literal is the least of what it does: it states the mode the body it opens runs in, and that
    body keeps the mode for as long as the statement stands there. A statement whose value nothing
    reads may be discarded, and this one may not, so the shape it is written in is not enough to
    decide it by.

    It is discarded all the same, whenever a binding standing beside it in the same body is removed:
    the sweep that drops a variable nothing reads back also drops the statements of that body which
    only evaluate a literal, and a directive is one of those by its shape alone. What removed the
    variable does not matter — an assignment nothing reads and a namespace object flattened into
    bare names each take the directive with them — and neither does whether the body is a function
    or the file.
    """

    @unittest.expectedFailure
    def test_a_directive_survives_the_removal_of_a_binding_beside_it(self):
        """
        Node prints `true` for all four programs of
        `A_BODY_WHOSE_DIRECTIVE_STANDS_BESIDE_A_BINDING_NOTHING_READS`, the two that read a property
        back printing the property first: each body opens with the directive and is strict for it.
        Each deobfuscation prints `false`, having removed the binding and the directive together.
        The first program with its directive left out prints `false` on both sides, so it is the
        removal of the directive and not the removal of the binding that moved the answer.
        """
        rows = A_BODY_WHOSE_DIRECTIVE_STANDS_BESIDE_A_BINDING_NOTHING_READS
        self.assertEqual(
            {source: _before_and_after_as_a_script(source) for source in rows},
            _each_program_still_prints(rows),
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestTheEvaluatorRunsABodyInTheModeItDeclares(TestBase):
    """
    A body that opens with the directive runs strict, and an assignment to a name no binding
    declares throws a `ReferenceError` there rather than creating a global. Running that body is how
    the tool answers what a call returns, and the mode the body declares is not carried into the
    run: the write is answered by the sloppy rule, the call is answered with a value it never
    produced, and the file that comes back prints that value where the program it came from threw.

    This is not the directive being lost. It is still written where the file wrote it, and both the
    file handed over and the file handed back declare the same mode; only the answer computed
    between them was computed under the other one.
    """

    @unittest.expectedFailure
    def test_an_assignment_to_an_undeclared_name_throws_in_a_strict_body(self):
        """
        Node prints `threw` for the first program of
        `A_STRICT_BODY_ASSIGNING_TO_AN_UNDECLARED_NAME` and refuses the second with a
        `ReferenceError` having printed nothing: the write is the same in both, and the first body
        catches what it throws while the second lets it end the program. The first deobfuscation
        prints `ok`, the value of the branch the throw never let run, and the second prints `1`,
        having gone on past the statement the file stopped at. The first program with its directive
        left out prints `ok` on both sides.
        """
        rows = A_STRICT_BODY_ASSIGNING_TO_AN_UNDECLARED_NAME
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            {source: (answer, answer) for source, answer in rows.items()},
        )


