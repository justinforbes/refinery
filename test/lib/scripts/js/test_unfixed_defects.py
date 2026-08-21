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
    node_executable,
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
from test.lib.scripts.js.test_parameter_grammar import (
    A_BINDING_THE_KIND_OF_FUNCTION_RESERVES,
    A_FUNCTION_EXPRESSION_NAME_ONLY_THE_ENCLOSING_KIND_RESERVES,
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


class TestABindingNamedByAWordItsFunctionKindReservesIsNoProgram(TestBase):
    """
    A class whose name is a word the enclosing function kind reserves is read as though the name
    were absent, and what the printer then writes is not JavaScript:
    `function* g() { class yield {} }` comes back as `class {\n    {;\n  }`, and the `await`
    twin inside an `async` function comes back the same way. The two function declarations of the
    corpus are read with their names now, so their text comes back exactly as it went in.

    Node refuses each of the four inputs this entry is quantified over, so what those four cost is
    not a program: it is that `refinery.lib.scripts.is_well_formed` answers `True` for the tree,
    which is the domain every fidelity law is stated over — a caller told the tree is well formed
    compares text that is not a program against the file it came from, and a printer under no
    obligation is asked for one anyway.
    """

    @unittest.expectedFailure
    def test_a_binding_the_kind_of_function_reserves_is_not_a_well_formed_program(self):
        """
        Node refuses all four files of
        `test.lib.scripts.js.test_parameter_grammar.A_BINDING_THE_KIND_OF_FUNCTION_RESERVES`, which
        the law in that module pins against the engine along with the controls that part the kind of
        function from the mode: the same declarations are read inside a plain function, except
        `class yield`, which every strict region refuses for a reason of its own.
        """
        rows = A_BINDING_THE_KIND_OF_FUNCTION_RESERVES
        self.assertEqual(
            {source: _well_formed(source) for source in rows},
            {source: False for source in rows},
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAnExpressionNamedByAWordOnlyTheEnclosingKindReservesIsAProgram(TestBase):
    """
    A function expression takes its name from its own kind and not from the kind of the function
    around it, so a plain expression named `yield` standing inside a generator is a program, and so
    is one named `await` inside an async function. Node prints `function` for

        function* g() { var f = function yield() { return 1; }; console.log(typeof f); } g().next();

    and for the `await` twin of it, and reads every file of the corpus this entry is quantified
    over. A directive is no defence for the `await` half, which no strict body reserves.

    The reservation the kind around the expression used to reach the name anyway, leaving the
    expression without one, so that what came back opened `var f = function(() {` — a text Node
    refuses. A declaration's name and a class expression's name are read under whatever encloses
    them instead, which is what makes the four files of
    `TestABindingNamedByAWordItsFunctionKindReservesIsNoProgram` no programs to begin with.
    """

    def test_printing_one_of_them_gives_a_program_that_runs_the_same_way(self):
        rows = A_FUNCTION_EXPRESSION_NAME_ONLY_THE_ENCLOSING_KIND_RESERVES
        self.assertEqual(
            {source: behavior(_printed(source)) for source in rows},
            {source: ('', None) for source in rows},
        )

    def test_the_deobfuscation_of_one_that_prints_keeps_what_it_prints(self):
        """
        A file that asks for the type of the expression keeps it alive through every pass, so what
        the tool writes for this one is what a caller of `refinery.js` is handed.
        """
        source = (
            'function* g() { var f = function yield() { return 1; };'
            ' console.log(typeof f); } g().next();'
        )
        self.assertEqual(_before_and_after(source), (('function\n', None), ('function\n', None)))


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


def _returned_from_a_body(body: str) -> str:
    """
    A script whose one function runs *body* and prints what it returned.

    A question asked inside a function is what puts it where the tool answers it at all, for the
    reason `_evaluated_in_a_body` gives. The body is written out whole here because the receivers
    below are built by statements rather than by one literal.
    """
    return F'function f() {{ {body} }}\nconsole.log(f());\n'


#: A `for...in` walk over an array holding a position no element was written in, mapped to what Node
#: prints for it. The walk reports the array's own keys, and such a position is not one of them
#: however it came about: written as an elision, passed over by a write beyond the end, or passed
#: over by moving `length` up.
A_FOR_IN_WALK_OVER_AN_ARRAY_HOLDING_A_HOLE = {
    _returned_from_a_body(
        "var v = [1, , 3]; var r = ''; for (var k in v) r += k; return r;"
    ): '02\n',
    _returned_from_a_body(
        "var v = [1]; v[2] = 3; var r = ''; for (var k in v) r += k; return r;"
    ): '02\n',
    _returned_from_a_body(
        "var v = [1]; v.length = 3; var r = ''; for (var k in v) r += k; return r;"
    ): '0\n',
}


#: A membership test over a position an array grew past without writing anything in it, mapped to
#: what Node prints for it. Growth moves how far the array reaches and never says that every
#: position below that was filled.
A_MEMBERSHIP_TEST_OVER_A_SLOT_GROWTH_PASSED_OVER = {
    _returned_from_a_body('var v = [1]; v.length = 3; return 1 in v;'): 'false\n',
    _returned_from_a_body('var v = [1]; v[2] = 3; return 1 in v;'): 'false\n',
}


#: A search of a grown array for the value a position it passed over would hold if it held one,
#: mapped to what Node prints for it. `indexOf` skips a position the array does not hold, so it
#: reports `-1` where the same search over `[1, undefined, 3]` reports `1`.
A_SEARCH_FOR_UNDEFINED_IN_A_SLOT_GROWTH_PASSED_OVER = {
    _returned_from_a_body('var v = [1]; v.length = 3; return v.indexOf(undefined);'): '-1\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAWalkOfAnArrayVisitsOnlyThePositionsItHolds(TestBase):
    """
    `for...in` visits an object's own keys, and an array's own keys are the positions it holds. A
    position nothing was ever written in is not one of them, so Node prints `02`, `02`, and `0` for
    the three programs of `A_FOR_IN_WALK_OVER_AN_ARRAY_HOLDING_A_HOLE`.

    Each walk is taken from the length instead and prints `012`, which is the walk of an array whose
    every position was written. The walk then hands its body an index the array has nothing at, and
    a body reading `v[k]` for each `k` it is given reads a slot the prototype chain owns.

    A walk of an array a `delete` left a hole in is answered correctly and is stated as law in
    `test.lib.scripts.js.deobfuscation.test_own_property_order`. The two holes are one hole to an
    engine, and this entry retires when one representation answers both.
    """

    @unittest.expectedFailure
    def test_a_position_no_element_was_written_in_is_visited_by_no_walk(self):
        rows = A_FOR_IN_WALK_OVER_AN_ARRAY_HOLDING_A_HOLE
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            _each_program_still_prints(rows),
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestGrowingAnArrayWritesNoElement(TestBase):
    """
    Moving `length` up and writing beyond the end both make an array reach further while leaving
    every position passed over empty, which is a hole exactly as an elision is. Node prints `false`
    for both programs of `A_MEMBERSHIP_TEST_OVER_A_SLOT_GROWTH_PASSED_OVER` and `-1` for the one of
    `A_SEARCH_FOR_UNDEFINED_IN_A_SLOT_GROWTH_PASSED_OVER`.

    Growth is recorded as elements instead, so the passed-over position answers as though it held
    `undefined`: the membership tests print `true` and the search prints `1`. The same two questions
    over a hole a `delete` made are answered correctly, which is what says it is growth rather than
    the hole that has no representation here.
    """

    @unittest.expectedFailure
    def test_a_position_growth_passed_over_is_in_no_array(self):
        rows = A_MEMBERSHIP_TEST_OVER_A_SLOT_GROWTH_PASSED_OVER
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            _each_program_still_prints(rows),
        )

    @unittest.expectedFailure
    def test_a_search_finds_no_undefined_in_a_position_growth_passed_over(self):
        rows = A_SEARCH_FOR_UNDEFINED_IN_A_SLOT_GROWTH_PASSED_OVER
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


#: A strict region assigning to one of the global names that is not writable, mapped to the behavior
#: Node gives it: the pair of what it prints and what it throws. The write is refused in every row,
#: so the statement does exactly one thing and that thing is throw. What varies is where the mode
#: comes from, how the assignment is spelled, and whether anything catches what it throws.
A_STRICT_REGION_ASSIGNING_TO_A_NON_WRITABLE_GLOBAL = {
    "'use strict';\nNaN = 1;\nconsole.log(1);\n": ('', 'TypeError'),
    "'use strict';\nundefined = 1;\nconsole.log(1);\n": ('', 'TypeError'),
    "'use strict';\nInfinity = 1;\nconsole.log(1);\n": ('', 'TypeError'),
    "'use strict';\n(NaN) = 1;\nconsole.log(1);\n": ('', 'TypeError'),
    "'use strict';\n[NaN] = [1];\nconsole.log(1);\n": ('', 'TypeError'),
    "'use strict';\n({p: undefined} = {p: 1});\nconsole.log(1);\n": ('', 'TypeError'),
    "(function () { 'use strict'; NaN = 1; console.log(1); })();\n": ('', 'TypeError'),
    "var out = 'L';\nclass C { static { NaN = 1; } }\nconsole.log(out);\n": ('', 'TypeError'),
    "'use strict';\ntry { NaN = 1; console.log('L'); }"
    " catch (e) { console.log(e.constructor.name); }\n": ('TypeError\n', None),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAWriteTheModeRefusesIsNotAWriteThatDoesNothing(TestBase):
    """
    `undefined`, `NaN` and `Infinity` name properties of the global object that no program may
    replace. An assignment to one of them is discarded where the assignment is sloppy code and is a
    `TypeError` where it is strict code, so in a strict region the statement has exactly one effect
    and throwing is it. It is removed in both, and everything the throw stood in front of then runs.

    Which region is strict is not what is misread: the mode is taken from wherever the file puts it
    and the removal follows it everywhere. A directive at the head of the script, one at the head of
    the function body holding the write, and a class static block, which is strict with no directive
    anywhere in the file, are all removed alike.

    Neither is this the removal of a statement that does nothing. The same write to `Object`, `Math`
    or `globalThis` — writable, every one of them — is removed from the same strict position, and
    Node prints `1` for those programs before and after; so is the same write to `NaN` behind a
    `var NaN` that binds the name locally; and so is every one of these writes in a file with no
    directive in it at all. `delete Object.prototype` in that position is left standing, with both
    sides refused by a `TypeError`.
    """

    @unittest.expectedFailure
    def test_an_assignment_to_a_non_writable_global_throws_in_a_strict_region(self):
        """
        Node refuses the first eight programs of
        `A_STRICT_REGION_ASSIGNING_TO_A_NON_WRITABLE_GLOBAL` with a `TypeError` having printed
        nothing, and prints `TypeError` for the last, which catches what the write throws. Every
        deobfuscation goes on past the statement its program stopped at, printing `1` for the first
        seven and `L` for the last two.

        The extent stops at the plain write. `NaN += 1`, `NaN++`, `NaN ||= 1` and `var q = NaN = 1`
        in the same strict position are all left standing and refused on both sides, and so is
        `NaN = 1, 0`, which is the same write with something behind it in the same statement.
        """
        rows = A_STRICT_REGION_ASSIGNING_TO_A_NON_WRITABLE_GLOBAL
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            {source: (answer, answer) for source, answer in rows.items()},
        )


#: A program whose function reads its variables out of a stack hanging off an object rather than out
#: of the rest parameter it declares, mapped to what Node prints for it. The truncation is written on
#: the qualified name and the rest parameter is named nowhere but in the parameter list, so it is the
#: qualified stack that each of these is about. The elements reach the function through the object:
#: one program stores them after the declaration, one writes them into the object literal, one
#: reaches the stack through a chain of two names, one holds two of them, one calls the function
#: twice, and one has the body write an element back for the file to read after the call.
A_STACK_REACHED_THROUGH_A_QUALIFIED_NAME = {
    'var NS = { F: {} };\n'
    'function f(...r) { NS.F.stk.length = 1; return NS.F.stk[0] * 2; }\n'
    'NS.F.stk = [5];\n'
    'console.log(f());\n': '10\n',

    'var NS = { F: { stk: [5] } };\n'
    'function f(...r) { NS.F.stk.length = 1; return NS.F.stk[0] * 2; }\n'
    'console.log(f(3));\n': '10\n',

    'var NS = { stk: [5] };\n'
    'function f(...r) { NS.stk.length = 1; return NS.stk[0] * 2; }\n'
    'console.log(f(3));\n': '10\n',

    'var NS = { F: { stk: [5, 7] } };\n'
    'function f(...r) { NS.F.stk.length = 2; return NS.F.stk[0] + NS.F.stk[1]; }\n'
    'console.log(f());\n': '12\n',

    'var NS = { F: { stk: [5] } };\n'
    'function f(...r) { NS.F.stk.length = 1; return NS.F.stk[0] * 2; }\n'
    'console.log(f() + f());\n': '20\n',

    'var NS = { F: { stk: [5] } };\n'
    'function f(...r) { NS.F.stk.length = 1; NS.F.stk[0] = NS.F.stk[0] + 1; }\n'
    'f();\n'
    'console.log(NS.F.stk[0]);\n': '6\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAStackHangingOffAnObjectIsNotTheArgumentsOfACall(TestBase):
    """
    A function can pack its variables into an array reached through a qualified name instead of into
    the rest parameter it declares. That array belongs to the object, so what it holds is whatever
    the program put there and the call that runs the function need pass nothing at all.

    Unpacking such a function into one taking plain parameters rewrites the callee alone. No call
    site is given the elements the object held, so a read that named a slot of the stack now names a
    parameter nobody passes, and a write that filled one now fills a parameter the object never
    sees. Where the stack is the rest parameter itself the same rewrite is sound, since there the
    elements are exactly the arguments of the call; that is the stack
    `test.lib.scripts.js.deobfuscation.test_restunpack` states its laws over.
    """

    @unittest.expectedFailure
    def test_the_elements_the_object_holds_survive_the_unpacking(self):
        """
        Node prints `10`, `10`, `10`, `12`, `20`, and `6` for the six programs of
        `A_STACK_REACHED_THROUGH_A_QUALIFIED_NAME`. Their deobfuscations print `NaN`, `6`, `6`,
        `NaN`, `NaN`, and `5`: the two programs whose call passes an argument read that argument
        where an element of the object stood, the three that pass none read a parameter no call
        supplies, and the last leaves the object holding the element it started with. The same
        computation with the stack written as the rest parameter itself, `f(...s)` truncated at
        `s.length = 1` and called as `f(5)`, prints `10` on both sides.
        """
        rows = A_STACK_REACHED_THROUGH_A_QUALIFIED_NAME
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            _each_program_still_prints(rows),
        )


#: A program whose function stores into a stack reached through a qualified name under a key that is
#: no parameter's index, mapped to what Node prints for it. The stack starts empty and the call
#: passes nothing in all but one, where the one index key is a parameter the call is handed the very
#: value the object holds: no element ever has to cross the call boundary, so what any of these
#: reports is the name the rewrite writes and nothing about where a value came from. The keys are an
#: identifier, a string, a negative index, and a decimal that is not the canonical spelling of the
#: one it resembles; the chain is two names long or one; and the last two ask, in a file with no
#: directive in it, how many properties the run left on the global object.
A_QUALIFIED_STACK_KEY_THAT_NAMES_NO_PARAMETER = {
    "'use strict';\n"
    'var NS = { F: { stk: [] } };\n'
    'function f(...r) { NS.F.stk.length = 0; NS.F.stk.a = 3; return NS.F.stk.a; }\n'
    'console.log(f());\n': '3\n',

    "'use strict';\n"
    'var NS = { F: { stk: [] } };\n'
    "function f(...r) { NS.F.stk.length = 0; NS.F.stk['zz'] = 3; return NS.F.stk['zz']; }\n"
    'console.log(f());\n': '3\n',

    "'use strict';\n"
    'var NS = { F: { stk: [] } };\n'
    'function f(...r) { NS.F.stk.length = 0; NS.F.stk[-1] = 3; return NS.F.stk[-1]; }\n'
    'console.log(f());\n': '3\n',

    "'use strict';\n"
    'var NS = { F: { stk: [] } };\n'
    "function f(...r) { NS.F.stk.length = 0; NS.F.stk['01'] = 3; return NS.F.stk['01']; }\n"
    'console.log(f());\n': '3\n',

    "'use strict';\n"
    'var NS = { stk: [] };\n'
    'function f(...r) { NS.stk.length = 0; NS.stk.a = 3; return NS.stk.a; }\n'
    'console.log(f());\n': '3\n',

    "'use strict';\n"
    'var NS = { F: { stk: [] } };\n'
    'function f(...r) { NS.F.stk.length = 0; NS.F.stk.a = 3; NS.F.stk.b = 4;'
    ' return NS.F.stk.a * NS.F.stk.b; }\n'
    'console.log(f());\n': '12\n',

    "'use strict';\n"
    'var NS = { F: { stk: [4] } };\n'
    'function f(...r) { NS.F.stk.length = 1; NS.F.stk.a = 3; return NS.F.stk[0] + NS.F.stk.a; }\n'
    'console.log(f(4));\n': '7\n',

    'var NS = { F: { stk: [] } };\n'
    'function f(...r) { NS.F.stk.length = 0; NS.F.stk.a = 3; return NS.F.stk.a; }\n'
    'var before = Object.getOwnPropertyNames(globalThis).length;\n'
    'f();\n'
    'console.log(Object.getOwnPropertyNames(globalThis).length - before);\n': '0\n',

    'var NS = { stk: [] };\n'
    'function f(...r) { NS.stk.length = 0; NS.stk.a = 3; return NS.stk.a; }\n'
    'var before = Object.getOwnPropertyNames(globalThis).length;\n'
    'f();\n'
    'console.log(Object.getOwnPropertyNames(globalThis).length - before);\n': '0\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAnUnpackedStackDeclaresTheLocalsItMints(TestBase):
    """
    Unpacking a rest-array stack turns each key into an identifier, and a key that names no
    parameter turns into an identifier the program never had. Such a name is the rewrite's to
    introduce: a binding for it exists only if the rewrite writes one.

    `refinery.lib.scripts.js.deobfuscation.restunpack.JsRestArrayUnpacking` writes it where the
    stack is a plain local, and writes none where the stack is reached through a qualified name.
    Every key of a qualified stack that is not an index the parameter list covers therefore comes
    back as a bare assignment to a name nothing declares, which is an implicit global where the
    function is sloppy code and a `ReferenceError` where it is strict code.

    This is the second thing that branch gets wrong and it is not the first. What
    `TestAStackHangingOffAnObjectIsNotTheArgumentsOfACall` pins is where a value comes from, an
    element the object held being sought in an argument the call never passed; here every value
    stays inside the body that computes it and only the binding is missing. Declaring the locals
    would leave that entry exactly as it is, and supplying the elements would leave this one exactly
    as it is.
    """

    @unittest.expectedFailure
    def test_a_key_that_names_no_parameter_is_declared_where_it_is_written(self):
        """
        Node prints `3`, `3`, `3`, `3`, `3`, `12` and `7` for the seven strict programs of
        `A_QUALIFIED_STACK_KEY_THAT_NAMES_NO_PARAMETER`, and `0` for the two sloppy ones, which run
        leaving the global object with the properties it already had. Each strict deobfuscation
        throws a `ReferenceError` having printed nothing, the body coming back as `v0 = 3;` and a
        read of `v0` with `v0` declared nowhere in the file; each sloppy one prints `1`, that same
        write having put the name on the global object.

        The same computations with the stack written as the rest parameter itself — `f(...s)` with
        `s.length = 0` and `s.a = 3` — reach the branch that declares what it mints, and print `3`
        and `0` on both sides.
        """
        rows = A_QUALIFIED_STACK_KEY_THAT_NAMES_NO_PARAMETER
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            _each_program_still_prints(rows),
        )


#: A region that runs strict, holding an assignment to a name no declaration binds, mapped to the
#: behavior Node gives it. Assigning to an unresolvable reference is a `ReferenceError` in strict
#: code where sloppy code creates a property of the global object, so the statement's one effect is
#: the throw and nothing written behind it runs. The mode is arrived at three ways, none of which
#: the statement itself states.
A_STRICT_REGION_ASSIGNING_TO_NO_BINDING = {
    "function f(b) { 'use strict'; var q = b + 1; undeclared_a = 1; return q; }"
    ' console.log(f(2));': ('', 'ReferenceError'),
    "'use strict'; undeclared_b = 1; console.log(3);": ('', 'ReferenceError'),
    'var out = 3; class C { static { undeclared_c = 1; } } console.log(out);':
        ('', 'ReferenceError'),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAWriteOnlyStrictCodeRefusesIsNotADeadStore(TestBase):
    """
    A store whose value nothing reads is removable only where storing is all it does. Where the name
    resolves to no binding and the write stands in strict code the assignment throws instead, and
    the sweep reads it as a store and deletes the throw along with everything the program never
    reached. The same write in sloppy code really is dead, and
    `test_unused.TestAWriteSloppyCodeAnswersIsADeadStore` pins that it is still removed.

    Refusing to remove it is not by itself the fix, and a repair that stops there makes a commoner
    program worse. A namespace flattening rewrites `NS.p = 1` to a bare `p = 1` and emits `var p`
    beside it; where a fold then answers every read of `p`, the declaration is swept as unread.
    Keeping the assignment while its declaration goes leaves a write to a name nothing binds — the
    very throw this entry is about, in a program that had none. The store and its declaration have
    to be decided together.
    """

    @unittest.expectedFailure
    def test_an_assignment_to_no_binding_throws_where_the_region_is_strict(self):
        rows = A_STRICT_REGION_ASSIGNING_TO_NO_BINDING
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            {source: (answer, answer) for source, answer in rows.items()},
        )


#: A module holding an indirect eval whose payload binds the name `await` and hoists nothing. The
#: eval runs its text as a script, whatever kind of code called it, and a script may bind that name;
#: the module around the eval may not spell it at all. The payload binds it as a function-expression
#: parameter because a `var` or a function declaration would be refused already — for scoping, a
#: global-eval hoist landing on the global object where a module-level one would not — and this
#: ledger is about the refusal that is missing, not the one that happens to stand in front of it.
A_PAYLOAD_ONLY_A_SCRIPT_MAY_SPELL = (
    'export {};\n'
    "(0, eval)('console.log(function (await) { return await + process.argv.length; }(1));');"
)


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestANameTheModuleReservesStaysBehindTheEval(TestBase):
    """
    §13.1.1 refuses `await` as a binding name wherever the goal symbol is Module — everywhere in
    the file, async code or not, directive or none — while a script under `'use strict'` binds it
    freely: the word is module-reserved and not strict-reserved. Splicing a payload that binds it
    into the module that called the eval therefore has to be refused, and the gate that decides,
    `collect_strict_violations` seeded with the destination's strictness, has no way to say so:
    strictness is the only destination fact the collector is handed, and the fact that matters here
    is the destination's goal symbol. The fix is the missing channel — the collector takes the
    destination's module-ness beside its strictness and reports the binding under it, and the
    reflection gate seeds both from the tree the payload would land in.

    Node prints `3` for the module and refuses its deobfuscation whole with a `SyntaxError`, the
    payload having come back spliced into the module with `function (await)` in it.
    """

    @unittest.expectedFailure
    def test_a_module_keeps_printing_what_the_eval_printed(self):
        source = A_PAYLOAD_ONLY_A_SCRIPT_MAY_SPELL
        self.assertEqual(
            (
                behavior(source, module=True),
                behavior(deobfuscate_source(source, module=True), module=True),
            ),
            (('3\n', None), ('3\n', None)),
        )


def _spelled_with_an_escaped_identifier(source: str) -> str:
    """
    *source* with the placeholder `ESCAPED_A` replaced by the unicode escape denoting the identifier
    `a`, and `ESCAPED_Q` by the one denoting `q`.

    Both escapes are assembled from `chr(92)` rather than written out. An escape written into this
    file is one flattening away from being the character it denotes, and an entry that no longer
    contains the spelling it asks about asks nothing at all.
    """
    with_a = source.replace('ESCAPED_A', F'{chr(92)}u0061')
    return with_a.replace('ESCAPED_Q', F'{chr(92)}u0071')


#: A program naming a property by an identifier written with a unicode escape, mapped to what Node
#: prints for it. The escape is the identifier and not four characters that resemble it, so the
#: literal writes the key `a`, a plain `.a` reads what it wrote, a plain read finds a key an escaped
#: spelling wrote, and a membership test over `q` finds the key an escaped `q` wrote.
A_PROPERTY_KEY_WRITTEN_WITH_AN_ESCAPE = {
    _spelled_with_an_escaped_identifier(
        _returned_from_a_body("return Object.keys({ ESCAPED_A: 1, b: 2 }).join('|');")
    ): 'a|b\n',
    _spelled_with_an_escaped_identifier(
        _returned_from_a_body('return { ESCAPED_A: 7 }.a;')
    ): '7\n',
    _spelled_with_an_escaped_identifier(
        _returned_from_a_body('var o = { a: 1 }; return o.ESCAPED_A;')
    ): '1\n',
    _spelled_with_an_escaped_identifier(
        _returned_from_a_body("return 'q' in { ESCAPED_Q: 1 };")
    ): 'true\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAPropertyKeyWrittenWithAnEscapeIsTheNameThatEscapeDenotes(TestBase):
    """
    The name a property key carries is the code points its escapes resolve to and not the characters
    it was typed with, so a key written with an escape and a read written plainly are one name, and
    so is the reverse pair. Node prints `a|b`, `7`, `1`, and `true` for the four programs of
    `A_PROPERTY_KEY_WRITTEN_WITH_AN_ESCAPE`.

    `test.lib.scripts.js.analysis.test_differential` pins the same defect where it costs code: a
    declaration written with an escape never matches the plain read of it, so the declaration reads
    as unused and is dropped. In a property key it costs an answer instead, which is worse. Each
    program here folds to a constant, and every one of those constants is wrong: the key comes back
    spelled with the backslash the file used, the two reads come back `undefined`, and the
    membership test comes back `false` for a key the object owns.
    """

    @unittest.expectedFailure
    def test_a_key_written_with_an_escape_is_the_name_that_escape_denotes(self):
        rows = A_PROPERTY_KEY_WRITTEN_WITH_AN_ESCAPE
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            _each_program_still_prints(rows),
        )


#: A program whose one function reads a global-object alias Node does not put on the global object,
#: mapped to the behavior Node gives it. The read is everything the function does and the call is
#: everything the program does before it prints, so what the read does is all that decides a row.
A_READ_OF_AN_ALIAS_THE_RUNNING_HOST_LACKS = {
    'function f() { return window; }\nf();\nconsole.log(1);\n': ('', 'ReferenceError'),
    'function f() { return self; }\nf();\nconsole.log(1);\n': ('', 'ReferenceError'),
    'function f() { return top; }\nf();\nconsole.log(1);\n': ('', 'ReferenceError'),
    'function f() { return frames; }\nf();\nconsole.log(1);\n': ('', 'ReferenceError'),
}


#: The same program written with `global`, the alias Node defines and a browser does not. Running it
#: decides nothing — both sides print `1` under the only engine this file can ask — so the answer is
#: pinned as the text a correct implementation writes rather than as what an engine makes of it.
A_READ_OF_THE_ALIAS_ONLY_ANOTHER_HOST_LACKS = (
    'function f() { return global; }\nf();\nconsole.log(1);\n'
)


class TestABareGlobalObjectAliasIsNotCertainToResolve(TestBase):
    """
    `window`, `global`, `self`, `top` and `frames` are names a host may put on its global object,
    and no host puts all of them there. A bare read of one may therefore find nothing, and finding
    nothing is a `ReferenceError`: Node refuses `window`, `self`, `top` and `frames`, a browser
    refuses `global`. `SemanticModel.read_may_throw` answers `False` for every one of the five,
    which asserts that whoever runs the file defines the name — the assertion it refuses to make
    for any other name the program neither declares nor assigns.

    A function whose body only reads one is then a function with no effect, its discarded call is
    removed, and the declaration goes with it, so a program whose one failure was that read comes
    back as one that runs to the end and prints. `globalThis` is the spelling the language mandates
    rather than the host, which is why `GUARANTEED_GLOBALS` holds it, and it is not what this entry
    is about.

    Fixing this is not free, because the same host assumption is made a second time elsewhere:
    `EffectModel._base_is_safe` clears a property access whose base is one of these five names as
    one that cannot throw on a nullish base, which is the identical claim that whoever runs the
    file defines the name, made about a member read rather than about a bare one. An
    implementation that stops vouching for the five here has to answer for that clause in the same
    breath, or the analysis holds two contradictory answers to one question: a bare `window` that
    may throw, and a `window.x` whose base is certain to be there.
    """

    @unittest.skipIf(node_executable() is None, 'node.js is not available')
    @unittest.expectedFailure
    def test_a_read_of_an_alias_the_running_host_lacks_still_throws(self):
        """
        Node refuses each program of `A_READ_OF_AN_ALIAS_THE_RUNNING_HOST_LACKS` having printed
        nothing, with a `ReferenceError` reading `window is not defined` and the same for `self`,
        `top` and `frames`. Every deobfuscation prints `1`: `console.log(1);` is the whole of what
        comes back for each of them.
        """
        rows = A_READ_OF_AN_ALIAS_THE_RUNNING_HOST_LACKS
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            {source: (answer, answer) for source, answer in rows.items()},
        )

    @unittest.expectedFailure
    def test_a_read_of_the_alias_the_running_host_defines_is_kept_all_the_same(self):
        """
        Node prints `1` for `A_READ_OF_THE_ALIAS_ONLY_ANOTHER_HOST_LACKS` and prints `1` for its
        deobfuscation, so running the two decides nothing about this alias and the text is what
        carries the answer. The text pinned is the one this program takes today when the name is
        one the analysis does not vouch for: written with `zzz` in place of `global`, it comes back
        with its function and its call in place and only the layout changed.
        """
        self.assertEqual(
            _folded(A_READ_OF_THE_ALIAS_ONLY_ANOTHER_HOST_LACKS),
            'function f() {\n  return global;\n}\nf();\nconsole.log(1);',
        )


#: A program reading a `let` or `const` binding from a point its declaration has not run past,
#: mapped to the behavior Node gives it. The read is reached four ways: through the initializer of a
#: later declarator, through an assignment, through `typeof`, and with no function in the file.
A_READ_IN_THE_DEAD_ZONE_OF_A_LEXICAL_BINDING = {
    'function f() { let v = q; let q = 1; }\n'
    'f();\nconsole.log(1);\n': ('', 'ReferenceError'),

    'function f() { let v = 0; v = q; const q = 1; }\n'
    'f();\nconsole.log(1);\n': ('', 'ReferenceError'),

    'function f() { { let v = typeof q; let q = 1; } }\n'
    'f();\nconsole.log(1);\n': ('', 'ReferenceError'),

    '{ let v = q; let q = 1; }\nconsole.log(1);\n': ('', 'ReferenceError'),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAReadOfALexicalBindingBeforeItsDeclarationThrows(TestBase):
    """
    A `let` or `const` binding exists from the moment its block is entered and holds no value until
    its declaration runs. A read in between resolves to it and throws a `ReferenceError` all the
    same, which is the one way a name is bound and unreadable at once. The analysis stops at the
    resolution, so `SemanticModel.read_may_throw` answers `False` and the read is one that cannot
    fail: the store holding it is a store nothing reads, the function it leaves empty has no
    effect, and the discarded call goes. What comes back runs to the end and prints.

    `typeof` is no defence, which is where a dead zone parts company with a name that denotes no
    binding at all. Node prints `1` for

        function f() { let v = 0; v = typeof zzz; } f(); console.log(1);

    where nothing binds `zzz`, and refuses the same program with the read moved into a dead zone. A
    `var` has no dead zone either, being initialized to `undefined` when the body is entered, so
    Node prints `1` for

        function f() { { let v = q; var q = 1; } } f(); console.log(1);

    and for the same program with the declaration written in front of the read; both of those calls
    are discarded rightly.

    Resolving the read correctly is not the whole of the fix. The sweep that removes the store asks
    no question of the read either, so a store holding a free name is removed from these same
    positions and its program comes back running too; a dead zone read is the case where the name
    does resolve and the answer is still that the read may not happen.
    """

    @unittest.expectedFailure
    def test_a_read_before_the_declaration_runs_still_throws(self):
        """
        Node refuses each program of `A_READ_IN_THE_DEAD_ZONE_OF_A_LEXICAL_BINDING` having printed
        nothing, the `typeof` row included, with a `ReferenceError` reading

            Cannot access 'q' before initialization

        Every deobfuscation prints `1`: the three programs that call a function come back as
        `void 0;` in front of the print, and the one written as a block comes back as an empty
        block.
        """
        rows = A_READ_IN_THE_DEAD_ZONE_OF_A_LEXICAL_BINDING
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            {source: (answer, answer) for source, answer in rows.items()},
        )


#: A program whose one failure is a read of a name nothing binds, mapped to the behavior Node gives
#: it. Each read stands where nothing goes on to use the value around it — a store no later
#: statement reads, an assignment to a name no later statement reads, and a container built into
#: such a store — so the expression holding the read is discarded and the read goes with it.
A_DISCARDED_READ_OF_A_NAME_NOTHING_BINDS = {
    'function f() { let v = zzz; }\nf();\nconsole.log(1);\n': ('', 'ReferenceError'),
    'function f() { var v = zzz; }\nf();\nconsole.log(1);\n': ('', 'ReferenceError'),
    'y = a;\nconsole.log(1);\n': ('', 'ReferenceError'),
    'var o = { p: g };\nconsole.log(1);\n': ('', 'ReferenceError'),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAReadOfANameNothingBindsThrowsWhereverItStands(TestBase):
    """
    Reading a name no binding resolves throws a `ReferenceError`, and the gate that decides whether
    evaluating an expression may be dropped — `refinery.lib.scripts.js.analysis.effects` and its
    `side_effect_free` — answers that such a read does nothing. Wherever an expression is discarded
    the read inside it is therefore discarded too, and the program comes back running to the end.
    No dead zone is involved and no host is: the name is bound nowhere, under every engine, and
    Node refuses each of these programs having printed nothing.

    `TestAnAllocationThatCannotBeBuiltAnswersNothing` pins the same gate at the allocation, where
    the program does ask a question of the value — what `typeof [zzz]` calls it, which branch
    `{p: zzz} ? 1 : 2` picks — and the answer is given without building it. Here nothing is asked:
    the expression is thrown away whole, so the same gate is reached with no fold in front of it.
    `TestAReadOfALexicalBindingBeforeItsDeclarationThrows` is the case where the name does resolve
    and the read may still not happen; the last paragraph of that entry is about this defect, the
    other half of the fix it needs.

    The hook for the fix already exists and is already documented for it: `side_effect_free` takes
    a *read_effect* whose contract says the read it rejects may fire the `with` object's getter or
    throw, and `EffectModel.is_side_effect_free` supplies only the getter half of that contract,
    which is `SemanticModel.read_has_dynamic_effect`. Widening it to reject every read that
    `SemanticModel.read_may_throw` rejects was measured, and it does correct every row below. It
    also breaks a program that is right today:

        X = 5; y = X; console.log(1);

    Node prints `1` for it and so does the deobfuscation as it stands. Under the widened hook the
    program comes back as `X;` in front of the print, which throws: the store-removal pass drops
    the write that establishes `X` in the same round in which the preserved right-hand side of
    `y = X` keeps the read of it standing. Widening the hook by itself therefore buys the four rows
    below at the price of a fifth answer, and a fix has to settle what that pass does with a store
    whose value it may no longer drop.
    """

    @unittest.expectedFailure
    def test_a_read_of_a_name_nothing_binds_is_not_dropped_with_its_expression(self):
        """
        Node refuses each program of `A_DISCARDED_READ_OF_A_NAME_NOTHING_BINDS` having printed
        nothing, with a `ReferenceError` reading `zzz is not defined` and the same for `a` and for
        `g`. Every deobfuscation prints `1`: the two programs that call a function come back as
        `void 0;` in front of the print, and the two written at the top of the file come back as
        the print alone.
        """
        rows = A_DISCARDED_READ_OF_A_NAME_NOTHING_BINDS
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            {source: (answer, answer) for source, answer in rows.items()},
        )


#: A program whose discarded read of a name nothing binds stands inside a `try`, mapped to what Node
#: prints for it. The read throws, the `catch` clause runs, and the program goes on to the end, so
#: the whole of the defect is one line of output and neither the program nor its deobfuscation ends
#: in an error. The last two rows are the controls: a `throw` and a member read on `null` reach the
#: same `catch` from the same block and are kept.
A_DISCARDED_READ_A_TRY_CATCHES = {
    'try {\n  var x = missing;\n} catch (e) {\n  console.log(2);\n}\nconsole.log(3);\n': '2\n3\n',
    'try {\n  let v = missing;\n} catch (e) {\n  console.log(2);\n}\nconsole.log(3);\n': '2\n3\n',
    'try {\n  y = missing;\n} catch (e) {\n  console.log(2);\n}\nconsole.log(3);\n': '2\n3\n',
    'try {\n  var o = { p: missing };\n} catch (e) {\n  console.log(2);\n}\nconsole.log(3);\n':
        '2\n3\n',
    'try {\n  throw 1;\n} catch (e) {\n  console.log(2);\n}\nconsole.log(3);\n': '2\n3\n',
    'try {\n  null.p;\n} catch (e) {\n  console.log(2);\n}\nconsole.log(3);\n': '2\n3\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAReadOfANameNothingBindsInsideATryIsCaught(TestBase):
    """
    This is the same gate `TestAReadOfANameNothingBindsThrowsWhereverItStands` pins — a
    discarded expression holding a read of a name no binding resolves is dropped, read and all —
    asked in the one place where dropping it costs no error. It is a separate entry rather than a
    row of `A_DISCARDED_READ_OF_A_NAME_NOTHING_BINDS` because every row of that corpus is a
    program Node refuses and a deobfuscation that runs, and the law over it is stated in those
    terms; here both sides run to the end and print, and only the first line of output tells them
    apart. A ledger of behavior-preservation defects that only ever compares an error against
    no error would not have this defect in it at all, which is the reason to keep it stated on
    its own.

    A `catch` clause is what turns the throw into output: the read fails, the clause runs, and the
    program carries on. Emptying the `try` block takes the clause's run away with it, so the line
    the clause prints is gone and the line after the statement is all that is left.

    The last two rows are the control. A `throw` and a member read on `null` reach the same clause
    from the same block, are not reads of a name, and are kept, so an entry that started passing by
    refusing to touch a `try` at all would be reported as an unexpected success here.
    """

    @unittest.expectedFailure
    def test_a_read_that_throws_inside_a_try_still_reaches_the_catch(self):
        """
        Node prints `2` and then `3` for every program of `A_DISCARDED_READ_A_TRY_CATCHES`, the
        first line from the `catch` clause the failed read reaches and the second from the statement
        after the `try`. The four deobfuscations whose block holds a read come back with that block
        emptied — `try {} catch (e) { console.log(2); }` — and print `3` alone.
        """
        rows = A_DISCARDED_READ_A_TRY_CATCHES
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            _each_program_still_prints(rows),
        )


#: A program that calls a function whose body reads a name nothing binds and prints the result,
#: mapped to the behavior Node gives it. The read stands in a store no later statement reads, which
#: is what lets the body be emptied down to the value it returns. The last two rows are the
#: controls: the same read written where no store holds it, and the returned value itself.
A_CALL_WHOSE_BODY_READS_A_NAME_NOTHING_BINDS = {
    'function f() {\n  var x = missing;\n  return 7;\n}\nconsole.log(f());\n':
        ('', 'ReferenceError'),
    'function f() {\n  var x = { p: missing };\n  return 7;\n}\nconsole.log(f());\n':
        ('', 'ReferenceError'),
    'var f = () => {\n  var x = missing;\n  return 7;\n};\nconsole.log(f());\n':
        ('', 'ReferenceError'),
    'function f() {\n  missing;\n  return 7;\n}\nconsole.log(f());\n': ('', 'ReferenceError'),
    'function f() {\n  return missing;\n}\nconsole.log(f());\n': ('', 'ReferenceError'),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestACallWhoseBodyReadsANameNothingBindsIsNotAValue(TestBase):
    """
    A call that cannot return, because evaluating its body throws, is not a call that may be written
    as the value it would have returned. The result here is not merely a call that survives where it
    should have gone: the call is replaced by a literal, so the program that refused to run comes
    back printing a number.

    Two gates have to be passed for that, and widening the one
    `TestAReadOfANameNothingBindsThrowsWhereverItStands` names would only close the first. The store
    holding the read is dropped because the gate that entry is about answers that the read does
    nothing, which leaves a body that only returns. What then replaces the call is decided by
    `EffectSummary.is_literal_replaceable`, and that property does not consult
    `EffectSummary.throws` at all — deliberately, since a literal replacement is meant for an
    evaluator that ran the call to a value, and such an evaluator reproduces the throw by
    throwing. `throws` is set for every function below; it is the second gate, not the summary,
    that has to learn the difference between a value an evaluator computed and one the body was
    read to have.

    The last two rows are the control, and they pass today: the same read written as a statement of
    its own and the same read written as the returned expression are both kept, so an entry that
    began passing by refusing to fold any call at all would be an unexpected success here.
    """

    @unittest.expectedFailure
    def test_a_call_that_cannot_return_is_not_replaced_by_what_it_would_have_returned(self):
        """
        Node refuses every program of `A_CALL_WHOSE_BODY_READS_A_NAME_NOTHING_BINDS` having printed
        nothing, with a `ReferenceError` reading

            missing is not defined

        The three deobfuscations whose read stands in a store come back as `console.log(7);` and
        print `7`.
        """
        rows = A_CALL_WHOSE_BODY_READS_A_NAME_NOTHING_BINDS
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            {source: (answer, answer) for source, answer in rows.items()},
        )


#: A program in which a read of a name nothing binds is an operand of a sequence expression whose
#: value is decided by a later operand, mapped to the behavior Node gives it. The read is evaluated
#: for its effect and its value is thrown away, which is what the operand of a sequence is for.
A_SEQUENCE_OPERAND_READING_A_NAME_NOTHING_BINDS = {
    'missing, console.log(1);\n': ('', 'ReferenceError'),
    '(missing, 0), console.log(1);\n': ('', 'ReferenceError'),
    'var v = (missing, 2);\nconsole.log(v);\n': ('', 'ReferenceError'),
    'console.log((missing, 1));\n': ('', 'ReferenceError'),
    'function f() {\n  missing, console.log(1);\n}\nf();\n': ('', 'ReferenceError'),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestASequenceOperandThatThrowsIsNotDropped(TestBase):
    """
    Every operand of a sequence expression is evaluated, in order, and every one but the last has
    its value discarded. Discarding the value is not licence to skip the evaluation: an operand that
    reads a name no binding resolves throws before the operands after it are reached, so a sequence
    whose first operand is such a read has no value at all and the program ends there.

    `TestAReadOfANameNothingBindsThrowsWhereverItStands` names one hook for its fix — the
    *read_effect* callback `side_effect_free` takes — and this is the case that claim does not
    reach. A sequence operand never passes through that callback:
    `JsSimplifications.visit_JsSequenceExpression` keeps an operand that `is_simple_expression`
    rejects or that `SemanticModel.read_has_dynamic_effect` accepts, and drops the rest. A bare
    identifier is simple, and `read_has_dynamic_effect` is the `with`-object question rather than
    the may-it-throw question, so the read is dropped by a route of its own and needs a fix of its
    own.

    The rows vary where the sequence stands, because that decides what is left behind rather than
    whether the operand goes: at the top of a statement, nested in another sequence, on the right of
    a declaration, inside a call's argument list, and inside a function body.
    """

    @unittest.expectedFailure
    def test_an_operand_whose_value_is_discarded_is_still_evaluated(self):
        """
        Node refuses every program of `A_SEQUENCE_OPERAND_READING_A_NAME_NOTHING_BINDS` having
        printed nothing, with a `ReferenceError` reading

            missing is not defined

        Every deobfuscation drops the operand and prints: the first four come back as
        `console.log(1);`, `console.log(1);`, `console.log(2);` and `console.log(1);`, and the last
        keeps the function with the read gone from its body.
        """
        rows = A_SEQUENCE_OPERAND_READING_A_NAME_NOTHING_BINDS
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            {source: (answer, answer) for source, answer in rows.items()},
        )


def _a_property_written_on_a_local_object(name: str) -> str:
    """
    A program that declares *name* as a local object, writes a property on it, and prints the object
    as JSON, so that a write that went missing is a line of output and not an error.
    """
    return (
        F'var {name} = {{}};\n'
        F'{name}.x = 1;\n'
        F'console.log(JSON.stringify({name}));\n'
    )


#: A program whose object is named by a local declaration, mapped to what Node prints for it. The
#: name is spelled seven ways: the six the model calls aliases of the global object, and one that
#: is no spelling of it at all, which is the control.
A_PROPERTY_WRITTEN_ON_AN_OBJECT_A_LOCAL_NAME_HOLDS = {
    _a_property_written_on_a_local_object(name): '{"x":1}\n'
    for name in ['globalThis', 'global', 'window', 'self', 'top', 'frames', 'obj']
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAPropertyWriteThroughAShadowedGlobalAliasSurvives(TestBase):
    """
    `window` names the global object only where nothing else binds it. A declaration of that name
    binds it, and from then on `window.x = 1` is a property write on whatever the declaration put
    there — a plain object here — which the program goes on to read back.

    The sweep that deletes a write of a global property never asks that question.
    `JsUnusedCodeRemoval._remove_dead_global_properties` takes a statement to be a global-property
    write when the base is an identifier whose *name* is one of
    `refinery.lib.scripts.js.deobfuscation.helpers.GLOBAL_OBJECT_ALIASES`, and decides from the
    spelling alone; `EffectModel._base_is_global_object` asks the model whether the name is bound
    before answering the same question, which is the answer this sweep needs.

    Two of the six spellings are wrong in the other direction and pass today, which is why they are
    rows and not omissions: the alias set the sweep reads holds four names, and the model's
    `refinery.lib.scripts.js.analysis.model.GLOBAL_OBJECT_ALIASES` holds six, so `top` and `frames`
    are outside the sweep's reach for a reason that has nothing to do with shadowing. The last row
    is the control: `obj` is a spelling of nothing, and its write is kept, so an entry that started
    passing by keeping every property write would be reported as an unexpected success.
    """

    @unittest.expectedFailure
    def test_a_write_on_an_object_a_local_name_holds_is_kept(self):
        """
        Node prints `{"x":1}` for every program of
        `A_PROPERTY_WRITTEN_ON_AN_OBJECT_A_LOCAL_NAME_HOLDS`, the property having been written on
        the object the declaration made. The four deobfuscations whose name is one the sweep
        reads — `globalThis`, `global`, `window` and `self` — come back with the assignment
        deleted and print `{}`.
        """
        rows = A_PROPERTY_WRITTEN_ON_AN_OBJECT_A_LOCAL_NAME_HOLDS
        self.assertEqual(
            {source: _before_and_after(source) for source in rows},
            _each_program_still_prints(rows),
        )


#: A program whose object literal names a property by its shorthand, mapped to the text a correct
#: deobfuscation writes for it. The one identifier of a shorthand is both the name of the property
#: and a read of the binding, so a constant reaches it the way it reaches any other read, and the
#: property still needs a name once the value is there. The last three rows are the controls, each
#: of them a substitution that lands where it belongs: into a key that is spelled out, into the
#: object a destructuring pattern takes apart, and into the default a shorthand in such a pattern
#: carries.
A_CONSTANT_REACHING_A_SHORTHAND_PROPERTY = {
    'var q = 1;\nconsole.log(JSON.stringify({ q }));\n':
        'console.log(JSON.stringify({ q: 1 }));',
    'var q = "a";\nconsole.log(JSON.stringify({ q }));\n':
        'console.log(JSON.stringify({ q: "a" }));',
    'var q = 1;\nvar w = 2;\nconsole.log(JSON.stringify({ q, w }));\n':
        'console.log(JSON.stringify({ q: 1, w: 2 }));',
    'var q = 1;\nvar o = { q, r: 2 };\nconsole.log(JSON.stringify(o));\n':
        'var o = { q: 1, r: 2 };\nconsole.log(JSON.stringify(o));',
    'var q = 1;\nconsole.log(JSON.stringify({ q: q }));\n':
        'console.log(JSON.stringify({ q: 1 }));',
    'var p = 5;\nvar o = { q: p };\nvar { q } = o;\nconsole.log(q);\n':
        'var o = { q: 5 };\nvar { q } = o;\nconsole.log(q);',
    'var d = 5;\nvar o = {};\nvar { q = d } = o;\nconsole.log(q);\n':
        'var o = {};\nvar { q = 5 } = o;\nconsole.log(q);',
}


class TestAConstantSubstitutedIntoAShorthandPropertyKeepsItsName(TestBase):
    """
    `{ q }` means `{ q: q }`, and what parts the two spellings is that the shorthand writes one
    identifier where the other writes two: the name of the property and the read of the binding are
    the same word. Node prints `{"q":1}` for

        var q = 1; console.log(JSON.stringify({ q }));

    and prints `{"q":1}` for `{ q: 1 }` written out in full, which is where the value of that read
    has to go. A property named by nothing is not a property, so the one thing a substitution here
    may not do is put the value where the name stood.

    `JsConstantInlining._substitute_constants` replaces the identifier through `_replace_in_parent`,
    and the parser stores one node in both the `key` and the `value` of a shorthand `JsProperty`, so
    the replacement lands on the name as well as on the read — twice over, once for each slot the
    walk reaches the node through. `JsSynthesizer.visit_JsProperty` writes a shorthand out by
    emitting its key alone, so what comes back is `{ 1 }`, `{ "a" }` and `{ 1, 2 }`, none of which
    any engine reads.

    The guard is written already and is written for exactly this:
    `refinery.lib.scripts.js.deobfuscation.helpers._substitute_use_position` refuses to substitute a
    non-computed key and clears `JsProperty.shorthand` when it replaces the value, so that the
    property is spelled in full. The inliner does not go through it.

    The entry is stated over the text the tool writes rather than over what running it prints,
    because the corrupt rows are not programs and cannot be run at all: Node answers `SyntaxError`
    for each of them, and answers it just as readily for any other way of breaking a file, so
    running them can say that something is wrong but never which value went where. The controls are
    the second reason. A fix that stopped substituting into an object literal altogether would leave
    every program here printing what it printed before, since refusing to touch a program cannot
    change what it does, and a behavior comparison would then report this entry as an unexpected
    success on the day the defect was covered over instead of fixed.

    Those controls are the last three rows, and each is the text the tool writes today. The first
    takes the same constant into a key that is spelled out and keeps the name beside it. The other
    two are the destructuring pattern, which was measured rather than assumed: a shorthand there
    names the binding the pattern writes and not one it reads, nothing is substituted over it, and a
    constant reaching the same statement lands on the object being taken apart in one row and on the
    default the shorthand carries in the other.
    """

    @unittest.expectedFailure
    def test_a_constant_substituted_into_a_shorthand_property_is_written_under_its_name(self):
        """
        Node prints `{"q":1}`, `{"q":"a"}`, `{"q":1,"w":2}`, `{"q":1,"r":2}`, `{"q":1}`, `5` and `5`
        for the seven programs of `A_CONSTANT_REACHING_A_SHORTHAND_PROPERTY`, and prints those same
        seven lines for the seven texts they are mapped to. The four whose property is written as a
        shorthand come back as `{ 1 }`, `{ "a" }`, `{ 1, 2 }` and `{ 1, r: 2 }`.
        """
        rows = A_CONSTANT_REACHING_A_SHORTHAND_PROPERTY
        self.assertEqual({source: _folded(source) for source in rows}, rows)
