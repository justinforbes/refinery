"""
A ledger of JavaScript defects that are known, understood, and not yet fixed.

The ones a release is held for are not here: `test.lib.scripts.js.test_release_blockers` holds
those, under the same rules, so that the question of whether the tool is fit to ship has one file
for an answer. An entry belongs there rather than here when a program an engine runs comes back
behaving differently over a shape real input plausibly holds. A behavior change only a shape
constructed for the defect reaches stays here whatever it costs, with the judgment of its
unlikelihood written on the entry that carries it; so does one that refuses to reduce
something, reduces it to something uglier, or mishandles a file no engine runs. Which
file an entry sits in says what it costs and never how well it is understood, so an entry moves
across when that is reassessed.

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

import inspect
import unittest

from collections import Counter

from test import TestBase
from test.lib.scripts.js.analysis.differential import (
    behavior,
    deobfuscate_source,
    module_graph_behavior,
    node_executable,
)
from test.lib.scripts.js.deobfuscation.test_array_length_reads import (
    A_COUNT_THE_FOLD_DOES_NOT_REACH,
)
from test.lib.scripts.js.deobfuscation.test_call_answers_a_wrapper import (
    a_string_array_whose_rotation_runs,
)
from test.lib.scripts.js.deobfuscation.test_escaped_identifiers import (
    AN_ESCAPED_ACCESSOR_TERMINAL,
    AN_ESCAPED_ASYNC_TERMINAL,
    AN_ESCAPED_KEYWORD_OPERATOR,
    AN_ESCAPED_STATIC_TERMINAL,
)
from test.lib.scripts.js.deobfuscation.test_stringarray import (
    A_PRESET_BESIDE_AN_ACCESSOR_CALL_NOTHING_CAN_ANSWER,
)
from test.lib.scripts.js.ledger import (
    Program,
    Reading,
    a_program,
    a_walk_of,
    an_accessor_at,
    before_and_after,
    before_and_after_in_a_host,
    each_program_still_prints,
    evaluated_in_a_body,
    folded,
    one_expected_failure_per_program,
    printed,
    prints,
    well_formed,
)
from test.lib.scripts.js.test_parameter_grammar import (
    A_BINDING_THE_KIND_OF_FUNCTION_RESERVES,
    A_FUNCTION_EXPRESSION_NAME_ONLY_THE_ENCLOSING_KIND_RESERVES,
)
from test.lib.scripts.js.test_parser_recovery import (
    A_POSITION_NAMING_A_BINDING_THE_FILE_CREATES,
    A_WORD_NO_MODULE_MAY_BIND,
)
from test.lib.scripts.js.test_template_literal import AN_ESCAPE_NEITHER_LITERAL_HAS
from test.lib.scripts.js.test_truncated_source import FOLDS_ANSWERED_WITH_A_PROGRAM

from refinery.lib.scripts import UnspellableNode
from refinery.lib.scripts.js.analysis.model import is_use_position
from refinery.lib.scripts.js.model import (
    JsBigIntLiteral,
    JsIdentifier,
    JsNumericLiteral,
    JsProperty,
    JsPropertyDefinition,
    JsStringLiteral,
)
from refinery.lib.scripts.js.parser import JsParser
from refinery.lib.scripts.js.synth import JsSynthesizer


_ASTRAL = chr(0x1F600)


def _sole_property_definition(source: str) -> JsPropertyDefinition:
    return [
        node for node in JsParser(source).parse().walk()
        if isinstance(node, JsPropertyDefinition)
    ][0]


def _sole_property(source: str) -> JsProperty:
    return [
        node for node in JsParser(source).parse().walk()
        if isinstance(node, JsProperty)
    ][0]


def _sole_string_literal(source: str) -> JsStringLiteral:
    return [
        node for node in JsParser(source).parse().walk()
        if isinstance(node, JsStringLiteral)
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
    A static block goes one step further and bans `await` outright, in every position and of its
    own accord, so that the word is no more a name there than it is an operator.
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
            printed(source), 'async function f() {\n  class C {\n    p = await;\n  }\n}'
        )

    @unittest.expectedFailure
    def test_the_await_operator_is_not_available_in_a_field_initializer(self):
        """
        Node refuses `async function f() { class C { p = await x; } }` with `SyntaxError:
        Unexpected identifier 'x'`, because `await` there is a name and no name may be followed by
        another one. Reading an operator the grammar does not offer in that position turns a file
        an engine rejects into a tree that claims to be a program.
        """
        self.assertEqual(well_formed('async function f() { class C { p = await x; } }'), False)

    @unittest.expectedFailure
    def test_a_class_static_block_bans_await_whatever_encloses_the_class(self):
        """
        Node refuses `class C { static { var await = 1; } }` with `SyntaxError: Unexpected reserved
        word`, and refuses the same class standing inside a plain function and inside an `async`
        one with the same message. The ban is the block's own, because the block is its own function
        context: no enclosure has to supply it and none can lift it.

        The `async` row is why this entry is quantified over three rather than written about one.
        The word is reserved throughout an async function's body, so a parser that lets that
        reservation reach through the class body into the block refuses that row for a reason of
        the enclosure's — the right answer by accident — while reading the other two as programs.
        """
        sources = [
            'class C { static { var await = 1; } }',
            'function f() { class C { static { var await = 1; } } }',
            'async function f() { class C { static { var await = 1; } } }',
        ]
        self.assertEqual(
            {source: well_formed(source) for source in sources},
            {source: False for source in sources},
        )

    @unittest.expectedFailure
    def test_yield_in_a_field_initializer_is_refused(self):
        """
        Node refuses `function* f() { class C { p = yield; } }` with `SyntaxError: Unexpected
        strict mode reserved word`. The initializer is not the generator's body, so `yield` is not
        the operator, and a class body is strict code, where `yield` is not a usable name either.
        """
        self.assertEqual(well_formed('function* f() { class C { p = yield; } }'), False)

    @unittest.expectedFailure
    def test_yield_is_banned_in_a_class_static_block(self):
        """
        Node refuses `function* f() { class C { static { yield; } } }` with `SyntaxError:
        Unexpected strict mode reserved word`, for the reason the field initializer is refused for:
        the block is its own context and it is strict.
        """
        self.assertEqual(well_formed('function* f() { class C { static { yield; } } }'), False)


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
            {source: well_formed(source) for source in rows},
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
            {source: behavior(printed(source)) for source in rows},
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
        self.assertEqual(before_and_after(source), (('function\n', None), ('function\n', None)))


class TestABindingNamedByAWordNoModuleMayBindIsNoProgram(TestBase):
    """
    An `import` or `export` declaration stands only in module code, and module code is strict with
    no directive saying so, so a binding one of them creates cannot be named by a word strict code
    reserves, by the one word only a module reserves, or by either of the two names strict code
    refuses to bind. `node --check` over a `.mjs` file refuses every word of
    `test.lib.scripts.js.test_parser_recovery.A_WORD_NO_MODULE_MAY_BIND` in every position of
    `A_POSITION_NAMING_A_BINDING_THE_FILE_CREATES` beside it, each with a `SyntaxError`:

        import yield from "m";               Unexpected strict mode reserved word
        import * as enum from "m";           Unexpected reserved word
        import { await } from "m";           Unexpected reserved word
        import { remote as eval } from "m";  Unexpected eval or arguments in strict mode

    The fifth position writes the binding with a `var` of its own before exporting it, so the
    declaration the host names in refusing `var yield;` and `export { yield };` together is that
    one. The same host binds `yield`, `let` and `eval` freely in a sloppy script, which is what
    makes the module and not the word the reason any of these is refused.

    The far side of the same declaration is a different position under no such restriction, a name
    reaching across the module boundary being an IdentifierName rather than one the file could
    refer to: the host reads `import { yield as v } from "m";` and `export * as await from "m";`.
    That half is already answered, by the law
    `TestAModuleTakesAWiderNameAcrossItsBoundaryThanItBinds` of the module the two corpora live in,
    which reads every word of both in every boundary position with no repair. A refusal that
    reaches a name across the boundary therefore turns that law red rather than this entry green.

    The words the language reserves outright are refused already in four of the five binding
    positions. The words only strict code and only a module reserve are read in all five, and the
    shorthand `import { yield } from "m";`, whose one word is a boundary name and a binding at
    once, is read for every word of the corpus. Each file so read prints back exactly as it went
    in, so what they cost is what the four files of
    `TestABindingNamedByAWordItsFunctionKindReservesIsNoProgram` cost:
    `refinery.lib.scripts.is_well_formed` answers `True` for a tree that is not a program, which is
    the domain every fidelity law is stated over, and a consumer reading that tree finds a module
    binding a name no module has.
    """

    @unittest.expectedFailure
    def test_a_binding_named_by_a_word_no_module_may_bind_is_not_a_well_formed_program(self):
        """
        Every file of the product is one the host refuses, and all of them are compared in a single
        answer so that any position or any word left reading is this entry still failing.
        """
        sources = [
            template.format(name=word)
            for template in A_POSITION_NAMING_A_BINDING_THE_FILE_CREATES.values()
            for word in A_WORD_NO_MODULE_MAY_BIND
        ]
        self.assertEqual(
            {source: well_formed(source) for source in sources},
            {source: False for source in sources},
        )

    @unittest.expectedFailure
    def test_the_shorthand_import_binds_the_word_the_shorthand_re_export_only_passes_on(self):
        """
        The word in `import { yield } from "m";` names a binding as well as the far side of the
        boundary, and the word in `export { yield } from "m";` names the far side twice over, which
        is why the host refuses the first and reads the second.
        """
        sources = ['import { yield } from "m";', 'export { yield } from "m";']
        self.assertEqual([well_formed(source) for source in sources], [False, True])


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


#: A function declaration written as the whole of an `if` clause, in each of the modes. Annex B.3.4
#: admits one in sloppy code, reading it as a block holding it; strict code has no such rule and the
#: grammar refuses a declaration where it requires a statement.
A_FUNCTION_DECLARATION_AS_AN_IF_CLAUSE = {
    'sloppy': a_program("""
        if (1) function W() { return 1; }
        console.log(W());
        """),
    'strict': a_program("""
        'use strict';
        if (1) function W() { return 1; }
        console.log(W());
        """),
}


class TestAFunctionDeclarationIsAClauseOnlyWhereAnnexBSaysSo(TestBase):
    """
    `if (x) function f() {}` is a program in sloppy code and no program in strict code: §B.3.4 gives
    the sloppy grammar an extra production reading the declaration as the block that would hold it,
    and strict code keeps the rule that a clause is a Statement, which a Declaration is not. Node
    runs the first and refuses the second with a `SyntaxError`.

    The parser reads both, so a file the engine would not load comes back as a deobfuscated program.
    That is mishandling of invalid input and never what a release is held for, which is why this is
    here and not in the other file; it is written down because the placement of a block-declared
    function is decided by the same production and a reader of that code will look for it.
    """

    @unittest.expectedFailure
    def test_a_clause_position_declaration_is_refused_in_strict_code(self):
        self.assertEqual(
            {
                mode: well_formed(source)
                for mode, source in A_FUNCTION_DECLARATION_AS_AN_IF_CLAUSE.items()
            },
            {'sloppy': True, 'strict': False},
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
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
        dropped = tuple(_dropped_source_characters(s, printed(s)) for s in sources)
        self.assertEqual(dropped, ('', ''))

    @unittest.expectedFailure
    def test_the_token_a_repair_steps_over_is_still_in_what_isprinted(self):
        """
        Node refuses all six of these: `missing ) after argument list` for the two argument lists,
        `Unexpected identifier 'b'` for the two parameter lists, `Unexpected string` for the catch
        parameter, and `Unexpected token 'break'` for the case clause. Standing where the grammar
        requires one token and finding another, `JsParser._expect` writes the token it wanted and
        steps over the one that was there, so `f('alpha' 'beta');` comes back as `f('alpha');` and
        the `break` of the case clause is nowhere in what comes back at all. No error node is built
        for the token that went and no other node holds its text, so the entire record of it is
        that the file is reported as one the parser repaired.
        """
        sources = [
            "f('alpha' 'beta');",
            "x = new C('alpha' 'beta');",
            'function f(a b) { return a; }',
            'class C { m(a b) {} }',
            "try { f(); } catch (e 'beta') {}",
            'switch (x) { case 1 break; }',
        ]
        self.assertEqual(
            [_dropped_source_characters(source, printed(source)) for source in sources],
            [''] * len(sources),
        )


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
        once = printed('x = /ab+')
        self.assertEqual(printed(once), once)

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
        once = [printed(source) for source in sources]
        self.assertEqual([printed(text) for text in once], once)

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
        once = [printed(source) for source in sources]
        self.assertEqual([printed(text) for text in once], once)

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
        once = [printed(source) for source in sources]
        self.assertEqual([printed(text) for text in once], once)

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
        once = printed('class D extends')
        self.assertEqual(printed(once), once)


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
        self.assertEqual(tuple(printed(source) for source in sources), tuple(sources))


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
            [folded(program) for program in programs],
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
            [before_and_after(source) for source in sources],
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
            [folded(F'console.log({read});') for read in counts],
            [F'console.log({count});' for count in counts.values()],
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
            {source: before_and_after(source) for source in rows},
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
            {source: before_and_after(source) for source in rows},
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
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
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
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
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
            {source: before_and_after(source) for source in rows},
            {source: (answer, answer) for source, answer in rows.items()},
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAConstAModuleExportsIsDeadUntilTheModuleRuns(TestBase):
    """
    A function declaration crosses a module boundary at link time: an importer in a cycle with its
    exporter can call the function before the exporter's body has run, and a `const` the function
    reads is then still in its dead zone, so the input throws `ReferenceError`. The model takes the
    export read to stand at the list's statement position, so the fold that puts the constant into
    the function proves its ordering over a walk the cycle never takes, and the deobfuscation
    returns the value where the input threw.

    Kept off the release gate for likelihood: the shape needs an import cycle whose second module
    calls back into the first at its own top level, which a single-file payload — the
    overwhelming shape of obfuscated malware — cannot spell at all. The fix is a link-time
    invocation point in `refinery.lib.scripts.js.analysis.dominance`, priced against every module
    fold there is.
    """

    @unittest.expectedFailure
    def test_the_cycle_still_throws_after_the_fold(self):
        exporter = inspect.cleandoc(
            """
            import { g } from './b.mjs';
            const x = 1;
            export { f };
            function f() { return x; }
            g();
            """
        )
        caller = inspect.cleandoc(
            """
            export function g() {}
            import { f } from './main.mjs';
            console.log(f());
            """
        )
        rewritten = deobfuscate_source(exporter, module=True)
        self.assertEqual(
            (
                module_graph_behavior({'b.mjs': caller, 'main.mjs': exporter}, 'main.mjs'),
                module_graph_behavior({'b.mjs': caller, 'main.mjs': rewritten}, 'main.mjs'),
            ),
            (('', 'ReferenceError'), ('', 'ReferenceError')),
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
    def test_a_module_keeps_printing_what_the_evalprinted(self):
        source = A_PAYLOAD_ONLY_A_SCRIPT_MAY_SPELL
        self.assertEqual(
            (
                behavior(source, module=True),
                behavior(deobfuscate_source(source, module=True), module=True),
            ),
            (('3\n', None), ('3\n', None)),
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
            {source: before_and_after(source) for source in rows},
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
            folded(A_READ_OF_THE_ALIAS_ONLY_ANOTHER_HOST_LACKS),
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
            {source: before_and_after(source) for source in rows},
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
            {source: before_and_after(source) for source in rows},
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
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )


#: A program that calls a function whose body reads a name nothing binds and prints the result,
#: mapped to the behavior Node gives it. The read stands in a store no later statement reads, which
#: is what lets the body be emptied down to the value it returns. The last two rows are the
#: A program handing a name nothing binds to a wrapper whose body reads the parameter under
#: `typeof`, mapped to what Node prints for it. At the call site the argument is read as a value
#: and throws; substituted into the `typeof` operand, the same spelling stands in the one position
#: a `ReferenceError` does not reach, so the inlined program prints `undefined` where the original
#: is caught. The second row is the control: a declared name prints the same under both.
A_THROWING_READ_TYPEOF_WOULD_MUTE = {
    'try {'
    ' console.log(function (a) { return typeof a; }(u));'
    " } catch (e) { console.log('caught'); }": 'caught\n',
    'var u = 1;'
    ' try {'
    ' console.log(function (a) { return typeof a; }(u));'
    " } catch (e) { console.log('caught'); }": 'number\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAReadMovedUnderTypeofKeepsItsThrow(TestBase):
    """
    `typeof` is the one operand position where reading a name nothing binds does not throw, so
    substituting a call-site argument into it erases the `ReferenceError` the call site raised.
    The admission — `refinery.lib.scripts.js.deobfuscation.helpers.is_safe_iife_inline` — counts
    a bare identifier argument as side-effect-free wherever the body uses it, because the effect
    gate `TestAReadOfANameNothingBindsThrowsWhereverItStands` pins answers that such a read does
    nothing. Even under the widening documented there, an argument moved into a `typeof` operand
    mutes the throw the widened gate would order, so the admission additionally has to refuse the
    position, not only classify the read.
    """

    @unittest.expectedFailure
    def test_an_argument_read_under_typeof_still_throws(self):
        """
        Node prints `caught` for the first program of `A_THROWING_READ_TYPEOF_WOULD_MUTE` and
        `number` for its control. The deobfuscation substitutes the argument into the operand and
        the first program comes back printing `undefined`.
        """
        rows = A_THROWING_READ_TYPEOF_WOULD_MUTE
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )


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
            {source: before_and_after(source) for source in rows},
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
            {source: before_and_after(source) for source in rows},
            {source: (answer, answer) for source, answer in rows.items()},
        )


class TestAStringNamedSpecifierTheGrammarBansIsRefused(TestBase):
    """
    An import or export list may name what it reads or writes across the module boundary with a
    string literal instead of a word, which is how a module reaches a name no identifier spells.
    What such a specifier names is the text the literal denotes, and
    `test.lib.scripts.js.test_module_export_names` states that law and the tool's compliance with
    it.

    Where the literal may stand is the part that is still wrong. Three positions the grammar bans
    it in are read all the same, so `refinery.lib.scripts.is_well_formed` answers `True` for a tree
    that spells no program, which is the domain every fidelity law is stated over. Nothing is
    answered wrongly: each of the three prints back as the text it was written as.

    Every answer below is Node's over a `.mjs` file, a module being the only kind of code these
    declarations appear in.
    """

    @unittest.expectedFailure
    def test_a_string_named_specifier_the_grammar_bans_is_not_a_well_formed_program(self):
        """
        Node refuses `export { 'a' };` with `SyntaxError: String literal module export names must
        be followed by a 'from' clause`, refuses `import { 'a' } from 'm';` with `SyntaxError:
        Unexpected reserved word`, a string being no name a module may bind a local to, and refuses
        the name holding an unpaired surrogate with `SyntaxError: Invalid module export name:
        contains unpaired surrogate`. It accepts `export { 'a' } from 'm';`,
        `import { 'a' as b } from 'm';` and `var a = 1; export { a as 'a' };`, so each refusal is
        about where the string stands and not about a string standing there at all.
        """
        unpaired_surrogate = F'{chr(92)}uD800'
        sources = [
            "export { 'a' };",
            "import { 'a' } from 'm';",
            F"var a = 1; export {{ a as '{unpaired_surrogate}' }};",
        ]
        self.assertEqual(
            {source: well_formed(source) for source in sources},
            {source: False for source in sources},
        )


class TestADeclarationTheGrammarRequiresAnInitializerForIsNoProgram(TestBase):
    """
    Two things decide together whether a declaration may be written with nothing assigned to it:
    the keyword it opens with, and whether what it binds is a name or a pattern. `var` and `let`
    bind a bare name and leave it undefined, `const` may not, and a destructuring target requires an
    initializer under every keyword, a pattern having nothing to take apart otherwise.
    `node --check` over a script gives:

        var x;      OK
        let x;      OK
        const x;    SyntaxError: Missing initializer in const declaration
        var [a];    SyntaxError: Missing initializer in destructuring declaration
        var {a};    SyntaxError: Missing initializer in destructuring declaration
        let [a];    SyntaxError: Missing initializer in destructuring declaration
        let {a};    SyntaxError: Missing initializer in destructuring declaration
        const [a];  SyntaxError: Missing initializer in destructuring declaration
        const {a};  SyntaxError: Missing initializer in destructuring declaration

    A `for` head is where the two answers part company. A for-in or a for-of head hands the binding
    its value on every pass, so a declaration standing there is written with no initializer whatever
    the keyword and whatever the target, and the host reads all eighteen of those files. The first
    clause of a C-style head is an ordinary declaration and both rules reach it unchanged: the host
    refuses `for (const x;;) {}` and `for (var [a];;) {}` with the two messages above, and reads
    `for (const x of o) {}` and `for (var [a] of o) {}`.

    Every file here comes back as the text it went in as, layout aside, so what they cost is what
    the four files of `TestABindingNamedByAWordItsFunctionKindReservesIsNoProgram` cost:
    `refinery.lib.scripts.is_well_formed` answers `True` for a tree that is not a program, which is
    the domain every fidelity law is stated over.
    """

    @unittest.expectedFailure
    def test_only_var_and_let_bind_a_bare_name_with_no_initializer(self):
        """
        The nine cells of the keyword against the target, in one answer, with the two files the
        host reads among them: a refusal that reaches `var x;` or `let x;` is a keyword rule
        applied where the target decides, and this entry still failing.
        """
        rows = {
            'var x;': True,
            'let x;': True,
            'const x;': False,
            'var [a];': False,
            'var {a};': False,
            'let [a];': False,
            'let {a};': False,
            'const [a];': False,
            'const {a};': False,
        }
        self.assertEqual({source: well_formed(source) for source in rows}, rows)

    @unittest.expectedFailure
    def test_a_for_head_lifts_the_requirement_only_where_it_supplies_the_value(self):
        """
        The same nine cells in each of the three heads a declaration stands in. The eighteen files
        of the two heads that iterate are read, so a fix that refuses a bare `const` or a bare
        pattern wherever it is written takes eighteen programs with it; the nine of the head that
        does not iterate are refused for the reasons the statement is refused for.
        """
        rows = {
            F'for ({kind} {target} {word} o) {{}}': True
            for kind in ('var', 'let', 'const')
            for target in ('x', '[a]', '{a}')
            for word in ('in', 'of')
        }
        rows.update({
            'for (var x;;) {}': True,
            'for (let x;;) {}': True,
            'for (const x;;) {}': False,
            'for (var [a];;) {}': False,
            'for (var {a};;) {}': False,
            'for (let [a];;) {}': False,
            'for (let {a};;) {}': False,
            'for (const [a];;) {}': False,
            'for (const {a};;) {}': False,
        })
        self.assertEqual({source: well_formed(source) for source in rows}, rows)


class TestAPropertyKeyIsSpelledByAName(TestBase):
    """
    An object literal names a property with an IdentifierName, with a string literal, with a
    numeric literal, or with a bracketed expression, and with nothing else. An IdentifierName is
    wider than a name the code around it could refer to, every reserved word spelling one, which is
    what makes `x = { if: 1 };` a program; it is a name all the same, and a punctuator is not one.
    `node --check` over a script refuses:

        x = { +: 1 };       SyntaxError: Unexpected token '+'
        x = { ,: 1 };       SyntaxError: Unexpected token ','
        x = { ;: 1 };       SyntaxError: Unexpected token ';'
        x = { ): 1 };       SyntaxError: Unexpected token ')'
        x = { %: 1 };       SyntaxError: Unexpected token '%'
        x = { ++: 1 };      SyntaxError: Unexpected token '++'
        x = { =>: 1 };      SyntaxError: Unexpected token '=>'
        x = { @: 1 };       SyntaxError: Invalid or unexpected token
        x = { #a: 1 };      SyntaxError: Unexpected identifier '#a'
        x = { +() {} };     SyntaxError: Unexpected token '+'
        class C { +() {} }  SyntaxError: Unexpected token '+'

    and reads `{ a: 1 }`, `{ if: 1 }`, `{ 'a b': 1 }`, `{ 1: 1 }`, `{ .5: 1 }`, `{ 08: 1 }`,
    `{ 1_0: 1 }`, `{ 1n: 1 }` and `{ [k]: 1 }`. Whatever token stands in the position is taken for
    the name and kept in a `refinery.lib.scripts.js.model.JsIdentifier` spelled by that token's
    text, so a key named `+` is a name to every consumer that reads one, the file prints back
    unchanged, and `refinery.lib.scripts.is_well_formed` calls the tree a program. The private name
    is the same reading one step further on: `#a` names a field of a class and nothing else, and an
    object literal that borrows it is no program either.

    A numeral carrying the BigInt suffix is what parts a fix from a careless one. It is a
    NumericLiteral like any other, so `x = { 1n: 1 };` is a program and Node answers the one key
    `1` for `Object.keys({ 1n: 1 })`. A key position narrowed to an identifier, a string and a
    plain decimal stops reading a file the language reads.
    """

    @unittest.expectedFailure
    def test_a_key_position_holding_no_name_is_not_a_well_formed_program(self):
        rows = {
            'x = { a: 1 };': True,
            'x = { if: 1 };': True,
            "x = { 'a b': 1 };": True,
            'x = { 1: 1 };': True,
            'x = { .5: 1 };': True,
            'x = { 08: 1 };': True,
            'x = { 1_0: 1 };': True,
            'x = { 1n: 1 };': True,
            'x = { [k]: 1 };': True,
            'x = { +: 1 };': False,
            'x = { ,: 1 };': False,
            'x = { ;: 1 };': False,
            'x = { ): 1 };': False,
            'x = { %: 1 };': False,
            'x = { ++: 1 };': False,
            'x = { =>: 1 };': False,
            'x = { @: 1 };': False,
            'x = { #a: 1 };': False,
            'x = { +() {} };': False,
            'class C { +() {} }': False,
        }
        self.assertEqual({source: well_formed(source) for source in rows}, rows)

    @unittest.expectedFailure
    def test_a_numeral_in_a_key_position_is_the_numeral_it_is_anywhere_else(self):
        """
        `x = 1n;` is read as the BigInt numeral it spells and `x = { 1n: 1 };` is not, though the
        same literal is written in both: the key is kept as a name whose text is `1n`, a spelling
        no identifier may carry and a property no object has. The decimal beside it is read as the
        numeral it is, so the position is not one that nothing reaches.
        """
        self.assertEqual(
            (
                type(_sole_property('x = { 1n: 1 };').key),
                type(_sole_property('x = { 1: 1 };').key),
            ),
            (JsBigIntLiteral, JsNumericLiteral),
        )


#: Programs asking how many own properties an object literal spelling `__proto__` has, mapped to
#: what Node prints for each. The three spellings answer differently: the shorthand gives the object
#: a property of that name, the two written with a colon set its prototype and give it none, and a
#: computed key gives it one again.
AN_OBJECT_LITERAL_SPELLING_PROTO = {
    'var __proto__ = 7; console.log(Object.keys({ __proto__ }).length);': '1\n',
    "console.log(Object.keys({ '__proto__': 7 }).length);": '0\n',
    'console.log(Object.keys({ __proto__: 1 }).length);': '0\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAnObjectLiteralSpellingProtoIsCounted(TestBase):
    """
    `__proto__` written as a key with a colon is not a key at all: it sets the object's prototype
    and the object has no property of that name, so `Object.keys` of such a literal answers one name
    fewer than the literal appears to hold. Written as a shorthand, or with the brackets of a
    computed key, it is an ordinary property like any other.

    Nothing here is answered wrongly — every program comes back doing what it did. What is refused
    is the fold: the interpreter declines a literal spelling that name at all rather than telling
    the three shapes apart, so a program whose whole point is the count comes back with the count
    still in it. The computed spelling is the one it does fold, and it folds correctly.

    `refinery.lib.scripts.js.deobfuscation.helpers.substitute_use_position` is where the shorthand
    is refused expansion, for the same reason and soundly: writing `{ __proto__ }` out as
    `{ __proto__: v }` is a different program.
    """

    @unittest.expectedFailure
    def test_an_object_literal_spelling_proto_folds_to_the_count_it_has(self):
        """
        Node prints `1`, `0` and `0` for the three programs of `AN_OBJECT_LITERAL_SPELLING_PROTO`.
        Each deobfuscation prints the same, and comes back with the literal and the call to
        `Object.keys` still standing where the count could have been.
        """
        rows = AN_OBJECT_LITERAL_SPELLING_PROTO
        self.assertEqual(
            {source: folded(source) for source in rows},
            {source: F'console.log({prints.strip()});' for source, prints in rows.items()},
        )


#: Declarations carrying an import-attribute clause, mapped to whether that clause is on an `import`
#: or on a re-`export`. The `import` rows are the controls: the clause is read there, and the two
#: forms are the same clause in the same place in the grammar.
A_DECLARATION_CARRYING_IMPORT_ATTRIBUTES = {
    "import j from './x.json' with { type: 'json' };": True,
    "import * as ns from './x.json' with { type: 'json' };": True,
    "export { default as j } from './x.json' with { type: 'json' };": False,
    "export * from './x.json' with { type: 'json' };": False,
    "export * as ns from './x.json' with { type: 'json' };": False,
}


class TestAReExportCarriesTheAttributesItWasWrittenWith(TestBase):
    """
    An import attribute clause stands on a re-export exactly as it stands on an import, and the
    parser has it only on the import. What follows the module specifier is read as a `with`
    statement instead, so `with { type: 'json' }` becomes a statement whose object is a read of a
    binding named `type` and whose body is the string, and the brace that closed the clause closes
    nothing: the text that comes back holds one more `}` than any file can.

    `refinery.lib.scripts.is_well_formed` answers `False` for each, so nothing is spliced anywhere,
    and what is wrong is that `refinery.js` writes the file at all. The attribute key becomes a read
    besides — `refinery.lib.scripts.js.analysis.model.is_use_position` counts it as one where the
    import form counts none — so a pass asking which names the module reads is told a name the
    module never mentions, and a rename would rewrite the key.
    """

    @unittest.expectedFailure
    def test_a_re_export_reads_its_attribute_clause_as_a_clause(self):
        """
        Each declaration of `A_DECLARATION_CARRYING_IMPORT_ATTRIBUTES` names one attribute, `type`,
        and no binding of that name, so no occurrence of it is a read. The three re-export rows
        report one read each and come back as text no engine parses.
        """
        rows = A_DECLARATION_CARRYING_IMPORT_ATTRIBUTES
        self.assertEqual(
            {source: (well_formed(source), _reads_named_type(source)) for source in rows},
            {source: (True, 0) for source in rows},
        )


def _reads_named_type(source: str) -> int:
    """
    How many occurrences of the name `type` in *source* stand where a binding is read or written.
    """
    tree = JsParser(source).parse()
    return sum(
        is_use_position(node) for node in tree.walk()
        if isinstance(node, JsIdentifier) and node.name == 'type'
    )


#: Programs whose `for-in` walk the language alone decides although the file wrote a prototype,
#: mapped to what Node prints for each. A property installed with `Object.defineProperty` and no
#: `enumerable` is not enumerable, and neither is an accessor installed the same way, so neither
#: reaches a walk. A `delete` takes a name off a chain rather than putting one on. And an own key
#: shadows an inherited one of the same name, so a receiver holding its own `z` walks `z` once
#: whatever `Object.prototype` holds.
A_WALK_A_WRITTEN_CHAIN_STILL_DECIDES = {
    a_walk_of('{a: 1}', 'Object.defineProperty(Object.prototype, "z", {value: 9});'):
        'a\n',
    a_walk_of('{a: 1}', an_accessor_at('Object.prototype', 'z')):
        'a\n',
    a_walk_of('{a: 1}', 'delete Object.prototype.toString;'):
        'a\n',
    a_walk_of('{z: 1, a: 2}', 'Object.prototype.z = 9;'):
        'za\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAWalkAWrittenChainStillDecidesIsStillWalked(TestBase):
    """
    A `for-in` is refused wherever the file wrote any prototype the receiver inherits from, which is
    a question about the whole chain where the walk is a question about each name on it. The four
    programs here write one and the walk is still the language's to decide, so each comes back with
    the loop standing where the names could have been folded in its place.

    The refusal is correct and the cost is recall, which is why this is a ledger entry rather than a
    release blocker. Closing it needs the per-name question: whether *this* key is enumerable on the
    chain, and whether the receiver's own slot shadows it. `EffectModel.chain_roots_unwritten` is
    the whole-chain question the walk asks today, and
    `refinery.lib.scripts.js.deobfuscation.interpreter.JsInterpreter._exec_for_in` says as much.
    """

    @unittest.expectedFailure
    def test_a_walk_a_written_chain_still_decides_is_folded(self):
        """
        Node prints `a`, `a`, `a` and `za` for the four programs of
        `A_WALK_A_WRITTEN_CHAIN_STILL_DECIDES`, and each deobfuscation prints the same. What none of
        them comes back as is the one `console.log` of those names that a walk over an untouched
        chain folds to.
        """
        rows = A_WALK_A_WRITTEN_CHAIN_STILL_DECIDES
        self.assertEqual(
            {source: folded(source) for source in rows},
            {source: F"console.log('{prints.strip()}');" for source, prints in rows.items()},
        )


#: Reads whose answer the receiver's own slot decides although the file wrote the chain behind it,
#: mapped to the text each could come back as. The namespace row writes the key onto the object
#: before reading it, so the chain is never consulted; the membership row asks for a name the
#: language puts on every object, which a write to a different key cannot take away.
A_READ_A_WRITTEN_CHAIN_DOES_NOT_REACH = {
    'Object.prototype.z = 9; var o = {}; o.z = 1; console.log(o.z);':
        'Object.prototype.z = 9;\nconsole.log(1);',
    "Object.prototype.q = 1; var o = {}; console.log('toString' in o);":
        'Object.prototype.q = 1;\nconsole.log(true);',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAReadAWrittenChainDoesNotReachIsStillAnswered(TestBase):
    """
    The other half of what the chain gate costs. Namespace flattening and the `in` operator both ask
    whether the file wrote any prototype the receiver inherits from, and hold back every key once
    one was written — including a key the file writes onto the receiver itself, which the chain
    never answers for, and a key on the chain that the write did not touch.

    Restoring this needs a write-side model rather than a wider read-side one: measured in Node, a
    write `o.z = 1` creates an own slot unless the chain holds that key as an accessor or as a
    non-writable data property, and in strict mode those two cases throw a `TypeError` rather than
    silently doing nothing. Until that exists, the own slot cannot be told from the chain's answer.
    """

    @unittest.expectedFailure
    def test_a_read_a_written_chain_does_not_reach_is_folded(self):
        """
        Node prints `1` and `true` for the two programs of `A_READ_A_WRITTEN_CHAIN_DOES_NOT_REACH`,
        and each deobfuscation prints the same. Neither comes back with the answer folded in place
        of the read.
        """
        rows = A_READ_A_WRITTEN_CHAIN_DOES_NOT_REACH
        self.assertEqual(
            {source: folded(source) for source in rows},
            A_READ_A_WRITTEN_CHAIN_DOES_NOT_REACH,
        )


#: Programs that reach `Object.prototype` through a name the file bound the receiver to, rather than
#: through the literal itself, mapped to what Node prints for each. Reading `__proto__` off a
#: binding, and handing that binding to a `getPrototypeOf` the file also bound, are the same gadget
#: written one indirection further out: what the spelling reaches is decided by the value the
#: binding holds, which the syntax at the write does not show.
A_PROTOTYPE_REACHED_THROUGH_A_BINDING = {
    'var a = {}; a.__proto__.z = 9; var o = {}; console.log(o.z);':
        '9\n',
    'var a = {}; var g = Object.getPrototypeOf; g(a).z = 9; var o = {}; console.log(o.z);':
        '9\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAPrototypeReachedThroughABindingIsStillWritten(TestBase):
    """
    `refinery.lib.scripts.js.deobfuscation.protospelling` writes each spelling that reaches an
    intrinsic prototype out as the name it reaches, which is what puts the write in front of every
    check already looking for one. It reads the receiver from the syntax, so a receiver held in a
    binding is one it declines: `test_a_receiver_the_syntax_does_not_decide_is_left_alone` pins that
    refusal, and these are what the refusal costs.

    Closing it needs the value the binding holds rather than new machinery — the model already
    answers what a binding is assigned, and the pass would consult it where the receiver is a name.
    The same is true of the callee: `g` is `Object.getPrototypeOf` and the file said so.

    Off the release gate deliberately: the one prototype write real obfuscation spells is the
    literal `X.prototype.m = f`, which is read; a write that reaches the prototype through
    a binding has so far had to be constructed to be seen.
    """

    @unittest.expectedFailure
    def test_a_prototype_reached_through_a_binding_answers_the_read(self):
        """
        Node prints `9` for both programs of `A_PROTOTYPE_REACHED_THROUGH_A_BINDING`, each of which
        reaches `Object.prototype` through a name rather than through a literal. Each deobfuscation
        prints `undefined`, and comes back having replaced the read with a variable nothing writes.
        """
        rows = A_PROTOTYPE_REACHED_THROUGH_A_BINDING
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )


#: Programs that reach `Object.prototype` by handing it to something that writes it, rather than by
#: writing through a member chain, mapped to what Node prints for each. The prototype is an ordinary
#: argument in each: to `Object.assign`, to `Reflect.set`, to `Reflect.deleteProperty`, and to a
#: function the file declares itself. The last row hands over `Object` rather than its prototype,
#: which reaches the same chain through one more member access and is a route of its own: the write
#: is spelled inside the callee, so nothing at the call site names the property it replaces.
A_PROTOTYPE_HANDED_TO_SOMETHING_THAT_WRITES_IT = {
    'Object.assign(Object.prototype, {z: 9}); var o = {}; console.log(o.z);':
        '9\n',
    'Reflect.set(Object.prototype, "z", 9); var o = {}; console.log(o.z);':
        '9\n',
    'Reflect.deleteProperty(Object.prototype, "toString");\n'
    + evaluated_in_a_body('{a: 1}', "'toString' in v"):
        'false\n',
    'function patch(p) { p.z = 9; } patch(Object.prototype); var o = {}; console.log(o.z);':
        '9\n',
    'function patch(o) { o.prototype.zz = 9; }\npatch(Object);\n'
    + evaluated_in_a_body('{a: 1}', 'v.zz'):
        '9\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAPrototypeHandedToAWriterIsStillWritten(TestBase):
    """
    A value that escapes into a call may be written by that call. `refinery.lib.scripts.js.analysis
    .effects` records that escape against the *keys* of every name it watches, `Object` among them,
    which is why `EffectModel.global_key_written` refuses each of these — and against the name
    itself only for the roots whose methods a fold is trusted to run, which `Object` is not. So
    `EffectModel.read_chain_intact` reports the chain intact, and every question about it is
    answered from the tables.

    A correct implementation records the escape once, against the name as much as against its keys,
    so that the two questions asked about one escape cannot disagree. Doing that and nothing else
    withdraws every chain answer from a file that hands `Object` to a call it cannot resolve, which
    the real samples do: `test.units.scripting.test_js` measures two of them coming back unreduced.
    So the escape has to be *followed* rather than assumed, which is the interprocedural precision
    this is deferred to — a callee whose writes are enumerable is a callee whose escape is not one.

    Off the release gate deliberately, with the entry above: prototype pollution through a
    writer is exploit vocabulary rather than dropper vocabulary, and no sample family
    observed so far hands its prototype to a call and reads the pollution back.
    """

    @unittest.expectedFailure
    def test_a_prototype_handed_to_a_writer_answers_the_read(self):
        """
        Node prints `9`, `9`, `false`, `9` and `9` for the five programs of
        `A_PROTOTYPE_HANDED_TO_SOMETHING_THAT_WRITES_IT`. Each deobfuscation answers as if the call
        it was handed to had not written it.
        """
        rows = A_PROTOTYPE_HANDED_TO_SOMETHING_THAT_WRITES_IT
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )


#: Programs that disable one of the mechanisms a prototype spelling goes through by writing it
#: through another such spelling, mapped to what Node prints for each. The first replaces the
#: `constructor` every plain object inherits, and the second replaces the `__proto__` accessor with
#: an own data property that shadows it; both write `Object.prototype` without naming `Object`.
A_MECHANISM_WRITTEN_THROUGH_A_SPELLING_OF_ITS_OWN = {
    'function C() {}\nC.prototype.q = 5;\n({}).__proto__.constructor = C;'
    '\nconsole.log(({}).constructor.prototype.q);':
        '5\n',
    'Object.defineProperty(({}).constructor.prototype, "__proto__", {value: 1});'
    '\n({}).__proto__.z = 9;\nconsole.log(({}).z);':
        'undefined\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAMechanismWrittenThroughASpellingIsStillWritten(TestBase):
    """
    `refinery.lib.scripts.js.deobfuscation.protospelling` gates each rewrite on
    `refinery.lib.scripts.js.analysis.effects.EffectModel.global_key_written`, which attributes a
    write to the name at the root of the chain it is written through. A chain rooted in a literal is
    attributed to no name, which is the whole reason the pass exists, so a write that disables one
    spelling's mechanism, made through another spelling, is invisible to the gate that would have
    refused it. The pass then rewrites a read of a mechanism the file had already replaced.

    Rewriting the write first closes only what the order reaches. The pass does re-read the models
    per rewrite, so a write spelled where the walk meets it before the read is attributed in time
    and the read behind it is refused —
    `test.lib.scripts.js.deobfuscation.test_protospelling` pins that much. What no order answers is
    the rest: the gate is a question about the whole program and the write may stand anywhere the
    walk reaches after the read, which the rows here are written to stand at.

    Closing it means attributing the write rather than rewriting it — teaching
    `refinery.lib.scripts.js.analysis.effects` that a member chain rooted in a literal receiver is
    rooted at the name `_PROTOTYPE_OWNERS` gives that receiver's prototype, so that one model build
    sees both spellings. The same step answers `var a = {}; a.__proto__.z = 9`, which
    `TestAPrototypeReachedThroughABindingIsStillWritten` records the pass as unable to read.
    """

    @unittest.expectedFailure
    def test_a_mechanism_written_through_a_spelling_is_still_written(self):
        """
        Node prints `5` and `undefined` for the two programs of
        `A_MECHANISM_WRITTEN_THROUGH_A_SPELLING_OF_ITS_OWN`, and each deobfuscation prints the
        same. Each comes back having rewritten a read whose mechanism the file replaced, and prints
        the answer that read has with the mechanism left alone.
        """
        rows = A_MECHANISM_WRITTEN_THROUGH_A_SPELLING_OF_ITS_OWN
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )


class TestTheStringArrayMachineryGoesOnceNothingReadsIt(TestBase):
    """
    `refinery.lib.scripts.js.deobfuscation.stringarray` keeps the array function, the accessor and
    the rotation IIFE while any of their names is still referenced, which is what
    `test.lib.scripts.js.deobfuscation.test_infrastructure_removal` states. The rows it keeps them
    for reach the end of a run with nothing referencing them after all: the reference stood in a
    function that was itself unreachable, and dead-code elimination took the function, and then the
    accessor, away. What is left is an array holder and a loop that rotates its contents, reachable
    from nothing and printed out in full.

    The pass cannot pick the work up again, because it recognizes the pattern by its accessor and
    there is no longer one. Closing it means the cleanup deciding from the array function and the
    rotation alone, and that in turn means proving the rotation loop terminates without the
    accessor calls its checksum is written in terms of — the simulation is what proves it today,
    and deleting a loop nothing proved terminates is the trade this pass just stopped making.
    """

    @unittest.expectedFailure
    def test_the_preset_folds_to_the_one_statement_it_computes(self):
        """
        Every string the program reads is resolved and the function reading the rest is
        unreachable, so nothing the machinery holds is read and the program is one call to
        `console.log`.
        """
        self.assertEqual(
            "console.log('test string');",
            deobfuscate_source(A_PRESET_BESIDE_AN_ACCESSOR_CALL_NOTHING_CAN_ANSWER),
        )


#: A call to a wrapper whose answer nothing keeps the wrapping of: the first `await`s it, and
#: awaiting a promise and awaiting the value it resolves to differ only in how many turns pass; the
#: second discards it, and a promise nobody holds is a value nobody reads. Each is mapped to the text
#: a fold that could see the call site would produce, measured by answering
#: `refinery.lib.scripts.js.model.wraps_return` with False.
A_WRAPPING_THE_CALL_SITE_TAKES_BACK_OFF = {
    "async function w(a) { return 'b'; }\n"
    '(async function () { console.log(await w(2)); })();\n':
        "(async function() {\n  console.log(await 'b');\n})();",
    'function send(u) { return u; }\n'
    'async function get(u) { return send(u); }\n'
    "get('http://example.test/payload');\n":
        "'http://example.test/payload';",
}


class TestAWrappingTheCallSiteTakesBackOffIsStillInlined(TestBase):
    """
    Refusing to answer a call to an `async` function is decided from the callee, and there are two
    call sites where the wrapping the callee adds is taken back off at once: an `await`ed call, and a
    call whose value is discarded. Both are reductions the guards give up, and the second is the one
    that costs triage — a downloader whose URL the fold used to surface now keeps the URL inside a
    body nothing reads out.

    Recovering them means a rule about the call site rather than about the callee, and the call site
    does not settle it on its own. A discarded wrapper that throws gives an unhandled rejection after
    the statement that follows it, where the direct call throws before it; an `await`ed one differs
    from its inlined form by up to two turns when the return is itself promise-valued. Terser
    (`inline.js:352`) and Closure (`InlineFunctions.java:357-362`) both refuse on the same predicate.
    """

    @unittest.expectedFailure
    def test_a_call_whose_wrapping_is_taken_off_is_answered(self):
        rows = A_WRAPPING_THE_CALL_SITE_TAKES_BACK_OFF
        self.assertEqual({source: folded(source) for source in rows}, rows)


class TestAStringArrayHolderNoLoopReadsIsStillResolved(TestBase):
    """
    The string array's rotation loop is what reads what the holder answered, and an `async` holder
    answers a promise, which has no `shift`. Where the loop actually turns the array over that makes
    the program throw, which is what
    `test.lib.scripts.js.deobfuscation.test_call_answers_a_wrapper` states. Where the checksum meets
    its target on the first pass the loop never touches the promise, the holder has already replaced
    itself with a plain function, and every later read answers the array — so the strings are
    resolvable and are no longer resolved.

    Separating the two means deciding whether the loop rotates before deciding whether the holder may
    be read, which is the rotation simulation the pass runs after it has recognized the holder.
    """

    @unittest.expectedFailure
    def test_a_holder_the_rotation_never_reads_is_answered(self):
        """
        Node prints `3`: the loop breaks on its first pass, so `arr` being a promise is never read,
        and `A` has replaced itself with the plain function every later call reads.
        """
        self.assertEqual(
            "console.log('3');",
            folded(a_string_array_whose_rotation_runs('async function', target=3)),
        )


class TestWellFormednessRefusesANonProgram(TestBase):
    """
    `refinery.lib.scripts.is_well_formed` is the domain over which fidelity is stated: it holds
    when every node in the tree spells something a parser agreed to read. A source no engine
    accepts is not such a tree, and a caller that is told otherwise compares a fabrication against
    the file it came from.

    What is left here is the class the parser reads without repairing anything: an assignment
    target the grammar refuses. A file that stops in the middle of a construct is answered
    correctly and its law lives in `test.lib.scripts.js.test_parser_recovery`.
    """

    @unittest.expectedFailure
    def test_an_arrow_function_is_not_an_update_target(self):
        """
        Node refuses `f = a => {}++` with `SyntaxError: Unexpected token '++'`. An update operator
        needs an operand it can write back to, and a function made on the spot is not a reference.
        """
        self.assertEqual(well_formed('f = a => {}++'), False)

    @unittest.expectedFailure
    def test_a_function_expression_is_not_an_update_target(self):
        """
        Node refuses `f = function () {}++` with `SyntaxError: Invalid left-hand side expression in
        postfix operation`, naming the same missing target the arrow form lacks.
        """
        self.assertEqual(well_formed('f = function () {}++'), False)


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


def _refuses_to_print(source: str) -> bool:
    """
    Whether `refinery.js` declines to write anything for *source*, which is the only answer that can
    be given for a buffer holding a literal no text spells.
    """
    try:
        folded(source)
    except UnspellableNode:
        return True
    else:
        return False


def _spelled_with_an_escaped_identifier(source: str) -> str:
    """
    *source* with each placeholder replaced by the unicode escape spelling the characters it names.

    The escapes are assembled from `chr(92)` rather than written out, because an escape written into
    this file is one flattening away from being the characters it denotes, and an entry that no
    longer holds the spelling it asks about asks nothing at all. A source that named a placeholder
    and came back without a backslash is that flattening having happened, and it is refused here
    rather than left to be discovered as an entry that quietly stopped asking anything.
    """
    result = source.replace('ESCAPED_IF', F'{chr(92)}u0069f')
    result = result.replace('ESCAPED_AIT', F'{chr(92)}u0061it')
    result = result.replace('ESCAPED_ET', F'{chr(92)}u0065t')
    if result != source and chr(92) not in result:
        raise AssertionError(F'the escape in {source!r} was flattened away')
    return result


#: Files the language refuses although nothing in them was fabricated by the parser, each mapped to
#: whether it is a module. Every one of them parses cleanly: what refuses them is an early error,
#: which is a rule about a tree rather than about the text a parser could not read.
A_FILE_REFUSED_WITH_NOTHING_FABRICATED = {
    _spelled_with_an_escaped_identifier(source): module
    for source, module in (
        ('function ESCAPED_IF(){ return 1; } console.log(2);', False),
        ('let let = 1; console.log(2);', False),
        ('var o = { __proto__: null, __proto__: {} }; console.log(2);', False),
        ('var await = 1; console.log(2);', True),
        ('var awESCAPED_AIT = 1; console.log(2);', True),
    )
}


#: Binding positions written as an object pattern whose shorthand names a reserved word. The one
#: node the parser builds there is the key and the binding at once, so the refusal that reaches
#: every other binding position — a declarator, an array pattern, a parameter — does not reach this
#: one, and `var { if: x } = o` is a program, which is why the refusal cannot simply move onto the
#: key.
A_BINDING_PATTERN_NAMING_A_RESERVED_WORD = tuple(
    _spelled_with_an_escaped_identifier(source) for source in (
        'var { ESCAPED_IF } = { if: 7 }; console.log(1);',
        'var { ESCAPED_IF = 1 } = {}; console.log(1);',
        'function f({ ESCAPED_IF }){ return 1; } console.log(f({}));',
    )
)


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAFileRefusedWithNothingFabricatedIsNotAnsweredWithAProgram(TestBase):
    """
    A file can parse with every token the source wrote and still be one no engine runs, because the
    rule refusing it is stated over the tree rather than over the text: a name whose escapes spell a
    reserved word, a `let` binding named `let`, two `__proto__` keys in one literal, and a module
    binding `await` are all read without anything being repaired or invented.

    `refinery.lib.scripts.is_well_formed` answers `True` for each, which is the honest answer to the
    question it asks — nothing was fabricated — and the wrong answer to the one every caller wants,
    which is whether the tree spells a program. What follows from that is a file the analyst is
    handed as though it ran: each of these comes back reduced, with the one thing wrong with it
    removed along with the code it stood in.

    `test.lib.scripts.js.test_parser_recovery` states the other half, where the parser did supply
    something and says so. Closing this one needs a refusal mechanism that does not exist yet, and
    the escaped-name row shows the shape it must have: the parser answers with the span it read
    wherever the model has a node kind for one, and a declared function's name is a slot that holds
    an identifier and nothing else.
    """

    @unittest.expectedFailure
    def test_a_file_the_language_refuses_is_refused(self):
        """
        Node refuses every program of `A_FILE_REFUSED_WITH_NOTHING_FABRICATED` with a `SyntaxError`
        and prints nothing for it. Each deobfuscation prints `2`.
        """
        rows = A_FILE_REFUSED_WITH_NOTHING_FABRICATED
        refused = ('', 'SyntaxError')
        self.assertEqual(
            {source: before_and_after(source, module=module) for source, module in rows.items()},
            {source: (refused, refused) for source in rows},
        )

    @unittest.expectedFailure
    def test_a_pattern_binding_a_reserved_word_is_no_program(self):
        """
        Node refuses every program of `A_BINDING_PATTERN_NAMING_A_RESERVED_WORD`, each of which
        binds a name whose escapes spell `if` through an object pattern. What each comes back as is
        refused too, so nothing runs that should not; what is wrong is that
        `refinery.lib.scripts.is_well_formed` answers `True` for all three, and that answer is what
        decides whether such a text may be spliced into a file that does run.
        """
        rows = A_BINDING_PATTERN_NAMING_A_RESERVED_WORD
        self.assertEqual(
            {source: well_formed(source) for source in rows},
            {source: False for source in rows},
        )


#: Further shapes of the repair `A_FILE_THE_PARSER_REPAIRED` is about, one written with no escape at
#: all so that the family is not read as being about escapes, and one spelling `let` where a
#: declaration would begin.
A_REPAIR_WITH_NOTHING_ESCAPED_ABOUT_IT = (
    "console.log('alpha' 'beta');",
    _spelled_with_an_escaped_identifier('lESCAPED_ET x = 1; console.log(x);'),
)


#: Every file whose parse needed a token the source did not write. Four of the tables come from
#: `test.lib.scripts.js.deobfuscation.test_escaped_identifiers`, where the law they belong to is
#: stated: a terminal word of the grammar is matched by the characters typed, so an escaped
#: spelling of `get`, `set`, `static`, `async`, `instanceof` or `in` is a name standing where the
#: grammar wanted a word, and the parser writes the separator that would have to be there.
A_FILE_THE_PARSER_REPAIRED = (
    *AN_ESCAPED_ACCESSOR_TERMINAL,
    *AN_ESCAPED_STATIC_TERMINAL,
    *AN_ESCAPED_ASYNC_TERMINAL,
    *AN_ESCAPED_KEYWORD_OPERATOR,
    *A_REPAIR_WITH_NOTHING_ESCAPED_ABOUT_IT,
)


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAFileTheParserRepairedIsNotAnsweredWithAProgram(TestBase):
    """
    Standing where the grammar requires one token and finding another, the parser writes the token
    it wanted and reads on. It records that it did — `refinery.lib.scripts.is_well_formed` answers
    `False` for every file here — and nothing between that record and the printer reads it, so what
    comes back is a program built out of text no engine agreed to read.

    Two of these are the expensive shape. `[] instanceof Array` and `'a' in {a: 1}` written with an
    escaped operator lose the operator and keep both operands, so the file comes back printing them;
    and `class C { get x(){} }` written the same way comes back declaring a field beside a method,
    which runs and prints a function where Node refuses the file outright.

    `test_a_file_the_language_refuses_is_refused` states the same cost for the files where nothing
    was repaired at all. That one needs a refusal mechanism to be built; this one needs only a
    reader for the record the parser already keeps.
    """

    @unittest.expectedFailure
    def test_a_file_the_parser_repaired_is_refused(self):
        """
        Node refuses every program of `A_FILE_THE_PARSER_REPAIRED` with a `SyntaxError` and prints
        nothing for it. Each deobfuscation is a file that parses, and five of them print.
        """
        rows = A_FILE_THE_PARSER_REPAIRED
        refused = ('', 'SyntaxError')
        self.assertEqual(
            {source: (well_formed(source), before_and_after(source)) for source in rows},
            {source: (False, (refused, refused)) for source in rows},
        )


#: A self-disabling wrapper called inside a `with` body whose scope object carries the wrapper's
#: name, mapped to what Node prints for it.
A_WITH_OBJECT_CARRYING_A_WRAPPER_NAME = {
    "var o = { W: function (a) { console.log('real', a); } };\n"
    'function W() { W = function () {}; }\n'
    'with (o) { W(1); }\n': 'real 1\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAWithObjectMayCarryTheNameAWrapperAnswersTo(TestBase):
    """
    A call inside a `with` body reads the scope object first, and the wrapper only where the object
    lacks the name. The wrapper expansion assumes the object lacks it. The assumption is
    deliberate: the files the pass exists for call their wrappers inside `with` dispatch blocks
    whose objects never carry the wrapper's name - an obfuscator that put it there would break its
    own program - and refusing every call a `with` body makes is measured to forfeit the whole
    recovery of one of the three real samples. Deciding the property's absence instead would take
    interprocedural object facts the analysis does not have.
    """

    @unittest.expectedFailure
    def test_a_call_the_scope_object_answers_is_left_standing(self):
        """
        Node prints `real 1` for the program of `A_WITH_OBJECT_CARRYING_A_WRAPPER_NAME`: the scope
        object's own `W` answers the call. The deobfuscation lowers the call to its argument and
        prints nothing.
        """
        rows = A_WITH_OBJECT_CARRYING_A_WRAPPER_NAME
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    def test_a_call_the_scope_object_does_not_answer_is_still_expanded(self):
        """
        The acceptance's other half, and the fizzbuzz-03 recovery in miniature: where the scope
        object lacks the name, the call is the wrapper's and still expands. A guard refusing every
        call a `with` body makes would flip the entry above to an unexpected success by forfeiting
        exactly this.
        """
        self.assertEqual(
            folded(
                'var o = { p: 1 };\n'
                'function W() { W = function () {}; }\n'
                'with (o) { W(console.log(1)); }\n'
                'console.log(2);\n'
            ),
            'var o = { p: 1 };\n'
            'with (o) {\n'
            '  console.log(1);\n'
            '}\n'
            'console.log(2);',
        )


#: A self-disabling wrapper rebound by a direct `eval` of a string no fold can read, mapped to
#: what Node prints for it.
A_WRAPPER_REBOUND_BY_AN_UNREADABLE_EVAL = {
    'function W() { W = function () {}; }\n'
    'eval(String(Math.random() < 2 && "W = function (a) { console.log(7, a); }"));\n'
    'W(1);\n': '7 1\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAnUnreadableEvalMayRebindAWrapper(TestBase):
    """
    A direct `eval` of a string nothing can fold may write any name in its scope, so after one
    runs, a wrapper's name may hold anything. The wrapper expansion accepts this: it is the
    acceptance namespace flattening and the dispatcher unwrapper already record, made because a
    reflective surface is exactly what the real obfuscated files carry on the way in, and a pass
    gated on one never runs and never clears the surface that was gating it. An `eval` a fold can
    read is not covered here: its assignment is inlined as real code before the expansion decides,
    and the expansion then sees the write.
    """

    @unittest.expectedFailure
    def test_a_call_after_the_eval_reaches_what_it_bound(self):
        """
        Node prints `7 1` for the program of `A_WRAPPER_REBOUND_BY_AN_UNREADABLE_EVAL`: the `eval`
        argument always evaluates to an assignment rebinding `W`. The deobfuscation lowers the
        call to its argument and prints nothing.
        """
        rows = A_WRAPPER_REBOUND_BY_AN_UNREADABLE_EVAL
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    def test_the_eval_argument_is_still_unread(self):
        """
        The entry above pins the acceptance only while nothing reads the string: a fold that
        learns to decide `Math.random() < 2` would flip it to an unexpected success by making the
        rebind visible, not by closing the acceptance. This holds the string unread.
        """
        source, = A_WRAPPER_REBOUND_BY_AN_UNREADABLE_EVAL
        self.assertEqual(
            folded(source),
            'eval(String(Math.random() < 2 && "W = function (a) { console.log(7, a); }"));\n1;',
        )


#: A self-disabling wrapper called inside a `with` body whose scope object lacks the wrapper's name
#: but watches it being looked for, mapped to what Node prints for it.
A_WITH_OBJECT_WATCHING_FOR_A_WRAPPER_NAME = {
    "var o = new Proxy({}, { has: function (t, k) { console.log('asked', k); return false; } });\n"
    'function W() { W = function () {}; }\n'
    'with (o) { W(console.log(1)); }\n'
    'console.log(2);\n': 'asked W\nasked console\n1\n2\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAWithObjectMayWatchForTheNameAWrapperAnswersTo(TestBase):
    """
    The other half of what a `with` body costs, and the half its scope object does not have to carry
    the name to observe: a call inside the body asks the object for every name it spells, and an
    object can answer that question with code. Expanding the call takes the question away, so a
    program that watched for the wrapper's name stops being asked it.

    This is the acceptance `TestAWithObjectMayCarryTheNameAWrapperAnswersTo` records, priced the
    same way and closed the same way: refusing every call a `with` body makes forfeits the whole
    recovery of one of the three real samples, and deciding that an object neither carries the name
    nor watches for it takes interprocedural object facts the analysis does not have.
    """

    @unittest.expectedFailure
    def test_the_question_the_expansion_takes_away_was_answered_by_code(self):
        """
        Node prints `asked W` and then `asked console` for the program of
        `A_WITH_OBJECT_WATCHING_FOR_A_WRAPPER_NAME`: the body asks the scope object for both names.
        The deobfuscation expands the call, and the output asks only for `console`.
        """
        rows = A_WITH_OBJECT_WATCHING_FOR_A_WRAPPER_NAME
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )


#: A self-disabling wrapper rebound through the global object under a name no fold can read, mapped
#: to what a host running the file as a classic script prints for it.
A_WRAPPER_REBOUND_UNDER_AN_UNREADABLE_KEY = {
    'function W() { W = function () {}; }\n'
    "globalThis[['W'].join('')] = function (a) { console.log('real', a); };\n"
    'W(1);\n': 'real 1\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAnUnreadableKeyMayRebindAWrapperThroughTheGlobalObject(TestBase):
    """
    A write to the global object under a computed key names whatever the key evaluates to, and a key
    no fold reads names anything at all — including a wrapper the file declares at its top level,
    which under the script execution model is a property of that same object.

    The wrapper expansion accepts this, for the reason `TestAnUnreadableEvalMayRebindAWrapper`
    states: a reflective surface is what the real obfuscated files carry on the way in, and a pass
    gated on one never runs and never clears the surface that was gating it. A key a fold can read
    is not covered here — the write is then an ordinary one the model records against the binding,
    and the expansion refuses.
    """

    @unittest.expectedFailure
    def test_a_call_after_the_write_reaches_what_it_bound(self):
        """
        A host prints `real 1` for the program of `A_WRAPPER_REBOUND_UNDER_AN_UNREADABLE_KEY`: the
        key spells the wrapper's own name. The deobfuscation lowers the call to its argument and
        prints nothing.
        """
        rows = A_WRAPPER_REBOUND_UNDER_AN_UNREADABLE_KEY
        self.assertEqual(
            {source: before_and_after_in_a_host(source) for source in rows},
            each_program_still_prints(rows),
        )

    def test_the_key_is_still_unread(self):
        """
        The entry above pins the acceptance only while nothing reads the key: a fold that learns to
        answer `['W'].join('')` would flip it to an unexpected success by making the write visible,
        not by closing the acceptance. This holds the key unread.
        """
        source, = A_WRAPPER_REBOUND_UNDER_AN_UNREADABLE_KEY
        self.assertEqual(
            folded(source),
            "globalThis[['W'].join('')] = function(a) {\n"
            "  console.log('real', a);\n"
            '};\n'
            '1;',
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestScriptCodeHasTwoMoreCommentOpeners(TestBase):
    """
    Script code reads two comment delimiters beyond `//` and `/*`, held over from the years a
    script was written inside an HTML comment so that a browser which did not know the tag printed
    nothing instead of the source. Each runs to the end of its line the way `//` does, and only one
    of them is free to stand anywhere.

    `<!--` opens a comment wherever a comment may open, the middle of an expression included, so
    `var y = x <!-- note` declares `y` and holds what `x` holds. `-->` opens one only where nothing
    but whitespace and comments precedes it on its line, the head of the file counting as such a
    line; anywhere else those three characters are the decrement operator and `>`, which is what
    makes `a-->b` the program `a-- > b`. Node reads a script holding either delimiter and refuses
    the same file read as a module with `SyntaxError: HTML comments are not allowed in modules`,
    the two being script grammar and nothing else; it refuses `console.log(1); --> note` and
    `var a = 1; /* c */ --> note` with `SyntaxError: Unexpected token '>'`, a statement in front of
    the delimiter on its line being what the positional restriction is about.

    Neither delimiter is read here, `<!--` being taken for `<` against `!` and `-->` for a
    decrement against `>`, and what that costs runs in both directions at once. Ten of the files
    below are refused, `refinery.lib.scripts.is_well_formed` answering `False` for programs a host
    runs; the eleventh, `var y = x <!-- note`, is called a program and comes back as
    `var y = x < !--note;`, which reads the word behind the delimiter and throws over it. What the
    printer writes for the rest is text resegmented into statements the file never held: a `-->`
    comment becomes a statement of its own with the comment text standing behind it as a second, so
    a file that printed `1` and `2` comes back printing `1` and then throwing.

    Off the release gate deliberately: both delimiters stand in the input in plain sight, so
    an analyst who meets one recognizes the HTML wrapper and cuts it — a work-around,
    where the gate is for the wrong answers an analyst cannot see.
    """

    @unittest.expectedFailure
    def test_a_file_a_host_reads_is_a_well_formed_program(self):
        """
        The delimiters in every position that decides one: at the head of a file, behind a
        statement, at the end of one, inside a function body, and behind whitespace, a comment and
        a comment that spans lines. The three files that hold neither delimiter and the two the
        host refuses are answered here already, and they stand in the same answer so that a fix
        reading `-->` wherever it is written turns `a-->b` red.
        """
        rows = {
            '<!-- note\nconsole.log(1);': True,
            'console.log(1); <!-- note': True,
            'console.log(1);\n<!--': True,
            'var x = 1;\nvar y = x <!-- note\nconsole.log(y);': True,
            '--> note\nconsole.log(1);': True,
            'console.log(1);\n--> note\nconsole.log(2);': True,
            'console.log(1);\n   --> note\nconsole.log(2);': True,
            'console.log(1);\n/* c */ --> note\nconsole.log(2);': True,
            'console.log(1); /* c\n */ --> note\nconsole.log(2);': True,
            'function f() {\n--> note\nreturn 1;\n}\nconsole.log(f());': True,
            'var a = 2, b = 0;\nconsole.log(a-->b);': True,
            "console.log('-->');": True,
            'console.log(`x\n--> y`);': True,
            'console.log(1); --> note': False,
            'var a = 1; /* c */ --> note': False,
        }
        self.assertEqual({source: well_formed(source) for source in rows}, rows)

    @unittest.expectedFailure
    def test_the_text_printed_for_one_prints_what_the_file_prints(self):
        """
        What each of these files prints is what the host prints for it, and the text the printer
        hands back has to print the same. `<!--` read as two operators leaves text the host refuses
        outright; `-->` read as a decrement leaves a program that prints the output ahead of the
        delimiter and then throws a `ReferenceError` over the word behind it, which is a file's
        second half going missing behind an answer that reads as one.
        """
        programs = {
            '<!-- note\nconsole.log(1);': '1\n',
            'console.log(1); <!-- note': '1\n',
            'console.log(1);\n<!--': '1\n',
            'var x = 1;\nvar y = x <!-- note\nconsole.log(y);': '1\n',
            '--> note\nconsole.log(1);': '1\n',
            'console.log(1);\n--> note\nconsole.log(2);': '1\n2\n',
            'console.log(1);\n   --> note\nconsole.log(2);': '1\n2\n',
            'console.log(1);\n/* c */ --> note\nconsole.log(2);': '1\n2\n',
            'console.log(1); /* c\n */ --> note\nconsole.log(2);': '1\n2\n',
            'function f() {\n--> note\nreturn 1;\n}\nconsole.log(f());': '1\n',
            'var a = 2, b = 0;\nconsole.log(a-->b);': 'true\n',
            "console.log('-->');": '-->\n',
            'console.log(`x\n--> y`);': 'x\n--> y\n',
        }
        self.assertEqual(
            {source: behavior(printed(source)) for source in programs},
            {source: (prints, None) for source, prints in programs.items()},
        )


#: An accessor an IIFE answers, over a closure the answered function writes through a member of or
#: reads the identity of, mapped to what Node prints for it. Measured against
#: `refinery.lib.scripts.js.deobfuscation.iifeaccessor._is_safe_to_promote` answering False, which
#: leaves both programs standing whole.
A_CLOSURE_THE_PROMOTED_ACCESSOR_STOPS_SHARING = {
    'var acc = (function () {\n'
    '  var t = [0];\n'
    '  return function (i) { t[0] = t[0] + i; return t[0]; };\n'
    '})();\n'
    'console.log(acc(1), acc(1));\n': '1 2\n',
    'var acc = (function () {\n'
    "  var t = ['a'];\n"
    '  return function () { return t; };\n'
    '})();\n'
    'console.log(acc() === acc());\n': 'true\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAPromotedClosureIsStillOneObjectAcrossCalls(TestBase):
    """
    An IIFE answering a function is inlined by moving what the IIFE declared into the answered
    function's body, which builds those declarations afresh on every call. That is equivalent only
    where nothing carries a value or an identity from one call to the next, and `_is_safe_to_promote`
    asks it of a closure name written through a bare identifier and of nothing else: a closure
    written through a member keeps no count, and one whose identity is compared is a different object
    each time it is answered.

    Off the release gate deliberately: the accessor closures real obfuscators emit are
    memo-caches, whose rebuilt state recomputes the same values, so a promotion changes
    their speed and nothing an engine reports; state a program can watch accumulating
    across calls has had to be constructed.
    """

    @unittest.expectedFailure
    def test_a_closure_the_accessor_keeps_writing_or_comparing_is_not_promoted(self):
        """
        Node prints `1 2` for the first program of `A_CLOSURE_THE_PROMOTED_ACCESSOR_STOPS_SHARING`
        and `true` for the second. The deobfuscation folds them to `console.log(1, 1);` and
        `console.log(['a'] === ['a']);`, which print `1 1` and `false`.
        """
        rows = A_CLOSURE_THE_PROMOTED_ACCESSOR_STOPS_SHARING
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )


#: A program whose block-declared function is kept out of the enclosing scope by a lexical binding
#: of the same name, mapped to the behavior an engine gives it. The `let` is what stops the copy, so
#: it is read by the placement whether or not anything else in the program names it.
A_LEXICAL_BINDING_THAT_STOPS_A_BLOCK_FUNCTION_ESCAPING = {
    'a let nothing else reads': Program(
        a_program("""
            function outer() {
              { let f = 1; { function f() { return 2; } } }
              console.log(typeof f);
            }
            outer();
            """),
        prints('undefined'),
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
@one_expected_failure_per_program(A_LEXICAL_BINDING_THAT_STOPS_A_BLOCK_FUNCTION_ESCAPING)
class TestALexicalBindingThatStopsABlockFunctionEscapingIsKept(TestBase):
    """
    A `let` between a block-declared function and the variable scope stops Annex B giving that
    function a `var` outside its block (§B.3.3.1), so the name means nothing after the block and
    `typeof` answers `undefined`. Reading the `let` is the whole of what it is for: no other
    statement of the program has to name it for it to decide that.

    `refinery.lib.scripts.js.deobfuscation.unused` removes a declarator nothing references, and a
    declarator whose only consumer is the placement of another declaration is one it counts as
    unreferenced. With the `let` gone the copy runs, the name reaches the enclosing scope, and the
    program comes back printing `function`.

    The removal is what is wrong here rather than the placement, which is why this is its own entry:
    `refinery.lib.scripts.js.analysis.model.annex_b_var_home` reads the `let` correctly, and answers
    a different question once a pass has deleted it.

    Off the release gate deliberately: no real file writes a `let` whose one consumer is the
    placement of a block function two blocks in, probed by a `typeof` after both.
    """


#: A program whose parameter default runs a direct `eval` declaring a name the body reads, mapped
#: to the behavior an engine gives it.
A_DIRECT_EVAL_IN_A_DEFAULT_DECLARING_A_NAME = {
    'a var the eval declares, read by the body': Program(
        a_program("""
            var v = 1;
            function f(x = eval('var v = 2')) { return v; }
            console.log(f());
            """),
        prints('2'),
    ),
    'a var the eval declares, read by a later default': Program(
        a_program("""
            var v = 1;
            function f(a = eval('var v = 2'), b = v) { return b; }
            console.log(f());
            """),
        prints('2'),
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
@one_expected_failure_per_program(A_DIRECT_EVAL_IN_A_DEFAULT_DECLARING_A_NAME)
class TestADirectEvalInADefaultDeclaresWhereItRuns(TestBase):
    """
    A direct `eval` in sloppy code declares a `var` in the variable scope it runs in, and the
    parameter list of a function carrying an expression is such a scope. The name it declares is
    therefore one the body reads instead of the outer one it would otherwise have read, and the
    program prints what the `eval` put there.

    The analysis resolves the body's read to the outer declaration and folds it, so the value the
    `eval` wrote is dropped. The root is not the parameter scope but the reach of a direct `eval`:
    `TestAParameterDefaultReadsPastTheBody` is about which scope a default reads from, and this one
    is about a scope whose contents no reading of the text gives.

    Off the release gate deliberately: no real file runs an `eval` in a parameter default to
    mint a binding, so the shape is this entry's own.
    """


#: A classic script reading one of its own top-level declarations through a name the file itself
#: installs on the global object, mapped to the behavior a host gives it. The installing statement is
#: what every row has in common: Node has no `window`, so a file meaning to read one has to put it
#: there, and a browser file that never installs one is already read correctly.
A_NAME_THE_FILE_INSTALLS_ON_THE_GLOBAL_OBJECT = {
    'read through the installed name': Program(
        a_program("""
            globalThis.window = globalThis;
            var q = 1;
            console.log(window.q);
            """),
        prints('1'),
        Reading.SCRIPT,
    ),
    'a guard reads the installed name': Program(
        a_program("""
            globalThis.window = globalThis;
            var q = function (a) { console.log('q', a); };
            var w = window || {};
            w.q(1);
            """),
        prints('q 1'),
        Reading.SCRIPT,
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
@one_expected_failure_per_program(A_NAME_THE_FILE_INSTALLS_ON_THE_GLOBAL_OBJECT)
class TestANameTheFileInstallsOnTheGlobalObjectHoldsIt(TestBase):
    """
    A name given the global object holds it, and a read through the name is a read of a global —
    the law `test.lib.scripts.js.deobfuscation.test_a_name_holding_the_global_object` states. It is
    answered from the value the name is declared with, and a name the file only ever assigns is
    declared with none: the binding minted for `globalThis.window = globalThis` carries no
    declaration, so both value queries decline for it, the name is taken for one bound to something
    other than the object, and every global read through it is recorded nowhere.

    Declining is deliberate and its reason is an ordering one. The value would have to come from the
    write, and the writes made through the global object are recorded by the same walk that would ask
    — so what a read is admitted on would be how far that walk had got, which is not a fact about
    the program. Answering needs the writes established before any read is admitted, which is a
    change to how the model is built rather than to what it knows.

    The second row is the shape that made this worth having: `var w = window || {}` is how a file
    meant for a browser and for something else names the object once. It is read correctly wherever
    `window` is the host's own name, and wrongly only where the file installs that name itself.

    Off the release gate deliberately: a browser file relies on the host's own `window`,
    which is read correctly, and installing the alias oneself is a shape only cross-host
    shims come near.
    """


#: A classic script handing the global object to a call from inside a function body, mapped to the
#: behavior a host gives it. Every row spells the object as the `this` of a function nothing calls as
#: a method, which §10.2.1.2 makes the global object for the duration of the call.
THE_GLOBAL_OBJECT_A_CALL_SUPPLIES_HANDED_ON = {
    'a write through it': Program(
        a_program("""
            var q = 1;
            function a(g, k) { g[k] = 2; }
            function f() { a(this, 'q'); }
            f();
            console.log(q);
            """),
        prints('2'),
        Reading.SCRIPT,
    ),
    'a read through it': Program(
        a_program("""
            var q = 1;
            function a(g, k) { console.log(g[k]); }
            function f() { a(this, 'q'); }
            f();
            """),
        prints('1'),
        Reading.SCRIPT,
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
@one_expected_failure_per_program(THE_GLOBAL_OBJECT_A_CALL_SUPPLIES_HANDED_ON)
class TestAReceiverACallSuppliesMayBeHandedOn(TestBase):
    """
    A function called with no receiver is given the global object as its `this`, and passing that on
    hands a second call every global the file declares. The references such a hand-over makes are
    recorded — `test.lib.scripts.js.deobfuscation.test_global_handed_to_a_call` states that law — but
    only for the spellings the text settles: an unshadowed alias, and the `this` a script's top level
    holds. Which receiver reaches a body is not decided anywhere, so a `this` written inside a
    function is not admitted, and a hand-over spelled with one is recorded nowhere.

    Admitting every `this` is measured, not assumed, and it is what this entry costs: obfuscator.io's
    self-defending wrapper passes its own `this` to a call, and a run that took that for the global
    object leaves `test_obfuscated_fizzbuzz_01` at twenty times its deobfuscated size. Closing this
    needs the receiver a call supplies, which nothing answers today.

    Off the release gate deliberately: the spellings real files hand the object on with — a
    UMD factory's top-level `this`, `(function (g) {...})(this)`, a `.call(this)`
    wrapper — are all top-level and admitted already, and a bare-called function
    forwarding its own `this` has so far had to be constructed.
    """


#: A program whose deobfuscation writes a global's name into a block where a `let` of the same name
#: shadows it, mapped to the behavior an engine gives it. The constant inliner substitutes an
#: initializer for a use of the name it was bound to, without asking whether the names the
#: initializer carries still resolve, where it lands, to what they resolved to where it was read.
A_SUBSTITUTED_NAME_A_BLOCK_SHADOWS_AT_ITS_USE = {
    'an initializer inlined into a shadowing block': Program(
        a_program("""
            function f() {
              var g = parseInt;
              {
                let parseInt = function (s) { return 99; };
                console.log(g('7'));
              }
            }
            f();
            """),
        prints('7'),
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
@one_expected_failure_per_program(A_SUBSTITUTED_NAME_A_BLOCK_SHADOWS_AT_ITS_USE)
class TestASubstitutedNameStillResolvesWhereItIsWritten(TestBase):
    """
    Substituting text for a name moves every name that text carries to a new position, and a name
    means what it resolves to where it stands. The program binds `parseInt` inside a block with
    `let` and calls the global one from that block through an alias bound outside it, so the call
    answers `7`. The deobfuscation writes `parseInt` into the block, where the shadow answers it,
    and the program comes back printing `99`.

    `refinery.lib.scripts.js.deobfuscation.constants` checks what the initializer's names held at
    the use (`_free_variables_preserved`) but not what its spelling resolves to there, a question
    its own cross-function inliner does ask before emitting an intrinsic alias's name. The rule that
    closes it: text may land only where every name it carries resolves to what it resolved to where
    the text was read. The wrapper inliner's twin of this defect is closed —
    `refinery.lib.scripts.js.deobfuscation.wrappers.JsCallWrapperInliner._forwarded_callee_reaches`
    refuses to graft a return expression whose forwarded callee a local at the call site would
    capture.

    Off the release gate deliberately: the shape needs a block-scoped `let` spelling the same name
    as a global that an alias bound outside the block still reaches, and that is not what an
    obfuscator does to names — it renames them apart, never toward a collision.
    """


#: A reflective-inlining pass retiring a single-use `Function`-constructor temporary whose only read
#: the pinned model counted was the invocation the pass folded, while a direct `eval` inlined in the
#: same pass splices in a second reference to the same temporary. The row is mapped to the behavior
#: Node gives the original, which the retirement is meant to preserve and does not.
A_RETIRED_TEMPORARY_A_SAME_PASS_EVAL_STILL_NAMES = {
    'an eval names the temporary the fold retired': Program(
        a_program("""
            const m = Function('return 41');
            console.log(m());
            eval('console.log(m.name)');
            """),
        prints('41', 'anonymous'),
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
@one_expected_failure_per_program(A_RETIRED_TEMPORARY_A_SAME_PASS_EVAL_STILL_NAMES)
class TestARetiredTemporaryIsStillNamedByASamePassEval(TestBase):
    """
    The reflective-inlining pass holds one model pinned for its whole run and retires a single-use
    temporary — the local whose sole value is a `Function` construction — once every read the pinned
    model shows was an invocation the pass inlined. A direct `eval` the same pass inlines splices in
    code the pinned model never read, and where that code names the temporary the retirement counts
    a live reference as none and drops the declaration out from under it.

    Node runs `m()` and prints `41`, then reads `m.name` through the eval and prints `anonymous`,
    the name a `Function` construction gives its function. The deobfuscation folds `m()` to `41`,
    retires `const m`, and the spliced `console.log(m.name)` is left reading an undeclared `m`, so
    the program comes back printing `41` and then throwing a `ReferenceError`.

    `refinery.lib.scripts.js.deobfuscation.reflection.JsReflectionInlining._retire_consumed_temporaries`
    compares the reads it consumed to `len(binding.reads)` taken from the model pinned before any
    inlining, which the same pass's eval splice is invisible to. The pin's soundness argument covers
    a fold against an intrinsic, which a live reflection surface withdraws trust from; it does not
    cover the retirement read-count, which is a structural fact the splice changes. The rule that
    closes it re-derives the reads against the post-inline tree, or leaves a temporary a same-pass
    eval could reach un-retired.

    Off the release gate deliberately: the shape needs a direct `eval` whose string names the very
    reflective temporary the fold retires, and that collision is this entry's own construction.
    """


#: A reflective-inlining pass folding a separated `Function`-constructor temporary's invocation
#: through the one value the pinned model holds for it, while a direct `eval` inlined in the same
#: pass reassigns that temporary. The row is mapped to the behavior Node gives the original.
A_REASSIGNED_TEMPORARY_A_SAME_PASS_EVAL_REBINDS = {
    'an eval rebinds the temporary before the fold reads it': Program(
        a_program("""
            var m = Function('return 1');
            eval('m = function () { return 2; };');
            console.log(m());
            """),
        prints('2'),
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
@one_expected_failure_per_program(A_REASSIGNED_TEMPORARY_A_SAME_PASS_EVAL_REBINDS)
class TestAReassignedTemporaryIsFoldedFromItsStaleValue(TestBase):
    """
    The pass resolves a separated `Function`-constructor temporary to the single value the pinned
    model holds for it and folds its invocation to what that construction returns. A direct `eval`
    the same pass inlines can reassign the temporary, and the pinned model, taken before the splice,
    still holds only the construction, so the fold reads a value the reassignment has replaced.

    Node reassigns `m` through the eval to a function returning `2` and prints `2`. The deobfuscation
    folds `m()` to `1`, the value of the `Function` construction the pinned model still holds, and
    the program comes back printing `1`.

    `refinery.lib.scripts.js.deobfuscation.reflection.JsReflectionInlining._resolved_constructor_call`
    resolves the temporary with `singular_value` on the model pinned before inlining, the same
    pinned-model-versus-splice root the retirement entry carries. The rule that closes it re-derives
    the value against the post-inline tree, or declines the fold for a temporary a same-pass eval
    could rebind.

    Off the release gate deliberately: the shape needs a direct `eval` reassigning the very
    reflective temporary the fold reads, and that is this entry's own construction.
    """


#: A classic script handing the global object as the receiver of an `apply` whose target is a
#: parameter the body reassigns to a function that reads its own `this`. The row is mapped to the
#: behavior a host gives it: the reassigned target reads the global through the handed receiver.
A_HANDED_GLOBAL_AN_APPLY_TARGET_REASSIGNMENT_READS = {
    'a reassigned apply target reads the handed global': Program(
        a_program("""
            var secret = 'S';
            function inner(host) { return host.secret; }
            function wrap(recv, payload) {
              payload = function () { return inner(this); };
              return payload.apply(recv, []);
            }
            console.log(wrap(this, function () {}));
            """),
        prints('S'),
        Reading.SCRIPT,
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
@one_expected_failure_per_program(A_HANDED_GLOBAL_AN_APPLY_TARGET_REASSIGNMENT_READS)
class TestAReassignedApplyTargetStillReadsTheHandedGlobal(TestBase):
    """
    The gate that keeps a global foldable decides a receiver handed to `apply`/`call` is unread when
    its target provably does not read its own `this`. Where that target is another parameter of the
    same call, the target is resolved through the parameter's original call-site argument, and a
    parameter the body reassigns is judged on that original alone. A parameter reassigned to a
    function that reads `this` is therefore taken for a this-free one, the handed global is called
    unobserved, and the global it carries is unfrozen and deleted.

    Under the script model the top-level `this` is the global object, so `inner(this)` reads the
    global `secret` and Node prints `S`. The deobfuscation deletes `var secret`, and the read yields
    `undefined`, so the program comes back printing `undefined`.

    `refinery.lib.scripts.js.analysis.model.SemanticModel._apply_receiver_is_safe` trusts the
    argument in its parameter map without inspecting the reassignments the body makes to that
    parameter. The rule that closes it takes the target from the set of values the parameter can
    hold — its argument and every value assigned to it — and calls the receiver safe only when every
    one of them is a function that does not read `this`. A blanket bail on any write instead would
    refuse the obfuscator's own `payload = null`, regressing the self-defending fold.

    Off the release gate deliberately: the shape needs an `apply` target parameter reassigned to a
    this-reader with the global handed as its receiver, and that is this entry's own construction.
    """


#: A classic script whose run-once wrapper carries the three shapes the self-defending detector keys
#: on — an empty-alternate conditional, a `payload.apply(recv, ...)`, and `payload = null` — for
#: reasons of its own, handed the global object as its receiver. The row is mapped to the behavior a
#: host gives it: the wrapper runs its payload once.
A_BENIGN_RUN_ONCE_WRAPPER_THE_DETECTOR_MATCHES = {
    'a once wrapper handed the global runs its payload': Program(
        a_program("""
            var once = function (context, fn) {
              var run = fn
                ? function () { var r = fn.apply(context, arguments); fn = null; return r; }
                : function () {};
              return run;
            };
            var boot = once(this, function () { console.log('ran'); });
            boot();
            """),
        prints('ran'),
        Reading.SCRIPT,
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
@one_expected_failure_per_program(A_BENIGN_RUN_ONCE_WRAPPER_THE_DETECTOR_MATCHES)
class TestABenignRunOnceWrapperIsNotSelfDefending(TestBase):
    """
    The structural self-defending detector matches a factory by three shapes it gathers one at a
    time from anywhere in the body: a conditional whose alternate is an empty function, a
    `payload.apply(recv, ...)`, and a `payload = null`. A run-once wrapper — the shape a `once` or
    memoize utility takes — spells all three for reasons of its own, and handed the global object as
    its receiver it matches the template, so the factory, the stored guard, and the payload are all
    removed and the call the wrapper stood in front of never runs.

    Under the script model the top-level `this` is the global object, so `once(this, ...)` is a
    hand-over the detector reads, and Node runs the payload and prints `ran`. The deobfuscation
    deletes the whole construction and the program comes back printing nothing.

    `refinery.lib.scripts.js.deobfuscation.antidbg._matches_self_defending_factory` does not
    correlate its three flags to one run-once branch, nor require that the function the factory
    returns is invoked only for its effect. The rule that closes it ties the flags to the same
    branch and leaves a wrapper whose result a program reads standing.

    Off the release gate deliberately: the shape needs a benign wrapper that spells all three
    template shapes and is handed the global object as its receiver, which a real `once` utility,
    taking a specific context rather than the global, does not come near.
    """
