"""
A ledger of JavaScript defects that are known, understood, and not yet fixed.

The ones a release is held for are not here: `test.lib.scripts.js.test_release_blockers` holds
those, under the same rules, so that the question of whether the tool is fit to ship has one file
for an answer. An entry belongs there rather than here when what it hands back looks clean and is
wrong; one that refuses to reduce something, or reduces it to something uglier, belongs here. Which
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

import unittest

from collections import Counter

from test import TestBase
from test.lib.scripts.js.analysis.differential import (
    behavior,
    deobfuscate_source,
    node_executable,
)
from test.lib.scripts.js.deobfuscation.test_array_length_reads import (
    A_COUNT_THE_FOLD_DOES_NOT_REACH,
)
from test.lib.scripts.js.deobfuscation.test_dispatcher import a_dispatcher
from test.lib.scripts.js.ledger import (
    before_and_after,
    each_program_still_prints,
    folded,
    printed,
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

from refinery.lib.scripts.js.model import (
    JsBigIntLiteral,
    JsExportAllDeclaration,
    JsExportSpecifier,
    JsIdentifier,
    JsImportSpecifier,
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
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )


def _the_module_export_name(source: str) -> str | None:
    """
    The name the one specifier of *source* reaches across the module boundary with, which is the
    text denoted by the string literal written in that position. A position holding anything other
    than a string literal denotes no text at all, and nothing is what that is reported as.
    """
    for node in JsParser(source).parse().walk_in_order():
        if isinstance(node, JsImportSpecifier):
            written = node.imported
        elif isinstance(node, (JsExportSpecifier, JsExportAllDeclaration)):
            written = node.exported
        else:
            continue
        return written.value if isinstance(written, JsStringLiteral) else None
    return None


class TestAModuleExportNameWrittenAsAStringIsAString(TestBase):
    """
    An import or export list may name what it reads or writes across the module boundary with a
    string literal instead of a word, which is how a module reaches a name no identifier spells.
    What such a specifier names is the text the literal denotes: the quotes are how it was written
    and are no part of it, and an escape in it stands for the character it denotes.

    The text of every one of these declarations prints back exactly as it was written, and that is
    not what is wrong. What is wrong is the tree behind it. The literal is read as though it were a
    word and kept as a `refinery.lib.scripts.js.model.JsIdentifier` whose name is the raw spelling,
    quotes and all, so a consumer reading that name reads a name no module has, two spellings of
    the one name are two names, and the rules the grammar attaches to a name written as a string
    are rules the parser is never in a position to apply.

    Every answer below is Node's over a `.mjs` file, a module being the only kind of code these
    declarations appear in: `node --check` for what it refuses, and running the file for what it
    prints.
    """

    @unittest.expectedFailure
    def test_a_name_written_as_a_string_is_the_text_that_string_denotes(self):
        """
        A module re-exporting `{ a as 'b c' }` and one importing `{ 'b c' as v }` from it print `1`
        together, and so do the same two with the `b` of the importing module's name written as a
        `u` escape: the name a boundary matches on is the text the literal denotes and not the way
        that text was spelled. An importing module asking for the name `"'b c'"` instead is refused
        with `SyntaxError: The requested module './mid.mjs' does not provide an export named
        ''b c''`, which is those quotes being taken for part of the name.
        """
        escaped = F'{chr(92)}u0062 c'
        sources = [
            "export { a as 'b c' } from 'm';",
            F"export {{ a as '{escaped}' }} from 'm';",
            "import { 'b c' as v } from 'm';",
            "export * as 'b c' from 'm';",
            "var a = 1; export { a as 'b c' };",
        ]
        self.assertEqual(
            [_the_module_export_name(source) for source in sources],
            ['b c'] * len(sources),
        )

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


#: A program that dispatches through a callee name built by an expression rather than written as a
#: string literal, mapped to what Node prints for it. The dispatcher is reached the ordinary way and
#: answers the ordinary way; only the spelling of the name keeps the unwrapper from rewriting it.
A_DISPATCH_THROUGH_A_COMPUTED_KEY = {
    a_dispatcher(
        dict_lines=['"f1": function() { var [a, b, c] = p; return a + b + c; }'],
        tail_lines=[
            'var k = "f" + 1;',
            'console.log((p = ["a", "b", "c"], d(k)));',
        ],
    ): 'abc\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestADispatcherIsKeptWhileACallToItSurvives(TestBase):
    """
    Unwrapping a dispatcher replaces each call made through it with a direct call to the function
    that call selects, and then removes the dispatcher declaration. The replacement is attempted per
    call site and declines the ones it cannot read, while the removal runs whatever it left behind,
    so a file holding one such call comes back calling a function nothing declares.

    A callee named by an expression is the shape that shows it, since the unwrapper needs a string
    literal to know which function a call selects. The failure is the removal's rather than that
    refusal's: what the unwrapper cannot rewrite it is right to leave alone, and what it leaves
    alone still needs the declaration it reads.
    """

    @unittest.expectedFailure
    def test_a_dispatch_through_a_computed_key_keeps_the_dispatcher(self):
        """
        Node prints `abc` for the one program of `A_DISPATCH_THROUGH_A_COMPUTED_KEY`, which spells
        its callee name `"f" + 1`. The deobfuscation prints nothing and throws a `ReferenceError`:
        the dispatcher declaration is gone and the call to it is not.
        """
        rows = A_DISPATCH_THROUGH_A_COMPUTED_KEY
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )
