"""
The JavaScript defects a release is held for.

Same form as `test.lib.scripts.js.test_unfixed_defects`, which these entries were separated out of,
and the same rules: every test states what a correct implementation would do, never what the code
does today, and is marked `unittest.expectedFailure`, so an entry that starts passing is reported as
an unexpected success and leaves this file only by being fixed. Where the question is one about
JavaScript rather than about this project, the answer was established with Node.js and is quoted in
the docstring of the test that pins it.

What sets these apart is what they cost rather than what they are. Each one hands back a file that
looks clean and is wrong: nothing throws, nothing is left half-rewritten, and the analyst reading it
gets no signal that the answer is not the one the language gives. An entry that merely refuses to
reduce something, or reduces it to something uglier, belongs in the other file. This file emptying
is the release gate.
"""
from __future__ import annotations

import unittest

from test import TestBase
from test.lib.scripts.js.analysis.differential import behavior, node_executable
from test.lib.scripts.js.deobfuscation.test_dispatcher import a_dispatcher
from test.lib.scripts.js.ledger import (
    before_and_after,
    each_program_still_prints,
    evaluated_in_a_body,
    folded,
    printed,
    returned_from_a_body,
    well_formed,
)
from test.lib.scripts.js.test_truncated_source import FOLDS_ANSWERED_WITH_A_PROGRAM

from refinery.lib.scripts import UnspellableNode


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
            (folded(program(identifier_key)), folded(program(literal_key))),
            (reads_as_the_two_surrogates, reads_as_the_two_surrogates),
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


_ASTRAL_LETTER = chr(0x1D465)


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
        returned_from_a_body("return Object.keys({ ESCAPED_A: 1, b: 2 }).join('|');")
    ): 'a|b\n',
    _spelled_with_an_escaped_identifier(
        returned_from_a_body('return { ESCAPED_A: 7 }.a;')
    ): '7\n',
    _spelled_with_an_escaped_identifier(
        returned_from_a_body('var o = { a: 1 }; return o.ESCAPED_A;')
    ): '1\n',
    _spelled_with_an_escaped_identifier(
        returned_from_a_body("return 'q' in { ESCAPED_Q: 1 };")
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
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
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
        self.assertEqual({source: folded(source) for source in rows}, rows)


def _a_module_reporting_it_loaded(module: str) -> str:
    """
    *module* with a statement printing `loaded` appended. A module that only declares and exports
    does nothing an engine reports, and the print is what makes the difference between a module that
    loads and one that does not an answer rather than a silence on both sides.
    """
    return F"{module}\nconsole.log('loaded');\n"


#: A module that exports a binding it declares, mapped to what Node prints for it. Every one of them
#: is a module an engine loads, which is the whole of what an export list has to be checked against:
#: a list compiles only where the module declares what it names.
A_MODULE_THAT_EXPORTS_A_BINDING_IT_DECLARES = {
    _a_module_reporting_it_loaded('var a = 1;\nexport { a };'): 'loaded\n',
    _a_module_reporting_it_loaded('let a = 1;\nexport { a };'): 'loaded\n',
    _a_module_reporting_it_loaded('const a = 1;\nexport { a };'): 'loaded\n',
    _a_module_reporting_it_loaded('const a = () => 1;\nexport { a };'): 'loaded\n',
    _a_module_reporting_it_loaded('let a;\nexport { a };'): 'loaded\n',
    _a_module_reporting_it_loaded('var a = 1;\nexport { a as b };'): 'loaded\n',
    _a_module_reporting_it_loaded('var a = 1, b = 2;\nexport { a, b };'): 'loaded\n',
    _a_module_reporting_it_loaded('var a = 1;\nexport { a as default };'): 'loaded\n',
    'var a = 1;\nexport { a };\nconsole.log(a);\n': '1\n',
    _a_module_reporting_it_loaded('function a() { return 1; }\nexport { a };'): 'loaded\n',
    _a_module_reporting_it_loaded('class a {}\nexport { a };'): 'loaded\n',
    _a_module_reporting_it_loaded(
        "import { format } from 'node:util';\nexport { format };"
    ): 'loaded\n',
    _a_module_reporting_it_loaded(
        "var format = 1;\nexport { format } from 'node:util';"
    ): 'loaded\n',
    _a_module_reporting_it_loaded('export var a = 1;'): 'loaded\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAnExportListReadsTheBindingItNames(TestBase):
    """
    A module can declare a binding and export it in two statements, the declaration and an export
    list naming it, and the name in that list is a read of that binding. `export { a };` is a module
    only where `a` is declared, and Node refuses one where it is not with `SyntaxError: Export 'a'
    is not defined in module`. That refusal is the linker's rather than the parser's, so it is what
    whoever imports the module gets as much as whoever runs it.

    A declaration whose only reader is such a list is read as unread and deleted, and the list is
    left standing over a name the module no longer declares. What comes back reads as the file it
    came from with one dead statement taken out of it: nothing throws while the tool runs, nothing
    is left half-rewritten, and the module is one no engine will load.

    Which declarations this reaches was measured. A variable declaration goes whatever keyword wrote
    it, with an initializer or without, whether the list renames what it exports, exports it as
    `default`, or names two bindings at once, and whether or not the module reads it somewhere else:
    a read a constant is substituted into is a read the declaration is no longer kept for.

    The last five rows are the controls a fix has to keep. Three of them export a name the module
    declares no local for — an export list carrying a `from` clause names a binding on the far side
    of the module boundary, a list re-exporting an import names what the import statement bound, and
    an `export var` is one statement and not two — and the other two are a function declaration
    and a class declaration, the latter of which is kept here whether or not any list names it.

    Reading every name in every export list as a local read is not a fix those controls survive: the
    `from` row declares a local under the name the list exports and the list does not name that
    local, so a constant substituted into it writes `export { 1 } from 'node:util'`, which Node
    refuses as readily as it refuses the export of a name nothing declared.
    """

    @unittest.expectedFailure
    def test_a_declaration_an_export_list_names_is_read_by_that_list(self):
        """
        Node prints `loaded` for thirteen of the fourteen modules of
        `A_MODULE_THAT_EXPORTS_A_BINDING_IT_DECLARES` and `1` for the one that reads what it
        exports. Nine of them are handed back as a module no engine loads, printing nothing and
        answering `SyntaxError` to anyone who asks for them.
        """
        rows = A_MODULE_THAT_EXPORTS_A_BINDING_IT_DECLARES
        self.assertEqual(
            {source: before_and_after(source, module=True) for source in rows},
            each_program_still_prints(rows),
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


#: A program that installs a value at the index a dispatcher's payload was written with no element
#: in, and dispatches through it, mapped to what Node prints for it. The dispatcher reaches its
#: callee's arguments by reading them off the payload, so a position the payload does not hold is
#: read from the prototype chain like any other.
A_DISPATCH_WHOSE_PAYLOAD_HOLDS_A_HOLE = {
    "Array.prototype[1] = 'X';\n" + a_dispatcher(
        dict_lines=['"f1": function() { var [a, b, c] = p; return a + b + c; }'],
        tail_lines=['console.log((p = ["first", , "third"], d("f1")));'],
    ): 'firstXthird\n',
}


#: A program that removes a name the language puts on the receiver's prototype chain and then asks
#: whether that name is `in` the receiver, mapped to what Node prints for it. Nothing else touches
#: the chain, so the deletion is the whole of what decides each answer.
A_MEMBERSHIP_TEST_FOR_A_KEY_THE_PROGRAM_DELETED = {
    evaluated_in_a_body(
        '{a: 1}',
        "'toString' in v",
        'delete Object.prototype.toString;',
    ): 'false\n',
    evaluated_in_a_body(
        '{a: 1}',
        "'hasOwnProperty' in v",
        'delete Object.prototype.hasOwnProperty;',
    ): 'false\n',
    evaluated_in_a_body(
        '[1, 2]',
        "'join' in v",
        'delete Array.prototype.join;',
    ): 'false\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestADispatcherPayloadHoleIsTheChainsToAnswer(TestBase):
    """
    Unwrapping a dispatcher into direct calls has to spell out each argument the payload carried,
    and a position the payload was written with no element in is not one it carries: what reaches
    the callee there is whatever the prototype chain answers for that index. Spelling it `undefined`
    is right only while that chain is the one the language describes, and the rewrite asks nothing
    about it.

    This is the law `test.lib.scripts.js.deobfuscation.test_inherited_reads` states of a read, at a
    site that decides it in the syntax rather than in the interpreter. The same program with the
    prototype left alone is `test_dispatcher_sparse_payload_preserves_arity` and prints
    `firstundefinedthird`, which is what a fix has to keep.
    """

    @unittest.expectedFailure
    def test_a_payload_position_no_element_was_written_in_finds_what_the_chain_holds(self):
        """
        Node prints `firstXthird` for the one program of `A_DISPATCH_WHOSE_PAYLOAD_HOLDS_A_HOLE`,
        which writes `X` at index 1 of `Array.prototype` before dispatching a payload whose second
        position holds no element. The deobfuscation prints `firstundefinedthird`, the answer the
        same dispatch has with the prototype untouched.
        """
        rows = A_DISPATCH_WHOSE_PAYLOAD_HOLDS_A_HOLE
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAKeyTheProgramDeletedIsInNothing(TestBase):
    """
    A name is reported present on a receiver's prototype chain from a table of what the language
    puts there, with nothing asked about what the program did to that chain. The reasoning is that
    a write to a prototype only ever adds a name, so a name the language already put there is there
    afterwards too — which is true of a write and false of a `delete`.

    The read side of the same programs is answered correctly, because absence is asked of the effect
    model and the deletion is recorded there as a write to the prototype's owner. So the model
    already knows; it is the presence side that does not ask, and telling a deletion from a write
    is what closes this.
    """

    @unittest.expectedFailure
    def test_a_key_the_program_deleted_is_in_nothing(self):
        """
        Node prints `false` for all three programs of
        `A_MEMBERSHIP_TEST_FOR_A_KEY_THE_PROGRAM_DELETED`, each of which deletes one name from a
        prototype its receiver inherits from and then asks for that same name. Each deobfuscation
        prints `true`, the answer the same question has with nothing deleted.
        """
        rows = A_MEMBERSHIP_TEST_FOR_A_KEY_THE_PROGRAM_DELETED
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )
