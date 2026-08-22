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
from test.lib.scripts.js.deobfuscation.test_escaped_identifiers import (
    AN_ESCAPED_ACCESSOR_TERMINAL,
    AN_ESCAPED_ASYNC_TERMINAL,
    AN_ESCAPED_KEYWORD_OPERATOR,
    AN_ESCAPED_STATIC_TERMINAL,
)
from test.lib.scripts.js.ledger import (
    an_accessor_at,
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


#: Programs writing a name onto `Object.prototype` and then reading that name off an object
#: literal that does not hold it, mapped to what Node prints for each. The four differ in how
#: the write reaches the prototype and in what it puts there: directly, through a name the file
#: bound it to, through `Object.defineProperty`, and as an accessor rather than a value, so that
#: the last of them answers the read by running the program's own code. The last also holds a
#: second key the literal does write, which the flattening keeps and is right to keep.
A_NAMESPACE_KEY_THE_CHAIN_ANSWERS = {
    'Object.prototype.z = 9; var o = {}; console.log(o.z);':
        '9\n',
    'var P = Object.prototype; P.z = 9; var o = {}; console.log(o.z);':
        '9\n',
    "Object.defineProperty(Object.prototype, 'z', {value: 9});"
    ' var o = {}; console.log(o.z);':
        '9\n',
    an_accessor_at('Object.prototype', 'z')
    + ' var o = {}; o.k = 1; console.log(o.k, o.z);':
        '1 G\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestANamespaceKeyTheChainAnswersIsNotAVariable(TestBase):
    """
    An object literal a file only ever reads properties off is flattened into plain variables, one
    per key, and the declaration it was built by is dropped. That is the same program only while a
    key the literal does not hold reads back `undefined`, and a write to `Object.prototype` is what
    makes one read back something else: the flattened variable answers nothing where the property
    answered what the chain holds.

    `refinery.lib.scripts.js.deobfuscation.helpers.property_provably_absent` is the question this
    is deciding and does not ask. The pass holds back a key the language itself puts on every
    object, which is the yes-side of it, and takes every other key as absent whatever the file did
    to the chain.

    `test_a_key_the_namespace_holds_is_flattened` is the other side of it and passes: where the
    literal owns the key, the chain never answers and the flattening is right.
    """

    @unittest.expectedFailure
    def test_a_key_the_chain_answers_is_not_flattened(self):
        """
        Node prints `9`, `9`, `9` and `1 G` for the four programs of
        `A_NAMESPACE_KEY_THE_CHAIN_ANSWERS`, each of which reads a key off an object literal that
        does not hold it after putting that key on `Object.prototype`. Each deobfuscation answers
        `undefined` in that key's place, and the accessor row runs no getter at all.
        """
        rows = A_NAMESPACE_KEY_THE_CHAIN_ANSWERS
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    def test_a_key_the_namespace_holds_is_flattened(self):
        """
        Node prints `1` for a program that writes the same key onto the literal before reading it,
        with the same write to `Object.prototype` standing in front of it. The own property is what
        answers there, so nothing about the chain reaches the read and the flattening keeps it.
        """
        source = 'Object.prototype.z = 9; var o = {}; o.z = 1; console.log(o.z);'
        self.assertEqual(before_and_after(source), (('1\n', None), ('1\n', None)))


#: Programs that remove a name the language puts on a plain object's chain and then ask whether that
#: name is `in` an object literal that never held it, mapped to what Node prints for each. The
#: question is asked at the top level, where `simplify` answers it, rather than inside a body where
#: the interpreter does, so these are the same defect as
#: `A_MEMBERSHIP_TEST_FOR_A_KEY_THE_PROGRAM_DELETED` at the other of the two sites that decide it.
A_TOP_LEVEL_MEMBERSHIP_TEST_FOR_A_KEY_THE_PROGRAM_DELETED = {
    "delete Object.prototype.toString; var o = {}; console.log('toString' in o);":
        'false\n',
    "delete Object.prototype.hasOwnProperty; var o = {}; console.log('hasOwnProperty' in o);":
        'false\n',
}


#: Membership tests whose answer the language alone decides, mapped to the text each comes back as
#: and what Node prints for it. Two ask a chain nothing touched, and two put a write to a prototype
#: no plain object inherits from in front of the same question. All four fold today and a fix has to
#: keep folding them: an entry that only pins a wrong answer is kept green by a pass that stops
#: answering at all.
A_MEMBERSHIP_TEST_THE_CHAIN_STILL_ANSWERS = {
    "var o = {}; console.log('toString' in o);": (
        'console.log(true);',
        'true\n',
    ),
    "var o = {}; console.log('zz' in o);": (
        'console.log(false);',
        'false\n',
    ),
    "delete Array.prototype.join; var o = {}; console.log('toString' in o);": (
        'delete Array.prototype.join;\nconsole.log(true);',
        'true\n',
    ),
    "delete Array.prototype.join; var o = {}; console.log('zz' in o);": (
        'delete Array.prototype.join;\nconsole.log(false);',
        'false\n',
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestATopLevelKeyTheProgramDeletedIsInNothing(TestBase):
    """
    `TestAKeyTheProgramDeletedIsInNothing` at the other site. The two `in` operators are answered by
    different code — a question asked inside a function body is the interpreter's and one asked at
    the top level is `simplify`'s — and each reads the same table of what the language puts on a
    chain without asking the model what the file did to it.

    `refinery.lib.scripts.js.deobfuscation.simplify.JsSimplifier._resolve_in` says as much: it asks
    `read_chain_intact` before answering `False` and not before answering `True`, on the reasoning
    that a name the tables list is there whatever the file wrote. A `delete` is what that reasoning
    does not cover.
    """

    @unittest.expectedFailure
    def test_a_key_the_program_deleted_is_in_nothing(self):
        """
        Node prints `false` for both programs of
        `A_TOP_LEVEL_MEMBERSHIP_TEST_FOR_A_KEY_THE_PROGRAM_DELETED`. Each deobfuscation folds the
        membership test to `true`, the answer it has with nothing deleted.
        """
        rows = A_TOP_LEVEL_MEMBERSHIP_TEST_FOR_A_KEY_THE_PROGRAM_DELETED
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    def test_a_membership_test_the_chain_answers_is_still_folded(self):
        """
        Node prints `true`, `false`, `true` and `false` for the four programs of
        `A_MEMBERSHIP_TEST_THE_CHAIN_STILL_ANSWERS`, and each deobfuscation comes back with the
        answer folded in place of the test. Writing `Array.prototype` reaches no chain a plain
        object has, so the two rows that do it are answered as if nothing had been written, and the
        `delete` itself stays because it is what puts the file in that state.
        """
        rows = A_MEMBERSHIP_TEST_THE_CHAIN_STILL_ANSWERS
        self.assertEqual(
            {source: (folded(source), before_and_after(source)) for source in rows},
            {
                source: (reduced, ((prints, None), (prints, None)))
                for source, (reduced, prints) in rows.items()
            },
        )


def _a_walk_of(receiver: str, installs: str = '') -> str:
    """
    A script that runs *installs* and then prints the names a `for-in` over *receiver* reaches, in
    the order it reaches them. The walk is written inside a function body, which is what puts it
    where the tool answers it at all: the same loop at the top level is left standing.

    The names are joined by appending to a string rather than through `Array.prototype.join`, so
    that a row installing something on `Array.prototype` asks only about the walk. Called through
    `join`, such a row comes back unreduced because the call cannot fold, and would report the walk
    as refused wherever it was in fact answered.
    """
    walk = F"var t = ''; for (var k in {receiver}) t += k; return t;"
    body = returned_from_a_body(walk)
    return F'{installs}\n{body}' if installs else body


#: Programs that put an enumerable name on `Object.prototype` and then walk a receiver with
#: `for-in`, mapped to what Node prints for each. The four differ in how the write reaches the
#: prototype and in what the walk runs over: directly, through a name the file bound it to, through
#: `Object.defineProperty` asking for an enumerable property, and over an array rather than an
#: object literal. Every plain object and every array inherits from `Object.prototype`, so the added
#: name is reached from all of them.
A_FOR_IN_WALK_REACHING_A_NAME_THE_CHAIN_WAS_GIVEN = {
    _a_walk_of('{a: 1}', 'Object.prototype.z = 9;'):
        'az\n',
    _a_walk_of('{a: 1}', 'var P = Object.prototype; P.z = 9;'):
        'az\n',
    _a_walk_of(
        '{a: 1}',
        'Object.defineProperty(Object.prototype, "z", {value: 9, enumerable: true});',
    ):
        'az\n',
    _a_walk_of('[7, 8]', 'Object.prototype.z = 9;'):
        '01z\n',
}


#: Walks whose names the language alone decides, mapped to the text each comes back as and what Node
#: prints for it. Two run over a chain nothing touched and two put a write to `Array.prototype` in
#: front of a walk over an object literal, which inherits from `Object.prototype` and not from that
#: one. All four fold today and a fix has to keep folding them.
A_FOR_IN_WALK_THE_CHAIN_STILL_ANSWERS = {
    _a_walk_of('{a: 1, b: 2}'): (
        "console.log('ab');",
        'ab\n',
    ),
    _a_walk_of('[7, 8]'): (
        "console.log('01');",
        '01\n',
    ),
    _a_walk_of('{a: 1, b: 2}', 'Array.prototype.z = 9;'): (
        "Array.prototype.z = 9;\nconsole.log('ab');",
        'ab\n',
    ),
    _a_walk_of('{a: 1, b: 2}', 'delete Array.prototype.join;'): (
        "delete Array.prototype.join;\nconsole.log('ab');",
        'ab\n',
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAForInWalkReachesTheNamesTheChainWasGiven(TestBase):
    """
    A `for-in` walk visits every enumerable name the receiver's whole prototype chain holds, not the
    own properties alone, and a name a file writes onto a prototype with an ordinary assignment is
    enumerable. The interpreter walks the own properties and stops, so a program written to carry a
    name on the chain comes back having visited one name fewer, with nothing to say it did.

    Everything the language itself puts on a chain is non-enumerable, which is why this is only ever
    wrong for a name the file put there — and why the whole question is one about what the
    file did, which the effect model is what knows.
    """

    @unittest.expectedFailure
    def test_a_name_written_onto_the_chain_is_walked(self):
        """
        Node prints `az`, `az`, `az` and `01z` for the four programs of
        `A_FOR_IN_WALK_REACHING_A_NAME_THE_CHAIN_WAS_GIVEN`. Each deobfuscation prints the walk with
        the inherited name missing, which is what the same walk answers with the chain untouched.
        """
        rows = A_FOR_IN_WALK_REACHING_A_NAME_THE_CHAIN_WAS_GIVEN
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    def test_a_walk_the_chain_answers_is_still_folded(self):
        """
        Node prints `ab`, `01`, `ab` and `ab` for the four programs of
        `A_FOR_IN_WALK_THE_CHAIN_STILL_ANSWERS`, and each deobfuscation comes back with the names
        folded in place of the loop. An object literal does not inherit from `Array.prototype`, so
        the two rows writing that one are answered as if nothing had been written.
        """
        rows = A_FOR_IN_WALK_THE_CHAIN_STILL_ANSWERS
        self.assertEqual(
            {source: (folded(source), before_and_after(source)) for source in rows},
            {
                source: (reduced, ((prints, None), (prints, None)))
                for source, (reduced, prints) in rows.items()
            },
        )


#: Programs that write a name onto `Object.prototype` without ever spelling `Object`, mapped to what
#: Node prints for each. Every one reaches the same object a literal `Object.prototype` names:
#: as the `__proto__` of an object literal, as what `Object.getPrototypeOf` answers for one, and
#: as the `prototype` of what its `constructor` is. These are the shapes a file uses when it does
#: not want the write found by looking for the name.
A_PROTOTYPE_WRITTEN_WITHOUT_NAMING_ITS_OWNER = {
    '({}).__proto__.z = 9; var o = {}; console.log(o.z);':
        '9\n',
    'Object.getPrototypeOf({}).z = 9; var o = {}; console.log(o.z);':
        '9\n',
    '({}).constructor.prototype.z = 9; var o = {}; console.log(o.z);':
        '9\n',
}


#: Programs whose `__proto__` reaches a prototype other than the one a plain object inherits
#: from, or reaches nothing the file may assume, mapped to what Node prints for each. An array
#: literal's `__proto__` is `Array.prototype`, which a plain object does not inherit from, and a
#: file that binds the name `Object` to something of its own is not writing the intrinsic at all.
A_PROTOTYPE_SPELLING_THAT_NAMES_A_DIFFERENT_OBJECT = {
    '[].__proto__.z = 9; var o = {}; console.log(o.z);':
        'undefined\n',
    '[].__proto__.z = 9; var a = [1]; console.log(a.z);':
        '9\n',
    'var Object = {prototype: {}}; Object.prototype.z = 9; console.log(({}).z);':
        'undefined\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAPrototypeWrittenWithoutNamingItsOwnerIsStillWritten(TestBase):
    """
    A write to a prototype is recorded against the name at the root of the member chain it was
    written through, so a chain rooted in something other than an identifier is recorded against
    nothing and every question about that prototype goes on answering as if the file had left it
    alone. Reaching `Object.prototype` through an object literal is the whole of what a prototype
    pollution gadget does, and all three spellings here are ways of writing it without the name.

    The read is then flattened into a variable that answers `undefined`, so the value the write put
    there is gone from the file as well as from the answer.
    """

    @unittest.expectedFailure
    def test_a_prototype_written_through_a_literal_answers_the_read(self):
        """
        Node prints `9` for all three programs of `A_PROTOTYPE_WRITTEN_WITHOUT_NAMING_ITS_OWNER`,
        each of which reaches `Object.prototype` without spelling `Object`. Each deobfuscation
        prints `undefined`, and comes back having replaced the read with a variable nothing writes.
        """
        rows = A_PROTOTYPE_WRITTEN_WITHOUT_NAMING_ITS_OWNER
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    def test_a_spelling_that_names_a_different_object_is_read_as_that_object(self):
        """
        Node prints `undefined`, `9` and `undefined` for the three programs of
        `A_PROTOTYPE_SPELLING_THAT_NAMES_A_DIFFERENT_OBJECT`, and each deobfuscation agrees. An
        array literal's `__proto__` is `Array.prototype`, and a file that declares `Object` itself
        names its own object with it.
        """
        rows = A_PROTOTYPE_SPELLING_THAT_NAMES_A_DIFFERENT_OBJECT
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )


#: Programs that reach `Object.prototype` by handing it to something that writes it, rather than by
#: writing through a member chain, mapped to what Node prints for each. The prototype is an ordinary
#: argument in each: to `Object.assign`, to `Reflect.set`, to `Reflect.deleteProperty`, and to a
#: function the file declares itself.
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
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAPrototypeHandedToAWriterIsStillWritten(TestBase):
    """
    A value that escapes into a call may be written by that call, which the effect model records —
    but only for the names it watches, and `Object` is not one of them. So a file that hands
    `Object.prototype` to anything at all keeps its chain reported intact, and every question about
    that chain is answered from the tables.

    `refinery.lib.scripts.js.analysis.effects` is where the watched set is chosen, and choosing it
    is what this costs: adding `Object` to it closes every row here and withdraws trust from a file
    that merely mentions the name, which is what the real samples do. It is deferred to the
    interprocedural precision that would let the escape be followed rather than assumed.
    """

    @unittest.expectedFailure
    def test_a_prototype_handed_to_a_writer_answers_the_read(self):
        """
        Node prints `9`, `9`, `false` and `9` for the four programs of
        `A_PROTOTYPE_HANDED_TO_SOMETHING_THAT_WRITES_IT`. Each deobfuscation answers as if the call
        it was handed to had not written it.
        """
        rows = A_PROTOTYPE_HANDED_TO_SOMETHING_THAT_WRITES_IT
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )


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


#: A program reaching `eval` through a name it bound to it and asking that name for a local of the
#: caller, mapped to what Node prints for it. Only a call written as the name `eval` is a direct
#: eval; every other way of reaching the same function runs the text in the global scope, where the
#: local is not, so what the program prints is the name of the error that raises.
AN_EVAL_REACHED_THROUGH_A_NAME_BOUND_TO_IT = {
    'function f(){ var loc = 7; var g = eval;'
    " try { return g('loc'); } catch (e) { return e.constructor.name; } }"
    ' console.log(f());': 'ReferenceError\n',
    'function f(){ var loc = 7; var g = eval;'
    " return typeof g('typeof loc'); }"
    ' console.log(f());': 'string\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestANameBoundToEvalIsNotADirectEval(TestBase):
    """
    `eval` is the one function the language treats differently depending on how the call was
    written. A call whose callee is the name `eval` runs its text in the calling scope; a call
    reaching the same function any other way runs it in the global scope, where the caller's locals
    are not. Substituting a name bound to `eval` for its value therefore turns one into the other,
    and the payload starts seeing bindings it could not have seen.

    `test.lib.scripts.js.deobfuscation.test_simplify` refuses this for two of the three ways of
    reaching it, `window.eval(code)` and `(0, eval)(code)`. A plain `var g = eval` is the third and
    is not refused, which is the same rule missing its own case rather than a new rule.
    """

    @unittest.expectedFailure
    def test_a_call_through_a_name_bound_to_eval_sees_no_local_of_its_caller(self):
        """
        Node prints `ReferenceError` for the first program of
        `AN_EVAL_REACHED_THROUGH_A_NAME_BOUND_TO_IT`, whose payload reads a local of the calling
        function, and `string` for the second, where a `typeof` guard makes the same read safe. The
        deobfuscation rewrites the call to a direct one and prints `7` for the first.
        """
        rows = AN_EVAL_REACHED_THROUGH_A_NAME_BOUND_TO_IT
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )


#: Programs calling a function the `async` keyword marks, mapped to what Node prints for each.
#: Calling one answers a promise whose resolution is the value the body returned, never that value,
#: so a program asking anything of what the call answered is asking it of the promise.
A_CALL_TO_AN_ASYNC_FUNCTION = {
    'async function m(){ return 7; }'
    ' m().then(function(v){ console.log(v); });': '7\n',
    'var o = { async m(){ return 7; } };'
    ' console.log(typeof o.m().then);': 'function\n',
    'async function m(){ return 7; }'
    ' console.log(m() instanceof Promise);': 'true\n',
}


#: The same shapes the fold is right about, which is what says the defect is the `async` keyword in
#: those two spellings and not the fold: an async function written as an expression or as an arrow,
#: a generator method, and an async method of a class are all left standing.
AN_ASYNC_SHAPE_THE_FOLD_LEAVES_ALONE = {
    'var m = async function(){ return 7; }; console.log(typeof m().then);': 'function\n',
    'var m = async () => 7; console.log(typeof m().then);': 'function\n',
    'class C { async m(){ return 7; } } console.log(typeof new C().m().then);': 'function\n',
    'var o = { *m(){ yield 7; } }; console.log(o.m().next().value);': '7\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestACallToAnAsyncFunctionAnswersAPromise(TestBase):
    """
    An `async` function does not answer what its body returned. Calling one answers a promise, and
    the value the body returned is what that promise resolves to, which is a different object with
    different members and a different type. Folding the call to the returned value is therefore not
    a reduction of the program but a replacement of it, and every question the file then asks of
    what the call answered is asked of the wrong thing.

    Two spellings are folded and four are not.  A declaration and a method of an object literal lose
    the keyword; the same function written as an expression, as an arrow, or as a method of a class,
    and a generator method beside them, are all left standing —
    `AN_ASYNC_SHAPE_THE_FOLD_LEAVES_ALONE` holds those and passes. So what is missing is not the
    rule but its reach: `refinery.lib.scripts.js.deobfuscation.helpers.try_inline_trivial_function`
    states the rule outright and whatever answers these two never asks it.

    Measured identical at `45d1e19a2`, so it is older than the work around it.
    """

    @unittest.expectedFailure
    def test_a_call_to_an_async_function_is_not_folded_to_what_its_body_returned(self):
        """
        Node prints `7`, `function` and `true` for the three programs of
        `A_CALL_TO_AN_ASYNC_FUNCTION`. Each deobfuscation replaces the call with `7`: the first
        throws a `TypeError` for calling `then` on a number, and the other two print `undefined`
        and `false`.
        """
        rows = A_CALL_TO_AN_ASYNC_FUNCTION
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    def test_an_async_function_written_another_way_is_left_standing(self):
        """
        Node prints `function` for the three programs of `AN_ASYNC_SHAPE_THE_FOLD_LEAVES_ALONE`
        that ask an async call for its `then`, and `7` for the generator. Each deobfuscation prints
        the same, which is what makes the entry above about two spellings rather than about the
        keyword.
        """
        rows = AN_ASYNC_SHAPE_THE_FOLD_LEAVES_ALONE
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )
