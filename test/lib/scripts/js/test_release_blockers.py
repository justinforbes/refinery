"""
The JavaScript defects a release is held for.

Same form as `test.lib.scripts.js.test_unfixed_defects`, which these entries were separated out of,
and the same rules: every test states what a correct implementation would do, never what the code
does today, and is marked `unittest.expectedFailure`, so an entry that starts passing is reported as
an unexpected success and leaves this file only by being fixed. Where the question is one about
JavaScript rather than about this project, the answer was established with Node.js and is written
into the row the entry holds, so that the only statement of it is the one that is executed.

What sets these apart is what they cost rather than what they are. Each one takes a program an
engine runs and hands back one that behaves differently: nothing throws, nothing is left
half-rewritten, and the analyst reading it gets no signal that the answer is not the one the
language gives. An entry that merely refuses to reduce something, or reduces it to something
uglier, belongs in the other file, and so does everything about a file no engine runs, however
clean the answer for one looks: mishandling invalid input is never what a release is held for.
This file emptying is the release gate.

An entry whose programs are spellings of one root is pinned by one test over all of them. An entry
whose programs have roots a fix may reach separately is pinned by one test per program instead,
installed by `_one_expected_failure_per_program` and named for the shape that program holds: a fix
that reaches some of the shapes and not the others is then reported as a fix rather than as nothing
at all, which one test over the whole family cannot do.
"""
from __future__ import annotations

import inspect
import unittest

from typing import Callable, Mapping

from test import TestBase
from test.lib.scripts.js.analysis.differential import behavior, node_executable
from test.lib.scripts.js.ledger import (
    Program,
    Reading,
    a_program,
    before_and_after,
    before_and_after_in_a_host,
    each_program_still_prints,
    evaluated_in_a_body,
    folded,
    printed,
    prints,
    well_formed,
)


def _one_expected_failure_per_program(
    rows: Mapping[str, Program],
) -> Callable[[type], type]:
    """
    Install on the class one expected-failure test per program of *rows*, each named for the shape
    its key labels and each asserting the one law every entry here asserts: the deobfuscation of a
    program behaves the way the program does.
    """
    def install(entry: type) -> type:
        for label, row in rows.items():
            def test(self, row=row):
                self.assertEqual(row.read(), row.required())
            test.__name__ = F'test_{label.replace(" ", "_").replace("-", "_")}_still_behaves_so'
            if hasattr(entry, test.__name__):
                raise AssertionError(F'{entry.__name__} already holds {test.__name__}')
            setattr(entry, test.__name__, unittest.expectedFailure(test))
        return entry
    return install


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


#: An argument to a method the object fold answers, whose evaluation the answer has to keep. The
#: first is used by nothing the body returns, so folding the call away takes the call to `g` with it
#: and the write to `SIDE` never happens; the second pair is used in the other order than it is
#: written, and substituting the arguments into the body puts their effects in the body's order.
#: Each is mapped to what Node prints for it.
AN_ARGUMENT_WHOSE_EFFECT_THE_FOLD_OWES = {
    'function g() { SIDE = 1; return 2; }'
    ' var o = { m: function (a) { return 7; } };'
    ' console.log(o.m(g()));'
    ' console.log(SIDE);': '7\n1\n',
    "var LOG = '';"
    " function p() { LOG += 'p'; return 1; }"
    " function q() { LOG += 'q'; return 2; }"
    ' var o = { m: function (a, b) { return b + a; } };'
    ' console.log(o.m(p(), q()), LOG);': '3 pq\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAnArgumentIsEvaluatedWhereItIsWritten(TestBase):
    """
    Substituting call-site arguments into a method body moves each argument to wherever the body
    reads it — which may be nowhere, and may be in another order. An argument the body never reads is
    then never evaluated, and a pair the body reads back to front runs back to front.

    `refinery.lib.scripts.js.deobfuscation.helpers.is_safe_iife_inline` states the rule this needs:
    an effectful argument must be used exactly once, unconditionally, and in declaration order.
    `refinery.lib.scripts.js.deobfuscation.objectfold` does not ask it, and asks
    `try_inline_trivial_function` for a substitution that admission would have refused.
    """

    @unittest.expectedFailure
    def test_each_argument_runs_once_and_in_the_order_it_is_written(self):
        """
        Node prints `7` then `1` for the first program of `AN_ARGUMENT_WHOSE_EFFECT_THE_FOLD_OWES`
        and `3 pq` for the second. The deobfuscation drops the call to `g` from the first, so it
        prints `7` and then throws `ReferenceError` for `SIDE`, and reverses the pair in the second,
        so it prints `3 qp`.
        """
        rows = AN_ARGUMENT_WHOSE_EFFECT_THE_FOLD_OWES
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
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


#: A program binding one name that holds a zero-width joiner, mapped to what Node prints for it.
#: U+200C is an identifier character, and obfuscators reach for it because nothing renders it. The
#: name is assembled from `chr` so that no invisible character stands in this file.
A_NAME_HOLDING_A_JOINER = {
    F'function f(a{chr(0x200C)}b) {{ return a{chr(0x200C)}b + 1; }} console.log(f(6));': '7\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestANameHoldingAJoinerIsOneName(TestBase):
    """
    The defect is in no pass: handed the same text directly, the deobfuscation answers correctly.
    The unit guesses the codec of its input bytes before parsing, and valid UTF-8 holding a joiner
    is guessed as cp1252, so the unit reads and rewrites a different program than the file holds.
    """

    @unittest.expectedFailure
    def test_a_name_holding_a_joiner_keeps_the_value_it_computes(self):
        """
        Node prints `7` for the program of `A_NAME_HOLDING_A_JOINER`, whose one function returns
        its argument plus one. Through the unit it comes back as `console.log(6);`.
        """
        rows = A_NAME_HOLDING_A_JOINER
        self.assertEqual(
            {source: (behavior(source), behavior(folded(source))) for source in rows},
            each_program_still_prints(rows),
        )


#: A program declaring a function inside a block, mapped to the behavior an engine gives it. The
#: block is where the name lives in strict code and the copy Annex B makes is what puts it outside
#: one in sloppy code, so the shapes here differ in the mode the block is read under and in where
#: the name is read from.
A_FUNCTION_DECLARED_INSIDE_A_BLOCK = {
    'a module': Program(
        a_program("""
            function outer() {
              { function W() { return 1; } }
              try { console.log(W()); } catch (e) { console.log('threw'); }
            }
            outer();
            export {};
            """),
        prints('threw'),
        Reading.ES_MODULE,
    ),
    'a script saying so': Program(
        a_program("""
            'use strict';
            function outer() {
              { function W() { return 1; } }
              try { console.log(W()); } catch (e) { console.log('threw'); }
            }
            outer();
            """),
        prints('threw'),
    ),
    'a function body saying so': Program(
        a_program("""
            function outer() {
              'use strict';
              { function W() { return 1; } }
              try { console.log(W()); } catch (e) { console.log('threw'); }
            }
            outer();
            """),
        prints('threw'),
    ),
    'a sloppy call before the block': Program(
        a_program("""
            function outer() {
              try { console.log(W()); } catch (e) { console.log('threw'); }
              { function W() { return 1; } }
            }
            outer();
            """),
        prints('threw'),
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
@_one_expected_failure_per_program(A_FUNCTION_DECLARED_INSIDE_A_BLOCK)
class TestAFunctionDeclaredInsideABlockIsDeclaredInsideIt(TestBase):
    """
    A function declared inside a plain block is a lexical binding of that block, and what reaches
    the enclosing variable scope is decided by the mode. Strict code puts nothing there: a call
    beside the block reads no binding and throws. Sloppy code creates a `var` of the name at the
    entry of the enclosing function and copies the block's function into it where the declaration
    runs, so a call before the block reads `undefined` and throws too, and only a call after it
    answers the function.

    `refinery.lib.scripts.js.analysis.model` places every block function in the enclosing variable
    scope whatever the mode, initialized as if it were declared there. The strict-mode overlay that
    knows the difference exists and is not consulted, and the point in the block at which the copy
    runs is not modeled at all, so every consumer reads a binding holding a function where the
    program holds none, and the folds downstream answer the call with the function's value.
    """


#: A program whose block is one no branch of it takes, mapped to the behavior an engine gives it.
#: What a dead block declares is declared all the same - a `var` and, in sloppy code, the `var` half
#: of a function declaration are created at the entry of the enclosing scope, before any branch is
#: decided - so a read of the name answers `undefined` rather than throwing.
A_DECLARATION_A_DEAD_BLOCK_HOLDS = {
    'a function no branch declares': Program(
        a_program("""
            if (0) { function W() { return 1; } }
            console.log(W);
            """),
        prints('undefined'),
    ),
    'a var no branch declares': Program(
        a_program("""
            if (0) { var v = 1; }
            console.log(v);
            """),
        prints('undefined'),
    ),
    'a function that is the whole of an if clause': Program(
        a_program("""
            function outer() {
              if (0) function W() { return 1; }
              console.log(W);
            }
            outer();
            """),
        prints('undefined'),
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
@_one_expected_failure_per_program(A_DECLARATION_A_DEAD_BLOCK_HOLDS)
class TestADeclarationADeadBlockHoldsIsStillDeclared(TestBase):
    """
    Removing a branch no condition takes removes the statements it ran, and a declaration is not
    only a statement: a `var` and the `var` half of a sloppy function declaration name something
    from the entry of the enclosing scope onwards, whether or not the branch holding them is ever
    reached. `refinery.lib.scripts.js.deobfuscation.deadcode` drops the branch whole, so the name
    the program answered `undefined` for is left bound to nothing and reading it throws.

    The two halves are one defect and are pinned as two rows because they are removed by one rule
    reading two shapes: `JsDeadCodeElimination._unwrap_branch` already keeps a `let` or `const` the
    branch declares, which is the mechanism the `var` half needs and does not use.
    """


class TestTheDeclarationsADeadBlockHeldAreGone(TestBase):
    """
    The control for `TestADeclarationADeadBlockHoldsIsStillDeclared`, read from the text and from
    nothing else, so that the entry cannot go quiet on a machine with no Node.js and cannot be
    answered by a program that fails for some other reason. It names the removal the entry is about:
    what comes back is the read alone, with nothing left declaring the name it reads.
    """

    def test_each_dead_block_is_removed_and_the_read_left_behind(self):
        self.assertEqual(
            {
                label: folded(row.text)
                for label, row in A_DECLARATION_A_DEAD_BLOCK_HOLDS.items()
            },
            {
                'a function no branch declares': 'console.log(W);',
                'a var no branch declares': 'console.log(v);',
                'a function that is the whole of an if clause': inspect.cleandoc(
                    """
                    function outer() {
                      console.log(W);
                    }
                    outer();
                    """
                ),
            },
        )


#: A classic script whose function is called with no receiver, mapped to the behavior a host gives
#: it. Sloppy code gives such a call the global object as its `this`, so the body reaches the
#: script's own top-level declarations through it.
A_SLOPPY_CALL_READING_THE_GLOBAL_OBJECT = {
    'a bare call': Program(
        a_program("""
            var q = function (a) { console.log('q', a); };
            function f() { this.q(1); }
            f();
            """),
        prints('q 1'),
        Reading.SCRIPT,
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
@_one_expected_failure_per_program(A_SLOPPY_CALL_READING_THE_GLOBAL_OBJECT)
class TestASloppyCallGivesItsBodyTheGlobalObject(TestBase):
    """
    A call written with no receiver passes `undefined` as the `this` of the call, and sloppy code
    replaces that with the global object before the body runs. So a sloppy function nothing calls as
    a method reads the global object through `this` just as its script's top level does, and a
    top-level declaration is reachable through it.

    This is the same gap as `TestATopLevelThisNamesTheGlobalObject` at one remove, and it is not the
    same fix: what a `this` in a function body holds is decided by every call reaching that body,
    where the `this` of a top level is decided by the position alone. The declaration is removed and
    the call left standing, so the program comes back throwing where it printed.
    """


#: A program reading the text of a function value, mapped to the behavior an engine gives it. The
#: read is written inside a function body, which is where the tool answers it at all: the same read
#: at the top of a file is left standing.
THE_TEXT_A_FUNCTION_WAS_WRITTEN_WITH = {
    'a declaration handed to String': Program(
        a_program("""
            function W(a) { return a + 1; }
            function f() { return String(W); }
            console.log(f());
            """),
        prints('function W(a) { return a + 1; }'),
    ),
    'the length of the text of a declaration': Program(
        a_program("""
            function W(a) { return a + 1; }
            function f() { return String(W).length; }
            console.log(f());
            """),
        prints('31'),
    ),
    'a local function value concatenated': Program(
        a_program("""
            function f() { var W = function (a) { return a + 1; }; return '' + W; }
            console.log(f());
            """),
        prints('function (a) { return a + 1; }'),
    ),
    'a local function value joined': Program(
        a_program("""
            function f() { var W = function (a) { return a + 1; }; return [W].join(''); }
            console.log(f());
            """),
        prints('function (a) { return a + 1; }'),
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
@_one_expected_failure_per_program(THE_TEXT_A_FUNCTION_WAS_WRITTEN_WITH)
class TestTheTextOfAFunctionIsTheTextItWasWrittenWith(TestBase):
    """
    Converting a function to a string answers the source text it was written with, character for
    character, and every conversion reaches it: `String`, a concatenation, a template, and the join
    an array performs on its elements. The interpreter has no value for that text and converts a
    function the way it converts a plain object, so each of these answers `[object Object]` and the
    length of one answers `15`.

    An obfuscator reads this text on purpose - a self-check comparing a function's own source
    against a stored length or hash is a common anti-tamper device - so answering it wrongly hands
    the analyst a program that takes the branch the original never took. The numeric row is the one
    that shows it: a length is a plain number, and nothing about `15` says it was not computed.
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
@_one_expected_failure_per_program(A_DIRECT_EVAL_IN_A_DEFAULT_DECLARING_A_NAME)
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
    """


#: A classic script reading one of its own top-level declarations through a local name a guard put
#: the global object into, mapped to the behavior a host gives it. `window` is installed first,
#: because Node has no such name and the guard would otherwise answer the empty object.
A_DECLARATION_READ_THROUGH_A_LOCAL_ALIAS = {
    'an alias a guard built': Program(
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
@_one_expected_failure_per_program(A_DECLARATION_READ_THROUGH_A_LOCAL_ALIAS)
class TestADeclarationReadThroughALocalAliasIsStillRead(TestBase):
    """
    `var w = window || {}` is how a script written to run in a browser and in something else names
    the global object once, and every read through the local name afterwards is a read of a global
    property. The model recognizes an access whose base is one of six spellings and nothing else, so
    a read through the local name is recorded nowhere and the declaration it reaches is removed.

    Two notions of a local alias already exist and disagree - `refinery.lib.scripts.js.deobfuscation
    .simplify` holds one and `refinery.lib.scripts.js.deobfuscation.unused` holds another - which is
    why this belongs to the model rather than to either of them.
    """


#: A classic script handing the global object to a function that writes a property of it, mapped to
#: the behavior a host gives it. The object is spelled both ways a top level may spell it.
THE_GLOBAL_OBJECT_HANDED_TO_A_CALL = {
    'the top-level this': Program(
        a_program("""
            var q = 1;
            function a(g, k) { g[k] = 2; }
            a(this, 'q');
            console.log(q);
            """),
        prints('2'),
        Reading.SCRIPT,
    ),
    'globalThis': Program(
        a_program("""
            var q = 1;
            function a(g, k) { g[k] = 2; }
            a(globalThis, 'q');
            console.log(q);
            """),
        prints('2'),
        Reading.SCRIPT,
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
@_one_expected_failure_per_program(THE_GLOBAL_OBJECT_HANDED_TO_A_CALL)
class TestTheGlobalObjectHandedToACallMayBeWrittenThrough(TestBase):
    """
    A call that is handed the global object may write any property of it, and a top-level
    declaration of a classic script is such a property. The model reads a write of a global property
    off the spelling of the base at the write, so a write whose base is a parameter holding the
    object is one it never sees, and the declaration keeps the value its initializer gave it.

    Nothing throws and nothing is left half-rewritten: the read is folded to the value the
    declaration held, and the program comes back printing a different number. The two rows are one
    defect asked of the two spellings a top level has for the object, so that a fix reading only the
    name is reported as reaching one of them.
    """
