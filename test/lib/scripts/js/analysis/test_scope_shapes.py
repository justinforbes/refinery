"""
Which scope holds a name, for the three shapes `refinery.lib.scripts.js.analysis.model` builds
wrongly: the parameter list of a function that carries an expression, the block a function is
declared in, and the top level of a classic script, whose `this` is the global object.

The defects themselves are entries of `test.lib.scripts.js.test_release_blockers`. What is here is
the other half of each of them, which a fix is answerable for just as much: the programs the model
already answers correctly and must go on answering, and the reductions a correct answer costs. A
scope that is split apart, a value that stops being folded, and a declaration that stops being
removed each buy correctness with recall, and a cost written down before the change is the only one
a reader can tell from a regression afterwards.

The receiver table is the third kind: it is a question about JavaScript rather than about this
project, asked of a classic script because that is the execution model the question exists in at
all. A decorator is the one position it does not name, because no engine here parses one - Node
answers a `SyntaxError` for the class this file's parser reads - so there is nothing to write
down.

SECURITY: every program here is hand-authored in this file and benign. No sample and no stored
obfuscator fixture may be fed to this.
"""
from __future__ import annotations

import unittest

from test import TestBase
from test.lib.scripts.js.analysis.differential import node_executable
from test.lib.scripts.js.ledger import (
    Program,
    Reading,
    a_program,
    folded,
    prints,
)


def _still_answered(rows: dict[str, Program]) -> dict[str, tuple]:
    return {label: row.read() for label, row in rows.items()}


def _as_it_answers_them(rows: dict[str, Program]) -> dict[str, tuple]:
    return {label: row.required() for label, row in rows.items()}


#: A program declaring a function inside a block that the model already answers correctly, mapped to
#: the behavior an engine gives it. Where the copy Annex B makes reaches the enclosing scope, the
#: name holds the function from the declaration onwards, and every one of the conditions that
#: suppresses the copy leaves the enclosing name holding what it held.
A_BLOCK_FUNCTION_THE_PROGRAM_STILL_ANSWERS_FOR = {
    'a call after the block': Program(
        a_program("""
            function outer() {
              { function W() { return 1; } }
              console.log(W());
            }
            outer();
            """),
        prints('1'),
    ),
    'a call inside the block': Program(
        a_program("""
            function outer() {
              { function W() { return 1; } console.log(W()); }
            }
            outer();
            """),
        prints('1'),
    ),
    'a lexical binding of the name suppresses the copy': Program(
        a_program("""
            function outer() {
              let W = 7;
              { function W() { return 1; } }
              console.log(W);
            }
            outer();
            """),
        prints('7'),
    ),
    'a parameter of the name suppresses the copy': Program(
        a_program("""
            function outer(W) {
              { function W() { return 1; } }
              console.log(W);
            }
            outer('outer');
            """),
        prints('outer'),
    ),
    'a destructuring catch parameter suppresses the copy': Program(
        a_program("""
            function outer() {
              try { throw { W: 7 }; } catch ({ W }) {
                { function W() { return 1; } }
                console.log(W);
              }
              console.log(typeof W);
            }
            outer();
            """),
        prints('7', 'undefined'),
    ),
    'a simple catch parameter is what the block reads': Program(
        a_program("""
            function outer() {
              try { throw 7; } catch (W) {
                { function W() { return 1; } }
                console.log(W);
              }
              console.log(typeof W);
            }
            outer();
            """),
        prints('7', 'function'),
    ),
    'the copy takes the value the block name holds where it runs': Program(
        a_program("""
            function outer() {
              var W = 7;
              { W = 9; function W() { return 1; } }
              console.log(W);
            }
            outer();
            """),
        prints('9'),
    ),
    'a label between the block and the declaration is transparent': Program(
        a_program("""
            function outer() {
              { lab: function W() { return 1; } }
              console.log(typeof W);
            }
            outer();
            """),
        prints('function'),
    ),
    'the name arguments has its creation suppressed and its copy run': Program(
        a_program("""
            function outer() {
              { function arguments() { return 1; } }
              console.log(typeof arguments);
            }
            outer(1);
            """),
        prints('function'),
    ),
    'a block no branch takes runs no copy': Program(
        a_program("""
            function outer(c) {
              if (c) { function W() { return 1; } }
              console.log(typeof W);
            }
            outer(0);
            """),
        prints('undefined'),
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestABlockFunctionIsWhereItsModeAndItsPositionPutIt(TestBase):
    """
    The half of the block-function family the model answers correctly today. Every condition B.3.3.1
    lists for suppressing the copy is here, because a fix that models the copy has to model each of
    them, and a fix that suppresses one too many is not visible in the entries the defect is pinned
    by: those are all programs the copy is wrong about, and suppressing every copy would answer all
    of them.
    """

    def test_a_block_function_the_program_answers_for_is_answered_the_same_way(self):
        rows = A_BLOCK_FUNCTION_THE_PROGRAM_STILL_ANSWERS_FOR
        self.assertEqual(_still_answered(rows), _as_it_answers_them(rows))


#: A program whose block function writes a name declared outside the block, mapped to the behavior
#: an engine gives it. The call is the only thing that writes the name, so a reader that cannot say
#: which function a call reaches has to say that any of them may have written it.
A_CALL_TO_A_BLOCK_FUNCTION_THAT_WRITES_PAST_ITS_BLOCK = {
    'a call in a loop body': Program(
        a_program("""
            var v = 1;
            for (let i = 0; i < 1; i++) { function f() { v = 2; } f(); }
            console.log(v);
            """),
        prints('2'),
    ),
    'a call in a plain block': Program(
        a_program("""
            var v = 1;
            { function m() { v = 2; } m(); }
            console.log(v);
            """),
        prints('2'),
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestACallWhoseCalleeCannotBeValuedStillWrites(TestBase):
    """
    The control that says which direction a refusal is safe in. `SemanticModel.singular_value` is
    read as a value, where declining to answer costs a fold and nothing else, and it is read through
    `EffectModel.function_of` to decide which functions a call may have run, where declining to
    answer is what a caller must not treat as `no function ran`.

    Both programs here write a name from inside a block-declared function and read it afterwards.
    Refusing to value the callee and leaving the two readers as they are turns each of them into a
    program printing `1`: the write inside the body is deleted, the declaration it wrote to is
    removed, and the read is folded to what the initializer held. So this is not a recall cost to be
    weighed but the thing that makes the refusal a wrong answer, and it belongs beside the refusal
    rather than after it.
    """

    def test_a_write_from_a_block_function_still_reaches_the_name_outside(self):
        rows = A_CALL_TO_A_BLOCK_FUNCTION_THAT_WRITES_PAST_ITS_BLOCK
        self.assertEqual(_still_answered(rows), _as_it_answers_them(rows))


#: A call to a block-declared function the deobfuscation folds today, mapped to the exact text it
#: answers with. Every one of them is a correct answer, and every one of them is given up by a model
#: that refuses to value a name Annex B assigns rather than declares.
A_REDUCTION_THE_BLOCK_FUNCTION_REFUSAL_GIVES_UP = {
    'a call after the block': a_program("""
        function outer() {
          { function W() { return 1; } }
          console.log(W());
        }
        outer();
        """),
    'a call inside the block': a_program("""
        function outer() {
          { function W() { return 1; } console.log(W()); }
        }
        outer();
        """),
    'a value taken out of the block': a_program("""
        function outer() {
          { function W() { return 1; } }
          var g = W;
          console.log(g());
        }
        outer();
        """),
}


class TestAReductionTheBlockFunctionRefusalGivesUpIsGivenUp(TestBase):
    """
    What the block-function fix costs, written down before it is paid. A name Annex B copies into
    the enclosing scope holds the function only from the point the declaration runs, and the copy
    takes whatever the block's own name holds at that point, so nothing that reads the enclosing
    name can be answered from the declaration alone. The three calls here are answered from it
    today and are answered correctly; a model that stops guessing stops answering them.

    Read from the text and from nothing else, because that is what changes: the programs go on
    printing `1` either way, so no engine can report the difference.
    """

    def test_each_call_to_a_block_function_is_still_folded(self):
        self.assertEqual(
            {
                label: folded(program)
                for label, program in A_REDUCTION_THE_BLOCK_FUNCTION_REFUSAL_GIVES_UP.items()
            },
            {
                'a call after the block': (
                    'function outer() {\n  {}\n  console.log(1);\n}\nouter();'),
                'a call inside the block': (
                    'function outer() {\n  {\n    console.log(1);\n  }\n}\nouter();'),
                'a value taken out of the block': (
                    'function outer() {\n  {}\n  console.log(1);\n}\nouter();'),
            },
        )


#: A program whose parameters and body answer for the same name and which the model already answers
#: correctly, mapped to the behavior an engine gives it. A parameter and a body `var` of one name
#: are one binding the argument initializes, and the name of a named function expression is a
#: binding of its own, outside the parameters and behind any of them that spells it.
A_PARAMETER_SCOPE_THE_PROGRAM_STILL_ANSWERS_FOR = {
    'a bare var of a parameter name keeps the argument': Program(
        a_program("""
            function f(x = 1) { console.log(x); var x; console.log(x); }
            f();
            """),
        prints('1', '1'),
    ),
    'a var of a parameter name is written where its initializer runs': Program(
        a_program("""
            function f(x = 1) { console.log(x); var x = 5; console.log(x); }
            f();
            """),
        prints('1', '5'),
    ),
    'a default reading a later parameter throws': Program(
        a_program("""
            function f(a = b, b = 2) { return a; }
            try { console.log(f()); } catch (e) { console.log(e.constructor.name); }
            """),
        prints('ReferenceError'),
    ),
    'a lexical binding beside the parameters is its own': Program(
        a_program("""
            function f(x = 5) { let args = [x, 1]; return args; }
            console.log(JSON.stringify(f()));
            """),
        prints('[5,1]'),
    ),
    'a default reads the name of its own function expression': Program(
        a_program("""
            var f = function g(x = g) { return typeof x; };
            console.log(f());
            """),
        prints('function'),
    ),
    'a body var of the name of its own function expression is undefined': Program(
        a_program("""
            var f = function g() { var g; return typeof g; };
            console.log(f());
            """),
        prints('undefined'),
    ),
    'a parameter spelling the name of its own function expression wins': Program(
        a_program("""
            var f = function g(g) { return g; };
            console.log(f(7));
            """),
        prints('7'),
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAParameterAndItsBodyAnswerForOneName(TestBase):
    """
    The half of the parameter-default family the model answers correctly today. Splitting the
    parameters of a function into a scope of their own is what the defect needs, and every one of
    these programs is answered by the two being one scope, so each of them is a way for the split to
    go too far: a body `var` that stops seeing the argument, a lexical binding that stops being
    reachable, or the name of a function expression put on the wrong side of the parameters.
    """

    def test_a_parameter_scope_the_program_answers_for_is_answered_the_same_way(self):
        rows = A_PARAMETER_SCOPE_THE_PROGRAM_STILL_ANSWERS_FOR
        self.assertEqual(_still_answered(rows), _as_it_answers_them(rows))


#: A position of a classic script, mapped to the behavior a host gives a program that reports from
#: it whether its `this` is the global object. This is the table a fix has to agree with: the
#: positions answering `true` are exactly the ones a top-level declaration is reachable from through
#: `this`, and the ones answering `false` are the ones where `this` is whatever a caller passed.
THE_RECEIVER_A_POSITION_OF_A_SCRIPT_HOLDS = {
    'the top level': Program(
        a_program('console.log(this === globalThis);'),
        prints('true'),
        Reading.SCRIPT,
    ),
    'a strict top level': Program(
        a_program("""
            'use strict';
            console.log(this === globalThis);
            """),
        prints('true'),
        Reading.SCRIPT,
    ),
    'an arrow at the top level': Program(
        a_program('(() => { console.log(this === globalThis); })();'),
        prints('true'),
        Reading.SCRIPT,
    ),
    'an arrow inside an arrow': Program(
        a_program('(() => (() => { console.log(this === globalThis); })())();'),
        prints('true'),
        Reading.SCRIPT,
    ),
    'a default of an arrow': Program(
        a_program('((a = (this === globalThis)) => { console.log(a); })();'),
        prints('true'),
        Reading.SCRIPT,
    ),
    'an arrow in an object literal': Program(
        a_program("""
            var o = { m: () => this === globalThis };
            console.log(o.m());
            """),
        prints('true'),
        Reading.SCRIPT,
    ),
    'a computed key of a class': Program(
        a_program('class C { [(console.log(this === globalThis), "k")]() {} }'),
        prints('true'),
        Reading.SCRIPT,
    ),
    'an extends clause of a class': Program(
        a_program('class C extends (console.log(this === globalThis), Object) {}'),
        prints('true'),
        Reading.SCRIPT,
    ),
    'a sloppy call with no receiver': Program(
        a_program("""
            function f() { console.log(this === globalThis); }
            f();
            """),
        prints('true'),
        Reading.SCRIPT,
    ),
    'a method of an object literal': Program(
        a_program("""
            var o = { m() { return this === globalThis; } };
            console.log(o.m());
            """),
        prints('false'),
        Reading.SCRIPT,
    ),
    'a getter of an object literal': Program(
        a_program("""
            var o = { get g() { return this === globalThis; } };
            console.log(o.g);
            """),
        prints('false'),
        Reading.SCRIPT,
    ),
    'a static block of a class': Program(
        a_program('class C { static { console.log(this === globalThis); } }'),
        prints('false'),
        Reading.SCRIPT,
    ),
    'a field initializer of a class': Program(
        a_program("""
            class C { f = console.log(this === globalThis); }
            new C();
            """),
        prints('false'),
        Reading.SCRIPT,
    ),
    'a strict call with no receiver': Program(
        a_program("""
            function f() { 'use strict'; console.log(this === undefined); }
            f();
            """),
        prints('true'),
        Reading.SCRIPT,
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestTheReceiverAPositionOfAScriptHolds(TestBase):
    """
    Where the `this` of a classic script is its global object. The boundary is the one
    `SemanticModel.walk_receiver_scope` already draws for a different purpose: an arrow carries the
    `this` of what encloses it, and the head of a class - its `extends` clause and its computed keys
    - is evaluated where the class is written, while a method, a getter, a field initializer and a
    static block each take one of their own.

    The last two rows are the pair that says the boundary is not the whole rule: a call written with
    no receiver hands the body `undefined`, and only sloppy code replaces that with the global
    object, so a body reached that way is the global object under one mode and not under the other.
    """

    def test_each_position_reports_the_receiver_it_holds(self):
        rows = THE_RECEIVER_A_POSITION_OF_A_SCRIPT_HOLDS
        self.assertEqual(_still_answered(rows), _as_it_answers_them(rows))


#: A classic script whose top level writes a global property under a key no reading of the text
#: gives, mapped to the exact text the deobfuscation answers with today. The write may put anything
#: anywhere on the global object, so a model that reads it has to stop removing what the file does
#: not name - and `z` here is exactly that.
A_REMOVAL_A_COMPUTED_WRITE_THROUGH_THIS_STANDS_BESIDE = a_program("""
    var z = 1;
    var k = 'q';
    this[k] = 2;
    console.log(3);
    """)


#: The same script with nothing written through `this` at all, which is the control: a top-level
#: declaration no reading of the file reaches is removed today and has to go on being removed, or
#: the fix has bought the entry it is for by keeping every declaration of every script.
A_TOP_LEVEL_DECLARATION_NOTHING_IN_THE_FILE_NAMES = a_program("""
    var q = function (a) { console.log(a); };
    console.log(1);
    """)


class TestARemovalTheGlobalObjectFixGivesUp(TestBase):
    """
    What the top-level `this` fix costs, written down before it is paid. Reading `this` as the
    global object means reading a write through it as a write of a global property, and a write
    under a computed key is a write of a property no reading of the text names, so every removal in
    the file it stands in has to stop. The control beside it is the file with no such write, whose
    removals must go on happening.

    Read from the text and from nothing else: both programs print what they printed either way.
    """

    def test_a_removal_a_computed_write_stands_beside_still_happens(self):
        self.assertEqual(
            folded(A_REMOVAL_A_COMPUTED_WRITE_THROUGH_THIS_STANDS_BESIDE),
            'this.q = 2;\nconsole.log(3);',
        )

    def test_a_top_level_declaration_nothing_names_is_still_removed(self):
        self.assertEqual(
            folded(A_TOP_LEVEL_DECLARATION_NOTHING_IN_THE_FILE_NAMES),
            'console.log(1);',
        )
