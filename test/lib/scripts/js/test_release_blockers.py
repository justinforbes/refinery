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
language gives. And each is reached by a shape real input plausibly holds: a wrong answer only
a shape constructed for the defect can reach costs no analyst anything, so it lives in the
other file however wrong it is, with the judgment of its unlikelihood written on the entry. An
entry that merely refuses to reduce something, or reduces it to something uglier, belongs in
the other file as well — unless the refusal forfeits the reduction of a whole class of
input, which costs what a wrong answer costs — and so does everything about a file no
engine runs, however clean the answer for one looks: mishandling invalid input is never what a
release is held for. This file emptying is the release gate.

An entry whose programs are spellings of one root is pinned by one test over all of them. An entry
whose programs have roots a fix may reach separately is pinned by one test per program instead,
installed by `test.lib.scripts.js.ledger.one_expected_failure_per_program` and named for the
shape that program holds: a fix that reaches some of the shapes and not the others is then
reported as a fix rather than as nothing at all, which one test over the whole family cannot do.
"""
from __future__ import annotations

import unittest

from test import TestBase
from test.lib.scripts.js.analysis.differential import (
    behavior,
    deobfuscate_source,
    host_behavior,
    node_executable,
)
from test.lib.scripts.js.ledger import (
    Program,
    a_program,
    before_and_after,
    each_program_still_prints,
    folded,
    one_expected_failure_per_program,
    prints,
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
@one_expected_failure_per_program(THE_TEXT_A_FUNCTION_WAS_WRITTEN_WITH)
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

    On the release gate since likelihood was reviewed: `var self = this` is the closure idiom of
    a whole era of JavaScript, so a local named `self` holding an ordinary object is a habit
    rather than a corner, and the write deleted here is one that `JSON.stringify(self)` or any
    second name for the object reads back.
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


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestANamedEntrypointIsKeptForTheHost(TestBase):
    """
    `refinery.units.scripting.js` takes entrypoint names on its command line, and a name given
    there is a promise about the output: the host calls the name once the file has loaded, so what
    it holds must survive, behave as it did, and keep reading its globals rather than folded copies
    of them, since a host that can reach a name can also have rewritten it. The promise is kept for
    a function held in a `var` and broken for the two commoner shapes: a function declaration named
    as an entrypoint is deleted, and a data global named as one is folded into its readers and
    removed.

    What admits this here is that the contract broken is the unit's own rather than the
    language's: the input can be any file at all, and the option is one the unit ships. The
    kept shapes and the module model's half of the contract are stated where these three rows
    came from, `test.lib.scripts.js.analysis.test_differential`.
    """

    def assertSameBehavior(
        self,
        source: str,
        *,
        calls: tuple[str, ...],
        entrypoints: tuple[str, ...] = (),
    ):
        deobfuscated = deobfuscate_source(source, entrypoints=entrypoints)
        self.assertEqual(
            host_behavior(source, calls=calls),
            host_behavior(deobfuscated, calls=calls),
            F'deobfuscation changed what a host observes; result was:\n{deobfuscated}',
        )

    @unittest.expectedFailure
    def test_a_declared_function_named_as_an_entrypoint_is_kept(self):
        source = 'function handler() { return 5; }'
        self.assertEqual(deobfuscate_source(source), '')
        self.assertSameBehavior(source, calls=('handler',), entrypoints=('handler',))

    @unittest.expectedFailure
    def test_a_helper_an_entrypoint_reaches_is_kept(self):
        source = (
            'function help() { console.log("hi!"); return 7; }'
            ' function handler() { return help(); }'
        )
        self.assertEqual(deobfuscate_source(source), '')
        self.assertSameBehavior(source, calls=('help',), entrypoints=('handler',))

    @unittest.expectedFailure
    def test_a_data_global_named_as_an_entrypoint_survives_unfolded(self):
        source = 'var VERSION = 3; console.log(VERSION);'
        self.assertEqual(
            deobfuscate_source(source, entrypoints=('VERSION',)),
            'var VERSION = 3;\nconsole.log(VERSION);',
        )
