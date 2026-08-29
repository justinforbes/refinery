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
