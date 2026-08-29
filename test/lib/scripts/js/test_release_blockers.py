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
from test.lib.scripts.js.analysis.differential import node_executable
from test.lib.scripts.js.ledger import (
    Program,
    a_program,
    one_expected_failure_per_program,
    prints,
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
