"""
Converting a function to a string answers the source it was written with. The parser records the
span of every function it reads, and the value domain hands that text back for `String(fn)`, for a
concatenation, and for the join an array performs on its elements; the length of one is the length
of that text. A function a transform built instead carries no span and is written back the way the
synthesizer would write it. Retired from `test.lib.scripts.js.test_release_blockers`.
"""
from __future__ import annotations

import unittest

from test import TestBase
from test.lib.scripts.js.analysis.differential import node_executable
from test.lib.scripts.js.ledger import before_and_after, folded

#: A program reading the text of a function value, mapped to the file `refinery.js` folds it to. The
#: read is written inside a function body, which is where the tool answers it at all; the folded
#: text holds the source the function was written with, character for character, as a string
#: literal. Each expected value was confirmed against Node: `String` of a declaration keeps its
#: name, a function expression drops it, an arrow is the arrow, and a method keeps its name.
A_FUNCTION_STRINGIFIES_TO_ITS_WRITTEN_SOURCE = {
    'a declaration handed to String': (
        'function W(a) { return a + 1; }\n'
        'function f() { return String(W); }\n'
        'console.log(f());\n',
        "console.log('function W(a) { return a + 1; }');",
    ),
    'the length of the text of a declaration': (
        'function W(a) { return a + 1; }\n'
        'function f() { return String(W).length; }\n'
        'console.log(f());\n',
        'console.log(31);',
    ),
    'a local function expression concatenated onto a string': (
        "function f() { var W = function (a) { return a + 1; }; return '' + W; }\n"
        'console.log(f());\n',
        "console.log('function (a) { return a + 1; }');",
    ),
    'a local function expression joined by an array': (
        "function f() { var W = function (a) { return a + 1; }; return [W].join(''); }\n"
        'console.log(f());\n',
        "console.log('function (a) { return a + 1; }');",
    ),
    'an arrow function handed to String': (
        'function f() { var g = (a) => a + 1; return String(g); }\n'
        'console.log(f());\n',
        "console.log('(a) => a + 1');",
    ),
    'a shorthand method handed to String': (
        'function f() { var o = { m(a) { return a + 1; } }; return String(o.m); }\n'
        'console.log(f());\n',
        "console.log('m(a) { return a + 1; }');",
    ),
    'a declaration interpolated into a template': (
        'function W(a) { return a + 1; }\n'
        'function f() { return `<${W}>`; }\n'
        'console.log(f());\n',
        "console.log('<function W(a) { return a + 1; }>');",
    ),
    'a declaration added to a number': (
        'function W(a) { return a + 1; }\n'
        'function f() { return W + 1; }\n'
        'console.log(f());\n',
        "console.log('function W(a) { return a + 1; }1');",
    ),
}


class TestAFunctionStringifiesToItsWrittenSource(TestBase):
    def test_the_fold_writes_the_source_the_function_was_written_with(self):
        for label, (source, expected) in A_FUNCTION_STRINGIFIES_TO_ITS_WRITTEN_SOURCE.items():
            with self.subTest(label):
                self.assertEqual(folded(source, module=True), expected)


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestTheStringifiedFunctionRunsTheSameBeforeAndAfter(TestBase):
    def test_the_deobfuscation_prints_what_the_program_prints(self):
        for label, (source, _) in A_FUNCTION_STRINGIFIES_TO_ITS_WRITTEN_SOURCE.items():
            with self.subTest(label):
                before, after = before_and_after(source)
                self.assertEqual(before, after)
