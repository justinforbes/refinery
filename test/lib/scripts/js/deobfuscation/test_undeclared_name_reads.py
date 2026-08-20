"""
Evaluating a bare name that denotes no binding throws a `ReferenceError`, so a function whose body
reads such a name is not a function whose call may be quietly discarded: discarding it turns a
program that refuses to run into one that runs and prints. Every program in
`_A_READ_WHOSE_NAME_MAY_NOT_EXIST` prints after the read that decides it, so a call that went
missing shows up as a line Node did not print for the original.

The other direction is the one Node cannot see, because a deobfuscator that reduces nothing
preserves behavior perfectly. `_REDUCED` is what must still go, and it asserts text: a name that a
declaration, a parameter, a `catch` clause or the specification itself binds is a name whose read
cannot throw, and a property name, an object key, a label and a `typeof` do not read a binding at
all.

Node is the authority for every value recorded in this module.
"""
from __future__ import annotations

import inspect
import unittest

from typing import NamedTuple

from test.lib.scripts.js.analysis.differential import behavior, node_executable
from test.lib.scripts.js.deobfuscation import TestJsDeobfuscator


class _Program(NamedTuple):
    """
    A program whose observable behavior turns on whether one name denotes a binding, together with
    what Node prints for it and the error it ends with, if any.
    """
    source: str
    printed: str
    error: str | None = None


class _Rewrite(NamedTuple):
    """
    A program and the text the deobfuscator leaves it as.
    """
    source: str
    result: str


_A_READ_WHOSE_NAME_MAY_NOT_EXIST: dict[str, _Program] = {
    'a bare expression statement': _Program(
        inspect.cleandoc(
            """
            function f() {
              missing;
            }
            f();
            console.log('after');
            """
        ),
        '',
        'ReferenceError',
    ),
    'a returned value': _Program(
        inspect.cleandoc(
            """
            function f() {
              return missing;
            }
            f();
            console.log('after');
            """
        ),
        '',
        'ReferenceError',
    ),
    'an argument to another call': _Program(
        inspect.cleandoc(
            """
            function f() {
              return String(missing);
            }
            f();
            console.log('after');
            """
        ),
        '',
        'ReferenceError',
    ),
    'a shorthand object key': _Program(
        inspect.cleandoc(
            """
            function f() {
              return { missing };
            }
            f();
            console.log('after');
            """
        ),
        '',
        'ReferenceError',
    ),
    'a member read off the name': _Program(
        inspect.cleandoc(
            """
            function f() {
              return typeof missing.x;
            }
            f();
            console.log('after');
            """
        ),
        '',
        'ReferenceError',
    ),
    'a name only a function nothing calls assigns': _Program(
        inspect.cleandoc(
            """
            function never() {
              later = 1;
            }
            function f() {
              return later;
            }
            f();
            console.log('after');
            """
        ),
        '',
        'ReferenceError',
    ),
    'a name an assignment creates only after the call': _Program(
        inspect.cleandoc(
            """
            function f() {
              return boot;
            }
            f();
            boot = 1;
            console.log('after');
            """
        ),
        '',
        'ReferenceError',
    ),
    'a host name the specification does not mandate': _Program(
        inspect.cleandoc(
            """
            function f() {
              return document;
            }
            f();
            console.log('after');
            """
        ),
        '',
        'ReferenceError',
    ),
    'a name an assignment has already run to create': _Program(
        inspect.cleandoc(
            """
            boot = 1;
            function f() {
              return boot;
            }
            console.log(f());
            """
        ),
        '1\n',
    ),
    'a typeof guard in front of a name the host defines': _Program(
        inspect.cleandoc(
            """
            globalThis.present = 5;
            function f() {
              if (typeof present === 'undefined') {
                return 'absent';
              }
              return present;
            }
            console.log(f());
            """
        ),
        '5\n',
    ),
    'a delete of a name the host defines': _Program(
        inspect.cleandoc(
            """
            globalThis.present = 5;
            function f() {
              return delete present;
            }
            console.log(f());
            console.log(typeof present);
            """
        ),
        'true\nundefined\n',
    ),
}
"""
Programs in which one name decides what happens, mapped to what Node makes of each. The last three
are the controls that a refusal cannot pass by refusing everything: the name is there by the time it
is read, and the program prints what it read.
"""

_REDUCED: dict[str, _Rewrite] = {
    'a name that spells the global object': _Rewrite(
        inspect.cleandoc(
            """
            function f() {
              return globalThis;
            }
            f();
            console.log('after');
            """
        ),
        "console.log('after');",
    ),
    'a global the specification mandates': _Rewrite(
        inspect.cleandoc(
            """
            function f() {
              return Math.max(1, 2);
            }
            console.log(f());
            """
        ),
        'console.log(2);',
    ),
    'a typeof of a global the specification mandates': _Rewrite(
        inspect.cleandoc(
            """
            function f() {
              return typeof JSON;
            }
            console.log(f());
            """
        ),
        "console.log('object');",
    ),
    'a declared global': _Rewrite(
        inspect.cleandoc(
            """
            var g = 3;
            function f() {
              return g;
            }
            console.log(f());
            """
        ),
        'console.log(3);',
    ),
    'a local declaration of the name': _Rewrite(
        inspect.cleandoc(
            """
            function f() {
              var missing = 3;
              return missing;
            }
            console.log(f());
            """
        ),
        'console.log(3);',
    ),
    'a catch parameter': _Rewrite(
        inspect.cleandoc(
            """
            function f() {
              try {
                return 1;
              } catch (missing) {
                return missing;
              }
            }
            console.log(f());
            """
        ),
        'console.log(1);',
    ),
    'an object key read back as a property name': _Rewrite(
        inspect.cleandoc(
            """
            function f() {
              var o = { missing: 7 };
              return o.missing;
            }
            console.log(f());
            """
        ),
        'console.log(7);',
    ),
    'a parameter': _Rewrite(
        inspect.cleandoc(
            """
            function f(missing) {
              return missing + 1;
            }
            console.log(f(4));
            """
        ),
        'console.log(5);',
    ),
    'the function nothing calls that holds the only assignment': _Rewrite(
        inspect.cleandoc(
            """
            function never() {
              later = 1;
            }
            console.log(later);
            """
        ),
        'console.log(later);',
    ),
}
"""
A program the deobfuscator must still reduce, mapped to the text it leaves it as. The last row asks
for both directions at once: the function is unreachable and goes, and the read that outlives it is
not answered with the value that function would have stored.
"""


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAReadOfANameThatMayNotDenoteABinding(TestJsDeobfuscator):

    def test_node_does_what_each_row_records(self):
        for name, program in _A_READ_WHOSE_NAME_MAY_NOT_EXIST.items():
            with self.subTest(name=name):
                self.assertEqual(behavior(program.source), (program.printed, program.error))

    def test_the_deobfuscated_program_does_the_same(self):
        for name, program in _A_READ_WHOSE_NAME_MAY_NOT_EXIST.items():
            deobfuscated = self._deobfuscate(program.source)
            with self.subTest(name=name):
                self.assertEqual(
                    behavior(deobfuscated),
                    (program.printed, program.error),
                    F'deobfuscation changed observable behavior; result was:\n{deobfuscated}',
                )


class TestAReadThatCannotThrowIsStillReduced(TestJsDeobfuscator):

    def test_each_program_is_left_as_the_text_the_row_records(self):
        for name, rewrite in _REDUCED.items():
            with self.subTest(name=name):
                self.assertEqual(self._deobfuscate(rewrite.source), rewrite.result)

    @unittest.skipIf(node_executable() is None, 'node.js is not available')
    def test_node_does_the_same_for_each_program_and_the_text_it_was_left_as(self):
        for name, rewrite in _REDUCED.items():
            with self.subTest(name=name):
                self.assertEqual(behavior(rewrite.source), behavior(rewrite.result))
