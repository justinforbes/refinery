"""
A call with no receiver gives its body the global object.

A function invoked as `f()` is passed `undefined` as its `this`, and sloppy code replaces that with
the global object before the body runs (ECMA-262 §10.2.1.2). So a function nothing calls as a method
reads a classic script's own top-level declarations through `this`, and a remover that does not
record those reads deletes a declaration the program still uses.

`refinery.lib.scripts.js.analysis.model.may_be_global_object_base` is what records them. It admits
every `this` rather than only the ones a bare call can reach, because the two directions cost
different things: admitting a receiver that turns out to be some other object keeps a declaration
nothing reaches, while missing one removes a declaration that runs. The rows below are the second
half of that trade — a receiver that is *not* the global object, whose own property must still be
the one that is read.

The two readers that decide a rewrite keep the narrow question, and are asked here too: a reflective
surface reported for any `this` would freeze every removal in the file, and a timer callee is a name
the pass acts on.

SECURITY: every program here is hand-authored in this file and benign. No sample and no stored
obfuscator fixture may be fed to this.
"""
from __future__ import annotations

import inspect
import unittest

from test import TestBase
from test.lib.scripts.js.analysis.differential import node_executable
from test.lib.scripts.js.ledger import before_and_after_in_a_host


def a_program(text: str) -> str:
    return inspect.cleandoc(text) + chr(10)


#: A classic script whose function is called with no receiver, mapped to what a host prints for it.
A_BARE_CALL_READS_THE_GLOBAL_OBJECT = {
    'a bare call reaches a declaration': a_program("""
        var q = function (a) { console.log('q', a); };
        function f() { this.q(1); }
        f();
        """),
    'a bare call reaches a value': a_program("""
        var q = 7;
        function f() { console.log(this.q); }
        f();
        """),
    'a bare call writes what the top level reads': a_program("""
        var q = 1;
        function f() { this.q = 2; }
        f();
        console.log(q);
        """),
}

#: A receiver that is not the global object, mapped to what a host prints for it. The property read
#: is the receiver's own, and a top-level name of the same spelling must not answer for it.
A_RECEIVER_THAT_IS_NOT_THE_GLOBAL_OBJECT = {
    'a method reads its own property': a_program("""
        var q = 1;
        var o = { q: 7, m: function () { return this.q; } };
        console.log(o.m());
        """),
    'a method writes its own property': a_program("""
        var q = 1;
        var o = { q: 7, m: function () { this.q = 9; return this.q; } };
        console.log(o.m(), q);
        """),
    'a constructed object reads its own property': a_program("""
        var q = 1;
        function C() { this.q = 7; }
        C.prototype.get = function () { return this.q; };
        console.log(new C().get());
        """),
    'a method reaches a reflective name of its own': a_program("""
        var o = {
          eval: function (s) { return 'method:' + s; },
          m: function () { return this.eval('x'); }
        };
        console.log(o.m());
        """),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestABareCallGivesItsBodyTheGlobalObject(TestBase):
    """
    Read under the script execution model, which is the one the unit runs by default and the only
    one in which a top-level declaration is a property of the global object at all. The module
    oracle cannot see these: there both sides throw.
    """

    def test_every_program_behaves_the_way_the_host_does(self):
        for label, source in A_BARE_CALL_READS_THE_GLOBAL_OBJECT.items():
            with self.subTest(label):
                before, after = before_and_after_in_a_host(source)
                self.assertEqual(after, before)


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAReceiverThatIsNotTheGlobalObjectKeepsItsOwnProperty(TestBase):
    """
    The cost side of admitting every `this`. Each program has a top-level name spelled the same as
    the property its receiver holds, so a reader that took the receiver for the global object would
    answer with the wrong one of the two.
    """

    def test_every_program_behaves_the_way_the_host_does(self):
        for label, source in A_RECEIVER_THAT_IS_NOT_THE_GLOBAL_OBJECT.items():
            with self.subTest(label):
                before, after = before_and_after_in_a_host(source)
                self.assertEqual(after, before)
