"""
A call handed the global object may read and write any global.

A classic script's top-level declarations are properties of the global object, so a call the object
is passed to reaches every one of them under its own name — `a(globalThis, 'q')` lets `a` write the
`q` the file declares. No identifier in the file names that reference, so a walk over the text finds
none, and a remover that reads only what the text spells folds a value across the write and deletes a
declaration the callee still reads.

`refinery.lib.scripts.js.analysis.model.SemanticModel._record_global_object_alias_references` records
those references, the way the references a mapped `arguments` object makes are recorded against the
parameters it aliases. The cost rows below are the other half: what the object is not, and which
bindings it does not carry, both of which must still fold.

SECURITY: every program here is hand-authored in this file and benign. No sample and no stored
obfuscator fixture may be fed to this.
"""
from __future__ import annotations

import unittest

from test import TestBase
from test.lib.scripts.js.analysis.differential import node_executable
from test.lib.scripts.js.ledger import a_program, before_and_after_in_a_host, folded


#: A classic script handing the global object to a call, mapped to what a host prints for it.
A_CALL_IS_HANDED_THE_GLOBAL_OBJECT = {
    'a write through it reaches a declaration': a_program("""
        var q = 1;
        function a(g, k) { g[k] = 2; }
        a(globalThis, 'q');
        console.log(q);
        """),
    'the top level hands over its own this': a_program("""
        var q = 1;
        function a(g, k) { g[k] = 2; }
        a(this, 'q');
        console.log(q);
        """),
    'a read through it is a read of the declaration': a_program("""
        var q = 1;
        function a(g, k) { console.log(g[k]); }
        a(this, 'q');
        """),
    'a call through it is a call of the declaration': a_program("""
        var q = function () { console.log('q'); };
        function a(g, k) { g[k](); }
        a(globalThis, 'q');
        """),
    'a function declaration is a property too': a_program("""
        function q() { console.log('q'); }
        function a(g, k) { g[k](); }
        a(globalThis, 'q');
        """),
    'a new expression receives it as well': a_program("""
        var q = 1;
        function C(g) { g.q = 2; }
        new C(globalThis);
        console.log(q);
        """),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestACallHandedTheGlobalObjectReachesEveryGlobal(TestBase):
    """
    Read under the script execution model, which is the one the unit runs by default and the only one
    in which a top-level declaration is a property of the global object at all.
    """

    def test_every_program_behaves_the_way_the_host_does(self):
        for label, source in A_CALL_IS_HANDED_THE_GLOBAL_OBJECT.items():
            with self.subTest(label):
                before, after = before_and_after_in_a_host(source)
                self.assertEqual(after, before)


class TestWhatIsNotTheGlobalObjectStillFolds(TestBase):
    """
    The cost side. A reference is recorded for every global the file declares at each hand-over, so
    the two ways of admitting one too many are asked here: a local spelled like the object, and a
    binding the object does not carry.
    """

    def test_a_local_spelled_like_the_object_is_not_it(self):
        source = a_program("""
            var q = 1;
            function f(window) { console.log(window); }
            f(0);
            console.log(q);
            """)
        self.assertEqual(folded(source), a_program("""
            function f(window) {
              console.log(window);
            }
            f(0);
            console.log(1);
            """).rstrip(chr(10)))

    def test_the_object_carries_no_local_of_the_callee(self):
        source = a_program("""
            function a(g) { return g; }
            function m() { var t = 3; a(globalThis); return t; }
            console.log(m());
            """)
        self.assertEqual(folded(source), a_program("""
            function a(g) {
              return g;
            }
            function m() {
              a(globalThis);
              return 3;
            }
            console.log(m());
            """).rstrip(chr(10)))
