from __future__ import annotations

import inspect
import unittest

from test import TestBase
from test.lib.scripts.js.analysis.differential import node_executable
from test.lib.scripts.js.deobfuscation import TestJsDeobfuscator
from test.lib.scripts.js.ledger import Program, Reading, a_program, prints

from refinery.lib.scripts.js.deobfuscation.antidbg import JsRemoveSelfDefending


class TestAntiDebug(TestJsDeobfuscator):

    _DEFENSE_CODE = (
        "var a = (function() {"
        "  var b = true;"
        "  return function(c, d) {"
        "    var e = b ? function() {"
        "      if (d) { var f = d.apply(c, arguments); return d = null, f; }"
        "    } : function() {};"
        "    return b = false, e;"
        "  };"
        "}()), g = a(this, function() {"
        "  return g.toString().search('(((.+)+)+)+$')"
        "    .toString().constructor(g).search('(((.+)+)+)+$');"
        "});"
    )

    def test_remove_self_defending_redos(self):
        source = self._DEFENSE_CODE + (
            "g();"
            "console.log('hello');"
        )
        self.assertEqual(self._deobfuscate(source), "console.log('hello');")

    def test_preserves_code_without_redos(self):
        source = inspect.cleandoc(
            """
            var x = 1;
            console.log(x);
            """
        )
        self.assertEqual(source, self._run_transformer(source, JsRemoveSelfDefending))

    def test_redos_factory_preserved_when_referenced(self):
        source = self._DEFENSE_CODE + (
            "g();"
            "var other = a(this, function() { return 42; });"
            "console.log(other);"
        )
        self.assertEqual(
            inspect.cleandoc(
                """
                var a = function() {
                  var b = true;
                  return function(c, d) {
                    var e = b ? function() {
                      if (d) {
                        var f = d.apply(c, arguments);
                        return d = null, f;
                      }
                    } : function() {};
                    return b = false, e;
                  };
                }();
                var other = a(this, function() {
                  return 42;
                });
                console.log(other);
                """
            ),
            self._deobfuscate(source),
        )

    def test_factory_removed_despite_same_name_in_other_scope(self):
        source = (
            'var fac = function() { return 1; };'
            " var g = fac('(((.+)+)+)+$');"
            ' g();'
            ' function other() { var fac = 7; return fac; }'
            ' console.log(other());'
        )
        self.assertEqual(
            inspect.cleandoc(
                """
                function other() {
                  var fac = 7;
                  return fac;
                }
                console.log(other());
                """
            ),
            self._run_transformer(source, JsRemoveSelfDefending),
        )


class TestRemoveSelfDefendingStructural(TestJsDeobfuscator):
    """
    Structural removal: the same run-once `apply`-payload factory template, payloads that carry no ReDoS
    string, three guard invocation shapes, sequence-operand entanglement, and preservation when the
    factory result is stored and read but never invoked.
    """

    _FACTORY = (
        "var a = (function() {"
        "  var b = true;"
        "  return function(c, d) {"
        "    var e = b ? function() {"
        "      if (d) { var f = d.apply(c, arguments); return d = null, f; }"
        "    } : function() {};"
        "    return b = false, e;"
        "  };"
        "}());"
    )

    def test_console_hijack_payload_immediate_guard_is_removed(self):
        source = (
            self._FACTORY
            + "a(this, function() { console.log('hijack'); })();"
            + " console.log('done');"
        )
        self.assertEqual(
            "console.log('done');",
            self._run_transformer(source, JsRemoveSelfDefending),
        )

    def test_regexp_payload_iife_wrapped_guard_is_removed(self):
        source = (
            self._FACTORY
            + "(function() { a(this, function() { return /test/.test('test'); })(); })();"
            + " console.log('done');"
        )
        self.assertEqual(
            "console.log('done');",
            self._run_transformer(source, JsRemoveSelfDefending),
        )

    def test_iife_wrapped_guard_as_sequence_operand_removes_only_guard(self):
        source = (
            self._FACTORY
            + "(function() { a(this, function() {})(); })(), setInterval(function() {}, 1000);"
            + " console.log('main');"
        )
        self.assertEqual(
            inspect.cleandoc(
                """
                setInterval(function() {}, 1000);
                console.log('main');
                """
            ),
            self._run_transformer(source, JsRemoveSelfDefending),
        )

    def test_stored_guard_as_sequence_operand_removes_only_guard(self):
        source = (
            self._FACTORY
            + "var g = a(this, function() {});"
            + " g(), console.log('main');"
        )
        self.assertEqual(
            "console.log('main');",
            self._run_transformer(source, JsRemoveSelfDefending),
        )

    def test_stored_result_only_read_is_not_removed(self):
        source = (
            self._FACTORY
            + "var g = a(this, function() { return 42; });"
            + " console.log(g);"
        )
        self.assertEqual(
            self._run_transformers(source),
            self._run_transformer(source, JsRemoveSelfDefending),
        )


#: Classic scripts whose run-once wrapper spells the template's shapes for reasons of its own,
#: handed the global object as its receiver, mapped to the behavior a host gives them: the wrapper
#: runs its payload. The structural detector must not take any of them for the self-defending
#: template.
A_BENIGN_RUN_ONCE_WRAPPER = {
    'the conditional tests the payload parameter': Program(
        a_program("""
            var once = function (context, fn) {
              var run = fn
                ? function () { var r = fn.apply(context, arguments); fn = null; return r; }
                : function () {};
              return run;
            };
            var boot = once(this, function () { console.log('ran'); });
            boot();
            """),
        prints('ran'),
        Reading.SCRIPT,
    ),
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestABenignRunOnceWrapperKeepsRunning(TestBase):

    def test_every_program_behaves_the_way_the_host_does(self):
        for label, row in A_BENIGN_RUN_ONCE_WRAPPER.items():
            with self.subTest(label):
                self.assertEqual(row.read(), row.required())
