from __future__ import annotations

import inspect
import unittest

from test.lib.scripts.js.analysis.differential import behavior, node_executable
from test.lib.scripts.js.deobfuscation import TestJsDeobfuscator

from refinery.lib.scripts.js.deobfuscation.iifeaccessor import JsIIFEAccessorPromoter


class TestIIFEAccessorPromoter(TestJsDeobfuscator):

    def _promote(self, source: str) -> str:
        return self._run_transformer(source, JsIIFEAccessorPromoter)

    def test_promotes_simple_accessor(self):
        source = inspect.cleandoc(
            """
            var get = function () {
                var data = [[72, 105], [66, 121, 101]];
                return function (i) { return data[i]; };
            }();
            """
        )
        result = self._promote(source)
        self.assertEqual(
            inspect.cleandoc(
                """
                function get(i) {
                  var data = [[72, 105], [66, 121, 101]];
                  return data[i];
                }
                """
            ),
            result,
        )

    def test_fold_xor_accessor_pattern_end_to_end(self):
        source = inspect.cleandoc(
            """
            var get = function () {
                var data = [[72, 105], [66, 121, 101]];
                var shift = 28;
                var mask = 42;
                return function (i) {
                    var a = data[i];
                    if (!a) return "";
                    var r = "";
                    for (var j = 0; j < a.length; j++) {
                        var k = j >> shift & j << mask & (shift ^ shift) & 2047;
                        r += String.fromCharCode(a[j] ^ k);
                    }
                    return r;
                };
            }();
            document.write(get(0));
            document.write(get(1));
            """
        )
        result = self._deobfuscate_iterative(source)
        self.assertEqual(
            inspect.cleandoc(
                """
                document.write('Hi');
                document.write('Bye');
                """
            ),
            result,
        )

    def test_does_not_promote_when_closure_is_mutated(self):
        source = inspect.cleandoc(
            """
            var counter = function() {
              var n = 0;
              return function() {
                n++;
                return n;
              };
            }();
            """
        )
        self.assertEqual(source, self._promote(source))

    def test_does_not_promote_when_param_collides_with_closure(self):
        source = inspect.cleandoc(
            """
            var get = function() {
              var data = [1, 2, 3];
              return function(data) {
                return data;
              };
            }();
            """
        )
        self.assertEqual(source, self._promote(source))

    def test_does_not_promote_non_literal_closure(self):
        source = inspect.cleandoc(
            """
            var get = function() {
              var data = computeData();
              return function(i) {
                return data[i];
              };
            }();
            """
        )
        self.assertEqual(source, self._promote(source))

    def test_promotes_through_parenthesised_iife(self):
        source = inspect.cleandoc(
            """
            var get = (function () {
                var data = [1, 2, 3];
                return function (i) { return data[i]; };
            })();
            """
        )
        result = self._promote(source)
        self.assertEqual(
            inspect.cleandoc(
                """
                function get(i) {
                  var data = [1, 2, 3];
                  return data[i];
                }
                """
            ),
            result,
        )

    def test_does_not_promote_self_referencing_named_function(self):
        source = inspect.cleandoc(
            """
            var get = function() {
              var data = [1, 2, 3];
              return function rec(i) {
                return i <= 0 ? data[0] : rec(i - 1);
              };
            }();
            """
        )
        self.assertEqual(source, self._promote(source))

    def test_promotes_when_arguments_used_only_in_nested_function(self):
        source = inspect.cleandoc(
            """
            var get = function () {
                var data = [1, 2, 3];
                return function (i) {
                    var inner = function () { return arguments[0]; };
                    return data[i];
                };
            }();
            """
        )
        result = self._promote(source)
        self.assertEqual(
            inspect.cleandoc(
                """
                function get(i) {
                  var data = [1, 2, 3];
                  var inner = function() {
                    return arguments[0];
                  };
                  return data[i];
                }
                """
            ),
            result,
        )

    def test_promotes_when_inner_function_contains_class_field_this(self):
        source = inspect.cleandoc(
            """
            var get = function () {
                var data = [1, 2, 3];
                return function (i) {
                    class Helper { value = this.x; }
                    return data[i];
                };
            }();
            """
        )
        result = self._promote(source)
        self.assertEqual(
            inspect.cleandoc(
                """
                function get(i) {
                  var data = [1, 2, 3];
                  class Helper {
                    value = this.x;
                  }
                  return data[i];
                }
                """
            ),
            result,
        )

    def test_the_closure_declarations_are_written_behind_the_inner_prologue(self):
        """
        The promoted function keeps the mode its inner body declared: the closure's declarations go
        below the Directive Prologue that body opens with, since a declaration written above the
        `'use strict'` would end the prologue and leave the function sloppy.
        """
        source = inspect.cleandoc(
            """
            var get = function () {
                var data = [[72, 105]];
                return function (i) { 'use strict'; return data[i]; };
            }();
            """
        )
        self.assertEqual(
            inspect.cleandoc(
                """
                function get(i) {
                  'use strict';
                  var data = [[72, 105]];
                  return data[i];
                }
                """
            ),
            self._promote(source),
        )

    def test_the_whole_prologue_is_stepped_over_and_not_the_directive_alone(self):
        """
        A directive the language does not recognize is one all the same, and a declaration wedged in
        front of it would end the run for the `'use strict'` standing ahead of it as well.
        """
        source = inspect.cleandoc(
            """
            var get = function () {
                var data = [[72, 105]];
                return function (i) { 'use strict'; 'other'; return data[i]; };
            }();
            """
        )
        self.assertEqual(
            inspect.cleandoc(
                """
                function get(i) {
                  'use strict';
                  'other';
                  var data = [[72, 105]];
                  return data[i];
                }
                """
            ),
            self._promote(source),
        )

    def test_a_body_that_opens_with_no_prologue_takes_the_declarations_at_its_head(self):
        source = inspect.cleandoc(
            """
            var get = function () {
                var data = [[72, 105]];
                return function (i) { return data[i]; };
            }();
            """
        )
        self.assertEqual(
            inspect.cleandoc(
                """
                function get(i) {
                  var data = [[72, 105]];
                  return data[i];
                }
                """
            ),
            self._promote(source),
        )


#: An accessor whose closure returns a function of each kind, mapped to the declaration the
#: promotion writes for it and to what Node prints when the program calls it. What a call answers is
#: decided by the kind: a plain function answers the value, an async one a promise, a generator an
#: iterator, and an async generator an iterator of promises. A declaration written without the kind
#: would answer something else, and one written without the `*` over a body holding a `yield` is not
#: even a text an engine reads.
_AN_ACCESSOR_RETURNING_A_FUNCTION_OF_EACH_KIND: dict[str, tuple[str, str]] = {
    'var get = function () { var d = [1, 2]; return function (i) { return d[i]; }; }();'
    ' console.log(get(0));': (
        inspect.cleandoc(
            """
            function get(i) {
              var d = [1, 2];
              return d[i];
            }
            console.log(get(0));
            """
        ), '1\n'),
    'var get = function () { var d = [1, 2]; return async function (i) { return d[i]; }; }();'
    ' get(0).then(function (v) { console.log(v); });': (
        inspect.cleandoc(
            """
            async function get(i) {
              var d = [1, 2];
              return d[i];
            }
            get(0).then(function(v) {
              console.log(v);
            });
            """
        ), '1\n'),
    'var get = function () { var d = [1, 2]; return function* (i) { yield d[i]; }; }();'
    ' console.log(get(0).next().value);': (
        inspect.cleandoc(
            """
            function* get(i) {
              var d = [1, 2];
              yield d[i];
            }
            console.log(get(0).next().value);
            """
        ), '1\n'),
    'var get = function () { var d = [1, 2]; return async function* (i) { yield d[i]; }; }();'
    ' get(0).next().then(function (v) { console.log(v.value); });': (
        inspect.cleandoc(
            """
            async function* get(i) {
              var d = [1, 2];
              yield d[i];
            }
            get(0).next().then(function(v) {
              console.log(v.value);
            });
            """
        ), '1\n'),
}


class TestAPromotedAccessorIsTheKindOfFunctionItReturned(TestJsDeobfuscator):

    def _promote(self, source: str) -> str:
        return self._run_transformer(source, JsIIFEAccessorPromoter)

    def test_the_declaration_is_written_the_way_the_row_records(self):
        rows = _AN_ACCESSOR_RETURNING_A_FUNCTION_OF_EACH_KIND
        self.assertEqual(
            {source: self._promote(source) for source in rows},
            {source: promoted for source, (promoted, _) in rows.items()},
        )

    @unittest.skipIf(node_executable() is None, 'node.js is not available')
    def test_node_prints_the_same_before_and_after_the_promotion(self):
        rows = _AN_ACCESSOR_RETURNING_A_FUNCTION_OF_EACH_KIND
        self.assertEqual(
            {source: (behavior(source), behavior(self._promote(source))) for source in rows},
            {
                source: ((printed, None), (printed, None))
                for source, (_, printed) in rows.items()
            },
        )
