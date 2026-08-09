from __future__ import annotations

import inspect

from test.lib.scripts.js.deobfuscation import TestJsDeobfuscator

from refinery.lib.scripts.js.deobfuscation.restunpack import JsRestArrayUnpacking


class TestVariableDemasking(TestJsDeobfuscator):

    def _demask(self, source: str) -> str:
        return self._run_transformer(source, JsRestArrayUnpacking)

    def test_simple_two_params(self):
        self.assertEqual(
            inspect.cleandoc(
                """
                var f = function(p0, p1) {
                  return p0 + p1;
                };
                """
            ),
            self._demask('var f = function(...s) { s.length = 2; return s[0] + s[1]; }'),
        )

    def test_simple_zero_params_with_locals(self):
        self.assertEqual(
            inspect.cleandoc(
                """
                var f = function() {
                  var v0;
                  v0 = 10;
                  return v0;
                };
                """
            ),
            self._demask('var f = function(...s) { s.length = 0; s.a = 10; return s.a; }'),
        )

    def test_simple_negative_keys(self):
        self.assertEqual(
            inspect.cleandoc(
                """
                var f = function(p0) {
                  var v0;
                  v0 = p0 + 1;
                  return v0;
                };
                """
            ),
            self._demask(
                'var f = function(...s) { s.length = 1; s[-42] = s[0] + 1; return s[-42]; }'
            ),
        )

    def test_frame_qualified(self):
        self.assertEqual(
            inspect.cleandoc(
                """
                var NS = {};
                NS.fn = function(p0) {
                  return p0 * 2;
                };
                """
            ),
            self._demask(
                'var NS = {}; NS.fn = function(...r) { NS.F.stk.length = 1; return NS.F.stk[0] * 2; }'
            ),
        )

    def test_skips_unresolvable_access(self):
        self.assertEqual(
            inspect.cleandoc(
                """
                var f = function(...s) {
                  s.length = 1;
                  return s[x];
                };
                """
            ),
            self._demask('var f = function(...s) { s.length = 1; return s[x]; }'),
        )

    def test_skips_rest_param_aliased(self):
        self.assertEqual(
            inspect.cleandoc(
                """
                var f = function(...s) {
                  s.length = 1;
                  foo(s);
                  return s[0];
                };
                """
            ),
            self._demask('var f = function(...s) { s.length = 1; foo(s); return s[0]; }'),
        )

    def test_nested_function_unpacked_in_own_scope(self):
        source = inspect.cleandoc(
            """
            var outer = function(...s) {
              s.length = 1;
              s.x = function(...t) { t.length = 0; t.a = 5; return t.a; };
              return s[0] + s.x();
            }
            """
        )
        self.assertEqual(
            inspect.cleandoc(
                """
                var outer = function(p0) {
                  var v0;
                  v0 = function() {
                    var v0;
                    v0 = 5;
                    return v0;
                  };
                  return p0 + v0();
                };
                """
            ),
            self._demask(source),
        )

    def test_frame_qualified_missing_accesses_skipped(self):
        self.assertEqual(
            inspect.cleandoc(
                """
                var f = function(...r) {
                  A.B.C.length = 2;
                  return A.X.C[0];
                };
                """
            ),
            self._demask('var f = function(...r) { A.B.C.length = 2; return A.X.C[0]; }'),
        )

    def test_skips_rest_param_captured_by_closure(self):
        source = inspect.cleandoc(
            """
            var f = function(...s) {
              s.length = 1;
              var g = function() {
                return s[0] + 1;
              };
              return s[0] + g();
            };
            """
        )
        self.assertEqual(source, self._demask(source))

    def test_skips_rest_param_named_by_eval(self):
        source = inspect.cleandoc(
            """
            var f = function(...s) {
              s.length = 1;
              eval("s[2] = 9");
              return s[0];
            };
            """
        )
        self.assertEqual(source, self._demask(source))

    _REST_LENGTH_BEFORE_AN_UNRELATED_ONE = inspect.cleandoc(
        """
        var f = function(...s) {
          s.length = LENGTH;
          NS.F.stk.length = 1;
          return NS.F.stk[0] + s[0];
        };
        """
    )
    """
    A rest array whose own truncation is written first and an unrelated chain whose length is
    written after it, with the count the pass reads left open. The second statement is a truncation
    of a shape this pass does accept, so it is what a match that gives up on the first one lands on.
    """

    def _rest_length_before_an_unrelated_one(self, length: str) -> str:
        return self._REST_LENGTH_BEFORE_AN_UNRELATED_ONE.replace('LENGTH', length)

    def test_binds_the_first_length_written_to_the_rest_array(self):
        """
        The control the refusals below are read against. Two is a parameter count, so the rest array
        is unpacked into two parameters and the unrelated chain is left exactly where it was. Node
        prints `8 [7]` for both programs with `NS.F.stk` starting as `[7, 8, 9]` and `f(1, 2, 3)`.
        """
        self.assertEqual(
            inspect.cleandoc(
                """
                var f = function(p0, p1) {
                  NS.F.stk.length = 1;
                  return NS.F.stk[0] + p0;
                };
                """
            ),
            self._demask(self._rest_length_before_an_unrelated_one('2')),
        )

    def test_skips_oversized_truncation_without_binding_a_later_length_assignment(self):
        """
        A million is no parameter list, so the pattern is not this one. Reading on until some other
        `.length =` fits would unpack the function around a stack it never had, dropping the rest
        parameter that the surviving statements still name.
        """
        source = self._rest_length_before_an_unrelated_one('1000000')
        self.assertEqual(source, self._demask(source))

    def test_skips_negative_truncation_without_binding_a_later_length_assignment(self):
        source = self._rest_length_before_an_unrelated_one('-1')
        self.assertEqual(source, self._demask(source))

    def test_skips_fractional_truncation_without_binding_a_later_length_assignment(self):
        source = self._rest_length_before_an_unrelated_one('1.5')
        self.assertEqual(source, self._demask(source))

    def test_skips_unreadable_truncation_without_binding_a_later_length_assignment(self):
        """
        The count is a name rather than a number, so nothing about it is known. That is the same
        answer as an unusable number and not a reason to look for a different statement.
        """
        source = self._rest_length_before_an_unrelated_one('n')
        self.assertEqual(source, self._demask(source))

    def test_skips_oversized_frame_truncation_without_binding_a_later_length_assignment(self):
        """
        The same refusal when the oversized truncation is the frame-qualified form, where the
        candidate that follows is another chain rather than the rest parameter.
        """
        source = inspect.cleandoc(
            """
            var g = function(...r) {
              NS.F.stk.length = 1000000;
              OT.G.buf.length = 1;
              return OT.G.buf[0];
            };
            """
        )
        self.assertEqual(source, self._demask(source))
