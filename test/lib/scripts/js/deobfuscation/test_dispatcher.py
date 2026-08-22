from __future__ import annotations

import unittest

from test import TestBase
from test.lib.scripts.js.analysis.differential import node_executable
from test.lib.scripts.js.deobfuscation import TestJsDeobfuscator
from test.lib.scripts.js.ledger import before_and_after, each_program_still_prints

from refinery.lib.scripts.js.analysis.cache import ModelCache
from refinery.lib.scripts.js.deobfuscation.dispatcher import JsDispatcherUnwrapper
from refinery.lib.scripts.js.model import JsFunctionDeclaration
from refinery.lib.scripts.js.parser import JsParser


def a_dispatcher(dict_lines: list, tail_lines: list) -> str:
    """
    The dispatcher scaffold: one function `d` that looks its callee up in a map by name and
    reads that callee's arguments out of the one shared payload array `p`. *dict_lines* spell
    the entries of the map and *tail_lines* the calls made through it.

    It is written here, where the law about unwrapping it is stated, and read from wherever
    else a program built around one is needed.
    """
    return '\n'.join((
        'var c = Object["create"](null);',
        'var p;',
        'function d(name, flag, rtype, lengths) {',
        '  var output;',
        '  var fns = {',
        *dict_lines,
        '  };',
        '  if (flag === "initF") { p = []; }',
        '  if (flag === "createF") {',
        '    output = c[name] || (c[name] = fns[name]);',
        '  } else {',
        '    output = fns[name]();',
        '  }',
        '  if (rtype === "wrapF") { return { "wk": output }; }',
        '  else { return output; }',
        '}',
        'function stub() {}',
        *tail_lines,
    ))


class TestDispatcherUnwrapping(TestJsDeobfuscator):

    def _make_dispatcher(self, dict_lines: list, tail_lines: list):
        return a_dispatcher(dict_lines, tail_lines)

    def test_single_function_direct_call(self):
        source = self._make_dispatcher(
            dict_lines=[
                '"abc": function() { var [x] = p; return x + 1; }'
            ],
            tail_lines=[
                'console.log((p = [5], d("abc")));'
            ]
        )
        self.assertEqual('console.log(6);', self._deobfuscate(source))

    def test_multi_function_dispatcher(self):
        source = self._make_dispatcher(
            dict_lines=[
                '"f1": function() { var [a, b] = p; return a + b; },',
                '"f2": function() { var [a, b] = p; return a * b; }',
            ],
            tail_lines=[
                'var x = (p = [2, 3], d("f1"));',
                'var y = (p = [x, 4], d("f2"));',
                'console.log(y);',
            ]
        )
        self.assertEqual('console.log(20);', self._deobfuscate(source))

    def test_wrapped_reference(self):
        source = self._make_dispatcher(
            dict_lines=[
                '"id": function() { var [x] = p; return x; }',
            ],
            tail_lines=[
                'var fn = new d("id", "createF", "wrapF")["wk"];',
                'console.log(fn(42));',
            ]
        )
        self.assertEqual('console.log(42);', self._deobfuscate(source))

    def test_boilerplate_removal(self):
        source = self._make_dispatcher(
            dict_lines=[
                '"k": function() { return 42; }',
            ],
            tail_lines=[
                'console.log(d("k"));'
            ]
        )
        result = self._deobfuscate(source)
        self.assertEqual('console.log(42);', result)

    def test_unwrapping_invalidates_shared_model_cache(self):
        """
        Unwrapping removes the dispatcher's empty stub function through the shared analysis cache.
        When neither the payload nor the cache declarator is in removable form, that stub deletion
        is the unwrapper's only boilerplate removal, so the removal itself must invalidate the shared
        cache: a later transform sharing the cache must not observe a model that still declares the
        already-deleted stub.
        """
        source = '\n'.join((
            'var c = {};',
            'var p = 0;',
            'function d(name, flag, rtype, lengths) {',
            '  var output;',
            '  var fns = {',
            '    "k": function() { return 42; }',
            '  };',
            '  if (flag === "initF") { p = []; }',
            '  if (flag === "createF") {',
            '    output = c[name] || (c[name] = fns[name]);',
            '  } else {',
            '    output = fns[name]();',
            '  }',
            '  if (rtype === "wrapF") { return { "wk": output }; }',
            '  else { return output; }',
            '}',
            'function stub() {}',
            'console.log(d("k"));',
        ))
        ast = JsParser(source).parse()
        cache = ModelCache(ast)
        transformer = JsDispatcherUnwrapper()
        transformer.models = cache
        transformer.visit(ast)
        remaining = sorted(
            node.id.name
            for node in ast.walk()
            if isinstance(node, JsFunctionDeclaration) and node.id is not None
        )
        self.assertEqual(['k'], remaining)
        self.assertIsNone(cache.model.lookup('stub', cache.model.root_scope))


#: A program that dispatches through a callee name built by an expression rather than written as a
#: string literal, mapped to what Node prints for it. The dispatcher is reached the ordinary way and
#: answers the ordinary way; only the spelling of the name keeps the unwrapper from reading it.
A_DISPATCH_THROUGH_A_COMPUTED_KEY = {
    a_dispatcher(
        dict_lines=['"f1": function() { var [a, b, c] = p; return a + b + c; }'],
        tail_lines=[
            'var k = "f" + 1;',
            'console.log((p = ["a", "b", "c"], d(k)));',
        ],
    ): 'abc\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestADispatcherIsKeptWhileACallToItSurvives(TestBase):
    """
    Unwrapping a dispatcher extracts the functions its table holds, replaces each call made through
    it with a direct call to the function that call selects, and removes the declaration. A call
    whose callee name is not written as a string literal is one the pass cannot read, and the whole
    unwrap is refused where one is present rather than the removal alone.

    Refusing the whole of it is what extraction costs: the extracted function reuses the statement
    nodes of the table entry it is built from and strips the payload destructuring out of them in
    place, so a dispatcher left standing beside the extracted functions would call bodies that no
    longer read the payload it writes.
    """

    def test_a_dispatch_through_a_computed_key_keeps_the_dispatcher(self):
        """
        Node prints `abc` for the one program of `A_DISPATCH_THROUGH_A_COMPUTED_KEY`, which spells
        its callee name `"f" + 1`, and the deobfuscation prints it too.
        """
        rows = A_DISPATCH_THROUGH_A_COMPUTED_KEY
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )


#: A program that installs a value at the index a dispatcher's payload was written with no element
#: in, and dispatches through it, mapped to what Node prints for it. The dispatcher reaches its
#: callee's arguments by reading them off the payload, so a position the payload does not hold is
#: read from the prototype chain like any other.
A_DISPATCH_WHOSE_PAYLOAD_HOLDS_A_HOLE = {
    "Array.prototype[1] = 'X';\n" + a_dispatcher(
        dict_lines=['"f1": function() { var [a, b, c] = p; return a + b + c; }'],
        tail_lines=['console.log((p = ["first", , "third"], d("f1")));'],
    ): 'firstXthird\n',
}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestADispatcherPayloadHoleIsTheChainsToAnswer(TestBase):
    """
    Unwrapping a dispatcher into direct calls has to spell out each argument the payload carried,
    and a position the payload was written with no element in is not one it carries: what reaches
    the callee there is whatever the prototype chain answers for that index. Spelling it `undefined`
    is right only while that chain is the one the language describes, so the rewrite asks the effect
    model whether it still is, and refuses the dispatch where it is not.

    This is the law `test.lib.scripts.js.deobfuscation.test_inherited_reads` states of a read, at a
    site that decides it in the syntax rather than in the interpreter. The same program with the
    prototype left alone is `test_dispatcher_sparse_payload_preserves_arity` and prints
    `firstundefinedthird`, which is what the refusal here must not cost.

    Retired from `test.lib.scripts.js.test_release_blockers` and kept as the regression it retired
    into.
    """

    def test_a_payload_position_no_element_was_written_in_finds_what_the_chain_holds(self):
        """
        Node prints `firstXthird` for the one program of `A_DISPATCH_WHOSE_PAYLOAD_HOLDS_A_HOLE`,
        which writes `X` at index 1 of `Array.prototype` before dispatching a payload whose second
        position holds no element, and the deobfuscation prints it too.
        """
        rows = A_DISPATCH_WHOSE_PAYLOAD_HOLDS_A_HOLE
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )
