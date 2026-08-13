from __future__ import annotations

import inspect

from test import TestBase

from refinery.lib.scripts import Node
from refinery.lib.scripts.js.model import (
    JsArrayPattern,
    JsAssignmentPattern,
    JsErrorNode,
    JsIdentifier,
    JsObjectPattern,
    JsProperty,
    JsRestElement,
    JsVariableDeclaration,
)
from refinery.lib.scripts.js.parser import JsParser

LET_IS_A_NAME = 'let is a name'
PARSE_ERROR = 'parse error'
NOT_A_BINDING_TARGET = 'not a binding target'

LET_BEFORE_A_BRACKET_ACROSS_A_LINE_BREAK = inspect.cleandoc("""
    let
    [a] = [42];
""")

LET_BEFORE_A_BRACE_ACROSS_A_LINE_BREAK = inspect.cleandoc("""
    let
    {a} = {a: 42};
""")

LET_BEFORE_AN_EQUALS_ACROSS_A_LINE_BREAK = inspect.cleandoc("""
    let
    = 42;
""")

A_LET_LOOP_COUNTER_CAPTURED_BY_A_CLOSURE = inspect.cleandoc("""
    var fns = [];
    for (let i = 0; i < 3; i++) fns.push(function () { return i; });
    log = fns.map(function (f) { return f(); });
""")

A_VAR_LOOP_COUNTER_CAPTURED_BY_A_CLOSURE = inspect.cleandoc("""
    var fns = [];
    for (var i = 0; i < 3; i++) fns.push(function () { return i; });
    log = fns.map(function (f) { return f(); });
""")

THE_NAME_LET_BOUND_BY_VAR_AND_THEN_ASSIGNED = inspect.cleandoc("""
    var let = 42;
    let = 43;
    log.push(let);
""")

A_FUNCTION_NAMED_LET = inspect.cleandoc("""
    function let() { return 'F'; }
    log.push(let());
""")

LET_BEFORE_A_BINDING_NAME = [
    ('let a = 42;', ['let declares a']),
    ('let a;', ['let declares a']),
    ('let a = 1, b = 2;', ['let declares a, b']),
    ('let of = 42;', ['let declares of']),
    ('let async = 42;', ['let declares async']),
    ('let yield = 42;', ['let declares yield']),
]

LET_BEFORE_A_DESTRUCTURING_PATTERN = [
    ('let [a] = [42];', ['let declares a']),
    ('let [, a] = [1, 42];', ['let declares a']),
    ('let [a = 42] = [];', ['let declares a']),
    ('let [...a] = [42];', ['let declares a']),
    ('let {a} = {a: 42};', ['let declares a']),
    ('let {b: a} = {b: 42};', ['let declares a']),
    ('let {a = 42} = {};', ['let declares a']),
    ('let {...a} = {b: 42};', ['let declares a']),
    (LET_BEFORE_A_BRACKET_ACROSS_A_LINE_BREAK, ['let declares a']),
    (LET_BEFORE_A_BRACE_ACROSS_A_LINE_BREAK, ['let declares a']),
]

LET_BEFORE_WHAT_CANNOT_BE_BOUND = [
    ('let = 42;', [LET_IS_A_NAME]),
    ('let.x = 42;', [LET_IS_A_NAME]),
    ('let(42);', [LET_IS_A_NAME]),
    ('let;', [LET_IS_A_NAME]),
    ('let++;', [LET_IS_A_NAME]),
    (LET_BEFORE_AN_EQUALS_ACROSS_A_LINE_BREAK, [LET_IS_A_NAME]),
    ("let: log.push('LABEL');", [LET_IS_A_NAME]),
]

LET_WHERE_AN_OPERAND_IS_EXPECTED = [
    ('let a = let;', ['let declares a', LET_IS_A_NAME]),
    ('log.push(typeof let);', [LET_IS_A_NAME]),
    ('var x = let.x; log.push(x);', ['var declares x', LET_IS_A_NAME]),
    ('log.push(let(1));', [LET_IS_A_NAME]),
    ('(let[0]);', [LET_IS_A_NAME]),
    ('(let) = 42;', [LET_IS_A_NAME]),
]

LET_IN_A_FOR_HEADER_THAT_BINDS = [
    ('for (let i = 0; i < 3; i++) log.push(i);', ['let declares i']),
    ('for (let i = 0, j = 1; i < 1; i++) log.push(i, j);', ['let declares i, j']),
    ('for (let x of [42]) log.push(x);', ['let declares x']),
    ('for (let x in {p: 1}) log.push(x);', ['let declares x']),
    ('for (let [a, b] of [[1, 2]]) log.push(a, b);', ['let declares a, b']),
    ('for (let {a} of [{a: 42}]) log.push(a);', ['let declares a']),
    ('for (let of of [42]) log.push(of);', ['let declares of']),
    (A_LET_LOOP_COUNTER_CAPTURED_BY_A_CLOSURE, ['var declares fns', 'let declares i']),
]

LET_IN_A_FOR_HEADER_THAT_BINDS_NOTHING = [
    ('for (let; log.length < 1; ) log.push(let.x);', [LET_IS_A_NAME] * 2),
    ("for (let.x; log.length < 1; ) log.push('B');", [LET_IS_A_NAME]),
    ('for (let = 0; let < 2; let++) log.push(let);', [LET_IS_A_NAME] * 4),
    ('for (let in {p: 1}) log.push(let);', [LET_IS_A_NAME] * 2),
    ('for ((let) of [42]) log.push(let);', [LET_IS_A_NAME] * 2),
    ('for ((let) in {p: 1}) log.push(let);', [LET_IS_A_NAME] * 2),
]

VAR_AND_CONST_ALWAYS_DECLARE = [
    ('var a = 42;', ['var declares a']),
    ('const a = 42;', ['const declares a']),
    ('var [a] = [42];', ['var declares a']),
    ('const [a] = [42];', ['const declares a']),
    ('var {b: a} = {b: 42};', ['var declares a']),
    ('const {b: a} = {b: 42};', ['const declares a']),
    ('for (var x of [42]) log.push(x);', ['var declares x']),
    ('for (const x of [42]) log.push(x);', ['const declares x']),
    ('for (var x in {p: 1}) log.push(x);', ['var declares x']),
    ('if (true) var a = 42;', ['var declares a']),
    (A_VAR_LOOP_COUNTER_CAPTURED_BY_A_CLOSURE, ['var declares fns', 'var declares i']),
]

THE_WORD_LET_OWNED_BY_ANOTHER_BINDER = [
    (THE_NAME_LET_BOUND_BY_VAR_AND_THEN_ASSIGNED, ['var declares let'] + [LET_IS_A_NAME] * 2),
    ('for (var let of [42]) log.push(let);', ['var declares let', LET_IS_A_NAME]),
    (A_FUNCTION_NAMED_LET, [LET_IS_A_NAME] * 2),
]

LET_IN_A_BLOCK_OR_A_SINGLE_STATEMENT_BODY = [
    ('{ let a = 42; log.push(a); }', ['let declares a']),
    ('{ let.x = 42; log.push(let.x); }', [LET_IS_A_NAME] * 2),
    ('if (true) let.x = 42;', [LET_IS_A_NAME]),
]


class TestJsLetDeclarationOrName(TestBase):

    def _binding_names(self, target: Node | None) -> list[str]:
        if target is None:
            return []
        if isinstance(target, JsIdentifier):
            return [target.name]
        if isinstance(target, JsArrayPattern):
            return [name for e in target.elements for name in self._binding_names(e)]
        if isinstance(target, JsObjectPattern):
            return [name for p in target.properties for name in self._binding_names(p)]
        if isinstance(target, JsProperty):
            return self._binding_names(target.value)
        if isinstance(target, JsAssignmentPattern):
            return self._binding_names(target.left)
        if isinstance(target, JsRestElement):
            return self._binding_names(target.argument)
        return [NOT_A_BINDING_TARGET]

    def _let_readings(self, source: str) -> list[str]:
        """
        What the parser made of every `let`, `var` and `const` in the source, in the order they are
        written. A word that introduced a declaration is named by the keyword it spelled together
        with the names that declaration binds, a word that was read as an ordinary identifier is
        named by itself, and a place where the parser gave up by `PARSE_ERROR`. No two of those
        readings can compare equal, so neither a swapped reading nor a declaration that binds the
        wrong name can pass for the one that was meant.
        """
        tree = JsParser(source).parse()
        readings: list[tuple[int, str]] = []
        bound: set[int] = set()
        for node in tree.walk():
            if isinstance(node, JsVariableDeclaration):
                names: list[str] = []
                for declarator in node.declarations:
                    target = declarator.id
                    names.extend(self._binding_names(target))
                    if target is not None:
                        bound.update(id(inner) for inner in target.walk())
                readings.append((node.offset, F'{node.kind.value} declares {", ".join(names)}'))
            elif isinstance(node, JsErrorNode):
                readings.append((node.offset, PARSE_ERROR))
        for node in tree.walk():
            if isinstance(node, JsIdentifier) and id(node) not in bound:
                if node.name in ('let', 'var', 'const'):
                    readings.append((node.offset, F'{node.name} is a name'))
        return [reading for _, reading in sorted(readings)]

    def test_let_before_a_binding_name_declares_that_name(self):
        for source, expected in LET_BEFORE_A_BINDING_NAME:
            with self.subTest(source=source):
                self.assertEqual(self._let_readings(source), expected)

    def test_let_before_a_pattern_declares_only_the_names_that_pattern_binds(self):
        for source, expected in LET_BEFORE_A_DESTRUCTURING_PATTERN:
            with self.subTest(source=source):
                self.assertEqual(self._let_readings(source), expected)

    def test_let_before_something_that_cannot_be_bound_is_an_ordinary_name(self):
        for source, expected in LET_BEFORE_WHAT_CANNOT_BE_BOUND:
            with self.subTest(source=source):
                self.assertEqual(self._let_readings(source), expected)

    def test_let_where_an_operand_is_expected_is_an_ordinary_name(self):
        for source, expected in LET_WHERE_AN_OPERAND_IS_EXPECTED:
            with self.subTest(source=source):
                self.assertEqual(self._let_readings(source), expected)

    def test_let_in_a_for_header_declares_the_binding_that_follows_it(self):
        for source, expected in LET_IN_A_FOR_HEADER_THAT_BINDS:
            with self.subTest(source=source):
                self.assertEqual(self._let_readings(source), expected)

    def test_let_in_a_for_header_is_an_ordinary_name_when_it_binds_nothing(self):
        for source, expected in LET_IN_A_FOR_HEADER_THAT_BINDS_NOTHING:
            with self.subTest(source=source):
                self.assertEqual(self._let_readings(source), expected)

    def test_var_and_const_declare_whatever_follows_them(self):
        for source, expected in VAR_AND_CONST_ALWAYS_DECLARE:
            with self.subTest(source=source):
                self.assertEqual(self._let_readings(source), expected)

    def test_the_word_let_is_an_ordinary_name_where_another_binder_owns_it(self):
        for source, expected in THE_WORD_LET_OWNED_BY_ANOTHER_BINDER:
            with self.subTest(source=source):
                self.assertEqual(self._let_readings(source), expected)

    def test_a_block_and_a_single_statement_body_decide_a_let_the_same_way(self):
        for source, expected in LET_IN_A_BLOCK_OR_A_SINGLE_STATEMENT_BODY:
            with self.subTest(source=source):
                self.assertEqual(self._let_readings(source), expected)
