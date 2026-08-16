"""
An `arguments` object whose elements alias the parameters of the function that has it.

Such an object is not a copy of the call's arguments: element `i` and parameter `i` are one location.
Writing the parameter is read back off the object, and writing the object is read back under the
parameter's name, so a write to either is a write neither a sweep over the parameter's uses nor a
sweep over the object's can see alone.

Which functions have one was measured with Node, across every axis below, and the measurement is what
each corpus records:

- The mode the body runs in. A sloppy body has the aliasing and a strict one an independent copy, and
  a body comes to be strict through its own Directive Prologue, through the prologue of the script or
  of any function around it, through standing anywhere in a class definition, or through being module
  code, which is strict with nothing saying so. A string that is not a directive changes no mode: one
  behind a statement, one written with an escape, one in parentheses and one inside a nested block
  all leave the body sloppy, however a rewrite that moves or unwraps it spells it afterwards.
- The kind of function. A declaration, an expression, a generator, an async function, a method, a
  setter, and a body the `Function` constructor compiles all have an object of their own. An arrow has
  none and reads the one belonging to the function around it; a nested function has its own, and the
  enclosing parameters are not what its elements alias.
- The shape of the parameter list. Aliasing needs a list of plain names: a default, a rest element or
  a destructuring pattern anywhere in the list gives every parameter in it an independent copy, and a
  list of no parameters has nothing for an element to alias. Two parameters spelled with one name are
  still a list of plain names.
- How the object is reached. `arguments[0]`, `arguments['0']` and a key computed at runtime name a
  parameter; `arguments.length`, `arguments.callee`, a key that is no index and an index past the end
  of the list observe no parameter's value; and a spread, an `Array.from`, an `apply`, a second name
  bound to the object and the object handed to a call put every parameter in reach. A write lands
  through the same spellings, except that deleting an element first breaks the link the write would
  otherwise have travelled.
- Where the reference stands. A block, a loop, a `try`, a `with`, an arrow at any depth and a direct
  `eval` all reach the enclosing object, whatever mode the eval's own code declares. A nested regular
  function introduces its own, and an arrow inside that nested function reads the nested one.

Every absolute value here is Node's. Each corpus is asserted twice: once against Node, which is what
makes the recorded value a measurement rather than a claim, and once against the text the
deobfuscation produces, which is the law.
"""
from __future__ import annotations

import inspect
import unittest

from test import TestBase
from test.lib.scripts.js.analysis.differential import behavior, node_executable

from refinery.units.scripting.js import js


def _deobfuscated(source: str, *, module: bool = False) -> str:
    return source.encode('utf8') | js(module=module) | str


def _printed(rows: dict[str, str]) -> dict[str, tuple[str, str | None]]:
    """
    What Node has to make of each program of *rows*: the text the row records, with nothing thrown.
    """
    return {source: (printed, None) for source, printed in rows.items()}


def _said_by_node(rows: dict[str, str]) -> dict[str, tuple[str, str | None]]:
    return {source: behavior(source) for source in rows}


def _said_after_deobfuscation(rows: dict[str, str]) -> dict[str, tuple[str, str | None]]:
    return {source: behavior(_deobfuscated(source)) for source in rows}


#: A sloppy function whose parameter list is a run of plain names, mapped to what Node prints for it.
#: One name is written and the other spelling of it is read, in both directions, so `2` is the value
#: the body wrote and `1` would be the one the call passed.
_A_LIST_OF_PLAIN_NAMES_IN_A_SLOPPY_BODY: dict[str, str] = {
    'function f(a) { a = 2; return arguments[0]; } console.log(f(1));': '2\n',
    'function f(a) { arguments[0] = 2; return a; } console.log(f(1));': '2\n',
    'function f(a, b) { b = 2; return arguments[1]; } console.log(f(0, 1));': '2\n',
    'function f(a, b) { arguments[1] = 2; return b; } console.log(f(0, 1));': '2\n',
    'var f = function (a) { a = 2; return arguments[0]; }; console.log(f(1));': '2\n',
    'var f = function g(a) { arguments[0] = 2; return a; }; console.log(f(1));': '2\n',
    'function* f(a) { a = 2; yield arguments[0]; } console.log(f(1).next().value);': '2\n',
    'async function f(a) { arguments[0] = 2; return a; } f(1).then(function (v) { console.log(v); });': '2\n',
    'var o = { m(a) { a = 2; return arguments[0]; } }; console.log(o.m(1));': '2\n',
    'var o = { set s(a) { arguments[0] = 2; console.log(a); } }; o.s = 1;': '2\n',
    'function f(a, a) { a = 2; return arguments[1]; } console.log(f(0, 1));': '2\n',
    "'use strict'; var f = new Function('a', 'a = 2; return arguments[0];'); console.log(f(1));": '2\n',
}

#: A function whose object aliases no parameter, mapped to what Node prints for it. `1` is the value
#: the call passed, which is what a read answers when the write it should have been read back from
#: reached an independent location.
_AN_OBJECT_THAT_ALIASES_NOTHING: dict[str, str] = {
    "function f(a) { 'use strict'; a = 2; return arguments[0]; } console.log(f(1));": '1\n',
    'function f(a) { "use strict"; arguments[0] = 2; return a; } console.log(f(1));': '1\n',
    "'use strict'; function f(a) { a = 2; return arguments[0]; } console.log(f(1));": '1\n',
    "function o() { 'use strict'; function f(a) { a = 2; return arguments[0]; } return f(1); } console.log(o());": '1\n',
    'class C { m(a) { a = 2; return arguments[0]; } } console.log(new C().m(1));': '1\n',
    'class C { m() { function f(a) { a = 2; return arguments[0]; } return f(1); } } console.log(new C().m());': '1\n',
    'function f(a = 0) { a = 2; return arguments[0]; } console.log(f(1));': '1\n',
    'function f(...a) { a[0] = 2; return arguments[0]; } console.log(f(1));': '1\n',
    'function f([a]) { arguments[0] = 2; return a; } console.log(f([1]));': '1\n',
    'function f({a}) { arguments[0] = 2; return a; } console.log(f({a: 1}));': '1\n',
    'function f(a, b = 0) { a = 2; return arguments[0]; } console.log(f(1, 0));': '1\n',
    'function f(a, ...b) { arguments[0] = 2; return a; } console.log(f(1));': '1\n',
    'function f() { var a = 1; arguments[0] = 2; return a; } console.log(f(1));': '1\n',
    'function f(a) { var g = (b) => { b = 2; return arguments[0]; }; return g(9); } console.log(f(1));': '1\n',
    'function f(a) { function g(b) { arguments[0] = 2; return a; } return g(9); } console.log(f(1));': '1\n',
}

#: A `use strict` that declares nothing, mapped to what Node prints for the sloppy function standing
#: beside it. A directive is a statement whose expression is the literal and which opens the list it
#: stands in, so a parenthesized one, one behind an ordinary statement, one written with an escape and
#: one inside a nested block are all ordinary string-valued statements.
_A_USE_STRICT_THAT_GOVERNS_NOTHING: dict[str, str] = {
    "function f(a) { ('use strict'); a = 2; return arguments[0]; } console.log(f(1));": '2\n',
    "function f(a) { 0; 'use strict'; a = 2; return arguments[0]; } console.log(f(1));": '2\n',
    "function f(a) { 'use\\u0020strict'; a = 2; return arguments[0]; } console.log(f(1));": '2\n',
    "function f(a) { { 'use strict'; } a = 2; return arguments[0]; } console.log(f(1));": '2\n',
    "function f(a) { if (true) { 'use strict'; } a = 2; return arguments[0]; } console.log(f(1));": '2\n',
    "{ 'use strict'; } function f(a) { a = 2; return arguments[0]; } console.log(f(1));": '2\n',
    "if (true) { 'use strict'; } function f(a) { a = 2; return arguments[0]; } console.log(f(1));": '2\n',
    "var s = 'use strict'; function f(a) { a = 2; return arguments[0]; } console.log(f(1));": '2\n',
}

#: An expression that reads through the object, mapped to what Node prints when a sloppy function of
#: one parameter returns it after writing that parameter. `2` is the written value read back, `1` is a
#: count rather than a value, and `undefined` is a key the list holds no element at.
_A_READ_THROUGH_THE_OBJECT: dict[str, str] = {
    "arguments['0']"                                : '2\n',
    "arguments['0' + '']"                           : '2\n',
    'arguments[i]'                                  : '2\n',
    "arguments['00']"                               : 'undefined\n',
    'arguments[5]'                                  : 'undefined\n',
    'arguments.length'                              : '1\n',
    'arguments.callee.length'                       : '1\n',
    'Object.keys(arguments).length'                 : '1\n',
    '[...arguments][0]'                             : '2\n',
    'Array.from(arguments)[0]'                      : '2\n',
    '((b) => b[0])(arguments)'                      : '2\n',
    '(function (x) { return x; }).apply(null, arguments)': '2\n',
    'JSON.stringify(arguments)'                     : '{"0":2}\n',
}

#: A statement that writes through the object, mapped to what Node prints when a sloppy function of
#: one parameter runs it and then returns the parameter. A key that is no index, an index past the end
#: of the list and the length are locations no parameter stands at, and a deleted element is one whose
#: link to its parameter is gone before the write lands.
_A_WRITE_THROUGH_THE_OBJECT: dict[str, str] = {
    "arguments['0'] = 2"                : '2\n',
    'arguments[i] = 2'                  : '2\n',
    'arguments[0]++'                    : '2\n',
    'arguments[0] += 1'                 : '2\n',
    'var b = arguments; b[0] = 2'       : '2\n',
    'Object.assign(arguments, {0: 2})'  : '2\n',
    "arguments['00'] = 2"               : '1\n',
    'arguments[5] = 2'                  : '1\n',
    'arguments.length = 0'              : '1\n',
    'delete arguments[0]; arguments[0] = 2': '1\n',
}

#: The tail of a sloppy one-parameter body that has already written its parameter, mapped to what Node
#: prints for it. `2` is the enclosing object answering; `7` is a nested function's own object, which
#: holds the argument that nested call was made with and no parameter of the function around it.
_WHERE_THE_REFERENCE_TO_THE_OBJECT_STANDS: dict[str, str] = {
    '{ return arguments[0]; }'                                          : '2\n',
    'for (var q = 0; q < 1; q++) { return arguments[0]; }'              : '2\n',
    'try { return arguments[0]; } finally { }'                          : '2\n',
    'with ({}) { return arguments[0]; }'                                : '2\n',
    'return (() => { return arguments[0]; })();'                        : '2\n',
    'return (() => arguments[0])();'                                    : '2\n',
    'return (() => (() => arguments[0])())();'                          : '2\n',
    'return ((x = arguments[0]) => x)();'                               : '2\n',
    "return eval('arguments[0]');"                                      : '2\n',
    'return eval("\'use strict\'; arguments[0]");'                      : '2\n',
    'return (function () { return arguments[0]; })(7);'                 : '7\n',
    'return (function () { return (() => arguments[0])(); })(7);'       : '7\n',
}

#: A program whose function has an object aliasing its parameters that nothing in it observes, mapped
#: to the text the deobfuscation writes for it. Each write is dead in fact and not merely unread by
#: name, so the tool is free to remove it and does.
_AN_ALIASING_NOTHING_OBSERVES: dict[str, str] = {
    'function f(a) { a = 2; return a; } console.log(f(1));':
        'console.log(2);',
    'function f(a, b) { a = 2; b = 3; return a + b; } console.log(f(0, 0));':
        'console.log(5);',
    'function f(a) { a = 2; return arguments.length; } console.log(f(1));': inspect.cleandoc(
        """
        function f(a) {
          return arguments.length;
        }
        console.log(f(1));
        """
    ),
    'function f(a, b) { b = 7; return arguments[0]; } console.log(f(1, 0));': inspect.cleandoc(
        """
        function f(a, b) {
          return arguments[0];
        }
        console.log(f(1, 0));
        """
    ),
    "function f(a) { 'use strict'; a = 2; return arguments[0]; } console.log(f(1));":
        inspect.cleandoc(
            """
            function f(a) {
              'use strict';
              return arguments[0];
            }
            console.log(f(1));
            """
        ),
    'function f(a = 0) { a = 2; return arguments[0]; } console.log(f(1));': inspect.cleandoc(
        """
        function f(a = 0) {
          return arguments[0];
        }
        console.log(f(1));
        """
    ),
}

#: A program whose write the aliasing carries to the other name, mapped to the text the deobfuscation
#: writes for it. Each is a row of `_AN_ALIASING_NOTHING_OBSERVES` with the object brought into it, so
#: that what the tool does with the aliasing is pinned as text and not only as an answer Node gives.
_AN_ALIASING_THAT_IS_OBSERVED: dict[str, str] = {
    'function f(a) { a = 2; return arguments[0]; } console.log(f(1));': inspect.cleandoc(
        """
        function f(a) {
          a = 2;
          return arguments[0];
        }
        console.log(f(1));
        """
    ),
    'function f(a) { arguments[0] = 2; return a; } console.log(f(1));': inspect.cleandoc(
        """
        function f(a) {
          arguments[0] = 2;
          return a;
        }
        console.log(f(1));
        """
    ),
}

#: A module, which is strict code with no directive saying so, mapped to what Node prints for it. The
#: same body spelled in a script prints `2`, so the file's kind is the whole of the difference.
_A_MODULE_IS_STRICT_WITH_NOTHING_SAYING_SO: dict[str, str] = {
    'function f(a) { a = 2; return arguments[0]; } export const v = f(1); console.log(v);': '1\n',
    'function f(a) { arguments[0] = 2; return a; } export const v = f(1); console.log(v);': '1\n',
}


def _reading(expression: str) -> str:
    return F'var i = 0; function f(a) {{ a = 2; return {expression}; }} console.log(f(1));'


def _writing(statement: str) -> str:
    return F'var i = 0; function f(a) {{ {statement}; return a; }} console.log(f(1));'


def _standing(tail: str) -> str:
    return F'function f(a) {{ a = 2; {tail} }} console.log(f(1));'


_READS = {_reading(k): v for k, v in _A_READ_THROUGH_THE_OBJECT.items()}

_WRITES = {_writing(k): v for k, v in _A_WRITE_THROUGH_THE_OBJECT.items()}

_PLACES = {_standing(k): v for k, v in _WHERE_THE_REFERENCE_TO_THE_OBJECT_STANDS.items()}


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestASloppyListOfPlainNamesIsAliasedByTheObject(TestBase):

    def test_node_reads_back_the_value_written_under_the_other_name(self):
        rows = _A_LIST_OF_PLAIN_NAMES_IN_A_SLOPPY_BODY
        self.assertEqual(_said_by_node(rows), _printed(rows))

    def test_the_deobfuscation_reads_it_back_too(self):
        rows = _A_LIST_OF_PLAIN_NAMES_IN_A_SLOPPY_BODY
        self.assertEqual(_said_after_deobfuscation(rows), _printed(rows))


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAFunctionWhoseObjectAliasesNothing(TestBase):

    def test_node_answers_each_read_with_the_value_the_call_passed(self):
        rows = _AN_OBJECT_THAT_ALIASES_NOTHING
        self.assertEqual(_said_by_node(rows), _printed(rows))

    def test_the_deobfuscation_answers_it_the_same_way(self):
        rows = _AN_OBJECT_THAT_ALIASES_NOTHING
        self.assertEqual(_said_after_deobfuscation(rows), _printed(rows))


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAUseStrictThatIsNoDirectiveLeavesTheBodySloppy(TestBase):

    def test_node_leaves_the_body_aliased(self):
        rows = _A_USE_STRICT_THAT_GOVERNS_NOTHING
        self.assertEqual(_said_by_node(rows), _printed(rows))

    def test_the_deobfuscation_leaves_it_aliased_too(self):
        rows = _A_USE_STRICT_THAT_GOVERNS_NOTHING
        self.assertEqual(_said_after_deobfuscation(rows), _printed(rows))


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestHowTheObjectIsReached(TestBase):

    def test_node_answers_each_read_the_way_the_row_records(self):
        self.assertEqual(_said_by_node(_READS), _printed(_READS))

    def test_the_deobfuscation_answers_each_read_the_same_way(self):
        self.assertEqual(_said_after_deobfuscation(_READS), _printed(_READS))

    def test_node_answers_each_write_the_way_the_row_records(self):
        self.assertEqual(_said_by_node(_WRITES), _printed(_WRITES))

    def test_the_deobfuscation_answers_each_write_the_same_way(self):
        self.assertEqual(_said_after_deobfuscation(_WRITES), _printed(_WRITES))


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestWhereTheReferenceToTheObjectStands(TestBase):

    def test_node_answers_each_placement_the_way_the_row_records(self):
        self.assertEqual(_said_by_node(_PLACES), _printed(_PLACES))

    def test_the_deobfuscation_answers_each_placement_the_same_way(self):
        self.assertEqual(_said_after_deobfuscation(_PLACES), _printed(_PLACES))


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestAModuleIsStrictCode(TestBase):

    def test_node_answers_the_module_with_the_value_the_call_passed(self):
        rows = _A_MODULE_IS_STRICT_WITH_NOTHING_SAYING_SO
        self.assertEqual(
            {source: behavior(source, module=True) for source in rows},
            _printed(rows),
        )

    def test_the_deobfuscation_answers_the_module_the_same_way(self):
        rows = _A_MODULE_IS_STRICT_WITH_NOTHING_SAYING_SO
        self.assertEqual(
            {
                source: behavior(_deobfuscated(source, module=True), module=True)
                for source in rows
            },
            _printed(rows),
        )


class TestTheAliasingIsHonouredWithoutForfeitingTheSimplification(TestBase):

    def test_a_write_the_aliasing_does_not_reach_is_removed(self):
        rows = _AN_ALIASING_NOTHING_OBSERVES
        self.assertEqual({source: _deobfuscated(source) for source in rows}, rows)

    def test_a_write_the_aliasing_reaches_is_left_standing(self):
        rows = _AN_ALIASING_THAT_IS_OBSERVED
        self.assertEqual({source: _deobfuscated(source) for source in rows}, rows)
