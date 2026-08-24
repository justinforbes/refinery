"""
A call to a function that wraps what its body returns answers the wrapper, never the value.

`async function m() { return 7; }` called answers a promise; `function* g() { return 7; }` called
answers a generator object whose completion value is `7` and which yields nothing at all. Neither is
`7`. A pass that reads such a body and answers the call with what the body returned hands back a
program that computes something else, and hands it back looking clean: nothing throws where the
program is only read, nothing is half-rewritten, and the analyst gets no signal that the answer is
not the one the language gives.

Eleven sites across ten passes do it. Each has its own entry here, and each entry is asked three
ways:

- the shape the pass exists for, folded, so that a guard which refuses too much is caught by the
  control rather than by silence — killing `stringarray._match_wrapper` outright leaves the rest of
  the suite green, and a site witnessed by nothing is a site a guard can quietly empty;
- the same shape with the keyword, read for what still stands in the output, which needs no engine
  and so is the reading that survives on a machine with no Node.js;
- the same shape run in Node before and after, which is what says the two agree.

**The read has to be one the two answers disagree on.** `typeof decode(P).then` for a *generator*
answers `undefined` in Node and here alike, because a generator object has no `.then` either, and the
row is then blind to a fold that is wrong the whole time. `.then` for `async`, `.next` for a generator
and for an async generator.

The tally `_what_wraps_and_what_calls_it` reads is deliberately its own and does not ask
`refinery.lib.scripts.js.model`: a test that asks the library whether the library is right reports
only that it is self-consistent.

SECURITY: every program here is hand-authored in this file and benign, except the base91 table and
decoder, which are the fixture `test.lib.scripts.js.deobfuscation.test_b91strings` builds and which
are written out in full below so that what the engine is handed is what is read here. No sample and
no stored obfuscator fixture may be fed to this.
"""
from __future__ import annotations

import unittest

from test import TestBase
from test.lib.scripts.js.analysis.differential import (
    deobfuscate_within,
    node_executable,
)
from test.lib.scripts.js.ledger import (
    before_and_after,
    each_program_still_prints,
    folded,
    well_formed,
)

from refinery.lib.scripts import Node
from refinery.lib.scripts.js.analysis.model import FUNCTION_NODES
from refinery.lib.scripts.js.model import (
    JsCallExpression,
    JsIdentifier,
    JsMemberExpression,
    JsMethodDefinition,
    JsProperty,
    JsStringLiteral,
    JsVariableDeclarator,
)
from refinery.lib.scripts.js.parser import JsParser

NL = chr(10)


def _lines(*parts: str) -> str:
    return NL.join(parts) + NL


def a_call_wrapper(kw: str = '', read: str = 'w(7)') -> str:
    """
    A wrapper that forwards its argument to another function, which is what a call wrapper is for.
    """
    return _lines(
        'function t(a) { return a; }',
        F'{kw}function w(a) {{ return t(a); }}',
        F'console.log({read});',
    )


def an_iife(read: str = '(function (a) { return a; })(7)') -> str:
    """
    A function expression called where it stands.
    """
    return _lines(F'console.log({read});')


def a_global_finder(kw: str = '', read: str = 'g() === globalThis') -> str:
    """
    A function every return of which names the global object, reached through a call it makes.
    """
    return _lines(
        F'{kw}function g() {{',
        '  var a = [function () { return globalThis; }];',
        '  var r = a[0]();',
        '  return r;',
        '}',
        F'console.log({read});',
    )


def a_callback(read: str = '[1, 2].filter(function (x) { return x > 1; }).length') -> str:
    """
    A predicate handed to an array method, which the interpreter runs rather than reads. Every
    promise and every generator object is truthy, so a wrapping predicate keeps both elements.
    """
    return _lines(F'console.log({read});')


def a_callback_inside_a_body(kw: str = '') -> str:
    return _lines(
        'function f() {',
        F'  return [1, 2].filter({kw}function (x) {{ return x > 1; }}).length;',
        '}',
        'console.log(f());',
    )


def a_rotation(kw: str = '', read: str = 'x.join("")') -> str:
    """
    A rotation over a literal array of ten elements, which is the floor the pass recognizes one at.
    """
    return _lines(
        F'{kw}function rot(arr, n) {{',
        '  for (var i = 0; i < n; i++) arr.push(arr.shift());',
        '  return arr;',
        '}',
        'var x = rot(["b", "c", "d", "e", "f", "g", "h", "i", "j", "a"], 9);',
        F'console.log({read});',
    )


def a_string_array(accessor_kw: str = '', holder_kw: str = '', read: str = 'ACC(0)') -> str:
    """
    A holder that replaces itself on its first call, an accessor that reads it by index, and a
    rotation loop whose sum matches its target at once, so the strings stand as written.
    """
    return _lines(
        F"{holder_kw}function A() {{ var s = ['hello', 'world'];"
        ' A = function () { return s; }; return A(); }',
        F'{accessor_kw}function ACC(i, k) {{ i = i - 0; var a = A();'
        ' var r = a[i]; return r; }',
        '(function (getArray, target) {',
        '  var arr = getArray();',
        '  while (true) {',
        '    var sum = 1;',
        "    if (sum === target) break; else arr['push'](arr['shift']());",
        '  }',
        '})(A, 1);',
        F'console.log({read});',
    )


AN_ENCODED_STRING = 'hJQxp9Pvj3X2QId3C4RuMOe1C4EpuSg2b/8JyqzSWjrQm+VgNNg='


def a_scramble(kw: str = '', read: str = F"typeof decode('{AN_ENCODED_STRING}').then") -> str:
    """
    A Scramble cipher instance behind a decode wrapper. `pb` and `decrypt` are stubs so that the
    program runs at all; the read is of the value the pass substitutes and not of what they compute.
    """
    return _lines(
        "function pb() { return 'K'; }",
        "function decrypt() { return 'X'; }",
        'class Scramble {',
        '  constructor(pw, salt) {',
        "    this.masterKey = pb(pw, salt, 200000, 32, 'sha256');",
        '    this.rounds = 3;',
        '  }',
        '  decode(input) { return decrypt(input, this.masterKey, this.rounds); }',
        '}',
        "var key = '2aaa9053353088d4d49b5bf32f403f2d85b3df97c9a9beedfcdbb1ecc27ba9c6';",
        "var salt = 'fec5863b88643968ecff0c2c8afecbaf';",
        'var instance = new Scramble(key, salt);',
        F'{kw}function decode(x) {{ return instance.decode(x); }}',
        F'console.log({read});',
    )


A_BASE91_ALPHABET = (
    '0,Fz)`Q(lH=j5gK[i8~mJt_b&qr/fW^Y2]?#|@.!$cLZ9BN>A1o7ye+D%IM}O6;pV:P*E3CRnxXSh{wvaTUuk4G'
    + chr(92) + '"s<d'
)


def a_base91_table(
    decoder_kw: str = '',
    accessor_kw: str = '',
    read: str = "typeof accessor(12).then",
) -> str:
    """
    The base91 table, decoder and caching accessor of
    `test.lib.scripts.js.deobfuscation.test_b91strings`, with the `bufferToString` that fixture
    leaves to its caller written in so that the program runs. Index 12 decodes to `log`.
    """
    return _lines(
        'function bufferToString(a) { return String.fromCharCode.apply(null, a); }',
        'var cache = {};',
        'var table = ["aa","bb","cc","dd","ee","ff","gg","hh","ii","jj",'
        '"fOg=r","lrCD^","#ZlH"];',
        F'{decoder_kw}function decode(str) {{',
        F'    var alpha = "{A_BASE91_ALPHABET}";',
        '    var raw = "" + (str || "");',
        '    var len = raw.length;',
        '    var ret = [];',
        '    var b = 0, n = 0, v = -1;',
        '    for (var i = 0; i < len; i++) {',
        '        var p = alpha.indexOf(raw[i]);',
        '        if (p === -1) continue;',
        '        if (v < 0) { v = p; }',
        '        else {',
        '            v += p * 91;',
        '            b |= v << n;',
        '            n += (v & 8191) > 88 ? 13 : 14;',
        '            do { ret.push(b & 0xff); b >>= 8; n -= 8; } while (n > 7);',
        '            v = -1;',
        '        }',
        '    }',
        '    if (v > -1) { ret.push((b | v << n) & 0xff); }',
        '    return bufferToString(ret);',
        '}',
        F'{accessor_kw}function accessor(index) {{',
        "    if (typeof cache[index] === 'undefined') {",
        '        return cache[index] = decode(table[index]);',
        '    }',
        '    return cache[index];',
        '}',
        F'console.log({read});',
    )


def a_self_disabling_wrapper(kw: str = '', read: str = 'W(a())') -> str:
    """
    A wrapper that overwrites itself with an empty function, called where its value is read.
    """
    return _lines(
        F'{kw}function W() {{ W = function () {{}}; }}',
        'var log = [];',
        "function a() { log.push('a'); return 1; }",
        F'console.log({read}, log.join(","));',
    )


def a_self_disabling_wrapper_as_a_statement(kw: str = '') -> str:
    return _lines(
        F'{kw}function W() {{ W = function () {{}}; }}',
        'var log = [];',
        "function a() { log.push('a'); return 1; }",
        'W(a());',
        'console.log(log.join(","));',
    )


A_DISPATCH = '(params = ["a", "b"], d("f1"))'


def a_dispatcher(outer: str = '', entry: str = '', read: str = A_DISPATCH) -> str:
    """
    An entry table selected by name behind a dispatcher taking a name, a flag and a return kind.

    The call site reads `(params = [...], d("f1")).then` and never `d("f1").then`: this pass does
    not unwrap the second shape at all, so a row written that way reports agreement for every
    keyword and for the control alike.
    """
    return _lines(
        'var cache = Object["create"](null);',
        'var params;',
        F'{outer}function d(name, flag, rtype, lengths) {{',
        '  var output;',
        F'  var fns = {{ "f1": {entry}function () {{',
        '    var [a, b] = params;',
        '    return a + b;',
        '  } };',
        '  if (flag === "initF") {',
        '    params = [];',
        '  }',
        '  if (flag === "createF") {',
        '    output = cache[name] || (cache[name] = fns[name]);',
        '  } else {',
        '    output = fns[name]();',
        '  }',
        '  if (rtype === "wrapF") {',
        '    return { "wk": output };',
        '  } else {',
        '    return output;',
        '  }',
        '}',
        F'console.log({read});',
    )


#: For each site, the shape that pass exists for and the text it folds to. A guard that refuses more
#: than the keyword is caught here rather than by a suite that stayed green because nothing witnessed
#: the site at all.
THE_SHAPE_EACH_PASS_IS_FOR = {
    'wrappers': (a_call_wrapper(), 'console.log(7);'),
    'simplify': (an_iife(), 'console.log(7);'),
    'globalfinder': (a_global_finder(), 'console.log(globalThis === globalThis);'),
    'interpreter': (a_callback(), 'console.log(1);'),
    'interpreter inside a body': (a_callback_inside_a_body(), 'console.log(1);'),
    'unshuffle': (
        a_rotation(),
        "var x = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'];"
        + NL + 'console.log(x.join(""));',
    ),
    'stringarray': (a_string_array(), "console.log('hello');"),
    'scramble': (a_scramble(), "console.log(typeof 'https://api.github.com'.then);"),
    'b91strings': (a_base91_table(), "console.log(typeof 'log'.then);"),
    'argwrap in statement position': (
        a_self_disabling_wrapper_as_a_statement(),
        _lines(
            'var log = [];',
            'function a() {',
            "  log.push('a');",
            '  return 1;',
            '}',
            'a();',
            'console.log(log.join(","));',
        ).rstrip(NL),
    ),
    'argwrap in expression position': (
        a_self_disabling_wrapper(),
        _lines(
            'var log = [];',
            'function a() {',
            "  log.push('a');",
            '  return 1;',
            '}',
            'console.log((a(), void 0), log.join(","));',
        ).rstrip(NL),
    ),
    'dispatcher': (a_dispatcher(), "console.log('ab');"),
}


#: Per site, the same shape with a keyword on the function whose call the pass answers, mapped to
#: what Node prints for it. Every value here was read off Node and none of it off this project.
A_CALL_ANSWERING_A_WRAPPER = {
    'wrappers': {
        a_call_wrapper('async ', 'typeof w(7).then'): 'function\n',
        a_call_wrapper('function* ', 'typeof w(7).next').replace(
            'function* function w', 'function* w'): 'function\n',
        a_call_wrapper('async function* ', 'typeof w(7).next').replace(
            'async function* function w', 'async function* w'): 'function\n',
    },
    'simplify': {
        an_iife('typeof (async function (a) { return a; })(7).then'): 'function\n',
        an_iife('typeof (function* (a) { return a; })(7).next'): 'function\n',
    },
    'globalfinder': {
        a_global_finder('async ', 'typeof g().then'): 'function\n',
        a_global_finder('function* ', 'typeof g().next').replace(
            'function* function g', 'function* g'): 'function\n',
    },
    'interpreter': {
        a_callback('[1, 2].filter(async function (x) { return x > 1; }).length'): '2\n',
        a_callback('[1, 2].filter(async (x) => x > 1).length'): '2\n',
        a_callback('[1, 2].filter(function* (x) { return x > 1; }).length'): '2\n',
    },
    'interpreter inside a body': {
        a_callback_inside_a_body('async '): '2\n',
    },
    'unshuffle': {
        a_rotation('async ', 'typeof x.then'): 'function\n',
        a_rotation('function* ', 'typeof x.next').replace(
            'function* function rot', 'function* rot'): 'function\n',
    },
    'stringarray': {
        a_string_array('async ', read='typeof ACC(0).then'): 'function\n',
        a_string_array('function* ', read='typeof ACC(0).next').replace(
            'function* function ACC', 'function* ACC'): 'function\n',
        a_string_array(holder_kw='function* ').replace(
            'function* function A', 'function* A'): 'undefined\n',
    },
    'scramble': {
        a_scramble('async '): 'function\n',
        a_scramble(
            'function* ',
            F"typeof decode('{AN_ENCODED_STRING}').next",
        ).replace('function* function decode', 'function* decode'): 'function\n',
    },
    'b91strings': {
        a_base91_table(accessor_kw='async '): 'function\n',
        a_base91_table(
            accessor_kw='function* ',
            read='typeof accessor(12).next',
        ).replace('function* function accessor', 'function* accessor'): 'function\n',
        a_base91_table(decoder_kw='async '): 'function\n',
    },
    'argwrap in expression position': {
        a_self_disabling_wrapper('async ', 'typeof W(a()).then'): 'function a\n',
        a_self_disabling_wrapper('function* ', 'typeof W(a()).next').replace(
            'function* function W', 'function* W'): 'function a\n',
    },
    'dispatcher': {
        a_dispatcher(outer='async ', read=F'typeof {A_DISPATCH}.then'): 'function\n',
        a_dispatcher(outer='*', read=F'typeof {A_DISPATCH}.next').replace(
            '*function d(', 'function* d(', 1): 'function\n',
    },
}


#: Shapes in which the keyword sits on a function whose call the pass is right to answer, so that a
#: guard written per pass rather than per function is caught. The string array's `async` holder
#: replaces itself on its first call and the rotation calls it before anything else does, so every
#: later call reads the plain replacement; the dispatcher's entries are built with the keyword
#: carried, which `dispatcher._build_extracted_function` already does.
A_WRAPPER_THE_PASS_IS_RIGHT_TO_ANSWER = {
    a_string_array(holder_kw='async '): (
        'hello' + NL,
        "console.log('hello');",
    ),
    an_iife('typeof (async (a) => a)(7).then'): (
        'function' + NL,
        'console.log(typeof (async a => a)(7).then);',
    ),
    a_dispatcher(read=F'typeof {A_DISPATCH}.then'): (
        'undefined' + NL,
        "console.log(typeof 'ab'.then);",
    ),
    a_dispatcher(entry='async ', read=F'typeof {A_DISPATCH}.then'): (
        'function' + NL,
        _lines(
            'async function f1(a, b) {',
            '  return a + b;',
            '}',
            'console.log(typeof f1("a", "b").then);',
        ).rstrip(NL),
    ),
    a_dispatcher(entry='*', read=F'typeof {A_DISPATCH}.next').replace(
        '*function ()', 'function* ()', 1): (
        'function' + NL,
        _lines(
            'function* f1(a, b) {',
            '  return a + b;',
            '}',
            'console.log(typeof f1("a", "b").next);',
        ).rstrip(NL),
    ),
    a_self_disabling_wrapper_as_a_statement('async '): (
        'a' + NL,
        _lines(
            'var log = [];',
            'function a() {',
            "  log.push('a');",
            '  return 1;',
            '}',
            'a();',
            'console.log(log.join(","));',
        ).rstrip(NL),
    ),
}


#: Programs whose fold lifts an operator out of the body that gives it meaning. Neither output is a
#: program at all: `await` outside an `async` body and `yield` outside a generator are syntax errors,
#: so `refinery.lib.scripts.is_well_formed` answers `False` for what comes back.
AN_OPERATOR_THE_BODY_GAVE_MEANING_TO = (
    'console.log(typeof (async function () { return await 7; })().then);' + NL,
    'console.log(typeof (function* () { return yield 7; })().next);' + NL,
)


def _wraps_what_it_returns(node: Node | None) -> bool:
    return isinstance(node, FUNCTION_NODES) and bool(
        getattr(node, 'is_async', False) or getattr(node, 'generator', False)
    )


def _the_names_a_wrapping_function_answers_to(root: Node) -> set[str]:
    names: set[str] = set()
    for node in root.walk():
        if not _wraps_what_it_returns(node):
            continue
        own = getattr(node, 'id', None)
        if isinstance(own, JsIdentifier):
            names.add(own.name)
        owner = node.parent
        if isinstance(owner, JsVariableDeclarator) and isinstance(owner.id, JsIdentifier):
            names.add(owner.id.name)
        if isinstance(owner, (JsProperty, JsMethodDefinition)):
            key = getattr(owner, 'key', None)
            if isinstance(key, JsIdentifier):
                names.add(key.name)
            elif isinstance(key, JsStringLiteral):
                names.add(key.value)
    return names


def _names_a_wrapping_function(callee: Node | None, names: set[str]) -> bool:
    if _wraps_what_it_returns(callee):
        return True
    if isinstance(callee, JsIdentifier):
        return callee.name in names
    if isinstance(callee, JsMemberExpression) and not callee.computed:
        key = callee.property
        return isinstance(key, JsIdentifier) and key.name in names
    return False


def _what_wraps_and_what_calls_it(source: str) -> tuple[int, int]:
    """
    How many functions in *source* wrap what their body returns, and how many calls name one.

    A pass that answers such a call takes the call away, and usually the function with it, so this
    pair moving is what a wrong answer looks like from outside the engine. A pass that reshapes
    either without answering the call — promoting an accessor, flattening a namespace, extracting an
    entry — carries the keyword along and leaves the pair where it was.
    """
    root = JsParser(source).parse()
    names = _the_names_a_wrapping_function_answers_to(root)
    wrapping = 0
    calls = 0
    for node in root.walk():
        if _wraps_what_it_returns(node):
            wrapping += 1
        if isinstance(node, JsCallExpression) and _names_a_wrapping_function(node.callee, names):
            calls += 1
    return wrapping, calls


def _each_row_keeps_what_wraps_and_what_calls_it(rows: dict[str, str]) -> dict[str, tuple[int, int]]:
    return {source: _what_wraps_and_what_calls_it(source) for source in rows}


def _after_the_fold(rows: dict[str, str]) -> dict[str, tuple[int, int]]:
    return {source: _what_wraps_and_what_calls_it(folded(source)) for source in rows}


class TestEachPassStillAnswersTheShapeItIsFor(TestBase):
    """
    The admission control, one row per site, with no keyword anywhere. Each states the text that pass
    folds its own shape to, so that a guard which refuses the shape rather than the keyword fails
    here. Without it a guard could empty a site and leave the suite green, which is what
    `stringarray._match_wrapper` is witnessed by nothing enough to allow.
    """

    def test_every_site_folds_the_shape_it_exists_for(self):
        self.assertEqual(
            {site: folded(source) for site, (source, _) in THE_SHAPE_EACH_PASS_IS_FOR.items()},
            {site: text for site, (_, text) in THE_SHAPE_EACH_PASS_IS_FOR.items()},
        )


class TestAWrapperThePassIsRightToAnswerIsStillAnswered(TestBase):
    """
    The other half of the control: a keyword on a function whose call the pass may answer. A guard
    written per pass rather than per function destroys these.
    """

    def test_each_of_them_folds_to_the_text_it_folded_to(self):
        rows = A_WRAPPER_THE_PASS_IS_RIGHT_TO_ANSWER
        self.assertEqual(
            {source: folded(source) for source in rows},
            {source: text for source, (_, text) in rows.items()},
        )

    @unittest.skipIf(node_executable() is None, 'node.js is not available')
    def test_each_of_them_prints_what_it_printed(self):
        rows = {
            source: prints
            for source, (prints, _) in A_WRAPPER_THE_PASS_IS_RIGHT_TO_ANSWER.items()
        }
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )


class TestTheDeobfuscationStillCallsAWrappingFunction(TestBase):
    """
    What each site leaves standing, read from the output text and from nothing else, so that the
    regression is reported on a machine with no Node.js as well.
    """

    @unittest.expectedFailure
    def test_no_site_answers_a_call_to_one(self):
        rows = {
            source: prints
            for group in A_CALL_ANSWERING_A_WRAPPER.values()
            for source, prints in group.items()
        }
        self.assertEqual(_after_the_fold(rows), _each_row_keeps_what_wraps_and_what_calls_it(rows))


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestTheProgramPrintsWhatItPrinted(TestBase):
    """
    Node is the oracle for every value here. Each test names what Node prints and asserts that the
    deobfuscation prints it too, with neither of the two throwing.
    """

    @unittest.expectedFailure
    def test_a_call_wrapper(self):
        """
        Node prints `function` for each: `w(7)` answers a promise for `async`, a generator object for
        a generator, and an async generator object for both, and each has the read the row asks for.
        """
        rows = A_CALL_ANSWERING_A_WRAPPER['wrappers']
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    @unittest.expectedFailure
    def test_an_iife(self):
        rows = A_CALL_ANSWERING_A_WRAPPER['simplify']
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    @unittest.expectedFailure
    def test_a_global_finder(self):
        rows = A_CALL_ANSWERING_A_WRAPPER['globalfinder']
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    @unittest.expectedFailure
    def test_a_callback_an_array_method_runs(self):
        """
        Node prints `2` for each: every promise and every generator object is truthy, so `filter`
        keeps both elements where the value each body returned would have kept one.
        """
        rows = A_CALL_ANSWERING_A_WRAPPER['interpreter']
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    @unittest.expectedFailure
    def test_a_callback_inside_a_function_body(self):
        rows = A_CALL_ANSWERING_A_WRAPPER['interpreter inside a body']
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    @unittest.expectedFailure
    def test_a_rotation(self):
        rows = A_CALL_ANSWERING_A_WRAPPER['unshuffle']
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    @unittest.expectedFailure
    def test_a_string_array(self):
        """
        Node prints `function` for the two accessor rows. The generator *holder* prints `undefined`:
        a generator's body does not run when it is called, so the self-reassignment never happens and
        the accessor reads an index of a generator object.
        """
        rows = A_CALL_ANSWERING_A_WRAPPER['stringarray']
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    @unittest.expectedFailure
    def test_a_scramble_decoder(self):
        rows = A_CALL_ANSWERING_A_WRAPPER['scramble']
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    @unittest.expectedFailure
    def test_a_base91_table(self):
        """
        The keyword is asked for on the accessor and on the decoder, because the accessor answers
        with what the decoder gave it and either one wrapping is enough to make the call answer a
        wrapper.
        """
        rows = A_CALL_ANSWERING_A_WRAPPER['b91strings']
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    @unittest.expectedFailure
    def test_a_self_disabling_wrapper_in_expression_position(self):
        """
        Node prints `function a`: the call answers a promise, and the argument ran on the way. The
        expansion writes `(a(), void 0)`, which has the argument's effect and no `.then`.
        """
        rows = A_CALL_ANSWERING_A_WRAPPER['argwrap in expression position']
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    @unittest.expectedFailure
    def test_a_dispatcher(self):
        rows = A_CALL_ANSWERING_A_WRAPPER['dispatcher']
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )


class TestNeitherEscapeLeavesAProgramNothingCanSpell(TestBase):
    """
    `await` and `yield` mean what they mean because of the body they stand in. Lifting the body's
    return expression to the call site takes that body away, and what comes back is not a program:
    `refinery.lib.scripts.is_well_formed`, reached through `test.lib.scripts.js.ledger.well_formed`,
    answers `False` for both.
    """

    @unittest.expectedFailure
    def test_the_fold_leaves_both_of_them_spellable(self):
        self.assertEqual(
            [well_formed(folded(source)) for source in AN_OPERATOR_THE_BODY_GAVE_MEANING_TO],
            [True, True],
        )


class TestARefusalStillLetsTheRunEnd(TestBase):
    """
    A pass that declines has to decline without reporting a change, or the pipeline is handed the
    same tree for as long as it is willing to look at it. Every shape the guards refuse is run under
    a bound here, so a refusal that spins is a failure and not a hang.
    """

    def test_every_refused_shape_is_deobfuscated_within_the_bound(self):
        rows = [
            source
            for group in A_CALL_ANSWERING_A_WRAPPER.values()
            for source in group
        ]
        self.assertEqual(
            [deobfuscate_within(source, 30.0) is not None for source in rows],
            [True] * len(rows),
        )
