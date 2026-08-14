"""
Reading a property off a string whose value the tool already knows.

Exactly two keys are decided by the string and by nothing else: `length`, and a canonical index
within range. Both are own data properties of the String exotic object the access goes through, so
no prototype can shadow them and no program can change what they answer — which is what licenses
replacing the access with the value it reads.

Every other key is the prototype chain's to answer, and a file is free to install one before the
read runs. An index past the end, an index below zero, a key that merely looks like an index
(`'01'`, `'-0'`, `'1.0'`, `' 1'`), a method name such as `charAt`, and a name nothing defines are
therefore left standing; the cases below run a file that installs each of them through Node to show
the read still finds it.

The string need not be spelled in the file. `atob`, `unescape`, `decodeURIComponent`, `JSON.parse`,
`repeat`, and `toLowerCase` all hand one back, and a read off the result is the same read off the
same value — including for a character above the basic multilingual plane, which is two code units
however the string holding it was built.

A property access is finally not always a read. Assigned, updated, deleted, or destructured, it is a
target that must survive as one. In callee position it may be replaced, because what these two keys
answer is a string or a number and never a function, so the call throws a `TypeError` before and
after alike.

Node is the authority for every absolute value here. A string is pinned as its UTF-16 code units so
that what is fixed is the value and not the tool's choice of escapes.
"""
from __future__ import annotations

import unittest

from test import TestBase
from test.lib.scripts.js.analysis.differential import behavior, code_units, node_executable

from refinery.units.scripting.js import js

_ASTRAL = chr(0x1F600)

_ASTRAL_LITERAL = F"'a{_ASTRAL}b'"

_ASTRAL_PRODUCED = "decodeURIComponent('a%F0%9F%98%80b')"

#: Each read that the string itself decides, mapped to the text the tool folds it to and to the
#: code-unit structure Node answers it with.
_DECIDED: dict[str, tuple[str, str]] = {
    "'abc'.length"                     : ('3', '3'),
    "'abc'['length']"                  : ('3', '3'),
    "'abc'['leng' + 'th']"             : ('3', '3'),
    "''.length"                        : ('0', '0'),
    "'abc'[0]"                         : ("'a'", 'S[97]'),
    "'abc'[2]"                         : ("'c'", 'S[99]'),
    "'abc'['1']"                       : ("'b'", 'S[98]'),
    "'abc'[1.0]"                       : ("'b'", 'S[98]'),
    "'abc'[0x1]"                       : ("'b'", 'S[98]'),
    "'abc'[-0]"                        : ("'a'", 'S[97]'),
    "'abc'[2 - 1]"                     : ("'b'", 'S[98]'),
    "'abc'[true ? 1 : 2]"              : ("'b'", 'S[98]'),
    "atob('SGVsbG8=').length"          : ('5', '5'),
    "atob('SGVsbG8=')[1]"              : ("'e'", 'S[101]'),
    "unescape('%48%65')[1]"            : ("'e'", 'S[101]'),
    """JSON.parse('"hi"').length"""    : ('2', '2'),
    "'ABC'.toLowerCase()[1]"           : ("'b'", 'S[98]'),
    "'ab'.repeat(5000).length"         : ('10000', '10000'),
    F'{_ASTRAL_LITERAL}.length'        : ('4', '4'),
    F'{_ASTRAL_LITERAL}[1]'            : (R"'\uD83D'", 'S[55357]'),
    F'{_ASTRAL_LITERAL}[2]'            : (R"'\uDE00'", 'S[56832]'),
    F'{_ASTRAL_LITERAL}[3]'            : ("'b'", 'S[98]'),
    F'{_ASTRAL_PRODUCED}.length'       : ('4', '4'),
    F'{_ASTRAL_PRODUCED}[1]'           : (R"'\uD83D'", 'S[55357]'),
    F'{_ASTRAL_PRODUCED}[2]'           : (R"'\uDE00'", 'S[56832]'),
    F'{_ASTRAL_PRODUCED}[3]'           : ("'b'", 'S[98]'),
}

#: Each read no prototype-free string decides, mapped to the text it is left standing as. Where the
#: two differ, the difference is the synthesizer spelling a computed key as a dot access; the access
#: is still there and still asks the engine.
_LEFT_STANDING: dict[str, str] = {
    "'abc'[3]"                : "'abc'[3]",
    "'abc'[9]"                : "'abc'[9]",
    "'abc'[4294967295]"       : "'abc'[4294967295]",
    "'abc'[-1]"               : "'abc'[-1]",
    "''[0]"                   : "''[0]",
    "'abc'['-0']"             : "'abc'['-0']",
    "'abc'['01']"             : "'abc'['01']",
    "'abc'['00']"             : "'abc'['00']",
    "'abc'['1.0']"            : "'abc'['1.0']",
    "'abc'[0.5]"              : "'abc'[0.5]",
    "'abc'[' 1']"             : "'abc'[' 1']",
    "'abc'[NaN]"              : "'abc'[NaN]",
    "'abc'[void 0]"           : "'abc'[void 0]",
    "'abc'.charAt"            : "'abc'.charAt",
    "'abc'['charAt']"         : "'abc'.charAt",
    "'abc'.toUpperCase"       : "'abc'.toUpperCase",
    "'abc'.constructor"       : "'abc'.constructor",
    "'abc'.__proto__"         : "'abc'.__proto__",
    "'abc'.nope"              : "'abc'.nope",
    "'abc'['nope']"           : "'abc'.nope",
    '(1).length'              : '(1).length',
    'true.length'             : 'true.length',
    F'{_ASTRAL_LITERAL}[4]'   : F'{_ASTRAL_LITERAL}[4]',
}


def _fold(expression: str) -> str:
    """
    The expression `refinery.js` folds *expression* to. It is placed in a `console.log` argument,
    which survives as a side effect, so nothing but the fold decides what comes back.
    """
    printed = F'console.log({expression});'.encode('utf8') | js() | str
    return printed.removeprefix('console.log(').removesuffix(');')


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class _NodeDecides(TestBase):
    """
    A base for the cases whose answer only a real engine has: it runs a snippet and its
    deobfuscation in Node and asserts the two behave identically.
    """

    def _preserves(self, source: str):
        deobfuscated = source.encode('utf8') | js() | str
        self.assertEqual(
            behavior(source),
            behavior(deobfuscated),
            F'deobfuscation changed observable behavior; result was:\n{deobfuscated}',
        )


class TestAReadTheStringDecides(TestBase):

    def test_length_and_an_index_in_range_fold_to_a_constant(self):
        for expression, (folded, _) in _DECIDED.items():
            with self.subTest(expression=expression):
                self.assertEqual(_fold(expression), folded)

    @unittest.skipIf(node_executable() is None, 'node.js is not available')
    def test_node_reads_the_folded_constant_from_the_access_it_replaced(self):
        expressions = list(_DECIDED)
        answers = code_units(expressions)
        replacements = code_units([_DECIDED[e][0] for e in expressions])
        for expression, answer, replacement in zip(expressions, answers, replacements):
            with self.subTest(expression=expression):
                expected = _DECIDED[expression][1]
                self.assertEqual((answer, replacement), (expected, expected))


class TestAReadTheStringDoesNotDecide(TestBase):

    def test_a_key_no_string_owns_is_left_standing(self):
        for expression, standing in _LEFT_STANDING.items():
            with self.subTest(expression=expression):
                self.assertEqual(_fold(expression), standing)

    def test_a_method_that_is_only_a_name_when_read_still_folds_when_called(self):
        self.assertEqual(_fold("'abc'['charAt'](1)"), "'b'")
        self.assertEqual(_fold("'abc'.toUpperCase()"), "'ABC'")


class TestNoPrototypeCanShadowAnOwnProperty(_NodeDecides):

    def test_a_prototype_length_does_not_reach_a_string_that_has_its_own(self):
        self._preserves("String.prototype.length = 99; console.log('abc'.length);")

    def test_a_prototype_index_does_not_reach_an_index_the_string_holds(self):
        self._preserves("String.prototype[0] = 'X'; console.log('abc'[0]);")

    def test_length_cannot_even_be_redefined_on_the_prototype(self):
        self._preserves(
            "Object.defineProperty(String.prototype, 'length', "
            '{get: function () { return 99; }});'
            " console.log('abc'.length);"
        )


class TestTheReadsThatAreLeftToTheEngine(_NodeDecides):

    def test_a_prototype_index_past_the_end_is_still_found(self):
        for source in [
            "String.prototype[3] = 'X'; console.log('abc'[3]);",
            "String.prototype[9] = 'X'; console.log('abc'[9]);",
            "Object.defineProperty(String.prototype, '5', {value: 'Q'}); console.log('abc'[5]);",
            F"String.prototype[4] = 'X'; console.log({_ASTRAL_PRODUCED}[4]);",
            F"String.prototype[4] = 'X'; console.log({_ASTRAL_LITERAL}[4]);",
        ]:
            with self.subTest(source=source):
                self._preserves(source)

    def test_a_prototype_name_that_is_not_an_index_is_still_found(self):
        for source in [
            "String.prototype.nope = 'Y'; console.log('abc'.nope);",
            "String.prototype['01'] = 'Z'; console.log('abc'['01']);",
            "String.prototype['-0'] = 'Z'; console.log('abc'['-0']);",
            "String.prototype[' 1'] = 'Z'; console.log('abc'[' 1']);",
        ]:
            with self.subTest(source=source):
                self._preserves(source)

    def test_a_method_name_is_a_function_and_not_what_calling_it_would_answer(self):
        for source in [
            "console.log(typeof 'abc'.charAt, typeof 'abc'['charAt']);",
            "console.log(typeof 'abc'.constructor, typeof 'abc'.toUpperCase);",
            "console.log('abc'.charAt.length, 'abc'.charAt.name);",
        ]:
            with self.subTest(source=source):
                self._preserves(source)

    def test_a_receiver_that_is_no_live_string_keeps_what_it_did(self):
        for source in [
            'console.log(undefined.length);',
            'console.log(null[0]);',
            'console.log((void 0).length);',
            'console.log((1).length, true.length, (0 / 0).length);',
            "console.log(new String('abc').length);",
        ]:
            with self.subTest(source=source):
                self._preserves(source)


class TestAnAccessThatIsNotARead(_NodeDecides):

    def test_a_destructuring_target_survives_as_a_target(self):
        for source in [
            "['abc'[0]] = ['z']; console.log('abc'[0]);",
            "({k: 'abc'[0]} = {k: 'z'}); console.log('abc'[0]);",
            "[decodeURIComponent('abc')[0]] = ['z']; console.log('abc'[0]);",
            "[{}.x, 'abc'.length] = [1, 2]; console.log('abc'.length);",
        ]:
            with self.subTest(source=source):
                self._preserves(source)

    def test_a_for_of_target_survives_as_a_target(self):
        for source in [
            "for ('abc'[0] of ['z']) ; console.log('abc'[0]);",
            "for ('abc'.length of [9]) ; console.log('abc'.length);",
        ]:
            with self.subTest(source=source):
                self._preserves(source)

    def test_an_assignment_that_a_strict_file_refuses_still_throws(self):
        for source in [
            "'use strict'; 'abc'.length = 9; console.log('not reached');",
            "'use strict'; 'abc'[0] = 'z'; console.log('not reached');",
            "'use strict'; console.log(delete 'abc'[0]);",
            "'use strict'; console.log(delete 'abc'.length);",
            "'use strict'; console.log('abc'.length++);",
        ]:
            with self.subTest(source=source):
                self._preserves(source)

    def test_a_produced_string_is_a_target_the_same_way_a_literal_is(self):
        for source in [
            "console.log(decodeURIComponent('abc').length = 9);",
            """console.log(JSON.parse('"abc"')[0]++);""",
            "console.log(delete 'ABC'.toLowerCase()[0]);",
            "console.log(decodeURIComponent('abc')[0] += 'q');",
        ]:
            with self.subTest(source=source):
                self._preserves(source)

    def test_calling_what_an_own_property_answers_throws_either_way(self):
        for source in [
            "console.log('abc'.length());",
            "console.log('abc'[0]());",
            "console.log('abc'.length?.());",
            "console.log(new ('abc'.length)());",
            "console.log('abc'.length`x`);",
            "'use strict'; console.log('abc'[1]());",
        ]:
            with self.subTest(source=source):
                self._preserves(source)
