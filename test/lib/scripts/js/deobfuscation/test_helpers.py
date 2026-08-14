from __future__ import annotations

import inspect
import struct
import unittest

from test.lib.scripts.js.analysis.differential import behavior, node_executable
from test.lib.scripts.js.deobfuscation import TestJsDeobfuscator

from refinery.lib.scripts.js.analysis.model import build_semantic_model
from refinery.lib.scripts.js.deobfuscation.helpers import (
    JS_NULL,
    JsBuffer,
    binding_has_references,
    escape_js_string,
    extract_literal_value,
    make_numeric_literal,
    make_string_literal,
    value_to_node,
)
from refinery.lib.scripts.js.lexer import decode_js_string_body
from refinery.lib.scripts.js.model import JsExpressionStatement, JsStringLiteral
from refinery.lib.scripts.js.parser import JsParser
from refinery.lib.scripts.js.synth import JsSynthesizer


class TestDeobfuscationHelpers(TestJsDeobfuscator):

    def test_make_string_literal_escapes_control_chars(self):
        self.assertEqual(make_string_literal('a\nb').raw, "'a\\nb'")
        self.assertEqual(make_string_literal('x\ry').raw, "'x\\ry'")
        self.assertEqual(make_string_literal('p\tq').raw, "'p\\tq'")
        self.assertEqual(make_string_literal('m\0n').raw, "'m\\0n'")

    def test_binding_has_references_ignores_shadowing_param(self):
        source = inspect.cleandoc(
            """
            var table = pool;
            function uses(table) { return table.length; }
            """
        )
        model = build_semantic_model(JsParser(source).parse())
        binding = model.lookup('table', model.root_scope)
        self.assertFalse(binding_has_references(model, binding))

    def test_binding_has_references_counts_genuine_use(self):
        source = inspect.cleandoc(
            """
            var table = pool;
            log(table.length);
            """
        )
        model = build_semantic_model(JsParser(source).parse())
        binding = model.lookup('table', model.root_scope)
        self.assertTrue(binding_has_references(model, binding))


class TestValueToNode(TestJsDeobfuscator):
    """
    `value_to_node` renders an interpreter value back into source. Every value it accepts must denote
    exactly the same thing once re-parsed, so these tests pin the rendered text for the cases where a
    plausible-looking rendering would mean something else, and pin refusal for the values that have no
    faithful literal form at all.
    """

    def _render(self, value: object) -> str | None:
        node = value_to_node(value)
        if node is None:
            return None
        return JsSynthesizer().convert(node)

    def test_renders_primitives(self):
        self.assertEqual("'abc'", self._render('abc'))
        self.assertEqual('42', self._render(42))
        self.assertEqual('-42', self._render(-42))
        self.assertEqual('1.5', self._render(1.5))
        self.assertEqual('true', self._render(True))
        self.assertEqual('false', self._render(False))
        self.assertEqual('null', self._render(JS_NULL))
        self.assertEqual('void 0', self._render(None))

    def test_renders_special_numbers(self):
        self.assertEqual('0 / 0', self._render(float('nan')))
        self.assertEqual('1e999', self._render(float('inf')))
        self.assertEqual('-1e999', self._render(float('-inf')))
        self.assertEqual('-0', self._render(-0.0))

    def test_renders_containers(self):
        self.assertEqual('[]', self._render([]))
        self.assertEqual('[1, 2]', self._render([1, 2]))
        self.assertEqual('[1, void 0]', self._render([1, None]))
        self.assertEqual('{}', self._render({}))
        self.assertEqual("{ 'a': 1 }", self._render({'a': 1}))

    def test_renders_proto_key_as_computed(self):
        """
        A `__proto__` entry in the value is an own data property. Only the computed key form creates
        one; the bare and quoted forms install a prototype and leave no own property behind.
        """
        self.assertEqual("{ ['__proto__']: 1 }", self._render({'__proto__': 1}))

    def test_renders_nested_proto_key_as_computed(self):
        self.assertEqual(
            "{ 'a': { ['__proto__']: 1 } }",
            self._render({'a': {'__proto__': 1}}),
        )

    def test_renders_ordinary_prototype_member_names_plainly(self):
        """
        Only `__proto__` is special. The other inherited member names are ordinary own properties when
        written as plain keys, so forcing them computed would be needless noise.
        """
        self.assertEqual("{ 'constructor': 1 }", self._render({'constructor': 1}))
        self.assertEqual("{ 'toString': 1 }", self._render({'toString': 1}))
        self.assertEqual("{ 'hasOwnProperty': 1 }", self._render({'hasOwnProperty': 1}))

    def test_refuses_buffer(self):
        """
        A Buffer has no literal form. Rendering its bytes as an array would produce a value of a
        different type, silently dropping `Buffer.isBuffer` and `.toString('hex')`.
        """
        self.assertIsNone(value_to_node(JsBuffer([65, 66])))
        self.assertIsNone(value_to_node(JsBuffer([])))

    def test_refuses_buffer_nested_in_container(self):
        self.assertIsNone(value_to_node([JsBuffer([65])]))
        self.assertIsNone(value_to_node({'b': JsBuffer([65])}))
        self.assertIsNone(value_to_node([[JsBuffer([65])]]))

    def test_refuses_non_string_key(self):
        self.assertIsNone(value_to_node({1: 'x'}))

    def test_refuses_unrepresentable_value(self):
        self.assertIsNone(value_to_node(object()))


JS_NUMBERS = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    42.0,
    -42.0,
    0.1,
    0.30000000000000004,
    -1.5,
    4.35,
    0.3333333333333333,
    9007199254740992.0,
    9007199254740994.0,
    1152921504606846976.0,
    1e16,
    1.2345678901234568e17,
    1e20,
    9.999999999999999e20,
    1e21,
    -1e21,
    1.2345678901234568e29,
    1e-6,
    1e-7,
    1.5e-7,
    2.220446049250313e-16,
    2.2250738585072014e-308,
    1e-323,
    5e-324,
    -5e-324,
    1.7976931348623157e308,
    float('inf'),
    float('-inf'),
]
"""
A sample of the JavaScript Number domain, weighted towards the members whose spelling is least
obvious: the two zeros, which nothing but their sign tells apart; the integers past 2^53, where a
double has fewer digits than its exact value; both boundaries where the language switches between
positional and exponential notation, and the one where Python switches at a different magnitude; the
smallest subnormals; and the infinities, which no arithmetic but a decimal literal reaches by
rounding.
"""

NUMBER_BITS_IN_NODE = inspect.cleandoc("""
    const view = new DataView(new ArrayBuffer(8));
    function bits(x) {
      view.setFloat64(0, x);
      return view.getBigUint64(0).toString(16).padStart(16, '0');
    }
""")


class TestNumericLiteralRoundTrip(TestJsDeobfuscator):
    """
    `make_numeric_literal` writes a Number as source text and `extract_literal_value` reads source
    text back into a value; the module declares the two inverses of each other. Whether they are is
    decided by what the text means, and neither of them may answer that: every spelling below goes
    to Node, which reports the 64 bits of the Number it read, and through `JsParser`, so that the
    reader is handed the tree the text really produces rather than the one the writer built.
    """

    def _bits(self, value: float) -> str:
        return struct.pack('>d', value).hex()

    def _spelling(self, value: int | float) -> str:
        node = make_numeric_literal(value)
        if node is None:
            self.fail(F'{value!r} was refused a spelling')
        return JsSynthesizer().convert(node)

    def _spellings(self) -> list[tuple[float, str]]:
        return [(value, self._spelling(value)) for value in JS_NUMBERS]

    def _node_reads(self, expressions: list[str]) -> list[str]:
        """
        The 64 bits of the Number Node reads from each of *expressions*, as hexadecimal. A bit
        pattern is the whole Number and nothing besides it, which is what makes it the thing to
        compare: it separates the two zeros, and it is not a printed form, so no spelling is ever
        measured against a reading of itself.
        """
        script = [NUMBER_BITS_IN_NODE]
        script.extend(F'console.log(bits({expression}));' for expression in expressions)
        output, error = behavior('\n'.join(script))
        self.assertIsNone(error, 'node.js refused the spellings')
        lines = output.split()
        self.assertEqual(len(expressions), len(lines))
        return lines

    def _reader_reads(self, text: str) -> str:
        script = JsParser(F'{text};').parse()
        statement = script.body[0]
        if not isinstance(statement, JsExpressionStatement) or statement.expression is None:
            self.fail(F'{text!r} did not parse as a single expression')
        recognized, value = extract_literal_value(statement.expression)
        if not recognized or not isinstance(value, float):
            return F'refused: {value!r}'
        return self._bits(value)

    @unittest.skipIf(node_executable() is None, 'node.js is not available')
    def test_every_spelling_denotes_the_number_it_was_asked_to_spell(self):
        spelled = self._spellings()
        reads = self._node_reads([F'({text})' for _, text in spelled])
        self.assertEqual(
            {text: self._bits(value) for value, text in spelled},
            {text: read for (_, text), read in zip(spelled, reads)},
        )

    def test_every_spelling_reads_back_as_the_number_it_was_asked_to_spell(self):
        spelled = self._spellings()
        self.assertEqual(
            {text: self._bits(value) for value, text in spelled},
            {text: self._reader_reads(text) for _, text in spelled},
        )

    @unittest.skipIf(node_executable() is None, 'node.js is not available')
    def test_every_spelling_denotes_the_same_number_where_the_value_names_are_rebound(self):
        """
        `NaN`, `Infinity` and `undefined` are ordinary identifiers that any scope may bind to
        something else, so a spelling that reached a Number through one of those names would mean
        whatever the scope supplies. A literal denotes its value in every scope, which is what this
        asks of each spelling by reading it inside a function that binds all three to other numbers.
        """
        spelled = self._spellings()
        reads = self._node_reads([
            F'(function (NaN, Infinity, undefined) {{ return ({text}); }})(1, 2, 3)'
            for _, text in spelled
        ])
        self.assertEqual(
            {text: self._bits(value) for value, text in spelled},
            {text: read for (_, text), read in zip(spelled, reads)},
        )

    @unittest.skipIf(node_executable() is None, 'node.js is not available')
    def test_an_integer_is_spelled_as_the_number_its_own_digits_denote(self):
        """
        A Python integer is not a Number, and what it names is what its digits round to: past 2^53 a
        different integer, and past the double range an infinity. Those digits are themselves a
        JavaScript numeric literal, so what Node reads from them is what the spelling has to agree
        with, without a Python-side Number entering the comparison at all.
        """
        integers = [
            2 ** 53 + 1,
            2 ** 60,
            -(2 ** 60),
            10 ** 400,
            -(10 ** 400),
        ]
        digits = self._node_reads([F'({integer})' for integer in integers])
        spellings = self._node_reads([F'({self._spelling(integer)})' for integer in integers])
        self.assertEqual(
            {str(integer): read for integer, read in zip(integers, digits)},
            {str(integer): read for integer, read in zip(integers, spellings)},
        )

    def test_the_one_number_with_no_literal_is_refused(self):
        """
        Every other Number has a literal that denotes it, the infinities included. NaN has none, so
        the writer answers nothing rather than a spelling that would mean something else.
        """
        self.assertIsNone(make_numeric_literal(float('nan')))


class TestNulFollowedByDigitSurvivesPrinting(TestJsDeobfuscator):
    """
    A NUL immediately before an ASCII digit is the one place the compact NUL escape cannot be used:
    before a 0 through 7 it joins the digit into a single legacy octal escape, a different character
    (Node reads a NUL written before a 7 that way as the one character U+0007), and before an 8 or 9
    it is an escape strict code refuses. Printing the value must keep the NUL and the digit apart,
    while a NUL that no digit follows keeps the compact escape it had.
    """

    def _printed_then_read(self, value: str) -> str:
        printed = JsSynthesizer().convert(make_string_literal(value))
        reparsed = JsParser(printed).parse()
        statement = reparsed.body[0]
        if not isinstance(statement, JsExpressionStatement):
            self.fail(F'{printed!r} did not parse as an expression statement')
        literal = statement.expression
        if not isinstance(literal, JsStringLiteral):
            self.fail(F'{printed!r} did not parse as a string literal')
        return literal.value

    def test_nul_before_seven_reads_back_as_a_nul_and_a_seven(self):
        value = '\x00' + '7'
        self.assertEqual(value, self._printed_then_read(value))

    def test_nul_before_zero_reads_back_as_a_nul_and_a_zero(self):
        value = '\x00' + '0'
        self.assertEqual(value, self._printed_then_read(value))

    def test_nul_before_a_letter_reads_back_as_a_nul_and_the_letter(self):
        value = '\x00' + 'a'
        self.assertEqual(value, self._printed_then_read(value))

    def test_nul_before_a_letter_keeps_the_compact_escape(self):
        self.assertEqual("'\\0a'", make_string_literal('\x00' + 'a').raw)


class TestEscapeAndDecodeInvertOnNulFollowedByDigit(TestJsDeobfuscator):
    """
    Escaping a value into a literal body and decoding a literal body are inverse operations, so a
    string put through both comes back unchanged. The case a naive escape would corrupt is a NUL
    immediately before a digit, where the compact NUL escape would merge with the digit into one
    character; every ASCII digit is checked here, the 8 and 9 that form no octal escape included.
    """

    def _round_trip(self, value: str) -> str:
        return decode_js_string_body(escape_js_string(value))

    def test_every_digit_after_a_nul_is_recovered(self):
        values = [F'\x00{digit}' for digit in '0123456789']
        self.assertEqual(
            {value: value for value in values},
            {value: self._round_trip(value) for value in values},
        )

    def test_a_letter_after_a_nul_is_recovered(self):
        value = '\x00' + 'a'
        self.assertEqual(value, self._round_trip(value))
