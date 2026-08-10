"""
What `refinery.lib.scripts.ps1.analysis.values.read` makes of an expression and what
`refinery.lib.scripts.ps1.analysis.values.render` writes a value back as, both held against what a
real Windows PowerShell 5.1 host made of the same expression. No host is run from here: the
measurements are the ones already taken and checked in, and this module reads them as data, so the
expectations below are the host's rather than ours.
"""
from __future__ import annotations

import decimal
import inspect
import re
import unittest

from typing import Callable, NamedTuple

from test.lib.scripts.ps1.test_oracle import TYPE_TRANSCRIPTS

from refinery.lib.scripts.ps1.analysis.values import (
    INFINITY,
    NOTHING,
    NULL,
    UNKNOWN,
    Ps1Constant,
    Ps1Fact,
    Ps1Outcome,
    Ps1Typed,
    convert,
    make_string_literal,
    read,
    render,
    type_of,
)
from refinery.lib.scripts.ps1.data import resolve_type
from refinery.lib.scripts.ps1.dotnet import Ps1TypeName
from refinery.lib.scripts.ps1.model import (
    Ps1AssignmentExpression,
    Ps1ExpressionStatement,
    Ps1HereString,
    Ps1ParenExpression,
    Ps1StringLiteral,
)
from refinery.lib.scripts.ps1.parser import Ps1Parser
from refinery.lib.scripts.ps1.synth import Ps1Synthesizer

_ROW_PATTERN = re.compile(
    r'\$t = (?P<expression>.+); Write-Output \(,\$t\); Write-Output \$t'
)

#: What counts as a single numeric literal in a measured row. The selection is textual on purpose:
#: it has to be independent of the parser it is used to test, or a lexer that stopped reading a
#: spelling would quietly withdraw the row that measures it instead of failing on it.
_NUMERAL = re.compile(r'[-+]?\(?[0-9][0-9a-zA-Z_.+-]*\)?')


def _type(name: str) -> Ps1TypeName:
    resolved = resolve_type(name)
    assert resolved is not None, name
    return resolved


INT32 = _type('System.Int32')
INT64 = _type('System.Int64')
STRING = _type('System.String')
CHAR = _type('System.Char')
BOOLEAN = _type('System.Boolean')
DOUBLE = _type('System.Double')
DECIMAL = _type('System.Decimal')
OBJECT_ARRAY = _type('System.Object[]')


def _boolean(rendered: str) -> bool:
    """
    The Boolean a host printed. Those two spellings are the only ones it prints, and a third is a
    `KeyError` rather than a truth value read out of the length of a word.
    """
    return {'True': True, 'False': False}[rendered]


#: How the Python payload of a fact is spelled for each .NET type a measured row carries. A type no
#: entry covers is a `KeyError` naming it rather than a comparison against a value read the wrong
#: way round. A Char and a String are spelled alike on purpose: what tells the two apart is the
#: `Ps1TypeName` a fact is stamped with, never the Python object it holds.
_PAYLOADS: dict[str, Callable[[str], int | float | bool | decimal.Decimal | str]] = {
    'System.Int32'   : int,
    'System.Int64'   : int,
    'System.Decimal' : decimal.Decimal,
    'System.Double'  : float,
    'System.Byte'    : int,
    'System.SByte'   : int,
    'System.Int16'   : int,
    'System.UInt16'  : int,
    'System.UInt32'  : int,
    'System.UInt64'  : int,
    'System.Boolean' : _boolean,
    'System.Char'    : str,
    'System.String'  : str,
}


def _transcript(expression: str, witness: str = '$t') -> tuple[str, ...]:
    """
    What a 5.1 host printed for the measured row that assigns `expression` to `$t` and then prints
    `$t` whole and `witness`. A row this module asks for and the corpus does not hold is a
    `KeyError` naming it: nothing here may run a host, so a case has to be measured in
    `test.lib.scripts.ps1.corpus` before it can be asked about.
    """
    row = F'$t = {expression}; Write-Output (,$t); Write-Output {witness}'
    return TYPE_TRANSCRIPTS[row]


def _printed(line: str) -> tuple[str, str]:
    """
    The type a transcript line's value carried and the text the host rendered it as.
    """
    kind, carried, rendered = line.split('\t')
    assert kind == 'OUT', line
    return carried, rendered


def _throws(transcript: tuple[str, ...]) -> bool:
    return transcript[0].startswith('THROW\t')


def _measured(expression: str) -> tuple[str, str]:
    """
    The .NET type and the rendered value a host printed for `expression`. Both witnesses name the
    type — the first for `$t` wrapped in a one-element array and unrolled back out of it, the
    second for `$t` written directly — and a row whose two witnesses disagree is not a measurement
    of one value.
    """
    wrapped, printed = _transcript(expression)
    stamp, reported = _printed(wrapped)
    carried, rendered = _printed(printed)
    assert (stamp, reported) == (carried, rendered), (wrapped, printed)
    return carried, rendered


def _measured_fact(expression: str) -> Ps1Fact:
    name, rendered = _measured(expression)
    return Ps1Constant(_type(name), _PAYLOADS[name](rendered))


def _measured_count(expression: str) -> tuple[str, int]:
    """
    The container type and the element count a host printed for an array shape. Such a row prints
    `$t.Count` as its second witness, because printing the array itself prints its elements and
    never says how many there are.
    """
    wrapped, counted = _transcript(expression, '$t.Count')
    container, _ = _printed(wrapped)
    counter, rendered = _printed(counted)
    assert counter == 'System.Int32', counted
    return container, int(rendered)


def _measured_shape(expression: str) -> tuple[Ps1TypeName, int]:
    container, count = _measured_count(expression)
    return _type(container), count


def _measured_rendering(expression: str) -> tuple[str, str]:
    """
    The container type a host stamped a collection with and the text it wrote the collection as.
    Writing one writes its elements, so this is the witness that says what they are: an element
    that is itself a collection is written as `System.Object[]` rather than as its contents.
    """
    wrapped, _ = _transcript(expression, '$t.Count')
    return _printed(wrapped)


def _measured_elements(expression: str) -> tuple[Ps1Fact, ...]:
    """
    The facts a host printed for the elements of a collection. Writing a collection out unrolls it,
    so every line after the first witness is one element under the type it carried.
    """
    transcript = _transcript(expression)
    elements: list[Ps1Fact] = []
    for line in transcript[1:]:
        name, rendered = _printed(line)
        elements.append(Ps1Constant(_type(name), _PAYLOADS[name](rendered)))
    return tuple(elements)


def _numeral_rows() -> dict[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {}
    for row, transcript in TYPE_TRANSCRIPTS.items():
        match = _ROW_PATTERN.fullmatch(row)
        if match is None:
            continue
        expression = match.group('expression')
        if _NUMERAL.fullmatch(expression):
            rows[expression] = transcript
    return rows


_NUMERAL_ROWS = _numeral_rows()

#: The fact a 5.1 host pins each measured numeral spelling to.
MEASURED: dict[str, Ps1Fact] = {
    expression: _measured_fact(expression)
    for expression, transcript in _NUMERAL_ROWS.items()
    if not _throws(transcript)
}

#: The measured spellings 5.1 itself has no value for. It reports NumericConstantTooLarge for a
#: hexadecimal literal that fills neither width, and reads a digit group broken by an underscore as
#: a command name, because 5.1 has no digit separator.
REFUSED: tuple[str, ...] = tuple(
    expression for expression, transcript in _NUMERAL_ROWS.items() if _throws(transcript)
)

#: Measured spellings the host pins to a value that the domain does not answer. `-(2147483648)` is
#: a unary minus over a parenthesized literal, which 5.1 types by the width of the literal that was
#: negated rather than by the width the negated value needs, and `1.5L` is a real with a long
#: suffix, which the lexer does not read as one literal at all.
UNANSWERED: tuple[str, ...] = (
    '-(2147483648)',
    '-(2147483647)',
    '1.5L',
    '2.5L',
)

DECIDED: tuple[str, ...] = tuple(
    expression for expression in MEASURED if expression not in UNANSWERED
)


def _read(expression: str) -> Ps1Fact:
    """
    The fact `read` answers for the right hand side of `$t = <expression>`, which is the slot every
    measured row puts its expression in.
    """
    statement = Ps1Parser(F'$t = {expression}').parse().body[0]
    assert isinstance(statement, Ps1ExpressionStatement)
    assignment = statement.expression
    assert isinstance(assignment, Ps1AssignmentExpression)
    return read(assignment.value)


def _elements(fact: Ps1Fact) -> tuple[Ps1Fact, ...]:
    """
    The element facts an array constant carries.
    """
    assert isinstance(fact, Ps1Constant), fact
    payload = fact.payload
    assert isinstance(payload, tuple), payload
    return payload


def _read_shape(expression: str) -> tuple[Ps1TypeName | None, int]:
    fact = _read(expression)
    return type_of(fact), len(_elements(fact))


def _spelled(fact: Ps1Fact) -> str:
    """
    The source text a fact is written back as, which is what a caller putting the value into a
    script emits. A fact that names a value always has one, so nothing to write is a broken
    contract here rather than a case to be handled.
    """
    node = render(fact)
    assert node is not None, fact
    return Ps1Synthesizer().convert(node)


#: What counts as a cast in a measured row: the type an accelerator names in brackets, and the
#: operand written against it. The selection is textual for the same reason the numeral one is, and
#: an operand may not open with a colon, so that a read of a static member — `[double]::NaN` — is
#: not taken for a value cast to `Double`.
_CAST = re.compile(r'\[(?P<target>[A-Za-z0-9_]+)\](?P<operand>[^:].*)')

#: A row that measures `-as` instead. `X -as [T]` and `[T]X` are different expressions and the
#: corpus measures both, so the two have to be told apart before either is read as an expectation.
_AS = re.compile(r'(?P<operand>.+) -as \[(?P<target>[A-Za-z0-9_]+)\]')


class _Cast(NamedTuple):
    """
    The two halves of a measured cast, as the row spells them.
    """
    target: str
    operand: str


def _rows(pattern: re.Pattern[str]) -> dict[str, _Cast]:
    matched: dict[str, _Cast] = {}
    for row in TYPE_TRANSCRIPTS:
        outer = _ROW_PATTERN.fullmatch(row)
        if outer is None:
            continue
        expression = outer.group('expression')
        inner = pattern.fullmatch(expression)
        if inner is not None:
            matched[expression] = _Cast(inner.group('target'), inner.group('operand'))
    return matched


CAST_ROWS = _rows(_CAST)
AS_ROWS = _rows(_AS)

#: The measured casts a host answered by throwing, which are therefore the rows that have no value
#: to be held against.
THROWN: tuple[str, ...] = tuple(
    expression for expression in CAST_ROWS if _throws(_transcript(expression))
)

#: The reason a row is declined for want of an operand rather than for want of a rule: `read`
#: evaluates nothing, so the inner cast pins no value for the outer one to convert.
_NO_OPERAND = 'the operand is a cast, and `read` evaluates nothing'

#: The measured casts whose value the domain declines to compute, under the reason `convert`
#: documents for each. Every row here is one a host printed a value for, so an entry states that
#: the domain answers less than the host does — deliberately, because the rule that would produce
#: the value is one no measurement covers, and a .NET rule guessed at is a wrong answer that looks
#: like a right one.
DECLINED: dict[str, tuple[str, ...]] = {
    'a String parses by .NET rules that Python does not share': (
        "[int]'5'",
        "[int]' 5 '",
        "[int]'0'",
        "[int]'0x10'",
        "[int]'1e3'",
        "[double]'1.5'",
        "[double]'1,5'",
        "[bool]''",
        "[bool]'a'",
    ),
    'a Double is written by .NET formatting rather than by Python': (
        '[string]1.5',
    ),
    'a collection joins on $OFS, which only a run decides': (
        "[string]('a', 'b')",
    ),
    'a Single is a width nothing here computes in': (
        '[single]1.5',
    ),
    _NO_OPERAND: (
        '[int][char]65',
        '[int][char]48',
    ),
}

DECLINED_CASTS: tuple[str, ...] = tuple(
    expression for group in DECLINED.values() for expression in group
)

#: Every measured way a script spells a collection, each of which the corpus counts the elements
#: of. Both operators are here on the same parts, because the point of measuring them is that they
#: build different arrays out of them.
COLLECTION_SHAPES: tuple[str, ...] = (
    '@()',
    ', 1',
    ',(1, 2)',
    '(1, 2), 3',
    '@(@(1, 2))',
    '@((1, 2))',
    '@(@(1, 2), 3)',
)

#: The measured casts whose target is a type 5.1 spells no literal for. Each expression is
#: therefore the only way a value of that type can be written back into a script, and the host
#: stamped the row with the type and the value that spelling produces.
CAST_SPELLINGS: tuple[str, ...] = (
    '[byte]5',
    '[sbyte]-5',
    '[int16]7',
    '[uint16]7',
    '[uint32]7',
    '[uint64]7',
    '[uint64]18446744073709551615',
    '[char]0',
    '[char]65',
    '[char]65535',
)


def _converted(expression: str) -> Ps1Outcome:
    """
    What `convert` makes of a measured cast row: the type the row names as its target, applied to
    the fact `read` makes of the operand it is written against.
    """
    row = CAST_ROWS[expression]
    return convert(_read(row.operand), _type(row.target))


class TestPs1MeasuredNumerals(unittest.TestCase):
    """
    A numeric literal's spelling decides its .NET type as much as its digits do, and every
    expectation here is what a 5.1 host printed for that spelling.
    """

    def test_every_numeral_the_corpus_measures_is_selected(self):
        self.assertEqual(
            len(_NUMERAL_ROWS), 36, 'a measured numeral was added or withdrawn')
        self.assertEqual(sorted(REFUSED), ['0xFFFFFFFFFFFFFFFFF', '1_0'])
        self.assertEqual(
            sorted(set(UNANSWERED) - set(MEASURED)), [], 'a spelling named here is not measured')

    def test_every_measured_numeral_reads_as_the_fact_the_host_printed(self):
        for expression in DECIDED:
            with self.subTest(expression):
                self.assertEqual(_read(expression), MEASURED[expression])

    def test_a_numeral_5_1_cannot_read_is_not_read_here_either(self):
        """
        Our lexer accepts both of these and hands the domain a number for each: the seventeen
        hexadecimal digits as an arbitrary-width integer, and the underscored group as ten. 5.1 has
        no value for either, so a fact here would be one no run of the script can produce.
        """
        for expression in REFUSED:
            with self.subTest(expression):
                self.assertEqual(_read(expression), UNKNOWN)

    def test_a_numeral_the_domain_does_not_read_is_refused_rather_than_guessed(self):
        for expression in UNANSWERED:
            with self.subTest(expression):
                self.assertEqual(_read(expression), UNKNOWN)

    @unittest.expectedFailure
    def test_a_negated_parenthesized_literal_keeps_the_width_of_the_literal_it_negates(self):
        """
        `-2147483648` is one literal that fits an Int32, while `-(2147483648)` negates the Int64
        literal `2147483648` and stays an Int64 — the same value under two types. The domain reads
        the first and refuses the second, so the width the parenthesis decides is not answered.
        """
        self.assertEqual(_read('-2147483648'), MEASURED['-2147483648'])
        for expression in ('-(2147483648)', '-(2147483647)'):
            with self.subTest(expression):
                self.assertEqual(_read(expression), MEASURED[expression])

    @unittest.expectedFailure
    def test_a_real_with_a_long_suffix_is_the_rounded_int64_the_host_prints(self):
        """
        5.1 prints Int64 2 for both `1.5L` and `2.5L`, which is the half-to-even rounding of a cast
        rather than a refusal to read the spelling.
        """
        for expression in ('1.5L', '2.5L'):
            with self.subTest(expression):
                self.assertEqual(_read(expression), MEASURED[expression])

    def test_the_same_printed_value_under_two_spellings_is_two_different_facts(self):
        """
        The host prints 255 for `0xFF` and for `0xFFL`, and 1.5 for `1.5` and for `1.5d`, and stamps
        each pair with two different types. A domain that carried the value alone could not tell the
        members of a pair apart, and every operation whose result type follows its operands' would
        then answer for the wrong one.
        """
        by_value: dict[str, dict[str, str]] = {}
        for expression in DECIDED:
            name, rendered = _measured(expression)
            by_value.setdefault(rendered, {})[name] = expression
        collisions = {
            rendered: spellings for rendered, spellings in by_value.items() if len(spellings) > 1
        }
        self.assertEqual(sorted(collisions), ['-1', '1.5', '1024', '255'])
        for rendered, spellings in collisions.items():
            with self.subTest(rendered):
                facts = {_read(expression) for expression in spellings.values()}
                self.assertEqual(len(facts), len(spellings))


class TestPs1FactLattice(unittest.TestCase):

    def test_nothing_known_and_no_value_are_different_facts(self):
        self.assertNotEqual(UNKNOWN, NULL)
        self.assertIsNone(type_of(UNKNOWN))
        self.assertIsNone(type_of(NULL))

    def test_a_variable_the_source_does_not_pin_is_unknown_while_the_null_literal_is_null(self):
        self.assertEqual(_read('$x'), UNKNOWN)
        self.assertEqual(_read('$null'), NULL)

    def test_type_of_reads_the_type_of_a_typed_fact_and_of_a_constant(self):
        self.assertEqual(type_of(Ps1Typed(CHAR)), CHAR)
        self.assertEqual(type_of(Ps1Constant(CHAR, 'A')), CHAR)

    def test_a_char_and_a_string_carrying_equal_payloads_are_different_facts(self):
        """
        5.1 answers `[char]65 -is [char]` True and `'A' -is [char]` False, prints 66 for
        `1 + [char]65` and throws on `1 + 'A'`, and refuses to replicate a Char where it replicates
        a String. Both carry the same `'A'`, so only the type they are stamped with tells them
        apart.
        """
        self.assertNotEqual(Ps1Constant(CHAR, 'A'), Ps1Constant(STRING, 'A'))
        self.assertNotEqual(type_of(Ps1Constant(CHAR, 'A')), type_of(Ps1Constant(STRING, 'A')))

    def test_a_fact_is_a_value_so_one_read_twice_is_one_fact(self):
        self.assertEqual(_read('0xFF'), _read('0xFF'))
        self.assertEqual(len({_read('0xFF'), _read('0xFF'), _read('0xFFL')}), 2)


class TestPs1Outcome(unittest.TestCase):

    def test_the_refusal_names_no_value_and_claims_no_absence_of_a_throw(self):
        self.assertEqual(NOTHING, Ps1Outcome(False, UNKNOWN))
        self.assertIsNone(type_of(NOTHING.value))

    def test_an_outcome_carries_a_type_and_a_possible_throw_at_once(self):
        outcome = Ps1Outcome(True, Ps1Typed(INT32))
        self.assertEqual(outcome, (True, Ps1Typed(INT32)))
        self.assertEqual(type_of(outcome.value), INT32)


class TestPs1LiteralFacts(unittest.TestCase):

    def test_a_string_literal_reads_as_the_string_it_spells(self):
        self.assertEqual(_read("'abc'"), Ps1Constant(STRING, 'abc'))
        self.assertEqual(_read('"abc"'), Ps1Constant(STRING, 'abc'))

    def test_a_here_string_reads_as_the_text_between_its_delimiters(self):
        source = inspect.cleandoc("""
            @'
            abc
            '@
        """)
        self.assertEqual(_read(source), Ps1Constant(STRING, 'abc'))

    def test_the_boolean_variables_read_as_booleans_in_any_casing(self):
        self.assertEqual(_read('$true'), Ps1Constant(BOOLEAN, True))
        self.assertEqual(_read('$TRUE'), Ps1Constant(BOOLEAN, True))
        self.assertEqual(_read('$false'), Ps1Constant(BOOLEAN, False))

    def test_a_parenthesis_around_a_literal_reads_through_to_the_literal(self):
        self.assertEqual(_read('(0xFF)'), MEASURED['0xFF'])
        self.assertEqual(_read('((1kb))'), MEASURED['1kb'])


class TestPs1StringSpelling(unittest.TestCase):

    def test_multiline_string_emitted_as_here_string(self):
        node = make_string_literal('line1\nline2')
        self.assertIsInstance(node, Ps1HereString)
        self.assertEqual(node.value, 'line1\nline2')
        self.assertIn("@'\n", node.raw)
        node2 = make_string_literal('no newlines')
        self.assertIsInstance(node2, Ps1StringLiteral)

    def test_make_string_literal_avoids_herestring_breakout(self):
        # A value with a line beginning with the here-string terminator `'@` must not be emitted as
        # a here-string, or it would close the string early; a safe multi-line value still may.
        unsafe = make_string_literal("a\n'@\nb")
        self.assertNotIsInstance(unsafe, Ps1HereString)
        safe = make_string_literal('a\nb')
        self.assertIsInstance(safe, Ps1HereString)


class TestPs1ValueSpelling(unittest.TestCase):
    """
    What `render` writes a fact back as. A value always has a spelling — a literal where 5.1 has
    one for its type, the cast of a decimal numeral where it does not — so a caller holding a
    value never has to choose between leaving the source alone and writing something else.
    Nothing to write is therefore reserved for a fact that names no value.
    """

    def test_only_a_fact_that_names_no_value_is_left_unwritten(self):
        self.assertIsNone(render(UNKNOWN))
        self.assertIsNone(render(Ps1Typed(INT32)))
        self.assertIsNone(render(Ps1Typed(OBJECT_ARRAY)))
        self.assertEqual(_spelled(NULL), '$Null')
        self.assertEqual(_read('$Null'), NULL)

    def test_a_fact_written_and_read_back_is_the_fact_it_was(self):
        facts = [MEASURED[expression] for expression in DECIDED]
        facts.extend(_read(spelling) for spelling in COLLECTION_SHAPES)
        facts.extend(_read(spelling) for spelling in (
            "'abc'",
            "''",
            '$True',
            '$False',
            '$Null',
            "'a', 1",
        ))
        for fact in facts:
            with self.subTest(repr(fact)):
                self.assertEqual(_read(_spelled(fact)), fact)

    def test_a_spelling_read_and_written_back_is_the_spelling_it_was(self):
        for spelling in (
            "'abc'",
            '255',
            '-1',
            '1024L',
            '10d',
            '1.5',
            '$True',
            '$False',
            '$Null',
            '@()',
            "'a', 1",
            ',(1, 2)',
        ):
            with self.subTest(spelling):
                self.assertEqual(_spelled(_read(spelling)), spelling)

    def test_a_numeral_carries_the_suffix_that_pins_its_type_and_no_other(self):
        """
        5.1 has two numeric literal suffixes: `L` names an Int64 and `d` a Decimal. An integer
        numeral written without one is an Int32 and a real without one a Double, so those two are
        written bare.
        """
        self.assertEqual(_spelled(Ps1Constant(INT32, 7)), '7')
        self.assertEqual(_spelled(Ps1Constant(INT64, 1)), '1L')
        self.assertEqual(_spelled(Ps1Constant(DECIMAL, decimal.Decimal(10))), '10d')
        self.assertEqual(_spelled(Ps1Constant(DOUBLE, 1.5)), '1.5')
        self.assertEqual(
            [MEASURED['007'], MEASURED['1L'], MEASURED['10d'], MEASURED['1.5']],
            [
                Ps1Constant(INT32, 7),
                Ps1Constant(INT64, 1),
                Ps1Constant(DECIMAL, decimal.Decimal(10)),
                Ps1Constant(DOUBLE, 1.5),
            ],
        )

    def test_a_value_of_a_type_no_literal_spells_is_written_as_the_cast_the_host_measured(self):
        self.assertEqual(sorted(set(CAST_SPELLINGS) - set(CAST_ROWS)), [])
        self.assertEqual(
            {expression: _spelled(_measured_fact(expression)) for expression in CAST_SPELLINGS},
            {expression: expression for expression in CAST_SPELLINGS},
        )

    def test_the_cast_a_value_is_written_as_has_no_read_twin(self):
        """
        `read` answers what the source pins and evaluates nothing, so it does not read a cast back
        and there is no round trip to hold this arm to. What says the spelling is right is the
        host, which stamped `[byte]5` a Byte 5.
        """
        for expression in CAST_SPELLINGS:
            with self.subTest(expression):
                self.assertEqual(_read(expression), UNKNOWN)
        self.assertEqual(_measured('[byte]5'), ('System.Byte', '5'))

    def test_a_char_and_a_one_character_string_are_not_written_alike(self):
        """
        Measured, `[char]65` is a Char and `'A'` a String: the two answer `-is [char]` differently,
        carry different methods and reach different numbers under `[int]`. One spelling for both
        would erase that at the one point where a value re-enters a script.
        """
        self.assertEqual(_measured('[char]65'), ('System.Char', 'A'))
        self.assertEqual(_spelled(Ps1Constant(CHAR, 'A')), '[char]65')
        self.assertEqual(_spelled(Ps1Constant(STRING, 'A')), "'A'")
        self.assertEqual(_read("'A'"), Ps1Constant(STRING, 'A'))

    def test_the_boolean_values_are_written_as_the_variables_that_hold_them(self):
        self.assertEqual(_spelled(Ps1Constant(BOOLEAN, True)), '$True')
        self.assertEqual(_spelled(Ps1Constant(BOOLEAN, False)), '$False')

    def test_a_double_that_no_literal_and_no_cast_reaches_is_left_unwritten(self):
        for payload in (INFINITY, -INFINITY, float('nan')):
            with self.subTest(payload):
                self.assertIsNone(render(Ps1Constant(DOUBLE, payload)))
        self.assertEqual(_spelled(MEASURED['100000000000000000000000000000000']), '1e+32')

    def test_a_collection_holds_each_element_under_the_type_the_host_printed_for_it(self):
        fact = _read("'a', 1")
        self.assertEqual(_elements(fact), _measured_elements("'a', 1"))
        self.assertEqual(_spelled(fact), "'a', 1")

    def test_the_array_operator_writes_the_empty_collection_and_no_other(self):
        """
        `@(...)` unrolls the collection it is handed, so writing a one-element collection with it
        would hand back what was inside instead: measured, `,(1, 2)` holds one element and
        `@((1, 2))` holds two. The comma operator is what writes every non-empty collection.
        """
        self.assertEqual(_measured_count('@()'), ('System.Object[]', 0))
        self.assertEqual(_measured_count(',(1, 2)'), ('System.Object[]', 1))
        self.assertEqual(_measured_count('@((1, 2))'), ('System.Object[]', 2))
        self.assertEqual(_spelled(_read('@()')), '@()')
        self.assertEqual(_spelled(_read(',(1, 2)')), ',(1, 2)')
        self.assertNotEqual(_read(',(1, 2)'), _read('@((1, 2))'))

    def test_a_collection_is_written_by_what_it_holds_and_not_by_how_it_was_spelled(self):
        """
        `@(@(1, 2))` and `@((1, 2))` are both the two-element collection `1, 2`, and `@(@(1, 2), 3)`
        is the two-element one whose first element is a collection. Each is written back as what it
        holds, which is a different spelling from the one it was read from and the same value.
        """
        self.assertEqual(_spelled(_read('@(@(1, 2))')), '1, 2')
        self.assertEqual(_spelled(_read('@((1, 2))')), '1, 2')
        self.assertEqual(_spelled(_read('@(@(1, 2), 3)')), '(1, 2), 3')
        for expression in ('@(@(1, 2))', '@((1, 2))', '@(@(1, 2), 3)'):
            with self.subTest(expression):
                self.assertEqual(
                    _read_shape(_spelled(_read(expression))), _measured_shape(expression))

    def test_a_collection_holding_a_value_with_no_spelling_is_not_written_shorter(self):
        self.assertIsNone(render(Ps1Constant(OBJECT_ARRAY, (Ps1Constant(INT32, 1), UNKNOWN))))
        self.assertEqual(_spelled(Ps1Constant(OBJECT_ARRAY, (Ps1Constant(INT32, 1),))), ',1')

    def test_a_string_value_is_written_as_a_literal_that_reads_back_as_it(self):
        for value in ('abc', '', "it's", 'a"b', 'a`b', 'a$b', 'a\nb', 'a\tb', "a\n'@\nb"):
            with self.subTest(value):
                self.assertEqual(
                    _read(_spelled(Ps1Constant(STRING, value))), Ps1Constant(STRING, value))


class TestPs1ArrayFacts(unittest.TestCase):
    """
    The two ways of writing a collection do not build the same array out of the same parts. The
    comma operator takes each operand whole; `@(...)` collects what a pipeline hands it, and a
    pipeline unrolls a collection one level on the way. Every expectation here is the length a 5.1
    host counted.
    """

    def test_a_measured_array_shape_reads_as_an_object_array_of_the_measured_length(self):
        self.assertEqual(
            {expression: _read_shape(expression) for expression in COLLECTION_SHAPES},
            {expression: _measured_shape(expression) for expression in COLLECTION_SHAPES},
        )

    def test_the_comma_operator_takes_its_operand_whole_where_the_array_operator_unrolls_it(self):
        """
        Measured, `,(1, 2)` counts one element and `@((1, 2))` counts two out of the same parts, so
        the collection `@(...)` was handed is not the one it built. Reading the two alike would let
        a caller lose or gain a level of nesting without anything in the answer saying so.
        """
        self.assertEqual(_measured_count(',(1, 2)'), ('System.Object[]', 1))
        self.assertEqual(_measured_count('@((1, 2))'), ('System.Object[]', 2))
        self.assertNotEqual(_read(',(1, 2)'), _read('@((1, 2))'))
        self.assertEqual(_read(',(1, 2)'), Ps1Constant(OBJECT_ARRAY, (_read('1, 2'),)))
        self.assertEqual(_read('@((1, 2))'), _read('1, 2'))

    def test_the_array_operator_unrolls_the_value_a_statement_produced_and_not_what_was_in_it(self):
        """
        `@(@(1, 2), 3)` and `(1, 2), 3` are written by the host alike — two elements, the first of
        them an `Object[]` — so the unrolling happened once, to the collection the statement
        produced, and did not reach into the collection that was inside it.
        """
        self.assertEqual(
            _measured_rendering('@(@(1, 2), 3)'), ('System.Object[]', 'System.Object[] 3'))
        self.assertEqual(
            _measured_rendering('(1, 2), 3'), _measured_rendering('@(@(1, 2), 3)'))
        self.assertEqual(_read('@(@(1, 2), 3)'), _read('(1, 2), 3'))
        self.assertEqual(
            [type_of(element) for element in _elements(_read('@(@(1, 2), 3)'))],
            [OBJECT_ARRAY, INT32],
        )

    def test_a_collection_the_array_operator_unrolled_holds_what_was_inside_it(self):
        self.assertEqual(_measured_count('@(@(1, 2))'), ('System.Object[]', 2))
        self.assertEqual(_read('@(@(1, 2))'), _read('1, 2'))
        self.assertEqual(_read('@(@(1, 2))'), _read('@((1, 2))'))

    def test_an_array_literal_carries_the_fact_of_each_element(self):
        self.assertEqual(
            _read('0xFF, 1kb'),
            Ps1Constant(OBJECT_ARRAY, (MEASURED['0xFF'], MEASURED['1kb'])),
        )

    def test_an_array_whose_element_is_not_pinned_is_not_read_as_a_shorter_array(self):
        self.assertNotIsInstance(_read('0xFF, $x'), Ps1Constant)


class TestPs1ReadOnlyReadsTheSource(unittest.TestCase):
    """
    `read` answers what the source pins and evaluates nothing, so an expression whose value only a
    run produces is refused however well known that value is. A guess here would be worse than no
    answer: the ledger's Char rows are all one erasure, and a Char answered as a one-character
    String changes what `1 + $t` and `$t * 3` do.
    """

    def test_an_expression_whose_value_only_an_evaluation_produces_is_refused(self):
        for expression in (
            '[char]65',
            "'AB'.Length",
            "5 + '5'",
            "'ABC'[0]",
            '"a$b"',
            '@{ a = 1 }',
            'Get-Date',
        ):
            with self.subTest(expression):
                self.assertEqual(_read(expression), UNKNOWN)

    def test_reading_nothing_at_all_is_unknown(self):
        self.assertEqual(read(None), UNKNOWN)
        self.assertEqual(read(Ps1ParenExpression()), UNKNOWN)


class TestPs1MeasuredCasts(unittest.TestCase):
    """
    What `[target] value` produces, held against what a 5.1 host printed for the same cast. Each
    row is answered exactly as the host answered it or declined, and the failure this class exists
    to catch is the third thing: a value, a type, or an absence of a throw that the host did not
    produce.
    """

    def test_every_cast_the_corpus_measures_is_selected(self):
        self.assertEqual(len(CAST_ROWS), 47, 'a measured cast was added or withdrawn')
        self.assertEqual(sorted(set(DECLINED_CASTS) - set(CAST_ROWS)), [])
        self.assertEqual(sorted(set(DECLINED_CASTS) & set(THROWN)), [])

    def test_a_measured_cast_the_domain_pins_is_pinned_to_the_fact_the_host_printed(self):
        pinned = {
            expression: _converted(expression)
            for expression in CAST_ROWS
            if expression not in THROWN
            and isinstance(_converted(expression).value, Ps1Constant)
        }
        self.assertEqual(
            pinned,
            {
                expression: Ps1Outcome(False, _measured_fact(expression))
                for expression in pinned
            },
        )

    def test_the_measured_casts_whose_value_is_declined_are_exactly_the_documented_ones(self):
        declined = [
            expression for expression in CAST_ROWS
            if expression not in THROWN
            and not isinstance(_converted(expression).value, Ps1Constant)
        ]
        self.assertEqual(sorted(declined), sorted(DECLINED_CASTS))

    def test_a_declined_cast_still_carries_the_type_the_host_reported(self):
        """
        A number the domain does not compute is still a value of the type the host stamped it with,
        and every declined cast whose operand is read has that type in the grid: `[int]'0x10'` is an
        Int32 whether or not this can say that .NET reads those digits as sixteen.
        """
        for expression in sorted(set(DECLINED_CASTS) - set(DECLINED[_NO_OPERAND])):
            with self.subTest(expression):
                name, _ = _measured(expression)
                self.assertEqual(_converted(expression).value, Ps1Typed(_type(name)))

    def test_a_cast_of_an_operand_the_source_does_not_pin_answers_nothing(self):
        """
        The host prints 65 for `[int][char]65` and 48 for `[int][char]48`, and `read` evaluates
        neither inner cast. A cast of a value nothing is known about has no type either: what
        `[int]` does is decided by what it is given, and over a String it throws.
        """
        for expression in DECLINED[_NO_OPERAND]:
            with self.subTest(expression):
                self.assertEqual(_converted(expression), NOTHING)

    def test_a_cast_the_host_threw_on_pins_no_value_and_reports_the_throw(self):
        """
        5.1 throws rather than wrapping where a value does not fit its target, and rather than
        yielding zero for a String that spells no number. The last is the cell shape a cast has most
        often — an Int32, or it throws — and the type it names is not a claim that a value came out.
        """
        self.assertEqual(
            {expression: _converted(expression) for expression in THROWN},
            {
                '[byte]300'       : Ps1Outcome(True, UNKNOWN),
                '[byte]-1'        : Ps1Outcome(True, UNKNOWN),
                '[int]2147483648' : Ps1Outcome(True, UNKNOWN),
                '[char]65536'     : Ps1Outcome(True, UNKNOWN),
                '[char]-1'        : Ps1Outcome(True, UNKNOWN),
                "[int]'abc'"      : Ps1Outcome(True, Ps1Typed(INT32)),
            },
        )

    def test_a_row_that_measures_as_is_not_a_cast_and_is_no_expectation_here(self):
        """
        A conversion that cannot be made yields `$null` for `-as` and throws for a cast, measured:
        the corpus prints `<null>` for `300 -as [byte]` where `[byte]300` throws. An `-as` row read
        as a cast's expectation would pin `convert` to `$null` for exactly the operands where the
        two expressions differ.
        """
        self.assertEqual(
            sorted(AS_ROWS), ["'abc' -as [int]", '300 -as [byte]', '5 -as [long]'])
        self.assertEqual(sorted(set(AS_ROWS) & set(CAST_ROWS)), [])
        self.assertEqual(_measured('300 -as [byte]'), ('', '<null>'))
        self.assertEqual(_throws(_transcript('[byte]300')), True)


class TestPs1CharConversions(unittest.TestCase):
    """
    A Char is the one value no literal spells, so every measured cast over one is a row `convert`
    has to be asked for directly. It is also the erasure the fact layer exists to undo: 65 and `A`
    are one Char under two casts, and a Char answered as a one-character String changes what
    `1 + $t` and `$t * 3` do.
    """

    def test_a_char_casts_to_the_number_and_to_the_text_the_host_printed(self):
        """
        Measured, `[int][char]65` is Int32 65 and `[string][char]65` is `A`.
        """
        self.assertEqual(
            convert(Ps1Constant(CHAR, 'A'), INT32), Ps1Outcome(False, Ps1Constant(INT32, 65)))
        self.assertEqual(
            convert(Ps1Constant(CHAR, 'A'), STRING), Ps1Outcome(False, Ps1Constant(STRING, 'A')))

    def test_a_char_reaches_no_target_the_corpus_does_not_measure(self):
        """
        The corpus measures a Char cast to Int32 and to String, and to nothing else. What
        `[bool][char]0` is remains a .NET rule no row records, so the type the grid names is the
        whole answer and the value is left where the measurement left it.
        """
        self.assertEqual(convert(Ps1Constant(CHAR, 'A'), BOOLEAN).value, Ps1Typed(BOOLEAN))
        self.assertEqual(convert(Ps1Constant(CHAR, 'A'), DOUBLE).value, Ps1Typed(DOUBLE))


class TestPs1ConvertRefusals(unittest.TestCase):
    """
    A cast reads its type from the grid and its value from a kernel, and the two axes fail apart:
    where the kernel has no rule the type still stands, and where the grid has no cell there is no
    answer at all.
    """

    def test_a_fact_that_is_not_a_value_is_never_converted_into_one(self):
        """
        `$null` and a value known only by its type are both looked up in the grid and neither is
        computed from, so `[int] $null` is an Int32 whose number is unanswered. A fact that names no
        type at all indexes no cell, and the cast answers nothing.
        """
        self.assertEqual(convert(UNKNOWN, INT32), NOTHING)
        self.assertEqual(convert(Ps1Typed(DOUBLE), INT32).value, Ps1Typed(INT32))
        self.assertEqual(convert(NULL, INT32).value, Ps1Typed(INT32))

    def test_a_cast_the_grid_does_not_cover_is_refused_rather_than_assumed(self):
        """
        The grid holds the targets the capture measured, and `System.DateTime` is not one of them. A
        cast to a type nothing measured produces no fact: neither the identity nor a nearest rule is
        an answer that was ever observed.
        """
        self.assertEqual(convert(Ps1Constant(INT32, 5), _type('System.DateTime')), NOTHING)

    def test_a_double_is_not_carried_into_a_decimal_by_python(self):
        """
        `[decimal]` of a Double converts by the digits the value is written with, where Python's
        `Decimal` of a `float` carries the binary expansion instead — `Decimal(1.1)` begins
        1.10000000000000008. Nothing measures the cast, so the type is the answer and the number is
        not.
        """
        self.assertEqual(convert(Ps1Constant(DOUBLE, 1.5), DECIMAL).value, Ps1Typed(DECIMAL))


if __name__ == '__main__':
    unittest.main()
