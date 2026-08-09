"""
What `refinery.lib.scripts.ps1.analysis.values.read` makes of an expression, held against what a
real Windows PowerShell 5.1 host made of the same expression. No host is run from here: the
measurements are the ones already taken and checked in, and this module reads them as data, so the
expectations below are the host's rather than ours.
"""
from __future__ import annotations

import decimal
import inspect
import re
import unittest

from typing import Callable

from test.lib.scripts.ps1.test_oracle import TYPE_TRANSCRIPTS

from refinery.lib.scripts.ps1.analysis.values import (
    NOTHING,
    NULL,
    UNKNOWN,
    Ps1Constant,
    Ps1Fact,
    Ps1Outcome,
    Ps1Typed,
    read,
    type_of,
)
from refinery.lib.scripts.ps1.data import resolve_type
from refinery.lib.scripts.ps1.dotnet import Ps1TypeName
from refinery.lib.scripts.ps1.model import (
    Ps1AssignmentExpression,
    Ps1ExpressionStatement,
    Ps1ParenExpression,
)
from refinery.lib.scripts.ps1.parser import Ps1Parser

_ROW_PATTERN = re.compile(
    r'\$t = (?P<expression>.+); Write-Output \(,\$t\); Write-Output \$t'
)

#: What counts as a single numeric literal in a measured row. The selection is textual on purpose:
#: it has to be independent of the parser it is used to test, or a lexer that stopped reading a
#: spelling would quietly withdraw the row that measures it instead of failing on it.
_NUMERAL = re.compile(r'[-+]?\(?[0-9][0-9a-zA-Z_.]*\)?')


def _type(name: str) -> Ps1TypeName:
    resolved = resolve_type(name)
    assert resolved is not None, name
    return resolved


INT32 = _type('System.Int32')
STRING = _type('System.String')
CHAR = _type('System.Char')
BOOLEAN = _type('System.Boolean')
OBJECT_ARRAY = _type('System.Object[]')

#: How the Python payload of a fact is spelled for each .NET type a measured numeral carries. A
#: type no entry covers is a `KeyError` naming it rather than a comparison against a value read the
#: wrong way round.
_PAYLOADS: dict[str, Callable[[str], int | float | decimal.Decimal]] = {
    'System.Int32'   : int,
    'System.Int64'   : int,
    'System.Decimal' : decimal.Decimal,
    'System.Double'  : float,
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


class TestPs1MeasuredNumerals(unittest.TestCase):
    """
    A numeric literal's spelling decides its .NET type as much as its digits do, and every
    expectation here is what a 5.1 host printed for that spelling.
    """

    def test_every_numeral_the_corpus_measures_is_selected(self):
        self.assertEqual(
            len(_NUMERAL_ROWS), 33, 'a measured numeral was added or withdrawn')
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


class TestPs1ArrayFacts(unittest.TestCase):

    def test_a_measured_array_shape_reads_as_an_object_array_of_the_measured_length(self):
        for expression in ('@()', ', 1'):
            with self.subTest(expression):
                container, count = _measured_count(expression)
                fact = _read(expression)
                self.assertEqual(
                    (type_of(fact), len(_elements(fact))),
                    (_type(container), count),
                )

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


if __name__ == '__main__':
    unittest.main()
