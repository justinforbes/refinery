"""
What `refinery.lib.scripts.ps1.deobfuscation.emulator._Ps1Interpreter` computes for an expression,
held against what a real Windows PowerShell 5.1 host printed for the same expression. No host is run
from here: the measurements are the ones already taken and checked in as
`test.lib.scripts.ps1.test_oracle.TYPE_TRANSCRIPTS`, and this module reads them as data.

**The interpreter's currency carries no .NET type.** A `System.Char` and a one-character
`System.String` are the same Python `str` to it, and a `System.Byte`, an `Int32` and an `Int64` are
the same Python `int`, so the recording says more than the interpreter can answer. `_CURRENCY` is
where that is decided: it names, for each measured .NET type, the one value of the currency a
correct interpreter has to produce, and every type that collapses onto another collapses there and
nowhere else. What it does *not* collapse is `int` against `float`, `str` and `bool` and `None`
against each other: those the currency does spell apart, the interpreter does produce each of them,
and the value that reaches a folded script is one of them rather than the other — so a mismatch
between two of those is a wrong answer and is recorded as one. A `Decimal` and a `Single` have no
counterpart in the currency at all and their rows are held against nothing, which is why the types
that leave the population are pinned by name beside the population itself.

Two populations, because refusing to answer and answering wrongly are different things and the
deobfuscator only survives one of them. `_measured_rows` is what 5.1 computed a value for, and a
divergence there is a wrong constant folded into the emitted script. `_thrown_rows` is what 5.1
refused, where the only safe answer is to refuse as well and any value at all is a fold of an
expression that in fact halts the script.

Both ledgers hold the *values* rather than prose, and are compared whole. An entry states what 5.1
printed and what the interpreter answered instead, so a failure can be read without leaving this
file; and because the wrong answer is pinned too, a divergence that changes its shape fails as
loudly as one that appears or is fixed.
"""
from __future__ import annotations

import functools
import re
import unittest

from typing import TYPE_CHECKING, Callable, NamedTuple

from test.lib.scripts.ps1.test_oracle import TYPE_TRANSCRIPTS

from refinery.lib.scripts.ps1.deobfuscation.emulator import (
    InvokeExpression,
    _Ps1Interpreter,
    _Ps1InterpreterError,
)
from refinery.lib.scripts.ps1.model import Ps1AssignmentExpression, Ps1ExpressionStatement
from refinery.lib.scripts.ps1.parser import Ps1Parser

if TYPE_CHECKING:
    from refinery.lib.scripts.ps1.deobfuscation.emulator import _Value

#: The two exceptions every caller of the interpreter catches — `Ps1FunctionEvaluator`,
#: `Ps1ForEachPipeline` and `evaluate_truthy` alike. Anything else escapes into the unit, so it is
#: neither an answer nor a refusal and is not counted as either.
_REFUSALS = (_Ps1InterpreterError, InvokeExpression)

_ROW = re.compile(r'\$t = (?P<expression>.+); Write-Output \(,\$t\); Write-Output \$t')


def _boolean(rendered: str) -> bool:
    return {'True': True, 'False': False}[rendered]


def _null(rendered: str) -> None:
    return {'<null>': None}[rendered]


#: The one value of the interpreter's currency that a measured witness denotes, by the .NET type the
#: recording stamped it with. A type with no entry here takes its rows out of the population rather
#: than being compared against an approximation of itself. A null value's witness names no type at
#: all, which is what the empty key is.
_CURRENCY: dict[str, Callable[[str], _Value]] = {
    'System.Byte'    : int,
    'System.SByte'   : int,
    'System.Int16'   : int,
    'System.UInt16'  : int,
    'System.Int32'   : int,
    'System.UInt32'  : int,
    'System.Int64'   : int,
    'System.UInt64'  : int,
    'System.Double'  : float,
    'System.Boolean' : _boolean,
    'System.Char'    : str,
    'System.String'  : str,
    ''               : _null,
}


class _Divergence(NamedTuple):
    measured: _Value
    computed: _Value


class _Coverage(NamedTuple):
    measured: int
    answered: int


#: How many rows of each measured .NET type are held against the interpreter, and how many of those
#: it computes a value for at all. A census rather than a total, so that a type whose rows stop
#: being compared cannot hide behind another type's growth — and a row that turns from an answer
#: into a refusal moves a number here rather than quietly leaving the comparison below.
COVERAGE: dict[str, _Coverage] = {
    ''               : _Coverage(3, 1),
    'System.Boolean' : _Coverage(88, 56),
    'System.Byte'    : _Coverage(5, 5),
    'System.Char'    : _Coverage(8, 6),
    'System.Double'  : _Coverage(33, 17),
    'System.Int16'   : _Coverage(1, 0),
    'System.Int32'   : _Coverage(100, 69),
    'System.Int64'   : _Coverage(26, 19),
    'System.SByte'   : _Coverage(2, 0),
    'System.String'  : _Coverage(31, 18),
    'System.UInt16'  : _Coverage(2, 0),
    'System.UInt32'  : _Coverage(4, 0),
    'System.UInt64'  : _Coverage(2, 0),
}

#: The measured types no value of the currency denotes, and how many rows each takes out of the
#: population. `Decimal` is not a `float` — 5.1 computes it exactly and to a different precision —
#: and `Single` is not one either, so a row carrying one has nothing here to be right about.
TYPES_OUTSIDE_THE_CURRENCY: dict[str, int] = {
    'System.Decimal' : 17,
    'System.Single'  : 1,
}

#: How many measured rows write a collection rather than one value. Such a row's two witnesses
#: disagree by design — writing a collection unrolls it, so the container's type stands on the first
#: line and its elements' on the rest — and the population is the rows whose witnesses agree.
COLLECTION_ROWS: int = 5

#: How many measured rows 5.1 answered by throwing. The population `ANSWERED_THROWS` is drawn from.
THROWING_ROWS: int = 35

#: Where the interpreter computes a value 5.1 did not. Each entry is a constant the deobfuscator
#: folds wrongly into the script it emits.
DIVERGENCES: dict[str, _Divergence] = {
    # A number that leaves the width of its type becomes a Double on 5.1, which is lossy and says
    # so. The interpreter keeps computing in Python integers, which are exact and unbounded, so the
    # answer is a different number written a different way.
    '2147483647 + 1'                     : _Divergence(2147483648.0, 2147483648),
    '-2147483648 / -1'                   : _Divergence(2147483648.0, 2147483648),
    '2147483647 * 2147483647'            : _Divergence(4.61168601413242e+18, 4611686014132420609),
    '512MB * 512MB'                      : _Divergence(2.88230376151712e+17, 288230376151711744),
    '9223372036854775807 + 2'            : _Divergence(9.22337203685478e+18, 9223372036854775809),
    '9223372036854775807L + 1'           : _Divergence(9.22337203685478e+18, 9223372036854775808),
    '9223372036854775807L - -1L'         : _Divergence(9.22337203685478e+18, 9223372036854775808),
    '$true + 9223372036854775807L'       : _Divergence(9.22337203685478e+18, 9223372036854775808),
    '-2147483648 - 9223372036854775807L' : _Divergence(-9.22337203900226e+18, -9223372039002259455),

    # The left operand decides the operation and the right one is converted to its type, so a
    # number on the left reads whatever stands on the right as a number; a Boolean the operator has
    # no method for falls back to Int32 first, which is why `$true + ''` is 1. The interpreter
    # dispatches on Python's types instead and concatenates.
    "0 + '5'"                            : _Divergence(5, '05'),
    "1 + '5'"                            : _Divergence(6, '15'),
    "5 + '5'"                            : _Divergence(10, '55'),
    "1 + '+5'"                           : _Divergence(6, '1+5'),
    "1 + ' 7 '"                          : _Divergence(8, '1 7 '),
    "1 + '  '"                           : _Divergence(1, '1  '),
    "1 + '1kb'"                          : _Divergence(1025, '11kb'),
    "1 + '1e3'"                          : _Divergence(1001.0, '11e3'),
    "1 + '1.5L'"                         : _Divergence(3, '11.5L'),
    "1 + '2147483648'"                   : _Divergence(2147483649, '12147483648'),
    "1 + '0xFFFFFFFF'"                   : _Divergence(0, '10xFFFFFFFF'),
    "12 + '0xabc'"                       : _Divergence(2760, '120xabc'),
    "$true + ''"                         : _Divergence(1, 'True'),
    '0 + [char]65'                       : _Divergence(65, '0A'),

    # A Char is a number wherever a number is wanted, and its code point is the number. The
    # interpreter holds it as a one-character string, which reads as zero.
    '[int][char]48'                      : _Divergence(48, 0),
    '[char]48 - 0.0'                     : _Divergence(48.0, 0.0),
    '1.5 * [char]48'                     : _Divergence(72.0, 0.0),
    '[char]48 -band [byte]255'           : _Divergence(48, 0),

    # A one-element collection is as true as the element inside it, and a Char is as true as the
    # code point it carries. The interpreter asks Python, for which a non-empty list is true and a
    # one-character string is true.
    '@(0) -and $true'                    : _Divergence(False, True),
    '@($false) -and $true'               : _Divergence(False, True),
    "(,'') -and $true"                   : _Divergence(False, True),
    '(,0.0) -and $true'                  : _Divergence(False, True),
    '(,@()) -and $true'                  : _Divergence(False, True),
    '-not @(0)'                          : _Divergence(True, False),
    '[char]0 -or $false'                 : _Divergence(False, True),
    '$true -and [char]0'                 : _Divergence(False, True),
    '-not [char]0'                       : _Divergence(True, False),

    # `-contains` converts each element to the type of the value it is asked about before comparing
    # it. The interpreter compares the Python objects, for which a string is never an integer.
    "@('1') -contains 1"                 : _Divergence(True, False),
    "@(1) -contains '1'"                 : _Divergence(True, False),

    # A PowerShell wildcard is not an fnmatch pattern: a backtick escapes the character behind it,
    # and `[!a]` is the two-character set `!a` rather than a negated class. Nor is `-match` Python's
    # `re.IGNORECASE`, which folds the long s onto s where .NET does not.
    "'a*' -like 'a`*'"                   : _Divergence(True, False),
    "'b' -like '[!a]'"                   : _Divergence(False, True),
    "'ſ' -match 's'"                     : _Divergence(False, True),

    # A conversion answers inside the width of its target or not at all: a hexadecimal string
    # reaches Int32 as a bit pattern, `-as` hands back `$null` where a cast would throw, and a shift
    # keeps the type it shifted rather than widening past it.
    "[int]'0xFFFFFFFF'"                  : _Divergence(-1, 4294967295),
    '300 -as [byte]'                     : _Divergence(None, 44),
    '[byte]1 -shl -1'                    : _Divergence(0, -2147483648),

    # `$null` on the left leaves the type to the right operand, so nothing plus a Boolean is that
    # Boolean rather than the number the interpreter converts it to.
    '$null + $true'                      : _Divergence(True, 1),

    # .NET writes a Double with an upper-case, signed exponent. The interpreter writes Python's own
    # spelling, and writes a whole one out in digits.
    '[string]1E20'                       : _Divergence('1E+20', '100000000000000000000'),
    '[string]0.0000001'                  : _Divergence('1E-07', '1e-07'),
    '[string]1.5E-7'                     : _Divergence('1.5E-07', '1.5e-07'),
}

#: Where 5.1 threw and the interpreter answered anyway, with the value it answered. Each entry is an
#: expression that stops the script being folded into a constant that lets it run on.
ANSWERED_THROWS: dict[str, _Value] = {
    # A conversion whose source does not fit its target throws. The interpreter truncates to the
    # width instead, so the value it folds is one the script could never have held.
    '[byte]-1'                      : 255,
    '[byte]300'                     : 44,
    '[byte]400'                     : 144,
    '[byte](200 * 2)'               : 144,
    "[byte]'-1'"                    : 255,
    "[byte]'0x100'"                 : 0,
    '[int]2147483648'               : 2147483648,
    '[char]65536'                   : chr(65536),

    # A string 5.1's number parser refuses. The interpreter reads it with Python's rules, or leaves
    # it a string and concatenates.
    "[int]'0b1010'"                 : 10,
    "[int]'0o17'"                   : 15,
    "[int]'1_0'"                    : 10,
    "'1_0' -band 15"                : 10,
    "1 + '1e400'"                   : '11e400',
    "16 + 'file'"                   : '16file',

    # `Convert.ToInt32` reads a based string as unsigned and rejects the sign in front of it.
    "[Convert]::ToInt32('-10', 16)" : -16,

    # A Char carries none of the String methods, so asking for one is a MethodNotFound.
    '([char]65).ToUpper()'          : 'A',
    '([char]65).Substring(0)'       : 'A',

    # A repeat count of four billion is a string .NET will not allocate.
    "'ab' * 0xFFFFFFFF"             : '',
}


class _Measured(NamedTuple):
    """
    A measured row reduced to what the interpreter can be held against: the .NET type 5.1 stamped
    the value with, and that value spelled in the interpreter's currency.
    """
    carried: str
    value: _Value


def _witness(line: str) -> tuple[str, str]:
    kind, carried, rendered = line.split('\t')
    assert kind == 'OUT', line
    return carried, rendered


def _rows() -> dict[str, tuple[str, ...]]:
    """
    Every measured row that assigns one expression to `$t` and writes it twice, keyed by the
    expression. The selection is textual so that it cannot depend on the interpreter it is used to
    test, and a row that stops being selected moves a pinned count rather than dropping out.
    """
    found = {}
    for row, transcript in TYPE_TRANSCRIPTS.items():
        match = _ROW.fullmatch(row)
        if match is not None:
            found[match.group('expression')] = transcript
    return found


def _valued_rows() -> dict[str, tuple[str, ...]]:
    return {
        expression: transcript
        for expression, transcript in _rows().items()
        if not transcript[0].startswith('THROW\t')
    }


@functools.lru_cache(maxsize=1)
def _measured_rows() -> dict[str, _Measured]:
    """
    Every measured row that carries one value the currency denotes, as that value. A row whose two
    witnesses differ wrote a collection and is not a measurement of one value; a row whose type has
    no entry in `_CURRENCY` is a measurement the currency cannot hold.
    """
    found = {}
    for expression, transcript in _valued_rows().items():
        if expression in _collection_rows():
            continue
        carried, rendered = _witness(transcript[0])
        if carried in _CURRENCY:
            found[expression] = _Measured(carried, _CURRENCY[carried](rendered))
    return found


@functools.lru_cache(maxsize=1)
def _collection_rows() -> tuple[str, ...]:
    return tuple(
        expression
        for expression, transcript in _valued_rows().items()
        if len(transcript) != 2 or transcript[0] != transcript[1]
    )


@functools.lru_cache(maxsize=1)
def _types_outside_the_currency() -> dict[str, int]:
    found: dict[str, int] = {}
    for expression, transcript in _valued_rows().items():
        if expression in _collection_rows():
            continue
        carried, _ = _witness(transcript[0])
        if carried not in _CURRENCY:
            found[carried] = found.get(carried, 0) + 1
    return found


@functools.lru_cache(maxsize=1)
def _thrown_rows() -> tuple[str, ...]:
    return tuple(
        expression
        for expression, transcript in _rows().items()
        if transcript[0].startswith('THROW\t')
    )


def _assigned(expression: str):
    """
    The expression as the interpreter meets it in a measured row: the right hand side of the
    assignment the row wrote it in, rather than the same text parsed on its own.
    """
    statement = Ps1Parser(F'$t = {expression}').parse().body[0]
    assert isinstance(statement, Ps1ExpressionStatement), expression
    assignment = statement.expression
    assert isinstance(assignment, Ps1AssignmentExpression), expression
    return assignment.value


class _Outcome(NamedTuple):
    answered: bool
    value: _Value


@functools.lru_cache(maxsize=None)
def _computed(expression: str) -> _Outcome:
    """
    What the interpreter makes of a measured expression, or that it declined to say. Only the two
    exceptions the interpreter's callers catch count as declining; any other one escapes, because a
    caller that does not catch it does not get a refusal from it either.
    """
    try:
        return _Outcome(True, _Ps1Interpreter()._eval(_assigned(expression)))
    except _REFUSALS:
        return _Outcome(False, None)


def _answers() -> dict[str, _Value]:
    return {
        expression: outcome.value
        for expression in _measured_rows()
        if (outcome := _computed(expression)).answered
    }


def _diverges(measured: _Value, computed: _Value) -> bool:
    """
    Whether the interpreter answered something other than the value the host printed. The Python
    type is part of the comparison because the currency spells `1` and `1.0` and `True` and `'1'`
    apart and the deobfuscator writes each of them into the script differently.
    """
    return type(measured) is not type(computed) or measured != computed


def _divergences() -> dict[str, _Divergence]:
    found = {}
    for expression, computed in _answers().items():
        measured = _measured_rows()[expression].value
        if _diverges(measured, computed):
            found[expression] = _Divergence(measured, computed)
    return found


def _answered_throws() -> dict[str, _Value]:
    return {
        expression: outcome.value
        for expression in _thrown_rows()
        if (outcome := _computed(expression)).answered
    }


class TestPs1InterpreterComputesWhatWindowsPowerShellComputed(unittest.TestCase):

    maxDiff = None

    def test_the_value_it_computes_is_the_measured_one_wherever_no_divergence_is_recorded(self):
        for expression, computed in _answers().items():
            if expression in DIVERGENCES:
                continue
            with self.subTest(expression):
                measured = _measured_rows()[expression].value
                self.assertEqual(computed, measured)
                self.assertIs(type(computed), type(measured))

    def test_the_expressions_it_computes_a_different_value_for_are_the_ones_recorded(self):
        self.assertEqual(_divergences(), DIVERGENCES)

    def test_the_rows_held_against_the_recording_are_the_ones_recorded(self):
        measured: dict[str, list[int]] = {}
        for expression, row in _measured_rows().items():
            counted = measured.setdefault(row.carried, [0, 0])
            counted[0] += 1
            counted[1] += _computed(expression).answered
        self.assertEqual(
            {carried: _Coverage(*counted) for carried, counted in measured.items()}, COVERAGE)

    def test_the_measured_types_no_value_of_the_currency_denotes_are_the_ones_recorded(self):
        self.assertEqual(_types_outside_the_currency(), TYPES_OUTSIDE_THE_CURRENCY)

    def test_the_measured_rows_that_write_a_collection_are_as_many_as_recorded(self):
        self.assertEqual(len(_collection_rows()), COLLECTION_ROWS)


class TestPs1InterpreterRefusesWhatWindowsPowerShellRefused(unittest.TestCase):
    """
    An expression a 5.1 host answers by throwing has no value, so an interpreter that produces one
    has folded away a script that in fact stops. Refusing to answer is the safe outcome and the only
    correct one here, which is why these rows are held apart from the ones that measure a value.
    """

    maxDiff = None

    def test_the_measured_throws_it_answers_anyway_are_the_ones_recorded(self):
        self.assertEqual(_answered_throws(), ANSWERED_THROWS)

    def test_the_measured_throws_are_as_many_as_recorded(self):
        self.assertEqual(len(_thrown_rows()), THROWING_ROWS)


class TestPs1InterpreterRaisesOnlyWhatItsCallersCatch(unittest.TestCase):
    """
    `Ps1FunctionEvaluator`, `Ps1ForEachPipeline` and `evaluate_truthy` each catch
    `_Ps1InterpreterError` and `InvokeExpression` and nothing else, so any other exception leaves
    the interpreter and takes the unit down with it.
    """

    def test_no_measured_expression_raises_anything_else(self):
        escaping = []
        for expression in (*_measured_rows(), *_thrown_rows()):
            try:
                _computed(expression)
            except Exception as error:
                escaping.append(F'{expression!r}: {type(error).__name__}')
        self.assertEqual(escaping, [])
