from __future__ import annotations

import unittest

from collections import Counter
from typing import Iterator

from refinery.lib.scripts.ps1 import data

BINARY_OPERATORS = [
    '+',
    '-',
    '*',
    '/',
    '%',
    '-band',
    '-bor',
    '-bxor',
    '-shl',
    '-shr',
    '-eq',
    '-ne',
    '-lt',
    '-le',
    '-gt',
    '-ge',
]

OPERAND_TYPES = [
    'System.Byte',
    'System.SByte',
    'System.Int16',
    'System.UInt16',
    'System.Int32',
    'System.UInt32',
    'System.Int64',
    'System.UInt64',
    'System.Single',
    'System.Double',
    'System.Decimal',
    'System.String',
    'System.Char',
    'System.Boolean',
    'System.Object[]',
    'System.Void',
]

CONVERSION_TARGETS = [
    'byte',
    'sbyte',
    'int16',
    'uint16',
    'int',
    'uint32',
    'long',
    'uint64',
    'single',
    'double',
    'decimal',
    'string',
    'char',
    'bool',
    'array',
]

NUMERIC_TYPES = [
    'System.Byte',
    'System.SByte',
    'System.Int16',
    'System.UInt16',
    'System.Int32',
    'System.UInt32',
    'System.Int64',
    'System.UInt64',
    'System.Single',
    'System.Double',
    'System.Decimal',
]

COLLECTION_TYPE = 'System.Object[]'

NULL_TYPE = 'System.Void'

SCALAR_TYPES = [
    name for name in OPERAND_TYPES if name not in (COLLECTION_TYPE, NULL_TYPE)
]

Cell = tuple[tuple[str, ...], bool, bool]

GridEntry = tuple[tuple[str, ...], 'data.OperatorOutcome | None']


def _cell(outcome: data.OperatorOutcome | None) -> Cell | None:
    """
    A grid cell as one exactly comparable value: the sorted names of the types it was seen to
    produce, whether it was seen to throw, and whether it was seen to yield `$null`.
    """
    if outcome is None:
        return None
    return (
        tuple(sorted(str(name) for name in outcome.types)),
        outcome.may_throw,
        outcome.may_be_null,
    )


def _binary(operator: str, left: str, right: str) -> Cell | None:
    return _cell(data.binary_outcome(operator, left, right))


def _cast(target: str, source: str) -> Cell | None:
    return _cell(data.conversion_outcome(target, source))


def _every_binary_cell() -> Iterator[GridEntry]:
    for operator in BINARY_OPERATORS:
        for left in OPERAND_TYPES:
            for right in OPERAND_TYPES:
                yield (operator, left, right), data.binary_outcome(operator, left, right)


def _every_conversion_cell() -> Iterator[GridEntry]:
    for target in CONVERSION_TARGETS:
        for source in OPERAND_TYPES:
            yield (target, source), data.conversion_outcome(target, source)


def _every_cell() -> Iterator[GridEntry]:
    yield from _every_binary_cell()
    yield from _every_conversion_cell()


def _classify(outcome: data.OperatorOutcome | None) -> str:
    if outcome is None:
        return 'uncovered'
    if outcome.single_type is not None:
        return 'type determined'
    if outcome.may_throw and not outcome.types and not outcome.may_be_null:
        return 'always throws'
    return 'value dependent'


def _recorded_outcomes() -> set[str]:
    recorded = set()
    for by_left in data._OPERATORS['binary'].values():
        for by_right in by_left.values():
            for entries in by_right.values():
                recorded.update(entries)
    for by_source in data._OPERATORS['conversions'].values():
        for entries in by_source.values():
            recorded.update(entries)
    return recorded


def _asymmetric_pairs(operator: str, types: list[str]) -> list[tuple[str, str]]:
    """
    Every unordered pair of operand types whose cell differs from the cell of the same pair with the
    operands exchanged, listed once each in the order the type axis declares them.
    """
    return [
        (left, right)
        for index, left in enumerate(types)
        for right in types[index:]
        if _binary(operator, left, right) != _binary(operator, right, left)
    ]


class TestPs1OperatorGridCoverage(unittest.TestCase):
    """
    The grid answers for every operator and cast the capture declares, over every operand type, and
    it answers regardless of how the operator or the type is spelled at the call site.
    """

    def test_the_grid_axes_are_the_ones_the_capture_declares(self):
        self.assertEqual(list(data._OPERATORS['binary']), BINARY_OPERATORS)
        self.assertEqual(list(data._OPERATORS['witnesses']), OPERAND_TYPES)
        self.assertEqual(list(data._OPERATORS['conversions']), CONVERSION_TARGETS)

    def test_every_binary_cell_is_present(self):
        cells = list(_every_binary_cell())
        self.assertEqual([key for key, outcome in cells if outcome is None], [])
        self.assertEqual(len(cells), 4096)

    def test_every_conversion_cell_is_present(self):
        cells = list(_every_conversion_cell())
        self.assertEqual([key for key, outcome in cells if outcome is None], [])
        self.assertEqual(len(cells), 240)

    def test_an_operator_is_one_operator_in_any_casing(self):
        for operator in BINARY_OPERATORS:
            self.assertEqual(
                _binary(operator.upper(), 'System.Int32', 'System.Int32'),
                _binary(operator, 'System.Int32', 'System.Int32'),
                operator,
            )
        self.assertEqual(
            _binary('-BXOR', 'System.Int32', 'System.Int32'),
            (('System.Int32',), False, False),
        )

    def test_a_cast_target_is_one_target_in_any_casing(self):
        for target in CONVERSION_TARGETS:
            self.assertEqual(
                _cast(target.upper(), 'System.String'), _cast(target, 'System.String'), target)

    def test_a_type_is_found_under_any_spelling_of_its_name(self):
        self.assertEqual(
            _binary('-bxor', 'int', 'long'),
            _binary('-bxor', 'System.Int32', 'System.Int64'),
        )
        self.assertEqual(
            _binary('+', 'object[]', 'object[]'),
            _binary('+', 'System.Object[]', 'System.Object[]'),
        )
        self.assertEqual(_cast('int', 'string'), _cast('int', 'System.String'))


class TestPs1OperatorCellIsNotAType(unittest.TestCase):
    """
    A cell is the set of outcomes observed over several witness values per operand type, not a
    result type. Every measurement below comes out of one `(operator, left type, right type)` triple
    and shows that the triple does not determine the answer.
    """

    def test_adding_two_integers_widens_to_double_on_overflow(self):
        self.assertEqual(
            _binary('+', 'System.Int32', 'System.Int32'),
            (('System.Double', 'System.Int32'), False, False),
        )

    def test_a_string_on_the_left_of_addition_decides_the_result_type(self):
        self.assertEqual(
            _binary('+', 'System.String', 'System.Int32'),
            (('System.String',), False, False),
        )

    def test_a_string_on_the_right_of_addition_is_parsed_as_a_number_or_throws(self):
        self.assertEqual(
            _binary('+', 'System.Int32', 'System.String'),
            (('System.Double', 'System.Int32'), True, False),
        )

    def test_a_bitwise_exclusive_or_of_two_integers_is_type_determined(self):
        self.assertEqual(
            _binary('-bxor', 'System.Int32', 'System.Int32'),
            (('System.Int32',), False, False),
        )

    def test_a_long_on_either_side_of_a_bitwise_exclusive_or_makes_the_result_long(self):
        self.assertEqual(
            _binary('-bxor', 'System.Int64', 'System.Int32'),
            (('System.Int64',), False, False),
        )
        self.assertEqual(
            _binary('-bxor', 'System.Int32', 'System.Int64'),
            (('System.Int64',), False, False),
        )

    def test_multiplying_a_char_by_an_integer_produces_no_type_at_all(self):
        self.assertEqual(_binary('*', 'System.Char', 'System.Int32'), ((), True, False))

    def test_a_collection_on_the_left_of_a_comparison_filters_instead_of_answering_a_boolean(self):
        self.assertEqual(
            _binary('-ne', 'System.Object[]', 'System.Int32'),
            (('System.Boolean', 'System.Object[]'), False, False),
        )

    def test_a_collection_on_the_right_of_a_comparison_does_not_filter(self):
        self.assertEqual(
            _binary('-eq', 'System.Int32', 'System.Object[]'),
            (('System.Boolean',), False, False),
        )

    def test_the_binary_grid_is_not_one_type_per_cell(self):
        census = Counter(_classify(outcome) for _, outcome in _every_binary_cell())
        self.assertEqual(census, Counter({
            'type determined' : 2270,  # noqa
            'value dependent' : 1606,  # noqa
            'always throws'   : 220,   # noqa
        }))

    def test_the_conversion_grid_is_not_one_type_per_cell(self):
        census = Counter(_classify(outcome) for _, outcome in _every_conversion_cell())
        self.assertEqual(census, Counter({
            'type determined' : 145,  # noqa
            'value dependent' : 91,   # noqa
            'always throws'   : 4,    # noqa
        }))


class TestPs1ConversionOutcomes(unittest.TestCase):

    def test_casting_an_integer_to_char_is_a_char_or_throws(self):
        self.assertEqual(_cast('char', 'System.Int32'), (('System.Char',), True, False))

    def test_casting_a_string_to_int_is_an_integer_or_throws(self):
        self.assertEqual(_cast('int', 'System.String'), (('System.Int32',), True, False))

    def test_casting_a_collection_to_string_always_succeeds(self):
        self.assertEqual(_cast('string', 'System.Object[]'), (('System.String',), False, False))

    def test_casting_a_scalar_to_array_wraps_it(self):
        self.assertEqual(_cast('array', 'System.Int32'), (('System.Object[]',), False, False))


def _single_type(operator: str, left: str, right: str):
    outcome = data.binary_outcome(operator, left, right)
    assert outcome is not None
    return outcome.single_type


def _cast_single_type(target: str, source: str):
    outcome = data.conversion_outcome(target, source)
    assert outcome is not None
    return outcome.single_type


class TestPs1OperatorOutcomeSingleType(unittest.TestCase):
    """
    `single_type` is the one answer a caller may act on without a value domain, so it is available
    only from a cell that always produces exactly that type.
    """

    def test_single_type_is_the_sole_type_of_a_cell_that_neither_throws_nor_is_null(self):
        self.assertEqual(
            _single_type('-bxor', 'System.Int32', 'System.Int32'),
            data.resolve_type('System.Int32'),
        )

    def test_single_type_is_none_when_the_cell_may_throw(self):
        self.assertEqual(_cast('int', 'System.String'), (('System.Int32',), True, False))
        self.assertIsNone(_cast_single_type('int', 'System.String'))

    def test_single_type_is_none_when_the_cell_may_be_null(self):
        self.assertEqual(
            _binary('+', 'System.Object[]', 'System.Object[]'),
            (('System.Object[]',), False, True),
        )
        self.assertIsNone(_single_type('+', 'System.Object[]', 'System.Object[]'))

    def test_single_type_is_none_when_the_cell_has_more_than_one_type(self):
        self.assertIsNone(_single_type('+', 'System.Int32', 'System.Int32'))

    def test_no_cell_in_either_grid_reports_a_single_type_it_does_not_always_produce(self):
        wrong = []
        for key, outcome in _every_cell():
            if outcome is None:
                wrong.append(key)
                continue
            certain = (
                len(outcome.types) == 1
                and not outcome.may_throw
                and not outcome.may_be_null
            )
            expected = next(iter(outcome.types)) if certain else None
            if outcome.single_type != expected:
                wrong.append(key)
        self.assertEqual(wrong, [])


class TestPs1OperatorGridUnknowns(unittest.TestCase):
    """
    An uncovered cell answers `None`, which a caller has to read as nothing being known rather than
    as the cell producing nothing.
    """

    def test_an_operator_the_grid_does_not_cover_is_none(self):
        self.assertIsNone(data.binary_outcome('-bnot', 'System.Int32', 'System.Int32'))
        self.assertIsNone(data.binary_outcome('-and', 'System.Boolean', 'System.Boolean'))
        self.assertIsNone(data.binary_outcome('-join', 'System.Object[]', 'System.String'))

    def test_a_type_name_that_does_not_resolve_is_none(self):
        self.assertIsNone(data.binary_outcome('+', 'NotARealType', 'System.Int32'))
        self.assertIsNone(data.binary_outcome('+', 'System.Int32', 'NotARealType'))
        self.assertIsNone(data.conversion_outcome('int', 'NotARealType'))

    def test_a_type_the_grid_has_no_row_for_is_none_although_the_name_resolves(self):
        self.assertIsNotNone(data.resolve_type('System.Net.WebClient'))
        self.assertIsNone(data.binary_outcome('+', 'System.Net.WebClient', 'System.Int32'))

    def test_a_cast_target_the_grid_does_not_cover_is_none(self):
        self.assertIsNone(data.conversion_outcome('hashtable', 'System.Int32'))
        self.assertIsNone(data.conversion_outcome('void', 'System.Int32'))


class TestPs1OperatorGridLaws(unittest.TestCase):
    """
    Laws of the capture rather than of the reader: a cell that a symmetry relates to another has to
    agree with it, and a capture that measured one of the two wrongly is where that shows up.
    """

    def test_the_bitwise_operators_are_commutative_over_the_scalar_types(self):
        for operator in ('-band', '-bor', '-bxor'):
            self.assertEqual(_asymmetric_pairs(operator, SCALAR_TYPES), [], operator)

    def test_the_bitwise_operators_are_not_commutative_over_a_collection_or_null(self):
        expected = [
            ('System.UInt32', 'System.Object[]'),
            ('System.UInt32', 'System.Void'),
            ('System.Int64', 'System.Object[]'),
            ('System.Int64', 'System.Void'),
            ('System.UInt64', 'System.Object[]'),
            ('System.UInt64', 'System.Void'),
        ]
        for operator in ('-band', '-bor', '-bxor'):
            self.assertEqual(_asymmetric_pairs(operator, OPERAND_TYPES), expected, operator)
        self.assertEqual(
            _binary('-bxor', 'System.Int64', 'System.Void'),
            (('System.Int64',), False, False),
        )
        self.assertEqual(_binary('-bxor', 'System.Void', 'System.Int64'), ((), True, False))

    def test_addition_is_commutative_in_its_outcome_over_the_numeric_types(self):
        self.assertEqual(_asymmetric_pairs('+', NUMERIC_TYPES), [])

    def test_addition_is_not_commutative_once_a_string_or_a_collection_is_involved(self):
        self.assertNotEqual(
            _binary('+', 'System.String', 'System.Int32'),
            _binary('+', 'System.Int32', 'System.String'),
        )
        self.assertEqual(
            _binary('+', 'System.Object[]', 'System.Int32'),
            (('System.Int32', 'System.Object[]'), False, False),
        )
        self.assertEqual(
            _binary('+', 'System.Int32', 'System.Object[]'),
            (('System.Int32',), True, False),
        )

    def test_every_recorded_outcome_is_a_marker_or_a_type_the_type_table_resolves(self):
        recorded = _recorded_outcomes()
        self.assertEqual(recorded, {
            'throw',
            'null',
            'System.Boolean',
            'System.Byte',
            'System.Char',
            'System.Decimal',
            'System.Double',
            'System.Int16',
            'System.Int32',
            'System.Int64',
            'System.Object[]',
            'System.SByte',
            'System.Single',
            'System.String',
            'System.UInt16',
            'System.UInt32',
            'System.UInt64',
        })
        unresolved = [
            name for name in sorted(recorded - {'throw', 'null'})
            if data.resolve_type(name) is None
        ]
        self.assertEqual(unresolved, [])


class TestPs1OperatorGridProvenance(unittest.TestCase):

    def test_the_grid_was_captured_on_an_authoritative_windows_powershell_5_1_host(self):
        host = data._OPERATORS['host']
        self.assertEqual(host['ps_version'].split('.')[:2], ['5', '1'])
        self.assertEqual(host['edition'], 'Desktop')
        self.assertEqual(host['culture'], 'en-US')
        self.assertIs(host['authoritative'], True)

    def test_the_schema_version_is_the_one_the_reader_expects(self):
        self.assertEqual(data._OPERATORS['schema']['version'], data.SCHEMA_VERSION)


if __name__ == '__main__':
    unittest.main()
