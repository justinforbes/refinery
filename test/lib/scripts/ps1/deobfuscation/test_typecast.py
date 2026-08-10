from __future__ import annotations

from test.lib.scripts.ps1.deobfuscation import TestPs1

from refinery.lib.scripts.ps1.deobfuscation import (
    Ps1ConstantFolding,
    Ps1TypeCasts,
)
from refinery.lib.scripts.ps1.parser import Ps1Parser


class TestPs1CharIntFolding(TestPs1):

    def test_char_int_literal(self):
        self.assertEqual(self._deobfuscate('[Char][int]83'), '[char]83')

    def test_char_literal_regression(self):
        self.assertEqual(self._deobfuscate('[Char]65'), '[char]65')

    def test_char_int_concat(self):
        result = self._deobfuscate_iterative('([Char][int]72 + [Char][int]105)')
        self.assertEqual(result.strip(), "'Hi'")

    def test_a_char_cast_of_a_negative_number_keeps_the_cast_that_throws(self):
        self.assertEqual(self._deobfuscate('$x = [Char][int](-65)'), '$x = [Char]-65')

    def test_int_identity_cast_stripped(self):
        result = self._deobfuscate('[int]42')
        self.assertEqual(result.strip(), '42')

    def test_a_string_of_digits_cast_to_int_is_the_number_it_spells(self):
        self.assertEqual(self._deobfuscate("$x = [int]'27'"), '$x = 27')

    def test_a_cast_of_an_indexed_string_element_is_the_number_that_element_spells(self):
        self.assertEqual(
            self._deobfuscate("$x = [int](('10', '20', '30')[2])"),
            '$x = 30',
        )

    def test_array_literal_index_scalar(self):
        result = self._deobfuscate("$x = ('hello', 'world', 'foo')[2]")
        self.assertIn('foo', result)
        self.assertNotIn('hello', result)

    def test_array_literal_index_nested(self):
        result = self._deobfuscate(
            "$x = ('a', 'b', 'c', 'd')[[int](('3', '1', '0')[2])]")
        self.assertNotIn("'b'", result)
        self.assertNotIn("'c'", result)
        self.assertNotIn("'d'", result)

    def test_char_int_multi_concat(self):
        result = self._deobfuscate_iterative(
            '([Char][int]83 + [Char][int]116 + [Char][int]111 + [Char][int]112)')
        self.assertEqual(result.strip(), "'Stop'")

    def test_char_int_partial_with_variable(self):
        result = self._deobfuscate_iterative(
            '([Char][int]83 + [Char][int]$x + [Char][int]112)')
        # $x is undefined -> $null -> [int]0 -> [char]0, which is a NUL character (not empty), so
        # the folded result is the three-character string "S\0p".
        self.assertIn("S`0p", result)


class TestPs1TypeCastExtra(TestPs1):

    def test_typecast_char(self):
        self.assertEqual(self._deobfuscate('[char]120'), '[char]120')

    def test_typecast_char_hex(self):
        self.assertEqual(self._deobfuscate('[char]0x41'), '[char]65')

    def test_a_string_cast_of_a_string_literal_is_that_literal(self):
        self.assertEqual(self._apply('$x = [string]"foo"', Ps1TypeCasts), "$x = 'foo'")

    def test_typecast_char_array(self):
        data = '[char[]](72,101,108,108,111)'
        result = self._deobfuscate(data)
        self.assertIn('Hello', result)

    def test_as_char_cast(self):
        self.assertEqual(self._deobfuscate('(45 -As [Char])'), '([char]45)')

    def test_a_type_cast_of_a_type_name_is_the_type_literal_naming_it(self):
        self.assertEqual(self._apply("$x = [Type]'Convert'", Ps1TypeCasts), '$x = [Convert]')

    def test_a_string_cast_inside_a_method_argument_is_the_string_it_produces(self):
        self.assertEqual(
            self._apply('''$snug.replAce(("M0I"),[strIng]"'")''', Ps1TypeCasts),
            '''$snug.replAce(("M0I"), "'")''',
        )

    def test_char_cast_in_bmp_folds(self):
        self.assertEqual('[char]65', self._apply('[char]65', Ps1TypeCasts))

    def test_char_zero_is_nul_character(self):
        # [char]0 is a NUL character, not an empty string: 'a'+[char]0+'b' is the 3-character
        # string "a\0b". Verified against PowerShell (.Length == 3, char codes 97, 0, 98); the NUL
        # is simply not rendered on the console, and is emitted as the `0 escape.
        result = self._apply("'a' + [char]0 + 'b'", Ps1TypeCasts, Ps1ConstantFolding)
        self.assertEqual(result, '"a`0b"')


class TestPs1ACastToATypeWithNoLiteralIsWrittenAsACast(TestPs1):
    """
    5.1 spells a literal for Int32, Int64, Decimal, Double, String and Boolean, and for none of the
    six integer widths: `[byte]5` is a Byte where `5` is an Int32, so a fold that dropped the
    bracket would change the type at the point the script asked for it. A cast to one of those
    targets is therefore written back as a cast of the value it produces, or left as it stands.
    """

    def test_a_hex_string_cast_to_uint16_keeps_the_uint16_cast(self):
        self.assertEqual(self._apply("$x = [uint16]'0xFFFF'", Ps1TypeCasts), '$x = [uint16]65535')

    def test_a_boolean_cast_to_int16_keeps_the_int16_cast(self):
        self.assertEqual(self._apply('$x = [int16]$true', Ps1TypeCasts), '$x = [int16]1')

    def test_a_byte_cast_of_a_numeral_is_left_standing(self):
        self._assertUnchanged('$x = [byte]5', Ps1TypeCasts)

    def test_an_sbyte_cast_of_a_negative_numeral_is_left_standing(self):
        self._assertUnchanged('$x = [sbyte]-5', Ps1TypeCasts)

    def test_an_int16_cast_of_a_numeral_is_left_standing(self):
        self._assertUnchanged('$x = [int16]7', Ps1TypeCasts)

    def test_a_uint32_cast_of_a_numeral_is_left_standing(self):
        self._assertUnchanged('$x = [uint32]7', Ps1TypeCasts)

    def test_a_uint64_cast_of_a_numeral_too_wide_for_int64_is_left_standing(self):
        self._assertUnchanged('$x = [uint64]18446744073709551615', Ps1TypeCasts)


class TestPs1ACastToATypeWithALiteralIsWrittenAsOne(TestPs1):
    """
    The other half. Where the language does spell a literal of the target's type, the cast is
    written away and what is left carries the same type: `[long]5` and `5L` are both the Int64 5,
    `[decimal]5` and `5d` the Decimal 5, `[double]5` and `5.0` the Double 5.
    """

    def test_an_int_cast_of_an_int32_numeral_is_that_numeral(self):
        self.assertEqual(self._apply('$x = [int]5', Ps1TypeCasts), '$x = 5')

    def test_a_long_cast_is_written_with_the_long_suffix(self):
        self.assertEqual(self._apply('$x = [long]5', Ps1TypeCasts), '$x = 5L')

    def test_a_decimal_cast_is_written_with_the_decimal_suffix(self):
        self.assertEqual(self._apply('$x = [decimal]5', Ps1TypeCasts), '$x = 5d')

    def test_a_double_cast_of_an_integer_numeral_is_written_as_a_real(self):
        self.assertEqual(self._apply('$x = [double]5', Ps1TypeCasts), '$x = 5.0')

    def test_a_double_cast_of_a_decimal_numeral_is_written_as_a_real(self):
        self.assertEqual(self._apply('$x = [double]1.5d', Ps1TypeCasts), '$x = 1.5')

    def test_a_string_cast_of_a_numeral_is_written_as_a_string_literal(self):
        self.assertEqual(self._apply('$x = [string]5', Ps1TypeCasts), "$x = '5'")

    def test_a_string_cast_of_a_decimal_numeral_keeps_the_digits_it_was_written_with(self):
        self.assertEqual(self._apply('$x = [string]10d', Ps1TypeCasts), "$x = '10'")

    def test_a_string_cast_of_a_boolean_is_the_word_it_renders_as(self):
        self.assertEqual(self._apply('$x = [string]$true', Ps1TypeCasts), "$x = 'True'")

    def test_a_string_cast_of_the_absent_value_is_the_empty_string(self):
        self.assertEqual(self._apply('$x = [string]$null', Ps1TypeCasts), "$x = ''")

    def test_a_bool_cast_of_zero_is_the_false_literal(self):
        self.assertEqual(self._apply('$x = [bool]0', Ps1TypeCasts), '$x = $False')

    def test_a_bool_cast_of_one_is_the_true_literal(self):
        self.assertEqual(self._apply('$x = [bool]1', Ps1TypeCasts), '$x = $True')

    def test_an_int_cast_of_a_boolean_is_one(self):
        self.assertEqual(self._apply('$x = [int]$true', Ps1TypeCasts), '$x = 1')

    def test_an_int_cast_of_the_absent_value_is_zero(self):
        self.assertEqual(self._apply('$x = [int]$null', Ps1TypeCasts), '$x = 0')


class TestPs1ACastToAnIntegerRoundsHalfToEven(TestPs1):
    """
    A real reaches an integer target by rounding rather than by truncation, and a half goes to the
    even neighbour: measured, `[int]1.5` and `[int]2.5` are both 2, `[int]1.4` is 1 and `[int]-1.5`
    is -2.
    """

    def test_a_half_below_an_odd_number_rounds_up_to_the_even_one(self):
        self.assertEqual(self._apply('$x = [int]1.5', Ps1TypeCasts), '$x = 2')

    def test_a_half_above_an_even_number_rounds_down_to_it(self):
        self.assertEqual(self._apply('$x = [int]2.5', Ps1TypeCasts), '$x = 2')

    def test_less_than_a_half_rounds_down(self):
        self.assertEqual(self._apply('$x = [int]1.4', Ps1TypeCasts), '$x = 1')

    def test_a_negative_half_rounds_to_the_even_neighbour(self):
        self.assertEqual(self._apply('$x = [int]-1.5', Ps1TypeCasts), '$x = -2')

    def test_the_rounded_value_carries_the_type_the_target_names(self):
        self.assertEqual(self._apply('$x = [long]1.5', Ps1TypeCasts), '$x = 2L')

    def test_a_decimal_numeral_reaches_an_integer_target_whole(self):
        self.assertEqual(self._apply('$x = [int]10d', Ps1TypeCasts), '$x = 10')


class TestPs1ACastThatStopsTheScriptIsLeftStanding(TestPs1):
    """
    A cast whose conversion cannot be made throws, and the throw is part of what the script does:
    writing the value the conversion would have had deletes the point at which the script stopped.
    Measured, `[byte]300`, `[byte]-1`, `[int]2147483648`, `[char]65536` and `[char]-1` each throw
    rather than wrapping, and so does every String the host's own parser rejects.
    """

    def test_a_byte_cast_above_the_byte_range_is_left_standing(self):
        self._assertUnchanged('$x = [byte]300', Ps1TypeCasts)

    def test_a_byte_cast_of_a_negative_numeral_is_left_standing(self):
        self._assertUnchanged('$x = [byte]-1', Ps1TypeCasts)

    def test_an_int_cast_above_the_int32_range_is_left_standing(self):
        self._assertUnchanged('$x = [int]2147483648', Ps1TypeCasts)

    def test_a_char_cast_above_the_utf16_code_unit_range_is_left_standing(self):
        self._assertUnchanged('$x = [char]65536', Ps1TypeCasts)

    def test_a_char_cast_of_a_negative_numeral_is_left_standing(self):
        self._assertUnchanged('$x = [char]-1', Ps1TypeCasts)

    def test_a_byte_cast_of_a_hex_string_one_digit_wider_than_the_target_is_left_standing(self):
        self._assertUnchanged("$x = [byte]'0x100'", Ps1TypeCasts)

    def test_a_byte_cast_of_a_negative_decimal_string_is_left_standing(self):
        # A decimal String keeps its sign where a hexadecimal one is a bit pattern, so `[byte]'-1'`
        # throws in the same script that reads `[byte]'0x80'` as 128.
        self._assertUnchanged("$x = [byte]'-1'", Ps1TypeCasts)


class TestPs1ACastOfAnOperandTheSourcePinsNothingForIsLeftStanding(TestPs1):
    """
    The pass answers a cast from what the operand *is*, so an operand the source settles nothing
    about leaves the cast where it stands whether or not the target has a literal.
    """

    def test_a_cast_of_a_variable_is_left_standing(self):
        self._assertUnchanged('$x = [int]$y', Ps1TypeCasts)

    def test_a_cast_of_a_variable_to_a_width_is_left_standing(self):
        self._assertUnchanged('$x = [byte]$y', Ps1TypeCasts)


class TestPs1AnAsConversionIsNotACast(TestPs1):
    """
    `-as` answers `$null` where the conversion cannot be made and the cast of the same operand
    throws: measured, `300 -as [byte]` is `$null` where `[byte]300` stops the script, and
    `'abc' -as [int]` is `$null` where `[int]'abc'` stops it. Writing either as the other changes
    what the script does.

    The two agree only where the conversion is settled, which is the one case an `-as` is folded
    in: `5 -as [long]` and `[long]5` are both the measured Int64 5, so what the fold has to keep
    for `5 -as [byte]` is the Byte that `[byte]5` measures as and for `'5' -as [int]` the Int32
    that `[int]'5'` does.
    """

    def test_an_as_conversion_whose_value_does_not_fit_is_left_standing(self):
        self._assertUnchanged('$x = 300 -as [byte]', Ps1TypeCasts)

    def test_the_cast_whose_value_does_not_fit_is_left_standing(self):
        self._assertUnchanged('$x = [byte]300', Ps1TypeCasts)

    def test_an_as_conversion_of_a_string_that_names_no_number_is_left_standing(self):
        self._assertUnchanged("$x = 'abc' -as [int]", Ps1TypeCasts)

    def test_the_cast_of_a_string_that_names_no_number_is_left_standing(self):
        self._assertUnchanged("$x = [int]'abc'", Ps1TypeCasts)

    def test_an_as_conversion_above_the_int32_range_is_left_standing(self):
        self._assertUnchanged('$x = 2147483648 -as [int]', Ps1TypeCasts)

    def test_a_settled_as_conversion_is_written_as_the_value_it_produces(self):
        self.assertEqual(self._apply('$x = 5 -as [long]', Ps1TypeCasts), '$x = 5L')

    def test_a_settled_as_conversion_to_a_width_is_written_as_the_cast_that_spells_it(self):
        self.assertEqual(self._apply('$x = 5 -as [byte]', Ps1TypeCasts), '$x = [byte]5')

    def test_a_settled_as_conversion_of_a_string_is_written_as_the_number_it_names(self):
        self.assertEqual(self._apply("$x = '5' -as [int]", Ps1TypeCasts), '$x = 5')


class TestPs1ACastOfAStringIsReadByTheHostsParserAndNotPythons(TestPs1):
    """
    A String operand is parsed rather than converted, and the parser is .NET's. Measured, 5.1 reads
    `'007'` as seven, `''` as zero, `'5.'` as five and `'.5'` as zero, rounds `'7.5'` to eight, and
    reads a hexadecimal String at the *target's* width, so `'0x80'` is 128 under `[byte]` and -128
    under `[sbyte]`. Python's own parser answers none of those the same way: `int(text, 0)` rejects
    `'007'`, `''`, `'5.'`, `'.5'` and `'7.5'` outright, reads `'0x80'` as 128 whatever the target
    is, and accepts `'1_0'`, `'0b1010'` and `'0o17'`, which 5.1 throws for.
    """

    def test_a_string_of_digits_with_leading_zeroes_is_a_decimal_number(self):
        self.assertEqual(self._apply("$x = [int]'007'", Ps1TypeCasts), '$x = 7')

    def test_the_empty_string_is_zero(self):
        self.assertEqual(self._apply("$x = [int]''", Ps1TypeCasts), '$x = 0')

    def test_a_string_of_nothing_but_whitespace_is_left_standing(self):
        # `[int]''` is 0 and `[int]'   '` throws, so a text that is only whitespace is not the
        # empty one with the whitespace stripped off it.
        self._assertUnchanged("$x = [int]'   '", Ps1TypeCasts)

    def test_whitespace_around_a_number_is_stripped(self):
        self.assertEqual(self._apply("$x = [int]' 5 '", Ps1TypeCasts), '$x = 5')

    def test_a_leading_plus_is_a_sign(self):
        self.assertEqual(self._apply("$x = [int]'+7'", Ps1TypeCasts), '$x = 7')

    def test_a_trailing_point_is_a_whole_number(self):
        self.assertEqual(self._apply("$x = [int]'5.'", Ps1TypeCasts), '$x = 5')

    def test_a_leading_point_is_a_fraction_that_rounds_to_zero(self):
        self.assertEqual(self._apply("$x = [int]'.5'", Ps1TypeCasts), '$x = 0')

    def test_a_fractional_string_rounds_half_to_even(self):
        self.assertEqual(self._apply("$x = [int]'7.5'", Ps1TypeCasts), '$x = 8')
        self.assertEqual(self._apply("$x = [int]'2.5'", Ps1TypeCasts), '$x = 2')

    def test_a_hexadecimal_string_is_read_at_the_width_of_the_target(self):
        self.assertEqual(self._apply("$x = [byte]'0x80'", Ps1TypeCasts), '$x = [byte]128')
        self.assertEqual(self._apply("$x = [sbyte]'0x80'", Ps1TypeCasts), '$x = [sbyte]-128')

    def test_a_hexadecimal_string_filling_the_int32_width_is_the_negative_number_it_denotes(self):
        self.assertEqual(self._apply("$x = [int]'0xFFFFFFFF'", Ps1TypeCasts), '$x = -1')

    def test_a_digit_separator_is_no_numeral_and_the_cast_is_left_standing(self):
        self._assertUnchanged("$x = [int]'1_0'", Ps1TypeCasts)

    def test_a_binary_prefix_is_no_numeral_and_the_cast_is_left_standing(self):
        self._assertUnchanged("$x = [int]'0b1010'", Ps1TypeCasts)

    def test_an_octal_prefix_is_no_numeral_and_the_cast_is_left_standing(self):
        self._assertUnchanged("$x = [int]'0o17'", Ps1TypeCasts)

    def test_a_multiplier_suffix_is_no_numeral_and_the_cast_is_left_standing(self):
        self._assertUnchanged("$x = [int]'1kb'", Ps1TypeCasts)

    def test_a_string_is_true_by_its_length_and_never_by_its_text(self):
        self.assertEqual(self._apply("$x = [bool]'0'", Ps1TypeCasts), '$x = $True')
        self.assertEqual(self._apply("$x = [bool]''", Ps1TypeCasts), '$x = $False')


class TestPs1ACastAlreadySpellingItsOwnValueIsNotRewritten(TestPs1):
    """
    The deobfuscation loop runs a pass until none of them reports a change, so a pass that answers
    a node with a tree spelling the same program never converges. Every value whose type has no
    literal is written back as the cast it was read from, which is exactly the shape that can be
    answered with itself.
    """

    def _reports_a_change(self, source: str) -> bool:
        transform = Ps1TypeCasts()
        transform.visit(Ps1Parser(source).parse())
        return transform.changed

    def test_a_byte_cast_of_the_value_it_already_spells_reports_no_change(self):
        self.assertFalse(self._reports_a_change('$x = [byte]128'))

    def test_a_uint16_cast_of_the_value_it_already_spells_reports_no_change(self):
        self.assertFalse(self._reports_a_change('$x = [uint16]65535'))

    def test_an_sbyte_cast_of_the_negative_value_it_already_spells_reports_no_change(self):
        self.assertFalse(self._reports_a_change('$x = [sbyte]-128'))

    def test_what_the_pass_writes_for_a_hex_string_is_a_fixed_point_of_the_pass(self):
        written = self._apply("$x = [sbyte]'0x80'", Ps1TypeCasts)
        self.assertEqual(written, '$x = [sbyte]-128')
        self.assertFalse(self._reports_a_change(written))

    def test_a_char_cast_of_the_value_it_already_spells_reports_no_change(self):
        self.assertFalse(self._reports_a_change('$x = [char]65'))

    def test_what_the_pass_writes_for_a_char_of_a_string_is_a_fixed_point_of_the_pass(self):
        written = self._apply("$x = [char]'A'", Ps1TypeCasts)
        self.assertEqual(written, '$x = [char]65')
        self.assertFalse(self._reports_a_change(written))


class TestPs1ACharIsCarriedAsACharAndWrittenAsACast(TestPs1):
    """
    Measured, `[char]65` is a `System.Char` and `'A'` a `System.String`: the two answer `-is [char]`
    differently, and `[char]65 + 1` is the String `A1` where `1 + [char]65` is the Int32 66. 5.1
    spells no literal for a Char, so the cast is its spelling and a cast that already is one is
    written back as one rather than as the one-character string it prints as.
    """

    def test_a_char_cast_of_a_one_character_string_is_written_as_the_code_point(self):
        self.assertEqual(self._apply("$x = [char]'A'", Ps1TypeCasts), '$x = [char]65')

    def test_a_char_cast_of_a_numeral_is_the_cast_it_already_is(self):
        self.assertEqual(self._apply('$x = [char]65', Ps1TypeCasts), '$x = [char]65')

    def test_a_char_cast_of_a_string_of_two_characters_is_left_standing(self):
        self._assertUnchanged("$x = [char]'AB'", Ps1TypeCasts)

    def test_a_char_cast_of_the_empty_string_is_left_standing(self):
        self._assertUnchanged("$x = [char]''", Ps1TypeCasts)

    def test_a_string_cast_of_a_char_is_the_one_character_string(self):
        self.assertEqual(self._apply('$x = [string][char]65', Ps1TypeCasts), "$x = 'A'")

    def test_an_int_cast_of_a_char_is_the_code_point_of_its_character(self):
        self.assertEqual(self._deobfuscate("$x = [int][char]'A'"), '$x = 65')
