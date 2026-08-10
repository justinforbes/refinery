from __future__ import annotations

import inspect
import unittest

from test.lib.scripts.ps1.deobfuscation import TestPs1

from refinery.lib.scripts.ps1.deobfuscation import Ps1ConstantFolding, Ps1ConstantInlining


class TestPS1StringConcatenations(TestPs1):

    def test_concat_basic(self):
        data = "'foo' + 'bar'"
        result = self._deobfuscate(data)
        self.assertIn('foobar', result)

    def test_concat_double_quotes(self):
        data = '"hel" + "lo"'
        result = self._deobfuscate(data)
        self.assertIn('hello', result)

    def test_uneven(self):
        result = self._deobfuscate("'T'+'b'+'c'")
        self.assertIn('Tbc', result)

    def test_concatenation(self):
        result = self._deobfuscate('"bla" + "foo" +"bar"')
        self.assertIn('blafoobar', result)

    def test_uneven_special_chars(self):
        result = self._deobfuscate('$t = "bla " + "\\foo" + "bar baz"')
        self.assertIn('bla \\foobar baz', result)

    def test_not_inside_string(self):
        result = self._deobfuscate('''$t="'bla ' + '\\foo'"; $t = $t + 'bar' + "baz"''')
        self.assertIn("'bla ' + '\\foo'", result)
        self.assertIn('barbaz', result)

    def test_variable_substitution(self):
        result = self._deobfuscate('''$y = "$y"+'$z'; Write-Output $y''')
        self.assertIn('$z', result)

    def test_split_format_string(self):
        result = self._deobfuscate(R'''"{0}$SEP{1}"-f 'Hello',"World"''')
        self.assertIn('Hello', result)
        self.assertIn('World', result)

    def test_invalid_format(self):
        result = self._deobfuscate(R'''"{0}{2}{1}"-f 'Hello',"World"''')
        self.assertIn('Hello', result)

    def test_all_single_quotes(self):
        result = self._deobfuscate(R"""'{0}{2}{1}'-f 'signa','ures','t'""")
        self.assertIn('signatures', result)

    def test_mixed_quotes(self):
        result = self._deobfuscate(R'''"{0}{2}{1}"-f 'signa','ures',"t"''')
        self.assertIn('signatures', result)

    def test_format_string_with_chars(self):
        result = self._deobfuscate('("{0}na{2}{1}"-f \'sig\',\'ures\',\'t\')')
        self.assertIn('signatures', result)

    def test_multiple_occurrences(self):
        result = self._deobfuscate(
            '"{10}{1}{0}{5}{9}{7}{8}{7}{3}{6}{2}{7}{4}{4}{10}{5}{1}"'
            "-f'v','n','r','x','s','o','p','e','-','k','i'"
        )
        self.assertIn('invoke-expression', result)

    def test_format_string_escaped_braces(self):
        result = self._deobfuscate('"{0} {{literal}}" -f "test"')
        self.assertIn('test {literal}', result)

    def test_format_string_only_escaped_braces(self):
        result = self._deobfuscate('"{{hello}}" -f "unused"')
        self.assertEqual(result, "'{hello}'")

    def test_string_concat_static_method(self):
        result = self._deobfuscate("[String]::Concat('a', 'b', 'c')")
        self.assertEqual(result, "'abc'")

    def test_join_single_string(self):
        result = self._deobfuscate("-Join @('hello')")
        self.assertEqual(result, "'hello'")

    def test_string_join_variadic_empty_separator(self):
        data = "[String]::Join('', 'Nam', 'e')"
        result = self._deobfuscate(data)
        self.assertIn("'Name'", result)

    def test_string_join_variadic_with_separator(self):
        data = "[String]::Join('-', 'a', 'b', 'c')"
        result = self._deobfuscate(data)
        self.assertIn("'a-b-c'", result)

    def test_replace_after_concat(self):
        result = self._deobfuscate("$([String]::Concat('h_llo')).Replace('_', 'e')")
        self.assertEqual(result, "'hello'")

    def test_string_join_static_with_separator(self):
        data = "[String]::Join(',', @('a','b','c'))"
        result = self._deobfuscate(data)
        self.assertIn('a,b,c', result)


class TestPS1StringReplace(TestPs1):

    def test_real_world_01(self):
        data = '''-RepLaCe"UVL",""""-CrePLAcE "MQo","``" -RepLaCe ("0"+"N"+"R"),"'"-CrePLAcE'eV5',"`$"-CrePLAcE  '31V',"|")'''
        result = self._deobfuscate(data, remove_junk=False)
        self.assertIn('0NR', result)

    def test_real_world_02(self):
        result = self._deobfuscate(
            '''"UVL0NR"-RepLaCe"UVL",""""-RepLaCe "0NR","'"-CrePLAcE  '31V',"|"))''', remove_junk=False)
        self.assertIn("'", result)

    def test_string_replace_method(self):
        data = '"haystack".Replace("hay","needle")'
        result = self._deobfuscate(data)
        self.assertIn('needlestack', result)

    def test_string_replace_operator(self):
        data = '"Hello World" -replace "World","Earth"'
        result = self._deobfuscate(data)
        self.assertIn('Hello Earth', result)

    def test_chained_replace_operator(self):
        data = '"ABCDEF" -replace \'AB\',\'ab\' -replace \'CD\',\'cd\' -replace \'EF\',\'ef\''
        result = self._deobfuscate(data)
        self.assertIn('abcdef', result)

    def test_replace_with_backslash_replacement(self):
        data = "'MarkXPath' -Replace 'X', '\\'"
        result = self._deobfuscate(data)
        self.assertIn('Mark\\Path', result)
        self.assertNotIn('-Replace', result)

    def test_chained_replace_on_herestring(self):
        data = (
            "(@'\n"
            "aXb cYd\n"
            "'@ -Replace 'X', '1' -Replace 'Y', '2')"
        )
        result = self._deobfuscate_iterative(data)
        self.assertIn('a1b', result)
        self.assertIn('c2d', result)
        self.assertNotIn('-Replace', result)

    def test_trivial(self):
        result = self._deobfuscate('''"Hello World".replace('l', "FOO")''')
        self.assertIn('HeFOOFOOo WorFOOd', result)

    def test_variable_substitution_01(self):
        result = self._deobfuscate(
            '''Write-Output "The $product costs `$100 for the average person." -replace '$', "\u20ac";''')
        self.assertIn('\u20ac', result)

    def test_variable_substitution_02(self):
        result = self._deobfuscate(
            '''Write-Output "The $product costs `$100 for the average person." -replace '$', "$currency";''')
        self.assertIn('currency', result)


class TestPs1ComparisonFolding(TestPs1):

    def test_eq_true(self):
        result = self._deobfuscate('(3 -eq 3)')
        self.assertIn('$True', result)

    def test_eq_false(self):
        result = self._deobfuscate('(3 -eq 4)')
        self.assertIn('$False', result)

    def test_lt_true(self):
        result = self._deobfuscate('(1 -lt 2)')
        self.assertIn('$True', result)

    def test_ge_false(self):
        result = self._deobfuscate('(-296 -ge 44)')
        self.assertIn('$False', result)

    def test_le_negative(self):
        result = self._deobfuscate('(81 -le -230)')
        self.assertIn('$False', result)

    def test_parenthesized_operands(self):
        result = self._deobfuscate('((5) -gt (3))')
        self.assertIn('$True', result)

    def test_non_constant_unchanged(self):
        result = self._deobfuscate('$x = $env:V\n($x -eq 3)', remove_junk=False)
        self.assertIn('-Eq', result)


class TestPs1StringEqualityFolding(TestPs1):

    def test_eq_false(self):
        self.assertEqual(self._apply("'m' -eq 'z'", Ps1ConstantFolding), '$False')

    def test_eq_true(self):
        self.assertEqual(self._apply("'abc' -eq 'abc'", Ps1ConstantFolding), '$True')

    def test_ne_true(self):
        self.assertEqual(self._apply("'a' -ne 'b'", Ps1ConstantFolding), '$True')

    def test_ne_false(self):
        self.assertEqual(self._apply("'a' -ne 'a'", Ps1ConstantFolding), '$False')

    def test_eq_case_insensitive_by_default(self):
        self.assertEqual(self._apply("'M' -eq 'm'", Ps1ConstantFolding), '$True')

    def test_ieq_case_insensitive(self):
        self.assertEqual(self._apply("'M' -ieq 'm'", Ps1ConstantFolding), '$True')

    def test_ceq_case_sensitive(self):
        self.assertEqual(self._apply("'M' -ceq 'm'", Ps1ConstantFolding), '$False')

    def test_cne_case_sensitive(self):
        self.assertEqual(self._apply("'M' -cne 'm'", Ps1ConstantFolding), '$True')

    def test_ordering_not_folded(self):
        # Only equality is folded for strings; culture-dependent ordering is left untouched.
        self.assertEqual(self._apply("'a' -lt 'b'", Ps1ConstantFolding), "'a' -lt 'b'")

    def test_sharp_s_not_equal_ss(self):
        self.assertEqual(self._apply("'ß'  -eq 'SS'", Ps1ConstantFolding), '$False')
        self.assertEqual(self._apply("'ß' -ieq 'ss'", Ps1ConstantFolding), '$False')


class TestPs1LogicalFolding(TestPs1):

    def test_and_true_false(self):
        self.assertEqual(self._apply('$true -and $false', Ps1ConstantFolding), '$False')

    def test_or_true_false(self):
        self.assertEqual(self._apply('$true -or $false', Ps1ConstantFolding), '$True')

    def test_xor_true_true(self):
        self.assertEqual(self._apply('$true -xor $true', Ps1ConstantFolding), '$False')

    def test_xor_true_false(self):
        self.assertEqual(self._apply('$true -xor $false', Ps1ConstantFolding), '$True')

    def test_null_operand_is_false(self):
        self.assertEqual(self._apply('$null -and $true', Ps1ConstantFolding), '$False')

    def test_unknown_operand_not_folded(self):
        # With one operand unresolved, the result is not constant and must be preserved.
        self.assertEqual(self._apply('$x -and $false', Ps1ConstantFolding), '$x -and $false')


class TestPs1RegexFolding(TestPs1):

    def test_regex_matches_simple(self):
        result = self._deobfuscate("[Regex]::Matches('abc123def', '\\d+').Groups.Captures.Groups.Value")
        self.assertIn("'123'", result)

    def test_regex_without_value_access_not_inlined(self):
        result = self._deobfuscate(
            """
            [Regex]::Matches('REFINERY', '[RF]') | Write-Output;
            """
        )
        self.assertIn('REFINERY', result)

    def test_regex_matches_dot_righttoleft(self):
        result = self._deobfuscate_iterative(
            "(-Join [Regex]::Matches('dlroW olleH', '.', 'RightToLeft'))")
        self.assertIn('Hello World', result)

    def test_regex_matches_dot_righttoleft_mapped(self):
        result = self._deobfuscate_iterative(
            "-Join([Regex]::Matches('dlroW olleH', '.', 'RightToLeft')|%{$_.Groups.Value})")
        self.assertIn('Hello World', result)

    def test_regex_matches_integer_option(self):
        result = self._deobfuscate_iterative(
            "(-Join [Regex]::Matches('olleH', '.', 64))")
        self.assertIn('Hello', result)

    def test_regression_props_of_constant_are_null(self):
        self.assertEqual(self._deobfuscate("'a'.Value"), '$Null')
        self.assertEqual(self._deobfuscate("'a'.fLaBu"), '$Null')

    def test_regex_matches_combined_options(self):
        result = self._deobfuscate(
            "[Regex]::Matches('aAbBcC', '[a-c]', 'IgnoreCase, RightToLeft').Value")
        self.assertIn("'C'", result)
        self.assertIn("'c'", result)
        self.assertNotIn('Value', result)

    def test_regex_match_single(self):
        result = self._deobfuscate("[Regex]::Match('abc123def456', '\\d+').Value")
        self.assertIn("'123'", result)
        self.assertNotIn('456', result)

    def test_regex_match_no_match(self):
        result = self._deobfuscate("[Regex]::Match('hello', '\\d+')|%{$_.Value}")
        self.assertIn("''", result)
        self.assertNotIn("Value", result)

    def test_regex_replace_static(self):
        result = self._deobfuscate("[Regex]::Replace('Hello World', 'World', 'Earth')")
        self.assertIn('Hello Earth', result)

    def test_regex_replace_with_pattern(self):
        result = self._deobfuscate("[Regex]::Replace('abc123def456', '\\d+', 'X')")
        self.assertIn('abcXdefX', result)

    def test_regex_matches_fully_qualified_type(self):
        result = self._deobfuscate(
            "[Text.RegularExpressions.Regex]::Matches('abc', '.')|%{$_.Value}")
        self.assertIn("'a'", result)
        self.assertIn("'b'", result)
        self.assertIn("'c'", result)
        self.assertNotIn("Value", result)

    def test_regex_join_chain(self):
        result = self._deobfuscate_iterative(
            "-Join [Regex]::Matches('!o!l!l!e!H', '[^!]', 'RightToLeft')")
        self.assertIn('Hello', result)


class TestPs1SubstringFolding(TestPs1):

    def test_substring_one_arg(self):
        result = self._deobfuscate("'Hello World'.Substring(6)")
        self.assertIn("'World'", result)
        self.assertNotIn('Substring', result)

    def test_substring_two_args(self):
        result = self._deobfuscate("'Hello World'.Substring(0, 5)")
        self.assertIn("'Hello'", result)
        self.assertNotIn('Substring', result)

    def test_substring_out_of_bounds(self):
        result = self._deobfuscate("'abc'.Substring(0, 10)")
        self.assertIn('Substring', result)


class TestPs1StringInsertRemoveFolding(TestPs1):

    def test_string_insert(self):
        result = self._deobfuscate("'hello'.Insert(0, 'X')")
        self.assertEqual(result, "'Xhello'")

    def test_string_insert_middle(self):
        result = self._deobfuscate("'hello'.Insert(2, 'XY')")
        self.assertEqual(result, "'heXYllo'")

    def test_string_insert_end(self):
        result = self._deobfuscate("'hello'.Insert(5, '!')")
        self.assertEqual(result, "'hello!'")

    def test_string_remove_one_arg(self):
        result = self._deobfuscate("'hello'.Remove(3)")
        self.assertEqual(result, "'hel'")

    def test_string_remove_two_args(self):
        result = self._deobfuscate("'hello'.Remove(1, 2)")
        self.assertEqual(result, "'hlo'")

    def test_string_insert_remove_chain(self):
        result = self._deobfuscate("'abcdef'.Remove(2, 1).Insert(0, 'X')")
        self.assertEqual(result, "'Xabdef'")


class TestPs1LengthFolding(TestPs1):

    def test_string_length(self):
        self.assertEqual(self._deobfuscate("'Hello'.Length"), '5')

    def test_array_length(self):
        self.assertEqual(self._deobfuscate('@(1, 2, 3).Length'), '3')

    def test_array_count(self):
        self.assertEqual(self._deobfuscate('@(1, 2, 3).Count'), '3')

    def test_string_length_via_variable(self):
        result = self._deobfuscate("$x = 'Hello'; Write-Host $x.Length")
        self.assertIn('5', result)


class TestPs1ControlCharStringLiteral(TestPs1):

    def test_format_newline_only_produces_backtick_escape(self):
        code = '"{0}`n{1}" -f "hello","world"'
        result = self._deobfuscate(code)
        self.assertIn('`n', result)
        self.assertIn('hello', result)
        self.assertIn('world', result)

    def test_tab_in_format_produces_backtick_escape(self):
        code = '"{0}`t{1}" -f "a","b"'
        result = self._deobfuscate(code)
        self.assertIn('`t', result)
        self.assertNotIn('\t', result)

    def test_mixed_newline_and_control_chars_produces_dq_string(self):
        code = '"{0}`n`t{1}" -f "a","b"'
        result = self._deobfuscate(code)
        self.assertNotIn("@'", result)
        self.assertIn('`n', result)
        self.assertIn('`t', result)

    def test_concat_with_control_chars_no_raw_embedding(self):
        code = "'hello' + \"`tworld\""
        result = self._deobfuscate(code)
        self.assertNotIn('\t', result)
        self.assertIn('hello', result)
        self.assertIn('world', result)


class TestPs1StringMultiplicationFolding(TestPs1):

    def test_string_times_int(self):
        result = self._deobfuscate("'x' * 5")
        self.assertIn('xxxxx', result)
        self.assertNotIn('*', result)

    def test_int_times_string(self):
        # `int * string` is governed by the integer left operand, so PowerShell coerces the right
        # side to a number (here a runtime error) rather than repeating the string; it must not be
        # folded to 'ababab'.
        result = self._deobfuscate("3 * 'ab'")
        self.assertNotIn('ababab', result)

    def test_string_multiply_in_expression(self):
        result = self._deobfuscate("$x = 'A' * 3 + 'B'")
        self.assertIn('AAAB', result)

    def test_string_multiply_zero(self):
        result = self._deobfuscate("'hello' * 0")
        self.assertNotIn('hello', result)
        self.assertNotIn('*', result)


class TestPs1RangeExpressionFolding(TestPs1):

    def test_ascending_range(self):
        result = self._deobfuscate('1..3')
        self.assertIn('1', result)
        self.assertIn('2', result)
        self.assertIn('3', result)

    def test_descending_range(self):
        result = self._deobfuscate('3..1')
        self.assertIn('3', result)
        self.assertIn('2', result)
        self.assertIn('1', result)

    def test_char_array_cast_on_range(self):
        result = self._deobfuscate("[char[]](65..67) -Join ''")
        self.assertIn('ABC', result)

    def test_single_element_range(self):
        result = self._deobfuscate('5..5')
        self.assertIn('5', result)
        self.assertNotIn('..', result)

    def test_range_used_as_index(self):
        result = self._deobfuscate("'ABCDEFG'[1..3] -Join ''")
        self.assertIn('BCD', result)
        self.assertNotIn('..', result)


class TestPs1UnaryOperatorFolding(TestPs1):

    def test_bnot_integer(self):
        self.assertEqual(self._apply('$x = -bnot 0', Ps1ConstantFolding), '$x = -1')

    def test_bnot_hex(self):
        # `0xFF00` fills neither width, so it is the Int32 65280 and its complement is -65281.
        self.assertEqual(self._apply('$x = -bnot 0xFF00', Ps1ConstantFolding), '$x = -65281')

    def test_not_zero(self):
        result = self._deobfuscate('-not 0')
        self.assertIn('$True', result)

    def test_not_nonzero(self):
        result = self._deobfuscate('-not 1')
        self.assertIn('$False', result)

    def test_not_true(self):
        result = self._deobfuscate('-not $True')
        self.assertIn('$False', result)

    def test_bang_false(self):
        result = self._deobfuscate('!$False')
        self.assertIn('$True', result)


class TestPs1ComplementIsFoldedAtTheWidthItsOperandTakes(TestPs1):
    """
    5.1 converts the operand of `-bnot` and complements the bits at the width the conversion
    reached, which is not the operand's own type. Measured, `-bnot [byte]5` and `-bnot [char]65` are
    Int32 -6 and -66, `-bnot 1L` is Int64 -2, and `-bnot [uint32]7` is UInt32 4294967288 — the same
    bits as -8 and not the same number. So the value that comes back carries a type as much as a
    number, and the emitted literal has to spell both.
    """

    def test_a_narrow_integer_operand_widens_to_an_int32(self):
        self.assertEqual(self._apply('$x = -bnot [byte]5', Ps1ConstantFolding), '$x = -6')

    def test_a_long_operand_stays_a_long(self):
        self.assertEqual(self._apply('$x = -bnot 1L', Ps1ConstantFolding), '$x = -2L')

    def test_an_unsigned_operand_keeps_its_unsigned_width(self):
        self.assertEqual(
            self._apply('$x = -bnot [uint32]7', Ps1ConstantFolding), '$x = [uint32]4294967288')

    def test_an_operand_5_1_converts_to_a_number_is_complemented_at_the_number(self):
        # Measured: `[int]$null` is 0, `[int]$true` is 1, `[int]'5'` is 5, `[int][char]65` is 65,
        # `[int]1.5` is 2 and `[int]10d` is 10, and each complement below is that number's.
        self.assertEqual(self._apply('$x = -bnot $null', Ps1ConstantFolding), '$x = -1')
        self.assertEqual(self._apply('$x = -bnot $true', Ps1ConstantFolding), '$x = -2')
        self.assertEqual(self._apply("$x = -bnot '5'", Ps1ConstantFolding), '$x = -6')
        self.assertEqual(self._apply('$x = -bnot [char]65', Ps1ConstantFolding), '$x = -66')
        self.assertEqual(self._apply('$x = -bnot 1.5', Ps1ConstantFolding), '$x = -3')
        self.assertEqual(self._apply('$x = -bnot 10d', Ps1ConstantFolding), '$x = -11')

    def test_a_complement_folded_to_a_negative_number_in_an_argument_is_parenthesized(self):
        self.assertEqual(
            self._deobfuscate('$t = -bnot 0xFF00; Write-Output $t'), 'Write-Output (-65281)')

    def test_a_string_that_spells_no_number_is_left_where_it_stands(self):
        # Measured, `-bnot 'abc'` throws where `-bnot '5'` is -6.
        self._assertUnchanged("$x = -bnot 'abc'", Ps1ConstantFolding)

    def test_a_real_whose_width_no_measurement_covers_is_left_where_it_stands(self):
        """
        Measured, `-bnot 1.5` is an Int32 -3 and `-bnot 3000000000.0` is a UInt32 1294967295: the
        width follows the magnitude, so an answer keyed on the operand being a Double would be
        -3000000001 under a type 5.1 never produced here.
        """
        self._assertUnchanged('$x = -bnot 3000000000.0', Ps1ConstantFolding)


class TestPs1AHexadecimalLiteralIsNegativeWhereverItsNumberIsRead(TestPs1):
    """
    `0xFFFFFFFF` is the Int32 -1 on 5.1 and not the magnitude its eight digits spell, measured. Each
    site below reads a number out of a literal, so each is a place where the same eight digits are
    one script under the value and a different script under the digits — and where reading the
    digits would have been off by 4294967296.

    Each assertion pairs the answer 5.1 gives for -1 at that site with the site itself.
    """

    def test_a_range_counts_from_the_negative_number_the_bound_names(self):
        self.assertEqual(self._apply('$x = 0xFFFFFFFF..1', Ps1ConstantFolding), '$x = -1, 0, 1')

    def test_an_index_into_an_array_counts_from_the_end(self):
        self.assertEqual(
            self._apply('$x = @(10, 20, 30)[0xFFFFFFFF]', Ps1ConstantFolding), '$x = 30')

    def test_an_index_into_a_string_counts_from_the_end(self):
        # 5.1 selects a Char here, which the fold spells as a one-character String; what this pins
        # is which character was selected.
        self.assertEqual(self._apply("$x = 'ABCDE'[0xFFFFFFFF]", Ps1ConstantFolding), "$x = 'E'")

    def test_a_conversion_reads_its_argument_as_the_negative_number(self):
        self.assertEqual(
            self._apply('$x = [Convert]::ToInt32(0xFFFFFFFF)', Ps1ConstantFolding), '$x = -1')

    def test_a_conversion_reads_its_base_as_the_number_the_literal_names(self):
        self.assertEqual(
            self._apply("$x = [Convert]::ToInt32('7f', 0x10)", Ps1ConstantFolding), '$x = 127')

    def test_a_byte_offset_is_read_as_the_number_the_literal_names(self):
        self.assertEqual(
            self._apply(
                '$x = [BitConverter]::ToString(@(0x41, 0x42, 0x43, 0x44), 0x1, 2)',
                Ps1ConstantFolding),
            "$x = '42-43'",
        )

    def test_a_repeat_count_is_read_as_the_number_the_literal_names(self):
        self.assertEqual(self._apply("$x = 'ab' * 0x3", Ps1ConstantFolding), "$x = 'ababab'")

    def test_a_format_argument_is_formatted_as_the_negative_number_it_is(self):
        """
        The two spellings below name the same eight digits and print differently: .NET writes the
        Int32 -1 as `-001` under `D3` and the Int64 4294967295 as its ten digits, which is what the
        magnitude reading of `0xFFFFFFFF` would have produced.
        """
        self.assertEqual(
            self._apply("$x = '{0:D3}' -f 0xFFFFFFFF", Ps1ConstantFolding), "$x = '-001'")
        self.assertEqual(
            self._apply("$x = '{0:D3}' -f 4294967295", Ps1ConstantFolding), "$x = '4294967295'")

    def test_a_loop_whose_bound_is_the_negative_number_never_runs(self):
        """
        `0 -lt 0xFFFFFFFF` is `0 -lt -1`, which is false, so the body never runs and the whole
        script is the initializer. Written in decimal the same eight digits are a positive Int64 and
        the loop runs, which is why the two are asserted together.
        """
        self.assertEqual(
            self._deobfuscate('for ($i = 0; $i -lt 0xFFFFFFFF; $i++) { Write-Output 1 }'),
            '$i = 0',
        )
        self.assertEqual(
            self._deobfuscate('for ($i = 0; $i -lt 4294967295; $i++) { Write-Output 1 }'),
            inspect.cleandoc("""
                for ($i = 0; $i -LT 4294967295; $i++) {
                  Write-Output 1
                }
            """),
        )


class TestPs1AnIntegerSiteDeclinesWhereFivePointOneThrows(TestPs1):
    """
    A negative number reaches these sites as an argument 5.1 refuses: a Char has no code point -1,
    a conversion has no base -16, an array offset of -1 is out of range, and a Byte cannot hold -1 —
    measured, `[byte]-1` throws. Each expression therefore terminates the script where it stands,
    and there is no value for a fold to put in its place.
    """

    def test_a_character_code_no_char_holds_is_not_folded(self):
        self._assertUnchanged('$x = [Convert]::ToChar(0xFFFFFFFF)', Ps1ConstantFolding)

    def test_a_conversion_base_no_conversion_accepts_is_not_folded(self):
        # `0xFFFFFFF0` is the Int32 -16, which is not base 16 and is no base at all.
        self._assertUnchanged("$x = [Convert]::ToInt32('11', 0xFFFFFFF0)", Ps1ConstantFolding)

    def test_a_number_no_byte_holds_is_not_folded_into_a_byte_array(self):
        self._assertUnchanged('$x = [BitConverter]::ToString(@(0xFFFFFFFF))', Ps1ConstantFolding)

    def test_an_offset_outside_the_array_is_not_folded(self):
        self._assertUnchanged(
            '$x = [BitConverter]::ToString(@(0x41, 0x42, 0x43, 0x44), 0xFFFFFFFF, 2)',
            Ps1ConstantFolding,
        )


class TestPs1AValueThatIsNoIntegerIsNotReadAsOne(TestPs1):
    """
    5.1 reaches a number from each of these by converting, and the conversion has a rule of its own:
    measured, `[int]1.5` is 2 and `[int]2.5` is 2, so it rounds half to even rather than truncating,
    and `[int]'5'` parses text. None of that is what the value *is*, so a site reading an integer
    declines and leaves the expression alone.

    What each assertion rules out is the other answer: a repeat count truncated from 1.5 to 1 would
    emit `'ab'` where 5.1 writes `abab`, and an index truncated from 1.5 to 1 would select 20 where
    5.1 selects 30.
    """

    def test_a_real_repeat_count_is_not_truncated_to_a_whole_one(self):
        self._assertUnchanged("$x = 'ab' * 1.5", Ps1ConstantFolding)

    def test_a_real_index_is_not_truncated_to_a_whole_one(self):
        self._assertUnchanged('$x = @(10, 20, 30)[1.5]', Ps1ConstantFolding)

    def test_a_real_byte_is_not_truncated_to_a_whole_one(self):
        self._assertUnchanged('$x = [BitConverter]::ToString(@(1.5))', Ps1ConstantFolding)

    def test_a_real_conversion_base_is_not_truncated_to_a_whole_one(self):
        self._assertUnchanged("$x = [Convert]::ToInt32('41', 16.0)", Ps1ConstantFolding)

    def test_a_repeat_count_of_a_type_that_holds_no_width_is_left_alone(self):
        # 5.1 writes '', `ab` and `ababab` for these three, so declining costs those three folds
        # and is the only answer that cannot invent a fourth.
        self._assertUnchanged("$x = 'ab' * $null", Ps1ConstantFolding)
        self._assertUnchanged("$x = 'ab' * $true", Ps1ConstantFolding)
        self._assertUnchanged("$x = 'ab' * '3'", Ps1ConstantFolding)


class TestPs1ReplicatingAStringANegativeNumberOfTimes(TestPs1):
    """
    Measured, `'ab' * 0xFFFFFFFF` throws `System.ArgumentOutOfRangeException` and the script writes
    nothing: 5.1 replicates a string zero or more times and refuses a negative count outright. A
    fold that answered the empty string would produce a value the input never had, so a `catch`
    around the line would run in the input and not in the output.
    """

    def test_a_negative_repeat_count_leaves_the_replication_where_it_stands(self):
        self._assertUnchanged("$x = 'ab' * 0xFFFFFFFF", Ps1ConstantFolding)

    def test_a_negative_repeat_count_written_in_decimal_is_no_different(self):
        self._assertUnchanged("$x = 'ab' * -1", Ps1ConstantFolding)

    def test_a_zero_repeat_count_is_the_empty_string_and_not_a_refusal(self):
        self.assertEqual(self._apply("$x = 'ab' * 0", Ps1ConstantFolding), "$x = ''")


class TestPs1ConvertFolding(TestPs1):

    def test_toint32_hex_base(self):
        result = self._deobfuscate("[Convert]::ToInt32('41', 16)")
        self.assertIn('65', result)
        self.assertNotIn('Convert', result)

    def test_toint32_binary_base(self):
        result = self._deobfuscate("[Convert]::ToInt32('01000001', 2)")
        self.assertIn('65', result)

    def test_toint32_decimal_string(self):
        result = self._deobfuscate("[Convert]::ToInt32('123')")
        self.assertIn('123', result)
        self.assertNotIn('Convert', result)

    def test_tobyte(self):
        result = self._deobfuscate("[Convert]::ToByte('FF', 16)")
        self.assertIn('255', result)

    def test_tochar(self):
        result = self._deobfuscate('[Convert]::ToChar(65)')
        self.assertIn('A', result)

    def test_toint32_octal_base(self):
        result = self._deobfuscate("[Convert]::ToInt32('77', 8)")
        self.assertIn('63', result)


class TestPs1NegativeIndexFolding(TestPs1):

    def test_string_negative_one(self):
        result = self._deobfuscate("'hello'[-1]")
        self.assertIn('o', result)
        self.assertNotIn('-1', result)

    def test_string_negative_two(self):
        result = self._deobfuscate("'ABCDE'[-2]")
        self.assertIn('D', result)

    def test_array_negative_one(self):
        result = self._deobfuscate('@(10, 20, 30)[-1]')
        self.assertIn('30', result)
        self.assertNotIn('-1', result)

    def test_string_multi_negative_index(self):
        result = self._deobfuscate("'ABCDE'[-1, -3] -Join ''")
        self.assertIn('EC', result)


class TestPs1FormatStringSpecifiers(TestPs1):

    def test_hex_uppercase(self):
        result = self._deobfuscate("'{0:X2}' -f 65")
        self.assertIn('41', result)

    def test_hex_lowercase(self):
        result = self._deobfuscate("'{0:x4}' -f 255")
        self.assertIn('00ff', result)

    def test_decimal_padding(self):
        result = self._deobfuscate("'{0:D3}' -f 7")
        self.assertIn('007', result)

    def test_alignment_right(self):
        result = self._deobfuscate("'{0,5}' -f 'hi'")
        self.assertIn('   hi', result)

    def test_alignment_left(self):
        result = self._deobfuscate("'{0,-5}' -f 'hi'")
        self.assertIn('hi   ', result)

    def test_alignment_with_format(self):
        result = self._deobfuscate("'{0,6:X2}' -f 255")
        self.assertIn('    FF', result)

    def test_existing_basic_format(self):
        result = self._deobfuscate('"{0}{2}{1}" -f "signa","ures","t"')
        self.assertIn('signatures', result)

    def test_hex_multi_arg(self):
        result = self._deobfuscate("'{0:X2}{1:X2}' -f 72, 105")
        self.assertIn('4869', result)


class TestPs1BitConverterFolding(TestPs1):

    def test_tostring_basic(self):
        result = self._deobfuscate('[BitConverter]::ToString(@(0x41, 0x42, 0x43))')
        self.assertIn('41-42-43', result)

    def test_tostring_single_byte(self):
        result = self._deobfuscate('[BitConverter]::ToString(@(0xFF))')
        self.assertNotIn('BitConverter', result)
        self.assertIn('FF', result)

    def test_tostring_with_offset_and_length(self):
        result = self._deobfuscate('[BitConverter]::ToString(@(0x41, 0x42, 0x43, 0x44), 1, 2)')
        self.assertIn('42-43', result)


class TestPs1EnvironmentVariableFolding(TestPs1):

    def test_comspec(self):
        result = self._deobfuscate("[Environment]::GetEnvironmentVariable('ComSpec')")
        self.assertIn('cmd.exe', result)
        self.assertNotIn('GetEnvironmentVariable', result)

    def test_os(self):
        result = self._deobfuscate("[Environment]::GetEnvironmentVariable('OS')")
        self.assertIn('Windows_NT', result)

    def test_unknown_variable_not_folded(self):
        result = self._deobfuscate("[Environment]::GetEnvironmentVariable('CUSTOM_VAR')")
        self.assertIn('GetEnvironmentVariable', result)


class TestPs1HashtableLookup(TestPs1):

    def test_basic_string_lookup(self):
        result = self._deobfuscate("@{'a'='hello'}['a']")
        self.assertIn('hello', result)
        self.assertNotIn('@{', result)

    def test_integer_value_lookup(self):
        result = self._deobfuscate("@{'x'=42}['x']")
        self.assertIn('42', result)

    def test_missing_key_not_folded(self):
        result = self._deobfuscate("@{'a'='hello'}['b']")
        self.assertIn('@{', result)


class TestPs1FoldingExtra(TestPs1):

    def test_format_string_basic(self):
        data = '"{0}{2}{1}" -f "signa","ures","t"'
        result = self._deobfuscate(data)
        self.assertIn('signatures', result)

    def test_format_string_in_parens(self):
        data = '("{0}{2}{1}"-f "signa","ures","t")'
        result = self._deobfuscate(data)
        self.assertIn('signatures', result)

    def test_format_string_single_quotes(self):
        data = "('{2}{0}{1}'-f'c','m','g')"
        result = self._deobfuscate(data)
        self.assertIn('gcm', result)

    def test_b64convert(self):
        data = '[System.Convert]::FromBase64String("AQID")'
        result = self._deobfuscate(data)
        self.assertIn('0x01', result)
        self.assertIn('0x02', result)
        self.assertIn('0x03', result)

    def test_b64convert_unqualified(self):
        data = '[Convert]::FromBase64String("AQID")'
        result = self._deobfuscate(data)
        self.assertIn('0x01', result)
        self.assertIn('0x02', result)
        self.assertIn('0x03', result)

    def test_b64convert_parenthesized_type(self):
        data = '([Convert])::FromBase64String("AQID")'
        result = self._deobfuscate(data)
        self.assertIn('0x01', result)
        self.assertIn('0x02', result)
        self.assertIn('0x03', result)

    def test_encoding_utf8(self):
        data = '[System.Text.Encoding]::UTF8.GetString(@(72, 101, 108, 108, 111))'
        result = self._deobfuscate(data)
        self.assertIn('Hello', result)

    def test_arithmetic_constant_folding(self):
        result = self._deobfuscate(
            '$foo=36665-36663\n'
            '$bar=345-BXor21\n'
            '$baz=744-BAND(3254-1221)*99-BXor743\n'
        ).lower()
        self.assertNotIn('36665', result)
        self.assertIn('$foo = 2', result)
        self.assertNotIn('bxor', result)
        self.assertIn('$bar = 332', result)
        self.assertNotIn('band', result)
        self.assertIn('$baz = 199', result)

    def test_split_constant_string(self):
        result = self._deobfuscate("'aXbYcZd'.Split('XYZ')")
        self.assertIn("'a'", result)
        self.assertIn("'b'", result)
        self.assertIn("'c'", result)
        self.assertIn("'d'", result)

    def test_join_scalar_string_is_noop(self):
        result = self._deobfuscate("-Join 'hello'")
        self.assertNotIn('-Join', result)
        self.assertNotIn('-join', result)
        self.assertIn('hello', result)

    def test_binary_split_single(self):
        result = self._deobfuscate("'aXbXc' -Split 'X'")
        self.assertIn("'a'", result)
        self.assertIn("'b'", result)
        self.assertIn("'c'", result)

    def test_binary_split_chained(self):
        result = self._deobfuscate("'aXbYc' -Split 'X' -Split 'Y'")
        self.assertIn("'a'", result)
        self.assertIn("'b'", result)
        self.assertIn("'c'", result)

    def test_shift_operations(self):
        result = self._deobfuscate("$x = 1 -Shl 4; $y = 256 -Shr 3")
        self.assertIn('16', result)
        self.assertIn('32', result)

    def test_format_expression_chained(self):
        result = self._deobfuscate('$x = "{0}" -f "a" -f "b"')
        self.assertIn("'a'", result)

    def test_shift_operators_use_int32_semantics(self):
        # `-shl`/`-shr` fold with .NET Int32 wraparound and shift-count masking.
        self.assertEqual('-2147483648', self._apply('1 -shl 31', Ps1ConstantFolding))
        self.assertEqual('1', self._apply('1 -shl 32', Ps1ConstantFolding))

    def test_integer_division_yields_double(self):
        self.assertEqual(self._apply('7 / 2', Ps1ConstantFolding), '3.5')

    def test_integer_modulo_keeps_dividend_sign(self):
        self.assertEqual(self._apply('-7 % 3', Ps1ConstantFolding), '-1')

    def test_replace_expands_group_reference(self):
        result = self._apply("'aXb' -replace '(a)X','$1Y'", Ps1ConstantFolding)
        self.assertEqual(result, "'aYb'")

    def test_replace_treats_backslash_literally(self):
        result = self._apply(r"'aXb' -replace '(a)X','Q\1Z'", Ps1ConstantFolding)
        self.assertEqual(result, r"'Q\1Zb'")

    def test_format_hex_negative_twos_complement(self):
        result = self._apply("'{0:X}' -f -1", Ps1ConstantFolding)
        self.assertEqual(result, "'FFFFFFFF'")

    def test_leading_zero_integer_is_decimal(self):
        self.assertEqual(self._apply('007 + 1', Ps1ConstantFolding), '8')


class TestPs1SelectionKeepsWhatBuildingTheContainerDid(TestPs1):

    def test_an_array_element_that_runs_a_command_is_not_dropped(self):
        self._assertUnchanged('$r = @(1, (Start-Process calc))[0]', Ps1ConstantFolding)

    def test_an_array_element_that_writes_a_variable_is_not_dropped(self):
        self._assertUnchanged('$r = @(1, $x++)[0]', Ps1ConstantFolding)

    def test_an_array_element_that_can_raise_is_not_dropped(self):
        # `[Int]'abc'` terminates where it stands, so the input runs the handler and a folded
        # output cannot: the payload path would be dead code. Purity does not answer this — the
        # cast is side-effect free and still raises.
        self.assertEqual(
            self._apply(
                "try { $r = @(1, [Int]'abc')[0] } catch { Start-Process calc }",
                Ps1ConstantFolding),
            "try {\n  $r = @(1, [Int]'abc')[0]\n} catch {\n  Start-Process calc\n}")

    def test_a_hashtable_value_that_runs_a_command_is_not_dropped(self):
        self.assertEqual(
            self._apply("$r = @{ a = 1; b = (Start-Process calc) }['a']", Ps1ConstantFolding),
            "$r = @{\n  a = 1\n  b = (Start-Process calc)\n}['a']")

    def test_a_hashtable_key_that_runs_a_command_is_not_dropped(self):
        # PowerShell rejects a bare subexpression key, so this is the spelling that reaches here.
        self.assertEqual(
            self._apply(
                '$r = @{ "$(Start-Process calc)" = 1; a = 2 }[\'a\']', Ps1ConstantFolding),
            '$r = @{\n  "$(Start-Process calc)" = 1\n  a = 2\n}[\'a\']')

    def test_an_array_of_literals_is_still_selected_from(self):
        self.assertEqual(self._apply('$r = @(1, 2)[0]', Ps1ConstantFolding), '$r = 1')

    def test_several_indices_into_an_array_of_literals_still_select(self):
        self.assertEqual(self._apply('$r = @(1, 2, 3)[0, 2]', Ps1ConstantFolding), '$r = 1, 3')

    def test_a_hashtable_of_literals_is_still_looked_up(self):
        self.assertEqual(
            self._apply("$r = @{ a = 1; b = 2 }['a']", Ps1ConstantFolding), '$r = 1')

    def test_a_character_of_a_string_leaves_no_evaluation_behind(self):
        # A string is a value and not a container of expressions, so the characters this does not
        # select are not work the folded script stops doing and no gate applies to them. The array
        # of the same shape one line up is refused.
        self.assertEqual(self._apply("$r = 'abc'[0]", Ps1ConstantFolding), "$r = 'a'")


class TestPs1ConstantInliningSelectsOnlyFromConstants(TestPs1):
    """
    `Ps1ConstantInlining` selects out of a container too, and the claim that it needs no gate is
    that its containers are already proven constant. These pin the claim rather than the gate.
    """

    def test_an_array_holding_a_command_is_not_a_constant_to_inline_from(self):
        self._assertUnchanged(
            '$a = @(1, (Start-Process calc))\n$r = $a[0]', Ps1ConstantInlining)

    def test_an_array_holding_an_increment_is_not_a_constant_to_inline_from(self):
        self._assertUnchanged('$a = @(1, $x++)\n$r = $a[0]', Ps1ConstantInlining)

    def test_an_array_holding_a_raising_cast_is_not_a_constant_to_inline_from(self):
        self._assertUnchanged("$a = @(1, [Int]'abc')\n$r = $a[0]", Ps1ConstantInlining)

    def test_an_array_of_literals_is_still_inlined_from(self):
        self.assertEqual(
            self._apply('$a = @(1, 2)\n$r = $a[0]', Ps1ConstantInlining), '$r = 1')

    def test_a_string_is_still_indexed_when_inlined(self):
        self.assertEqual(
            self._apply("$a = 'abc'\n$r = $a[0]", Ps1ConstantInlining), "$r = 'a'")


class TestPs1FoldingRedirections(TestPs1):

    def test_a_redirected_regex_pipeline_is_not_folded_into_its_value(self):
        self._assertUnchanged(
            "$m = [regex]::Matches('abc', 'b') | ForEach-Object {\n  $_.Value\n} > C:\\o.txt",
            Ps1ConstantFolding)

    def test_an_unredirected_regex_pipeline_is_still_folded(self):
        self.assertEqual(
            self._apply(
                "$m = [regex]::Matches('abc', 'b') | %{ $_.Value }", Ps1ConstantFolding),
            "$m = 'b'")


class TestPs1CountingAnArrayKeepsWhatBuildingItDid(TestPs1):

    def test_the_length_of_an_array_holding_a_command_is_not_folded(self):
        self.assertEqual(
            self._apply('$x = @(1, (Start-Process calc)).Length', Ps1ConstantFolding),
            '$x = @(1, (Start-Process calc)).Length')

    def test_the_count_of_an_array_holding_an_increment_is_not_folded(self):
        self._assertUnchanged('$x = @(1, $y++).Count', Ps1ConstantFolding)

    def test_the_length_of_an_array_holding_a_raising_cast_is_not_folded(self):
        self._assertUnchanged("$x = @(1, [Int]'abc').Length", Ps1ConstantFolding)

    def test_the_length_of_an_array_of_literals_is_still_folded(self):
        self.assertEqual(self._apply('$x = @(1, 2).Length', Ps1ConstantFolding), '$x = 2')

    def test_the_length_of_a_string_is_still_folded(self):
        self.assertEqual(self._apply("$x = 'abc'.Length", Ps1ConstantFolding), '$x = 3')

    def test_a_refused_selection_leaves_every_parent_pointer_true(self):
        source = '$r = @(1, 2, (Start-Process calc))[0, 1]'
        self._assertTreeIsIntact(source, source, Ps1ConstantFolding)

    def test_a_refused_count_leaves_every_parent_pointer_true(self):
        source = '$x = @(1, (Start-Process calc)).Length'
        self._assertTreeIsIntact(source, source, Ps1ConstantFolding)

    def test_a_repeated_index_into_an_array_is_not_folded(self):
        # Folding would put the array's own node in two slots of the result, where `Node.parent`
        # holds one holder: a later replacement rewrites one occurrence of two and a walk counts it
        # twice. Copying it would run whatever it holds twice instead.
        self._assertUnchanged('$r = @(1, 2, 3)[0, 0]', Ps1ConstantFolding)

    def test_a_repeated_index_reached_through_a_negative_one_is_not_folded_either(self):
        self._assertUnchanged('$r = @(1, 2, 3)[0, -3]', Ps1ConstantFolding)

    def test_distinct_indices_into_the_same_array_still_select(self):
        self.assertEqual(self._apply('$r = @(1, 2, 3)[2, 0]', Ps1ConstantFolding), '$r = 3, 1')

    def test_a_repeated_index_into_a_string_still_folds(self):
        # A character is built fresh per index out of a value that was never a node, so nothing is
        # shared and the shape obfuscators actually write keeps working.
        self.assertEqual(self._apply("$r = 'abc'[0, 0, 1]", Ps1ConstantFolding), "$r = 'a', 'a', 'b'")


class TestPs1ArrayReverseIsAppliedWhereItIsWritten(TestPs1):
    """
    `[Array]::Reverse` reverses in place and returns nothing, so folding it away means moving its
    effect back to the statement that built the array. That is only faithful where the array is
    what was passed and where nothing observes it in between.

    Every test here is a defect the tool still has, marked as a failure that is expected so that it
    ratchets in both directions: the fix has to unmark them. Each is carried a second time by
    `test/lib/scripts/ps1/corpus.py`, where the same three scripts are run on a 5.1 host and their
    transcripts compared, so neither witness rests on the other. What is asserted here is the
    conservative correct output — the transform declining — and a fold that reversed at the right
    point would be correct too and would rewrite these.
    """

    @unittest.expectedFailure
    def test_reversing_a_string_leaves_the_string_unchanged(self):
        # 5.1 binds a String to the `System.Array` parameter by converting it to a fresh `Char[]`,
        # reverses that copy and discards it, so the script prints `abc`.
        source = "$s = 'abc'\n[Array]::Reverse($s)\nWrite-Output $s"
        self.assertEqual(self._deobfuscate(source), source)

    @unittest.expectedFailure
    def test_a_read_between_the_assignment_and_the_reversal_sees_the_original_order(self):
        # 5.1 prints 1 and then 3: the first read runs before the reversal does.
        source = '$x = 1, 2, 3\nWrite-Output $x[0]\n[Array]::Reverse($x)\nWrite-Output $x[0]'
        self.assertEqual(self._deobfuscate(source), source)

    @unittest.expectedFailure
    def test_an_element_written_between_the_assignment_and_the_reversal_is_reversed_with_the_rest(
        self,
    ):
        # 5.1 leaves `3 2 9`: the write lands before the reversal, so it moves to the far end.
        source = '$x = 1, 2, 3\n$x[0] = 9\n[Array]::Reverse($x)\nWrite-Output $x'
        self.assertEqual(self._deobfuscate(source), source)
