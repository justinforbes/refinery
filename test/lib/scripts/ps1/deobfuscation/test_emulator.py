from __future__ import annotations

from inspect import cleandoc

from test.lib.scripts.ps1.deobfuscation import TestPs1

from refinery.lib.scripts.ps1 import data
from refinery.lib.scripts.ps1.deobfuscation import (
    Ps1ForEachPipeline,
    Ps1FunctionEvaluator,
)
from refinery.lib.scripts.ps1.deobfuscation.emulator import _NO_OPERATOR_METHOD_ON_BOOLEAN

#: The Int32 that every measurement in `TestPs1WhichSideOfWhichOperatorABooleanMayStandOn` was taken
#: against, as the grid names its type.
INT32 = 'System.Int32'


def _boolean_cell(operator: str, right: str) -> data.OperatorOutcome:
    """
    The grid cell a Boolean left operand and `right` index. A cell the grid does not cover raises
    here, naming what was asked for, rather than answering `None`: an operator or a type the grid
    stopped covering has to fail the comparison it stands in and not drop out of it.
    """
    cell = data.binary_outcome(operator, 'System.Boolean', right)
    if cell is None:
        raise KeyError(F'the grid has no cell for a Boolean {operator} {right}')
    return cell


class TestPs1FunctionEvaluator(TestPs1):

    def test_stride_extraction(self):
        data = (
            "Function F ([String]$s){"
            "For($i=1; $i -lt $s.Length-1; $i+=2)"
            "{$r=$r+$s.Substring($i, 1)};$r;}"
            "$x = F 'HaEbLcLdOeX'"
            "\nWrite-Output $x"
        )
        result = self._deobfuscate(data)
        self.assertIn('abcde', result)
        self.assertNotIn('function', result.lower())

    def test_multiple_call_sites(self):
        data = (
            "Function D ([String]$s){"
            "For($i=1; $i -lt $s.Length-1; $i+=2)"
            "{$r=$r+$s.Substring($i, 1)};$r;}"
            "$a = D 'XaYbZcX'\n"
            "$b = D 'P1Q2R3X'\n"
            "Write-Output $a\nWrite-Output $b"
        )
        result = self._deobfuscate(data)
        self.assertIn('abc', result)
        self.assertIn('123', result)
        self.assertNotIn('function', result.lower())

    def test_nonconstant_arg_preserved(self):
        data = (
            "Function D ([String]$s){"
            "For($i=1; $i -lt $s.Length-1; $i+=2)"
            "{$r=$r+$s.Substring($i, 1)};$r;}"
            "$y = D $input"
        )
        result = self._deobfuscate(data)
        self.assertIn('$Input', result)
        self.assertIn('function', result.lower())

    def test_while_loop_variant(self):
        data = (
            "Function W ([String]$s){"
            "$i=0; $r=''; "
            "While($i -lt $s.Length){$r=$r+$s.Substring($i, 1); $i+=2};"
            "$r;}"
            "$x = W 'HEeLlLlOo'\nWrite-Output $x"
        )
        result = self._deobfuscate(data)
        self.assertIn('Hello', result)

    def test_foreach_tochararray(self):
        data = (
            "Function Rev ([String]$s){"
            "$a = $s.ToCharArray(); $r = '';"
            "ForEach($c in $a){$r = $c + $r};"
            "$r;}"
            "$x = Rev 'olleH'\nWrite-Output $x"
        )
        result = self._deobfuscate(data)
        self.assertIn('Hello', result)

    def test_if_inside_function(self):
        data = (
            "Function C ([String]$s){"
            "$r = '';"
            "For($i=0; $i -lt $s.Length; $i+=1){"
            "If ($i % 2 -eq 0){$r = $r + $s.Substring($i, 1)}"
            "}; $r;}"
            "$x = C 'HxExLxLxO'\nWrite-Output $x"
        )
        result = self._deobfuscate(data)
        self.assertIn('HELLO', result)

    def test_function_definition_kept_when_not_all_resolved(self):
        data = (
            "Function D ([String]$s){"
            "For($i=1; $i -lt $s.Length-1; $i+=2)"
            "{$r=$r+$s.Substring($i, 1)};$r;}"
            "$a = D 'XaYbX'\n"
            "$b = D $var"
        )
        result = self._deobfuscate(data)
        self.assertIn('ab', result)
        self.assertIn('function', result.lower())

    def test_return_statement(self):
        data = (
            "Function Dec ([String]$s){"
            "$r = '';"
            "For($i=0; $i -lt $s.Length; $i+=2){"
            "$r = $r + $s.Substring($i, 1)"
            "}; return $r;}"
            "$x = Dec 'HxExLxLxOx'\nWrite-Output $x"
        )
        result = self._deobfuscate(data)
        self.assertIn('HELLO', result)

    def test_do_while_loop(self):
        data = (
            "Function D ([String]$s){"
            "$i = 0; $r = '';"
            "Do{$r = $r + $s.Substring($i, 1); $i += 2}"
            "While($i -lt $s.Length);"
            "$r;}"
            "$x = D 'HxExLxLxOx'\nWrite-Output $x"
        )
        result = self._deobfuscate(data)
        self.assertIn('HELLO', result)

    def test_new_object_byte_array(self):
        data = (
            "Function F ([Int]$n){"
            "$a = New-Object byte[] $n;"
            "$r = '';"
            "For($i=0; $i -lt $n; $i+=1){$r = $r + $a[$i]};"
            "$r;}"
            "$x = F 3\n"
            "Write-Output $x"
        )
        result = self._deobfuscate(data)
        self.assertIn('000', result)

    def test_convert_tobyte_static(self):
        data = (
            "Function F ([String]$s){"
            "$r = [convert]::ToByte($s, 16);"
            "$r;}"
            "$x = F 'FF'\n"
            "Write-Output $x"
        )
        result = self._deobfuscate(data)
        self.assertIn('255', result)

    def test_encoding_getstring(self):
        data = (
            "Function F {"
            "$a = New-Object byte[] 3;"
            "$a[0] = 72; $a[1] = 105; $a[2] = 33;"
            "[System.Text.Encoding]::ASCII.GetString($a);}"
            "$x = F\n"
            "Write-Output $x"
        )
        result = self._deobfuscate(data)
        self.assertIn('Hi!', result)

    def test_hex_xor_decode_function(self):
        data = (
            "Function F ([String]$s){\n"
            "$a = New-Object byte[] ($s.Length / 2)\n"
            "For($i=0; $i -lt $s.Length; $i+=2){\n"
            "$a[$i/2] = [convert]::ToByte($s.Substring($i, 2), 16)\n"
            "$a[$i/2] = ($a[$i/2] -bxor 128)\n"
            "}\n"
            "[String][System.Text.Encoding]::ASCII.GetString($a)\n"
            "}\n"
            "$x = F 'C8E5ECECEF'\n"
            "Write-Output $x\n"
        )
        result = self._deobfuscate(data)
        self.assertIn('Hello', result)
        self.assertNotIn('function', result.lower())

    def test_base64_xor_decode_function(self):
        data = (
            "Function F ([String]$s, [Byte]$k) {\n"
            "$a = [System.Convert]::FromBase64String($s)\n"
            "For ($i = 0; $i -lt $a.Length; $i++) {\n"
            "$a[$i] = $a[$i] -bxor $k\n"
            "}\n"
            "return [System.Text.Encoding]::ASCII.GetString($a)\n"
            "}\n"
            "$x = F 'aEVMTE8=' 0x20\n"
            "Write-Output $x\n"
        )
        result = self._deobfuscate(data)
        self.assertIn('Hello', result)
        self.assertNotIn('function', result.lower())

    def test_named_parameters(self):
        data = (
            "function F { Param([string]$a, [string]$b); $a + $b }\n"
            "$x = F -a 'Hel' -b 'lo'\n"
            "Write-Output $x"
        )
        result = self._deobfuscate(data)
        self.assertIn('Hello', result)
        self.assertNotIn('function', result.lower())

    def test_named_parameters_unordered(self):
        data = (
            "function G { Param([string]$first, [string]$second); $second + $first }\n"
            "$x = G -second 'World' -first 'Hello'\n"
            "Write-Output $x"
        )
        result = self._deobfuscate(data)
        self.assertIn('WorldHello', result)
        self.assertNotIn('function', result.lower())

    def test_constant_inlining_respects_function_scope(self):
        data = (
            "$a = 'INLINED'\n"
            "function F { Param([string]$a); $a }\n"
            "$x = F -a 'Hello'\n"
            "Write-Output $x"
        )
        result = self._deobfuscate(data)
        self.assertIn('Hello', result)

    def test_iex_trampoline_function(self):
        data = (
            "function Wrapper { Param([string]$code); Invoke-Expression $code > $Null 2> $Null }\n"
            "function Builder { Param([string]$a, [string]$b); $r = $a + $b; Wrapper '$r' }\n"
            "Builder -a 'Write-Host ' -b 'Hello'"
        )
        result = self._deobfuscate(data)
        self.assertIn('Write-Host', result)
        self.assertIn('Hello', result)
        self.assertNotIn('function Builder', result)

    def test_decoy_function_producing_garbage_is_pruned(self):
        data = (
            "function IexWrap { Param([string]$c); Invoke-Expression $c }\n"
            "function Decoy { Param([string]$a); IexWrap ($a + '!!!###@@@') }\n"
            "function Real { Param([string]$a, [string]$b); IexWrap ($a + $b) }\n"
            "Decoy -a '!!!###@@@'\n"
            "Real -a 'Write-Host ' -b 'OK'"
        )
        result = self._deobfuscate(data)
        self.assertNotIn('function Decoy', result)
        self.assertNotIn('Decoy', result.split('Write-Host')[0])
        self.assertIn('Write-Host', result)
        self.assertIn('OK', result)

    def test_helper_only_called_from_function_bodies_is_pruned(self):
        data = (
            "function Helper { Param([string]$x); Invoke-Expression $x }\n"
            "function Caller { Param([string]$s); Helper $s }\n"
            "Caller -s 'Write-Host Done'"
        )
        result = self._deobfuscate(data)
        self.assertNotIn('function Helper', result)
        self.assertNotIn('function Caller', result)
        self.assertIn('Write-Host', result)
        self.assertIn('Done', result)


class TestPs1ForEachPipeline(TestPs1):

    def test_foreach_pipeline_char_convert(self):
        data = "'72z101z108z108z111'.Split('z') | %{ ([Char]([Convert]::ToInt16(($_.ToString()), 10))) }"
        result = self._deobfuscate(data)
        self.assertEqual(result, "'H', 'e', 'l', 'l', 'o'")

    def test_foreach_pipeline_negative_integers(self):
        data = "((-83,-71,-65,-75,-107,-70,-75,-64,-110,-83,-75,-72,-79,-80) | %{ [char]($_ + 180) }) -join ''"
        result = self._deobfuscate(data)
        self.assertIn('amsiInitFailed', result)

    def test_foreach_pipeline_mixed_sign_integers(self):
        data = "(-4, 1, -17) | %{ [char]($_ + 104) }"
        result = self._deobfuscate(data)
        self.assertIn('d', result)
        self.assertIn('i', result)
        self.assertIn('W', result)

    def test_foreach_pipeline_expandable_string_hex_decode(self):
        data = "'46 75 6E' -split ' ' | %{[char][byte]\"0x$_\"}"
        result = self._deobfuscate(data)
        self.assertEqual(result, "'F', 'u', 'n'")

    def test_foreach_pipeline_expandable_string_with_subexpr(self):
        data = "@('A','B','C') | %{\"item: $( $_ )\"}"
        result = self._deobfuscate(data)
        self.assertIn('item: A', result)

    def test_foreach_pipeline_split_join_chain(self):
        data = (
            "$s = '48 65 6C 6C 6F'\n"
            "$r = $s -split ' ' | ForEach-Object {[char][byte]\"0x$_\"}\n"
            "$r -join ''"
        )
        result = self._deobfuscate_iterative(data)
        self.assertIn('Hello', result)

    def test_foreach_pipeline_replace_operator(self):
        data = "@('Hello','World') | %{$_ -replace 'o','0'}"
        result = self._deobfuscate(data)
        self.assertIn('Hell0', result)
        self.assertIn('W0rld', result)

    def test_foreach_pipeline_array_expression(self):
        data = "@(65,66,67) | %{[char]$_}"
        result = self._deobfuscate(data)
        self.assertIn('A', result)
        self.assertIn('B', result)


class TestPs1EmulatorExtra(TestPs1):

    def test_cast_wrapped_array_pipeline(self):
        result = self._deobfuscate(
            "[String]([Char[]] (72,101,108,108,111) | "
            "ForEach-Object { [Char]($_ -BXor 0) })")
        self.assertIn('Hello', result)

    def test_a_string_cast_over_an_xor_pipeline_joins_on_the_default_ofs(self):
        result = self._deobfuscate("[String]([Char[]] (127,78,88,95) | % { [Char]($_ -BXor 0x2B) })")
        self.assertEqual(result, "'T e s t'")

    def test_function_multiple_outputs_form_array(self):
        result = self._apply("function f { 'a'; 'b' }; $x = f", Ps1FunctionEvaluator)
        self.assertEqual(result, "$x = 'a', 'b'")

    def test_function_trailing_assignment_emits_nothing(self):
        # g only assigns, so it returns $null; the call must not fold to the assigned value, so
        # `$y = g` is left as-is.
        result = self._apply("function g { $r = 'hidden' }; $y = g", Ps1FunctionEvaluator)
        self.assertEqual(result, cleandoc("""
            function g {
              $r = 'hidden'
            }
            $y = g
        """))

    def test_emulated_shift_uses_int32_semantics(self):
        # The emulator must shift with the same .NET semantics as constant folding; a raw Python
        # shift would evaluate `1 -shl 32` to 4294967296 instead of 1.
        result = self._apply(
            'function f ($n) { $n -shl 32 }\n$x = f 1', Ps1FunctionEvaluator)
        self.assertIn('$x = 1', result)
        self.assertNotIn('4294967296', result)

    def test_foreach_pipeline_yields_array_not_joined_string(self):
        # Multi-character results stay an array (so indexing selects an element), unlike the
        # single-character char-build case which is joined.
        result = self._apply("('foo','bar' | %{ $_ })[1]", Ps1ForEachPipeline)
        self.assertEqual(result, "('foo', 'bar')[1]")

    def test_nested_function_reads_enclosing_scope(self):
        # Plain reads are transitive through the scope chain: C sees A's $g via the call stack.
        # Verified against PowerShell (returns 'X').
        result = self._apply(
            "function A { $g = 'X'; B } function B { C } function C { return $g }; $o = A",
            Ps1FunctionEvaluator)
        self.assertEqual(result, cleandoc("""
            function C {
              return $g
            }
            $o = 'X'
        """))

    def test_compound_assignment_does_not_read_enclosing_scope(self):
        # `$acc += 'C'` reads only the local scope, so against an enclosing-scope $acc it starts
        # from $null. Verified against PowerShell: the result is 'C', not 'ABC'.
        result = self._apply(
            "function A { $acc = 'AB'; B } function B { $acc += 'C'; return $acc }; $o = A",
            Ps1FunctionEvaluator)
        self.assertEqual(result, "$o = 'C'")

    def test_psitem_is_pipeline_item(self):
        result = self._apply("(97,98,99) | % { [char]$PSItem }", Ps1ForEachPipeline)
        self.assertEqual(result, "'a', 'b', 'c'")

    def test_foreach_over_string_is_scalar(self):
        # PowerShell iterates a foreach over a string exactly once (the string is a scalar).
        result = self._apply(
            "function f($s){ $n = 0; foreach($c in $s){ $n = $n + 1 }; return $n }; $r = f 'abc'",
            Ps1FunctionEvaluator)
        self.assertEqual(result, '$r = 1')

    def test_iex_in_foreach_pipeline_does_not_crash(self):
        # The InvokeExpression signal is caught rather than escaping the pipeline; the script is
        # left unchanged.
        result = self._deobfuscate("('calc','notepad') | % { iex $_ }")
        self.assertEqual(result, cleandoc("""
            ('calc', 'notepad') | ForEach-Object {
              Invoke-Expression $_
            }
        """))

    def test_recursive_function_does_not_crash(self):
        # Unbounded recursion converts to a graceful bail (no RecursionError); the script is left
        # unchanged.
        result = self._deobfuscate("function f($x){ f $x }; f 1")
        self.assertEqual(result, cleandoc("""
            function f {
              Param($x)
              f $x
            }
            f 1
        """))

    def test_a_redirected_call_is_not_folded_into_its_value(self):
        # Regression: the replacement is an expression and an expression carries no redirections,
        # so folding `f > C:\log` into the string `f` returns printed it to the console and left
        # the file unwritten. Every spelling is refused, including the merges that leave the output
        # stream alone, because none of them survives the substitution either.
        for redirection in (r'> C:\log.txt', r'>> C:\log.txt', '1>&2', '2>&1', r'*> C:\log.txt'):
            with self.subTest(redirection):
                source = F"function f {{ 'FOLDED' }}\n$x = f {redirection}"
                self.assertEqual(self._apply(source, Ps1FunctionEvaluator), cleandoc(F"""
                    function f {{
                      'FOLDED'
                    }}
                    $x = f {redirection}
                """))

    def test_a_body_that_opens_a_file_is_not_folded_into_the_value_it_returns(self):
        # The call site carries no redirection here; the body does. Folding the call deletes the
        # body, so the file the redirection names is never created although the value is right.
        source = cleandoc("""
            function F {
              $a = New-Object byte[] 4 > C:\\o.txt
              'V'
            }
            $x = F
        """)
        self.assertEqual(self._apply(source, Ps1FunctionEvaluator), source)

    def test_a_body_discarding_to_null_is_still_folded(self):
        # `> $Null` is PowerShell's discard and creates nothing, so there is no work the value
        # failed to capture. Reading it as a file would switch the trampoline resolution off for
        # the spelling obfuscators use most.
        source = cleandoc("""
            function F {
              $a = New-Object byte[] 4 > $Null
              'V'
            }
            $x = F
        """)
        self.assertEqual(self._apply(source, Ps1FunctionEvaluator), "$x = 'V'")

    def test_a_call_to_a_function_that_emitted_nothing_goes_with_its_definition(self):
        # The body hands out no code and produces no value, so the call is accounted for with
        # nothing installed in its place, and the definition it named is left uncalled.
        source = cleandoc("""
            function Silent {
              Invoke-Expression ''
            }
            Silent
            Write-Host 'after'
        """)
        self.assertEqual(self._apply(source, Ps1FunctionEvaluator), "Write-Host 'after'")

    def test_an_exported_definition_is_not_removed_after_its_calls_are_folded(self):
        # Regression: folding a call into its value is safe whoever else can reach the name, but
        # deleting the definition afterwards is a name-keyed removal, and an exported name has a
        # caller this tree does not contain. A `.psm1` lost the definition here, and the value it
        # had been folded into was then a bare literal at the root that junk removal stripped as
        # console text — so the module's whole payload went, under the default model, with an export
        # statement standing right there saying it was reachable.
        for export in (
            'Export-ModuleMember -Function f',
            "& 'Microsoft.PowerShell.Core\\Export-ModuleMember' -Function f",
        ):
            with self.subTest(export):
                result = self._apply(
                    F"{export}\nfunction f {{ 'FOLDED' }}\n$x = f", Ps1FunctionEvaluator)
                self.assertIn('function f', result)
                self.assertIn('FOLDED', result)

    def test_a_definition_no_export_names_is_still_removed_after_folding(self):
        # The other three unknowns `is_readable` carries are risks this pass takes deliberately, so
        # gating on the whole verdict would stop it resolving the `iex` trampolines above. Only the
        # export withholds.
        for opener in ('', 'Invoke-Expression $code\n', '& $dispatch\n'):
            with self.subTest(opener or '(closed world)'):
                result = self._apply(
                    F"{opener}function f {{ 'FOLDED' }}\n$x = f", Ps1FunctionEvaluator)
                self.assertNotIn('function f', result)
                self.assertIn("$x = 'FOLDED'", result)


class TestPs1AnArgumentIsFoldedOnlyWhereItsTypeSurvivesTheInterpreter(TestPs1):
    """
    The interpreter's values are Python objects and carry no .NET type, so binding one is a claim
    that the object stands for the value the source wrote. Where it does not, the call is left
    alone: a fold that dropped the type would answer `[byte] 5` with the Int32 5, which is a
    different value in every place the difference shows.
    """

    def _identity_call(self, argument: str) -> str:
        return cleandoc(F"""
            function f {{
              Param($n)
              $n
            }}
            $x = f {argument}
        """)

    def test_a_numeral_written_wider_than_its_magnitude_is_not_bound(self):
        self._assertUnchanged(self._identity_call('1L'), Ps1FunctionEvaluator)
        self.assertEqual(self._apply(self._identity_call('1'), Ps1FunctionEvaluator), '$x = 1')

    def test_a_numeral_written_at_the_width_its_magnitude_takes_is_bound(self):
        self.assertEqual(
            self._apply(self._identity_call('2147483648L'), Ps1FunctionEvaluator),
            '$x = 2147483648L',
        )

    def test_a_width_cast_is_not_bound_as_the_numeral_written_inside_it(self):
        self._assertUnchanged(self._identity_call('([byte]5)'), Ps1FunctionEvaluator)
        self._assertUnchanged(self._identity_call('([uint32]7)'), Ps1FunctionEvaluator)
        self.assertEqual(self._apply(self._identity_call('(5)'), Ps1FunctionEvaluator), '$x = 5')

    def test_a_char_is_not_bound_as_the_string_that_carries_the_same_character(self):
        self._assertUnchanged(self._identity_call('([char]65)'), Ps1FunctionEvaluator)
        self.assertEqual(self._apply(self._identity_call("'A'"), Ps1FunctionEvaluator), "$x = 'A'")

    def test_a_boolean_is_bound_and_handed_back_as_the_boolean_it_is(self):
        self.assertEqual(
            self._apply(self._identity_call('$true'), Ps1FunctionEvaluator), '$x = $True')
        self.assertEqual(self._apply('$true | % { $_ }', Ps1ForEachPipeline), '$True')

    def test_a_pipeline_source_declines_the_same_values_a_parameter_declines(self):
        self._assertUnchanged(cleandoc("""
            [byte]5 | % {
              $_
            }
        """), Ps1ForEachPipeline)
        self._assertUnchanged(cleandoc("""
            ([char]65) | % {
              $_
            }
        """), Ps1ForEachPipeline)
        self.assertEqual(self._apply('5 | % { $_ }', Ps1ForEachPipeline), '5')


class TestPs1AHexadecimalArgumentDenotesThePatternItFills(TestPs1):
    """
    A hexadecimal numeral names a bit pattern in the width its digits fill, not the magnitude the
    digits read as: 5.1 makes `0xFFFFFFFF` the Int32 -1, so a body that adds one to it produces 0
    and never 4294967296.
    """

    def _call(self, argument: str, body: str = '$n') -> str:
        return cleandoc(F"""
            function f ($n) {{ {body} }}
            $x = f {argument}
        """)

    def test_a_pattern_that_fills_an_int32_binds_the_negative_it_denotes(self):
        self.assertEqual(self._apply(self._call('0xFFFFFFFF'), Ps1FunctionEvaluator), '$x = -1')
        self.assertEqual(
            self._apply(self._call('0xFFFFFFFF', '$n + 1'), Ps1FunctionEvaluator), '$x = 0')

    def test_a_pattern_narrower_than_the_width_it_fills_binds_its_magnitude(self):
        self.assertEqual(self._apply(self._call('0xFF'), Ps1FunctionEvaluator), '$x = 255')
        self.assertEqual(
            self._apply(self._call('0x7FFFFFFF', '$n + 1'), Ps1FunctionEvaluator),
            '$x = 2147483648L',
        )

    def test_the_long_suffix_binds_the_digits_read_as_a_number(self):
        self.assertEqual(
            self._apply(self._call('0xFFFFFFFFL'), Ps1FunctionEvaluator), '$x = 4294967295L')

    def test_a_pipeline_reads_a_pattern_the_same_way_a_parameter_does(self):
        self.assertEqual(self._apply('0xFFFFFFFF | % { $_ + 1 }', Ps1ForEachPipeline), '0')


class TestPs1AProducedValueIsWrittenAsTheExpressionThatSpellsIt(TestPs1):

    def _folded(self, body: str, arguments: str) -> str:
        return self._apply(cleandoc(F"""
            function f ($a, $b) {{ {body} }}
            $x = f {arguments}
        """), Ps1FunctionEvaluator)

    def test_a_body_that_compares_folds_to_the_boolean_it_produced(self):
        self.assertEqual(self._folded('$a -eq $b', '1 1'), '$x = $True')
        self.assertEqual(self._folded('$a -eq $b', '1 2'), '$x = $False')

    def test_a_body_that_divides_folds_to_the_fraction_it_produced(self):
        self.assertEqual(self._folded('$a / $b', '3 2'), '$x = 1.5')
        self.assertEqual(self._folded('$a / $b', '4 2'), '$x = 2')

    def test_a_block_writes_each_item_under_the_type_it_produced(self):
        self.assertEqual(
            self._apply('(1, 2) | % { $_ -eq 1 }', Ps1ForEachPipeline), '$True, $False')
        self.assertEqual(self._apply('(3, 2) | % { $_ / 2 }', Ps1ForEachPipeline), '1.5, 1')


class TestPs1AnEmissionThatDidNotHappenIsNotFoldedIntoOne(TestPs1):
    """
    Producing nothing is not producing `$null`: measured on 5.1, `@(g).Count` is 0 for a body that
    emits nothing and 1 for `@($null)`, and `g | %{ }` runs the block no times where `$null | %{ }`
    runs it once. So nothing may stand in the place of an emission that never happened, and the
    call is left where it is.
    """

    def test_a_body_that_emits_nothing_leaves_its_call_alone(self):
        self._assertUnchanged(cleandoc("""
            function f {}
            $x = f
        """), Ps1FunctionEvaluator)
        self.assertEqual(self._apply(cleandoc("""
            function f { 'v' }
            $x = f
        """), Ps1FunctionEvaluator), "$x = 'v'")

    def test_a_block_that_emits_nothing_for_every_item_leaves_its_pipeline_alone(self):
        self._assertUnchanged('(1, 2) | % {}', Ps1ForEachPipeline)

    def test_an_item_that_emitted_nothing_contributes_nothing_to_the_stream(self):
        self.assertEqual(
            self._apply('@(1, 2) | % { if ($_ -eq 1) { $_ } }', Ps1ForEachPipeline), '1')


class TestPs1APipelineOverAScalarRunsItsBlockOverThatOneItem(TestPs1):

    def test_a_number_is_one_item(self):
        self.assertEqual(self._apply('5 | % { $_ + 1 }', Ps1ForEachPipeline), '6')

    def test_a_string_is_one_item_and_not_a_run_of_its_characters(self):
        self.assertEqual(self._apply("'abc' | % { $_.Length }", Ps1ForEachPipeline), '3')
        self.assertEqual(self._apply("'abc' | % { $_ }", Ps1ForEachPipeline), "'abc'")

    def test_the_one_item_still_contributes_everything_its_block_emits(self):
        self.assertEqual(self._apply('5 | % { $_, $_ }', Ps1ForEachPipeline), '5, 5')

    def test_an_array_source_runs_the_block_over_each_of_its_items(self):
        self.assertEqual(self._apply('@(5, 6) | % { $_ + 1 }', Ps1ForEachPipeline), '6, 7')


class TestPs1EmulatorRedirections(TestPs1):

    def test_a_redirected_foreach_pipeline_is_not_folded_into_its_value(self):
        self._assertUnchanged(
            "@('a', 'b') | ForEach-Object {\n  $_\n} > C:\\o.txt", Ps1ForEachPipeline)

    def test_a_merge_on_a_foreach_pipeline_is_refused_too(self):
        self._assertUnchanged(
            "@('a', 'b') | ForEach-Object {\n  $_\n} 2>&1", Ps1ForEachPipeline)

    def test_an_unredirected_foreach_pipeline_is_still_folded(self):
        self.assertEqual(
            self._apply("@('a', 'b') | ForEach-Object { $_ }", Ps1ForEachPipeline), "'a', 'b'")

    def test_a_definition_a_redirected_call_still_names_is_kept(self):
        # The redirected call cannot be folded, so the name still has a caller when the definition
        # removal reads the counter. Emitting the fold without the definition leaves the script
        # calling a function it no longer defines.
        result = self._apply(
            "function F { 'V' }\n$x = F\nF > C:\\o.txt", Ps1FunctionEvaluator)
        self.assertIn('function F', result)
        self.assertIn("$x = 'V'", result)

    def test_a_redirecting_call_to_a_function_that_emitted_nothing_is_kept(self):
        # PowerShell creates the target as it sets the redirection up, so the statement produces the
        # file although the body it names writes nothing into it; deleting it as observing nothing
        # loses the one thing it did. Every spelling is refused, including the merges that create no
        # file, because a merge still moves records the console would otherwise show.
        for redirection in (r'> C:\o.txt', r'>> C:\o.txt', '2>&1', '1>&2', r'*> C:\o.txt'):
            with self.subTest(redirection):
                self._assertUnchanged(cleandoc(F"""
                    function Silent {{
                      Invoke-Expression ''
                    }}
                    Silent {redirection}
                    Write-Host 'after'
                """), Ps1FunctionEvaluator)

    def test_a_value_a_discard_took_away_is_not_read_back(self):
        # `$a = j > $Null` binds `$a` to `$null`, so `$a.Length` is not 4.
        source = cleandoc("""
            function F {
              $a = New-Object byte[] 4 > $Null
              $a.Length
            }
            $x = F
        """)
        self.assertEqual(self._apply(source, Ps1FunctionEvaluator), source)


class TestPs1APipelineSourceIsWhatTheCastAroundItMakesOfIt(TestPs1):
    """
    Measured on 5.1: `[char[]]'ab'` is two Chars where the string it was written around is one item,
    `[int[]]('1', '2')` is two Int32s where the strings inside it would concatenate, and
    `[string[]](1, 2)` is two Strings where the numbers inside it would add. A block run over what
    stands inside the cast therefore runs the wrong number of times or over the wrong values.
    """

    def test_a_cast_that_decides_how_many_items_there_are_is_not_read_past(self):
        self._assertUnchanged(cleandoc("""
            [char[]]'ab' | % {
              $_
            }
        """), Ps1ForEachPipeline)

    def test_a_cast_that_decides_what_the_items_are_worth_is_not_read_past(self):
        for source in [
            cleandoc("""
                [int[]]('1', '2') | % {
                  $_ + 1
                }
            """),
            cleandoc("""
                [string[]](1, 2) | % {
                  $_ + 1
                }
            """),
        ]:
            with self.subTest(source):
                self._assertUnchanged(source, Ps1ForEachPipeline)

    def test_a_cast_the_numbers_written_inside_it_already_answer_is_still_folded(self):
        self.assertEqual(
            self._apply('[Char[]](72, 73) | % { [char]($_ -bxor 0) }', Ps1ForEachPipeline),
            "'H', 'I'",
        )


class TestPs1AnArrayCastAnElementDoesNotFitIsAThrowAndNotARename(TestPs1):
    """
    Measured on 5.1: `[byte[]](300, 1)`, `[byte[]](-1, 1)`, `[char[]](-1)` and `[int[]](2147483648)`
    each raise `Value was either too large or too small`, and so does the pipeline written over
    them, which therefore runs its block no times and produces nothing. Reading the cast as a name
    for the numbers inside it hands back a collection the script never produces.
    """

    def test_a_pipeline_over_a_cast_no_element_of_which_fits_is_left_alone(self):
        for source in [
            cleandoc("""
                [byte[]](300, 1) | % {
                  $_
                }
            """),
            cleandoc("""
                [byte[]](-1, 1) | % {
                  $_
                }
            """),
            cleandoc("""
                [char[]](-1) | % {
                  $_
                }
            """),
            cleandoc("""
                [int[]](2147483648) | % {
                  $_
                }
            """),
        ]:
            with self.subTest(source):
                self._assertUnchanged(source, Ps1ForEachPipeline)

    def test_a_pipeline_over_a_cast_every_element_fits_is_still_folded(self):
        """
        These pin a ledgered erasure, not an exact answer: on 5.1 `[byte[]](5, 6)` carries two
        `System.Byte` and the numerals written back carry `System.Int32`, so what survives the fold
        is the count and the magnitudes and not the element type. It is kept because the same
        reading is what resolves `[Char[]](…) | %{ [char]($_ -bxor $k) }`, the shape real loaders
        are written in, and because the erasure is bounded by the range check the tests above pin.
        """
        self.assertEqual(self._apply('[byte[]](5, 6) | % { $_ }', Ps1ForEachPipeline), '5, 6')
        self.assertEqual(self._apply('[byte[]](255, 1) | % { $_ }', Ps1ForEachPipeline), '255, 1')
        self.assertEqual(
            self._apply('[Char[]](72, 73) | % { [char]($_ -bxor 0) }', Ps1ForEachPipeline),
            "'H', 'I'",
        )


class TestPs1AnArrayCastToATypeThatDoesNotResolveIsWhereTheScriptStops(TestPs1):
    """
    Windows PowerShell 5.1 has no `[short]`, `[ushort]`, `[uint]` or `[ulong]` accelerator — those
    arrived in later versions — so `[ushort[]](1, 2)` is `Unable to find type` and the pipeline
    written over it runs its block no times at all. The integer widths 5.1 does have name a type
    the numbers inside the cast survive, and those are the ones the fold stands on.
    """

    def test_a_pipeline_over_a_cast_5_1_has_no_type_for_is_left_alone(self):
        for spelling in ['short', 'ushort', 'uint', 'ulong']:
            with self.subTest(spelling):
                self._assertUnchanged(cleandoc(F"""
                    [{spelling}[]](1, 2) | % {{
                      $_ + 1
                    }}
                """), Ps1ForEachPipeline)

    def test_a_pipeline_over_a_width_5_1_resolves_is_still_folded(self):
        for spelling in [
            'byte',
            'sbyte',
            'int',
            'int16',
            'int32',
            'int64',
            'long',
            'uint16',
            'uint32',
            'uint64',
            'char',
        ]:
            with self.subTest(spelling):
                self.assertEqual(
                    self._apply(F'[{spelling}[]](1, 2) | % {{ $_ + 1 }}', Ps1ForEachPipeline),
                    '2, 3',
                )


class TestPs1ANullWrittenAmongThePipelineSourceIsOneOfItsItems(TestPs1):
    """
    Measured on 5.1: `1, $null, 2 | % { $_ }` produces three objects, the middle one `$null`. A
    source read as the two numbers it names runs the block twice and hands back a collection one
    item shorter than the one the script produces.
    """

    def test_a_null_between_two_numbers_is_not_dropped_from_the_source(self):
        self._assertUnchanged(cleandoc("""
            1, $null, 2 | % {
              $_
            }
        """), Ps1ForEachPipeline)

    def test_a_source_of_numbers_alone_is_still_folded(self):
        self.assertEqual(self._apply('1, 2 | % { $_ }', Ps1ForEachPipeline), '1, 2')


class TestPs1WhichSideOfWhichOperatorABooleanMayStandOn(TestPs1):
    """
    Measured on 5.1 against the Int32 `2`, for every binary arithmetic and bitwise operator it has,
    with `$true` and `$false` on each side. An operator dispatches to a method on its *left*
    operand's type, and the three that `Boolean` carries none of are `*`, `-shl` and `-shr`: each
    answers `The operation '[System.Boolean] * [System.Int32]' is not defined` for both Booleans, so
    which one stands there decides nothing. The other seven convert it and answer a number for both,
    the Boolean deciding only which number: `$true / 2` is the Double 0.5 where `$false / 2` is the
    Int32 0, since left of a division the Boolean is the dividend. A Boolean on the *right* is
    converted before any dispatch happens and is a value for all ten, except that there `$false` is
    the divisor zero, which makes `2 / $false` and `2 % $false` `Attempted to divide by zero`.
    """

    def _call(self, expression: str, argument: str) -> str:
        return cleandoc(F"""
            function f {{
              Param($n)
              {expression}
            }}
            $x = f {argument}
        """)

    def test_the_operators_a_boolean_left_operand_has_no_method_for_leave_their_call_alone(self):
        for operator in ['*', '-shl', '-shr']:
            for argument in ['$true', '$false']:
                with self.subTest(F'{argument} {operator} 2'):
                    self._assertUnchanged(
                        self._call(F'$n {operator} 2', argument), Ps1FunctionEvaluator)

    def test_the_operators_a_boolean_left_operand_has_a_method_for_fold_to_their_number(self):
        for operator, from_true, from_false in [
            ('+', '3', '2'),
            ('-', '-1', '-2'),
            ('/', '0.5', '0'),
            ('%', '1', '0'),
            ('-band', '0', '0'),
            ('-bor', '3', '2'),
            ('-bxor', '3', '2'),
        ]:
            for argument, expected in [('$true', from_true), ('$false', from_false)]:
                with self.subTest(F'{argument} {operator} 2'):
                    self.assertEqual(
                        self._apply(self._call(F'$n {operator} 2', argument), Ps1FunctionEvaluator),
                        F'$x = {expected}',
                    )

    def test_a_boolean_right_operand_folds_for_every_one_of_the_ten_operators(self):
        for operator, expected in [
            ('+', '3'),
            ('-', '1'),
            ('*', '2'),
            ('/', '2'),
            ('%', '0'),
            ('-band', '0'),
            ('-bor', '3'),
            ('-bxor', '3'),
            ('-shl', '4'),
            ('-shr', '1'),
        ]:
            with self.subTest(operator):
                self.assertEqual(
                    self._apply(self._call(F'2 {operator} $n', '$true'), Ps1FunctionEvaluator),
                    F'$x = {expected}',
                )

    def test_a_false_right_operand_folds_for_every_operator_it_is_not_the_divisor_of(self):
        for operator, expected in [
            ('+', '2'),
            ('-', '2'),
            ('*', '0'),
            ('-band', '0'),
            ('-bor', '2'),
            ('-bxor', '2'),
            ('-shl', '2'),
            ('-shr', '2'),
        ]:
            with self.subTest(F'2 {operator} $false'):
                self.assertEqual(
                    self._apply(self._call(F'2 {operator} $n', '$false'), Ps1FunctionEvaluator),
                    F'$x = {expected}',
                )

    def test_the_false_standing_right_of_a_division_is_the_divisor_zero(self):
        for operator in ['/', '%']:
            with self.subTest(F'2 {operator} $false'):
                self._assertUnchanged(
                    self._call(F'2 {operator} $n', '$false'), Ps1FunctionEvaluator)

    def test_a_block_reads_a_boolean_operand_exactly_as_a_function_body_does(self):
        self.assertEqual(self._apply('$true | % { $_ -bor 2 }', Ps1ForEachPipeline), '3')
        self.assertEqual(self._apply('$false | % { $_ -bor 2 }', Ps1ForEachPipeline), '2')
        self.assertEqual(self._apply('$true | % { $_ / 2 }', Ps1ForEachPipeline), '0.5')
        self.assertEqual(self._apply('$false | % { $_ / 2 }', Ps1ForEachPipeline), '0')
        self.assertEqual(self._apply('$true | % { 2 -shl $_ }', Ps1ForEachPipeline), '4')
        for source in [
            cleandoc("""
                $true | % {
                  $_ -shr 2
                }
            """),
            cleandoc("""
                $false | % {
                  $_ -shr 2
                }
            """),
            cleandoc("""
                $false | % {
                  2 % $_
                }
            """),
        ]:
            with self.subTest(source):
                self._assertUnchanged(source, Ps1ForEachPipeline)

    def test_the_operators_refused_are_the_ones_the_grid_says_always_throw_for_a_boolean(self):
        """
        The list is hand written and the grid is measured, so it is the measurement the list has to
        keep agreeing with: an operator whose Boolean cell stops always throwing, and one that
        becomes so, are both a row of the capture the refusal no longer follows. The whole operator
        axis is asked rather than the ten measured above, so an operator the grid does not cover
        cannot enter the list until a capture covers it either.
        """
        operators = list(data._OPERATORS['binary'])
        self.assertEqual(len(operators), 16)
        self.assertEqual(
            {operator for operator in operators if _boolean_cell(operator, INT32).always_throws},
            set(_NO_OPERATOR_METHOD_ON_BOOLEAN),
        )

    def test_the_boolean_row_reaches_past_the_int32_the_refusal_is_pinned_to(self):
        """
        The refusal is by operand where the grid answers by cell, so the two part company wherever
        the rest of the Boolean row disagrees with the column it was measured against. Every
        disagreement the capture holds is listed: `*` always throws for every right operand but a
        Decimal, `+` and `-` for a Decimal alone, and `/` and `%` add the two whose throw is the
        divisor rather than the Boolean standing left of it.
        """
        operators = list(data._OPERATORS['binary'])
        types = list(data._OPERATORS['witnesses'])
        self.assertEqual((len(operators), len(types)), (16, 16))
        disagreeing = {}
        for operator in operators:
            measured = _boolean_cell(operator, INT32).always_throws
            found = [one for one in types if _boolean_cell(operator, one).always_throws != measured]
            if found:
                disagreeing[operator] = found
        self.assertEqual(disagreeing, {
            '+': ['System.Decimal'],
            '-': ['System.Decimal'],
            '*': ['System.Decimal'],
            '/': ['System.Decimal', 'System.Object[]', 'System.Void'],
            '%': ['System.Decimal', 'System.Object[]', 'System.Void'],
        })


class TestPs1AHexadecimalNumeralInsideABodyDenotesThePatternItFills(TestPs1):
    """
    The same reading as for an argument, measured at the position an emulated body writes it in:
    5.1 answers `0xFFFFFFFF` with the Int32 -1 in a function body and in a `%{ }` block alike, so
    adding one to it is 0 at both, and `0xFFFFFFFFL` is 4294967295 at both.
    """

    def _call(self, body: str) -> str:
        return cleandoc(F"""
            function f {{ {body} }}
            $x = f
        """)

    def test_a_function_body_reads_a_pattern_that_fills_an_int32_as_the_negative_it_names(self):
        self.assertEqual(self._apply(self._call('0xFFFFFFFF'), Ps1FunctionEvaluator), '$x = -1')
        self.assertEqual(self._apply(self._call('0xFFFFFFFF + 1'), Ps1FunctionEvaluator), '$x = 0')

    def test_a_function_body_reads_a_pattern_narrower_than_its_width_as_its_magnitude(self):
        self.assertEqual(self._apply(self._call('0xFF'), Ps1FunctionEvaluator), '$x = 255')

    def test_a_function_body_reads_the_long_suffix_as_the_digits_read_as_a_number(self):
        self.assertEqual(
            self._apply(self._call('0xFFFFFFFFL'), Ps1FunctionEvaluator), '$x = 4294967295L')

    def test_a_block_reads_a_numeral_exactly_as_a_function_body_does(self):
        self.assertEqual(self._apply('1 | % { 0xFFFFFFFF }', Ps1ForEachPipeline), '-1')
        self.assertEqual(self._apply('1 | % { 0xFFFFFFFF + 1 }', Ps1ForEachPipeline), '0')
        self.assertEqual(self._apply('1 | % { 0xFF }', Ps1ForEachPipeline), '255')
        self.assertEqual(self._apply('1 | % { 0xFFFFFFFFL }', Ps1ForEachPipeline), '4294967295L')


class TestPs1ANumeralWithAMultiplierSuffixIsAnIntegerAndNotAFraction(TestPs1):
    """
    Measured on 5.1: `1kb` is the Int32 1024 and `2gb` the Int64 2147483648, a magnitude no Int32
    holds, where `1.5` and `1e3` are Doubles and `1.5d` a Decimal. The parser files every one of
    them as a real literal, so what a body produced has to be written back at the type the numeral
    it was written as has, and not at the one the node is named after.
    """

    def _call(self, body: str) -> str:
        return cleandoc(F"""
            function f {{ {body} }}
            $x = f
        """)

    def test_a_multiplier_suffix_is_written_back_as_the_integer_numeral_it_names(self):
        self.assertEqual(self._apply(self._call('1kb'), Ps1FunctionEvaluator), '$x = 1024')
        self.assertEqual(self._apply(self._call('2gb'), Ps1FunctionEvaluator), '$x = 2147483648L')
        self.assertEqual(self._apply(self._call('1kb + 1'), Ps1FunctionEvaluator), '$x = 1025')

    def test_a_numeral_5_1_reads_as_a_double_is_written_back_as_a_real_numeral(self):
        self.assertEqual(self._apply(self._call('1.5'), Ps1FunctionEvaluator), '$x = 1.5')
        self.assertEqual(self._apply(self._call('1e3'), Ps1FunctionEvaluator), '$x = 1000.0')
        self.assertEqual(self._apply(self._call('2.5kb'), Ps1FunctionEvaluator), '$x = 2560.0')

    def test_a_decimal_is_declined_rather_than_handed_back_as_a_double(self):
        self._assertUnchanged(cleandoc("""
            function f {
              1.5d
            }
            $x = f
        """), Ps1FunctionEvaluator)
        self._assertUnchanged(cleandoc("""
            1 | % {
              1.5d
            }
        """), Ps1ForEachPipeline)

    def test_a_block_reads_a_real_literal_exactly_as_a_function_body_does(self):
        self.assertEqual(self._apply('1 | % { 1kb }', Ps1ForEachPipeline), '1024')
        self.assertEqual(self._apply('1 | % { 2gb }', Ps1ForEachPipeline), '2147483648L')
        self.assertEqual(self._apply('1 | % { 1kb + 1 }', Ps1ForEachPipeline), '1025')
        self.assertEqual(self._apply('1 | % { 1.5 }', Ps1ForEachPipeline), '1.5')
        self.assertEqual(self._apply('1 | % { 1e3 }', Ps1ForEachPipeline), '1000.0')
