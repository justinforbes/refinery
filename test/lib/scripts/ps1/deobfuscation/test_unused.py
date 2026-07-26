from __future__ import annotations

from inspect import cleandoc

from test.lib.scripts.ps1.deobfuscation import TestPs1

from refinery.lib.scripts.ps1.deobfuscation import (
    Ps1DeadStoreElimination,
    Ps1JunkStatementRemoval,
    Ps1UnusedVariableRemoval,
)


class TestPs1UnusedVariableRemoval(TestPs1):

    def test_unused_constant_assignment_removed(self):
        result = self._deobfuscate("$x = 'hello'; Write-Host done")
        self.assertNotIn('$x', result)
        self.assertNotIn('hello', result)
        self.assertIn('done', result)

    def test_multiple_unused_removed(self):
        result = self._deobfuscate("$a = 1; $b = 2; Write-Host done")
        self.assertNotIn('$a', result)
        self.assertNotIn('$b', result)
        self.assertIn('done', result)

    def test_used_variable_kept(self):
        result = self._deobfuscate("$x = 'hello'; Write-Host $x")
        self.assertIn('hello', result)

    def test_side_effect_rhs_preserved(self):
        result = self._deobfuscate("$x = Start-Process notepad; Write-Host done")
        self.assertNotIn('$x', result)
        self.assertIn('Start-Process', result)
        self.assertIn('done', result)

    def test_increment_removed(self):
        result = self._deobfuscate("$x = 0; $x++; Write-Host done")
        self.assertNotIn('$x', result)
        self.assertIn('done', result)

    def test_foreach_variable_preserved(self):
        result = self._deobfuscate(
            "foreach ($item in @(1,2,3)) { Write-Host 'hi' }")
        self.assertIn('foreach', result.lower())

    def test_scoped_variable_preserved(self):
        result = self._deobfuscate("$script:x = 42; Write-Host done")
        self.assertIn('$script:x', result)

    def test_parameter_preserved(self):
        result = self._deobfuscate(
            "function Test { Param($x); Write-Host done }; Test")
        self.assertIn('$x', result)

    def test_compound_assignment_removed(self):
        result = self._deobfuscate("$x = 0; $x += 1; Write-Host done")
        self.assertNotIn('$x', result)
        self.assertIn('done', result)

    def test_self_referential_folded(self):
        result = self._deobfuscate("$x = 0; $x = $x + 1; Write-Host $x")
        self.assertNotIn('$x', result)
        self.assertIn('1', result)


class TestPs1JunkStatementRemoval(TestPs1):

    def test_void_cast_removed(self):
        result = self._deobfuscate('[Void]([Math]::Sqrt(144)); Write-Host done')
        self.assertNotIn('Sqrt', result)
        self.assertIn('done', result)

    def test_void_cast_around_a_call_preserved(self):
        # A `[Void]` wrapper discards the value of the call, not the call. Treating the cast alone
        # as a discard deleted the payload out of a script whose whole point is to reveal it.
        result = self._deobfuscate('[Void](Start-Process notepad); Write-Host done')
        self.assertIn('Start-Process', result)
        self.assertIn('done', result)

    def test_void_cast_around_a_call_preserved_inside_a_function(self):
        result = self._deobfuscate(
            'function f { Write-Host hi; [Void](Remove-Item C:\\important) }\nf')
        self.assertIn('Remove-Item', result)

    def test_a_void_cast_does_not_stand_in_for_the_value_it_replaced(self):
        # Keeping the wrapped call must not cost the body its return value: a discard emits nothing,
        # so it cannot be the survivor that makes dropping the real output safe.
        result = self._deobfuscate(
            "function f { [Void](Remove-Item C:\\important); 'payload' }\nf")
        self.assertIn('Remove-Item', result)
        self.assertIn('payload', result)

    def test_an_advanced_function_is_not_junk(self):
        # A `process` block is a body like any other; reading the empty unnamed body as an inert
        # function deleted the definition and every call to it.
        result = self._deobfuscate(
            'function f { process { Start-Process notepad } }\nf\nWrite-Host done')
        self.assertIn('Start-Process', result)
        self.assertIn('function f', result)
        self.assertIn('done', result)

    def test_a_member_invoking_foreach_is_not_a_discard(self):
        result = self._deobfuscate(
            'Get-Process | ForEach-Object -MemberName Kill\nWrite-Host done')
        self.assertIn('Kill', result)
        self.assertIn('done', result)

    def test_a_member_invoking_foreach_is_not_excused_by_a_discarding_block(self):
        # `-MemberName Kill` runs on every input item whatever the block beside it does with `$_`,
        # so a visible discard may not vouch for it.
        for source in (
            'Get-Process | ForEach-Object { [Void]$_ } -MemberName Kill\nWrite-Host done',
            'Get-Process | ForEach-Object { [Void]$_ } Kill\nWrite-Host done',
            'Get-Process | ForEach-Object { $Null = $_ } -MemberName Kill\nWrite-Host done',
        ):
            with self.subTest(source):
                result = self._deobfuscate(source)
                self.assertIn('Kill', result)
                self.assertIn('done', result)

    def test_a_member_named_by_a_here_string_is_still_a_member(self):
        # The member name is a string whichever way it is quoted, and a table of literal forms that
        # names only the single-quoted one hands the here-string spelling back to the junk pass.
        for source in (
            "Get-ChildItem C:\\ | ForEach-Object { $Null = $_ } -MemberName @'\nDelete\n'@",
            "Get-Process | ForEach-Object { [Void]$_ } @'\nKill\n'@",
        ):
            with self.subTest(source):
                self.assertIn('ForEach-Object', self._deobfuscate(F'{source}\nWrite-Host done'))

    def test_a_foreach_body_hidden_in_a_named_or_param_block_is_kept(self):
        # The parser fills either `body` or the named blocks, so a block that carries its work in
        # `end { ... }` or in a parameter default reports an empty statement list. Reading that as
        # "this body discards everything" deleted the recursive removal along with the statement.
        for source in (
            'Get-ChildItem C:\\ | ForEach-Object { end { Remove-Item C:\\important -Recurse } }',
            '1..3 | ForEach-Object { begin { Start-Process notepad } process { [Void]$_ } }',
            '1..3 | ForEach-Object -Process { param($p = (Start-Process notepad)) [Void]$_ }',
        ):
            with self.subTest(source):
                self.assertIn('ForEach-Object', self._deobfuscate(F'{source}\nWrite-Host done'))

    def test_a_computed_member_name_keeps_the_statement_that_computes_it(self):
        result = self._deobfuscate('if ($true) { $x.$(Start-Process notepad) }\nWrite-Host done')
        self.assertIn('Start-Process', result)
        self.assertIn('done', result)

    def test_a_hash_literal_key_that_runs_a_command_is_kept(self):
        result = self._deobfuscate('if ($true) { @{ $(Start-Process notepad) = 1 } }\nWrite-Host x')
        self.assertIn('Start-Process', result)

    def test_a_junk_function_stays_removable_when_it_declares_parameters(self):
        # A bare parameter declaration evaluates nothing, so the pruned body is still inert.
        result = self._deobfuscate('function j($x) { $Null = 915 }\nj\nWrite-Host done')
        self.assertNotIn('function', result)
        self.assertIn('done', result)

    def test_a_foreach_block_held_in_a_variable_is_kept(self):
        # `-End $sb` runs whatever scriptblock the variable holds; nothing in the source says what,
        # so the discarding process block proves nothing about the statement as a whole.
        for source in (
            'Param($sb)\n1..3 | ForEach-Object -Process { [Void]$_ } -End $sb\nWrite-Host done',
            '1..3 | ForEach-Object { [Void]$_ } $Global:sb\nWrite-Host done',
        ):
            with self.subTest(source):
                self.assertIn('ForEach-Object', self._deobfuscate(source))

    def test_a_parameter_default_that_runs_a_command_keeps_the_function(self):
        # The default is evaluated on every call that omits the argument, so neither the definition
        # nor the call site is a no-op.
        result = self._deobfuscate(
            'function f { param($x = (Start-Process notepad)) }\nf\nWrite-Host done')
        self.assertIn('Start-Process', result)
        self.assertIn('done', result)

    def test_a_splatted_argument_keeps_the_command(self):
        # `@options` supplies parameters that are nowhere in the source, `-OutVariable` among them.
        result = self._deobfuscate('Get-Date @options\nWrite-Host $d')
        self.assertIn('Get-Date', result)

    def test_a_discard_terminator_may_not_hide_a_call_in_its_arguments(self):
        result = self._deobfuscate(
            '1 | Out-Null -InputObject (Start-Process notepad)\nWrite-Host done')
        self.assertIn('Start-Process', result)
        self.assertIn('done', result)

    def test_a_redirected_command_is_kept(self):
        result = self._deobfuscate('Get-Date > C:\\out.txt\nWrite-Host done')
        self.assertIn('out.txt', result)
        self.assertIn('done', result)

    def test_an_out_parameter_call_is_kept(self):
        # The statement is what fills `$n`; deleting it leaves the read below with nothing.
        result = self._deobfuscate('[Int]::TryParse($s, [ref]$n)\nWrite-Host $n')
        self.assertIn('TryParse', result)

    def test_a_temporary_file_creation_is_kept(self):
        result = self._deobfuscate('[IO.Path]::GetTempFileName()\nWrite-Host done')
        self.assertIn('GetTempFileName', result)
        self.assertIn('done', result)

    def test_an_array_mutation_on_a_live_variable_is_kept(self):
        result = self._deobfuscate('[Array]::Reverse($buffer)\nWrite-Host done')
        self.assertIn('Reverse', result)
        self.assertIn('done', result)

    def test_a_command_that_fills_a_variable_is_kept(self):
        # `-OutVariable d` is what sets `$d`. The parsed parameter name keeps its leading dash and
        # PowerShell binds abbreviations, so a name table consulted by exact match matches nothing.
        for source, marker in (
            ('Get-Date -OutVariable d\nWrite-Host $d', 'OutVariable'),
            ('Get-Date -OutVar d\nWrite-Host $d', 'OutVar'),
            ('Get-Date -ov d\nWrite-Host $d', '-ov'),
            ('Get-ChildItem -ErrorVariable e\nWrite-Host $e', 'ErrorVariable'),
            ('Get-Date | Out-Null -ErrorVariable e\nWrite-Host $e', 'ErrorVariable'),
            ('Get-Random -SetSeed 5\nWrite-Host done', 'SetSeed'),
        ):
            with self.subTest(source):
                self.assertIn(marker, self._deobfuscate(source))

    def test_a_constructor_argument_that_runs_a_command_is_kept(self):
        result = self._deobfuscate(
            "New-Object String 'x' (Start-Process notepad)\nWrite-Host done")
        self.assertIn('Start-Process', result)
        self.assertIn('done', result)

    def test_a_data_section_does_not_stand_in_for_a_return_value(self):
        # `data d { 42 }` binds `$d` and emits nothing, so it cannot be the survivor that makes
        # dropping the function's real output safe.
        result = self._deobfuscate("function f { data d { 42 }; 'payload' }\nf")
        self.assertIn('payload', result)

    def test_out_null_pipeline_removed(self):
        result = self._deobfuscate('[Math]::Pow(2, 8) | Out-Null; Write-Host done')
        self.assertNotIn('Pow', result)
        self.assertIn('done', result)

    def test_pure_static_method_removed(self):
        result = self._deobfuscate('[Math]::Sqrt(36); Write-Host done')
        self.assertNotIn('Sqrt', result)
        self.assertIn('done', result)

    def test_pure_cmdlet_removed(self):
        result = self._deobfuscate('Get-Random -Minimum 1 -Maximum 100; Write-Host done')
        self.assertNotIn('Get-Random', result)
        self.assertIn('done', result)

    def test_pure_instance_method_removed(self):
        result = self._deobfuscate('(Get-Date).ToString("yyyy"); Write-Host done')
        self.assertNotIn('ToString', result)
        self.assertIn('done', result)

    def test_side_effect_command_preserved(self):
        result = self._deobfuscate('Start-Sleep -s 1; Write-Host done')
        self.assertIn('Start-Sleep', result)
        self.assertIn('done', result)

    def test_uncalled_function_removed(self):
        result = self._deobfuscate(
            'function Junk { Get-Random }; Write-Host done')
        self.assertNotIn('Junk', result)
        self.assertIn('done', result)

    def test_called_function_preserved(self):
        result = self._deobfuscate(
            'function Helper { Get-Random }; Helper; Write-Host done')
        self.assertIn('Helper', result)

    def test_expandable_string_removed(self):
        result = self._deobfuscate('"noise ${x} text"; Write-Host done')
        self.assertNotIn('noise', result)
        self.assertIn('done', result)

    def test_string_literal_removed(self):
        result = self._deobfuscate("'junk string'; Write-Host done")
        self.assertNotIn('junk', result)
        self.assertIn('done', result)

    def test_pure_pipeline_removed(self):
        result = self._deobfuscate(
            'Get-Date | Out-String; Write-Host done')
        self.assertNotIn('Get-Date', result)
        self.assertIn('done', result)

    def test_empty_body_guard(self):
        result = self._deobfuscate('[Math]::Sqrt(36)')
        self.assertIn('Sqrt', result)

    def test_nested_body_junk_removed(self):
        result = self._deobfuscate(
            'while ($True) { [Void]"noise"; Write-Host running; break }')
        self.assertNotIn('noise', result)
        self.assertIn('running', result)

    def test_subexpression_body_preserved(self):
        result = self._deobfuscate("$x = $($a.Name + '.' + $a.Extension)")
        self.assertIn('.Name', result)
        self.assertIn('.Extension', result)

    def test_scriptblock_body_preserved(self):
        result = self._deobfuscate('1,2,3 | ForEach-Object { $_.ToString() }')
        self.assertIn('.ToString()', result)

    def test_transitive_function_calls_preserved(self):
        result = self._deobfuscate(
            'function Inner { Get-Date }\n'
            'function Outer { Inner }\n'
            'Outer\n'
        )
        self.assertIn('Inner', result)
        self.assertIn('Get-Date', result)


class TestPs1UnusedExtra(TestPs1):

    def test_junk_removal_keeps_indirectly_called_function(self):
        # A function reachable only through the call operator on a variable (`& $f`) must survive,
        # because the dynamic target cannot be proven different from it.
        result = self._apply(
            "function Invoke-Payload { Write-Host 'x' }\n& $f", Ps1JunkStatementRemoval)
        self.assertIn('Invoke-Payload', result)

    def test_junk_removal_keeps_collection_mutation(self):
        # `.Remove` mutates a collection in place, so a discarded `$list.Remove(...)` is not junk.
        result = self._apply(
            "$list = [System.Collections.ArrayList]@(1, 2, 3)\n$list.Remove(2)",
            Ps1JunkStatementRemoval)
        self.assertIn('.Remove(2)', result)

    def test_unused_variable_read_in_function_is_kept(self):
        # The read inside Run keeps the assignment alive (PowerShell dynamic scoping).
        result = self._apply(
            "$x = 'payload'; function Run { iex $x }; Run", Ps1UnusedVariableRemoval)
        self.assertEqual(result, cleandoc("""
            $x = 'payload'
            function Run {
              iex $x
            }
            Run
        """))

    def test_unused_variable_scoped_read_is_kept(self):
        # The $script:x read keeps the $x assignment alive.
        result = self._apply(
            "$x = 'keepme'; function f { Write-Host $script:x }; f", Ps1UnusedVariableRemoval)
        self.assertEqual(result, cleandoc("""
            $x = 'keepme'
            function f {
              Write-Host $script:x
            }
            f
        """))

    def test_dead_multiassign_all_targets_removed(self):
        # Every target of `$a, $b, $c = 1, 2, 3` is unread, so the whole multi-assignment is dead.
        result = self._apply(
            "$a, $b, $c = 1, 2, 3\nWrite-Host 'keep'", Ps1UnusedVariableRemoval)
        self.assertEqual(result, "Write-Host 'keep'")

    def test_live_multiassign_cotarget_keeps_statement(self):
        # `$b` is read, so the multi-assignment survives even though its co-target `$a` is dead.
        result = self._apply(
            '$a, $b = 1, 2\nWrite-Output $b', Ps1UnusedVariableRemoval)
        self.assertEqual(result, '$a, $b = 1, 2\nWrite-Output $b')

    def test_pure_new_object_dead_store_removed(self):
        # `New-Object System.Object` has no side effect, so an unread store of it is removable.
        result = self._apply(
            "$x = New-Object System.Object\nWrite-Host 'keep'", Ps1UnusedVariableRemoval)
        self.assertEqual(result, "Write-Host 'keep'")

    def test_impure_new_object_store_kept(self):
        # `New-Object System.Net.WebClient` is not proven pure, so its RHS must be preserved — as a
        # discard, not as a bare statement. The construction returns the client, which the dead
        # store swallowed and a bare expression would write to the output stream instead.
        result = self._apply(
            "$x = New-Object System.Net.WebClient\nWrite-Host 'keep'", Ps1UnusedVariableRemoval)
        self.assertEqual(result, "$Null = New-Object System.Net.WebClient\nWrite-Host 'keep'")

    def test_dropping_a_dead_store_does_not_start_emitting_its_value(self):
        # A bare expression statement writes its value to the output stream; the assignment being
        # removed swallowed it. Rewriting the store to the value alone therefore made the
        # deobfuscated script print what the sample never printed, and inside a function body it
        # changed the return value — a deobfuscator has to preserve what the script does.
        for source in (
            "$unused = [Loader]'payload'",
            '$unused = New-Object System.Net.WebClient',
        ):
            with self.subTest(source):
                result = self._apply(
                    F"{source}\nWrite-Host 'keep'", Ps1UnusedVariableRemoval)
                self.assertTrue(result.startswith('$Null = '), result)

    def test_null_discard_pure_removed(self):
        # `$null = <pure>` is PowerShell's discard idiom; with a side-effect-free RHS it is junk.
        result = self._apply(
            "$null = [Environment]::UserName\nWrite-Host 'keep'", Ps1JunkStatementRemoval)
        self.assertEqual(result, "Write-Host 'keep'")

    def test_null_discard_side_effect_kept(self):
        # A `$null =` discard whose RHS is a command call has a side effect and must be preserved.
        result = self._apply(
            "$null = Remove-Item C:\\x\nWrite-Host 'keep'", Ps1JunkStatementRemoval)
        self.assertEqual(result, "$null = Remove-Item C:\\x\nWrite-Host 'keep'")


class TestPs1InertFunctionRemoval(TestPs1):

    def test_inert_function_and_call_removed(self):
        result = self._apply(
            "function j { $Null = 915 }\nj\nWrite-Host 'keep'", Ps1JunkStatementRemoval)
        self.assertEqual(result, "Write-Host 'keep'")

    def test_inert_function_multiple_calls_removed(self):
        result = self._apply(
            "function j { $Null = 915 }\nj\nj\nj\nWrite-Host 'keep'", Ps1JunkStatementRemoval)
        self.assertEqual(result, "Write-Host 'keep'")

    def test_emitting_function_kept(self):
        result = self._apply(
            "function f { 42 }\nf\nWrite-Host 'keep'", Ps1JunkStatementRemoval)
        self.assertEqual(result, "function f {\n  42\n}\nf\nWrite-Host 'keep'")

    def test_effectful_function_kept(self):
        result = self._apply(
            "function f { Write-Host 'real' }\nf", Ps1JunkStatementRemoval)
        self.assertEqual(result, "function f {\n  Write-Host 'real'\n}\nf")

    def test_dynamic_dispatch_preserves_all_functions(self):
        result = self._apply(
            "function j { $Null = 1 }\nj\n& $name\nWrite-Host 'keep'",
            Ps1JunkStatementRemoval)
        self.assertEqual(result, "function j {}\nj\n& $name\nWrite-Host 'keep'")

    def test_function_with_argful_call_kept(self):
        result = self._apply(
            "function j { $Null = 1 }\nj 'arg'\nWrite-Host 'keep'", Ps1JunkStatementRemoval)
        self.assertEqual(result, "function j {}\nj 'arg'\nWrite-Host 'keep'")

    def test_function_captured_result_kept(self):
        result = self._apply(
            "function j { $Null = 1 }\n$x = j\nWrite-Host 'keep'", Ps1JunkStatementRemoval)
        self.assertEqual(result, "function j {}\n$x = j\nWrite-Host 'keep'")

    def test_a_scope_qualified_definition_is_reached_by_an_unqualified_call(self):
        # `function global:f` defines what a later bare `f` runs. Keying the definition under the
        # qualified spelling left the callgraph unable to enter the body, so nothing inside it
        # counted as reachable and the definition itself read as never called.
        for definition in ('global:f', 'script:f', 'local:f', 'private:f'):
            with self.subTest(definition):
                result = self._apply(
                    F"function {definition} {{ Start-Process calc }}\nf\nWrite-Host 'keep'",
                    Ps1JunkStatementRemoval)
                self.assertIn('Start-Process', result)

    def test_param_block_function_module_preserved(self):
        result = self._apply(cleandoc("""
            function Ge {
              [CmdletBinding()]
              param (
                [parameter(ValueFromPipeline=$true)]
                $frk=$env:ComputerName
              )
              Write-Host $frk
            }
        """), Ps1JunkStatementRemoval)
        self.assertIn('function Ge', result)
        self.assertIn('Write-Host', result)

class TestPs1DiscardedObjectRemoval(TestPs1):

    def test_bare_hash_literal_removed(self):
        result = self._apply(
            "@{ a = 1; b = 2 }\nWrite-Host 'keep'", Ps1JunkStatementRemoval)
        self.assertEqual(result, "Write-Host 'keep'")

    def test_pscustomobject_hash_removed(self):
        result = self._apply(
            "[pscustomobject]@{ Name = 'x'; Value = 42 }\nWrite-Host 'keep'",
            Ps1JunkStatementRemoval)
        self.assertEqual(result, "Write-Host 'keep'")

    def test_synchronized_hashtable_removed(self):
        result = self._apply(
            "[Collections.Hashtable]::Synchronized(@{})\nWrite-Host 'keep'",
            Ps1JunkStatementRemoval)
        self.assertEqual(result, "Write-Host 'keep'")

    def test_void_foreach_pipeline_removed(self):
        result = self._apply(
            "(1, 2, 3) | ForEach-Object { [Void]$_ }\nWrite-Host 'keep'",
            Ps1JunkStatementRemoval)
        self.assertEqual(result, "Write-Host 'keep'")

    def test_null_assign_foreach_pipeline_removed(self):
        result = self._apply(
            "(1, 2, 3) | ForEach-Object { $Null = $_ }\nWrite-Host 'keep'",
            Ps1JunkStatementRemoval)
        self.assertEqual(result, "Write-Host 'keep'")

    def test_hash_with_impure_value_kept(self):
        result = self._apply(
            "@{ x = (Start-Process notepad) }", Ps1JunkStatementRemoval)
        self.assertEqual(result, '@{\n  x = (Start-Process notepad)\n}')

    def test_emitting_foreach_kept(self):
        result = self._apply(
            "(1, 2, 3) | ForEach-Object { $_ }", Ps1JunkStatementRemoval)
        self.assertEqual(result, '(1, 2, 3) | ForEach-Object {\n  $_\n}')

    def test_null_assign_foreach_side_effect_kept(self):
        result = self._apply(
            "(1, 2, 3) | ForEach-Object { $Null = Remove-Item $_ }",
            Ps1JunkStatementRemoval)
        self.assertEqual(result, '(1, 2, 3) | ForEach-Object {\n  $Null = Remove-Item $_\n}')


class TestPs1DeadStoreElimination(TestPs1):

    def test_overwritten_store_removed(self):
        result = self._apply("$x = 1\n$x = 2\nWrite-Host $x", Ps1DeadStoreElimination)
        self.assertEqual(result, '$x = 2\nWrite-Host $x')

    def test_chain_all_but_last_removed(self):
        result = self._apply("$x = 1\n$x = 2\n$x = 3\nWrite-Host $x", Ps1DeadStoreElimination)
        self.assertEqual(result, '$x = 3\nWrite-Host $x')

    def test_dead_store_before_for_removed(self):
        result = self._apply(
            "$i = 33\nfor ($i = 0; $i -LT 5; $i++) { Write-Host $i }",
            Ps1DeadStoreElimination)
        self.assertNotIn('$i = 33', result)
        self.assertIn('for', result)

    def test_multiple_dead_stores_before_for_removed(self):
        result = self._apply(
            "$i = 33\n$i = 44\n$i = 55\nfor ($i = 0; $i -LT 5; $i++) { Write-Host $i }",
            Ps1DeadStoreElimination)
        self.assertNotIn('$i = 33', result)
        self.assertNotIn('$i = 44', result)
        self.assertNotIn('$i = 55', result)
        self.assertIn('for', result)

    def test_intervening_read_keeps_store(self):
        result = self._apply(
            "$x = 1\nWrite-Host $x\n$x = 2\nWrite-Host $x", Ps1DeadStoreElimination)
        self.assertIn('$x = 1', result)
        self.assertIn('$x = 2', result)

    def test_impure_rhs_preserved_as_standalone(self):
        result = self._apply(
            "$x = Remove-Item foo\n$x = 5\nWrite-Host $x", Ps1DeadStoreElimination)
        self.assertIn('Remove-Item', result)
        self.assertNotIn('$x = Remove-Item', result)
        self.assertIn('$x = 5', result)

    def test_different_variables_independent(self):
        result = self._apply(
            "$x = 1\n$y = 2\n$x = 3\nWrite-Host $x $y", Ps1DeadStoreElimination)
        self.assertNotIn('$x = 1', result)
        self.assertIn('$y = 2', result)
        self.assertIn('$x = 3', result)

    def test_scoped_variable_not_killed(self):
        result = self._apply(
            "$script:x = 1\n$script:x = 2\nWrite-Host $script:x", Ps1DeadStoreElimination)
        self.assertEqual(result, '$script:x = 1\n$script:x = 2\nWrite-Host $script:x')

    def test_control_flow_flushes_pending(self):
        result = self._apply(
            "$x = 1\nif ($c) { Write-Host $x }\n$x = 2\nWrite-Host $x",
            Ps1DeadStoreElimination)
        self.assertEqual(result, '$x = 1\nif ($c) {\n  Write-Host $x\n}\n$x = 2\nWrite-Host $x')

    def test_dead_store_inside_nested_function_removed(self):
        result = self._apply(
            'function f { $i = 5\n$i = 3\nWrite-Host $i }',
            Ps1DeadStoreElimination)
        self.assertEqual(result, 'function f {\n  $i = 3\n  Write-Host $i\n}')

    def test_dead_store_scriptblock_local_does_not_flush_outer(self):
        result = self._apply(cleandoc(
            """
            $inner = 1
            $cb = {
              $inner = 99
            }
            $inner = 2
            Write-Host $inner
            """
        ), Ps1DeadStoreElimination)
        self.assertEqual(result, '$cb = {\n  $inner = 99\n}\n$inner = 2\nWrite-Host $inner')

    def test_dead_store_read_in_captured_scriptblock_is_kept(self):
        # Regression: $x = 1 is read only through a captured (stored, never invoked) scriptblock
        # nested inside an array expression. The block could observe that value when later invoked,
        # so the store is live and must not be removed. The former per-body walk skipped nested
        # scriptblocks and deleted it; sourcing reads from the shared model closes that path.
        result = self._apply(cleandoc(
            """
            $x = 1
            $arr = @( { Write-Host $x } )
            $x = 2
            Write-Host $x
            """
        ), Ps1DeadStoreElimination)
        self.assertEqual(
            result, '$x = 1\n$arr = @({\n  Write-Host $x\n})\n$x = 2\nWrite-Host $x')

    def test_dead_store_read_by_compound_assignment_is_kept(self):
        # Regression: the target of `$x += 1` reads the variable before writing it, so the store it
        # observes is live. Excluding every assignment target from the read set — rather than only
        # the target of a plain `=`, which replaces the value unread — deleted `$x = 1`.
        result = self._apply(cleandoc(
            """
            $x = 5
            $a = ($x += 1)
            $x = 7
            Write-Host $x
            Write-Host $a
            """
        ), Ps1DeadStoreElimination)
        self.assertEqual(
            result, '$x = 5\n$a = ($x += 1)\n$x = 7\nWrite-Host $x\nWrite-Host $a')


class TestPs1PayloadRetention(TestPs1):
    """
    Every case here is a shape where a removal decision was taken over a name or a node that stood
    for more code than the pass could see. Keeping junk costs nothing; deleting a call that runs is
    the failure this suite exists to catch, so each test asserts the payload survives rather than
    asserting an exact rendering.
    """

    def test_a_nested_redefinition_keeps_the_name_from_being_inert(self):
        # Regression: `acting` was collected from top-level statements while call sites were
        # collected tree-wide, so the empty definition made the name inert and the call that reaches
        # the payload definition — an `if` body is not a new scope in PowerShell — was removed.
        result = self._deobfuscate(cleandoc(
            """
            function j { }
            if ($env:c) { function j { Start-Process calc } }
            j
            """
        ))
        self.assertIn('Start-Process calc', result)
        self.assertRegex(result, r'(?m)^j$')

    def test_a_payload_definition_inside_a_function_keeps_its_call(self):
        result = self._deobfuscate(cleandoc(
            """
            function Outer { function j { Start-Process calc }; j }
            function j { }
            Outer
            """
        ))
        self.assertIn('Start-Process calc', result)
        self.assertIn('Outer', result)

    def test_a_statement_valued_dead_assignment_is_kept_whole(self):
        # Regression: an assignment whose value is a statement (`$x = if (...) { ... }`, and the
        # switch/foreach/try forms beside it) has no expression statement to be rewritten into, and
        # the branch that could not rewrite it deleted the statement along with the branch bodies.
        for value in (
            'if ($env:c) { Start-Process calc }',
            'switch ($env:c) { 1 { Start-Process calc } }',
            'foreach ($i in 1..2) { Start-Process calc }',
        ):
            with self.subTest(value):
                self.assertIn(
                    'Start-Process calc', self._deobfuscate(F'$x = {value}\nWrite-Host done'))
                self.assertIn(
                    'Start-Process calc',
                    self._deobfuscate(F'$x = {value}\n$x = 1\nWrite-Host $x'))

    def test_a_scope_qualified_local_function_is_not_alias_inlined(self):
        # Regression: local definitions were keyed by their written spelling, so `function
        # global:gci` did not shield the call `gci`, which was rewritten to `Get-ChildItem` and then
        # deleted as a pure cmdlet — taking the definition with it.
        result = self._deobfuscate(cleandoc(
            """
            function global:gci { Start-Process calc }
            gci
            Write-Host done
            """
        ))
        self.assertIn('Start-Process calc', result)
        self.assertNotIn('Get-ChildItem', result)


class TestPs1NameRemovalNeedsTheWholeStory(TestPs1):
    """
    Both name-keyed removals in this module reason from the definitions and calls standing in the
    tree. That is the whole story only while the tree is: an open world holds definitions and calls
    in a file the walk never read, and an identity-namespace assignment binds a name by a spelling
    neither scan recognizes. Each test asserts the payload path survives, never an exact rendering.
    """

    def test_an_open_world_keeps_the_call_to_an_apparently_inert_function(self):
        # Regression: the empty `j` standing here is not the `j` the dot-sourced file defines, so
        # the call reaches code this tree does not contain.
        for opener in (". '.\\stage2.ps1'", 'Invoke-Expression $code', 'Import-Module .\\m.psm1'):
            with self.subTest(opener):
                result = self._deobfuscate(cleandoc(
                    F"""
                    function j {{ }}
                    {opener}
                    j
                    Write-Host 'keep'
                    """
                ))
                self.assertRegex(result, r'(?m)^j$')

    def test_an_open_world_keeps_a_function_nothing_in_the_tree_calls(self):
        # Regression: the `iex` is exactly what calls it, and its call site is not in this tree.
        result = self._deobfuscate(cleandoc(
            """
            function Payload { Start-Process calc }
            Invoke-Expression $code
            Write-Host 'keep'
            """
        ))
        self.assertIn('Start-Process calc', result)

    def test_an_identity_assignment_keeps_the_names_it_could_bind(self):
        # `${function:j} = { ... }` is a definition of `j` the definition scan does not read, and
        # `${alias:q} = 'j'` is a call to `j` the call scan does not read.
        rebound = self._deobfuscate(cleandoc(
            """
            function j { }
            ${function:j} = { Start-Process calc }
            j
            Write-Host done
            """
        ))
        self.assertIn('Start-Process calc', rebound)
        self.assertRegex(rebound, r'(?m)^j$')
        aliased = self._deobfuscate(cleandoc(
            """
            function j { Start-Process calc }
            ${alias:q} = 'j'
            q
            Write-Host done
            """
        ))
        self.assertIn('Start-Process calc', aliased)

    def test_a_closed_world_still_prunes_an_inert_function_and_its_calls(self):
        result = self._deobfuscate(cleandoc(
            """
            function j { $Null = 915 }
            j
            Write-Host 'keep'
            """
        ))
        self.assertEqual(result, "Write-Host 'keep'")
