from __future__ import annotations

from test.lib.scripts.ps1.deobfuscation import TestPs1

from refinery.lib.scripts.ps1.deobfuscation import Ps1WildcardResolution


class TestPs1VariableDriveResolution(TestPs1):

    def test_get_item_variable_value_resolved(self):
        result = self._deobfuscate("(Get-Item 'Variable:E*t').Value.InvokeCommand")
        self.assertEqual(result.strip(), '$ExecutionContext.InvokeCommand')

    def test_get_variable_value_resolved(self):
        result = self._deobfuscate('(Get-Variable ExecutionContext).Value')
        self.assertEqual(result.strip(), '$ExecutionContext')

    def test_get_item_variable_without_value_preserved(self):
        result = self._deobfuscate("(Get-Item 'Variable:E*t')")
        self.assertNotIn('$ExecutionContext', result)

    def test_member_alias_resolved(self):
        result = self._deobfuscate('$x | Member')
        self.assertIn('Get-Member', result)

    def test_variable_alias_resolved(self):
        result = self._deobfuscate('Variable ExecutionContext')
        self.assertIn('Get-Variable', result)

    def test_variable_drive_path_separator_stripped(self):
        result = self._deobfuscate('(Get-Item Variable:/hb).Value')
        self.assertIn('$hb', result)
        self.assertNotIn('/', result)

    def test_set_item_variable_becomes_assignment(self):
        result = self._deobfuscate("Set-Item Variable:/G7E 'hello'")
        self.assertEqual(result.strip(), "$G7E = 'hello'")

    def test_set_item_variable_multi_value(self):
        result = self._deobfuscate(
            "Set-Item Variable:/G7E $env:Temp '\\NGLClient.exe'"
        )
        self.assertIn('$G7E', result)
        self.assertIn('=', result)
        self.assertIn('env:Temp', result)
        self.assertIn('NGLClient', result)

    def test_set_variable_becomes_assignment(self):
        result = self._deobfuscate("Set-Variable foo 42")
        self.assertEqual(result.strip(), '$foo = 42')

    def test_set_variable_named_params(self):
        result = self._deobfuscate("Set-Variable -Name foo -Value 'bar'")
        self.assertEqual(result.strip(), "$foo = 'bar'")

    def test_set_variable_with_integer_name(self):
        result = self._deobfuscate("Set-Variable 0 'hello'\n$0")
        self.assertIn('hello', result)
        self.assertNotIn('Set-Variable', result)

    def test_get_variable_value_only_resolved(self):
        result = self._deobfuscate('Get-Variable ExecutionContext -ValueOnly')
        self.assertEqual(result.strip(), '$ExecutionContext')

    def test_get_variable_value_only_abbreviated(self):
        result = self._deobfuscate('Get-Variable Cf -ValueO')
        self.assertEqual(result.strip(), '$Cf')

    def test_get_variable_value_abbreviated_short(self):
        for switch in ('-V', '-Va', '-Val', '-Valu', '-Value', '-ValueO', '-ValueOn', '-ValueOnl', '-ValueOnly'):
            with self.subTest(switch=switch):
                result = self._deobfuscate(F'Get-Variable Cf {switch}')
                self.assertEqual(result.strip(), '$Cf')

    def test_get_childitem_variable_drive_resolved(self):
        result = self._deobfuscate("(Get-ChildItem 'Variable:ExecutionContext').Value")
        self.assertEqual(result.strip(), '$ExecutionContext')

    def test_gci_variable_drive_resolved(self):
        result = self._deobfuscate("(gci 'Variable:X').Value")
        self.assertIn('$X', result)
        self.assertNotIn('gci', result)

    def test_get_variable_value_only_member_access(self):
        result = self._deobfuscate(
            '(Get-Variable ExecutionContext -ValueOnly).InvokeCommand'
        )
        self.assertIn('$ExecutionContext', result)
        self.assertIn('InvokeCommand', result)
        self.assertNotIn('Get-Variable', result)

    def test_where_object_wildcard_paren_wrapped_pipeline(self):
        result = self._deobfuscate(
            '((New-Object Net.WebClient) | Get-Member) | ? { $_.Name -ilike \'Do*e\' }'
        )
        self.assertIn('DownloadFile', result)
        self.assertNotIn('Do*e', result)

    def test_new_object_type_resolution_in_pipeline(self):
        result = self._deobfuscate(
            '(New-Object Net.WebClient | Get-Member)[6].Name'
        )
        self.assertNotIn('[6].Name', result)

    def test_where_object_wildcard_variable_type_inferred(self):
        code = (
            "$x = New-Object Net.WebClient;"
            " ($x | Get-Member) | ? { $_.Name -ilike 'Do*e' }"
        )
        result = self._deobfuscate(code, remove_junk=False)
        self.assertIn('DownloadFile', result)
        self.assertNotIn('Do*e', result)


class TestPs1WildcardResolution(TestPs1):

    def test_wildcard_variable_get_item(self):
        result = self._deobfuscate("(Get-Item Variable:E*t).Value")
        self.assertIn('$ExecutionContext', result)
        self.assertNotIn('Variable:', result)

    def test_wildcard_variable_ambiguous(self):
        result = self._deobfuscate("Get-Item Variable:P*")
        self.assertIn('Variable:P*', result)

    def test_wildcard_cmdlet_getcmdlets(self):
        result = self._deobfuscate("$x.GetCmdlets('*w-*ct')")
        self.assertIn('New-Object', result)
        self.assertNotIn('GetCmdlets', result)

    def test_wildcard_cmdlet_invoke(self):
        result = self._deobfuscate("$x.Invoke('*w-*ct')")
        self.assertIn('New-Object', result)

    def test_wildcard_member_filter(self):
        result = self._deobfuscate("[IO.StreamReader] | Get-Member | ? { $_.Name -ilike 'ReadT*d' }")
        self.assertIn('ReadToEnd', result)

    def test_wildcard_member_filter_no_space_before_operator(self):
        result = self._deobfuscate("[IO.StreamReader] | Get-Member | ? { $_.Name-ilike'ReadT*d' }")
        self.assertIn('ReadToEnd', result)

    def test_wildcard_where_get_command(self):
        result = self._deobfuscate("Get-Command | ? { $_.Name -ilike '*w-*ct' }")
        self.assertIn('New-Object', result)

    def test_wildcard_where_unknown_source(self):
        result = self._deobfuscate("$obj | ? { $_.Name -ilike '*ts' }", remove_junk=False)
        self.assertNotIn('Exists', result)
        self.assertIn("'*ts'", result)

    def test_getcommandname_wildcard_resolved(self):
        result = self._deobfuscate(
            "$ExecutionContext.InvokeCommand.GetCommandName('*w-*ct', $True, $True)"
        )
        self.assertIn('New-Object', result)
        self.assertNotIn('GetCommandName', result)

    def test_getcommand_wildcard_resolved(self):
        result = self._deobfuscate(
            "$ExecutionContext.InvokeCommand.GetCommand('*w-*ct', 'All')"
        )
        self.assertIn('New-Object', result)
        self.assertNotIn('GetCommand', result)

    def test_getcommand_exact_name_resolved(self):
        result = self._deobfuscate(
            "$ExecutionContext.InvokeCommand.GetCommand('New-Object', 'Cmdlet')"
        )
        self.assertIn('New-Object', result)
        self.assertNotIn('GetCommand', result)

    def test_childitem_variable_resolved(self):
        result = self._deobfuscate(
            "$Y = 'hello'; (ChildItem Variable:\\Y).Value"
        )
        self.assertNotIn('ChildItem', result)
        self.assertNotIn('Variable:', result)

    def test_get_variable_name_wildcard(self):
        result = self._deobfuscate("(Get-Variable '*mdr*').Name")
        self.assertIn('MaximumDriveCount', result)
        self.assertNotIn('Get-Variable', result)

    def test_get_variable_name_wildcard_indexed_join(self):
        result = self._deobfuscate_iterative(
            "(Get-Variable '*mdr*').Name[3, 11, 2] -Join ''"
        )
        self.assertIn('iex', result.lower())
        self.assertNotIn('Get-Variable', result)


class TestPs1WildcardExtra(TestPs1):

    def test_where_object_wildcard_not_over_resolved(self):
        data = "$obj.PSObject.Methods | ? { $_.Name -ilike '*ts' }"
        result = self._deobfuscate(data, remove_junk=False)
        self.assertNotIn('Exists', result)
        self.assertIn("'*ts'", result)


class TestPs1WildcardRedirections(TestPs1):

    def test_a_redirected_variable_read_is_not_rewritten_to_the_variable(self):
        # The rewrite installs an expression and an expression carries no redirections, so the
        # value went to the console and `C:\out.txt` was never created — PowerShell creates the
        # target as it sets the redirection up, whatever the command then writes.
        self._assertUnchanged(
            'Get-Variable payload -ValueOnly > C:\\out.txt', Ps1WildcardResolution)

    def test_a_redirected_variable_write_is_not_rewritten_to_an_assignment(self):
        self._assertUnchanged("Set-Variable x 'v' > C:\\out.txt", Ps1WildcardResolution)

    def test_an_unredirected_variable_read_is_still_rewritten(self):
        self.assertEqual(
            self._apply('Get-Variable payload -ValueOnly', Ps1WildcardResolution), '$payload')

    def test_a_refused_rewrite_leaves_every_parent_pointer_true(self):
        source = 'Get-Variable payload -ValueOnly > C:\\out.txt'
        self._assertTreeIsIntact(source, source, Ps1WildcardResolution)

    def test_a_refused_variable_write_leaves_every_parent_pointer_true(self):
        # `Set-Variable` builds its replacement out of the argument the command already holds, so a
        # refusal that does not put it back leaves that argument naming a node the pass threw away.
        source = "Set-Variable x 'v' > C:\\out.txt"
        self._assertTreeIsIntact(source, source, Ps1WildcardResolution)


class TestPs1VariableCommandScope(TestPs1):
    """
    Rewriting a variable command into a variable reference has to carry the scope the command names
    and decline when the language has no way to name it. A `-Scope` argument reaches the parser as a
    switch followed by a positional, so a reading that takes every positional as an argument both
    names the wrong variable and appends the scope's own spelling to its value.

    Every expectation is measured on 5.1 — see `temp/ps1/census_measurements.md`.
    """

    def test_a_scope_the_language_can_name_is_carried_onto_the_reference(self):
        for scope, rendered in (
            ('Global', '$global:y'),
            ('Script', '$script:y'),
            ('Private', '$private:y'),
            ('Local', '$y'),
        ):
            with self.subTest(scope):
                self.assertEqual(
                    self._apply(F"Set-Variable y 'b' -Scope {scope}", Ps1WildcardResolution),
                    F"{rendered} = 'b'")

    def test_a_scope_no_qualifier_reaches_declines_the_rewrite(self):
        """
        Measured: `-Scope 1` writes the *caller's* scope, which no qualifier names. An unrecognised
        word and a computed scope are the same case, and rewriting any of them to a plain
        assignment would move the write into the scope the command stands in.
        """
        for scope in ('1', 'Foo', '$s'):
            with self.subTest(scope):
                self._assertUnchanged(F"Set-Variable y 'b' -Scope {scope}", Ps1WildcardResolution)

    def test_a_name_written_before_the_scope_is_still_the_name(self):
        self.assertEqual(
            self._apply("Set-Variable -Scope Global y 'b'", Ps1WildcardResolution),
            "$global:y = 'b'")

    def test_the_value_of_a_parameter_is_not_part_of_the_value_assigned(self):
        for source in (
            "Set-Variable y 'b' -Option ReadOnly",
            "Set-Variable y 'b' -Description 'd'",
            "Set-Variable y 'b' -Visibility Public",
        ):
            with self.subTest(source):
                self.assertEqual(self._apply(source, Ps1WildcardResolution), "$y = 'b'")

    def test_a_qualified_name_beside_a_scope_declines_the_rewrite(self):
        """
        Measured: `Set-Variable global:y 'b' -Scope Script` leaves both `$global:y` and `$script:y`
        as they were, so neither qualifier stands for what the command does.
        """
        self._assertUnchanged("Set-Variable global:y 'b' -Scope Script", Ps1WildcardResolution)

    def test_a_read_carries_its_scope_and_declines_what_it_cannot_name(self):
        self.assertEqual(
            self._apply('Get-Variable -Scope Global y -ValueOnly', Ps1WildcardResolution),
            '$global:y')
        self._assertUnchanged('Get-Variable y -ValueOnly -Scope 1', Ps1WildcardResolution)
