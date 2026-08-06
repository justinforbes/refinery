from __future__ import annotations

from inspect import cleandoc

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
        self.assertEqual(
            self._apply("Set-Variable y 'b' -Description 'd'", Ps1WildcardResolution),
            "$y = 'b'")

    def test_a_parameter_an_assignment_cannot_carry_declines_the_rewrite(self):
        """
        Measured: after `New-Variable x -Option ReadOnly` a later `$x = …` raises
        `SessionStateUnauthorizedAccessException` where after `$x = …` it succeeds, so the option
        decides what every later store in the script does and an assignment states the opposite.
        `-Force` is what lets a write land on a name so protected, and `-PassThru` makes the command
        emit an object an assignment does not emit.

        `-Description` is the floor: it is metadata no read or write observes, so declining on any
        parameter at all would cost the rewrite for nothing.
        """
        for source in (
            "Set-Variable y 'b' -Option ReadOnly",
            "Set-Variable y 'b' -Option Constant",
            "Set-Variable y 'b' -Force",
            "Set-Variable y 'b' -PassThru",
            "Set-Variable y 'b' -Visibility Private",
        ):
            with self.subTest(source):
                self._assertUnchanged(source, Ps1WildcardResolution)

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

    def test_the_value_property_carries_the_scope_the_ValueOnly_spelling_does(self):
        """
        `(Get-Variable y -Scope Global).Value` and `Get-Variable y -ValueOnly -Scope Global` read
        the same variable, so a rewrite that drops the qualifier on one of them silently reads a
        shadowing local instead of the global the script named.
        """
        self.assertEqual(
            self._apply('(Get-Variable y -Scope Global).Value', Ps1WildcardResolution),
            '$global:y')
        self._assertUnchanged('(Get-Variable y -Scope 1).Value', Ps1WildcardResolution)

    def test_the_name_property_is_the_name_whichever_scope_is_read(self):
        self.assertEqual(
            self._apply('(Get-Variable y -Scope 1).Name', Ps1WildcardResolution), "'y'")

    def test_a_subject_bound_by_name_is_read_like_a_positional_one(self):
        """
        A `-Name` or `-Path` argument leaves no free positional behind, so a reading that looks only
        at positionals declines every named spelling of the same command.
        """
        for source, expected in (
            ('Get-Variable -Name y -ValueOnly', '$y'),
            ('(Get-Variable -Name y).Value', '$y'),
            ('(Get-Item -Path Variable:y).Value', '$y'),
            ("Set-Item -Path Variable:y -Value 'b'", "$y = 'b'"),
            ("Set-Item Variable:y -Value 'b'", "$y = 'b'"),
            ("Set-Item -Path Variable:y 'b'", "$y = 'b'"),
        ):
            with self.subTest(source):
                self.assertEqual(self._apply(source, Ps1WildcardResolution), expected)


class TestPs1NumberedVariableNames(TestPs1):
    """
    Measured on 5.1 by comparing the variables a line creates against the ones that existed before
    it ran: `Set-Variable 007 v` creates `007`, `Set-Variable 0x10 v` creates `0x10` and
    `Set-Variable 1_000 v` creates `1_000`. A number in command-argument position becomes the text
    it is written as, never the text of the value it denotes — `007` is not `7` and `0x10` is not
    `16`. Naming the value instead invents a variable the script never mentions and leaves the one
    it does mention unresolved.
    """

    def test_a_read_spelled_the_way_the_name_was_written_resolves(self):
        for written in ('007', '0x10', '1_000'):
            with self.subTest(written):
                result = self._deobfuscate(cleandoc(F"""
                    Set-Variable {written} 'payload'
                    ${written}
                """), remove_junk=False)
                self.assertEqual(result.strip(), "'payload'")

    def test_a_read_spelled_as_the_value_the_number_denotes_does_not_resolve(self):
        for written, denoted in (('007', '7'), ('0x10', '16'), ('1_000', '1000')):
            with self.subTest(written):
                result = self._deobfuscate(cleandoc(F"""
                    Set-Variable {written} 'payload'
                    ${denoted}
                """), remove_junk=False)
                self.assertEqual(result.strip(), cleandoc(F"""
                    ${written} = 'payload'
                    ${denoted}
                """))

    def test_the_assignment_a_write_becomes_targets_the_text_the_number_was_written_as(self):
        for source, expected in (
            ("Set-Variable 007 'v'", "$007 = 'v'"),
            ("Set-Variable 0x10 'v'", "$0x10 = 'v'"),
            ("Set-Variable 1_000 'v'", "$1_000 = 'v'"),
            ("Set-Variable -Name 0x10 -Value 'v'", "$0x10 = 'v'"),
            ("Set-Item Variable:0x10 'v'", "$0x10 = 'v'"),
        ):
            with self.subTest(source):
                self.assertEqual(self._apply(source, Ps1WildcardResolution), expected)

    def test_the_reference_a_read_becomes_names_the_text_the_number_was_written_as(self):
        for source, expected in (
            ('Get-Variable 007 -ValueOnly', '$007'),
            ('Get-Variable 0x10 -ValueOnly', '$0x10'),
            ('Get-Variable 1_000 -ValueOnly', '$1_000'),
            ('(Get-Item Variable:0x10).Value', '$0x10'),
        ):
            with self.subTest(source):
                self.assertEqual(self._apply(source, Ps1WildcardResolution), expected)

    def test_the_value_property_names_what_the_ValueOnly_spelling_names(self):
        """
        `Get-Variable 0x10 -ValueOnly` and `(Get-Variable 0x10).Value` read the same variable, so
        the two spellings have to agree on which one that is.
        """
        self.assertEqual(
            self._apply('Get-Variable 0x10 -ValueOnly', Ps1WildcardResolution), '$0x10')
        self.assertEqual(
            self._apply('(Get-Variable 0x10).Value', Ps1WildcardResolution), '$0x10')

    def test_the_name_property_is_the_text_the_number_was_written_as(self):
        self.assertEqual(
            self._apply('(Get-Variable 0x10).Name', Ps1WildcardResolution), "'0x10'")
