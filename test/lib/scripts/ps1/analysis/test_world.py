from __future__ import annotations

from test import TestBase

from refinery.lib.scripts.ps1.analysis.world import build_closed_world
from refinery.lib.scripts.ps1.parser import Ps1Parser


class Ps1TypeWorldTest(TestBase):
    """
    The closed-world predicate is the soundness floor under the member gate: a present-member purity
    grant is trusted only where no code the script runs could have mutated the type system or the
    command table, or executed opaque code that might. What matters is that every such opener is seen
    as one and an ordinary script is not, since a false "closed" deletes a read that runs code and a
    false "open" only keeps one.
    """

    @staticmethod
    def _closed(source: str) -> bool:
        script = Ps1Parser(source).parse()
        return build_closed_world(script).world_closed_at(None)


class TestPs1WorldOpeners(Ps1TypeWorldTest):

    def test_invoke_expression_opens_the_world(self):
        # Any Invoke-Expression, not only an opaque one: a constant argument can still carry a type
        # mutation, and it is inlined by a later pass than the one that consults this.
        for source in (
            'iex $x',
            "Invoke-Expression 'Get-Date'",
            'Invoke-Expression (-Join $chars)',
        ):
            with self.subTest(source):
                self.assertFalse(self._closed(source))

    def test_opaque_dispatch_opens_the_world(self):
        for source in ('& $f', '. $f', '& (Get-Command $n)', '. $env:x'):
            with self.subTest(source):
                self.assertFalse(self._closed(source))

    def test_a_type_system_mutation_opens_the_world(self):
        for source in (
            'Update-TypeData -TypeName System.String -MemberName M -MemberType ScriptProperty -Value { 1 }',
            '$x | Add-Member -MemberType ScriptProperty -Name M -Value { 1 }',
            'Import-Module Foo',
            'ipmo Foo',
        ):
            with self.subTest(source):
                self.assertFalse(self._closed(source))

    def test_a_reflective_type_mutation_opens_the_world(self):
        for source in (
            "[System.Management.Automation.PSObject+TypeAccelerators]::Add('x', [int])",
            "[System.Management.Automation.PSObject+TypeAccelerators]::Remove('x')",
            '$obj.PSObject.Members.Add($member)',
        ):
            with self.subTest(source):
                self.assertFalse(self._closed(source))

    def test_command_identity_mutation_opens_the_world(self):
        for source in (
            "Set-Alias utd Update-TypeData",
            "New-Alias utd Update-TypeData",
            "Set-Item alias:utd Update-TypeData",
            "New-Item function:foo -Value { 1 }",
            '$alias:x = 1',
            '${function:Get-Date} = $blk',
            '${function:Get-Date} = (Get-Content f.ps1)',
            '${function:Get-Date} += { 1 }',
            '${function:Get-Date}, $y = { 1 }, 2',
        ):
            with self.subTest(source):
                self.assertFalse(self._closed(source))

    def test_scriptblock_execution_opens_the_world(self):
        for source in (
            '[scriptblock]::Create($s)',
            '$sb.Invoke()',
            '$sb.InvokeReturnAsIs()',
            '$sb.InvokeWithContext($f, $v)',
            '@(1).ForEach($sb)',
            '$ExecutionContext.InvokeCommand.InvokeScript($s)',
        ):
            with self.subTest(source):
                self.assertFalse(self._closed(source))

    def test_dot_sourcing_a_file_opens_the_world(self):
        for source in (". 'helper.ps1'", '. .\\helper.ps1'):
            with self.subTest(source):
                self.assertFalse(self._closed(source))

    def test_a_mutation_hidden_in_a_function_body_opens_the_world(self):
        # The predicate walks the whole tree, so a mutator inside a defined function is seen even
        # before any call to it.
        source = 'function Reset { Update-TypeData -TypeName System.String -MemberName M }\nGet-Date'
        self.assertFalse(self._closed(source))


class TestPs1WorldClosed(Ps1TypeWorldTest):

    def test_an_ordinary_script_is_closed(self):
        for source in (
            "$x = 1\nWrite-Output $x",
            "$Null = [Environment]::UserName",
            "'abcdef'.Length",
            '[Math]::Max(1, 2)',
            'Get-ChildItem -Recurse | Where-Object { $_.Name }',
            '&{ 42 }',
            'New-Object System.Net.WebClient',
        ):
            with self.subTest(source):
                self.assertTrue(self._closed(source))


class TestPs1ShadowedCommands(Ps1TypeWorldTest):
    """
    The shadow set records which command names the script redefines, so the analysis stops trusting
    the metadata for them. It is separate from the closed/open verdict — defining a function does not
    open the world (a benign helper would neuter every grant), it only distrusts that one name.
    """

    def test_a_redefinition_shadows_its_name(self):
        world = build_closed_world(Ps1Parser(
            "function Get-Date { 1 }\n"
            "filter Out-Null { $_ }\n"
            "${function:New-Object} = { 2 }\n"
            "$alias:gc = 'Get-Content'\n"
        ).parse())
        for name in ('get-date', 'out-null', 'new-object', 'gc'):
            with self.subTest(name):
                self.assertTrue(world.command_shadowed(name))
        self.assertFalse(world.command_shadowed('get-childitem'))

    def test_a_defined_function_does_not_open_the_world(self):
        self.assertTrue(self._closed('function Get-Date { 1 }\n$x = 2'))

    def test_a_redefinition_binding_a_visible_block_does_not_open_the_world(self):
        # `${function:X} = { ... }` is `function X { ... }` in the other spelling, and the block
        # stands in the tree either way. Opening on it killed every member grant in the script over
        # a name the shadow set already distrusts.
        for source in (
            '${function:Get-Date} = { 1 }',
            '${function:Get-Date} = ({ 1 })',
            '${function:global:Get-Date} = { 1 }',
        ):
            with self.subTest(source):
                self.assertTrue(self._closed(source))
                self.assertTrue(
                    build_closed_world(Ps1Parser(source).parse()).command_shadowed('get-date'))

    def test_a_mutation_inside_a_visible_block_still_opens_the_world(self):
        # The relaxation above rests entirely on the walk reaching into the block, so a mutation
        # hidden in one has to be caught by presence like any other statement. Every place a block
        # can carry code is checked, because reaching only the plain body would make the relaxation
        # a way to smuggle a mutation past the gate.
        for body in (
            'Update-TypeData -TypeName System.String -MemberName M -Value { 1 }',
            'param($p = (Update-TypeData -TypeName System.String -MemberName M -Value { 1 }))',
            'begin { Import-Module Evil }',
            'process { iex $x }',
            'end { Add-Member -InputObject $o -Name N -Value { 1 } }',
        ):
            with self.subTest(body):
                self.assertFalse(self._closed(F'${{function:Get-Date}} = {{ {body} }}'))

    def test_a_scope_qualified_redefinition_shadows_the_name_a_call_resolves_to(self):
        # A qualifier selects which scope table the definition is written to; it is not part of the
        # name. `function global:Get-Date` is what a later bare `Get-Date` runs, so a shadow set
        # holding the qualified spelling answers every consumer's question about the wrong name.
        for source in (
            'function global:Get-Date { 1 }',
            'function local:Get-Date { 1 }',
            'function private:Get-Date { 1 }',
            'function script:Get-Date { 1 }',
            'function global:script:Get-Date { 1 }',
            'filter global:Get-Date { 1 }',
            '${function:global:Get-Date} = { 1 }',
        ):
            with self.subTest(source):
                world = build_closed_world(Ps1Parser(source).parse())
                self.assertTrue(world.command_shadowed('get-date'))

    def test_a_multi_assignment_shadows_every_identity_slot_it_writes(self):
        world = build_closed_world(Ps1Parser(
            '${function:Get-Date}, $y, $alias:gc = { 1 }, 2, 3').parse())
        self.assertTrue(world.command_shadowed('get-date'))
        self.assertTrue(world.command_shadowed('gc'))
        self.assertFalse(world.command_shadowed('y'))

    def test_a_module_qualified_name_does_not_shadow_the_bare_one(self):
        # `Module\Get-Date` names one module's export and leaves the bare name alone, so stripping
        # the qualifier off this spelling would distrust a command nothing redefined.
        world = build_closed_world(Ps1Parser('function Module\\Get-Date { 1 }').parse())
        self.assertFalse(world.command_shadowed('get-date'))


class TestPs1OffTreeCodeOpensTheWorld(Ps1TypeWorldTest):
    """
    Every construct that runs code this tree does not contain has to open the world, whatever
    spelling it wears. The mutations that matter — an Extended Type System member, a type
    accelerator, an exported command — are runspace-global, so neither the operator used to reach
    the code nor the scope it runs in contains them.
    """

    def test_running_another_script_file_opens_the_world(self):
        for source in (
            ". '.\\stage2.ps1'",
            "& '.\\stage2.ps1'",
            'stage2.ps1',
            '& $PSScriptRoot\\stage2.ps1',
        ):
            with self.subTest(source):
                self.assertFalse(self._closed(source))

    def test_running_a_block_supplied_as_data_opens_the_world(self):
        for source in (
            'Invoke-Command -ScriptBlock $sb',
            'Start-Job -ScriptBlock $sb',
            'New-Module -ScriptBlock $sb',
        ):
            with self.subTest(source):
                self.assertFalse(self._closed(source))

    def test_defining_types_or_importing_identity_opens_the_world(self):
        for source in ('Add-Type -TypeDefinition $src', 'Import-Alias .\\a.csv'):
            with self.subTest(source):
                self.assertFalse(self._closed(source))

    def test_a_module_qualifier_does_not_hide_an_opener(self):
        # A qualifier selects which module's export is meant; it does not make the command something
        # the deny-list has never heard of. Only the quoted spelling is covered: the lexer splits an
        # unquoted `Module\\Command` at the backslash, so the name never reaches this predicate
        # whole.
        for source in (
            "& 'Microsoft.PowerShell.Utility\\Invoke-Expression' $x",
            "& 'Microsoft.PowerShell.Core\\Import-Module' Foo",
            "& 'Microsoft.PowerShell.Utility\\Update-TypeData' -TypeName System.String",
        ):
            with self.subTest(source):
                self.assertFalse(self._closed(source))

    def test_the_execution_context_chain_is_matched_at_any_depth(self):
        # `$ExecutionContext.SessionState.InvokeCommand` reaches the same object the short spelling
        # does, so accepting only one depth leaves the other reading as an ordinary member call.
        for source in (
            '$ExecutionContext.InvokeCommand.InvokeScript($s)',
            '$ExecutionContext.SessionState.InvokeCommand.InvokeScript($s)',
            '$ExecutionContext.SessionState.InvokeCommand.NewScriptBlock($s)',
        ):
            with self.subTest(source):
                self.assertFalse(self._closed(source))


class TestPs1TypeDefinitionsOpenTheWorld(Ps1TypeWorldTest):
    """
    A `class` or `enum` puts a type into the session under a name the collected metadata never
    described — the same thing `Add-Type` does, and it is on the deny-list. The case that matters is
    a name the metadata *did* describe, because every purity grant keyed on a resolved type then
    vouches for a body standing in this very script.
    """

    def test_a_script_defined_type_opens_the_world(self):
        for source in (
            'class Loader { Loader([String]$s) { } }',
            'class Math { static [int] Abs([int]$x) { return 1 } }',
            'enum Version { A = 1 }',
        ):
            with self.subTest(source):
                self.assertFalse(self._closed(source))


class TestPs1QualifiedOpenerSpellings(Ps1TypeWorldTest):
    """
    A qualifier selects which module or scope an opener is taken from; it does not make the command
    something the deny-list has never heard of. Every qualifier the deny-list does not strip is a
    spelling that runs the opener while the world reports closed.
    """

    def test_a_scope_qualifier_does_not_hide_an_opener(self):
        for source in (
            "& 'global:iex' $x",
            "& 'script:Import-Module' Foo",
            "& 'global:Set-Alias' x Update-TypeData",
        ):
            with self.subTest(source):
                self.assertFalse(self._closed(source))

    def test_a_qualified_provider_path_still_addresses_command_identity(self):
        # `Microsoft.PowerShell.Core\Function::Get-Date` is what `function:Get-Date` abbreviates, so
        # a test that only knows the short spelling reads the long one as an ordinary file path.
        for source in (
            'Set-Item function:Get-Date -Value { 1 }',
            "Set-Item 'Function::Get-Date' -Value { 1 }",
            "Set-Item 'Microsoft.PowerShell.Core\\Function::Get-Date' -Value { 1 }",
            "New-Item -Path 'Microsoft.PowerShell.Core\\Alias::gd' -Value Get-Date",
        ):
            with self.subTest(source):
                self.assertFalse(self._closed(source))

    def test_an_ordinary_path_argument_is_not_an_identity_write(self):
        for source in ('Set-Content C:\\tmp\\x.txt -Value 1', 'Get-Content .\\notes.txt'):
            with self.subTest(source):
                self.assertTrue(self._closed(source))
