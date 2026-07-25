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
            '${function:Get-Date} = { 1 }',
            '$alias:x = 1',
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
