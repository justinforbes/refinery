from __future__ import annotations

from inspect import cleandoc

from test.lib.scripts.ps1.deobfuscation import TestPs1

#: A discarded call whose removal needs the .NET type name to still denote what the metadata says,
#: so that every row below is decided by the statement written above it and by nothing else.
_JUNK = '$Null = [System.Guid]::NewGuid()'


class TestPs1TrustedEvalExcusesTheConstructsThatRunUnreadableCode(TestPs1):
    """
    Under `refinery.lib.scripts.ps1.options.Ps1DeobfuscationOptions.trust_eval` a construct that
    runs code this analysis cannot read is assumed to leave the .NET type system and the command
    table as it found them, so junk written below it is removed. Each row is one such construct, and
    each is asserted twice: kept without the option, which is the only sound answer and the default,
    and removed with it.
    """

    #: Each opener is paired with what has to be written under it to keep it alive, so that the one
    #: statement a row measures is the junk and never the opener: a `[scriptblock]::Create` whose
    #: result nothing reads is itself a dead store the option would collect, and such a row would
    #: pass while saying nothing about the junk.
    _RUNS_UNREADABLE_CODE = (
        ('Invoke-Expression $c', ''),
        ('Invoke-Command -ScriptBlock $b', ''),
        ('Start-Job -ScriptBlock $b', ''),
        ('& $f', ''),
        ('. $f', ''),
        ('. C:/stage2.ps1', ''),
        ("Set-Alias Copy-Item 'dI6tW'.Remove(3, 3)", ''),
        ('Set-Item alias:zzq Update-TypeData', ''),
        ("Import-Alias 'C:/a.csv'", ''),
        ("$function:zzq = 'Get-Date'", ''),
        ('$b = [scriptblock]::Create($s)', 'Write-Host $b'),
        ('$r = $sb.Invoke()', 'Write-Host $r'),
        ('$r = $ExecutionContext.InvokeCommand.InvokeScript($s)', 'Write-Host $r'),
    )

    def _rows(self):
        for opener, tail in self._RUNS_UNREADABLE_CODE:
            yield opener, '\n'.join(line for line in (opener, _JUNK, tail) if line)

    def test_the_junk_below_one_is_kept_by_default(self):
        for opener, script in self._rows():
            with self.subTest(opener):
                self._assertKept(script)

    def test_the_junk_below_one_is_removed_under_the_option(self):
        for opener, script in self._rows():
            with self.subTest(opener):
                self._assertRemoved(F'{script}\n', F'{_JUNK}\n', trust_eval=True)


class TestPs1TrustedEvalDoesNotExcuseAMutationTheScriptWritesDown(TestPs1):
    """
    The option is an assumption about code that cannot be read, not a licence to disbelieve a
    statement the walk can see. A mutation of the type system written out in the script opens the
    world under both models, and the junk below it is kept under both.
    """

    _MUTATES_IN_PLAIN_SIGHT = (
        'Add-Type -TypeDefinition $src',
        'Update-TypeData -TypeName System.Int32 -MemberName X -Value 1',
        'Add-Member -InputObject $o -MemberType NoteProperty -Name X -Value 1',
        'Import-Module Zzq',
        'New-Module -ScriptBlock $b',
        'class Zzq {}',
        cleandoc("""
            enum Zzq {
              A
            }
        """),
        "[PSObject+TypeAccelerators]::Add('zzq', [int])",
        '$o.PSObject.Members.Add($m)',
    )

    def test_the_junk_below_one_is_kept_by_default(self):
        for mutation in self._MUTATES_IN_PLAIN_SIGHT:
            with self.subTest(mutation):
                self._assertKept(F'{mutation}\n{_JUNK}')

    def test_the_junk_below_one_is_kept_under_the_option_too(self):
        for mutation in self._MUTATES_IN_PLAIN_SIGHT:
            with self.subTest(mutation):
                self._assertKept(F'{mutation}\n{_JUNK}', trust_eval=True)


class TestPs1TrustedEvalChangesNothingAboutAScriptThatRunsNoUnreadableCode(TestPs1):
    """
    A script with nothing to excuse comes out of both models byte for byte the same, so that the
    option is measured as the one thing it is rather than as a second deobfuscation mode.
    """

    _CLOSED_WORLD_SCRIPTS = (
        cleandoc("""
            $a = 1 + 1
            $Null = [Math]::Abs(-3)
            Write-Host $a
        """),
        cleandoc("""
            function Zzq {
              'x'
            }
            Zzq
        """),
        cleandoc("""
            $q = 'abc'.Substring(1)
            Write-Output $q
        """),
    )

    def test_both_models_produce_the_same_output(self):
        for script in self._CLOSED_WORLD_SCRIPTS:
            with self.subTest(script):
                self.assertEqual(
                    self._deobfuscate(script),
                    self._deobfuscate(script, trust_eval=True),
                )


class TestPs1TrustedEvalReachesTheAnswersOnePayloadDispatcherUsedToWithhold(TestPs1):
    """
    The three families a real sample's payload dispatcher keeps standing, each reduced to the
    statement that carries it, and none of them the discarded static call the classes above measure
    with. Together they are the reason the option exists.
    """

    def test_a_discarded_construction_below_an_eval_is_removed(self):
        self._assertRemoved(
            'Invoke-Expression $c\n$Null = New-Object System.Object\n',
            '$Null = New-Object System.Object\n',
            trust_eval=True,
        )

    def test_a_command_inside_a_block_below_an_eval_stops_refusing(self):
        self._assertRemoved(
            '$Null = [int[]](1, 2, 3 | ForEach-Object {\n'
            '  Get-Random\n'
            '}) -Join "-"\n'
            'Invoke-Expression $h\n',
            '$Null = [int[]](1, 2, 3 | ForEach-Object {\n'
            '  Get-Random\n'
            '}) -Join "-"\n',
            trust_eval=True,
        )

    def test_a_definition_and_its_only_call_below_an_eval_are_removed(self):
        self._assertDeobfuscatesTo(
            """
            Invoke-Expression $c
            function Zzq {
              $Null = 1
            }
            Zzq
            """,
            'Invoke-Expression $c',
            trust_eval=True,
        )
