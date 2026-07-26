from __future__ import annotations

from test import TestBase

from refinery.lib.scripts import Statement, _remove_from_parent
from refinery.lib.scripts.ps1.analysis.effects import (
    BodyRole,
    StatementEffect,
    body_is_inert,
    body_role,
    is_pure_constant,
    is_side_effect_free,
    output_is_covered,
    output_observed,
    pruning_erases_body,
    statement_effect,
)
from refinery.lib.scripts.ps1.analysis.types import TypeOracle
from refinery.lib.scripts.ps1.analysis.world import Ps1TypeWorld
from refinery.lib.scripts.ps1.model import (
    Ps1ArrayExpression,
    Ps1CommandInvocation,
    Ps1DataSection,
    Ps1ExpressionStatement,
    Ps1FunctionDefinition,
    Ps1IfStatement,
    Ps1InvokeMember,
    Ps1ScriptBlock,
    Ps1SubExpression,
    Ps1UnaryExpression,
)
from refinery.lib.scripts.ps1.parser import Ps1Parser


#: A type world with no mutation or capability leak, the context in which the member gate performs
#: its full type reasoning. `TestPs1Purity` asserts type facts (this member read is a plain .NET
#: property) against it, because Position A makes a present-member grant conditional on the world
#: being closed; the open-world behaviour it guards is exercised in `TestPs1ClosedWorld`.
_CLOSED_WORLD = TypeOracle(world=Ps1TypeWorld(True, frozenset()))

#: An oracle carrying no world at all, which is what a caller holds before any model has been built.
#: Every present-member grant is withheld, so a read this module proves pure is still kept. Named
#: rather than defaulted: a test asserting open-world behaviour has to say so, and the effect layer
#: no longer lets a call site acquire this answer by omission.
_NO_WORLD = TypeOracle()


class Ps1EffectsTest(TestBase):

    @staticmethod
    def _pure(node) -> bool:
        return is_side_effect_free(node, _CLOSED_WORLD)

    @staticmethod
    def _effect(stmt) -> StatementEffect:
        return statement_effect(stmt, _CLOSED_WORLD)

    @staticmethod
    def _inert(node) -> bool:
        return body_is_inert(node, _CLOSED_WORLD)

    @staticmethod
    def _parse(source: str):
        return Ps1Parser(source).parse()

    @classmethod
    def _statement(cls, source: str):
        return cls._parse(source).body[0]

    @classmethod
    def _expression(cls, source: str):
        statement = cls._statement(source)
        assert isinstance(statement, Ps1ExpressionStatement)
        assert statement.expression is not None
        return statement.expression

    @classmethod
    def _first(cls, source: str, kind):
        return next(node for node in cls._parse(source).walk() if isinstance(node, kind))


class TestPs1Purity(Ps1EffectsTest):

    def test_effect_free_expressions(self):
        for source in (
            "'a' + 'b'",
            '[Math]::Abs(-3)',
            '[Convert]::ToBase64String($b)',
            '$s.Substring(0, 2)',
            'Get-Date',
            'New-Object System.Text.StringBuilder',
        ):
            with self.subTest(source):
                self.assertTrue(self._pure(self._expression(source)))

    def test_expressions_that_change_the_world(self):
        for source in (
            'Start-Process notepad',
            'Remove-Item x',
            "[System.IO.File]::WriteAllText('a', 'b')",
            '$x++',
            '$s.Invoke()',
        ):
            with self.subTest(source):
                self.assertFalse(self._pure(self._expression(source)))

    def test_an_unrecognized_construct_is_assumed_impure(self):
        # The allow-list is the whole safety argument: anything it does not name has to come back
        # impure, however harmless it looks.
        for source in ('New-Object System.Net.WebClient', '& $f', '$obj.Frobnicate()'):
            with self.subTest(source):
                self.assertFalse(self._pure(self._expression(source)))

    def test_a_pipeline_cmdlet_is_as_pure_as_the_body_it_runs(self):
        # A scriptblock body is a sequence of statements, so purity of the cmdlet has to be decided
        # at the statement layer: a body of discards is as harmless as one of bare pure expressions.
        for source, pure in (
            ('1..3 | ForEach-Object { $_ }', True),
            ('1..3 | ForEach-Object { $Null = $_ }', True),
            ('1..3 | ForEach-Object { [Void]$_ }', True),
            ('1..3 | Where-Object { $Null = $_ }', True),
            ('1..3 | ForEach-Object { $x = $_ }', False),
            ('1..3 | ForEach-Object { Write-Host $_ }', False),
            ('1..3 | ForEach-Object { [Void](Start-Process notepad) }', False),
        ):
            with self.subTest(source):
                self.assertIs(self._pure(self._expression(source)), pure)

    def test_a_pipeline_cmdlet_body_is_read_for_every_such_cmdlet(self):
        # Three of the four pipeline cmdlets also name a plain pure cmdlet, so an allow-list that
        # answers on the name alone never reaches the body and calls every one of these pure.
        for source in (
            'Where-Object { Start-Process notepad }',
            'Select-Object { Start-Process notepad }',
            'Sort-Object { Start-Process notepad }',
            '1..3 | ForEach-Object { Start-Process notepad }',
        ):
            with self.subTest(source):
                self.assertFalse(self._pure(self._expression(source)))

    def test_a_scriptblock_argument_is_read_through_every_block_it_owns(self):
        # The parser fills either `body` or the named blocks, so a block that carries its work in
        # `begin`/`process`/`end` or in a parameter default reports an empty statement list. Judging
        # the cmdlet by that list calls a command that runs on every input item pure.
        for source in (
            'Get-Process | Where-Object { begin { Start-Process notepad } process { $_ } }',
            '1..3 | ForEach-Object { end { Start-Process notepad } }',
            '1..3 | ForEach-Object { param($p = (Start-Process notepad)) [Void]$_ }',
        ):
            with self.subTest(source):
                self.assertFalse(self._pure(self._expression(source)))

    def test_a_computed_member_name_is_an_expression_the_read_evaluates(self):
        # `$x.$(...)` runs the subexpression to decide which member to read, before any read.
        for source in (
            '$x.$(Start-Process notepad)',
            '[IO.Path]::$(Start-Process notepad)',
            '$x.$(Remove-Item C:\\important).Length',
        ):
            with self.subTest(source):
                self.assertFalse(self._pure(self._expression(source)))

    def test_a_provable_pure_member_read_is_removable(self):
        # The object is pure to evaluate and the member read runs no code: a curated reflection
        # property (`Process.ProcessName`, resolved through the static call's return type), any read
        # on a sealed value type (`DateTime.Ticks`), or a reflection field, which is a bare memory
        # slot with no getter (`Math.PI`, `Int32.MaxValue`). The chained read resolves the receiver
        # of `.ManagedThreadId` through `CurrentThread`, itself a curated pure read.
        for source in (
            '[Diagnostics.Process]::GetCurrentProcess().ProcessName',
            '(Get-Date).Ticks',
            '[Environment]::UserName',
            '[Threading.Thread]::CurrentThread.ManagedThreadId',
            '[Math]::PI',
            '[Int]::MaxValue',
            '[String]::Empty',
        ):
            with self.subTest(source):
                self.assertTrue(self._pure(self._expression(source)))

    def test_a_cast_to_a_type_the_metadata_cannot_resolve_is_kept(self):
        # Converting a string to a custom type runs that type's constructor, and `Add-Type`, a
        # PowerShell `class` and `[Reflection.Assembly]::Load` each make such a name denote code the
        # metadata never saw. Granting on the operand alone deleted the conversion — and the call
        # inside it — even under a closed world, which is where the name is most trusted.
        for source in ("[Loader]'payload'", '[Some.Unknown.Type]$x', '[NotAType]42'):
            with self.subTest(source):
                self.assertFalse(self._pure(self._expression(source)))

    def test_a_cast_to_a_collected_type_is_still_removable(self):
        # The guard above is a resolution check, not a blanket denial: the ordinary casts an
        # obfuscator emits by the dozen have to keep pruning.
        for source in ("[Int]'42'", '[Void]1', '[String]42', '[Char[]]$s', '[ordered]@{ a = 1 }'):
            with self.subTest(source):
                self.assertTrue(self._pure(self._expression(source)))

    def test_a_quoted_member_name_reads_the_same_member_as_a_bare_one(self):
        # A quoted member name (`.'Ticks'`) is one spelling of a literal member, not a computed one,
        # so the gate resolves it to the same member and reaches the same verdict as the bare form:
        # the sealed-value read is removable and the Extended Type System getter is kept. Only a
        # name the engine computes at runtime (`.$(...)`) leaves the member unknown and stays
        # impure.
        self.assertTrue(self._pure(self._expression("(Get-Date).'Ticks'")))
        self.assertFalse(self._pure(self._expression("(Get-Process).'Path'")))
        self.assertFalse(self._pure(self._expression('(Get-Date).$($x)')))

    def test_a_member_read_is_kept_unless_the_getter_is_proven_inert(self):
        # The soundness core: a property getter may run code or throw, and a read is removed only
        # when it is proven not to. Returning the object's own purity — which this gate replaces —
        # deleted every one of these. `Process.Path` is an Extended Type System member that shells
        # out; `Process.ExitCode` throws until the process exits; `IPAddress.Address` throws by
        # address family, which is why that type is not a whole-surface grant; casting to the
        # supertype `object` leaves the runtime type free to carry an effectful member the supertype
        # lacks; and an object whose type is not resolved could be anything at all.
        for source in (
            '(Get-Process).Path',
            '[Diagnostics.Process]::GetCurrentProcess().ExitCode',
            "([ipaddress]'::1').Address",
            '([object]$x).Path',
            '$Host.UI',
            '$reader.EndOfStream',
        ):
            with self.subTest(source):
                self.assertFalse(self._pure(self._expression(source)))

    def test_a_forwarding_cmdlet_result_read_is_kept(self):
        # A cmdlet's [OutputType] is only a lower bound: one that forwards its input emits types it
        # never declares. `Get-Random -InputObject $x` returns an element of $x -- a Process if $x
        # holds processes -- so proving `.Path` pure over its declared numeric outputs would delete
        # a live ETS getter. The oracle trusts a declaration only for a curated closed set, so a
        # read on any forwarding command's result is left unresolved and kept.
        for source in (
            '(Get-Random -InputObject $procs).Path',
            '(Get-Random -InputObject $x).ProcessName',
            '(Get-Content $p).Length',
        ):
            with self.subTest(source):
                self.assertFalse(self._pure(self._expression(source)))

    def test_a_static_field_read_is_gated_but_an_instance_field_is_not(self):
        # Reading a static field runs the declaring type's static constructor on first touch, so it
        # is not unconditionally pure the way an instance field is; it is removable only when the
        # type or the read is granted. `Math.PI` and `Int32.MaxValue` are granted; `IO.Path`'s
        # separator fields are not, and its cctor could do anything, so they are kept.
        self.assertFalse(self._pure(self._expression('[IO.Path]::DirectorySeparatorChar')))
        self.assertTrue(self._pure(self._expression('[Math]::PI')))
        self.assertTrue(self._pure(self._expression('[Int]::MaxValue')))

    def test_a_hash_literal_evaluates_its_keys_as_well_as_its_values(self):
        for source in (
            '@{ $(Start-Process notepad) = 1 }',
            '@{ 1 = $(Start-Process notepad) }',
        ):
            with self.subTest(source):
                self.assertFalse(self._pure(self._expression(source)))

    def test_a_cmdlet_is_no_purer_than_the_arguments_it_evaluates(self):
        # Being a pure transform says nothing about what the operands cost to produce: the cmdlet
        # runs whatever it is handed before it transforms anything.
        for source in (
            'Out-String -InputObject (Start-Process notepad)',
            'Measure-Object -InputObject (Start-Process notepad)',
            'Get-Item (Remove-Item C:\\important)',
            'Where-Object -InputObject (Start-Process notepad) { $_ }',
        ):
            with self.subTest(source):
                self.assertFalse(self._pure(self._expression(source)))

    def test_a_member_invoking_foreach_has_no_body_to_vouch_for_it(self):
        # `ForEach-Object -MemberName Delete` calls that member on every input item. A body check
        # that proves a property of the scriptblocks it saw proves nothing when there are none.
        for source in (
            'Get-ChildItem | ForEach-Object -MemberName Delete',
            'Get-Process | ForEach-Object Kill',
            'Get-Process | ForEach-Object $handler',
        ):
            with self.subTest(source):
                self.assertFalse(self._pure(self._expression(source)))

    def test_an_in_place_mutator_is_pure_only_on_a_temporary(self):
        # `[Array]::Reverse` rewrites what it is given. Reversing a value nothing else can reach is
        # unobservable; reversing a variable is the mutation the rest of the script reads back.
        for source, pure in (
            ("[Array]::Reverse('ab'.ToCharArray())", True),
            ('[Array]::Reverse((1, 2, 3))', True),
            ('[Array]::Reverse($buffer)', False),
            ('[Array]::Sort($buffer)', False),
            ('[Array]::Clear($buffer, 0, 2)', False),
            ('[Array]::Reverse($this.Items)', False),
            ('[Array]::Reverse($pair[0])', False),
        ):
            with self.subTest(source):
                self.assertIs(self._pure(self._expression(source)), pure)

    def test_an_out_parameter_writes_the_callers_storage(self):
        # `[ref]$x` hands the callee somewhere to put its result. Every `TryParse` on the numeric,
        # date and network types takes one, so a whole-type grant on its own calls them all pure and
        # the deobfuscator drops the statement that produced the value the script goes on to read.
        for source in (
            '[Int]::TryParse($s, [ref]$n)',
            '[Int32]::TryParse($s, [ref]$n)',
            '[Int64]::TryParse($s, [ref]$n)',
            '[Double]::TryParse($s, [ref]$n)',
            '[Decimal]::TryParse($s, [ref]$n)',
            '[DateTime]::TryParse($s, [ref]$d)',
            '[TimeSpan]::TryParse($s, [ref]$t)',
            '[IPAddress]::TryParse($s, [ref]$a)',
            '[Guid]::TryParse($s, [ref]$g)',
            '[Version]::TryParse($s, [ref]$v)',
            '[Char]::TryParse($s, [ref]$c)',
            '[Int]::TryParse($s, [ref]$obj.Slot)',
            '$dict.TryGetValue($k, [ref]$v)',
        ):
            with self.subTest(source):
                self.assertFalse(self._pure(self._expression(source)))

    def test_a_call_that_only_returns_its_result_stays_pure(self):
        # The rule is about being handed writable storage, not about the call having arguments.
        for source in (
            '[Int]::Parse($s)',
            '[Math]::Max($a, $b)',
            "[String]::Join(',', $parts)",
            '[Convert]::ToBase64String($b)',
        ):
            with self.subTest(source):
                self.assertTrue(self._pure(self._expression(source)))

    def test_a_bare_out_argument_is_caught_by_the_signature(self):
        # `TryParse` binds its second parameter by reference, and PowerShell lets the caller pass
        # the target without a `[ref]` cast. The syntactic check sees only `[ref]$n`; the collected
        # signature is what tells the effect layer that a bare `$r` in that position is written, so
        # a call filling a live variable is not mistaken for a pure transform and dropped. Only a
        # storage location can be written back through, so a temporary in that slot stays pure.
        for source, pure in (
            ('[Int]::TryParse($s, $r)', False),
            ('[Int32]::TryParse($s, $r)', False),
            ('[DateTime]::TryParse($s, $d)', False),
            ('[IPAddress]::TryParse($s, $a)', False),
            ('[Int]::TryParse($s, $obj.Slot)', False),
            ('[Int]::TryParse($s, $arr[0])', False),
            ('[Int]::Parse($s)', True),
        ):
            with self.subTest(source):
                self.assertIs(self._pure(self._expression(source)), pure)

    def test_a_types_spelling_does_not_change_its_purity(self):
        # `int`, `Int32` and the qualified name are one type, and a generic is one type however its
        # argument is spelled; resolving every spelling through the collected data lands them on a
        # single canonical key, so the verdict cannot depend on which an obfuscated script chose.
        # This is the property that retired the dual-spelling allow-list entries.
        for variants in (
            ('[int]::Parse($s)', '[Int32]::Parse($s)', '[System.Int32]::Parse($s)'),
            (
                "New-Object 'Collections.Generic.List[byte]'",
                "New-Object 'System.Collections.Generic.List[byte]'",
            ),
        ):
            with self.subTest(variants):
                verdicts = {self._pure(self._expression(v)) for v in variants}
                self.assertEqual(verdicts, {True})

    def test_a_member_that_writes_whatever_it_is_handed(self):
        # A whole-type grant asserts that no member of the type writes. `[IO.Path]` is pure apart
        # from the one member that creates a file on disk, and that one takes no arguments to be
        # judged by.
        self.assertFalse(self._pure(self._expression('[IO.Path]::GetTempFileName()')))
        self.assertTrue(self._pure(self._expression('[IO.Path]::Combine($a, $b)')))
        self.assertTrue(self._pure(self._expression('[IO.Path]::GetFileName($p)')))

    def test_a_parameter_that_names_a_variable_the_command_fills(self):
        # `-OutVariable d` sets `$d`. The parsed parameter name carries its leading dash and
        # PowerShell binds any unambiguous abbreviation, so matching the documented spelling alone
        # recognizes none of these and the deobfuscator drops the statement that fills the variable.
        for source in (
            'Get-Date -OutVariable d',
            'Get-Date -outvariable d',
            'Get-Date -OutVar d',
            'Get-Date -ov d',
            'Get-Date -ov:$d',
            'Get-Process -ErrorVariable e',
            'Get-ChildItem -PipelineVariable p',
            'Get-Content x -WarningVariable w',
            'Get-Random -SetSeed 5',
        ):
            with self.subTest(source):
                self.assertFalse(self._pure(self._expression(source)))

    def test_the_write_parameter_derivation_fails_loud_when_data_drops_one(self):
        # The out-variable write parameters are derived from the collected common parameters. If a
        # regenerated surface stopped flagging one, the set would silently shrink and a real
        # `-OutVariable` write would be judged pure and dropped; the derivation floors itself
        # against that and refuses to build rather than failing open in the deletion direction.
        from unittest.mock import patch

        from refinery.lib.scripts.ps1 import data
        from refinery.lib.scripts.ps1.analysis import effects

        reduced = {
            name: aliases
            for name, aliases in data.COMMON_PARAMETERS.items()
            if name != 'outvariable'
        }
        with patch.object(data, 'COMMON_PARAMETERS', reduced):
            with self.assertRaises(ValueError):
                effects._writing_parameters()

    def test_a_command_that_only_reads_stays_pure(self):
        for source in (
            'Get-Date -Format o',
            'Get-ChildItem -Recurse',
            'Get-Item x -Force',
            'Get-Random -Maximum 5',
            '1..3 | Select-Object -First 2',
        ):
            with self.subTest(source):
                self.assertTrue(self._pure(self._expression(source)))

    def test_a_constructor_is_judged_by_every_argument_it_is_handed(self):
        # `New-Object` binds two positional parameters. An accessor that reports the first two and
        # drops the rest leaves the trailing argument unexamined, and a call it runs is deleted.
        for source in (
            "New-Object String 'x' (Start-Process notepad)",
            'New-Object Text.StringBuilder (Start-Process notepad)',
            'New-Object Text.StringBuilder ([ref]$n)',
        ):
            with self.subTest(source):
                self.assertFalse(self._pure(self._expression(source)))

    def test_a_type_grants_purity_to_its_members_one_by_one(self):
        # A type whose static surface mixes readers with process- and environment-level writers
        # cannot be trusted wholesale, so membership is per method.
        for source, pure in (
            ("[Environment]::GetFolderPath('Desktop')", True),
            ("[Environment]::GetEnvironmentVariable('PATH')", True),
            ('[Environment]::Exit(0)', False),
            ("[Environment]::SetEnvironmentVariable('k', 'v')", False),
        ):
            with self.subTest(source):
                self.assertIs(self._pure(self._expression(source)), pure)

    def test_a_redirection_writes_a_file_however_pure_the_command_is(self):
        for source in (
            'Get-Date > C:\\out.txt',
            'Get-Content a.txt >> b.txt',
            'Get-Process 2> C:\\err.txt',
        ):
            with self.subTest(source):
                self.assertFalse(self._pure(self._expression(source)))

    def test_no_combining_form_launders_an_effect(self):
        # Purity is compositional: an impure operand must poison every expression built over it,
        # otherwise a pass could delete the effect by wrapping it.
        for source in (
            '1 + (Start-Process notepad)',
            '@(1, (Start-Process notepad))',
            '-(Start-Process notepad)',
            '((Start-Process notepad))',
            '@{ k = (Start-Process notepad) }',
            '$(Start-Process notepad)',
            '1..3 | ForEach-Object { Start-Process notepad }',
        ):
            with self.subTest(source):
                self.assertFalse(self._pure(self._expression(source)))


class TestPs1MemberGateWorld(Ps1EffectsTest):
    """
    Position A: a present-member purity grant — a property read, a static or instance method call, a
    constructor — is trusted only under a closed type world. The type reasoning that proves the read
    inert is the same whether the world is open or closed; what the world decides is whether that
    proof may be acted on, because an Extended Type System mutation the script could run would make
    the read effectful. Each read below is one the type layer proves pure, withheld under an oracle
    carrying no world and granted only under a closed one.
    """

    def test_each_grant_is_withheld_when_the_world_is_open(self):
        for source in (
            "'abcdef'.Length",
            '(Get-Date).Ticks',
            '[Environment]::UserName',
            '[Math]::Max(1, 2)',
            '$s.Trim()',
            "[Array]::Reverse('ab'.ToCharArray())",
            'New-Object System.Version(1, 2)',
        ):
            with self.subTest(source):
                self.assertFalse(is_side_effect_free(self._expression(source), _NO_WORLD))
                self.assertTrue(self._pure(self._expression(source)))

    def test_a_denied_read_stays_impure_even_under_a_closed_world(self):
        # The world gates grants, never denies: a getter that runs code or throws, an in-place
        # mutator on shared storage, or an out-parameter is kept whether the world is open or
        # closed.
        for source in (
            '(Get-Process).Path',
            '[Diagnostics.Process]::GetCurrentProcess().ExitCode',
            '[Array]::Reverse($buffer)',
            '[IO.Path]::GetTempFileName()',
        ):
            with self.subTest(source):
                self.assertFalse(is_side_effect_free(self._expression(source), _NO_WORLD))
                self.assertFalse(self._pure(self._expression(source)))


class TestPs1CommandShadowing(Ps1EffectsTest):
    """
    A command the script redefines as a function is not the built-in the metadata describes, so —
    even under a closed type world — the oracle must not type its result and the gate must not grant
    its purity or read it as a discarding sink. A command the script does not redefine is unaffected.
    """

    @staticmethod
    def _shadowing(*names: str) -> TypeOracle:
        return TypeOracle(world=Ps1TypeWorld(True, frozenset(names)))

    def test_a_shadowed_commands_read_and_call_are_impure(self):
        oracle = self._shadowing('get-date', 'new-object')
        for source in ('(Get-Date).Ticks', 'Get-Date', 'New-Object System.Version'):
            with self.subTest(source):
                self.assertFalse(is_side_effect_free(self._expression(source), oracle))

    def test_a_shadowed_pipeline_sink_does_not_discard(self):
        oracle = self._shadowing('out-null', 'foreach-object')
        for source in ('1 | Out-Null', '1 | ForEach-Object { [Void]$_ }'):
            with self.subTest(source):
                self.assertIs(
                    statement_effect(self._statement(source), oracle), StatementEffect.EFFECT)

    def test_an_unshadowed_command_stays_pure_beside_a_shadowed_one(self):
        oracle = self._shadowing('get-date')
        self.assertTrue(is_side_effect_free(self._expression('Get-ChildItem'), oracle))
        self.assertTrue(is_side_effect_free(self._expression('New-Object System.Version'), oracle))


class TestPs1StatementEffect(Ps1EffectsTest):

    def test_a_statement_that_only_yields_a_value_is_output(self):
        for source in ('42', "'hi'", '$x', '1 + 1', 'Get-Date'):
            with self.subTest(source):
                self.assertIs(self._effect(self._statement(source)), StatementEffect.OUTPUT)

    def test_a_statement_that_does_something_is_an_effect(self):
        for source in ('Write-Host hi', '$x = 1', '$x++', 'if ($a) { }'):
            with self.subTest(source):
                self.assertIs(self._effect(self._statement(source)), StatementEffect.EFFECT)

    def test_the_discard_idioms_emit_nothing(self):
        for source in (
            '$Null = 5',
            '[Void]1',
            '1..3 | Out-Null',
            '1..3 | ForEach-Object { [Void]$_ }',
            '1..3 | ForEach-Object { $Null = $_ }',
        ):
            with self.subTest(source):
                self.assertIs(self._effect(self._statement(source)), StatementEffect.DISCARD)

    def test_a_discard_idiom_wrapped_around_an_effect_is_still_an_effect(self):
        # A discard idiom throws away a value, never the work that produced it. Obfuscated scripts
        # wrap real calls in exactly these idioms, so a discard that skips the operand check makes
        # the deobfuscator delete the payload it is supposed to surface.
        for source in (
            '$Null = Start-Process notepad',
            '[Void](Start-Process notepad)',
            '[Void]$(Remove-Item C:\\important)',
            '1..3 | ForEach-Object { [Void](Start-Process notepad) }',
            '1..3 | ForEach-Object { $Null = Start-Process notepad }',
        ):
            with self.subTest(source):
                self.assertIs(self._effect(self._statement(source)), StatementEffect.EFFECT)

    def test_a_discard_of_a_harmless_value_stays_a_discard(self):
        for source in ('$Null = 5', '[Void]1', '[Void]$x', "[Void]('a' + 'b')"):
            with self.subTest(source):
                self.assertIs(self._effect(self._statement(source)), StatementEffect.DISCARD)

    def test_a_foreach_is_judged_by_all_of_its_work_not_only_its_blocks(self):
        # `ForEach-Object` takes its work through its arguments, and a discarding block sitting
        # beside that work says nothing about it: `-MemberName Delete` invokes a member on every
        # input item and `-End $sb` runs whatever scriptblock the variable holds. Reading the
        # question off "a block was seen" let the visible discard vouch for both.
        for source in (
            'Get-Process | ForEach-Object { [Void]$_ } -MemberName Kill',
            'Get-Process | ForEach-Object { [Void]$_ } Kill',
            'Get-ChildItem | ForEach-Object { $Null = $_ } -MemberName Delete',
            '1..3 | ForEach-Object -Process { [Void]$_ } -End $sb',
            '1..3 | ForEach-Object { [Void]$_ } $sb',
            "Get-Process | ForEach-Object { [Void]$_ } -MemberName @'\nKill\n'@",
            "Get-ChildItem | ForEach-Object { $Null = $_ } @'\nDelete\n'@",
        ):
            with self.subTest(source):
                self.assertIs(self._effect(self._statement(source)), StatementEffect.EFFECT)
                self.assertFalse(self._pure(self._expression(source)))

    def test_a_foreach_body_is_read_through_every_block_it_owns(self):
        # A discarding `body` says nothing about work the same block carries in a named or `param`
        # block, and the parser fills only one of the two.
        for source in (
            '1..3 | ForEach-Object { end { Remove-Item C:\\important } }',
            '1..3 | ForEach-Object { begin { Start-Process notepad } process { [Void]$_ } }',
            '1..3 | ForEach-Object { param($p = (Start-Process notepad)) [Void]$_ }',
        ):
            with self.subTest(source):
                self.assertIs(self._effect(self._statement(source)), StatementEffect.EFFECT)

    def test_a_foreach_whose_work_is_all_visible_is_still_a_discard(self):
        for source in (
            '1..3 | ForEach-Object { [Void]$_ }',
            '1..3 | ForEach-Object -Process { $Null = $_ }',
            '1..3 | ForEach-Object -Begin { [Void]$_ } -Process { $Null = $_ }',
            '1..3 | ForEach-Object -InputObject 5 -Process { [Void]$_ }',
        ):
            with self.subTest(source):
                self.assertIs(self._effect(self._statement(source)), StatementEffect.DISCARD)

    def test_a_splatted_argument_hides_the_parameters_it_supplies(self):
        # `@options` can carry `-OutVariable` as easily as `-Format`, and none of it is in the
        # source, so there is nothing to judge the command by.
        for source in ('Get-Date @options', 'Get-ChildItem @options | Out-Null'):
            with self.subTest(source):
                self.assertIs(self._effect(self._statement(source)), StatementEffect.EFFECT)

    def test_a_pure_pipeline_cmdlet_still_yields_a_value_a_caller_may_want(self):
        # Purity and emission answer different questions: `Where-Object` performs no side effect,
        # yet the filtered value it puts on the pipeline is not junk.
        statement = self._statement('1..3 | Where-Object { $_ }')
        self.assertTrue(self._pure(self._expression('1..3 | Where-Object { $_ }')))
        self.assertIs(self._effect(statement), StatementEffect.EFFECT)

    def test_pure_constants_are_a_strict_refinement_of_output(self):
        # The dead-code pass prunes only pure constants and the junk pass prunes the whole OUTPUT
        # set. That is only defensible while the candidate sets stay nested.
        for source in ('42', '-3', '(7)', '$Null', '$True', '$False', '+9', '3.5'):
            with self.subTest(source):
                self.assertTrue(is_pure_constant(self._expression(source)))
                self.assertIs(self._effect(self._statement(source)), StatementEffect.OUTPUT)

    def test_a_string_literal_is_not_a_prunable_constant(self):
        # A bare string is very often the point of the script, so it is deliberately left out of the
        # constant set even though it is side-effect free.
        expression = self._expression("'hi'")
        self.assertTrue(self._pure(expression))
        self.assertFalse(is_pure_constant(expression))

    def test_a_computed_expression_is_not_a_constant(self):
        for source in ('1 + 1', '$x', '[Math]::Abs(-3)'):
            with self.subTest(source):
                self.assertFalse(is_pure_constant(self._expression(source)))


class TestPs1EffectInvariant(Ps1EffectsTest):
    """
    A regression list of shapes that were each, at some point, deleted along with real work: a
    statement the passes are allowed to drop must not contain a call the expression layer rejects.

    Read this as a list, not as a property. Sweeping the same check over every PowerShell snippet in
    the test tree reports nothing at all, including on shapes confirmed to be live data-loss bugs at
    the time — because it asks `is_side_effect_free` about the sub-expressions of a statement whose
    classification already consulted it, so the two layers sharing one wrong belief looks like
    agreement. `ForEach-Object { $Null = $_ } -MemberName Delete` was exactly that: `DISCARD` at the
    statement layer, pure at the expression layer, silently deleted, and invisible here.

    A check that would have caught it has to ask an oracle this module does not supply — real
    PowerShell, or a corpus labelled by hand with what each statement actually does.
    """

    def _violations(self, source: str):
        script = self._parse(source)
        found = []
        for node in script.walk():
            if not isinstance(node, Statement):
                continue
            if self._effect(node) is StatementEffect.EFFECT:
                continue
            for sub in node.walk():
                if sub is node:
                    continue
                if isinstance(sub, (Ps1CommandInvocation, Ps1InvokeMember)):
                    if not self._pure(sub):
                        found.append(sub)
                elif isinstance(sub, Ps1UnaryExpression) and sub.operator in ('++', '--'):
                    found.append(sub)
        return found

    def test_a_removable_statement_never_hides_work(self):
        for source in (
            '[Void](Start-Process notepad)',
            '$Null = Start-Process notepad',
            '$Null = 5',
            '[Void]1',
            '1 | Out-Null -InputObject (Start-Process notepad)',
            '1..3 | ForEach-Object { [Void](Start-Process notepad) }',
            '1..3 | ForEach-Object { $_ } | Out-Null',
            '1..3 | ForEach-Object { $Null = $_ }',
            'Get-Process | ForEach-Object -MemberName Kill',
            'Get-Date > C:\\out.txt',
            'Get-Date -OutVariable d',
            '[Environment]::Exit(0)',
            '[Array]::Reverse($buffer)',
            '[Int]::TryParse($s, [ref]$n)',
            '[Int]::TryParse($s, $n)',
            '[IO.Path]::GetTempFileName()',
            "New-Object String 'x' (Start-Process notepad)",
            '[Void]$a[$i++]',
            '$Null = $x++',
            "$Null = 'a' + $(Start-Process notepad)",
            "@(1, 2) | Where-Object { $_ -GT 1 } | ForEach-Object { [Void](Remove-Item $_) }",
        ):
            with self.subTest(source):
                self.assertEqual(self._violations(source), [])


class TestPs1BodyRole(Ps1EffectsTest):

    def test_the_script_root_is_its_own_role(self):
        self.assertIs(body_role(self._parse('42')), BodyRole.SCRIPT)

    def test_a_body_whose_value_the_caller_observes(self):
        for source in ('function f { 42 }', '&{ 42 }', '. { 42 }'):
            with self.subTest(source):
                block = self._first(source, Ps1ScriptBlock)
                self.assertIs(body_role(block), BodyRole.RETURNING)

    def test_a_captured_body_is_opaque(self):
        for source in (
            '$cb = { 42 }',
            '&{ 42 } | Out-Null',
            'Foo-Bar { 42 }',
            'Get-Item | ForEach-Object { 42 }',
        ):
            with self.subTest(source):
                block = self._first(source, Ps1ScriptBlock)
                self.assertIs(body_role(block), BodyRole.OPAQUE)

    def test_a_subexpression_is_opaque(self):
        self.assertIs(body_role(self._first('$x = $( 42 )', Ps1SubExpression)), BodyRole.OPAQUE)

    def test_an_array_expression_owns_no_prunable_body(self):
        # `@( ... )` holds a captured value and is kept out of the pruning walks by having no role
        # at all. Teaching the body accessor about it would silently make its contents prunable.
        self.assertIsNone(body_role(self._first('$x = @( 42 )', Ps1ArrayExpression)))

    def test_a_node_that_owns_no_body_has_no_role(self):
        self.assertIsNone(body_role(self._expression('42')))

    def test_a_nested_block_does_not_inherit_its_owner_role(self):
        # Pinned, not endorsed. The same `if` body classifies three ways depending only on who owns
        # it, although its value is observed exactly when its owner's is. Resolving this needs
        # reachability, so the traces are asserted here to keep any change deliberate.
        for source, expected in (
            ('if ($x) { 1 }', BodyRole.NESTED),
            ('function f { if ($x) { 1 } }', BodyRole.RETURNING),
            ('&{ if ($x) { 1 } }', BodyRole.NESTED),
        ):
            with self.subTest(source):
                block = self._first(source, Ps1IfStatement).clauses[0][1]
                self.assertIs(body_role(block), expected)

    def test_a_block_inside_a_captured_body_is_opaque(self):
        for source in ('$cb = { if ($x) { 1 } }', '$y = $( if ($x) { 1 } )'):
            with self.subTest(source):
                block = self._first(source, Ps1IfStatement).clauses[0][1]
                self.assertIs(body_role(block), BodyRole.OPAQUE)


class TestPs1EmitSafety(Ps1EffectsTest):

    def test_only_a_returning_body_has_an_output_to_protect(self):
        protected = {role for role in BodyRole if output_observed(role)}
        self.assertEqual(protected, {BodyRole.RETURNING})

    def test_only_the_script_root_may_not_be_emptied(self):
        guarded = {role for role in BodyRole if pruning_erases_body(role, [])}
        self.assertEqual(guarded, {BodyRole.SCRIPT})

    def test_a_surviving_statement_never_trips_the_erasure_guard(self):
        survivors = list(self._parse('Write-Host hi').body)
        for role in BodyRole:
            with self.subTest(role):
                self.assertFalse(pruning_erases_body(role, survivors))

    def test_a_function_definition_alone_does_not_carry_a_bodys_output(self):
        for source in ('function f { Write-Host hi }', '&{ function f { Write-Host hi } }'):
            with self.subTest(source):
                definition = self._first(source, Ps1FunctionDefinition)
                self.assertFalse(output_is_covered([definition]))

    def test_any_other_survivor_covers_the_output(self):
        for source in ('Get-Item x', '42', 'if ($a) { 42 }', '($x = 1)'):
            with self.subTest(source):
                self.assertTrue(output_is_covered(list(self._parse(source).body)))

    def test_a_statement_holding_only_silent_statements_does_not_cover_the_output(self):
        # A body-bearing statement emits whatever the statements inside it emit, so it is descended
        # into rather than granted emission for its shape. Answering `True` for an empty branch is
        # what let one stand in for the value a `RETURNING` body exists to produce.
        for source in (
            'if ($a) { }',
            'if ($a) { $Null = 1 } else { $x = 2 }',
            'foreach ($i in $x) { $Null = $i }',
            'while ($a) { }',
            'do { } while ($a)',
            'for ($i = 0; $i -lt 3; $i++) { }',
            'switch ($a) { 1 { } }',
            'try { } catch { }',
        ):
            with self.subTest(source):
                self.assertFalse(output_is_covered(list(self._parse(source).body)))

    def test_a_statement_holding_an_emitting_statement_covers_the_output(self):
        # The counterpart: descent has to find a real emitter, including one below a `catch` clause,
        # whose block sits a node deeper than every other body.
        for source in (
            'if ($a) { 42 }',
            'if ($a) { $Null = 1 } else { 42 }',
            'foreach ($i in $x) { $i }',
            'while ($a) { 42 }',
            'switch ($a) { 1 { 42 } }',
            'try { } catch { 42 }',
            'try { } finally { 42 }',
            'if ($a) { foreach ($i in $x) { 42 } }',
        ):
            with self.subTest(source):
                self.assertTrue(output_is_covered(list(self._parse(source).body)))

    def test_a_redirection_of_the_output_stream_stops_it_covering(self):
        # Sending output to a file or merging it into another stream puts it where the enclosing
        # body cannot see it. Reading the wrong end of a merge inverts the answer, so both
        # directions are pinned: `1>&2` silences emission and `2>&1` leaves it alone.
        for source, covers in (
            (r'Get-Item x > C:\log.txt' , False),  # noqa
            (r'Get-Item x >> C:\log.txt', False),  # noqa
            (r'Get-Item x *> C:\log.txt', False),  # noqa
            (r'Get-Item x 1>&2'         , False),  # noqa
            (r'Get-Item x 2>&1 > C:\log', False),  # noqa
            (r'Get-Item x 2>&1'         , True),   # noqa
            (r'Get-Item x 3>&1'         , True),   # noqa
            (r'Get-Item x 2> C:\err.txt', True),   # noqa
        ):
            with self.subTest(source):
                self.assertEqual(output_is_covered(list(self._parse(source).body)), covers)

    def test_a_statement_that_only_binds_does_not_cover_the_output(self):
        # An assignment yields nothing to the pipeline, so it cannot stand in for the value a
        # `RETURNING` body exists to produce. Same for the named `data` section, which is an
        # assignment in block clothing — `data d { 42 }` binds `$d` rather than emitting `42`.
        for source in ('$x = 1', '$x += 1', '$a, $b = 1, 2', 'data d { 42 }'):
            with self.subTest(source):
                self.assertFalse(output_is_covered(list(self._parse(source).body)))

    def test_an_unnamed_data_section_does_emit(self):
        self.assertTrue(output_is_covered(list(self._parse('data { 42 }').body)))

    def test_emit_safety_reads_only_the_sequence_it_is_given(self):
        # The contract that used to be broken: a caller holds statements hoisted out of a block it
        # just pruned, whose `parent` still points at the block they came from, and statements that
        # are not parented into any body yet. The verdict has to be the same either way.
        for source in (
            'function f { Write-Host hi; 42 }',
            'function f { function g { Write-Host hi } }',
            'function f { }',
            '&{ if ($true) { Write-Host hi }; 42 }',
        ):
            with self.subTest(source):
                block = self._first(source, Ps1ScriptBlock)
                survivors = list(block.body)
                before = (
                    output_is_covered(survivors),
                    pruning_erases_body(BodyRole.SCRIPT, survivors),
                )
                for statement in survivors:
                    _remove_from_parent(statement)
                after = (
                    output_is_covered(survivors),
                    pruning_erases_body(BodyRole.SCRIPT, survivors),
                )
                self.assertEqual(before, after)

    def test_the_erasure_guard_answers_per_candidate_set(self):
        # `[Void]1; 42` at script root. The dead-code pass prunes only pure constants, so the
        # `[Void]1` survives and dropping `42` is allowed; the junk pass also removes the discard,
        # so nothing would survive and it must decline. One shared guard, two candidate sets, two
        # answers — the passes are not interchangeable. Pinned so that unifying them is a decision.
        script = self._parse('[Void]1\n42')
        constants = {
            statement for statement in script.body
            if isinstance(statement, Ps1ExpressionStatement)
            and is_pure_constant(statement.expression)
        }
        junk = {
            statement for statement in script.body
            if self._effect(statement) is not StatementEffect.EFFECT
        }
        self.assertEqual(len(constants), 1)
        self.assertEqual(len(junk), 2)
        self.assertFalse(pruning_erases_body(
            BodyRole.SCRIPT, [s for s in script.body if s not in constants]))
        self.assertTrue(pruning_erases_body(
            BodyRole.SCRIPT, [s for s in script.body if s not in junk]))

    def test_a_discard_never_covers_a_bodys_output(self):
        # A discard idiom emits nothing whatever its operand costs, so it cannot stand in for the
        # value a `RETURNING` body exists to produce — counting it silences the body.
        for source in (
            'function f { [Void](Start-Process notepad) }',
            'function f { [Void]$sb.Append(1) }',
            'function f { $Null = Start-Process notepad }',
            'function f { Get-Item x | Out-Null }',
        ):
            with self.subTest(source):
                block = self._first(source, Ps1ScriptBlock)
                self.assertFalse(output_is_covered(list(block.body)))

    def test_a_statement_that_acts_still_covers_the_output(self):
        for source in ('function f { Write-Host hi }', 'function f { Get-Item x }'):
            with self.subTest(source):
                block = self._first(source, Ps1ScriptBlock)
                self.assertTrue(output_is_covered(list(block.body)))

    def test_a_named_block_body_is_never_inert(self):
        # The parser fills either `body` or the named blocks, so an advanced function reports an
        # empty statement list. Reading that as "nothing happens here" deletes the function.
        for source in (
            'function f { process { Start-Process notepad } }',
            'function f { begin { Start-Process notepad } }',
            'function f { end { Start-Process notepad } }',
            'function f { param($a) process { Write-Host $a } }',
        ):
            with self.subTest(source):
                self.assertFalse(self._inert(self._first(source, Ps1FunctionDefinition).body))

    def test_a_parameter_block_is_code_the_call_runs(self):
        # A parameter default is evaluated on every call that omits the argument, and `get_body`
        # reports none of it. Reading the empty statement list as "nothing happens here" deletes the
        # function together with the call that runs the command in its default.
        for source in (
            'function f { param($x = (Start-Process notepad)) }',
            'function f { param($x = $(Remove-Item C:\\important)) }',
            'function f { param([ValidateScript({ Start-Process notepad })]$x) }',
            'function f { param([Parameter(Mandatory)]$x) }',
        ):
            with self.subTest(source):
                self.assertFalse(self._inert(self._first(source, Ps1FunctionDefinition).body))

    def test_a_parameter_block_that_only_declares_names_runs_nothing(self):
        # Declaring a parameter binds storage and evaluates nothing, so a junk function keeps being
        # removable once its body is pruned away. Only a default value or an attribute is code.
        for source in (
            'function f($a) { }',
            'function j($x) { $Null = 915 }',
            'function f { param($x, $y = 1) }',
            'function f { param([String]$x) }',
        ):
            with self.subTest(source):
                self.assertTrue(self._inert(self._first(source, Ps1FunctionDefinition).body))

    def test_a_data_section_captures_the_block_it_binds(self):
        # `data d { 42 }` binds the block's value to `$d`, so pruning into it is as destructive as
        # pruning into `$(...)`.
        block = self._first('data d { 42 }', Ps1DataSection).body
        self.assertIs(body_role(block), BodyRole.OPAQUE)

    def test_a_body_of_pure_discards_is_inert(self):
        for source in ('function j { $Null = 915 }', 'function j { }', 'function j { [Void]1 }'):
            with self.subTest(source):
                self.assertTrue(self._inert(self._first(source, Ps1FunctionDefinition).body))

    def test_a_body_that_emits_or_acts_is_not_inert(self):
        for source in ('function j { Write-Host hi }', 'function j { 42 }', 'function j { $x++ }'):
            with self.subTest(source):
                self.assertFalse(self._inert(self._first(source, Ps1FunctionDefinition).body))

    def test_a_definition_without_a_body_is_inert(self):
        self.assertTrue(self._inert(None))


class TestPs1OpenWorldNameTrust(Ps1EffectsTest):
    """
    A world that is open is not merely a statement about the type system: every opener — a
    dot-sourced file, an imported module, an `iex`, an item cmdlet writing the `function:` provider,
    an opaque dispatch — can bind an arbitrary command name to code this tree does not contain. The
    shadow set holds only the redefinitions written where the classifier can see them, so an open
    world has to withdraw name trust wholesale or the two facts contradict each other: the world
    reports that any name may have been rebound while the gate keeps granting the built-in's purity.
    """

    #: A world nothing was proven to redefine, but in which something can redefine anything.
    OPEN = TypeOracle(world=Ps1TypeWorld(False, frozenset()))

    def test_an_open_world_grants_no_command_purity(self):
        for source in ('Get-Date', 'New-Object System.Version', '(Get-Date)'):
            with self.subTest(source):
                self.assertTrue(is_side_effect_free(self._expression(source), _CLOSED_WORLD))
                self.assertFalse(is_side_effect_free(self._expression(source), self.OPEN))

    def test_an_open_world_has_no_discarding_pipeline_sink(self):
        for source in ('1 | Out-Null', '1 | ForEach-Object { [Void]$_ }'):
            with self.subTest(source):
                self.assertIs(
                    statement_effect(self._statement(source), _CLOSED_WORLD),
                    StatementEffect.DISCARD)
                self.assertIs(
                    statement_effect(self._statement(source), self.OPEN), StatementEffect.EFFECT)

    def test_an_oracle_without_a_world_withholds_name_trust_too(self):
        # The two questions the oracle answers must fail in the same direction, or a caller that
        # forgets the world gets a member grant refused and a name grant handed to it.
        self.assertFalse(is_side_effect_free(self._expression('Get-Date'), _NO_WORLD))
