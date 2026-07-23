from __future__ import annotations

from test import TestBase

from refinery.lib.scripts import _remove_from_parent
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
from refinery.lib.scripts.ps1.model import (
    Ps1ArrayExpression,
    Ps1DataSection,
    Ps1ExpressionStatement,
    Ps1FunctionDefinition,
    Ps1IfStatement,
    Ps1ScriptBlock,
    Ps1SubExpression,
)
from refinery.lib.scripts.ps1.parser import Ps1Parser


class Ps1EffectsTest(TestBase):

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
                self.assertTrue(is_side_effect_free(self._expression(source)))

    def test_expressions_that_change_the_world(self):
        for source in (
            'Start-Process notepad',
            'Remove-Item x',
            "[System.IO.File]::WriteAllText('a', 'b')",
            '$x++',
            '$s.Invoke()',
        ):
            with self.subTest(source):
                self.assertFalse(is_side_effect_free(self._expression(source)))

    def test_an_unrecognized_construct_is_assumed_impure(self):
        # The allow-list is the whole safety argument: anything it does not name has to come back
        # impure, however harmless it looks.
        for source in ('New-Object System.Net.WebClient', '& $f', '$obj.Frobnicate()'):
            with self.subTest(source):
                self.assertFalse(is_side_effect_free(self._expression(source)))

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
                self.assertIs(is_side_effect_free(self._expression(source)), pure)

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
                self.assertFalse(is_side_effect_free(self._expression(source)))

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
                self.assertFalse(is_side_effect_free(self._expression(source)))

    def test_a_member_invoking_foreach_has_no_body_to_vouch_for_it(self):
        # `ForEach-Object -MemberName Delete` calls that member on every input item. A body check
        # that proves a property of the scriptblocks it saw proves nothing when there are none.
        for source in (
            'Get-ChildItem | ForEach-Object -MemberName Delete',
            'Get-Process | ForEach-Object Kill',
            'Get-Process | ForEach-Object $handler',
        ):
            with self.subTest(source):
                self.assertFalse(is_side_effect_free(self._expression(source)))

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
                self.assertIs(is_side_effect_free(self._expression(source)), pure)

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
                self.assertIs(is_side_effect_free(self._expression(source)), pure)

    def test_a_redirection_writes_a_file_however_pure_the_command_is(self):
        for source in (
            'Get-Date > C:\\out.txt',
            'Get-Content a.txt >> b.txt',
            'Get-Process 2> C:\\err.txt',
        ):
            with self.subTest(source):
                self.assertFalse(is_side_effect_free(self._expression(source)))

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
                self.assertFalse(is_side_effect_free(self._expression(source)))


class TestPs1StatementEffect(Ps1EffectsTest):

    def test_a_statement_that_only_yields_a_value_is_output(self):
        for source in ('42', "'hi'", '$x', '1 + 1', 'Get-Date'):
            with self.subTest(source):
                self.assertIs(statement_effect(self._statement(source)), StatementEffect.OUTPUT)

    def test_a_statement_that_does_something_is_an_effect(self):
        for source in ('Write-Host hi', '$x = 1', '$x++', 'if ($a) { }'):
            with self.subTest(source):
                self.assertIs(statement_effect(self._statement(source)), StatementEffect.EFFECT)

    def test_the_discard_idioms_emit_nothing(self):
        for source in (
            '$Null = 5',
            '[Void]1',
            '1..3 | Out-Null',
            '1..3 | ForEach-Object { [Void]$_ }',
            '1..3 | ForEach-Object { $Null = $_ }',
        ):
            with self.subTest(source):
                self.assertIs(statement_effect(self._statement(source)), StatementEffect.DISCARD)

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
                self.assertIs(statement_effect(self._statement(source)), StatementEffect.EFFECT)

    def test_a_discard_of_a_harmless_value_stays_a_discard(self):
        for source in ('$Null = 5', '[Void]1', '[Void]$x', "[Void]('a' + 'b')"):
            with self.subTest(source):
                self.assertIs(statement_effect(self._statement(source)), StatementEffect.DISCARD)

    def test_a_pure_pipeline_cmdlet_still_yields_a_value_a_caller_may_want(self):
        # Purity and emission answer different questions: `Where-Object` performs no side effect,
        # yet the filtered value it puts on the pipeline is not junk.
        statement = self._statement('1..3 | Where-Object { $_ }')
        self.assertTrue(is_side_effect_free(self._expression('1..3 | Where-Object { $_ }')))
        self.assertIs(statement_effect(statement), StatementEffect.EFFECT)

    def test_pure_constants_are_a_strict_refinement_of_output(self):
        # The dead-code pass prunes only pure constants and the junk pass prunes the whole OUTPUT
        # set. That is only defensible while the candidate sets stay nested.
        for source in ('42', '-3', '(7)', '$Null', '$True', '$False', '+9', '3.5'):
            with self.subTest(source):
                self.assertTrue(is_pure_constant(self._expression(source)))
                self.assertIs(statement_effect(self._statement(source)), StatementEffect.OUTPUT)

    def test_a_string_literal_is_not_a_prunable_constant(self):
        # A bare string is very often the point of the script, so it is deliberately left out of the
        # constant set even though it is side-effect free.
        expression = self._expression("'hi'")
        self.assertTrue(is_side_effect_free(expression))
        self.assertFalse(is_pure_constant(expression))

    def test_a_computed_expression_is_not_a_constant(self):
        for source in ('1 + 1', '$x', '[Math]::Abs(-3)'):
            with self.subTest(source):
                self.assertFalse(is_pure_constant(self._expression(source)))


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
        for source in ('Write-Host hi', '42', 'if ($a) { }', '$x = 1'):
            with self.subTest(source):
                self.assertTrue(output_is_covered(list(self._parse(source).body)))

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
            if statement_effect(statement) is not StatementEffect.EFFECT
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
        for source in ('function f { Write-Host hi }', 'function f { $x = 1 }'):
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
                self.assertFalse(body_is_inert(self._first(source, Ps1FunctionDefinition).body))

    def test_a_data_section_captures_the_block_it_binds(self):
        # `data d { 42 }` binds the block's value to `$d`, so pruning into it is as destructive as
        # pruning into `$(...)`.
        block = self._first('data d { 42 }', Ps1DataSection).body
        self.assertIs(body_role(block), BodyRole.OPAQUE)

    def test_a_body_of_pure_discards_is_inert(self):
        for source in ('function j { $Null = 915 }', 'function j { }', 'function j { [Void]1 }'):
            with self.subTest(source):
                self.assertTrue(body_is_inert(self._first(source, Ps1FunctionDefinition).body))

    def test_a_body_that_emits_or_acts_is_not_inert(self):
        for source in ('function j { Write-Host hi }', 'function j { 42 }', 'function j { $x++ }'):
            with self.subTest(source):
                self.assertFalse(body_is_inert(self._first(source, Ps1FunctionDefinition).body))

    def test_a_definition_without_a_body_is_inert(self):
        self.assertTrue(body_is_inert(None))
