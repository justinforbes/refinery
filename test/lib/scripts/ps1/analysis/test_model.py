from __future__ import annotations

from test import TestBase

from refinery.lib.scripts.ps1.analysis.model import (
    Ps1OccurrenceRole,
    ScopeKind,
    build_semantic_model,
    declares_binding,
    is_mutated_in_place,
    is_substitutable_position,
    is_write_occurrence,
    observes_previous_value,
    occurrence_role,
    replaces_value,
)
from refinery.lib.scripts.ps1.model import Ps1AssignmentExpression, Ps1Variable
from refinery.lib.scripts.ps1.parser import Ps1Parser


class TestPs1SemanticModel(TestBase):

    @staticmethod
    def _model(source: str):
        return build_semantic_model(Ps1Parser(source).parse())

    @staticmethod
    def _assignment_value(ast, target_name: str):
        for node in ast.walk():
            if isinstance(node, Ps1AssignmentExpression) and isinstance(node.target, Ps1Variable):
                if node.target.name.lower() == target_name:
                    return node.value
        raise AssertionError(F'no assignment to ${target_name}')

    def _script_binding(self, source: str, name: str):
        model = self._model(source)
        return model, model.script_scope.bindings.get(name)

    def test_script_root_is_a_script_scope(self):
        model = self._model("$a = 1\nWrite-Host $a")
        self.assertIs(model.script_scope.kind, ScopeKind.SCRIPT)

    def test_if_and_loop_bodies_introduce_no_scope(self):
        model = self._model("$a = 1\nif ($a) { $b = 2 }\nwhile ($a) { $c = 3 }")
        self.assertEqual(model.script_scope.children, [])
        self.assertIn('b', model.script_scope.bindings)
        self.assertIn('c', model.script_scope.bindings)

    def test_function_body_is_a_function_scope(self):
        model = self._model("function f { $x = 1 }")
        self.assertEqual(len(model.script_scope.children), 1)
        self.assertIs(model.script_scope.children[0].kind, ScopeKind.FUNCTION)

    def test_bare_scriptblock_is_a_scriptblock_scope(self):
        model = self._model("$cb = { $x = 1 }")
        self.assertEqual(len(model.script_scope.children), 1)
        self.assertIs(model.script_scope.children[0].kind, ScopeKind.SCRIPTBLOCK)

    def test_unread_variable_is_dead(self):
        _, binding = self._script_binding("$x = 'hi'\nWrite-Host done", 'x')
        self.assertIsNotNone(binding)
        assert binding is not None
        self.assertTrue(binding.is_dead)

    def test_read_variable_is_live(self):
        _, binding = self._script_binding("$x = 'hi'\nWrite-Host $x", 'x')
        assert binding is not None
        self.assertFalse(binding.is_dead)

    def test_read_inside_function_keeps_outer_binding_live(self):
        # PowerShell dynamic scoping: a bare read inside a function references the caller's
        # variable, so the outer store the function might observe stays live.
        _, binding = self._script_binding(
            "$x = 'payload'\nfunction Run { iex $x }\nRun", 'x')
        assert binding is not None
        self.assertFalse(binding.is_dead)

    def test_qualified_read_keeps_binding_live(self):
        # A $script:x read resolves to the script-scope binding and keeps it live.
        _, binding = self._script_binding(
            "$x = 'keepme'\nfunction f { Write-Host $script:x }\nf", 'x')
        assert binding is not None
        self.assertFalse(binding.is_dead)

    def test_local_qualified_read_resolves_in_the_scope_of_the_reference(self):
        # $local: names the scope of the reference itself — the same scope a $local: write
        # binds in — so the read must keep the function-local binding live, not the script one.
        model = self._model('function f { $y = 1; Write-Host $local:y }')
        binding = model.script_scope.children[0].bindings['y']
        self.assertFalse(binding.is_dead)

    def test_class_property_declaration_is_not_a_read(self):
        # A class property declares a member, a namespace distinct from the script's variables,
        # so it must not keep a script variable of the same name alive.
        model = self._model('$x = 1\nclass C { [int]$x }')
        self.assertTrue(model.script_scope.bindings['x'].is_dead)

    def test_scriptblock_assignment_is_write_local(self):
        # A bare assignment inside a scriptblock creates a scriptblock-local binding, distinct
        # from the enclosing $inner; it does not add a write to the outer binding.
        model = self._model("$inner = 1\n$cb = { $inner = 99 }\nWrite-Host $inner")
        outer = model.script_scope.bindings['inner']
        block_scope = model.script_scope.children[0]
        self.assertIn('inner', block_scope.bindings)
        self.assertIsNot(outer, block_scope.bindings['inner'])
        self.assertEqual(len(outer.writes), 1)

    def test_reads_in_scope_sees_read_nested_in_scriptblock(self):
        # The reconciliation: a read of a script variable nested inside a captured scriptblock
        # is seen by the read set the dead-store sweep flushes against, so the store survives.
        ast = Ps1Parser("$x = 1\n$arr = @( { Write-Host $x } )").parse()
        model = build_semantic_model(ast)
        value = self._assignment_value(ast, 'arr')
        self.assertIn('x', model.reads_in_scope(value, model.script_scope))

    def test_reads_in_scope_sees_compound_assignment_target(self):
        # A `+=` target observes the variable before writing it, unlike a plain `=` target, which
        # replaces the value unread.
        ast = Ps1Parser("$x = 1\n$a = ($x += 1)").parse()
        model = build_semantic_model(ast)
        value = self._assignment_value(ast, 'a')
        self.assertIn('x', model.reads_in_scope(value, model.script_scope))

    def test_reads_in_scope_ignores_write_only_scriptblock(self):
        # A scriptblock that only assigns (write-local) is not a read of the outer variable.
        ast = Ps1Parser("$inner = 1\n$cb = { $inner = 99 }").parse()
        model = build_semantic_model(ast)
        value = self._assignment_value(ast, 'cb')
        self.assertNotIn('inner', model.reads_in_scope(value, model.script_scope))

    def test_environment_variable_is_a_distinct_binding(self):
        # $env:X and $X are different namespaces: an unread env write is dead, an unread script
        # write keyed the same name is independently tracked.
        model = self._model("$env:X = 'v'\n$X = 1\nWrite-Host $X")
        self.assertIn('env:x', model.script_scope.bindings)
        self.assertIn('x', model.script_scope.bindings)
        self.assertTrue(model.script_scope.bindings['env:x'].is_dead)
        self.assertFalse(model.script_scope.bindings['x'].is_dead)

    def test_scope_of_maps_nested_node_to_its_scriptblock(self):
        model = self._model("function f { $x = 1 }")
        function_scope = model.script_scope.children[0]
        inner = next(
            n for n in model.root.walk()
            if isinstance(n, Ps1Variable) and n.name.lower() == 'x')
        self.assertIs(model.scope_of(inner), function_scope)


class TestPs1WriteOccurrenceKinds(TestBase):
    """
    Which write positions also read the variable. A caller that removes a write on the grounds that
    nothing reads the binding any more has to count these, and the difference between `=` and `+=`
    is the whole of it.
    """

    @staticmethod
    def _write(source: str, name: str = 'x') -> Ps1Variable:
        for node in Ps1Parser(source).parse().walk():
            if isinstance(node, Ps1Variable) and node.name.lower() == name:
                if is_write_occurrence(node):
                    return node
        raise AssertionError(F'no write of ${name}')

    def test_a_plain_assignment_replaces_without_observing(self):
        write = self._write("$x = 'a'")
        self.assertTrue(replaces_value(write))
        self.assertFalse(observes_previous_value(write))

    def test_a_compound_assignment_observes_the_previous_value(self):
        for source in ("$x += 'a'", '$x -= 1', '$x *= 2', '$x /= 2', '$x %= 2'):
            with self.subTest(source):
                write = self._write(source)
                self.assertFalse(replaces_value(write))
                self.assertTrue(observes_previous_value(write))

    def test_an_increment_observes_the_previous_value(self):
        for source in ('$x++', '$x--'):
            with self.subTest(source):
                self.assertTrue(observes_previous_value(self._write(source)))

    def test_a_value_handed_in_from_outside_observes_nothing(self):
        for source in ('foreach ($x in 1, 2) { }', 'function f { param($x) }'):
            with self.subTest(source):
                write = self._write(source)
                self.assertFalse(replaces_value(write))
                self.assertFalse(observes_previous_value(write))

    @staticmethod
    def _read(source: str, name: str = 'x') -> Ps1Variable:
        for node in Ps1Parser(source).parse().walk():
            if isinstance(node, Ps1Variable) and node.name.lower() == name:
                if not is_write_occurrence(node):
                    return node
        raise AssertionError(F'no read of ${name}')

    def test_an_assignment_through_a_part_of_a_variable_mutates_it_in_place(self):
        for source in (
            "$x[0] = 'z'",
            '$x.Length = 5',
            "$x[0][1] = 'z'",
            '$x.A.B = 5',
            "($x)[0] = 'z'",
            "([array]$x)[0] = 'z'",
            "$x[0], $x[1] = 'p', 'q'",
            "$x[0] += 'z'",
        ):
            with self.subTest(source):
                self.assertTrue(is_mutated_in_place(self._read(source)))

    def test_a_variable_an_assignment_does_not_store_through_is_not_mutated_in_place(self):
        for source in (
            'Write-Host $x[0]',
            'Write-Host $x.Length',
            "$a[$x] = 'z'",
            "$a[0] = $x",
        ):
            with self.subTest(source):
                self.assertFalse(is_mutated_in_place(self._read(source)))

    def test_a_variable_an_assignment_replaces_outright_is_not_mutated_in_place(self):
        for source in ("$x = 'a'", "$x, $y = 'p', 'q'", "[string]$x = 'a'"):
            with self.subTest(source):
                self.assertFalse(is_mutated_in_place(self._write(source)))


class TestPs1OccurrenceRoles(TestBase):
    """
    What each occurrence of a name does to the value it holds, and the two questions that are not
    the role: whether a value may be installed in its place, and whether it brings the binding into
    existence. Every transform used to assemble its own answer from a handful of positional
    predicates, and `[ref]$n` came out a plain read in all of them at once.
    """

    @staticmethod
    def _occurrence(source: str, name: str = 'x', index: int = 0) -> Ps1Variable:
        found = [
            node for node in Ps1Parser(source).parse().walk()
            if isinstance(node, Ps1Variable) and node.name.lower() == name
        ]
        if len(found) <= index:
            raise AssertionError(F'no occurrence {index} of ${name} in {source!r}')
        return found[index]

    def _role(self, source: str, name: str = 'x', index: int = 0) -> Ps1OccurrenceRole:
        return occurrence_role(self._occurrence(source, name, index))

    def test_an_occurrence_that_only_observes_the_value_is_a_read(self):
        for source in ('Write-Host $x', '$y = $x + 1', 'Write-Host $x[0]', 'Get-Item @x'):
            with self.subTest(source):
                self.assertIs(self._role(source), Ps1OccurrenceRole.READ)

    def test_an_occurrence_handed_a_value_from_outside_replaces_it(self):
        for source in (
            "$x = 'a'",
            "[string]$x = 'a'",
            "$x, $y = 'p', 'q'",
            'foreach ($x in 1, 2) { }',
            'function f { param($x) }',
        ):
            with self.subTest(source):
                self.assertIs(self._role(source), Ps1OccurrenceRole.WRITE_REPLACING)

    def test_an_occurrence_that_reads_what_it_writes_observes_the_previous_value(self):
        for source in ("$x += 'a'", '$x -= 1', '$x++', '$x--'):
            with self.subTest(source):
                self.assertIs(self._role(source), Ps1OccurrenceRole.WRITE_OBSERVING)

    def test_a_reference_the_callee_may_store_through_observes_the_previous_value(self):
        """
        `[Int]::TryParse($s, [ref]$n)` assigns `$n`. The occurrence is a write however it reads, and
        reading it as anything else lets a value from before the call reach a use after it.
        """
        for source in (
            "[void][int]::TryParse('7', [ref]$x)",
            "[void][int]::TryParse('7', ([ref]$x))",
            "[void][int]::TryParse('7', [management.automation.psreference]$x)",
        ):
            with self.subTest(source):
                self.assertIs(self._role(source), Ps1OccurrenceRole.WRITE_OBSERVING)

    def test_an_occurrence_an_assignment_stores_through_keeps_its_own_value(self):
        for source in ("$x[0] = 'z'", '$x.Length = 5', "$x[0][1] = 'z'", "$x[0] += 'z'"):
            with self.subTest(source):
                self.assertIs(self._role(source), Ps1OccurrenceRole.WRITE_THROUGH)

    def test_a_class_property_declaration_references_no_variable(self):
        self.assertIs(
            self._role('class C { [int]$x }'), Ps1OccurrenceRole.NOT_A_REFERENCE)

    def test_a_plain_read_is_the_only_position_a_value_may_be_installed_in(self):
        self.assertTrue(is_substitutable_position(self._occurrence('Write-Host $x')))
        for source in (
            "$x = 'a'",
            "$x[0] = 'z'",
            "[void][int]::TryParse('7', [ref]$x)",
            "$x += 'a'",
        ):
            with self.subTest(source):
                self.assertFalse(is_substitutable_position(self._occurrence(source)))

    def test_a_splatted_read_is_not_a_position_a_value_may_be_installed_in(self):
        """
        `@x` spreads an array over a command's parameters, so `Get-Item @x` with `$x` holding
        `'-Path', 'C:\\'` binds `-Path`. The array written in its place is one positional argument
        instead, which is a different command.
        """
        splatted = self._occurrence('Get-Item @x')
        self.assertIs(occurrence_role(splatted), Ps1OccurrenceRole.READ)
        self.assertFalse(is_substitutable_position(splatted))

    def test_every_write_but_a_reference_brings_the_binding_into_existence(self):
        for source in ("$x = 'a'", "$x += 'a'", '$x++', 'foreach ($x in 1, 2) { }'):
            with self.subTest(source):
                self.assertTrue(declares_binding(self._occurrence(source)))
        for source in ("[void][int]::TryParse('7', [ref]$x)", 'Write-Host $x', "$x[0] = 'z'"):
            with self.subTest(source):
                self.assertFalse(declares_binding(self._occurrence(source)))


class TestPs1ReferenceAttribution(TestBase):
    """
    Where a `[ref]$n` occurrence lands in the model. It is resolved the way a read is — PowerShell
    looks the name up and creates nothing — and recorded the way a write is, because the callee
    stores through it.
    """

    @staticmethod
    def _model(source: str):
        return build_semantic_model(Ps1Parser(source).parse())

    def test_a_reference_is_a_write_of_the_binding_and_not_a_read(self):
        model = self._model("$n = 0\n[void][int]::TryParse('7', [ref]$n)")
        binding = model.script_scope.bindings['n']
        self.assertEqual(len(binding.writes), 2)
        self.assertEqual(len(binding.reads), 0)

    def test_a_reference_keeps_the_binding_alive_although_nothing_reads_it(self):
        """
        The store the callee performs is only observable through the binding, so a pass that deletes
        `$n = 0` for having no readers deletes the storage the call writes into.
        """
        model = self._model("$n = 0\n[void][int]::TryParse('7', [ref]$n)")
        binding = model.script_scope.bindings['n']
        self.assertEqual(len(binding.uses), 1)
        self.assertFalse(binding.is_dead)

    def test_a_reference_inside_a_body_declares_nothing_there(self):
        """
        `[ref]$n` resolves by ordinary lookup, so the name it names is the enclosing one. A local
        binding invented for it would hide the outer binding the callee actually stores through.
        """
        model = self._model("$n = 0\nfunction f { [void][int]::TryParse('7', [ref]$n) }")
        function_scope = model.script_scope.children[0]
        self.assertIs(function_scope.kind, ScopeKind.FUNCTION)
        self.assertNotIn('n', function_scope.bindings)
        self.assertEqual(len(model.script_scope.bindings['n'].writes), 2)

    def test_a_binding_whose_only_use_is_a_compound_assignment_is_not_dead(self):
        """
        `Binding.reads` holds no occurrence here and the value is observed all the same, which is
        the shape `Binding.uses` exists for.
        """
        model = self._model("$x = 'a'\n$x += 'b'")
        binding = model.script_scope.bindings['x']
        self.assertEqual(len(binding.reads), 0)
        self.assertEqual(len(binding.uses), 1)
        self.assertFalse(binding.is_dead)


class TestPs1NamedReferenceAttribution(TestBase):
    """
    Where a name addressed as a string lands in the model. A recognizer that identifies
    `Set-Variable x` while the model files the reference on the wrong binding — or on none — reads
    as working and corrupts exactly as before, so these assert the binding, its scope and its
    occurrence counts rather than the recognizer's own answer.
    """

    @staticmethod
    def _model(source: str):
        return build_semantic_model(Ps1Parser(source).parse())

    def test_a_named_write_creates_the_binding_when_nothing_else_mentions_it(self):
        """
        `$p` appears nowhere as a variable, so unless the census is consulted while the model is
        built there is no binding for the reference to be filed against.
        """
        model = self._model('Get-Process -OutVariable p')
        binding = model.script_scope.bindings['p']
        self.assertEqual(len(binding.writes), 1)
        self.assertFalse(binding.is_read)

    def test_a_named_write_lands_on_the_binding_the_assignments_use(self):
        model = self._model("$x = 'calc'\nGet-Process -OutVariable x")
        binding = model.script_scope.bindings['x']
        self.assertEqual(len(binding.writes), 2)

    def test_a_named_read_is_a_use_that_keeps_the_binding_alive(self):
        model = self._model("$a = 'x'\nGet-Variable a")
        binding = model.script_scope.bindings['a']
        self.assertEqual(len(binding.reads), 1)
        self.assertFalse(binding.is_dead)

    def test_an_appending_out_variable_observes_the_previous_value(self):
        model = self._model("$a = 'x'\nGet-Process -OutVariable +a")
        binding = model.script_scope.bindings['a']
        self.assertEqual(len(binding.uses), 1)
        self.assertFalse(binding.is_dead)

    def test_a_bare_named_write_in_a_body_binds_that_body_and_not_the_script(self):
        """
        Measured: `Set-Variable d 'INNER'` inside a function writes the function's own scope. Filing
        it at the script scope would let a later read of the caller's `$d` see a value the function
        never gave it.
        """
        model = self._model("$d = 'OUTER'\nfunction f { Set-Variable d 'INNER' }")
        function_scope = model.script_scope.children[0]
        self.assertIn('d', function_scope.bindings)
        self.assertEqual(len(function_scope.bindings['d'].writes), 1)
        self.assertEqual(len(model.script_scope.bindings['d'].writes), 1)

    def test_a_globally_scoped_named_write_binds_the_script_scope(self):
        model = self._model("function f { Set-Variable d 'V' -Scope Global }")
        self.assertIn('d', model.script_scope.bindings)
        self.assertNotIn('d', model.script_scope.children[0].bindings)

    def test_an_environment_item_write_binds_the_key_the_variable_reads_under(self):
        model = self._model("Set-Item Env:ComSpec 'evil'\nWrite-Host $env:ComSpec")
        binding = model.script_scope.bindings['env:comspec']
        self.assertEqual(len(binding.writes), 1)
        self.assertEqual(len(binding.reads), 1)

    def test_an_unbinding_replaces_the_value_rather_than_observing_it(self):
        """
        `Remove-Variable a` does not read `$a`, it removes it. Recording it as an occurrence that
        observes the value would make the assignment before it look like something the removal still
        needs, and would keep a store alive that nothing reads.
        """
        model = self._model("$a = 'x'\nRemove-Variable a")
        roles = [write.role for write in model.script_scope.bindings['a'].writes]
        self.assertIn(Ps1OccurrenceRole.WRITE_REPLACING, roles)
        self.assertNotIn(Ps1OccurrenceRole.READ, roles)
        self.assertNotIn(Ps1OccurrenceRole.WRITE_OBSERVING, roles)

    def test_a_computed_name_this_cannot_place_puts_the_scope_it_writes_in_doubt(self):
        """
        `-Scope Global` reaches the script scope out of any body, at a moment nothing here orders,
        so every binding of that scope is in doubt for as long as the tree stands.

        A computed name landing in the scope it is *written* in is deliberately not recorded here.
        That write happens at a point, and `Ps1VariableFlow.unattributable_writes` holds it there,
        which is what leaves the reads before it answerable.
        """
        model = self._model("$x = 'a'\nSet-Variable $n 'b' -Scope Global")
        self.assertTrue(model.script_scope.writes_unreadable_names)
        model = self._model("$x = 'a'\nSet-Variable $n 'b'")
        self.assertFalse(model.script_scope.writes_unreadable_names)

    def test_a_literal_named_write_leaves_the_scope_out_of_doubt(self):
        model = self._model("$x = 'a'\nSet-Variable y 'b'")
        self.assertFalse(model.script_scope.writes_unreadable_names)

    def test_a_scope_the_lexical_chain_cannot_name_puts_the_script_in_doubt(self):
        """
        Measured: `-Scope 1` writes the caller's scope, which is not an ancestor of anything here,
        so no binding can be shown to be safe from it.
        """
        model = self._model("function f { Set-Variable x 'b' -Scope 1 }")
        self.assertTrue(model.script_scope.writes_unreadable_names)
