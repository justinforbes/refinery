from __future__ import annotations

from test import TestBase

from refinery.lib.scripts.ps1.analysis.types import TypeOracle, resolve_expression_type
from refinery.lib.scripts.ps1.analysis.world import Ps1TypeWorld
from refinery.lib.scripts.ps1.data import resolve_type
from refinery.lib.scripts.ps1.model import Ps1ExpressionStatement
from refinery.lib.scripts.ps1.parser import Ps1Parser


class Ps1TypeOracleTest(TestBase):

    #: The world every script that redefines nothing and runs no opaque code has. Typing questions
    #: are asked through it because a command name is only trustworthy in a closed world, so an
    #: oracle carrying no world at all answers nothing and would test the absence, not the typing.
    CLOSED = Ps1TypeWorld(True, frozenset())

    @staticmethod
    def _expr(source: str):
        statement = Ps1Parser(source).parse().body[0]
        assert isinstance(statement, Ps1ExpressionStatement)
        assert statement.expression is not None
        return statement.expression


class TestPs1TypeOracleCandidates(Ps1TypeOracleTest):
    """
    The candidate set is the primitive the member gate reasons over: a conclusion about a value has
    to hold for every type the value could carry, so what matters is that a genuinely multi-valued
    result is reported as such and an unknowable one as the empty set, never as a single guess.
    """

    def setUp(self):
        self.oracle = TypeOracle(world=self.CLOSED)

    def test_a_cmdlet_contributes_every_output_type_it_declares(self):
        # Get-Date carries [OutputType([datetime], [string])]; both are candidates, since which one
        # a call yields depends on its arguments.
        self.assertEqual(
            self.oracle.candidate_types(self._expr('(Get-Date)')),
            frozenset({resolve_type('System.DateTime'), resolve_type('System.String')}),
        )

    def test_an_untrusted_cmdlet_contributes_no_candidates(self):
        # [OutputType] is only a lower bound: a command that forwards its input emits types it never
        # declares, so the oracle trusts a declaration only for a curated closed set. Get-Random
        # forwards -InputObject and Get-Command's declaration is not vouched for, so both are left
        # untyped — which is what keeps a member read on their result (e.g. (Get-Command X).Name).
        self.assertEqual(self.oracle.candidate_types(self._expr('Get-Random')), frozenset())
        self.assertEqual(self.oracle.candidate_types(self._expr('Get-Command')), frozenset())

    def test_a_static_call_takes_the_return_its_overloads_agree_on(self):
        # GetCurrentProcess has one overload returning Process; the single-type ladder cannot type a
        # call at all, so this is a return the oracle resolves and resolve_expression_type does not.
        self.assertEqual(
            self.oracle.candidate_types(self._expr('[Diagnostics.Process]::GetCurrentProcess()')),
            frozenset({resolve_type('System.Diagnostics.Process')}),
        )

    def test_a_static_call_with_disagreeing_overloads_is_empty(self):
        # Math.Max is overloaded across every numeric type; with no argument typing the return is
        # not decidable, so the oracle reports nothing rather than picking one.
        self.assertEqual(self.oracle.candidate_types(self._expr('[Math]::Max(1, 2)')), frozenset())

    def test_a_static_call_surfaces_an_imprecise_supertype_return(self):
        # Convert.ChangeType is declared to return Object; the oracle reports that supertype as-is.
        # Narrowing it is not the oracle's job — a caller that must not act on a supertype is the
        # one that neutralises it.
        self.assertEqual(
            self.oracle.candidate_types(self._expr('[Convert]::ChangeType($x, [int])')),
            frozenset({resolve_type('System.Object')}),
        )

    def test_an_instance_method_call_is_not_resolved(self):
        # String.Substring returns a string, but selecting an instance overload needs the receiver
        # type and argument types; that is deferred, so the call contributes no candidate.
        self.assertEqual(self.oracle.candidate_types(self._expr('$s.Substring(0, 2)')), frozenset())

    def test_a_command_without_a_declared_output_is_empty(self):
        # Write-Host carries no [OutputType]; an undeclared output is unknown, not empty, so the
        # oracle contributes nothing rather than reading absence as "emits nothing".
        self.assertEqual(self.oracle.candidate_types(self._expr('Write-Host x')), frozenset())

    def test_single_type_forms_are_delegated_to_the_free_function(self):
        for source in ("'abc'", '42', 'New-Object Net.WebClient', '[int]'):
            with self.subTest(source):
                single = resolve_expression_type(self._expr(source))
                self.assertIsNotNone(single)
                self.assertEqual(self.oracle.candidate_types(self._expr(source)), frozenset({single}))

    def test_an_untyped_variable_is_empty_without_a_model(self):
        # The empty oracle carries no variable typing; a bare local resolves to nothing until a
        # model populates the oracle. Its automatic-variable typing still applies, so this is a
        # plain local.
        self.assertEqual(self.oracle.candidate_types(self._expr('$notavariabletype')), frozenset())

    def test_a_populated_oracle_resolves_a_typed_variable(self):
        # The typing a model supplies is what an empty oracle lacks: given it, the same read the
        # empty oracle cannot type now carries the variable's type.
        webclient = resolve_type('System.Net.WebClient')
        assert webclient is not None
        oracle = TypeOracle({'client': webclient}, self.CLOSED)
        self.assertEqual(
            oracle.candidate_types(self._expr('$client')), frozenset({webclient}))
        self.assertEqual(self.oracle.candidate_types(self._expr('$client')), frozenset())


class TestPs1TypeOracleResolve(Ps1TypeOracleTest):
    """
    The single-type view is the candidate set collapsed, and its contract with the free function it
    generalizes is load-bearing: it must agree wherever the free function commits to a type, and only
    ever resolve more, so a consumer of the narrower answer is never silently handed a wider one.
    """

    def setUp(self):
        self.oracle = TypeOracle(world=self.CLOSED)

    def test_a_lone_candidate_is_the_answer(self):
        self.assertEqual(
            self.oracle.resolve(self._expr('[Diagnostics.Process]::GetCurrentProcess()')),
            resolve_type('System.Diagnostics.Process'),
        )

    def test_an_ambiguous_or_unknown_expression_is_none(self):
        self.assertIsNone(self.oracle.resolve(self._expr('(Get-Date)')))
        self.assertIsNone(self.oracle.resolve(self._expr('[Math]::Max(1, 2)')))
        self.assertIsNone(self.oracle.resolve(self._expr('$undecidable')))

    def test_resolve_agrees_with_the_free_function_where_it_commits(self):
        for source in ("'abc'", '42', 'New-Object Net.WebClient', '[int]', '[datetime]$t'):
            with self.subTest(source):
                single = resolve_expression_type(self._expr(source))
                self.assertIsNotNone(single)
                self.assertEqual(self.oracle.resolve(self._expr(source)), single)

    def test_resolve_is_a_strict_superset_of_the_free_function(self):
        # The free function has no arm for a static method call and returns None; the oracle
        # resolves it. This is the widening that keeps resolve out of the transforms that depend on
        # the narrower answer.
        call = self._expr('[Diagnostics.Process]::GetCurrentProcess()')
        self.assertIsNone(resolve_expression_type(call))
        self.assertEqual(self.oracle.resolve(call), resolve_type('System.Diagnostics.Process'))


if __name__ == '__main__':
    import unittest
    unittest.main()
