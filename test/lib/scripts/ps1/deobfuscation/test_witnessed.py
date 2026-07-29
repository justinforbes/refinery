from __future__ import annotations

import unittest

from collections.abc import Callable
from contextlib import ExitStack
from unittest.mock import patch

from test import TestBase
from test.lib.scripts.ps1 import test_deobfuscation
from test.lib.scripts.ps1.deobfuscation import (
    test_deadcode,
    test_emulator,
    test_folding,
    test_iexinline,
    test_removal,
    test_unused,
    test_wildcards,
)

from refinery.lib.scripts import owning_list
from refinery.lib.scripts.ps1.analysis import effects
from refinery.lib.scripts.ps1.analysis.effects import OutputSink, is_side_effect_free
from refinery.lib.scripts.ps1.deobfuscation import (
    deadcode,
    emulator,
    folding,
    removal,
    substitution,
    unused,
    wildcards,
)
from refinery.lib.scripts.ps1.model import Ps1ExpressionStatement


def _witness(witness: type) -> str:
    """
    The dotted name `unittest.TestLoader` resolves. Spelled from the class rather than by hand,
    because `unittest.loader.loadTestsFromNames` answers a name it cannot resolve with a test that
    *errors*, and a probe cannot tell that apart from a guard its witnesses noticed. Written this
    way a renamed witness is an `AttributeError` here, before any probe runs.
    """
    return F'{witness.__module__}.{witness.__qualname__}'


_DEAD_CODE = _witness(test_deadcode.TestPs1DeadCodeElimination)
_DEAD_CODE_EXTRA = _witness(test_deadcode.TestPs1DeadCodeExtra)
_DEAD_CODE_TREE = _witness(test_deadcode.TestPs1DeadCodeLeavesTheTreeConsistent)
_EMULATOR = _witness(test_emulator.TestPs1FunctionEvaluator)
_EMULATOR_EXTRA = _witness(test_emulator.TestPs1EmulatorExtra)
_HANDLER = _witness(test_removal.TestPs1DeadCodeEliminationDoesNotUnhookAHandler)
_IEX_REDIRECTIONS = _witness(test_iexinline.TestPs1IexRedirections)
_INTEGRATION = _witness(test_deobfuscation.TestPs1Integration)
_KEPT_EITHER_WAY = _witness(test_unused.TestPs1OutputSomethingElseHoldsIsKeptEitherWay)
_KEPT_WHEN_ASKED = _witness(test_unused.TestPs1BareOutputIsKeptWhenAsked)
_PLAN = _witness(test_removal.TestPs1RemovalPlan)
_REFERENCE = _witness(test_unused.TestPs1RemovalLeavesNoDanglingReference)
_SELECTION = _witness(test_folding.TestPs1SelectionKeepsWhatBuildingTheContainerDid)
_STRIPPED_BY_DEFAULT = _witness(test_unused.TestPs1BareOutputIsStrippedByDefault)
_TREE = _witness(test_unused.TestPs1RemovalLeavesTheTreeConsistent)
_UNUSED = _witness(test_unused.TestPs1UnusedVariableRemoval)
_WILDCARD_REDIRECTIONS = _witness(test_wildcards.TestPs1WildcardRedirections)


def _accepted_before_the_vetoes(plan: removal.Ps1RemovalPlan) -> list:
    return [proposal.statement for proposal in plan._proposals.values()]


def _plans_accepted_before_the_vetoes(plans: removal.Ps1RemovalPlans) -> list:
    accepted = [
        statement
        for plan in plans._plans.values()
        for statement in _accepted_before_the_vetoes(plan)
    ]
    accepted.extend(proposal.statement for proposal in plans._rewrites.values())
    return accepted


def _propose_repairing_only_a_replacement(
    self: removal.Ps1RemovalPlan,
    statement,
    replacement=None,
) -> None:
    self._proposals[id(statement)] = proposal = removal._Proposal(
        statement, list(replacement or ()))
    if proposal.replacement:
        removal.reattach(statement)


def _commit_interleaving_verdicts_and_edits(self: removal.Ps1RemovalPlans) -> bool:
    """
    `Ps1RemovalPlans.commit` with its verdicts left lazy, which is the whole mutation: read out of a
    generator inside the loop that edits, each plan decides after the plan before it has moved the
    tree. Kept a token away from the original so that a `commit` which grows a phase reads as stale
    here rather than quietly modelling a different bug.
    """
    verdicts = ((plan, plan._allowed()) for plan in self._plans.values())
    rewrites = self._installable()
    landed = []
    moved = False
    for plan, allowed in verdicts:
        was_moved, installed = plan._apply(allowed)
        moved = moved or was_moved
        landed.extend(installed)
    for proposal, replacement in rewrites:
        if not removal._replace_in_parent(proposal.statement, replacement):
            continue
        landed.append(replacement)
        moved = True
    removal._restore(landed)
    return moved


def _withdraw_rediscovering_the_owning_list(self: removal.Ps1RemovalPlans, statement) -> None:
    """
    The withdrawal as it read before the plan was remembered beside the statement: the owning list
    is looked up again, and a statement the tree has moved since is withdrawn from nothing at all.
    """
    if self._rewrites.pop(id(statement), None) is not None:
        return
    self._filed.pop(id(statement), None)
    owner = owning_list(statement)
    if owner is not None:
        self._plan_for(*owner).withdraw(statement)


def _apply_repairing_every_allowed_replacement(original: Callable) -> Callable:
    def mutated(self, allowed):
        moved, _ = original(self, allowed)
        landed = [statement for proposal in allowed for statement in proposal.replacement]
        return moved, landed if moved else []
    return mutated


def _dead_bindings_over_every_candidate(original: Callable) -> Callable:
    def mutated(self, candidates, dissolving):
        return original(self, candidates, {
            id(mutation) for mutations in candidates.values() for mutation in mutations})
    return mutated


def _resolved_reaching_the_host_always(self: effects.Ps1OutputFlow, path) -> OutputSink:
    """
    The destination question with the call graph taken out of it: every body writes to the host, so
    every write to the output stream is text on a console and deletable. This is what the model
    looked like before a function's callers were consulted, and what it collapses back to if the
    resolution ever stops distinguishing.
    """
    return OutputSink.HOST


def _leaves_anything_counting_its_own_residue(original: Callable) -> staticmethod:
    def mutated(plans, node, installed):
        return original(plans, node, set())
    return staticmethod(mutated)


def _try_body_survivors_hoisting_anything_pure(body: list, oracle) -> list | None:
    """
    The two predicates as they read before fault-freedom separated them: a body qualifies if every
    statement is side-effect free, and every side-effect-free statement is carried out. Purity is
    not an answer to whether a statement raises, so this hoists `[Int]'abc'` past the empty `catch`
    that was swallowing it.
    """
    survivors = []
    for stmt in body:
        if not isinstance(stmt, Ps1ExpressionStatement):
            return None
        if stmt.expression is None:
            continue
        if is_side_effect_free(stmt.expression, oracle):
            survivors.append(stmt)
            continue
        if deadcode._is_injected_noise_bareword(stmt.expression, oracle):
            continue
        return None
    return survivors



class TestPs1RemovalGuardsAreWitnessed(TestBase):
    """
    Each test removes one guard in memory and runs the tests that are supposed to notice. A guard
    whose removal leaves them green is unprotected however its docstring reads, and the next
    refactor takes it out for nothing; the failure names the guard that lost its last witness.
    """

    @staticmethod
    def _run(witnesses: list[str], mutations=()) -> unittest.TestResult:
        suite = unittest.defaultTestLoader.loadTestsFromNames(witnesses)
        result = unittest.TestResult()
        with ExitStack() as stack:
            for mutation in mutations:
                stack.enter_context(mutation)
            suite.run(result)
        return result

    @staticmethod
    def _reported(outcomes) -> list[str]:
        return sorted(str(test) for test, _ in outcomes)

    @staticmethod
    def _methods(outcomes) -> set[str]:
        """
        The test methods among `outcomes`, read through `unittest.TestCase.subTest`'s wrapper where
        one is present: a failed subtest reports as a `_SubTest` whose own method name is `runTest`,
        so reading it directly names every such witness identically and matches any probe at all.
        """
        return {getattr(test, 'test_case', test)._testMethodName for test, _ in outcomes}

    def _assertWitnessed(self, witnesses: list[str], *mutations, notices: str) -> None:
        """
        A witness has to be green before the mutation and red after it, red by *failing*, and the
        test named in `notices` has to be among the ones that failed.

        All three halves are load bearing. Without the first, a witness already red for a reason of
        its own satisfies every probe that names it, and the guard behind it stops being measured at
        the moment someone is editing near it. Without the second, an error counts: a mutation whose
        calling convention no longer matches the guard it replaces raises at every call site and is
        reported as a guard the tests noticed, which is the one answer this file must never give.

        Without the third, a probe names a *class* and any failure in it satisfies the probe — so a
        guard whose own test stopped covering it stays green as long as some unrelated neighbour in
        the same class notices something. `notices` says which test is the witness, and the class
        stays in the list because the surrounding tests are what establish the mutation did not
        simply break everything.
        """
        intact = self._run(witnesses)
        self.assertEqual(
            self._reported(intact.failures + intact.errors),
            [],
            F'{witnesses} do not pass with this guard in place')
        without = self._run(witnesses, mutations)
        self.assertEqual(
            self._reported(without.errors),
            [],
            F'the mutation broke {witnesses} rather than being noticed by them')
        self.assertTrue(
            without.failures,
            F'{intact.testsRun} tests in {witnesses} all pass without this guard')
        self.assertIn(
            notices,
            self._methods(without.failures),
            F'{notices} is not among the tests that noticed; '
            F'{sorted(self._methods(without.failures))} did')

    def test_the_handler_veto_is_witnessed(self):
        self._assertWitnessed(
            [_HANDLER],
            patch.object(removal, '_removes_a_handler', lambda statement: False),
            notices='test_a_trap_inside_a_protected_try_body_is_kept')

    def test_regranting_what_a_replacement_adopted_is_witnessed(self):
        self._assertWitnessed(
            [_DEAD_CODE, _PLAN, _TREE],
            patch.object(removal, '_restore', lambda landed: None),
            notices='test_a_live_store_holding_a_dead_one')

    def test_releasing_what_a_registration_adopted_is_witnessed(self):
        self._assertWitnessed(
            [_DEAD_CODE_TREE, _PLAN],
            patch.object(removal.Ps1RemovalPlan, 'propose', _propose_repairing_only_a_replacement),
            notices='test_a_replacement_the_caller_discarded_is_given_back_too')

    def test_repairing_only_what_an_edit_landed_is_witnessed(self):
        mutated = _apply_repairing_every_allowed_replacement(removal.Ps1RemovalPlan._apply)
        self._assertWitnessed(
            [_PLAN],
            patch.object(removal.Ps1RemovalPlan, '_apply', mutated),
            notices='test_the_edit_that_landed_decides_alone_what_is_repaired')

    def test_reaching_every_verdict_before_the_first_edit_is_witnessed(self):
        self._assertWitnessed(
            [_PLAN],
            patch.object(
                removal.Ps1RemovalPlans, 'commit', _commit_interleaving_verdicts_and_edits),
            notices='test_every_verdict_is_reached_before_the_first_edit_lands')

    def test_accepted_on_a_single_plan_is_witnessed(self):
        self._assertWitnessed(
            [_PLAN, _REFERENCE],
            patch.object(removal.Ps1RemovalPlan, 'accepted', property(_accepted_before_the_vetoes)),
            notices='test_an_assignment_the_fault_veto_keeps_keeps_the_variable_it_reads')

    def test_accepted_across_a_batch_of_plans_is_witnessed(self):
        self._assertWitnessed(
            [_PLAN, _REFERENCE],
            patch.object(
                removal.Ps1RemovalPlans,
                'accepted',
                property(_plans_accepted_before_the_vetoes)),
            notices='test_an_assignment_the_fault_veto_keeps_keeps_the_variable_it_reads')

    def test_a_withdrawal_reaching_the_plan_that_holds_it_is_witnessed(self):
        self._assertWitnessed(
            [_PLAN],
            patch.object(
                removal.Ps1RemovalPlans, 'withdraw', _withdraw_rediscovering_the_owning_list),
            notices='test_a_withdrawal_reaches_the_plan_that_holds_the_proposal')

    def test_liveness_asking_what_survives_is_witnessed(self):
        self._assertWitnessed(
            [_REFERENCE],
            patch.object(
                unused.Ps1UnusedVariableRemoval, '_dead_bindings',
                _dead_bindings_over_every_candidate(
                    unused.Ps1UnusedVariableRemoval._dead_bindings)),
            notices='test_a_kept_value_keeps_the_variable_it_reads')

    def test_the_target_slot_gate_on_a_dissolving_value_is_witnessed(self):
        self._assertWitnessed(
            [_REFERENCE],
            patch.object(unused, 'assignment_target_is_all_variables', lambda target: True),
            notices='test_a_target_slot_that_is_not_a_variable_keeps_the_whole_statement_alive')

    def test_the_erasure_guard_discounting_its_own_residue_is_witnessed(self):
        self._assertWitnessed(
            [_UNUSED, _EMULATOR, _INTEGRATION],
            patch.object(
                unused.Ps1UnusedVariableRemoval, '_leaves_anything',
                _leaves_anything_counting_its_own_residue(
                    unused.Ps1UnusedVariableRemoval._leaves_anything)),
            notices='test_a_script_reduced_to_this_passs_own_discards_is_left_alone')

    def test_carrying_only_fault_free_statements_out_of_a_try_is_witnessed(self):
        self._assertWitnessed(
            [_DEAD_CODE_EXTRA],
            patch.object(
                deadcode, '_try_body_survivors', _try_body_survivors_hoisting_anything_pure),
            notices='test_a_try_body_that_may_raise_keeps_its_construct')

    def test_discarding_a_hoisted_for_initializer_is_witnessed(self):
        # `for (5; $False; ) { }` puts nothing on the output; the bare statement `5` does. Hoisting
        # one plainly is the mirror image of deleting output.
        self._assertWitnessed(
            [_DEAD_CODE_EXTRA, _DEAD_CODE_TREE],
            patch.object(
                deadcode, '_hoisted_initializer',
                lambda expr: Ps1ExpressionStatement(expression=expr)),
            notices='test_a_hoisted_for_initializer_does_not_start_printing_its_value')

    def test_resolving_a_destination_rather_than_assuming_one_is_witnessed(self):
        # The analysis half of the conditional guard. Reading every body as writing to the host is
        # what a model with no call graph does, and it is the mutation the switch cannot cover for:
        # a value someone stores is not the switch's to give away under either setting.
        self._assertWitnessed(
            [_KEPT_EITHER_WAY],
            patch.object(effects.Ps1OutputFlow, 'resolved', _resolved_reaching_the_host_always),
            notices='test_a_value_a_caller_stores_is_kept')

    def test_reading_the_preserve_switch_at_all_is_witnessed(self):
        # A conditional guard goes vacuous in two more ways than an unconditional one, and neither
        # is the analysis being wrong. This is the switch never consulted.
        self._assertWitnessed(
            [_KEPT_WHEN_ASKED],
            patch.object(unused, 'bare_output_is_preserved', lambda options: False),
            notices='test_a_bare_literal_at_the_root_is_kept')

    def test_the_preserve_switch_being_off_by_default_is_witnessed(self):
        # And this is the switch stuck on, which reads as a deobfuscator that simply strips nothing.
        self._assertWitnessed(
            [_STRIPPED_BY_DEFAULT],
            patch.object(unused, 'bare_output_is_preserved', lambda options: True),
            notices='test_a_bare_literal_at_the_root_is_stripped')

    def test_the_fault_gate_on_a_deleted_write_is_witnessed(self):
        # What separates `42` from `[Int]'abc'`, both of which are output and only one of which
        # terminates the script where it stands.
        self._assertWitnessed(
            [_KEPT_EITHER_WAY],
            patch.object(unused, 'is_fault_free', lambda node: True),
            notices='test_a_statement_that_can_raise_is_kept')

    def test_the_redirection_gate_on_a_deleted_write_is_witnessed(self):
        # Patched in `effects`, where the outward walk reads it, because a redirection decides where
        # the value goes and not whether the statement around it is shaped like a discard.
        self._assertWitnessed(
            [_KEPT_EITHER_WAY],
            patch.object(effects, '_redirection_takes_output_away', lambda redirection: False),
            notices='test_a_value_a_redirection_moves_elsewhere_is_kept')

    def test_the_refusal_to_substitute_away_a_redirection_is_witnessed(self):
        # The rule itself, patched in the module that owns it. Both callers are named, because the
        # two shapes reach it by different routes: an expression in a slot asks through
        # `substitute`, and a pass returning a replacement from `visit_X` asks before it returns.
        self._assertWitnessed(
            [_IEX_REDIRECTIONS, _WILDCARD_REDIRECTIONS],
            patch.object(substitution, 'may_substitute', lambda *parts: True),
            notices='test_a_redirected_iex_statement_is_not_replaced_by_the_code_it_holds')

    def test_asking_the_rule_where_a_wildcard_rewrite_decides_is_witnessed(self):
        # `refinery.lib.scripts.Transformer.generic_visit` installs whatever a `visit_X` returns, so
        # a pass that does not ask is a pass whose replacement lands unexamined.
        self._assertWitnessed(
            [_WILDCARD_REDIRECTIONS],
            patch.object(wildcards, 'may_substitute', lambda *parts: True),
            notices='test_a_redirected_variable_read_is_not_rewritten_to_the_variable')

    def test_the_interpreter_refusing_a_body_that_opens_a_file_is_witnessed(self):
        # The emulator's own half: the call site carries no redirection, the body does, and folding
        # the call deletes the body along with the file it creates.
        self._assertWitnessed(
            [_EMULATOR_EXTRA],
            patch.object(emulator, 'opens_a_redirection_target', lambda node: False),
            notices='test_a_body_that_opens_a_file_is_not_folded_into_the_value_it_returns')

    def test_the_gate_on_an_element_a_selection_drops_is_witnessed(self):
        # Selecting out of a literal container evaluates the whole container, so what the selection
        # does not carry forward is work the folded script no longer does.
        self._assertWitnessed(
            [_SELECTION],
            patch.object(folding, 'may_be_dropped', lambda node, oracle: True),
            notices='test_an_array_element_that_runs_a_command_is_not_dropped')

    def test_the_fault_half_of_that_gate_is_witnessed(self):
        # And the half a purity argument alone would have granted: `[Int]'abc'` is side-effect free
        # and raises where it stands.
        self._assertWitnessed(
            [_SELECTION],
            patch.object(effects, 'is_fault_free', lambda node: True),
            notices='test_an_array_element_that_can_raise_is_not_dropped')
