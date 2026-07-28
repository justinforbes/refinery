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
    test_removal,
    test_unused,
)

from refinery.lib.scripts import owning_list
from refinery.lib.scripts.ps1.deobfuscation import removal, unused


def _witness(witness: type) -> str:
    """
    The dotted name `unittest.TestLoader` resolves. Spelled from the class rather than by hand,
    because `unittest.loader.loadTestsFromNames` answers a name it cannot resolve with a test that
    *errors*, and a probe cannot tell that apart from a guard its witnesses noticed. Written this
    way a renamed witness is an `AttributeError` here, before any probe runs.
    """
    return F'{witness.__module__}.{witness.__qualname__}'


_DEAD_CODE = _witness(test_deadcode.TestPs1DeadCodeElimination)
_DEAD_CODE_TREE = _witness(test_deadcode.TestPs1DeadCodeLeavesTheTreeConsistent)
_EMULATOR = _witness(test_emulator.TestPs1FunctionEvaluator)
_HANDLER = _witness(test_removal.TestPs1DeadCodeEliminationDoesNotUnhookAHandler)
_INTEGRATION = _witness(test_deobfuscation.TestPs1Integration)
_PLAN = _witness(test_removal.TestPs1RemovalPlan)
_REFERENCE = _witness(test_unused.TestPs1RemovalLeavesNoDanglingReference)
_TREE = _witness(test_unused.TestPs1RemovalLeavesTheTreeConsistent)


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


def _leaves_anything_counting_its_own_residue(original: Callable) -> staticmethod:
    def mutated(plans, node, installed):
        return original(plans, node, set())
    return staticmethod(mutated)



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

    def _assertWitnessed(self, witnesses: list[str], *mutations) -> None:
        """
        A witness has to be green before the mutation and red after it, and red by *failing*.

        Both halves are load bearing. Without the first, a witness already red for a reason of its
        own satisfies every probe that names it, and the guard behind it stops being measured at the
        moment someone is editing near it. Without the second, an error counts: a mutation whose
        calling convention no longer matches the guard it replaces raises at every call site and is
        reported as a guard the tests noticed, which is the one answer this file must never give.
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

    def test_the_handler_veto_is_witnessed(self):
        self._assertWitnessed(
            [_HANDLER],
            patch.object(removal, '_removes_a_handler', lambda statement: False))

    def test_regranting_what_a_replacement_adopted_is_witnessed(self):
        self._assertWitnessed(
            [_DEAD_CODE, _PLAN, _TREE],
            patch.object(removal, '_restore', lambda landed: None))

    def test_releasing_what_a_registration_adopted_is_witnessed(self):
        self._assertWitnessed(
            [_DEAD_CODE_TREE, _PLAN],
            patch.object(removal.Ps1RemovalPlan, 'propose', _propose_repairing_only_a_replacement))

    def test_repairing_only_what_an_edit_landed_is_witnessed(self):
        mutated = _apply_repairing_every_allowed_replacement(removal.Ps1RemovalPlan._apply)
        self._assertWitnessed(
            [_PLAN],
            patch.object(removal.Ps1RemovalPlan, '_apply', mutated))

    def test_reaching_every_verdict_before_the_first_edit_is_witnessed(self):
        self._assertWitnessed(
            [_PLAN],
            patch.object(
                removal.Ps1RemovalPlans, 'commit', _commit_interleaving_verdicts_and_edits))

    def test_accepted_on_a_single_plan_is_witnessed(self):
        self._assertWitnessed(
            [_PLAN, _REFERENCE],
            patch.object(removal.Ps1RemovalPlan, 'accepted', property(_accepted_before_the_vetoes)))

    def test_accepted_across_a_batch_of_plans_is_witnessed(self):
        self._assertWitnessed(
            [_PLAN, _REFERENCE],
            patch.object(
                removal.Ps1RemovalPlans,
                'accepted',
                property(_plans_accepted_before_the_vetoes)))

    def test_a_withdrawal_reaching_the_plan_that_holds_it_is_witnessed(self):
        self._assertWitnessed(
            [_PLAN],
            patch.object(
                removal.Ps1RemovalPlans, 'withdraw', _withdraw_rediscovering_the_owning_list))

    def test_liveness_asking_what_survives_is_witnessed(self):
        self._assertWitnessed(
            [_REFERENCE],
            patch.object(
                unused.Ps1UnusedVariableRemoval, '_dead_bindings',
                _dead_bindings_over_every_candidate(
                    unused.Ps1UnusedVariableRemoval._dead_bindings)))

    def test_the_target_slot_gate_on_a_dissolving_value_is_witnessed(self):
        self._assertWitnessed(
            [_REFERENCE],
            patch.object(unused, 'assignment_target_is_all_variables', lambda target: True))

    def test_the_erasure_guard_discounting_its_own_residue_is_witnessed(self):
        self._assertWitnessed(
            [_EMULATOR, _INTEGRATION],
            patch.object(
                unused.Ps1UnusedVariableRemoval, '_leaves_anything',
                _leaves_anything_counting_its_own_residue(
                    unused.Ps1UnusedVariableRemoval._leaves_anything)))
