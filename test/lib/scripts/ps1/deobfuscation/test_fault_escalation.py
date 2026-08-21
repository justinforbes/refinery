from __future__ import annotations

import inspect
import unittest

from test.lib.scripts.ps1.deobfuscation import TestPs1

#: A discarded conversion that Windows PowerShell 5.1 answers with an implicit terminating error.
#: Left to itself it is reported and stepped over, so it neither stops the script nor leaves a value
#: behind. Every claim below is about the shapes in which deleting it does change what runs.
_RAISE = "$Null = [Int]'abc'"

#: The type Windows PowerShell 5.1 gives that error, so a handler filtered on it matches the raise.
_MATCHING = '[System.Management.Automation.RuntimeException]'

#: A type the raise is not, so a handler filtered on it never matches the raise.
_DIFFERENT = '[System.IO.IOException]'

#: A statement written after the raise. Whether it runs is the whole question wherever the raise
#: ends the script or abandons the remainder of a block.
_FOLLOWER = "Write-Host 'FOLLOWER_RAN'"

#: A handler body that writes to the host, so that a handler which runs is one the output names.
_HANDLER = "Write-Host 'HANDLER_RAN'"

#: A second handler body, for the shapes where an error raised inside one handler is taken by
#: another, so that the output names which of the two ran.
_OUTER_HANDLER = "Write-Host 'OUTER_HANDLER_RAN'"

#: An acting statement that is never a removal candidate, so its survival says only that the pass
#: did not empty the script wholesale.
_ANCHOR = "Write-Host 'ANCHOR_SURVIVES'"

#: A condition the analysis cannot decide, so that a branch is neither taken nor folded away.
_OPAQUE = '$args'


class _Ps1FaultEscalation(TestPs1):

    def _assertDeobfuscatesTo(self, source: str, expected: str) -> None:
        """
        Both arguments are written as ordinary indented PowerShell, and `expected` is rendered
        through the synthesizer before the comparison, so that brace layout cannot be mistaken for a
        statement having been removed.
        """
        self.assertEqual(
            self._deobfuscate(inspect.cleandoc(source)),
            self._apply(inspect.cleandoc(expected)),
        )

    def _assertKept(self, source: str) -> None:
        self._assertDeobfuscatesTo(source, source)


class TestPs1AStopPreferenceMakesTheRaiseEndTheScript(_Ps1FaultEscalation):
    """
    An implicit terminating error is reported and stepped over, but under
    `$ErrorActionPreference = 'Stop'` it ends the script instead. Nothing written after the raise
    runs, neither in the raise's own block nor in any block enclosing it. Deleting the raise starts
    running all of it, so the raise and the assignment that arms it both survive.

    The deobfuscator reads no preference variable when it decides a removal. It deletes the raise
    and keeps the assignment, turning a script that stopped there into one that runs on.
    """

    @unittest.expectedFailure
    def test_a_raising_cast_under_a_stop_preference_is_kept(self):
        self._assertKept(F"""
            $ErrorActionPreference = 'Stop'
            {_RAISE}
            {_FOLLOWER}
        """)

    @unittest.expectedFailure
    def test_a_raising_cast_in_a_branch_under_a_stop_preference_is_kept(self):
        self._assertKept(F"""
            $ErrorActionPreference = 'Stop'
            if ({_OPAQUE}) {{
              {_RAISE}
            }}
            {_FOLLOWER}
        """)


class TestPs1APreferenceThatResumesLeavesTheRaiseRemovable(_Ps1FaultEscalation):
    """
    Every preference other than `Stop` reports the error at most and resumes at the next statement,
    so the script reaches the same statement whether the raise is there or not. These are the shapes
    a refusal keyed to the preference variable being assigned at all would break.
    """

    def test_a_raising_cast_under_a_continue_preference_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            $ErrorActionPreference = 'Continue'
            {_RAISE}
            {_ANCHOR}
        """, F"""
            $ErrorActionPreference = 'Continue'
            {_ANCHOR}
        """)

    def test_a_raising_cast_under_a_silently_continue_preference_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            $ErrorActionPreference = 'SilentlyContinue'
            {_RAISE}
            {_ANCHOR}
        """, F"""
            $ErrorActionPreference = 'SilentlyContinue'
            {_ANCHOR}
        """)


class TestPs1ATrapWhoseTypeFilterMissesTheErrorEndsTheScript(_Ps1FaultEscalation):
    """
    A `trap` whose type filter does not match the error is not merely inert. With no other `trap` in
    scope to take the error, it ends the script: the body of that `trap` never runs and neither does
    anything written after the raise, whether the body is empty or writes to the host. Deleting the
    raise starts running the rest of the script.

    The deobfuscator reads a `trap` it cannot match as no `trap` at all and deletes the raise.
    """

    def test_a_raising_cast_under_an_empty_trap_whose_filter_misses_is_kept(self):
        self._assertKept(F"""
            trap {_DIFFERENT} {{ }}
            {_RAISE}
            {_FOLLOWER}
        """)

    def test_a_raising_cast_under_a_live_trap_whose_filter_misses_is_kept(self):
        self._assertKept(F"""
            trap {_DIFFERENT} {{ {_HANDLER} }}
            {_RAISE}
            {_FOLLOWER}
        """)


class TestPs1ATrapBodyThatReachesBreakEndsTheScript(_Ps1FaultEscalation):
    """
    A `trap` body that reaches `break` rethrows the error once it has run, which ends the script.
    Nothing written after the raise runs, so deleting the raise starts running it. The one-word
    variant of the same `trap` that reaches `continue` instead is licensed to lose the raise.

    The deobfuscator reads no `trap` body when it decides a removal, so it deletes the raise and the
    `trap` with it and lets the script run to its end.
    """

    def test_a_raising_cast_under_a_trap_that_breaks_is_kept(self):
        self._assertKept(F"""
            trap {{ break }}
            {_RAISE}
            {_FOLLOWER}
        """)

    def test_a_raising_cast_under_a_trap_that_writes_then_breaks_is_kept(self):
        self._assertKept(F"""
            trap {{
              {_HANDLER}
              break
            }}
            {_RAISE}
            {_FOLLOWER}
        """)


class TestPs1ATrapThatTakesTheErrorAndSwallowsLeavesTheRaiseRemovable(_Ps1FaultEscalation):
    """
    A `trap` that matches the error and reaches `continue` suppresses it and resumes at the next
    statement, so the script runs the same code with the raise as without it. A `trap` whose filter
    misses ends the script only when no other `trap` in scope takes the error, so an untyped one
    written beside it swallows and the script does not end after all.
    """

    def test_a_raising_cast_under_a_matching_trap_that_continues_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            trap {_MATCHING} {{ continue }}
            {_RAISE}
            {_FOLLOWER}
        """, _FOLLOWER)

    def test_a_raising_cast_under_an_untyped_trap_that_continues_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            trap {{ continue }}
            {_RAISE}
            {_FOLLOWER}
        """, _FOLLOWER)

    def test_a_raising_cast_a_continuing_trap_takes_from_a_missing_filter_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            trap {_DIFFERENT} {{ }}
            trap {{ continue }}
            {_RAISE}
            {_FOLLOWER}
        """, _FOLLOWER)


class TestPs1ACatchWhoseTypeFilterMissesLeavesTheRaiseRemovable(_Ps1FaultEscalation):
    """
    A `catch` whose type filter does not match the error is the sharp opposite of a `trap` whose
    filter does not match. The error leaves the construct unhandled, the script resumes at the
    statement written after it, and the `catch` body never runs, so the raise may go. The clause
    here carries the same filter and the same empty body as the `trap` that ends the script, and
    only the keyword differs.
    """

    def test_a_raising_cast_under_an_empty_catch_whose_filter_misses_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            try {{ {_RAISE} }} catch {_DIFFERENT} {{ }}
            {_FOLLOWER}
        """, _FOLLOWER)


class TestPs1ATypedCatchThatMissesDoesNotShieldAnEnclosingCatch(_Ps1FaultEscalation):
    """
    A type filter decides whether a `catch` handles the error, so a `catch` that does not match
    passes it on to the enclosing `catch`, whose body then runs. The raise is what makes that
    handler run and must survive.

    The deobfuscator reads any `catch` written around the raise as taking it, deletes the raise, and
    leaves the enclosing handler with nothing that can reach it.
    """

    def test_a_raising_cast_a_missing_filter_passes_to_a_live_outer_catch_is_kept(self):
        self._assertKept(F"""
            try {{
              try {{ {_RAISE} }} catch {_DIFFERENT} {{ }}
            }} catch {{
              {_HANDLER}
            }}
        """)


class TestPs1ATypedCatchThatMatchesShieldsAnEnclosingCatch(_Ps1FaultEscalation):
    """
    The same nesting with a filter that does match the error is handled by the inner `catch`, so the
    enclosing handler never runs and the script carries on inside the outer `try` block. The removal
    is then unobservable and stays allowed.
    """

    def test_a_raising_cast_an_empty_matching_filter_swallows_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            try {{
              try {{ {_RAISE} }} catch {_MATCHING} {{ }}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """, F"""
            try {{
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """)


class TestPs1AnEmptyCatchDoesNotCoverWhatFollowsTheRaiseInItsBlock(_Ps1FaultEscalation):
    """
    An empty `catch` swallows the error, but the raise still abandons the rest of its `try` block,
    so a statement written after it there never runs. The raise is the only reason that statement is
    dead, and it cannot be dropped while the statement stands.

    The deobfuscator deletes the raise and leaves the statement behind, which starts running it.
    """

    @unittest.expectedFailure
    def test_a_raising_cast_before_another_statement_of_the_same_try_block_is_kept(self):
        self._assertKept(F"""
            try {{
              {_RAISE}
              {_FOLLOWER}
            }} catch {{ }}
            {_ANCHOR}
        """)


class TestPs1AnEmptyCatchCoversARaiseThatIsLastInItsBlock(_Ps1FaultEscalation):
    """
    With the same statement written before the raise rather than after it, nothing in the `try`
    block is abandoned: the statement runs, the empty `catch` swallows, and the script carries on.
    The raise may go.
    """

    def test_a_raising_cast_after_every_other_statement_of_its_try_block_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            try {{
              {_FOLLOWER}
              {_RAISE}
            }} catch {{ }}
            {_ANCHOR}
        """, F"""
            try {{
              {_FOLLOWER}
            }} catch {{ }}
            {_ANCHOR}
        """)


class TestPs1ATrapGuardsTheBlockItIsWrittenIn(_Ps1FaultEscalation):
    """
    A `trap` guards the statement block it is written in, and where blocks nest, the innermost one
    that declares a `trap` is the one that takes the error. A raise written in the same block as a
    live `trap` is what makes that `trap` run, and a raise beside the inner of two live traps is
    what makes the inner one run and the outer one not.

    The deobfuscator never consults a `trap` when it decides a removal, so it deletes the raise in
    both and keeps handlers nothing can trigger.
    """

    def test_a_raising_cast_beside_a_live_trap_in_the_same_nested_block_is_kept(self):
        self._assertKept(F"""
            if ({_OPAQUE}) {{
              trap {{ {_HANDLER} }}
              {_RAISE}
            }}
            {_ANCHOR}
        """)

    def test_a_raising_cast_beside_the_innermost_of_two_live_traps_is_kept(self):
        self._assertKept(F"""
            trap {{ {_FOLLOWER} }}
            if ({_OPAQUE}) {{
              trap {{ {_HANDLER} }}
              {_RAISE}
            }}
            {_ANCHOR}
        """)


class TestPs1ATrapTheRaisingBlockDoesNotReachLeavesItRemovable(_Ps1FaultEscalation):
    """
    A `trap` declared in a block the raise is not in never sees the error, and it does not end the
    script over it either, so a live `trap` written in a sibling block leaves the raise exactly as
    removable as no handler at all would. Where the innermost `trap` in reach swallows, the live one
    enclosing it is never offered the error, so that raise is removable too.

    These are the shapes a refusal keyed to a `trap` appearing anywhere in the script would break.

    Both leave the `trap` itself standing, which is what a handler with nothing left to handle costs
    and not what it means. `Write-Host` is a command, and no reading here shows a command unable to
    raise, so deleting the swallowing `trap` would expose whatever remains in its block to the live
    `trap` around it — a different handler, on the path 5.1 takes when the host does fail.
    """

    def test_a_raising_cast_outside_the_block_that_declares_the_trap_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            if ({_OPAQUE}) {{
              trap {{ {_HANDLER} }}
            }}
            {_RAISE}
            {_ANCHOR}
        """, F"""
            if ({_OPAQUE}) {{
              trap {{ {_HANDLER} }}
            }}
            {_ANCHOR}
        """)

    def test_a_raising_cast_whose_innermost_trap_continues_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            trap {{ {_HANDLER} }}
            if ({_OPAQUE}) {{
              trap {{ continue }}
              {_RAISE}
              {_FOLLOWER}
            }}
            {_ANCHOR}
        """, F"""
            trap {{ {_HANDLER} }}
            if ({_OPAQUE}) {{
              trap {{ continue }}
              {_FOLLOWER}
            }}
            {_ANCHOR}
        """)


class TestPs1ATrapInATryBlockRunsInsteadOfTheCatch(_Ps1FaultEscalation):
    """
    A `trap` declared inside a `try` block takes the error before the `catch` clause is offered it,
    so the `trap` body runs and the `catch` body does not. The raise is what makes the `trap` run,
    and it survives.
    """

    def test_a_raising_cast_beside_a_live_trap_inside_a_guarded_try_block_is_kept(self):
        self._assertKept(F"""
            try {{
              trap {{ {_HANDLER} }}
              {_RAISE}
            }} catch {{
              {_FOLLOWER}
            }}
            {_ANCHOR}
        """)


class TestPs1ATrapInATryBlockThatSwallowsLeavesTheRaiseRemovable(_Ps1FaultEscalation):
    """
    Because the `trap` inside the `try` block takes the error first, a `trap` body that suppresses
    it leaves the `catch` clause unreached, and one that emits nothing leaves nothing else to
    observe. Either way the script runs the same code with the raise as without it.

    The deobfuscator refuses both removals because it sees a `catch` clause around the raise, and it
    never asks whether a `trap` in the block took the error before that clause could.
    """

    @unittest.expectedFailure
    def test_a_raising_cast_a_continuing_trap_takes_before_a_live_catch_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            try {{
              trap {{ continue }}
              {_RAISE}
            }} catch {{
              {_HANDLER}
            }}
            {_ANCHOR}
        """, _ANCHOR)

    @unittest.expectedFailure
    def test_a_raising_cast_an_empty_trap_takes_before_a_live_catch_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            try {{
              trap {{ }}
              {_RAISE}
            }} catch {{
              {_HANDLER}
            }}
            {_ANCHOR}
        """, _ANCHOR)


class TestPs1ATrapBodyThatOnlyProducesAValueIsALiveHandler(_Ps1FaultEscalation):
    """
    A `trap` body that neither assigns nor calls anything is still a live handler: it runs when its
    block raises, and the value it produces is written to the output stream. The raise is what makes
    that value appear, so it survives.

    The deobfuscator deletes the raise and the `trap` with it, silencing an output the script made.
    """

    @unittest.expectedFailure
    def test_a_raising_cast_under_a_trap_whose_body_is_a_bare_value_is_kept(self):
        self._assertKept(F"""
            trap {{ 5 }}
            {_RAISE}
            {_ANCHOR}
        """)


class TestPs1ATrapBodyThatProducesNothingLeavesTheRaiseRemovable(_Ps1FaultEscalation):
    """
    A `trap` with an empty body writes nothing and lets execution resume, so it changes no code the
    script runs and the raise under it may go. The bare value that makes the other `trap` live is
    written only when its block raises, so with nothing raising in that block the `trap` writes
    nothing and may go itself.
    """

    def test_a_raising_cast_under_a_trap_whose_body_is_empty_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            trap {{ }}
            {_RAISE}
            {_ANCHOR}
        """, _ANCHOR)

    def test_a_trap_whose_body_is_a_bare_value_with_nothing_raising_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            trap {{ 5 }}
            {_ANCHOR}
        """, _ANCHOR)


class TestPs1AReadOfErrorObservesARaiseNoHandlerTook(_Ps1FaultEscalation):
    """
    Windows PowerShell 5.1 records every terminating error in `$Error` whether or not a handler ran,
    so a script with no `catch` and no `trap` anywhere can still branch on the raise having happened
    and can still read the record it left. The raise is what puts that record there and survives.

    The deobfuscator decides a removal from handlers alone. It finds none, deletes the raise, and
    leaves a script whose `$Error` is empty where the original's was not.
    """

    @unittest.expectedFailure
    def test_a_raising_cast_a_later_count_of_error_observes_is_kept(self):
        self._assertKept(F"""
            {_RAISE}
            if ($Error.Count) {{
              {_HANDLER}
            }}
            {_ANCHOR}
        """)

    @unittest.expectedFailure
    def test_a_raising_cast_a_later_read_of_its_error_record_observes_is_kept(self):
        self._assertKept(F"""
            {_RAISE}
            Write-Host $Error[0].Exception.Message
        """)


class TestPs1ARaiseNoReadOfErrorFollowsIsRemovable(_Ps1FaultEscalation):
    """
    With the same read of `$Error` written before the raise rather than after it, the read answers
    the same on both scripts, because nothing has raised yet when it runs. Nothing observes the
    raise and it may go.
    """

    def test_a_raising_cast_after_the_only_read_of_error_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            if ($Error.Count) {{
              {_HANDLER}
            }}
            {_RAISE}
            {_ANCHOR}
        """, F"""
            if ($Error.Count) {{
              {_HANDLER}
            }}
            {_ANCHOR}
        """)


class TestPs1ARaiseInATrapBodyEndsThatBody(_Ps1FaultEscalation):
    """
    A terminating error raised inside a `trap` body ends that body and escapes it. At script scope
    the escaped error ends the script, so neither the rest of the `trap` body nor the statement the
    script would have resumed at runs; inside a function it ends only the function and the caller
    carries on. Where something does guard the block the `trap` belongs to, the escaped error goes
    to that guard, and an enclosing `catch` clause or a second, live `trap` runs its body over it.
    In each of these the raise inside the `trap` body decides what runs next, so it survives.

    The deobfuscator reads a `trap` body as statements no error can leave. It deletes the raise
    there and runs the remainder of the body that the original abandoned.
    """

    @unittest.expectedFailure
    def test_a_raising_cast_before_another_statement_of_the_same_trap_body_is_kept(self):
        self._assertKept(F"""
            trap {{
              {_HANDLER}
              {_RAISE}
              {_FOLLOWER}
            }}
            {_RAISE}
            {_ANCHOR}
        """)

    @unittest.expectedFailure
    def test_a_raising_cast_in_the_trap_body_of_a_function_is_kept(self):
        self._assertKept(F"""
            function Invoke-Thing {{
              trap {{
                {_HANDLER}
                {_RAISE}
                {_FOLLOWER}
              }}
              {_RAISE}
            }}
            Invoke-Thing
            {_ANCHOR}
        """)

    def test_a_raising_cast_in_a_trap_body_an_enclosing_catch_takes_is_kept(self):
        self._assertKept(F"""
            try {{
              trap {{
                {_HANDLER}
                {_RAISE}
                {_FOLLOWER}
              }}
              {_RAISE}
            }} catch {{
              {_OUTER_HANDLER}
            }}
            {_ANCHOR}
        """)

    def test_a_raising_cast_in_a_trap_body_an_enclosing_trap_takes_is_kept(self):
        self._assertKept(F"""
            trap {{ {_OUTER_HANDLER} }}
            if ({_OPAQUE}) {{
              trap {{
                {_HANDLER}
                {_RAISE}
                {_FOLLOWER}
              }}
              {_RAISE}
            }}
            {_ANCHOR}
        """)


class TestPs1ATrapBodyNothingTriggersLeavesTheRaiseInItRemovable(_Ps1FaultEscalation):
    """
    A `trap` body runs only when its block raises, so with nothing raising there the statements of
    the body never run and a raise among them ends nothing. It may go while the `trap` stands, and
    this is the shape a refusal keyed to a raise being written inside a `trap` body would break.
    """

    def test_a_raising_cast_in_the_body_of_a_trap_no_raise_triggers_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            trap {{
              {_HANDLER}
              {_RAISE}
              {_FOLLOWER}
            }}
            {_ANCHOR}
        """, F"""
            trap {{
              {_HANDLER}
              {_FOLLOWER}
            }}
            {_ANCHOR}
        """)


class TestPs1ATrapTakesTheErrorsOfTheNamedBlockItIsWrittenIn(_Ps1FaultEscalation):
    """
    An advanced function splits its body across `begin`, `process`, `end` and `dynamicparam`
    blocks, and a `trap` guards the named block it is written in. A raise in the same named block
    as a live `trap` is what makes that `trap` run, and execution then resumes at the next statement
    of that block, so both the raise and what follows it survive.

    The deobfuscator never consults a `trap` when it decides a removal, so it deletes the raise from
    either block and keeps a handler nothing can trigger.
    """

    def test_a_raising_cast_beside_a_live_trap_in_the_same_process_block_is_kept(self):
        self._assertKept(F"""
            function Invoke-Thing {{
              process {{
                trap {{ {_HANDLER} }}
                {_RAISE}
                {_FOLLOWER}
              }}
            }}
            Invoke-Thing
        """)

    def test_a_raising_cast_beside_a_live_trap_in_the_same_begin_block_is_kept(self):
        self._assertKept(F"""
            function Invoke-Thing {{
              begin {{
                trap {{ {_HANDLER} }}
                {_RAISE}
                {_FOLLOWER}
              }}
              process {{
                {_ANCHOR}
              }}
            }}
            Invoke-Thing
        """)


class TestPs1ATrapInAnotherNamedBlockLeavesTheRaiseRemovable(_Ps1FaultEscalation):
    """
    The same `trap`, written in a named block other than the one that raises, is never offered the
    error: the raise in the `process` block is reported, that block carries on to its next
    statement, and the `trap` written in `begin` does not run. The script runs the same code with
    the raise as without it and the raise may go; writing that same `trap` in the block that raises
    is the whole of what makes it stay.
    """

    def test_a_raising_cast_in_the_process_block_a_begin_block_trap_never_sees_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            function Invoke-Thing {{
              begin {{
                trap {{ {_HANDLER} }}
              }}
              process {{
                {_RAISE}
                {_FOLLOWER}
              }}
            }}
            Invoke-Thing
        """, F"""
            function Invoke-Thing {{
              begin {{
                trap {{ {_HANDLER} }}
              }}
              process {{
                {_FOLLOWER}
              }}
            }}
            Invoke-Thing
        """)
