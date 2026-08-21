from __future__ import annotations

import inspect
import unittest

from test.lib.scripts.ps1.deobfuscation import TestPs1

#: A discarded conversion that Windows PowerShell 5.1 answers with a terminating error: `[Int]'abc'`
#: has no conversion, and the assignment to `$Null` means nothing observes a value it never makes.
#: It emits nothing and mutates nothing, which is exactly what makes a cleanup pass want to drop it.
_RAISE = "$Null = [Int]'abc'"

#: A second spelling of the same fault, so that no claim below rests on the cast in particular.
_RAISE_DIV = '$Null = 1/0'

#: Statements that Windows PowerShell 5.1 runs to completion. Each is a discarded value that emits
#: nothing and mutates nothing, exactly like `_RAISE`, and differs from it only in raising no error
#: for a handler to observe. A pass that decides a removal by asking whether a statement can raise
#: therefore has to delete every one of these from the shapes in which it has to keep `_RAISE`.
#: They are spelled twelve different ways so that no such claim rests on one expression.
_QUIET_CAST = "$Null = [Int]'42'"
_QUIET_DIVISION = '$Null = 1/1'
_QUIET_PRODUCT = '$Null = 6 * 7'
_QUIET_LENGTH = "$Null = 'abcdef'.Length"
_QUIET_REMAINDER = '$Null = 10 % 3'
_QUIET_CHAR = '$Null = [Char]65'
_QUIET_CONCATENATION = "$Null = 'ab' + 'cd'"
_QUIET_STRING = '$Null = [String]12'
_QUIET_CONJUNCTION = '$Null = 3 -Band 1'
_QUIET_NEGATION = '$Null = -Not $True'
_QUIET_BOOL = "$Null = [Bool]'x'"
_QUIET_COUNT = '$Null = @(1, 2, 3).Count'

#: An acting statement that is never a removal candidate, so its survival says only that the pass
#: did not empty the script wholesale.
_ANCHOR = "Write-Host 'ANCHOR_SURVIVES'"

#: A handler body that writes to the host. Writing to the host is what makes a handler live, and a
#: live handler and an empty one are opposite licences rather than degrees of the same one.
_HANDLER = "Write-Host 'HANDLER_RAN'"

#: A `finally` body that writes to the host. It runs on the faulting and non-faulting path alike.
_CLEANUP = "Write-Host 'CLEANUP_RAN'"

#: A condition and an enumerable that the analysis cannot decide. One it could fold away would take
#: its body with it on its own merits, and a body that lost its only statement that way would say
#: nothing about the statement that used to be in it.
_OPAQUE = '$args'

#: A call that is never a removal candidate, so a `try` written around it survives into the output
#: as a handler the script demonstrably still contains.
_REQUEST = 'Invoke-WebRequest $u'


class _Ps1FaultObservability(TestPs1):

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

    def _assertRemoved(self, source: str, statement: str) -> None:
        """
        The expected output is `source` with `statement` gone and nothing else touched, so the pair
        of arguments spells out one removal rather than a whole rewritten script. Naming a statement
        that `source` does not contain would leave the expectation saying nothing, so it is refused.
        """
        self.assertIn(statement, source)
        self._assertDeobfuscatesTo(source, source.replace(statement, ''))


class TestPs1ARaisingStatementDirectlyInAGuardedTryBlockIsKept(_Ps1FaultObservability):
    """
    A `catch` clause with a body runs when the `try` block it belongs to raises a terminating error,
    so deleting the only statement that can raise means the handler no longer runs. The statement
    therefore survives, however little else there is to observe about it.
    """

    def test_a_raising_cast_directly_in_the_try_block_is_kept(self):
        self._assertKept(F"""
            try {{
              {_RAISE}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """)

    def test_a_raising_division_directly_in_the_try_block_is_kept(self):
        self._assertKept(F"""
            try {{
              {_RAISE_DIV}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """)


class TestPs1AStatementNestedInAGuardedTryBlock(_Ps1FaultObservability):
    """
    A `catch` clause observes an error raised anywhere inside its `try` block, not only one raised
    by a statement written directly in it. A branch body, a loop body, a `switch` case body and a
    scriptblock invoked in place with `&` are all inside the block, so a raising statement in any of
    them is one the handler depends on and none of them may be deleted.

    The deobfuscator recognizes a handler only for a statement whose immediate holder is the `try`
    block itself. One nesting level is enough to hide the handler from it, so it deletes each of
    these and leaves behind a `catch` body that can no longer run.

    Every shape is written twice. The second of the pair puts a statement that runs to completion
    where the raising one stood: no handler can observe it, deleting it is the job, and the nesting
    must not save it either. The pair tells a pass that has found the handler apart from one that
    has merely stopped deleting.
    """

    @unittest.expectedFailure
    def test_a_raising_cast_in_a_nested_if_body_is_kept(self):
        self._assertKept(F"""
            try {{
              if ({_OPAQUE}) {{ {_RAISE} }}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """)

    def test_a_quiet_cast_in_a_nested_if_body_is_removed(self):
        self._assertRemoved(F"""
            try {{
              if ({_OPAQUE}) {{ {_QUIET_CAST} }}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """, _QUIET_CAST)

    @unittest.expectedFailure
    def test_a_raising_cast_in_a_nested_foreach_body_is_kept(self):
        self._assertKept(F"""
            try {{
              foreach ($i in {_OPAQUE}) {{ {_RAISE} }}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """)

    def test_a_quiet_product_in_a_nested_foreach_body_is_removed(self):
        self._assertRemoved(F"""
            try {{
              foreach ($i in {_OPAQUE}) {{ {_QUIET_PRODUCT} }}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """, _QUIET_PRODUCT)

    @unittest.expectedFailure
    def test_a_raising_cast_in_a_nested_while_body_is_kept(self):
        self._assertKept(F"""
            try {{
              while ({_OPAQUE}) {{ {_RAISE} }}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """)

    def test_a_quiet_length_in_a_nested_while_body_is_removed(self):
        self._assertRemoved(F"""
            try {{
              while ({_OPAQUE}) {{ {_QUIET_LENGTH} }}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """, _QUIET_LENGTH)

    @unittest.expectedFailure
    def test_a_raising_cast_in_a_nested_switch_case_body_is_kept(self):
        self._assertKept(F"""
            try {{
              switch ({_OPAQUE}) {{ 1 {{ {_RAISE} }} }}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """)

    def test_a_quiet_remainder_in_a_nested_switch_case_body_is_removed(self):
        self._assertRemoved(F"""
            try {{
              switch ({_OPAQUE}) {{ 1 {{ {_QUIET_REMAINDER} }} }}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """, _QUIET_REMAINDER)

    @unittest.expectedFailure
    def test_a_raising_cast_in_a_scriptblock_invoked_in_place_is_kept(self):
        self._assertKept(F"""
            try {{
              & {{ {_RAISE} }}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """)

    def test_a_quiet_char_in_a_scriptblock_invoked_in_place_is_removed(self):
        self._assertRemoved(F"""
            try {{
              & {{ {_QUIET_CHAR} }}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """, _QUIET_CHAR)

    @unittest.expectedFailure
    def test_a_raising_division_in_a_nested_if_body_is_kept(self):
        self._assertKept(F"""
            try {{
              if ({_OPAQUE}) {{ {_RAISE_DIV} }}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """)

    def test_a_quiet_division_in_a_nested_if_body_is_removed(self):
        self._assertRemoved(F"""
            try {{
              if ({_OPAQUE}) {{ {_QUIET_DIVISION} }}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """, _QUIET_DIVISION)


class TestPs1AStatementInAFunctionAGuardedTryBlockCalls(_Ps1FaultObservability):
    """
    A terminating error raised in a function reaches the `catch` clause guarding the call, so a
    function body is inside the `try` block for this purpose even though it is written outside it.

    The deobfuscator empties the function body, because the call site is what the `try` block holds
    and the raising statement is somewhere else entirely.

    A statement that runs to completion in that same function body reaches no handler at all, so
    emptying the body is the right answer for it and the two differ only in the raise.
    """

    @unittest.expectedFailure
    def test_a_raising_cast_in_a_function_the_try_block_calls_is_kept(self):
        self._assertKept(F"""
            function Invoke-Thing {{
              {_RAISE}
            }}
            try {{
              Invoke-Thing
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """)

    def test_a_quiet_concatenation_in_a_function_the_try_block_calls_is_removed(self):
        self._assertRemoved(F"""
            function Invoke-Thing {{
              {_QUIET_CONCATENATION}
            }}
            try {{
              Invoke-Thing
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """, _QUIET_CONCATENATION)


class TestPs1AnEmptyCatchSwallowsSoTheRaisingStatementIsRemovable(_Ps1FaultObservability):
    """
    An empty `catch` swallows the error and lets execution resume after the construct, so a script
    whose `try` block holds only the raising statement runs on to the same next statement whether
    that statement is there or not. Removing it is then not observable and remains allowed.

    An empty inner `catch` swallows before any outer `catch` is offered the error, so it licenses
    the removal even when the script does contain a live handler — one wrapped around it, or one
    beside it. These are the shapes a refusal keyed to a handler appearing anywhere in the script
    would break.
    """

    def test_a_raising_cast_guarded_only_by_an_empty_catch_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            try {{ {_RAISE} }} catch {{ }}
            {_ANCHOR}
        """, _ANCHOR)

    def test_a_raising_cast_an_empty_inner_catch_shields_from_a_live_outer_catch_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            try {{
              try {{ {_RAISE} }} catch {{ }}
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

    def test_a_raising_cast_under_an_empty_catch_beside_a_live_catch_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            try {{ {_RAISE} }} catch {{ }}
            try {{ {_REQUEST} }} catch {{ {_HANDLER} }}
            {_ANCHOR}
        """, F"""
            try {{ {_REQUEST} }} catch {{ {_HANDLER} }}
            {_ANCHOR}
        """)


class TestPs1AFinallyAloneDoesNotGuardTheRaisingStatement(_Ps1FaultObservability):
    """
    A `finally` body runs on the faulting path and on the non-faulting path, so no removal can
    change whether it runs. A `try` with no `catch` clause therefore leaves the raising statement
    exactly as removable as it would be with no construct written around it at all.
    """

    def test_a_raising_cast_in_a_try_whose_only_clause_is_a_finally_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            try {{ {_RAISE} }} finally {{ {_CLEANUP} }}
            {_ANCHOR}
        """, F"""
            {_CLEANUP}
            {_ANCHOR}
        """)


class TestPs1AnInnerFinallyDoesNotShieldAnOuterCatch(_Ps1FaultObservability):
    """
    Because a `finally` does not swallow, the error raised under one goes on to the nearest
    enclosing `catch`, which here has a body. The raising statement is what makes that handler run
    and must survive.

    The deobfuscator reads the inner `try` as unguarded, deletes the raising statement, then
    dissolves the construct and hoists the `finally` body into the outer block, leaving a `catch`
    clause nothing reaches.

    A statement that runs to completion under that same inner `finally` reaches no handler, so the
    whole sequence is the right answer for it and the two differ only in the raise.
    """

    @unittest.expectedFailure
    def test_a_raising_cast_under_an_inner_finally_inside_a_live_outer_catch_is_kept(self):
        self._assertKept(F"""
            try {{
              try {{ {_RAISE} }} finally {{ {_CLEANUP} }}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """)

    def test_a_quiet_string_under_an_inner_finally_inside_a_live_outer_catch_is_removed(self):
        """
        An inner `try` whose block is empty runs its `finally` and nothing else, so the expected
        output is the construct gone with the cleanup standing in the outer block in its place.
        """
        self._assertDeobfuscatesTo(F"""
            try {{
              try {{ {_QUIET_STRING} }} finally {{ {_CLEANUP} }}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """, F"""
            try {{
              {_CLEANUP}
              {_ANCHOR}
            }} catch {{
              {_HANDLER}
            }}
        """)


class TestPs1ALiveTrapGuardsItsWholeScope(_Ps1FaultObservability):
    """
    A `trap` handles a terminating error raised anywhere in the scope it is written in, whether it
    stands above the raising statement or below it, and whether the error is raised at the top of
    that scope or inside a construct nested in it. A live `trap` in scope makes the raising
    statement the reason the handler runs, so the statement survives.

    The deobfuscator never consults a `trap` when it decides a removal. It deletes the raising
    statement in each of these and keeps the `trap`, which is then a handler nothing can trigger.

    What a `trap` guards is the raise, so a statement that runs to completion gives it nothing to
    handle wherever in the scope it stands. Every shape is written a second time with such a
    statement, which the pass must go on deleting.
    """

    @unittest.expectedFailure
    def test_a_raising_cast_below_a_live_trap_in_the_same_scope_is_kept(self):
        self._assertKept(F"""
            trap {{ {_HANDLER} }}
            {_RAISE}
            {_ANCHOR}
        """)

    def test_a_quiet_conjunction_below_a_live_trap_in_the_same_scope_is_removed(self):
        self._assertRemoved(F"""
            trap {{ {_HANDLER} }}
            {_QUIET_CONJUNCTION}
            {_ANCHOR}
        """, _QUIET_CONJUNCTION)

    @unittest.expectedFailure
    def test_a_raising_cast_above_a_live_trap_in_the_same_scope_is_kept(self):
        self._assertKept(F"""
            {_RAISE}
            trap {{ {_HANDLER} }}
            {_ANCHOR}
        """)

    def test_a_quiet_negation_above_a_live_trap_in_the_same_scope_is_removed(self):
        self._assertRemoved(F"""
            {_QUIET_NEGATION}
            trap {{ {_HANDLER} }}
            {_ANCHOR}
        """, _QUIET_NEGATION)

    @unittest.expectedFailure
    def test_a_raising_cast_nested_in_a_scope_a_live_trap_guards_is_kept(self):
        self._assertKept(F"""
            trap {{ {_HANDLER} }}
            if ({_OPAQUE}) {{
              {_RAISE}
              {_ANCHOR}
            }}
        """)

    def test_a_quiet_bool_nested_in_a_scope_a_live_trap_guards_is_removed(self):
        self._assertRemoved(F"""
            trap {{ {_HANDLER} }}
            if ({_OPAQUE}) {{
              {_QUIET_BOOL}
              {_ANCHOR}
            }}
        """, _QUIET_BOOL)

    @unittest.expectedFailure
    def test_a_raising_cast_in_a_function_scope_a_live_trap_guards_is_kept(self):
        self._assertKept(F"""
            function Invoke-Thing {{
              trap {{ {_HANDLER} }}
              {_RAISE}
            }}
            Invoke-Thing
            {_ANCHOR}
        """)

    def test_a_quiet_count_in_a_function_scope_a_live_trap_guards_is_removed(self):
        self._assertRemoved(F"""
            function Invoke-Thing {{
              trap {{ {_HANDLER} }}
              {_QUIET_COUNT}
            }}
            Invoke-Thing
            {_ANCHOR}
        """, _QUIET_COUNT)


class TestPs1ATrapThatSwallowsOrIsOutOfScopeLeavesTheStatementRemovable(_Ps1FaultObservability):
    """
    A `trap { continue }` suppresses the error and resumes at the next statement of its scope, so
    the script reaches the same statement either way and the removal is not observable — the
    handler goes with it because nothing is left for it to handle. A live `trap` written in another
    scope never sees an error raised at script scope, so it does not guard the statement either.

    These are the shapes a refusal keyed to a `trap` appearing anywhere in the script would break.
    """

    def test_a_raising_cast_under_a_trap_that_continues_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            trap {{ continue }}
            {_RAISE}
            {_ANCHOR}
        """, _ANCHOR)

    def test_a_raising_cast_outside_the_scope_of_a_live_trap_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            function Invoke-Thing {{
              trap {{ {_HANDLER} }}
              {_REQUEST}
            }}
            Invoke-Thing
            {_RAISE}
            {_ANCHOR}
        """, F"""
            function Invoke-Thing {{
              trap {{ {_HANDLER} }}
              {_REQUEST}
            }}
            Invoke-Thing
            {_ANCHOR}
        """)


class TestPs1ARaisingStatementNoHandlerGuardsIsRemoved(_Ps1FaultObservability):
    """
    With no `catch` and no `trap` anywhere, there is no handler whose running the removal could
    change, and the raising statement is junk the pass exists to delete. It is deleted wherever it
    stands: at script scope, in a branch body, or in the body of a function that is called.
    """

    def test_a_raising_cast_at_script_scope_with_no_handler_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            {_RAISE}
            {_ANCHOR}
        """, _ANCHOR)

    def test_a_raising_cast_in_a_branch_body_with_no_handler_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            if ({_OPAQUE}) {{
              {_RAISE}
              Write-Host 'BRANCH_RUNS'
            }}
            {_ANCHOR}
        """, F"""
            if ({_OPAQUE}) {{
              Write-Host 'BRANCH_RUNS'
            }}
            {_ANCHOR}
        """)

    def test_a_raising_cast_in_a_called_function_with_no_handler_is_removed(self):
        self._assertDeobfuscatesTo(F"""
            function Invoke-Thing {{ {_RAISE} }}
            Invoke-Thing
            {_ANCHOR}
        """, _ANCHOR)
