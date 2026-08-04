"""
The writes no reading of one script attributes to a name.

Every layer above reasons about a name through its occurrences, so a write with no occurrence
anywhere is a write that layer folds straight across. There are two ways a script performs one, and
they arrive from opposite directions:

- it addresses the name as a *string* it computes, which
  `refinery.lib.scripts.ps1.analysis.naming` recognises — `Set-Variable $n 'v'`;
- it runs *code this analysis never sees* — `Invoke-Expression $c`, `. 'stage2.ps1'`, `. $sb` —
  which may write anything at all.

To a consumer they are one fact: at this point, some binding of this scope may have changed and
nothing says which. `writes_nobody_can_attribute` is that one question, and it is the only thing a
consumer should have to ask.

**This says nothing about when.** The point is the whole of what is known, and it is enough: a read
that reaches its write without passing the point observes what it always would have. Turning the
fact into a property of the enclosing scope instead was tried and it deadlocks — marking a scope in
doubt for an `Invoke-Expression` stops the payload variable folding, so the call never becomes
literal, so it never expands, so the mark never lifts, and a loader that used to unpack came back as
the obfuscator wrote it.

**The recognition is deny-side.** A spelling this does not resolve is not a call declared harmless;
it is one whose effect is unknown, so a miss here performs a corruption where a miss in a grant table
such as `refinery.lib.scripts.ps1.analysis.effects` merely withholds a rewrite. That is also why no
`refinery.lib.scripts.ps1.analysis.types.TypeOracle` is consulted: a script that shadows `iex` runs
something else, and something else is opaque too.

**Only the scope the code stands in.** Measured on 5.1 (`temp/ps1/census_measurements.md`): `&`
opens a child scope, so `& 'stage2.ps1'` and `& $sb` leave the caller's `$x` alone, and so does
`$ExecutionContext.InvokeCommand.InvokeScript('$x = 1')` — which runs in a scope of its own however
much it looks like `Invoke-Expression`. All three are deliberately absent. The narrow hole that
leaves is a *qualified* write from inside one of them — `& { iex '$script:x = 1' }` does reach the
caller — recorded as a hole rather than paid for by treating every call operator as a caller-scope
write.
"""
from __future__ import annotations

from refinery.lib.scripts import Node
from refinery.lib.scripts.ps1.analysis.naming import Ps1NameTarget, unreadable_name_target
from refinery.lib.scripts.ps1.ast import resolve_command_name
from refinery.lib.scripts.ps1.model import Ps1CommandInvocation, Ps1ScriptBlock

#: Commands that run arbitrary code supplied as data, in the scope of whatever calls them. Resolved
#: through `refinery.lib.scripts.ps1.ast.resolve_command_name`, so `iex` arrives here canonical.
#:
#: `Start-Job` and `Start-ThreadJob` are absent although
#: `refinery.lib.scripts.ps1.analysis.world` counts them: they run their block in another runspace,
#: which has no access to these variables at all. `Invoke-Command` is absent for the same kind of
#: reason — without `-NoNewScope` it opens a child scope, and
#: `refinery.lib.scripts.ps1.analysis.blocks` already places a literal block handed to it there.
_UNREADABLE_CODE_COMMANDS = frozenset({
    'invoke-expression',
})


def writes_nobody_can_attribute(node: Node) -> bool:
    """
    Whether *node* performs a write no reading of the source attributes to a name, landing in the
    scope *node* stands in.

    The two sources are one fact to a consumer, so they are answered together: a command addressing
    a name it computes, and a call running code this analysis never sees.
    """
    if isinstance(node, Ps1CommandInvocation):
        if unreadable_name_target(node) is Ps1NameTarget.LOCAL:
            return True
    return runs_unreadable_code(node)


def runs_unreadable_code(node: Node) -> bool:
    """
    Whether *node* runs code this analysis never sees, in the scope *node* stands in.

    A dot-invocation of anything but an inline block is one, and the target is not what decides it:
    `. 'stage2.ps1'` runs a file, `. $sb` runs whatever the expression yields, and `. f` runs a
    function's body *in the caller's scope* rather than the child scope an ordinary call opens — all
    three measured — so even a function this tree defines writes somewhere no call graph here
    accounts for. Only `. { … }` is exempt, because its body is written where it runs and every
    layer already reads it.
    """
    if not isinstance(node, Ps1CommandInvocation):
        return False
    if resolve_command_name(node) in _UNREADABLE_CODE_COMMANDS:
        return True
    return (
        node.invocation_operator == '.'
        and node.name is not None
        and not isinstance(node.name, Ps1ScriptBlock)
    )
