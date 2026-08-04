"""
Where a PowerShell script block runs: at what point, in whose scope, and how many times.

Three places in this package already answer part of this privately, each drawing the line somewhere
else — `refinery.lib.scripts.analysis.cycles.CycleModel.repeats` walks lexically and says so in its
own docstring, `refinery.lib.scripts.ps1.analysis.model.Ps1SemanticModel` binds every bare write to
the scope it is written in, and `refinery.lib.scripts.ps1.analysis.effects` draws the
stored-versus-invoked line a third time. A block is a value, so the code around it is where it was
*written*, which is a different question from where it runs, and every pass that has needed the
second has had to guess it from the first.

**The answers here come from real PowerShell 5.1, not from the shape of the syntax.** `. { }` runs in
the caller's scope and `& { }` opens a child one, which is the pair the whole question turns on; a
`ForEach-Object` or `Where-Object` body also runs in the caller's scope, including when the block
reaches the cmdlet through a variable, and so does a `catch` or `finally` body; a `function` body, an
`Invoke-Command -ScriptBlock` without `-NoNewScope`, and a calculated property's block all open a
child scope. A `trap` body opens a child scope too, which
`refinery.lib.scripts.ps1.analysis.model.Ps1SemanticModel` does not model — its body is a `Block`
rather than a `Ps1ScriptBlock`, so no block here stands for it.

**`CHILD` is the answer that has to be earned.** Calling a body `CALLER` that is really `CHILD` adds a
kill nobody performs, which only ever loses an inlining; calling one `CHILD` that is really `CALLER`
drops a kill somebody does perform, which silently keeps a stale value. So `UNKNOWN` is projected
exactly as `CALLER` is by `writes_reaching_caller`, and only a position that *proves* a child scope
answers `CHILD`.

**`.` is the invoker's scope, not the writer's.** `function TakeDot([scriptblock] $b) { . $b }`
dot-sources into `TakeDot`'s scope, so `TakeDot { $x = 'b' }` leaves the original caller's `$x` alone.
That is why a literal `. { }` is decidable from where it sits and a block handed anywhere else is not.

`ForEach-Object -Begin { }` and `-End { }` run once where `-Process { }` runs per input object, and
every one of them is reported `REPEATED` here anyway. The parser does not bind a parameter name to
the value that follows it — `-Begin` arrives as a switch argument and its block as the next
positional one — so telling them apart means inferring the association from argument order, and
inferring it wrongly reports a body that iterates as running once, which is the direction that keeps
a stale value. The precision is not worth depending on that shape; a begin block loses an inlining
and nothing else.
"""
from __future__ import annotations

import enum

from dataclasses import dataclass
from typing import Iterator

from refinery.lib.scripts import Node
from refinery.lib.scripts.ps1.analysis.model import is_write_occurrence
from refinery.lib.scripts.ps1.analysis.naming import Ps1NameTarget, unreadable_name_target
from refinery.lib.scripts.ps1.ast import resolve_command_name
from refinery.lib.scripts.ps1.model import (
    Ps1CommandArgument,
    Ps1CommandInvocation,
    Ps1FunctionDefinition,
    Ps1ScopeModifier,
    Ps1Script,
    Ps1ScriptBlock,
    Ps1Variable,
)

#: The commands that run a scriptblock argument once per input object. A hit *withholds* the
#: single-visit reading a caller would otherwise take, so this is a deny-list and is read through
#: `refinery.lib.scripts.ps1.ast.resolve_command_name`, which follows `%` and `?` to the full names —
#: the opposite of what a grant table may do. It is deliberately not floored against the collected
#: command metadata for the same reason: a spelling the capture host never reported must still be
#: allowed to match.
_ITERATING_COMMANDS = frozenset({
    'foreach-object',
    'where-object',
})


class Ps1BlockReach(enum.Enum):
    """
    When the body runs relative to the point it is written at.

    `IMMEDIATE` — the statement that mentions the block runs it.
    `FUNCTION` — it is a named function's body, run by that function's call sites.
    `STORED` — its value is kept rather than run, so when it runs is not a question this layer holds.
    `UNKNOWN` — it is handed to something that may or may not run it.
    """
    IMMEDIATE = 'immediate'
    FUNCTION  = 'function'   # noqa
    STORED    = 'stored'     # noqa
    UNKNOWN   = 'unknown'    # noqa


class Ps1BlockScope(enum.Enum):
    """
    Whose variables the body's bare writes land in.

    `CALLER` — the scope of the code that runs it, so its writes are the caller's writes.
    `CHILD` — a fresh scope, so its writes are invisible outside and a name it assigns shadows.
    `UNKNOWN` — not decidable here, and treated as `CALLER` everywhere the difference is a kill.
    """
    CALLER  = 'caller'   # noqa
    CHILD   = 'child'    # noqa
    UNKNOWN = 'unknown'  # noqa


class Ps1BlockIteration(enum.Enum):
    """
    How often the site runs the body.

    `ONCE` — one invocation per visit to the site.
    `REPEATED` — the site runs it per input object, so a fact taken from one visit is not a fact.
    `UNKNOWN` — not decidable here.
    """
    ONCE     = 'once'     # noqa
    REPEATED = 'repeated'  # noqa
    UNKNOWN  = 'unknown'  # noqa


@dataclass(frozen=True)
class Ps1BlockFacts:
    """
    What is known about one `refinery.lib.scripts.ps1.model.Ps1ScriptBlock`. `site` is the element
    whose evaluation runs the body, and is `None` whenever that element is not in this script or is
    not decidable — a function body's callers, a stored block's eventual invocation.
    """
    reach: Ps1BlockReach
    scope: Ps1BlockScope
    iteration: Ps1BlockIteration
    site: Node | None


_UNPLACED = Ps1BlockFacts(
    reach=Ps1BlockReach.UNKNOWN,
    scope=Ps1BlockScope.UNKNOWN,
    iteration=Ps1BlockIteration.UNKNOWN,
    site=None,
)


def _invoked_directly(block: Ps1ScriptBlock) -> Ps1CommandInvocation | None:
    """
    The invocation that runs *block* by naming it, as `& { }` and `. { }` do, or `None`.
    """
    parent = block.parent
    if isinstance(parent, Ps1CommandInvocation) and parent.name is block:
        return parent
    return None


def _handed_to_command(block: Ps1ScriptBlock) -> Ps1CommandInvocation | None:
    """
    The invocation *block* is an argument of, named or positional, or `None`. The block reaching a
    command as an argument says nothing about whether the command runs it — `f { }`,
    `Invoke-Command -ScriptBlock { }` and `ForEach-Object { }` are one shape — so the caller still
    has to recognise the command.
    """
    parent = block.parent
    if isinstance(parent, Ps1CommandArgument):
        parent = parent.parent
    if isinstance(parent, Ps1CommandInvocation) and parent.name is not block:
        return parent
    return None


def classify_block(block: Ps1ScriptBlock) -> Ps1BlockFacts:
    """
    The facts readable from where *block* sits.
    """
    parent = block.parent
    if isinstance(parent, Ps1FunctionDefinition) and parent.body is block:
        return Ps1BlockFacts(
            reach=Ps1BlockReach.FUNCTION,
            scope=Ps1BlockScope.CHILD,
            iteration=Ps1BlockIteration.UNKNOWN,
            site=None,
        )
    invocation = _invoked_directly(block)
    if invocation is not None:
        scope = {
            '.': Ps1BlockScope.CALLER,
            '&': Ps1BlockScope.CHILD,
        }.get(invocation.invocation_operator, Ps1BlockScope.UNKNOWN)
        return Ps1BlockFacts(
            reach=Ps1BlockReach.IMMEDIATE,
            scope=scope,
            iteration=Ps1BlockIteration.ONCE,
            site=invocation,
        )
    command = _handed_to_command(block)
    if command is not None and resolve_command_name(command) in _ITERATING_COMMANDS:
        return Ps1BlockFacts(
            reach=Ps1BlockReach.IMMEDIATE,
            scope=Ps1BlockScope.CALLER,
            iteration=Ps1BlockIteration.REPEATED,
            site=command,
        )
    if command is not None:
        return _UNPLACED
    return Ps1BlockFacts(
        reach=Ps1BlockReach.STORED,
        scope=Ps1BlockScope.UNKNOWN,
        iteration=Ps1BlockIteration.UNKNOWN,
        site=None,
    )


class Ps1BlockModel:
    """
    Where each script block of one root runs. Facts are read off the tree on first request and kept,
    since the tree is fixed for as long as this model lives.
    """

    def __init__(self, root: Ps1Script):
        self.root = root
        self._facts: dict[int, Ps1BlockFacts] = {}
        self._caller_writes: dict[int, tuple[Ps1Variable, ...]] = {}
        self._caller_unattributable: dict[int, bool] = {}

    def facts(self, block: Ps1ScriptBlock) -> Ps1BlockFacts:
        """
        What is known about where *block* runs.
        """
        found = self._facts.get(id(block))
        if found is None:
            found = self._facts[id(block)] = classify_block(block)
        return found

    def may_write_caller_scope(self, block: Ps1ScriptBlock) -> bool:
        """
        Whether a bare write inside *block* may land in the scope of the code that runs it. True for
        everything but a proven child scope — see the module docstring for why that asymmetry is the
        safe one.
        """
        return self.facts(block).scope is not Ps1BlockScope.CHILD

    def writes_reaching_caller(self, block: Ps1ScriptBlock) -> tuple[Ps1Variable, ...]:
        """
        The bare write occurrences inside *block* that land in the scope of whatever runs it — the
        ones written directly in its body, and those of any nested block that reaches its own caller
        in turn. Empty for a proven child scope, since nothing a child scope writes outlives it.

        That the recursion stops at a child scope is what makes `& { . { $x = 'b' } }` write nothing
        outside: the inner dot writes the `&` block's scope, and that scope ends with it. Qualified
        writes are left out because a `$script:` or `$global:` write names its scope outright and
        reaches the same binding whichever body it sits in, so it is not a fact about where the block
        runs.
        """
        found = self._caller_writes.get(id(block))
        if found is None:
            if not self.may_write_caller_scope(block):
                found = ()
            else:
                found = tuple(self._collect_writes(block))
            self._caller_writes[id(block)] = found
        return found

    def unattributable_writes_reaching_caller(self, block: Ps1ScriptBlock) -> bool:
        """
        Whether *block* runs a write whose name this cannot read — `Set-Variable $n 'v'` — into the
        scope of whatever runs it. The name is unknown, so the write may have landed on any binding
        of that scope, and a caller can place *when* it happened without knowing *what* it hit.

        Only the writes a command places in its own scope are reported. One that names a scope
        outright reaches the same binding whichever body it sits in, so it is not a fact about where
        this block runs, and reporting it here would have it stop at a child scope that does not
        stop it — the same reason `writes_reaching_caller` leaves a `$script:` write out.
        """
        found = self._caller_unattributable.get(id(block))
        if found is None:
            found = self._caller_unattributable[id(block)] = (
                self.may_write_caller_scope(block)
                and self._runs_unattributable_write(block)
            )
        return found

    def _runs_unattributable_write(self, block: Ps1ScriptBlock) -> bool:
        stack: list[Node] = list(block.children())
        while stack:
            node = stack.pop()
            if isinstance(node, Ps1ScriptBlock):
                if self.unattributable_writes_reaching_caller(node):
                    return True
                continue
            if (
                isinstance(node, Ps1CommandInvocation)
                and unreadable_name_target(node) is Ps1NameTarget.LOCAL
            ):
                return True
            stack.extend(node.children())
        return False

    def _collect_writes(self, block: Ps1ScriptBlock) -> Iterator[Ps1Variable]:
        stack: list[Node] = list(block.children())
        while stack:
            node = stack.pop()
            if isinstance(node, Ps1ScriptBlock):
                yield from self.writes_reaching_caller(node)
                continue
            if (
                isinstance(node, Ps1Variable)
                and node.scope is Ps1ScopeModifier.NONE
                and is_write_occurrence(node)
            ):
                yield node
            stack.extend(node.children())

    def body_site(self, owner: Node) -> tuple[Node, bool] | None:
        """
        The `refinery.lib.scripts.analysis.cycles.BodySite` answer for a body: the element that runs
        *owner*, and whether it runs it more than once.

        `None` for the script root, which nothing in the script runs, and for any block whose site is
        not decidable — a stored block, a function body, a block handed to a command that may or may
        not invoke it. The cycle walk reads that as *fall back to where the block is written*, which
        is what it did everywhere before this model existed, so answering nothing changes nothing.
        """
        if not isinstance(owner, Ps1ScriptBlock):
            return None
        facts = self.facts(owner)
        if facts.site is None:
            return None
        return facts.site, facts.iteration is Ps1BlockIteration.REPEATED


def build_block_model(root: Ps1Script) -> Ps1BlockModel:
    """
    Build the `Ps1BlockModel` for a script.
    """
    return Ps1BlockModel(root)
