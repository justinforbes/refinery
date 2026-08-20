"""
The flow-sensitive reading of the closed-world model: whether the type world is closed at *one*
particular read, rather than anywhere the script runs. `refinery.lib.scripts.ps1.analysis.world`
answers the whole-run question over a bare tree walk and stays a leaf value object; this layer adds
the position, which is a control-flow question, the way
`refinery.lib.scripts.ps1.analysis.commands` layers command identity on the same model's shadow set.

A member read the metadata proves inert still runs code once a leak has re-pointed that member
through the Extended Type System or remapped its type accelerator. A read *no such leak has run
before* observes what it always would have, so it is inert like any other — and the whole-run model
cannot tell the two apart, keeping every read in a file that leaks once anywhere. This model floods
forward through the control-flow graph from every opener and grants a read only where the flood does
not reach it: no code that could have mutated the world runs on any path to that read.

The gate is deny-side — a wrong grant deletes code that had an effect, a refusal only costs recall —
so every uncertainty fails toward *open*:

- A `class` or `enum` definition opens the world at no position: the engine compiles it before the
  first statement runs, so it stands before every read. Its presence anywhere returns the whole-run
  verdict for the whole file.
- A root `process` block re-runs once per pipeline input, which the per-body graph models as
  straight-line with no back edge, so a leak late in it precedes a read early in it on the next
  item. Its presence returns the whole-run verdict.
- An opener the graph cannot place — a parameter default, a node in no body's graph — poisons
  nothing a flood could reach, so it too returns the whole-run verdict.

An opener written inside a scriptblock or function body is *lifted* to the root-graph statement that
runs that body — the value where the block is written, the `function` statement that defines it —
and floods from there, because a stored block cannot execute before the statement that creates it
and a function cannot be called before its definition runs. A read inside such a body, by contrast,
is refused wholesale: a body may be entered again by a later call, so a read written after a leak in
source can run before it at runtime, which the intraprocedural graph does not order.

The gate rests on one assumption the graph cannot enforce: that the code a leak runs does not
re-execute this script's own earlier statements. A script that dot-sources or invokes its own file
(`. $PSCommandPath`) runs its statements a second time, after the first run's leaks, and a read
granted on the first pass then runs after them — a control-flow edge no per-body graph carries.
The portable spellings of such a re-run all pass through the names PowerShell reveals a script's
own path or text under — `$PSCommandPath`, `$MyInvocation`, `$PSScriptRoot`, the call stack, the
process arguments — so a script that spells any of them anywhere is refused whole, the same
fallback the placeless openers take. What remains is a script that hits its own file without
naming it: through a leak's opaque payload, or through a hard-coded path that happens to be its
own location, which an analysis that never learns where the script lies cannot recognize. Both
stay out of contract — refusing every leaking script for what an unseen payload might do is the
whole-run verdict this model exists to replace — and are stated rather than left silent.
"""
from __future__ import annotations

from refinery.lib.scripts import Node, tree_version
from refinery.lib.scripts.analysis.cfg import (
    CfgNode,
    ControlFlowGraph,
    ControlFlowModel,
    reachable_from_any,
)
from refinery.lib.scripts.ps1.analysis.world import Ps1TypeWorld, world_openers
from refinery.lib.scripts.ps1.model import (
    Ps1ClassDefinition,
    Ps1EnumDefinition,
    Ps1HereString,
    Ps1InvokeMember,
    Ps1MemberAccess,
    Ps1Script,
    Ps1StringLiteral,
    Ps1Variable,
)


class Ps1WorldReach:
    """
    The world as the purity gate reads it at a position: the whole-run facts of a
    `refinery.lib.scripts.ps1.analysis.world.Ps1TypeWorld`, plus `closed_at`, which asks whether the
    world is closed at one read rather than anywhere. Held in a
    `refinery.lib.scripts.ps1.analysis.cache.Ps1ModelCache` slot and threaded through
    `refinery.lib.scripts.ps1.analysis.effects` in place of the leaf world.

    Only `closed_at` is position-sensitive. `may_trust_command_name` is the whole-run verdict
    unchanged — command identity opens along a different axis than the type system, so a read's
    flow relaxation is not sound to apply to a name (see
    `Ps1TypeWorld.may_trust_command_name`) — and the remaining facts delegate to the leaf model, so
    a caller threads one object rather than two that can drift.

    Built without flow context — `Ps1WorldReach(world)` — it answers `closed_at` with the whole-run
    verdict at every position, which is what a caller that did not build the graphs, or that holds a
    world nothing was measured over, gets. "We looked and it is open here" and "we did not look" are
    the same refusal, deliberately, since both keep the read.

    A `closed_at` answer is bound to the tree the graphs were built over, so the model records that
    tree's version and refuses — falls back to the whole-run verdict — once the tree has changed
    under it. A transform reading this through the fresh
    `refinery.lib.scripts.ps1.analysis.cache.Ps1ModelCache` slot never sees a stale one, since the
    cache rebuilds it on the same version bump; a transform that captured it and then edited does,
    and the refusal keeps the reads it would otherwise mis-place. This is why holding it across an
    edit is sound without auditing each pass for whether it moves a read past a leak.
    """

    def __init__(
        self,
        world: Ps1TypeWorld,
        *,
        root: Ps1Script | None = None,
        control_flow: ControlFlowModel | None = None,
        root_graph: ControlFlowGraph | None = None,
        poisoned: frozenset[int] = frozenset(),
        refuse: bool = False,
        build_version: int = 0,
    ):
        self._world = world
        self._root = root
        self._control_flow = control_flow
        self._root_graph = root_graph
        self._poisoned = poisoned
        self._refuse = refuse
        self._build_version = build_version

    @property
    def closed_for_the_whole_run(self) -> bool:
        return self._world.closed_for_the_whole_run

    @property
    def closed_but_for_alias_bindings(self) -> bool:
        return self._world.closed_but_for_alias_bindings

    @property
    def shadowed_names(self) -> frozenset[str]:
        return self._world.shadowed_names

    def command_shadowed(self, name: str) -> bool:
        return self._world.command_shadowed(name)

    def may_trust_command_name(self, name: str) -> bool:
        return self._world.may_trust_command_name(name)

    def closed_at(self, node) -> bool:
        """
        Whether the type world is closed at `node`: no opener can have run on any path that reaches
        the statement evaluating it. A closed-for-the-whole-run world answers `True` everywhere,
        since it has no opener to reach anything; otherwise the answer is `False` unless `node`
        locates into the root graph outside the poisoned region.

        Anything the graph cannot place — a `node` in no body, or one in a scriptblock or function
        body, whose second invocation this intraprocedural model cannot order against a leak — is
        refused, which is the fail-closed direction: a wrong grant deletes an effect, a refusal
        keeps a read.
        """
        if self._world.closed_for_the_whole_run:
            return True
        if self._refuse or self._control_flow is None or self._root_graph is None:
            return False
        if self._root is not None and tree_version(self._root) != self._build_version:
            return False
        located = self._control_flow.locate(node)
        if located is None or located[0] is not self._root_graph:
            return False
        return id(located[1]) not in self._poisoned


def build_world_reach(
    root: Ps1Script,
    world: Ps1TypeWorld,
    control_flow: ControlFlowModel,
) -> Ps1WorldReach:
    """
    The flow-sensitive world for *root*, over its whole-run `world` and its control-flow graphs. A
    world already closed for the whole run carries no position — nothing opens it — so it is wrapped
    unmeasured; otherwise every opener is lifted into the root graph and the poisoned region is the
    forward flood from all of them.

    The verdict falls back to the whole-run answer at every position — a `Ps1WorldReach` built with
    `refuse` — whenever a position-less opener is present (a `class`/`enum` definition, a root
    `process` block that re-runs), an opener cannot be placed in the root graph, or the script
    names its own path (`_names_own_path`), through which a leak could re-run the statements before
    it. Each is the fail-closed direction the module docstring states: the flood cannot bound where
    such an opener ran, so no read is granted over it.
    """
    if world.closed_for_the_whole_run:
        return Ps1WorldReach(world)
    root_graph = control_flow.graph_of(root)
    if root_graph is None:
        return Ps1WorldReach(world, refuse=True)
    refuse = root.process_block is not None or _names_own_path(root)
    sources: list[CfgNode] = []
    for opener in world_openers(root):
        if isinstance(opener, (Ps1ClassDefinition, Ps1EnumDefinition)):
            refuse = True
            continue
        landing = _lift_to_root(control_flow, opener, root_graph)
        if landing is None:
            refuse = True
            continue
        sources.append(landing)
    # An open world yields at least one opener, since `world_openers` and `build_closed_world` read
    # the same `_opens_world`; empty sources here would mean that coupling broke, so fail closed
    # rather than flood from nothing and grant every position.
    if refuse or not sources:
        return Ps1WorldReach(world, refuse=True)
    poisoned = reachable_from_any(sources)
    return Ps1WorldReach(
        world,
        root=root,
        control_flow=control_flow,
        root_graph=root_graph,
        poisoned=poisoned,
        build_version=tree_version(root),
    )


def _lift_to_root(
    control_flow: ControlFlowModel,
    opener: Node,
    root_graph: ControlFlowGraph,
) -> CfgNode | None:
    """
    The root-graph control-flow node from which *opener* poisons forward, climbing out of every
    nested body it sits in, or `None` when it cannot be placed at all.

    An opener in the root graph is its own statement's node. One inside a scriptblock or function
    body locates into that body's own graph; the block or definition is a value written at a point
    in the body around it, so the climb re-locates that owner and repeats until it lands in the root
    graph. The landing is sound as a flood source: a stored block cannot run before the statement
    that creates it, and a function cannot be called before its definition executes, so poisoning
    from that statement forward reaches every point the opener's effect could. A `None` climb — a
    parameter default the graphs place nowhere — is the caller's signal to fall back to the
    whole-run verdict.
    """
    node: Node = opener
    while True:
        located = control_flow.locate(node)
        if located is None:
            return None
        graph, cfg_node = located
        if graph is root_graph:
            return cfg_node
        node = graph.owner


#: The names under which PowerShell reveals a running script's own path or text: the automatic
#: variables, the call-stack cmdlet, and the process-arguments member. Lowercase, because the
#: language matches none of them case-sensitively.
_SELF_PATH_NAMES = frozenset({
    'pscommandpath',
    'myinvocation',
    'psscriptroot',
    'pscallstack',
    'getcommandlineargs',
})


def _names_own_path(root: Ps1Script) -> bool:
    """
    Whether *root* spells, anywhere, one of the names through which a running script reaches its
    own path or text. The flood poisons forward only, so a statement that re-runs the script's file
    puts every leak before every read; the portable spellings of such a re-run all pass through one
    of these names, and which statement would perform it — under an `if` guard, inside a payload —
    is not decidable from here, so the whole script is refused instead.

    A variable or member name must match one exactly; a string value need only contain one, so that
    a bareword argument (`Get-Variable MyInvocation`) and a quoted payload (`'. $PSCommandPath'`)
    trip the guard as surely as the bare variable read does.
    """
    for node in root.walk():
        if isinstance(node, Ps1Variable):
            if node.name.lower() in _SELF_PATH_NAMES:
                return True
            continue
        if isinstance(node, (Ps1StringLiteral, Ps1HereString)):
            value = node.value.lower()
            if any(name in value for name in _SELF_PATH_NAMES):
                return True
            continue
        if isinstance(node, (Ps1MemberAccess, Ps1InvokeMember)):
            member = node.member
            if isinstance(member, str) and member.lower() in _SELF_PATH_NAMES:
                return True
    return False
