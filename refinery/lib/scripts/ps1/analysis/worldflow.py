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

from typing import Callable

from refinery.lib.scripts import Node, tree_version
from refinery.lib.scripts.analysis.cfg import (
    CfgNode,
    ControlFlowGraph,
    ControlFlowModel,
    reachable_from_any,
)
from refinery.lib.scripts.ps1.analysis.world import Ps1TypeWorld, Ps1WorldMeasurement
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
    `refinery.lib.scripts.ps1.analysis.effects` in place of the leaf world. It answers only the
    three questions the passes ask of it — `closed_at`, `may_trust_command_name`, and
    `closed_for_the_whole_run`; the remaining leaf facts are read straight off
    `Ps1ModelCache.closed_world`, which no staleness can touch because the cache rebuilds it, so
    mirroring them here would be answers with no reader.

    Only `closed_at` is position-sensitive. `may_trust_command_name` is the whole-run verdict —
    command identity opens along a different axis than the type system, so a read's flow relaxation
    is not sound to apply to a name (see `Ps1TypeWorld.may_trust_command_name`).

    Built without flow context — `Ps1WorldReach(world)` — it answers `closed_at` with the whole-run
    verdict at every position, which is what a caller that did not build the graphs, or that holds a
    world nothing was measured over, gets. "We looked and it is open here" and "we did not look" are
    the same refusal, deliberately, since both keep the read.

    Every answer is bound to the tree the model was built over: `build_world_reach` stamps that
    tree's version onto the wrapper, and once the tree changes under it every answer returns its
    fail-closed pole — `closed_at` and `may_trust_command_name` and `closed_for_the_whole_run` all
    read `False`. A transform reading this through the fresh
    `refinery.lib.scripts.ps1.analysis.cache.Ps1ModelCache` slot never sees a stale one, since the
    cache rebuilds it on the same version bump. A pass that instead captures the wrapper and holds
    it across its own edits reads the fail-closed pole from the first edit on: it loses recall until
    the next pass rebuilds against the changed tree, never soundness. A rootless synthetic wrapper
    carries no stamp and so never goes stale — it has no tree to change under it.
    """

    def __init__(
        self,
        world: Ps1TypeWorld,
        *,
        root: Ps1Script | None = None,
        control_flow: ControlFlowModel | None = None,
        poisoned: frozenset[int] = frozenset(),
        refuse: bool = False,
        build_version: int = 0,
    ):
        self._world = world
        self._root = root
        self._control_flow = control_flow
        self._poisoned = poisoned
        self._refuse = refuse
        self._build_version = build_version

    @property
    def _stale(self) -> bool:
        """
        Whether the tree has changed since this reach model was built, so its every answer is about
        a script that no longer stands. A wrapper carrying a `root` — one `build_world_reach` made —
        notices; a rootless one built around a synthetic world cannot, and answers from the leaf
        forever, which is what a caller that never named a tree wants.
        """
        return self._root is not None and tree_version(self._root) != self._build_version

    @property
    def closed_for_the_whole_run(self) -> bool:
        return not self._stale and self._world.closed_for_the_whole_run

    def may_trust_command_name(self, name: str) -> bool:
        return not self._stale and self._world.may_trust_command_name(name)

    def closed_at(self, node) -> bool:
        """
        Whether the type world is closed at `node`: no opener can have run on any path that reaches
        the statement evaluating it. A closed-for-the-whole-run world answers `True` everywhere,
        since it has no opener to reach anything; otherwise the answer is `False` unless `node`
        locates into the root graph outside the poisoned region.

        Staleness is read first, before the whole-run shortcut: a wrapper built over a tree that has
        since changed answers `False` even where its stale verdict is closed, because an edit could
        have opened a world the old walk read shut. Anything the graph cannot place — a `node` in no
        body, or one in a scriptblock or function body, whose second invocation this intraprocedural
        model cannot order against a leak — is refused too, which is the fail-closed direction: a
        wrong grant deletes an effect, a refusal keeps a read.
        """
        if self._stale:
            return False
        if self._world.closed_for_the_whole_run:
            return True
        if self._refuse or self._control_flow is None or self._root is None:
            return False
        located = self._control_flow.locate(node)
        if located is None or located[0].owner is not self._root:
            return False
        return id(located[1]) not in self._poisoned


def build_world_reach(
    measurement: Ps1WorldMeasurement,
    control_flow_of: Callable[[], ControlFlowModel],
) -> Ps1WorldReach:
    """
    The flow-sensitive world for `measurement.root`, over its whole-run verdict and its opener
    positions. A world already closed for the whole run carries no position — nothing opens it — so
    it is wrapped with no graph, and `control_flow_of` is never called: a clean script never pays
    for a control-flow build it would not read. Otherwise every opener is lifted into the root graph
    and the poisoned region is the forward flood from all of them.

    The verdict falls back to the whole-run answer at every position — a `Ps1WorldReach` built with
    `refuse` — whenever a position-less opener is present (a `class`/`enum` definition, a root
    `process` block that re-runs), an opener cannot be placed in the root graph, or the script names
    its own path (`_names_own_path`), through which a leak could re-run the statements before it.
    Each is the fail-closed direction the module docstring states: the flood cannot bound where such
    an opener ran, so no read is granted over it.

    Every wrapper — closed, refused, or measured — is stamped with the root and the version the
    measurement was taken at, so each notices the tree changing under a pass that holds it. A
    measurement already stale against the current tree is refused whole: its opener list may miss a
    leak an edit introduced, and stamping the stale version makes the wrapper read stale at once, so
    no answer is trusted.
    """
    world = measurement.world
    root = measurement.root
    version = measurement.build_version
    if tree_version(root) != version:
        return Ps1WorldReach(world, root=root, refuse=True, build_version=version)
    if world.closed_for_the_whole_run:
        return Ps1WorldReach(world, root=root, build_version=version)
    control_flow = control_flow_of()
    root_graph = control_flow.graph_of(root)
    if root_graph is None:
        return Ps1WorldReach(world, root=root, refuse=True, build_version=version)
    refuse = root.process_block is not None or _names_own_path(root)
    sources: list[CfgNode] = []
    for opener in measurement.openers:
        if isinstance(opener, (Ps1ClassDefinition, Ps1EnumDefinition)):
            refuse = True
            continue
        landing = _lift_to_root(control_flow, opener, root_graph)
        if landing is None:
            refuse = True
            continue
        sources.append(landing)
    # A measured open world has at least one opener by construction — the verdict and `openers` come
    # from the one `measure_world` walk — so an empty `sources` here means every opener was a
    # class/enum or unplaceable, which already set `refuse`. The guard is a belt-and-braces floor:
    # flooding from nothing would grant every position.
    if refuse or not sources:
        return Ps1WorldReach(world, root=root, refuse=True, build_version=version)
    poisoned = reachable_from_any(sources)
    return Ps1WorldReach(
        world,
        root=root,
        control_flow=control_flow,
        poisoned=poisoned,
        build_version=version,
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
