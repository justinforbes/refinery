"""
Which write a PowerShell variable read observes.

This is the join of three layers that already exist and have never been asked one question together:
`refinery.lib.scripts.ps1.analysis.model.Ps1SemanticModel` says which occurrences name the same
binding, `refinery.lib.scripts.analysis.cfg.ControlFlowModel` says what runs before what, and
`refinery.lib.scripts.ps1.analysis.blocks.Ps1BlockModel` says whose variables a script block writes.
The join is possible because the graphs and the scopes now partition the script the same way — one
graph and one scope per `refinery.lib.scripts.ps1.model.Ps1ScriptBlock`, plus one of each for the
root — so a binding's writes and a read of it land in comparable places.

**A write is a write.** Whether its value happens to be a constant is the *caller's* question and is
asked after this one, never before it. The pass this replaces sorted writes into two tables by that
test and only one table killed, so `if ($c) { $x = 'b' }` silently kept the value from before the
branch while `if ($c) { $x = $y }` correctly refused — same position, same graph, opposite answers.
Nothing here may reintroduce that split.

**One refusal this layer owes its callers that is not graph-theoretic: a store that did not finish.**
Dominance says a statement ran, not that its store completed —
`refinery.lib.scripts.analysis.liveness` states the same asymmetry as the reason its transfer function
is not the textbook one — so `try { [int]$x = 'abc' } catch { Write-Host $x }` must not publish
`'abc'`, because the run that enters the handler is exactly the run in which the cast raised and the
store never happened. Dominance cannot see this: the handler *is* dominated by the statement that
failed to store it. `_reached_after_completing` is the rule, and it is deliberately about the first
edge only — once control has left the statement normally, the store is done.

**A read that cannot be ordered against its writes is not answered.** A block is a value, so `$b = {
Write-Host $x }` may be invoked before or after any statement here; ordering the read inside it
against the script around it reads the script as if the block ran there. That is not a special case —
it falls out of `WRITE_IN_ANOTHER_BODY`, since the read and the writes then sit in different graphs.
A read whose writes are all in its *own* body is answered normally, block or not, which is why there
is no blanket refusal for blocks.

A write inside a `trap` is a case this layer gets right for the wrong reason, and the distinction
matters if either half is touched. 5.1 runs a trap body in a child scope, so the write cannot reach a
read outside the trap at all; `refinery.lib.scripts.ps1.analysis.model.Ps1SemanticModel` does not model
that and binds it to the enclosing scope. What refuses the answer is the kill rule — the trap's write
is another write of the same binding, and the exceptional edge into the handler with the resume edge
back out puts it on a path between any earlier write and that read. A rule keyed on the handler being
entered exceptionally was tried instead and removed: it refuses `catch { $x = 'b'; Write-Host $x }`,
where the handler's own store is genuinely what its own read sees.
"""
from __future__ import annotations

import enum

from refinery.lib.scripts.analysis.cfg import CfgNode, ControlFlowGraph, ControlFlowModel
from refinery.lib.scripts.analysis.dominance import DominatorModel
from refinery.lib.scripts.analysis.reaching import ReachabilityQuery
from refinery.lib.scripts.ps1.analysis.blocks import Ps1BlockModel, Ps1BlockReach
from refinery.lib.scripts.ps1.analysis.model import Binding, Ps1SemanticModel
from refinery.lib.scripts.ps1.model import Ps1ScriptBlock, Ps1Variable


class Ps1FlowUnknown(enum.Flag):
    """
    Why a binding's values cannot be tracked. A flag rather than a boolean because the reasons are
    not the same kind of fact and a caller may be able to live with one and not another — and because
    one predicate folding several unrelated refusals is the shape this package has already had to
    take apart once.
    """
    NONE = 0
    #: A write occurrence the control-flow graphs do not place. Its point of evaluation is not a
    #: point these graphs hold, so nothing can be ordered against it.
    UNPLACED_WRITE = enum.auto()
    #: A write in a different body from a read of the same binding. Whether one body runs before
    #: another is a question about calls, which this layer does not answer.
    WRITE_IN_ANOTHER_BODY = enum.auto()
    #: The binding is reachable through a scope qualifier or a dynamic scope, so an occurrence that
    #: does not appear in `reads` or `writes` at all may still touch it.
    REACHED_BY_QUALIFIER = enum.auto()
    #: A body that may write this binding runs at a time this layer cannot place — a stored block, a
    #: block handed to a command that may or may not invoke it.
    WRITTEN_BY_DEFERRED_BODY = enum.auto()


class Ps1VariableFlow:
    """
    Which write each variable read of one script observes, over that script's semantic, control-flow
    and block models. Build it through `build_variable_flow`.
    """

    def __init__(
        self,
        semantic: Ps1SemanticModel,
        flow: ControlFlowModel,
        blocks: Ps1BlockModel,
    ):
        self.semantic = semantic
        self.flow = flow
        self.blocks = blocks
        self._dominators = DominatorModel(flow)
        self._between = ReachabilityQuery(self._dominators)
        self._unknowns: dict[int, Ps1FlowUnknown] = {}
        self._completed: dict[tuple[int, int], frozenset[int]] = {}

    def reaching_definition(self, read: Ps1Variable) -> Ps1Variable | None:
        """
        The write occurrence whose value *read* observes, or `None` when no single write does.

        `None` is the answer to every kind of doubt — the binding is unknown, the read is in a body
        this layer cannot place, two writes reach, a write between them may have changed the value —
        so a caller may treat a returned occurrence as the one and only value the read can see, and
        must treat `None` as knowing nothing.
        """
        binding = self.semantic.binding_of(read)
        if binding is None:
            return None
        if self.unknowns(binding) is not Ps1FlowUnknown.NONE:
            return None
        located = self.flow.locate(read)
        if located is None:
            return None
        graph, use = located
        definitions = [
            (write, self.flow.locate(write)[1]) for write in binding.writes
        ]
        found = self._between.reaching_definition(
            graph,
            use,
            definitions,
            self._block_kills(graph, binding),
        )
        if found is None:
            return None
        definition = self.flow.locate(found)[1]
        if id(use) not in self._reached_after_completing(graph, definition):
            return None
        return found

    def unknowns(self, binding: Binding) -> Ps1FlowUnknown:
        """
        Every reason *binding*'s values cannot be tracked, or `Ps1FlowUnknown.NONE` when there is
        none. Fixed for as long as the tree is, so it is computed once per binding.
        """
        found = self._unknowns.get(id(binding))
        if found is None:
            found = self._unknowns[id(binding)] = self._compute_unknowns(binding)
        return found

    def _compute_unknowns(self, binding: Binding) -> Ps1FlowUnknown:
        found = Ps1FlowUnknown.NONE
        if binding.dynamic_or_qualified:
            found |= Ps1FlowUnknown.REACHED_BY_QUALIFIER
        graphs: set[int] = set()
        for occurrence in (*binding.writes, *binding.reads):
            placed = self.flow.locate(occurrence)
            if placed is None:
                found |= Ps1FlowUnknown.UNPLACED_WRITE
                continue
            graphs.add(id(placed[0]))
        if len(graphs) > 1:
            found |= Ps1FlowUnknown.WRITE_IN_ANOTHER_BODY
        if self._deferred_body_writes(binding):
            found |= Ps1FlowUnknown.WRITTEN_BY_DEFERRED_BODY
        return found

    def _deferred_body_writes(self, binding: Binding) -> bool:
        """
        Whether a block whose run time this layer cannot place may write *binding*. A stored block is
        a value: it may be invoked before or after any statement here, so a write inside it defeats
        every ordering the graphs could establish.

        The binding's own body is not one of those blocks. A stored block's statements are perfectly
        ordered against *each other* however late the block runs, and counting it against itself
        makes every binding local to a stored block unanswerable — which reads as caution and is
        simply a wrong reading of the question.
        """
        for block in self._blocks_of(binding.scope.node):
            if block is binding.scope.node:
                continue
            facts = self.blocks.facts(block)
            if facts.reach not in (Ps1BlockReach.STORED, Ps1BlockReach.UNKNOWN):
                continue
            for write in self.blocks.writes_reaching_caller(block):
                if write.name.lower() == binding.name:
                    return True
        return False

    @staticmethod
    def _blocks_of(owner) -> list[Ps1ScriptBlock]:
        return [node for node in owner.walk() if isinstance(node, Ps1ScriptBlock)]

    def _block_kills(self, graph: ControlFlowGraph, binding: Binding) -> set[int]:
        """
        The nodes of *graph* that run a script block writing *binding* into the scope that invokes
        it. A `. { $x = 'b' }` is one statement to the graph and its store is invisible in the tree
        around it, so without this the caller's `$x` reads as never having been touched.
        """
        kills: set[int] = set()
        for block in self._blocks_of(graph.owner):
            if not any(
                write.name.lower() == binding.name
                for write in self.blocks.writes_reaching_caller(block)
            ):
                continue
            placed = self.flow.locate(block)
            if placed is not None and placed[0] is graph:
                kills.add(id(placed[1]))
        return kills

    def _reached_after_completing(
        self,
        graph: ControlFlowGraph,
        definition: CfgNode,
    ) -> frozenset[int]:
        """
        The ids of the nodes reached from *definition* by first leaving it *normally* — that is, on
        a path where the statement holding the definition ran to completion.

        A node reached only by throwing out of *definition* is left out, and that is the whole point:
        `try { [int]$x = 'abc' } catch { … }` enters the handler because the cast failed, which is
        exactly the run in which the store never happened. Dominance cannot see this, since the
        handler is dominated by the statement that failed to store. Once control has left the
        statement normally the store is done, so every edge after the first step is followed.
        """
        key = (id(graph), id(definition))
        found = self._completed.get(key)
        if found is not None:
            return found
        seen: set[int] = set()
        stack: list[CfgNode] = []
        for successor in definition.successors:
            if graph.is_exceptional(definition, successor):
                continue
            if id(successor) not in seen:
                seen.add(id(successor))
                stack.append(successor)
        while stack:
            node = stack.pop()
            for successor in node.successors:
                if id(successor) not in seen:
                    seen.add(id(successor))
                    stack.append(successor)
        found = frozenset(seen)
        self._completed[key] = found
        return found


def build_variable_flow(
    semantic: Ps1SemanticModel,
    flow: ControlFlowModel,
    blocks: Ps1BlockModel,
) -> Ps1VariableFlow:
    """
    Build the `Ps1VariableFlow` for a script from the three models it joins.
    """
    return Ps1VariableFlow(semantic, flow, blocks)
