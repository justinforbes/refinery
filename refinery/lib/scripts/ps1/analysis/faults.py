"""
Where a terminating error raised at a point in a PowerShell script goes.

The routing itself is already in the control-flow graph:
`refinery.lib.scripts.ps1.analysis.cfg` joins every statement of a guarded region to the handler
that may be offered its errors, chains `catch` clauses and `trap` sets in the order the engine
consults them, and edges to the body exit where the error gets past all of them. What is here is
the reading of that graph — which handlers a point reaches, whether the error may leave the body,
and the transpose, which points a handler is reachable from.

**Three questions, and the difference between them is the whole of this module.** *Does an error
raised here reach a handler that acts* is a property of a **position**, and it is what a pass asks
before it empties a guarded body. *Would deleting this statement change which handler runs* is a
property of a **statement**, and it additionally needs to know whether the statement can raise at
all — `refinery.lib.scripts.ps1.analysis.effects.fault_is_observed` asks that one, because the
predicate that answers it lives there. *Is this handler still reachable from anything that raises*
is the **transpose**, and it is what a pass asks before deleting a `trap`, where the statement being
removed cannot itself raise and every error it re-routes belongs to something else.

A body boundary is where this stops. The graphs are per body, so an error that leaves a function
reaches whatever guards the *call*, and no graph here holds both. `leaves_the_body` reports that
rather than guessing, and a caller reads it as *unknown* wherever a caller might be guarding.
"""
from __future__ import annotations

from typing import Iterator, NamedTuple

from refinery.lib.scripts import Node
from refinery.lib.scripts.analysis.cfg import CfgNode, ControlFlowGraph, ControlFlowModel
from refinery.lib.scripts.ps1.model import (
    Ps1BreakStatement,
    Ps1CatchClause,
    Ps1ContinueStatement,
    Ps1Script,
    Ps1TrapStatement,
)


class Ps1FaultRouting(NamedTuple):
    """
    Where a terminating error raised at one point may go: the handlers it may be offered to, in no
    particular order, and whether it may get past all of them and leave the body.

    `handlers` holds `Ps1CatchClause` and `Ps1TrapStatement` nodes and is a *may* set — a type
    filter is matched by inheritance at run time, so a clause that cannot be shown to miss is
    reported alongside the one after it.

    The two fields answer different questions and a caller reads both. A handler that acts makes the
    error observable however the rest of the routing looks; `leaves_the_body` beside a `trap` is the
    escalation that ends the body, and beside nothing at all it is the ordinary unhandled error that
    5.1 reports and steps over.
    """
    handlers: tuple[Ps1CatchClause | Ps1TrapStatement, ...]
    leaves_the_body: bool


#: What the graphs place nothing for, and therefore claim nothing about. Every query answers `None`
#: for a node it cannot place, and every caller reads that as *the error may go anywhere*.
UNPLACED = None


def handler_acts(handler: Ps1CatchClause | Ps1TrapStatement) -> bool:
    """
    Whether running *handler* changes what the script does.

    An empty `catch { }` swallows the error and lets execution resume after the construct, so a
    script that never raises reaches the same next statement — which is why an obfuscator writes one
    and why removing what it guards costs nothing. A `trap` is the same rule with one more spelling:
    a body holding only an unlabelled `break` or `continue` decides how the error is disposed of and
    emits nothing, while a body holding anything else — a call, an assignment, or a bare value the
    engine writes to the output stream — is a handler whose running is observable.
    """
    body = handler.body.body if handler.body is not None else ()
    if isinstance(handler, Ps1CatchClause):
        return bool(body)
    for statement in body:
        if isinstance(statement, (Ps1BreakStatement, Ps1ContinueStatement)):
            if statement.label is not None:
                return True
            continue
        return True
    return False


class Ps1FaultReach:
    """
    The fault routing of one script, read off its control-flow graphs.

    Built over `refinery.lib.scripts.analysis.cfg.ControlFlowModel` and nothing else: where an error
    goes is decided by the graph the builder already wired, so this needs no semantic model, no call
    graph and no world. It is a view rather than a solver — every answer is one walk over the
    exceptional edges, memoized per node for the life of the model, and the model itself is
    discarded whenever the tree moves.
    """

    def __init__(self, control_flow: ControlFlowModel):
        self._control_flow = control_flow
        self._forward: dict[int, Ps1FaultRouting] = {}
        self._backward: dict[int, bool] = {}
        self._handled: set[int] | None = None

    def routing_at(self, node: Node) -> Ps1FaultRouting | None:
        """
        Where an error raised at *node* may go, or `None` when the graphs place *node* nowhere —
        a body they do not descend into, or an expression evaluated at no point they model.
        """
        located = self._control_flow.locate(node)
        if located is None:
            return UNPLACED
        graph, start = located
        remembered = self._forward.get(id(start))
        if remembered is None:
            remembered = self._forward[id(start)] = self._route(graph, start)
        return remembered

    def points_in(self, node: Node) -> Iterator[Node]:
        """
        The points inside *node* at which the graphs say something is evaluated, *node* itself
        included where it is one.

        Deleting a construct deletes everything it holds, so the errors it stops raising are its
        body's as well as its own — and each is at a position of its own, since a `try` nested in it
        routes its body's errors to its own `catch`. What the graphs place is exactly the list of
        such positions, which is why it is read off them rather than guessed from the node types: a
        `for` loop is three points and a body, an `if` is one point per test, and neither construct
        has a node standing for the statement as a whole.

        A construct that yields nothing at all is one the graphs model nowhere, and a caller reads
        that as a subtree it cannot judge rather than as one that raises nothing.
        """
        if self._control_flow.node_of(node) is not None:
            yield node
        for inner in node.walk():
            if inner is not node and self._control_flow.node_of(inner) is not None:
                yield inner

    def observed_at(self, node: Node) -> bool:
        """
        Whether an error raised at *node* changes which code runs: some handler it reaches acts, a
        `trap` set may decline it and end the body, or the graphs place *node* nowhere.

        This is the question about the **position** and not about the statement standing there —
        it answers the same for a statement that cannot raise at all, which is what a caller
        weighing whether a guarded body may be emptied wants to know.

        A `trap` beside `leaves_the_body` is escalation: the set was offered the error and may have
        declined it, and 5.1 then ends the body rather than reporting the error and stepping over
        it, so everything written after the raise stops running. A `catch` that misses does not do
        that — the sharp asymmetry between the two keywords — so the reading is keyed to the `trap`
        and not to the escape.

        **An error that gets past a function's own handlers is the caller's**, and no graph here
        holds both ends of a call. Measured: the same function whose error is reported and stepped
        over when called at script scope abandons its remaining statements and runs the `catch` of a
        `try` written around the *call*, however many bodies deep the raise is. So a raise in a body
        something may call is refused wherever the script holds a handler that acts — and granted
        where it holds none, since a `catch` that is not written cannot be the one guarding the call.
        """
        located = self._control_flow.locate(node)
        if located is None:
            return True
        graph, _ = located
        routing = self.routing_at(node)
        if routing is None:
            return True
        if any(handler_acts(handler) for handler in routing.handlers):
            return True
        if routing.leaves_the_body and any(
            isinstance(handler, Ps1TrapStatement) for handler in routing.handlers
        ):
            return True
        if routing.handlers and not routing.leaves_the_body:
            return False
        return not isinstance(graph.owner, Ps1Script) and self._handled_elsewhere(graph)

    def _handled_elsewhere(self, body: ControlFlowGraph) -> bool:
        """
        Whether this script writes a handler that acts in some body other than *body*. What guards a
        call is unknowable from the called body's graph, and this is what settles it in the direction
        of a grant: a handler written in the very body that raises is one the error has already got
        past, and a script with none anywhere else has no call site the raise could matter at.

        A named block is not a body of its own here, which is why the comparison is per graph: a
        `trap` in `begin` and a raise in `process` share one script block, and neither guards the
        other.
        """
        if self._handled is None:
            self._handled = {
                id(graph)
                for graph in self._control_flow.graphs.values()
                if any(
                    isinstance(node.element, (Ps1CatchClause, Ps1TrapStatement))
                    and handler_acts(node.element)
                    for node in graph.nodes
                )
            }
        return bool(self._handled - {id(body)})

    def leaves_the_body(self, node: Node) -> bool:
        """
        Whether an error raised at *node* may get past every handler of the body it is in, so that
        where it goes next is a question about the caller. `True` for a node the graphs cannot
        place, which is the same answer for the same reason.
        """
        routing = self.routing_at(node)
        return routing is None or routing.leaves_the_body

    def removing_a_handler_is_observed(self, handler: Node) -> bool:
        """
        Whether deleting *handler* may change which code runs — the transpose, and the question
        asked before a `trap` is deleted.

        A `trap` cannot itself raise, so the forward question answers nothing about removing one:
        what changes is where the errors of *other* statements go. Two things have to be true for
        that to matter, and asking only one gets it wrong in either direction.

        **Something has to still reach it.** Walking the exceptional edges backwards from the
        handler's own node finds exactly the statements whose errors it may be offered, and a
        handler nothing reaches is one whose deletion re-routes nothing — which is how a `trap` left
        behind by the removal of the only raise in its block becomes removable in turn. The walk
        crosses other handlers, because a `catch` that misses hands the error on and the statement
        that raised it is behind that clause rather than at it.

        **And what it would fall back to has to act.** Deleting a handler sends its errors to
        whatever the graph records as its fallback, so the question is asked again there: an empty
        `catch` beside it swallows exactly as it did, and a `trap` at script scope with nothing
        outside it lets an error be reported and stepped over, which is what an unhandled error
        already did. A fallback the graph does not record is refused, because a handler whose
        counterfactual is unknown is one whose removal cannot be judged.

        **A `trap` set that may decline is load bearing for declining**, not only for handling. It
        is what turns an error 5.1 would report and step over into one that ends the body, so
        deleting it starts running everything written after the raise — `trap [System.IO.IOException]
        { }` guards nothing and is still the whole reason a script stops where it does.
        """
        located = self._control_flow.locate(handler)
        if located is None:
            return True
        graph, start = located
        remembered = self._backward.get(id(start))
        if remembered is None:
            remembered = self._backward[id(start)] = self._removal_matters(graph, start)
        return remembered

    def _removal_matters(self, graph: ControlFlowGraph, start: CfgNode) -> bool:
        if not self._raisers(graph, start):
            return False
        routing = self._route(graph, start)
        if isinstance(start.element, Ps1TrapStatement) and routing.leaves_the_body:
            return True
        if any(handler_acts(handler) for handler in routing.handlers):
            return True
        fallback = graph.fallback_of(start)
        if fallback is None:
            return True
        return self._observed_from(graph, fallback)

    def _observed_from(self, graph: ControlFlowGraph, arrival: CfgNode) -> bool:
        """
        Whether an error arriving at *arrival* changes what runs: *arrival* is a handler that acts,
        or the routing onward from it reaches one.
        """
        element = arrival.element
        if isinstance(element, (Ps1CatchClause, Ps1TrapStatement)) and handler_acts(element):
            return True
        routing = self._route(graph, arrival)
        if any(handler_acts(reached) for reached in routing.handlers):
            return True
        return routing.leaves_the_body and any(
            isinstance(reached, Ps1TrapStatement) for reached in routing.handlers
        )

    def _route(self, graph: ControlFlowGraph, start: CfgNode) -> Ps1FaultRouting:
        handlers: list[Ps1CatchClause | Ps1TrapStatement] = []
        leaves = False
        for node in self._exceptional_closure(graph, start, forward=True):
            if node is graph.exit:
                leaves = True
            elif isinstance(node.element, (Ps1CatchClause, Ps1TrapStatement)):
                handlers.append(node.element)
        return Ps1FaultRouting(tuple(handlers), leaves)

    def _raisers(self, graph: ControlFlowGraph, start: CfgNode) -> bool:
        for node in self._exceptional_closure(graph, start, forward=False):
            element = node.element
            if element is None:
                continue
            if not isinstance(element, (Ps1CatchClause, Ps1TrapStatement)):
                return True
        return False

    @staticmethod
    def _exceptional_closure(
        graph: ControlFlowGraph, start: CfgNode, *, forward: bool,
    ) -> Iterator[CfgNode]:
        """
        The nodes reachable from *start* over exceptional edges alone, in either direction, without
        *start* itself. One sweep rather than a walk per edge kind: the graph marks an edge by the
        pair of nodes it joins, so the direction decides only which end of the pair the walk holds.
        """
        seen = {id(start)}
        stack = [start]
        while stack:
            current = stack.pop()
            adjacent = current.successors if forward else current.predecessors
            for node in adjacent:
                edge = (current, node) if forward else (node, current)
                if not graph.is_exceptional(*edge) or id(node) in seen:
                    continue
                seen.add(id(node))
                yield node
                stack.append(node)


def build_fault_reach(control_flow: ControlFlowModel) -> Ps1FaultReach:
    """
    The `Ps1FaultReach` over one script's control-flow graphs.
    """
    return Ps1FaultReach(control_flow)
