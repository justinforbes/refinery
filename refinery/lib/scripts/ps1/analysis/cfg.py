"""
PowerShell's contribution to the shared control-flow substrate: which node types are which
control-flow shape, and where the parts of each are stored.

Everything structural lives in `refinery.lib.scripts.analysis.cfg`. What is here is the dispatch, the
accessors, and the places PowerShell's answer differs from the shape JavaScript established:

- **`switch` does not fall through, it keeps matching.** Every clause is tested against the value and
  every matching one runs, so an arm's exits may reach any *later* arm rather than only the next.
  That is `refinery.lib.scripts.analysis.cfg.ArmFlow.CUMULATIVE`, and reading it as C-style
  fallthrough would drop the path in which the first and third clauses match and the second does not.
  It also *enumerates its input*, so `continue` inside it advances to the next input value rather
  than to an enclosing loop, and a `default` clause does not make it exhaustive: over an empty
  collection no clause runs at all.
- **`if`/`elseif` is one flat statement**, not a nest, so the chain of tests is built through
  `branch_chain` and each `elseif` condition gets a node of its own.
- **`trap` guards the statement block it is written in**, not the scope around it: it catches for
  that block entire, statements written above it included, and for nothing outside it, so one
  written inside an `if` body is no handler for the body around it. Where blocks nest, the innermost
  block declaring one is the only set consulted, and a set that may fail to match passes the error
  outward rather than letting it resume. `continue` inside it resumes at the statement after the one
  that threw. See `_Builder.block`.
- **`exit` is a terminator like `return`**, modelled as one: control does not continue in this body.
  That it leaves the *script* rather than the function is a claim about the caller's body, which a
  per-body graph does not make; keeping the caller's paths is the conservative reading.
- **A label is a field of the construct**, not a statement wrapping it, so `park_label` is called
  from the dispatch rather than through `labelled`, and the `:` the lexer keeps on the declaration
  is stripped before a jump can be matched against it.

`Ps1Code` also splits one body across `begin`/`process`/`end`/`dynamicparam` blocks, so `owner.body`
alone is not what an advanced function runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from refinery.lib.scripts import Block, Node
from refinery.lib.scripts.analysis.cfg import (
    ArmFlow,
    CfgBuilder,
    CfgNode,
    ControlFlowGraph,
    ControlFlowModel,
    build_control_flow,
)
from refinery.lib.scripts.ps1.ast import get_named_blocks, string_value
from refinery.lib.scripts.ps1.model import (
    Ps1BreakStatement,
    Ps1Code,
    Ps1ContinueStatement,
    Ps1DoLoop,
    Ps1ExitStatement,
    Ps1ForEachLoop,
    Ps1ForLoop,
    Ps1IfStatement,
    Ps1Jump,
    Ps1ReturnStatement,
    Ps1Script,
    Ps1ScriptBlock,
    Ps1SwitchStatement,
    Ps1ThrowStatement,
    Ps1TrapStatement,
    Ps1TryCatchFinally,
    Ps1WhileLoop,
)

#: The nodes that own a control-flow graph of their own, beside the script itself. A script block is
#: PowerShell's anonymous function: it is a *value* where it is written and runs somewhere else
#: entirely — as the body of `ForEach-Object`, through `&` or `.`, or as the body of a function,
#: which is the only script block a `Ps1FunctionDefinition` has and why keying on the definition
#: instead adds no graph but leaves every other block without one. A block with no graph is climbed
#: out of, so its statements are reported as running where the block is *written*: they claim to
#: precede everything after that point and lose their order among themselves.
#:
#: This is the same partition `refinery.lib.scripts.ps1.analysis.model.Ps1SemanticModel.scope_of`
#: makes — one body, one scope, one graph. A `trap` divides more finely still: it belongs to the
#: statement block it is written in, of which a body is only the outermost.
FUNCTION_NODES = (Ps1ScriptBlock,)

_LOOP_NODES = (
    Ps1WhileLoop,
    Ps1DoLoop,
    Ps1ForLoop,
    Ps1ForEachLoop,
)


def _jump_label(statement: Ps1Jump) -> str | None:
    """
    The label a `break` or `continue` names, or `None` when it names none or names one this cannot
    read. PowerShell allows the label to be any expression — `break $name` is legal — and a label
    that is not a static string is one no lexical target can be matched against, so it is treated as
    unlabelled, which resolves to the innermost enclosing target and is the conservative reading.
    """
    return string_value(statement.label) if statement.label is not None else None


def _construct_label(label: str | None) -> str | None:
    """
    The name a labelled loop or `switch` declares, without the `:` the lexer keeps on the token.

    A jump names its target without the colon, so the declaration and the reference are two
    spellings of one name and comparing them unnormalized never matches — which resolves every
    labelled jump to nothing and sends it out of the body instead of out of the construct it names.
    """
    return label[1:] if label is not None and label.startswith(':') else label


def _declared_traps(statements: Sequence[Node]) -> list[Ps1TrapStatement]:
    """
    The `trap` statements *statements* declares, in source order.

    A `trap` belongs to the statement block it is written in and to no other: it catches for that
    block entire, including the statements written above it, and an error raised outside the block
    is never offered to it. Collecting the traps of a nested block for the block around it hands
    those errors to a handler no run reaches, and — where the nested one is typed — reports a filter
    that may miss as guarding statements it does not guard at all.
    """
    return [statement for statement in statements if isinstance(statement, Ps1TrapStatement)]


@dataclass
class _TrapBody:
    """
    What building one `trap` body reported back: the nodes at which control resumes the block the
    `trap` guards, and whether the body reaches the `break` that rethrows.

    `resumes` collects the explicit `continue` spellings only. A body that runs off its end resumes
    too — the error record is written and the block carries on — but that is its exit frontier,
    which the caller already holds.

    `rethrows` is decided through the builder's jump-target stack rather than by finding the
    keyword: a `break` naming a loop written inside the body leaves that loop, and the `trap` goes
    on swallowing.
    """
    resumes: list[CfgNode] = field(default_factory=list)
    rethrows: bool = False


class _Builder(CfgBuilder):
    """
    The PowerShell dispatch over `refinery.lib.scripts.analysis.cfg.CfgBuilder`.
    """

    def __init__(self, owner: Node):
        super().__init__(owner)
        self._trap_body: _TrapBody | None = None

    def block(self, statements: Sequence[Node], frontier: list[CfgNode]) -> list[CfgNode]:
        """
        One statement block, with every `trap` it declares installed as a handler over the whole of
        it.

        A `trap` is not a guarded block. It is declared somewhere in the block and catches for that
        block entire, including the statements written above it, so it cannot be pushed at the point
        it appears the way a `try` is — it is pushed before the block is walked at all. Several
        traps in one block are peers and are chained the way
        `refinery.lib.scripts.analysis.cfg.CfgBuilder.guarded` chains several `catch` clauses,
        because which one runs depends on the type of the error and none of them is guaranteed.

        Where blocks nest, the innermost one declaring a `trap` is the only set the error is offered
        to, which pushing that set on the handler stack already says. The set passes the error
        outward — to the `catch` or `trap` guarding the block, and to the body exit when nothing
        guards it — exactly when it may fail to take it: every `trap` in it carrying a type filter,
        so none is known to match, or one of them reaching the `break` that runs the body and
        rethrows. Nothing else in a body reaches the exit on an exceptional edge without a handler
        having been offered the error, which is what tells an error that ends the script apart from
        one that is reported and stepped over.

        A trap does not guard itself: an error raised inside a trap body is offered to whatever
        the set itself unwinds to, so the trap bodies are built under that and only the block's own
        statements are built under the set. Building them under the set instead gives every node of
        every trap body an exceptional edge back to a handler that already has a normal edge into
        it, which is a cycle through the trap that no run can take — and reads as a body that
        repeats. `refinery.lib.scripts.analysis.cfg.CfgBuilder.guarded` pops before building its
        `catch` bodies for the same reason. Where nothing guards the block, the trap body unwinds to
        the body exit: the error ends the body, which is the one place a `trap` turns a fault that
        would have been reported and stepped over into one that stops everything after it.

        `continue` inside a trap resumes at the statement following the one that threw, which is a
        shape this graph cannot express exactly. Every resumption point is therefore taken to reach
        *every* node the block created (`_link_resumptions`), which is the over-approximation: it
        claims more paths than exist, where claiming fewer would let an analysis call a statement
        after a trap unreachable, or let a store before one look dead because the resumption that
        reads it was never modelled.
        """
        traps = _declared_traps(statements)
        if not traps:
            return self.sequence(statements, frontier)
        entries: list[CfgNode] = []
        for trap in traps:
            handler = self.detached_node(trap)
            if entries:
                self.exceptional_edge(entries[-1], handler)
            entries.append(handler)
        resumes: list[CfgNode] = []
        escapes = True
        enclosing = self._trap_body
        unwinding = self.unwinding()
        self._handlers.append(unwinding)
        for trap, handler in zip(traps, entries):
            self._trap_body = built = _TrapBody()
            resumes += self._body(trap.body, [handler]) + built.resumes
            if not trap.type_name and not built.rethrows:
                escapes = False
        self._handlers.pop()
        self._trap_body = enclosing
        if escapes:
            self.exceptional_edge(entries[-1], unwinding)
        guarded_from = len(self.cfg.nodes)
        self._handlers.append(entries[0])
        exits = self.sequence(statements, frontier)
        self._handlers.pop()
        landing = [node for node in self.cfg.nodes[guarded_from:] if node.element is not None]
        if resumes:
            self._link_resumptions(resumes, landing)
        return exits

    def _link_resumptions(self, resumes: list[CfgNode], landing: list[CfgNode]) -> None:
        """
        Wire every trap resumption to every landing point of the block and to the exit, the
        over-approximation `block` documents, through one synthetic hub rather than an edge from
        each resume to each landing. A resume reaching a landing through the hub is the same
        reachability an analysis reads off a direct edge — the hub carries no element, so it locates
        into no query and generates no dataflow fact — while the edge count falls from the product
        of the two sets to their sum, which is what a guarded body holding hundreds of resumes over
        thousands of landings costs when every resume names every landing directly.
        """
        hub = CfgNode(None)
        self.cfg.nodes.append(hub)
        self.add_edge(hub, self.cfg.exit)
        for node in landing:
            self.add_edge(hub, node)
        for resume in resumes:
            self.add_edge(resume, hub)

    def body_blocks(self, owner: Node) -> Sequence[Sequence[Node]]:
        """
        The statement blocks *owner* runs, in the order it runs them.

        An advanced function fills `begin`/`process`/`end`/`dynamicparam` instead of `body`, and
        reading only `body` for one would report an empty graph for a function that runs a great
        deal — the same trap `refinery.lib.scripts.ps1.ast.get_named_blocks` exists to warn about.
        `dynamicparam` is pulled to the front because the engine evaluates it during parameter
        binding, before `begin`, while that accessor reports the blocks in the order they are
        declared in.

        They are kept apart rather than concatenated because each is a statement block of its own: a
        `trap` written in `begin` is offered nothing that `process` raises, and joining them hands
        it errors no run ever will.
        """
        if not isinstance(owner, Ps1Code):
            return []
        blocks = sorted(
            get_named_blocks(owner),
            key=lambda block: block is not owner.dynamicparam_block,
        )
        return [block.body for block in blocks] + [owner.body]

    def statement(self, statement: Node, frontier: list[CfgNode]) -> list[CfgNode]:
        if isinstance(statement, Block):
            return self.block(statement.body, frontier)
        if isinstance(statement, _LOOP_NODES):
            self.park_label(_construct_label(statement.label))
        if isinstance(statement, Ps1IfStatement):
            return self._if(statement, frontier)
        if isinstance(statement, Ps1WhileLoop):
            return self.loop_head_tested(statement, statement.body, frontier)
        if isinstance(statement, Ps1DoLoop):
            return self.loop_tail_tested(statement, statement.body, frontier)
        if isinstance(statement, Ps1ForLoop):
            return self.loop_counted(
                statement.initializer,
                statement.condition,
                statement.iterator,
                statement.body,
                frontier,
            )
        if isinstance(statement, Ps1ForEachLoop):
            return self.loop_head_tested(statement, statement.body, frontier)
        if isinstance(statement, Ps1SwitchStatement):
            return self._switch(statement, frontier)
        if isinstance(statement, Ps1TryCatchFinally):
            return self._try(statement, frontier)
        if isinstance(statement, Ps1TrapStatement):
            return list(frontier)
        if isinstance(statement, Ps1ExitStatement):
            return self.terminate(statement, frontier, exceptional=False)
        if isinstance(statement, Ps1ThrowStatement):
            return self.terminate(statement, frontier, exceptional=True)
        if isinstance(statement, Ps1ReturnStatement):
            return self.terminate(statement, frontier, exceptional=False)
        if isinstance(statement, Ps1BreakStatement):
            label = _jump_label(statement)
            trap = self._trap_body
            if trap is not None and not self.has_break_target(label):
                trap.rethrows = True
            return self.jump_out(statement, label, frontier)
        if isinstance(statement, Ps1ContinueStatement):
            label = _jump_label(statement)
            trap = self._trap_body
            if trap is not None and not self.has_continue_target(label):
                return self._resume(trap.resumes, statement, frontier)
            return self.jump_back(statement, label, frontier)
        return self.opaque(statement, frontier)

    def _resume(
        self, resumes: list[CfgNode], statement: Node, frontier: list[CfgNode],
    ) -> list[CfgNode]:
        """
        A `continue` inside a `trap` body, which is not a back-jump: it resumes the guarded block at
        the statement after the one that threw. The node is recorded for `block` to link to every
        landing point once that block exists, and control does not fall through it here.

        Reading it as a loop back-jump instead resolves it to nothing and wires it to the body exit,
        which deletes the one path the trap exists to create — and leaves the whole resumption model
        unexercised, because this is the spelling that resumes.
        """
        node = self.node(statement)
        self.link(frontier, node)
        resumes.append(node)
        return []

    def _if(self, statement: Ps1IfStatement, frontier: list[CfgNode]) -> list[CfgNode]:
        """
        The chain, keyed so that the whole statement is the first test's node: a caller that locates
        the `if` itself must reach a node, and the first condition is where control first arrives.
        """
        clauses: list[tuple[Node, Node | None]] = []
        for index, (condition, block) in enumerate(statement.clauses):
            clauses.append((statement if index == 0 else condition, block))
        if not clauses:
            return self.opaque(statement, frontier)
        return self.branch_chain(clauses, statement.else_block, frontier)

    def _switch(self, statement: Ps1SwitchStatement, frontier: list[CfgNode]) -> list[CfgNode]:
        """
        The clauses as arms of an iterated, cumulative dispatch.

        A `default` clause does not make the construct exhaustive the way it does in a language
        whose `switch` tests one value: PowerShell's enumerates its input, and over an empty
        collection no clause runs at all — not even `default`. Reporting it exhaustive drops the
        edge that leaves the switch without entering any arm, which is what makes the statement
        after it look unreachable.

        An arm is handed over as the block it is rather than as the statements in it, because a
        clause body is a statement block of its own and a `trap` written in one guards that clause
        and nothing else.
        """
        arms: list[Sequence[Node]] = [
            [block] if block is not None else [] for _, block in statement.clauses
        ]
        self.park_label(_construct_label(statement.label))
        return self.dispatch(
            statement,
            arms,
            frontier,
            arm_flow=ArmFlow.CUMULATIVE,
            exhaustive=False,
            iterated=True,
        )

    def _try(self, statement: Ps1TryCatchFinally, frontier: list[CfgNode]) -> list[CfgNode]:
        """
        PowerShell allows several typed `catch` clauses where JavaScript allows one binding, so the
        clauses are chained: the guarded block reaches the first, and each clause reaches the next,
        because which one runs depends on the exception type and none of them is guaranteed.

        A clause carrying no type takes every error, so a set holding one leaves nothing to pass
        outward — which is what makes an empty `catch` shield the `catch` or `trap` around it where
        a typed one hands the error straight on to it.

        The `finally` body travels as its block for the same reason a `switch` arm does: it is a
        statement block, and a `trap` written in one guards it alone.
        """
        clauses = statement.catch_clauses
        finalizer = statement.finally_block
        return self.guarded(
            statement.try_block,
            [(clause, clause.body) for clause in clauses],
            finalizer,
            (finalizer,) if finalizer is not None else (),
            frontier,
            escapes=all(clause.types for clause in clauses),
        )


def build_ps1_control_flow(root: Ps1Script) -> dict[int, ControlFlowGraph]:
    """
    One control-flow graph per script block — see `FUNCTION_NODES` — and one for the script itself.
    """
    return build_control_flow(root, _Builder, FUNCTION_NODES)


def build_control_flow_model(root: Ps1Script) -> ControlFlowModel:
    """
    The `refinery.lib.scripts.analysis.cfg.ControlFlowModel` for a script root.
    """
    return ControlFlowModel(build_ps1_control_flow(root))
