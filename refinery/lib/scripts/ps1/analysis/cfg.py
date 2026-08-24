"""
PowerShell's dispatch over the shared control-flow builder: which node types are which shape, and
where the parts of each are stored. Everything structural lives in
`refinery.lib.scripts.analysis.cfg`.
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
    Ps1CatchClause,
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

#: The nodes owning a control-flow graph of their own, beside the script itself. A script block is
#: PowerShell's anonymous function, and the only one a `Ps1FunctionDefinition` holds.
FUNCTION_NODES = (Ps1ScriptBlock,)

_LOOP_NODES = (
    Ps1WhileLoop,
    Ps1DoLoop,
    Ps1ForLoop,
    Ps1ForEachLoop,
)


#: The type names a `catch` clause may carry that take every terminating error 5.1 raises, in the
#: spellings 5.1 resolves. A bare name resolves against `System` alone, so `[Exception]` names
#: `System.Exception` while `[RuntimeException]` names nothing and its clause matches no error at
#: all — which is why the short form of the second is absent here.
_CATCHES_EVERY_ERROR = frozenset({
    'exception',
    'system.exception',
    'management.automation.runtimeexception',
    'system.management.automation.runtimeexception',
})


def _names_every_error(name: str) -> bool:
    return not name.strip() or name.strip().lower() in _CATCHES_EVERY_ERROR


def _catch_takes_every_error(clause: Ps1CatchClause) -> bool:
    if not clause.types:
        return True
    return any(_names_every_error(name) for name in clause.types)


def _jump_label(statement: Ps1Jump) -> str | None:
    return string_value(statement.label) if statement.label is not None else None


def _construct_label(label: str | None) -> str | None:
    return label[1:] if label is not None and label.startswith(':') else label


def _declared_traps(statements: Sequence[Node]) -> list[Ps1TrapStatement]:
    return [statement for statement in statements if isinstance(statement, Ps1TrapStatement)]


@dataclass
class _TrapBody:
    resumes: list[CfgNode] = field(default_factory=list)
    rethrows: bool = False


class _Builder(CfgBuilder):
    def __init__(self, owner: Node):
        super().__init__(owner)
        self._trap_body: _TrapBody | None = None

    def block(self, statements: Sequence[Node], frontier: list[CfgNode]) -> list[CfgNode]:
        """
        One statement block, with every `trap` it declares installed as a handler over the whole of
        it: a `trap` catches for the block it is written in entire, statements above it included, so
        the set cannot be pushed at the point it appears the way a `try` is.
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
        handling = list(entries)
        resumes: list[CfgNode] = []
        escapes = True
        enclosing = self._trap_body
        outer_targets = self._targets
        outer_label = self._pending_label
        self._targets = []
        self._pending_label = None
        self._handlers.append(self.unwinding())
        for trap, handler in zip(traps, entries):
            self._trap_body = built = _TrapBody()
            body_from = len(self.cfg.nodes)
            resumes += self._body(trap.body, [handler]) + built.resumes
            handling += self.cfg.nodes[body_from:]
            if _names_every_error(trap.type_name) and not built.rethrows:
                escapes = False
        self._handlers.pop()
        self._trap_body = enclosing
        self._targets = outer_targets
        self._pending_label = outer_label
        self.close_handler_set(entries, escapes=escapes)
        self._handlers.append(entries[0])
        if not resumes:
            exits = self.sequence(statements, frontier)
        else:
            self.mark_hub_bound(handling)
            with self.resumption(resumes):
                exits = self._resumable_sequence(statements, frontier)
        self._handlers.pop()
        return exits

    def _resumable_sequence(
        self, statements: Sequence[Node], frontier: list[CfgNode],
    ) -> list[CfgNode]:
        """
        The guarded statements of a block whose `trap` set resumes it, threaded as `sequence` does
        and joined to the forward-only half of resumption as each statement is built.
        """
        for statement in statements:
            frontier = self.resume_past(self.statement(statement, frontier))
        return frontier

    def body_blocks(self, owner: Node) -> Sequence[Sequence[Node]]:
        """
        The statement blocks *owner* runs, `dynamicparam` first: the engine evaluates it during
        parameter binding, before `begin`. They stay apart because each is a statement block of its
        own, and a `trap` written in `begin` is offered nothing `process` raises.
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
        node = self.node(statement)
        self.link(frontier, node)
        resumes.append(node)
        return []

    def _if(self, statement: Ps1IfStatement, frontier: list[CfgNode]) -> list[CfgNode]:
        clauses: list[tuple[Node, Node | None]] = []
        for index, (condition, block) in enumerate(statement.clauses):
            clauses.append((statement if index == 0 else condition, block))
        if not clauses:
            return self.opaque(statement, frontier)
        return self.branch_chain(clauses, statement.else_block, frontier)

    def _switch(self, statement: Ps1SwitchStatement, frontier: list[CfgNode]) -> list[CfgNode]:
        """
        The clauses as arms of an iterated, cumulative dispatch. A `default` clause does not make
        the construct exhaustive: PowerShell's `switch` enumerates its input, and over an empty
        collection no clause runs at all.
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
        clauses = statement.catch_clauses
        finalizer = statement.finally_block
        return self.guarded(
            statement.try_block,
            [(clause, clause.body) for clause in clauses],
            finalizer,
            (finalizer,) if finalizer is not None else (),
            frontier,
            escapes=not any(_catch_takes_every_error(clause) for clause in clauses),
        )


def build_ps1_control_flow(root: Ps1Script) -> dict[int, ControlFlowGraph]:
    return build_control_flow(root, _Builder, FUNCTION_NODES)


def build_control_flow_model(root: Ps1Script) -> ControlFlowModel:
    return ControlFlowModel(build_ps1_control_flow(root))
