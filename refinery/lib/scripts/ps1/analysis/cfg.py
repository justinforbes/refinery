"""
PowerShell's contribution to the shared control-flow substrate: which node types are which
control-flow shape, and where the parts of each are stored.

Everything structural lives in `refinery.lib.scripts.analysis.cfg`. What is here is the dispatch, the
accessors, and the four places PowerShell's answer differs from the shape JavaScript established:

- **`switch` does not fall through, it keeps matching.** Every clause is tested against the value and
  every matching one runs, so an arm's exits may reach any *later* arm rather than only the next.
  That is `refinery.lib.scripts.analysis.cfg.ArmFlow.CUMULATIVE`, and reading it as C-style
  fallthrough would drop the path in which the first and third clauses match and the second does not.
- **`if`/`elseif` is one flat statement**, not a nest, so the chain of tests is built through
  `branch_chain` and each `elseif` condition gets a node of its own.
- **`trap` is a scope-wide handler**, not a guarded block: it catches for the whole body it is
  declared in, including statements written above it, and `continue` inside it resumes at the
  statement after the one that threw. See `_Builder.build`.
- **`exit` leaves the script, not the function.** It is a terminator like `return`, but nothing after
  it in *any* enclosing body runs, which no JavaScript statement does.

`Ps1Code` also splits one body across `begin`/`process`/`end`/`dynamicparam` blocks, so `owner.body`
alone is not what an advanced function runs.
"""
from __future__ import annotations

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
from refinery.lib.scripts.ps1.ast import get_body, get_named_blocks, string_value
from refinery.lib.scripts.ps1.model import (
    Ps1BreakStatement,
    Ps1Code,
    Ps1ContinueStatement,
    Ps1DoLoop,
    Ps1ExitStatement,
    Ps1ForEachLoop,
    Ps1ForLoop,
    Ps1FunctionDefinition,
    Ps1IfStatement,
    Ps1ReturnStatement,
    Ps1Script,
    Ps1SwitchStatement,
    Ps1ThrowStatement,
    Ps1TrapStatement,
    Ps1TryCatchFinally,
    Ps1WhileLoop,
)

#: The nodes that own a control-flow graph of their own. A class or enum definition owns none: its
#: methods are `Ps1FunctionDefinition` bodies in their own right and are found by the same walk.
FUNCTION_NODES = (Ps1FunctionDefinition,)

_LOOP_NODES = (
    Ps1WhileLoop,
    Ps1DoLoop,
    Ps1ForLoop,
    Ps1ForEachLoop,
)


def _jump_label(statement: Node) -> str | None:
    """
    The label a `break` or `continue` names, or `None` when it names none or names one this cannot
    read. PowerShell allows the label to be any expression — `break $name` is legal — and a label
    that is not a static string is one no lexical target can be matched against, so it is treated as
    unlabelled, which resolves to the innermost enclosing target and is the conservative reading.
    """
    label = getattr(statement, 'label', None)
    return string_value(label) if label is not None else None


class _Builder(CfgBuilder):
    """
    The PowerShell dispatch over `refinery.lib.scripts.analysis.cfg.CfgBuilder`.
    """

    def build(self) -> ControlFlowGraph:
        """
        The body, with any `trap` it declares installed as a handler over the whole of it.

        A `trap` is not a guarded block. It is declared somewhere in a body and catches for that
        body entire, including the statements written above it, so it cannot be pushed at the point
        it appears the way a `try` is — it is pushed before the body is walked at all.

        `continue` inside a trap resumes at the statement following the one that threw, which is a
        shape this graph cannot express exactly. The handler's exits are therefore linked to *every*
        node the body created, which is the over-approximation: it claims more paths than exist,
        where claiming fewer would let an analysis call a statement after a trap unreachable, or let
        a store before one look dead because the resumption that reads it was never modelled.
        """
        statements = self.body_statements(self.cfg.owner)
        traps = [node for node in statements if isinstance(node, Ps1TrapStatement)]
        resumes: list[CfgNode] = []
        for trap in traps:
            handler = self.node(trap)
            self._handlers.append(handler)
            resumes += self._body(trap.body, [handler])
        frontier = self.sequence(statements, [self.cfg.entry])
        for _ in traps:
            self._handlers.pop()
        self.link(frontier, self.cfg.exit)
        if resumes:
            landing = [
                node for node in self.cfg.nodes
                if node is not self.cfg.entry and node.element is not None
            ]
            for resume in resumes:
                self.link([resume], self.cfg.exit)
                for node in landing:
                    self.add_edge(resume, node)
        return self.cfg

    def body_statements(self, owner: Node) -> list[Node]:
        """
        The statements *owner* runs. An advanced function fills `begin`/`process`/`end` instead of
        `body`, and reading only `body` for one would report an empty graph for a function that runs
        a great deal — the same trap `refinery.lib.scripts.ps1.ast.get_named_blocks` exists to warn
        about.
        """
        code = owner if isinstance(owner, Ps1Code) else getattr(owner, 'body', None)
        if isinstance(code, Ps1Code):
            statements: list[Node] = []
            for block in get_named_blocks(code):
                statements.extend(block.body)
            statements.extend(code.body)
            return statements
        body = get_body(owner)
        if body is not None:
            return list(body)
        return [code] if isinstance(code, Node) else []

    def statement(self, statement: Node, frontier: list[CfgNode]) -> list[CfgNode]:
        if isinstance(statement, Block):
            return self.sequence(statement.body, frontier)
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
            return self.jump_out(statement, _jump_label(statement), frontier)
        if isinstance(statement, Ps1ContinueStatement):
            return self.jump_back(statement, _jump_label(statement), frontier)
        return self.opaque(statement, frontier)

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
        arms: list[Sequence[Node]] = []
        exhaustive = False
        for condition, block in statement.clauses:
            arms.append(list(block.body) if block is not None else [])
            if condition is None:
                exhaustive = True
        self._pending_label = statement.label
        return self.dispatch(
            statement, arms, frontier, arm_flow=ArmFlow.CUMULATIVE, exhaustive=exhaustive)

    def _try(self, statement: Ps1TryCatchFinally, frontier: list[CfgNode]) -> list[CfgNode]:
        """
        PowerShell allows several typed `catch` clauses where JavaScript allows one binding, so the
        clauses are chained: the guarded block reaches the first, and each clause reaches the next,
        because which one runs depends on the exception type and none of them is guaranteed.
        """
        clauses = statement.catch_clauses
        return self.guarded(
            statement.try_block,
            [(clause, clause.body) for clause in clauses],
            statement.finally_block,
            list(statement.finally_block.body) if statement.finally_block is not None else (),
            frontier,
        )


def build_ps1_control_flow(root: Ps1Script) -> dict[int, ControlFlowGraph]:
    """
    One control-flow graph per function definition and one for the script itself.
    """
    return build_control_flow(root, _Builder, FUNCTION_NODES)


def build_control_flow_model(root: Ps1Script) -> ControlFlowModel:
    """
    The `refinery.lib.scripts.analysis.cfg.ControlFlowModel` for a script root.
    """
    return ControlFlowModel(build_ps1_control_flow(root))
