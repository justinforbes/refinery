"""
Eliminate dead code branches guarded by constant conditions.

This transformer prunes unreachable branches from ``if``/``else`` statements when the test is a
literal whose truthiness can be determined statically.
"""
from __future__ import annotations

from refinery.lib.scripts import Node, Statement, Transformer
from refinery.lib.scripts.js.deobfuscation.helpers import is_statically_evaluable, is_truthy
from refinery.lib.scripts.js.model import (
    JsBlockStatement,
    JsIfStatement,
    JsScript,
)


class JsDeadCodeElimination(Transformer):
    """
    Remove unreachable code guarded by constant conditions.
    """

    def visit_JsScript(self, node: JsScript):
        self.generic_visit(node)
        self._prune_body(node, node.body)
        return None

    def visit_JsBlockStatement(self, node: JsBlockStatement):
        self.generic_visit(node)
        self._prune_body(node, node.body)
        return None

    def _prune_body(self, parent: Node, body: list[Statement]):
        result: list[Statement] = []
        changed = False
        for stmt in body:
            replacement = self._try_prune(stmt)
            if replacement is not None:
                result.extend(replacement)
                changed = True
            else:
                result.append(stmt)
        if changed:
            body.clear()
            body.extend(result)
            for stmt in result:
                stmt.parent = parent
            self.mark_changed()

    @staticmethod
    def _try_prune(stmt: Statement) -> list[Statement] | None:
        if not isinstance(stmt, JsIfStatement):
            return None
        if stmt.test is None or not is_statically_evaluable(stmt.test):
            return None
        truthy = is_truthy(stmt.test)
        if truthy is None:
            return None
        if truthy:
            return JsDeadCodeElimination._unwrap_branch(stmt.consequent)
        return JsDeadCodeElimination._unwrap_branch(stmt.alternate)

    @staticmethod
    def _unwrap_branch(branch: Statement | None) -> list[Statement]:
        """
        Extract the statements from a branch. If the branch is a block, return its body list
        contents; if it is a bare statement, wrap it in a single-element list.
        """
        if branch is None:
            return []
        if isinstance(branch, JsBlockStatement):
            return list(branch.body)
        return [branch]
