"""
Hoist void subexpressions out of expandable strings, replacing the expandable
string with a plain string literal of its text parts. The hoisted statements
are inserted around the parent statement preserving their side effects. Only
operates on expandable strings where ALL subexpressions are void-producing
(assignments only — a command inside `$( ... )` contributes its output to the
string, so it is never hoisted).

Safety constraint: subexpressions from leftmost expandable strings are
inserted BEFORE the parent statement (they were going to run first anyway).
Subexpressions from other expandable strings are inserted AFTER the parent
statement to preserve execution order.
"""
from __future__ import annotations

from refinery.lib.scripts import Block, BodyEdit, Transformer, _replace_in_parent
from refinery.lib.scripts.ps1.ast import get_body
from refinery.lib.scripts.ps1.deobfuscation.helpers import make_string_literal
from refinery.lib.scripts.ps1.model import (
    Ps1AssignmentExpression,
    Ps1BinaryExpression,
    Ps1Code,
    Ps1ExpandableString,
    Ps1ExpressionStatement,
    Ps1StringLiteral,
    Ps1SubExpression,
)


class Ps1ExpandableStringHoist(Transformer):
    """
    Extract void subexpressions from expandable strings into preceding or following statements,
    then replace the expandable string with a plain string literal.
    """

    def visit(self, node):
        for container in list(node.walk()):
            body = get_body(container)
            if body is None:
                continue
            i = 0
            while i < len(body):
                before, after, replaced = self._extract_void_subexpressions(body[i])
                if before or after:
                    edit = BodyEdit(container)
                    edit.splice(body[i], [*before, body[i], *after])
                    edit.apply()
                    i += len(before) + len(after)
                if replaced:
                    self.mark_changed()
                i += 1
        return None

    @staticmethod
    def _is_void_statement(stmt) -> bool:
        """
        A statement is void when it contributes nothing to an interpolating string. Only assignments
        qualify: a command invocation inside `$( ... )` writes its output into the surrounding
        string, so hoisting it out would drop that interpolated value and leak the command's output
        to the pipeline.
        """
        if not isinstance(stmt, Ps1ExpressionStatement):
            return False
        return isinstance(stmt.expression, Ps1AssignmentExpression)

    @staticmethod
    def _is_leftmost(node) -> bool:
        """
        Check whether *node* sits in the leftmost evaluation position of its enclosing expression
        tree. An expandable string is leftmost when every ancestor
        `refinery.lib.scripts.ps1.model.Ps1BinaryExpression` has it (or the subtree containing it)
        as its `left` operand. This guarantees the subexpressions would have been the first thing
        evaluated, so hoisting them before the statement does not change execution order.
        """
        child = node
        parent = node.parent
        while parent is not None:
            if isinstance(parent, Ps1BinaryExpression):
                if parent.left is not child:
                    return False
            if isinstance(parent, (Ps1ExpressionStatement, Ps1Code, Block)):
                break
            child = parent
            parent = parent.parent
        return True

    def _extract_void_subexpressions(self, stmt) -> tuple[list, list, bool]:
        """
        Walk the statement tree, find expandable strings whose every part is either literal text or
        a void subexpression, replace them with string literals, and return
        `(before_stmts, after_stmts, replaced)`.

        Every part has to be accounted for and not only the subexpressions. A part that is neither
        — the `$env:APPDATA` in `"$env:APPDATA$($z=1)\\dropper.exe"` — has no text this can write
        down, so rewriting the string around it would silently delete what it interpolated and leave
        a literal the script never said.

        `replaced` is reported separately from the two lists because a subexpression with an empty
        body hoists no statement and still rewrites the string, so the lists alone cannot say
        whether the tree moved.

        The hoisted statements are collected only once the string has actually been replaced. The
        walk is a snapshot, so a string nested inside one already rewritten is no longer reachable
        from its parent, and hoisting its subexpressions out while it stands would run them twice.
        """
        before: list = []
        after: list = []
        replaced = False
        for node in list(stmt.walk_in_order()):
            if not isinstance(node, Ps1ExpandableString):
                continue
            if not all(
                isinstance(part, (Ps1StringLiteral, Ps1SubExpression))
                for part in node.parts
            ):
                continue
            subs = [p for p in node.parts if isinstance(p, Ps1SubExpression)]
            if not subs:
                continue
            if not all(
                all(self._is_void_statement(s) for s in sub.body)
                for sub in subs
            ):
                continue
            text_parts: list[str] = []
            for part in node.parts:
                if isinstance(part, Ps1StringLiteral):
                    text_parts.append(part.value)
            collected: list = []
            for sub in subs:
                collected.extend(sub.body)
            leftmost = self._is_leftmost(node)
            if not _replace_in_parent(node, make_string_literal(''.join(text_parts))):
                continue
            if leftmost:
                before.extend(collected)
            else:
                after.extend(collected)
            replaced = True
        return before, after, replaced
