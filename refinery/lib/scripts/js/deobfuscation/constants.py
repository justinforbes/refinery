"""
Inline constant variable references in JavaScript.
"""
from __future__ import annotations

from typing import NamedTuple

from refinery.lib.scripts import Node, _clone_node, _remove_from_parent, _replace_in_parent
from refinery.lib.scripts.js.deobfuscation.helpers import ScopeProcessingTransformer, is_literal
from refinery.lib.scripts.js.model import (
    JsArrowFunctionExpression,
    JsArrayExpression,
    JsAssignmentExpression,
    JsAwaitExpression,
    JsCallExpression,
    JsClassExpression,
    JsFunctionDeclaration,
    JsFunctionExpression,
    JsIdentifier,
    JsMemberExpression,
    JsNewExpression,
    JsObjectExpression,
    JsStringLiteral,
    JsTaggedTemplateExpression,
    JsUpdateExpression,
    JsVariableDeclaration,
    JsVariableDeclarator,
    JsYieldExpression,
)

_FUNCTION_NODES = (JsFunctionDeclaration, JsFunctionExpression, JsArrowFunctionExpression)


class _Candidate(NamedTuple):
    declarator: JsVariableDeclarator
    statement: JsVariableDeclaration
    init: Node


def _walk_scope(root: Node):
    """
    Walk the AST under *root* without descending into nested function bodies. The function boundary
    node itself is yielded (so its identifier can be inspected) but its children are not visited.
    """
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, _FUNCTION_NODES) and node is not root:
            continue
        for child in node.children():
            stack.append(child)


def _is_side_effect_free(node: Node) -> bool:
    """
    Return whether evaluating *node* is guaranteed to produce no observable side effects and
    the result is a primitive value (not an object, array, or function).
    """
    for n in node.walk():
        if isinstance(n, (
            JsCallExpression,
            JsNewExpression,
            JsAssignmentExpression,
            JsUpdateExpression,
            JsYieldExpression,
            JsAwaitExpression,
            JsTaggedTemplateExpression,
            JsMemberExpression,
            JsObjectExpression,
            JsArrayExpression,
            JsFunctionExpression,
            JsArrowFunctionExpression,
            JsClassExpression,
        )):
            return False
    return True


def _identifier_leaves(node: Node) -> set[str]:
    """
    Collect the names of all `JsIdentifier` leaves in an expression tree.
    """
    return {n.name for n in node.walk() if isinstance(n, JsIdentifier)}


class JsConstantInlining(ScopeProcessingTransformer):
    """
    Inline variables that are assigned once and never mutated. Literal-valued variables are inlined
    at all use sites; single-use variables with side-effect-free initializers are inlined when no
    intervening mutation could alter the referenced identifiers.
    """

    def __init__(self, max_inline_length: int = 64):
        super().__init__()
        self.max_inline_length = max_inline_length

    def _process_scope(self, scope: Node) -> None:
        while True:
            candidates, mutated = self._collect_candidates(scope)
            if not candidates:
                return
            ref_counts = self._count_references(scope, candidates)
            literals = self._classify_literals(candidates, ref_counts)
            if literals:
                inlined = self._substitute(scope, literals, candidates)
                self._remove_dead(literals, ref_counts, inlined)
                if inlined:
                    continue
            expressions = self._classify_expressions(candidates, ref_counts, mutated)
            if expressions:
                inlined = self._substitute(scope, expressions, candidates)
                self._remove_dead(expressions, ref_counts, inlined)
                if inlined:
                    continue
            return

    @staticmethod
    def _collect_candidates(
        scope: Node,
    ) -> tuple[dict[str, _Candidate], set[str]]:
        candidates: dict[str, _Candidate] = {}
        mutated: set[str] = set()
        for node in _walk_scope(scope):
            if isinstance(node, JsVariableDeclaration):
                for decl in node.declarations:
                    if not isinstance(decl, JsVariableDeclarator):
                        continue
                    if not isinstance(decl.id, JsIdentifier):
                        continue
                    if decl.init is None:
                        continue
                    name = decl.id.name
                    if name in candidates:
                        mutated.add(name)
                    else:
                        candidates[name] = _Candidate(decl, node, decl.init)
            if isinstance(node, JsAssignmentExpression) and isinstance(node.left, JsIdentifier):
                mutated.add(node.left.name)
            if isinstance(node, JsUpdateExpression) and isinstance(node.argument, JsIdentifier):
                mutated.add(node.argument.name)
        for name in mutated:
            candidates.pop(name, None)
        return candidates, mutated

    @staticmethod
    def _count_references(
        scope: Node,
        candidates: dict[str, _Candidate],
    ) -> dict[str, int]:
        decl_ids = {id(c.declarator.id) for c in candidates.values()}
        counts: dict[str, int] = {name: 0 for name in candidates}
        for node in _walk_scope(scope):
            if not isinstance(node, JsIdentifier):
                continue
            if id(node) in decl_ids:
                continue
            if node.name not in counts:
                continue
            parent = node.parent
            if isinstance(parent, JsAssignmentExpression) and parent.left is node:
                continue
            counts[node.name] += 1
        return counts

    def _classify_literals(
        self,
        candidates: dict[str, _Candidate],
        ref_counts: dict[str, int],
    ) -> dict[str, _Candidate]:
        result: dict[str, _Candidate] = {}
        for name, candidate in candidates.items():
            count = ref_counts[name]
            init = candidate.init
            if count == 0:
                continue
            if not is_literal(init):
                continue
            if (
                count > 1
                and isinstance(init, JsStringLiteral)
                and len(init.value) > self.max_inline_length
            ):
                continue
            result[name] = candidate
        return result

    @staticmethod
    def _classify_expressions(
        candidates: dict[str, _Candidate],
        ref_counts: dict[str, int],
        mutated: set[str],
    ) -> dict[str, _Candidate]:
        result: dict[str, _Candidate] = {}
        for name, candidate in candidates.items():
            count = ref_counts[name]
            init = candidate.init
            if is_literal(init) or count != 1:
                continue
            if not _is_side_effect_free(init):
                continue
            if _identifier_leaves(init) & mutated:
                continue
            result[name] = candidate
        return result

    def _substitute(
        self,
        scope: Node,
        to_inline: dict[str, _Candidate],
        candidates: dict[str, _Candidate],
    ) -> dict[str, int]:
        decl_ids = {id(c.declarator.id) for c in candidates.values()}
        inlined: dict[str, int] = {}
        for node in list(_walk_scope(scope)):
            if not isinstance(node, JsIdentifier):
                continue
            if id(node) in decl_ids:
                continue
            name = node.name
            if name not in to_inline:
                continue
            parent = node.parent
            if isinstance(parent, JsAssignmentExpression) and parent.left is node:
                continue
            _replace_in_parent(node, _clone_node(to_inline[name].init))
            self.mark_changed()
            inlined[name] = inlined.get(name, 0) + 1
        return inlined

    def _remove_dead(
        self,
        to_inline: dict[str, _Candidate],
        ref_counts: dict[str, int],
        inlined: dict[str, int],
    ) -> None:
        for name, candidate in to_inline.items():
            remaining = ref_counts[name] - inlined.get(name, 0)
            if remaining > 0:
                continue
            stmt = candidate.statement
            if len(stmt.declarations) == 1:
                _remove_from_parent(stmt)
            else:
                decl = candidate.declarator
                try:
                    stmt.declarations.remove(decl)
                except ValueError:
                    continue
            self.mark_changed()
