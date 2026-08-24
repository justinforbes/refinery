"""
The obfuscator converts statement sequences into calls to a self-disabling no-op function whose
arguments carry all side effects. This transformer detects the pattern structurally, expands each
call back into individual statements, and removes the wrapper definition.
"""
from __future__ import annotations

from refinery.lib.scripts import Expression, Node, _remove_from_parent, _replace_in_parent
from refinery.lib.scripts.js.analysis.cache import model_cache
from refinery.lib.scripts.js.deobfuscation.helpers import (
    ScriptLevelTransformer,
    binding_has_references,
)
from refinery.lib.scripts.js.model import (
    JsAssignmentExpression,
    JsBlockStatement,
    JsCallExpression,
    JsExpressionStatement,
    JsFunctionDeclaration,
    JsFunctionExpression,
    JsIdentifier,
    JsNumericLiteral,
    JsScript,
    JsSequenceExpression,
    JsSpreadElement,
    JsSwitchCase,
    JsUnaryExpression,
    wraps_return,
)


def _is_expression_wrapper(node: JsFunctionDeclaration) -> bool:
    """
    Test whether a function declaration matches the self-disabling wrapper pattern:

        function NAME() {
            NAME = function() {};
        }
    """
    if node.id is None or node.body is None:
        return False
    if node.params:
        return False
    if not isinstance(node.body, JsBlockStatement):
        return False
    body = node.body.body
    if len(body) != 1:
        return False
    stmt = body[0]
    if not isinstance(stmt, JsExpressionStatement):
        return False
    expr = stmt.expression
    if not isinstance(expr, JsAssignmentExpression):
        return False
    if expr.operator != '=':
        return False
    if not isinstance(expr.left, JsIdentifier):
        return False
    if expr.left.name != node.id.name:
        return False
    rhs = expr.right
    if not isinstance(rhs, JsFunctionExpression):
        return False
    if rhs.params:
        return False
    if isinstance(rhs.body, JsBlockStatement) and rhs.body.body:
        return False
    return True


def _find_expression_wrappers(root: Node) -> dict[str, JsFunctionDeclaration]:
    """
    The self-disabling wrappers in *root*, by the name each one answers to. The declaration is kept
    rather than the name alone because what a call to it answers decides which of the two expansions
    is available, and that is a property of the function rather than of the call.
    """
    wrappers: dict[str, JsFunctionDeclaration] = {}
    for node in root.walk():
        if isinstance(node, JsFunctionDeclaration) and _is_expression_wrapper(node):
            assert node.id is not None
            wrappers[node.id.name] = node
    return wrappers


class JsAssignmentsAsFunctionArgs(ScriptLevelTransformer):
    """
    Detect self-disabling wrapper functions and expand their call sites: a call in statement position
    becomes the individual argument statements, and a call embedded in a larger expression becomes the
    equivalent comma sequence in place, so evaluation order is preserved.

    Only the statement-position expansion is available for a wrapper whose call answers a promise or
    a generator object. In statement position what the call answered is discarded, so every kind of
    wrapper expands alike; the sequence answers `undefined`, which is what the call answers only when
    the wrapper is a plain function.
    """

    @staticmethod
    def _sequence_lowering(arguments: list[Expression]) -> Expression:
        """
        The value a self-disabling wrapper call `W(a, b)` computes — its arguments evaluated left to
        right, then `undefined` — expressed in place so nothing is reordered: the comma sequence
        `(a, b, void 0)`, or a bare `void 0` when there are no arguments.
        """
        void_0 = JsUnaryExpression(operator='void', operand=JsNumericLiteral(value=0, raw='0'))
        if not arguments:
            return void_0
        return JsSequenceExpression(expressions=[*arguments, void_0])

    def _process_script(self, node: JsScript):
        wrappers = _find_expression_wrappers(node)
        if not wrappers:
            return
        unwrapped = False
        for ast_node in list(node.walk()):
            if not isinstance(ast_node, JsCallExpression):
                continue
            if not isinstance(ast_node.callee, JsIdentifier):
                continue
            if ast_node.callee.name not in wrappers:
                continue
            if any(isinstance(arg, JsSpreadElement) for arg in ast_node.arguments):
                continue
            if (
                isinstance(parent := ast_node.parent, JsExpressionStatement)
                and isinstance(pp := parent.parent, (JsBlockStatement, JsScript, JsSwitchCase))
            ):
                body = pp.body
                try:
                    idx = body.index(parent)
                except ValueError:
                    continue
                new_stmts = [
                    JsExpressionStatement(expression=arg) for arg in ast_node.arguments
                ]
                body[idx:idx + 1] = new_stmts
                for stmt in new_stmts:
                    stmt.parent = pp
                unwrapped = True
            elif not wraps_return(wrappers[ast_node.callee.name]):
                _replace_in_parent(ast_node, self._sequence_lowering(ast_node.arguments))
                unwrapped = True
        if not unwrapped:
            return
        self.mark_changed()
        model = model_cache(self, node).model
        for ast_node in list(node.walk()):
            if not isinstance(ast_node, JsFunctionDeclaration):
                continue
            if ast_node.id is None:
                continue
            if ast_node.id.name not in wrappers:
                continue
            binding = model.binding_of(ast_node.id)
            if not binding_has_references(model, binding, exclude=ast_node):
                _remove_from_parent(ast_node)
