"""
Remove the obfuscator.io self-defending anti-tamper pattern.

The obfuscator emits a run-once wrapper factory and one or more guard sites. Each guard hands the
global object and a payload function to the factory and invokes the result:

    FACTORY = (function () {
        var flag = true;
        return function (recv, payload) {
            var run = flag
                ? function () { if (payload) { var x = payload.apply(recv, arguments); return payload = null, x; } }
                : function () {};
            return flag = false, run;
        };
    }());
    var guard = FACTORY(this, function () { /* anti-analysis payload */ });
    guard();

This transformer detects the pattern in two independent ways and removes the factory and each guard
invocation:

- By the ReDoS signature string `(((.+)+)+)+$` carried by the payload.
- Structurally, by the run-once `apply`-payload factory template, which covers payloads that do not
  carry the ReDoS string (e.g. a console-hijack or a RegExp self-test).

Both detectors feed the same shared removal executor.
"""
from __future__ import annotations

from refinery.lib.scripts import _remove_from_parent
from refinery.lib.scripts.js.analysis.cache import model_cache
from refinery.lib.scripts.js.deobfuscation.helpers import (
    ScriptLevelTransformer,
    binding_has_references,
    remove_declarator,
)
from refinery.lib.scripts.js.model import (
    JsAssignmentExpression,
    JsBlockStatement,
    JsCallExpression,
    JsConditionalExpression,
    JsExpressionStatement,
    JsFunctionExpression,
    JsIdentifier,
    JsMemberExpression,
    JsNullLiteral,
    JsParenthesizedExpression,
    JsScript,
    JsSequenceExpression,
    JsStringLiteral,
    JsThisExpression,
    JsVariableDeclaration,
    JsVariableDeclarator,
    FUNCTION_NODES,
    strip_parens,
)

_REDOS_SIGNATURE = '(((.+)+)+)+$'


def _is_global_receiver(node) -> bool:
    node = strip_parens(node)
    if isinstance(node, JsThisExpression):
        return True
    return isinstance(node, JsIdentifier) and node.name in ('globalThis', 'window', 'self', 'global')


def _matches_self_defending_factory(model, fn) -> bool:
    """
    True when `fn` matches the run-once `apply`-payload factory template: exactly two plain-identifier
    params `(recv, payload)`; a conditional whose alternate is an empty function expression; a call of
    the form `payload.apply(recv, ...)` or `payload.call(recv, ...)` resolved through the param bindings.
    """
    if not isinstance(fn, FUNCTION_NODES) or len(fn.params) != 2:
        return False
    if not all(isinstance(p, JsIdentifier) for p in fn.params):
        return False
    recv_b = model.binding_of(fn.params[0])
    payload_b = model.binding_of(fn.params[1])
    if recv_b is None or payload_b is None:
        return False
    empty_alt = False
    apply_shape = False
    payload_nulled = False
    for n in fn.walk():
        if isinstance(n, JsConditionalExpression):
            alt = strip_parens(n.alternate)
            if isinstance(alt, JsFunctionExpression) and alt.body is not None and not alt.body.body:
                empty_alt = True
        if isinstance(n, JsCallExpression):
            callee = strip_parens(n.callee)
            if not isinstance(callee, JsMemberExpression):
                continue
            prop = callee.property
            prop_name = getattr(prop, 'name', None) or getattr(prop, 'value', None)
            if prop_name not in ('apply', 'call'):
                continue
            base = strip_parens(callee.object)
            arg0 = strip_parens(n.arguments[0]) if n.arguments else None
            if (
                isinstance(base, JsIdentifier)
                and model.resolve(base) is payload_b
                and isinstance(arg0, JsIdentifier)
                and model.resolve(arg0) is recv_b
            ):
                apply_shape = True
        if isinstance(n, JsAssignmentExpression) and n.operator == '=':
            lhs = strip_parens(n.left)
            rhs = strip_parens(n.right)
            if (
                isinstance(lhs, JsIdentifier)
                and model.resolve(lhs) is payload_b
                and isinstance(rhs, JsNullLiteral)
            ):
                payload_nulled = True
    return empty_alt and apply_shape and payload_nulled


def _removal_unit(call: JsCallExpression) -> JsCallExpression:
    """
    Return the IIFE call that wraps `call` when `call` is the sole statement of an immediately-invoked
    function body, otherwise return `call` itself.
    """
    es = call.parent
    if not isinstance(es, JsExpressionStatement):
        return call
    block = es.parent
    if not isinstance(block, JsBlockStatement) or len(block.body) != 1:
        return call
    fn = block.parent
    if not isinstance(fn, FUNCTION_NODES):
        return call
    outer = fn.parent
    while isinstance(outer, JsParenthesizedExpression):
        outer = outer.parent
    if isinstance(outer, JsCallExpression) and strip_parens(outer.callee) is fn:
        return outer
    return call


def _remove_expr(node) -> None:
    """
    Remove `node` as an expression: strips to the innermost non-paren ancestor, then removes the
    sequence operand, the enclosing expression statement, or the node itself.
    """
    cur = node
    p = cur.parent
    while isinstance(p, JsParenthesizedExpression):
        cur, p = p, p.parent
    if isinstance(p, JsSequenceExpression):
        _remove_from_parent(cur)
    elif isinstance(p, JsExpressionStatement):
        _remove_from_parent(p)
    else:
        _remove_from_parent(cur)


class JsRemoveSelfDefending(ScriptLevelTransformer):
    """
    Detect and remove the obfuscator.io self-defending factory+guard pattern, keyed both by the ReDoS
    signature string and by the structural run-once `apply`-payload template.
    """

    def _process_script(self, node: JsScript):
        for literal in list(node.walk()):
            if isinstance(literal, JsStringLiteral) and _REDOS_SIGNATURE in literal.value:
                self._remove_redos(literal, node)
        self._remove_structural(node)

    def _remove_redos(self, redos_literal: JsStringLiteral, root: JsScript) -> None:
        guard_decl = redos_literal.parent
        while guard_decl is not None and not isinstance(guard_decl, JsVariableDeclarator):
            guard_decl = guard_decl.parent
        if guard_decl is None or not isinstance(guard_decl.id, JsIdentifier):
            return
        if not isinstance(guard_decl.init, JsCallExpression):
            return
        callee = guard_decl.init.callee
        if isinstance(callee, JsIdentifier):
            factory_name = callee.name
        elif isinstance(callee, JsFunctionExpression):
            factory_name = None
        else:
            return
        guard_name = guard_decl.id.name
        co_names: set[str] = set()
        if factory_name is None:
            for arg in guard_decl.init.arguments:
                if isinstance(arg, JsIdentifier):
                    co_names.add(arg.name)
        var_decl = guard_decl.parent
        if not isinstance(var_decl, JsVariableDeclaration):
            return
        body_parent = var_decl.parent
        if isinstance(body_parent, JsScript):
            body = body_parent.body
        elif isinstance(body_parent, JsBlockStatement):
            body = body_parent.body
        else:
            return
        for stmt in list(body):
            if (
                isinstance(stmt, JsExpressionStatement)
                and isinstance(stmt.expression, JsCallExpression)
                and isinstance(stmt.expression.callee, JsIdentifier)
                and stmt.expression.callee.name == guard_name
                and not stmt.expression.arguments
            ):
                _remove_from_parent(stmt)
        remove_declarator(guard_decl)
        cleanup_names = {factory_name} if factory_name is not None else co_names
        for name in cleanup_names:
            model = model_cache(self, root).model
            for stmt in list(body):
                if not isinstance(stmt, JsVariableDeclaration):
                    continue
                for d in list(stmt.declarations):
                    if (
                        isinstance(d, JsVariableDeclarator)
                        and isinstance(d.id, JsIdentifier)
                        and d.id.name == name
                    ):
                        binding = model.binding_of(d.id)
                        if not binding_has_references(model, binding):
                            remove_declarator(d)
        self.mark_changed()

    def _remove_structural(self, root: JsScript) -> None:
        model = model_cache(self, root).model
        immediate_guards: list[JsCallExpression] = []
        stored_guard_bindings: list = []
        factory_names: set[str] = set()

        for node in list(root.walk()):
            if not isinstance(node, JsCallExpression) or len(node.arguments) < 2:
                continue
            if not _is_global_receiver(node.arguments[0]):
                continue
            fn = model._target_function_of_call(node)
            if fn is None or not _matches_self_defending_factory(model, fn):
                continue
            callee = strip_parens(node.callee)
            if isinstance(callee, JsIdentifier):
                factory_names.add(callee.name)
            parent = node.parent
            while isinstance(parent, JsParenthesizedExpression):
                parent = parent.parent
            if isinstance(parent, JsCallExpression) and strip_parens(parent.callee) is node:
                immediate_guards.append(parent)
            elif isinstance(parent, JsVariableDeclarator) and isinstance(parent.id, JsIdentifier):
                binding = model.binding_of(parent.id)
                if binding is None:
                    continue
                guard_called = any(
                    isinstance(ref.parent, JsCallExpression)
                    and strip_parens(ref.parent.callee) is ref
                    for ref in model.references(binding)
                )
                if guard_called:
                    stored_guard_bindings.append(binding)

        if not immediate_guards and not stored_guard_bindings:
            return

        for guard_call in immediate_guards:
            _remove_expr(_removal_unit(guard_call))

        for gb in stored_guard_bindings:
            for ref in list(model.references(gb)):
                call = ref.parent
                while isinstance(call, JsParenthesizedExpression):
                    call = call.parent
                if isinstance(call, JsCallExpression) and strip_parens(call.callee) is ref:
                    _remove_expr(call)
            for decl_site in list(gb.declarations):
                d = decl_site.parent
                if isinstance(d, JsVariableDeclarator):
                    remove_declarator(d)

        model = model_cache(self, root).model
        for binding in list(model.root_scope.bindings.values()):
            if binding.name not in factory_names:
                continue
            if not binding_has_references(model, binding):
                for decl_site in list(binding.declarations):
                    d = decl_site.parent
                    if isinstance(d, JsVariableDeclarator):
                        remove_declarator(d)

        self.mark_changed()
