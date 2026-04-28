"""
Inline properties of locally-defined constant object literals.

When the obfuscator lifts string literals and operator wrappers into a local object, this
transformer detects the pattern and replaces all member-access reads with the inlined property
values. Function-valued properties that are trivial wrappers (single return statement whose body is
an expression using only parameters) are inlined at the call site.
"""
from __future__ import annotations

from refinery.lib.scripts import (
    Node,
    _clone_node,
    _replace_in_parent,
)
from refinery.lib.scripts.js.deobfuscation.helpers import (
    ScopeProcessingTransformer,
    extract_identifier_params,
    is_closed_expression,
    remove_declarator,
    string_value,
    substitute_params,
)
from refinery.lib.scripts.js.model import (
    JsBlockStatement,
    JsCallExpression,
    JsFunctionExpression,
    JsIdentifier,
    JsMemberExpression,
    JsObjectExpression,
    JsProperty,
    JsReturnStatement,
    JsScript,
    JsStringLiteral,
    JsVariableDeclaration,
    JsVariableDeclarator,
)


def _property_key(prop: JsProperty) -> str | None:
    """
    Extract the string key from a property node. Handles both string-literal keys
    (`{'key': ...}`) and plain identifier keys (`{key: ...}`).
    """
    if prop.computed:
        return None
    if isinstance(prop.key, JsStringLiteral):
        return prop.key.value
    if isinstance(prop.key, JsIdentifier):
        return prop.key.name
    return None


def _build_property_map(
    obj: JsObjectExpression,
) -> dict[str, Node] | None:
    """
    Build a map from string key to value node for every property in the object literal.
    Returns `None` if any property cannot be statically keyed (computed key, spread, etc.).
    """
    result: dict[str, Node] = {}
    for prop in obj.properties:
        if not isinstance(prop, JsProperty):
            return None
        key = _property_key(prop)
        if key is None or prop.value is None:
            return None
        result[key] = prop.value
    return result


def _access_key(node: JsMemberExpression) -> str | None:
    """
    Extract the string key from a member-access expression. Handles both computed
    (`obj['key']`) and dot (`obj.key`) accesses.
    """
    if node.computed:
        return string_value(node.property)
    if isinstance(node.property, JsIdentifier):
        return node.property.name
    return None


def _try_inline_call(
    func: JsFunctionExpression,
    call_args: list,
) -> Node | None:
    """
    If *func* is a trivial wrapper (single return whose expression uses only the function's
    parameters), substitute call-site arguments into a clone of the return expression. Returns the
    inlined expression or `None` if the function is not a simple wrapper.
    """
    if func.body is None or not isinstance(func.body, JsBlockStatement):
        return None
    body = func.body.body
    if len(body) != 1:
        return None
    stmt = body[0]
    if not isinstance(stmt, JsReturnStatement) or stmt.argument is None:
        return None
    param_names = extract_identifier_params(func.params)
    if param_names is None:
        return None
    if len(call_args) != len(param_names):
        return None
    expr = stmt.argument
    if not is_closed_expression(expr, set(param_names)):
        return None
    return substitute_params(expr, param_names, call_args)


class JsObjectFold(ScopeProcessingTransformer):
    """
    Inline properties of locally-defined constant objects. Processes at function-scope and
    script-scope boundaries because JavaScript `var` declarations are function-scoped.
    """

    def _process_scope(self, scope: Node) -> None:
        if isinstance(scope, JsScript):
            body = scope.body
        elif isinstance(scope, JsBlockStatement):
            body = scope.body
        else:
            return
        for candidate in list(self._find_candidates(body)):
            obj_name, declarator, prop_map = candidate
            if not self._is_safe_to_fold(scope, obj_name, declarator):
                continue
            if self._inline_references(scope, obj_name, prop_map):
                remove_declarator(declarator)
                self.mark_changed()

    @staticmethod
    def _find_candidates(body: list):
        """
        Yield tuples of (name, declarator_node, property_map) for each variable declarator in
        *body* that initializes a variable to an object literal with all statically-keyed
        properties.
        """
        for stmt in body:
            if not isinstance(stmt, JsVariableDeclaration):
                continue
            for decl in stmt.declarations:
                if not isinstance(decl, JsVariableDeclarator):
                    continue
                if not isinstance(decl.id, JsIdentifier):
                    continue
                if not isinstance(decl.init, JsObjectExpression):
                    continue
                prop_map = _build_property_map(decl.init)
                if prop_map is None:
                    continue
                yield decl.id.name, decl, prop_map

    @staticmethod
    def _is_safe_to_fold(root: Node, name: str, declarator: JsVariableDeclarator) -> bool:
        """
        Verify that the variable is never reassigned, passed as an argument, or used in any
        context other than `obj['key']` or `obj.key` member access.
        """
        decl_name_node = declarator.id
        for node in root.walk():
            if node is decl_name_node:
                continue
            if not isinstance(node, JsIdentifier) or node.name != name:
                continue
            p = node.parent
            if isinstance(p, JsMemberExpression) and p.object is node:
                continue
            return False
        return True

    @staticmethod
    def _inline_references(
        root: Node,
        name: str,
        prop_map: dict[str, Node],
    ) -> bool:
        """
        Replace all `obj['key']` accesses with the corresponding property value. For function-valued
        properties called as `obj['key'](args)`, inline the call. Returns whether any replacement
        was made.
        """
        changed = False
        for node in list(root.walk()):
            if not isinstance(node, JsMemberExpression):
                continue
            if not isinstance(node.object, JsIdentifier) or node.object.name != name:
                continue
            key = _access_key(node)
            if key is None or key not in prop_map:
                continue
            value = prop_map[key]
            parent = node.parent
            if (
                isinstance(parent, JsCallExpression)
                and parent.callee is node
                and isinstance(value, JsFunctionExpression)
            ):
                replacement = _try_inline_call(value, parent.arguments)
                if replacement is not None:
                    _replace_in_parent(parent, replacement)
                    changed = True
                    continue
            _replace_in_parent(node, _clone_node(value))
            changed = True
        return changed
