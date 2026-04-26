"""
Inline trivial function call wrappers.

A call wrapper is a small function whose only purpose is to forward a call to another function
after rearranging or arithmetically transforming its arguments. This is a common obfuscation
technique that adds a layer of indirection around every call site. The transformer detects these
wrappers and substitutes each call site with the inlined return expression.
"""
from __future__ import annotations

from refinery.lib.scripts import (
    Node,
    Transformer,
    _clone_node,
    _remove_from_parent,
    _replace_in_parent,
)
from refinery.lib.scripts.js.deobfuscation.helpers import is_simple_expression
from refinery.lib.scripts.js.model import (
    JsCallExpression,
    JsFunctionDeclaration,
    JsIdentifier,
    JsReturnStatement,
    JsScript,
)

from typing import NamedTuple, Sequence


class _WrapperInfo(NamedTuple):
    """
    Describes a detected call wrapper function.
    """
    node: JsFunctionDeclaration
    name: str
    param_names: list[str]
    return_expression: Node


def _is_leaf_safe(node: Node, param_names: set[str]) -> bool:
    """
    Check whether every leaf in the expression tree is either a literal or a reference to one of
    the wrapper's parameters. This ensures the return expression has no free variables that would
    make inlining unsafe.
    """
    children = list(node.children())
    if not children:
        if isinstance(node, JsIdentifier):
            return node.name in param_names
        return is_simple_expression(node)
    return all(_is_leaf_safe(child, param_names) for child in children)


def _detect_wrapper(node: JsFunctionDeclaration) -> _WrapperInfo | None:
    """
    Test whether a function declaration is a call wrapper. A call wrapper has one or more
    identifier parameters, a body consisting of a single return statement, and the returned
    expression is a call whose argument sub-expressions reference only the wrapper's parameters
    and literal constants.
    """
    if node.id is None or node.body is None:
        return None
    if not node.params:
        return None
    param_names: list[str] = []
    for p in node.params:
        if not isinstance(p, JsIdentifier):
            return None
        param_names.append(p.name)
    body = node.body.body
    if len(body) != 1:
        return None
    stmt = body[0]
    if not isinstance(stmt, JsReturnStatement) or stmt.argument is None:
        return None
    call = stmt.argument
    if not isinstance(call, JsCallExpression):
        return None
    if not isinstance(call.callee, JsIdentifier):
        return None
    allowed_names = set(param_names)
    allowed_names.add(call.callee.name)
    for arg in call.arguments:
        if not _is_leaf_safe(arg, allowed_names):
            return None
    return _WrapperInfo(node, node.id.name, param_names, call)


def _collect_wrappers(root: Node) -> dict[str, _WrapperInfo]:
    """
    Walk the entire AST and collect all function declarations that qualify as call wrappers.
    """
    wrappers: dict[str, _WrapperInfo] = {}
    for node in root.walk():
        if isinstance(node, JsFunctionDeclaration):
            info = _detect_wrapper(node)
            if info is not None:
                wrappers[info.name] = info
    return wrappers


def _substitute_params(
    expression: Node,
    param_names: list[str],
    arguments: Sequence[Node],
) -> Node:
    """
    Deep-clone the wrapper's return expression and replace every parameter identifier with the
    corresponding call-site argument (also cloned to allow multiple substitutions).
    """
    cloned = _clone_node(expression)
    mapping = {name: arg for name, arg in zip(param_names, arguments)}
    for node in list(cloned.walk()):
        if isinstance(node, JsIdentifier) and node.name in mapping:
            _replace_in_parent(node, _clone_node(mapping[node.name]))
    return cloned


class JsCallWrapperInliner(Transformer):
    """
    Detect trivial call wrapper functions and inline them at every call site.
    """

    def visit_JsScript(self, node: JsScript):
        wrappers = _collect_wrappers(node)
        if not wrappers:
            return None
        inlined = False
        for ast_node in list(node.walk()):
            if not isinstance(ast_node, JsCallExpression):
                continue
            if not isinstance(ast_node.callee, JsIdentifier):
                continue
            info = wrappers.get(ast_node.callee.name)
            if info is None:
                continue
            if len(ast_node.arguments) != len(info.param_names):
                continue
            if not all(is_simple_expression(a) for a in ast_node.arguments):
                continue
            replacement = _substitute_params(
                info.return_expression,
                info.param_names,
                ast_node.arguments,
            )
            _replace_in_parent(ast_node, replacement)
            inlined = True
        if not inlined:
            return None
        exclude_ids: set[int] = set()
        for info in wrappers.values():
            for n in info.node.walk():
                exclude_ids.add(id(n))
        referenced: set[str] = set()
        for n in node.walk():
            if id(n) in exclude_ids:
                continue
            if isinstance(n, JsIdentifier) and n.name in wrappers:
                referenced.add(n.name)
        for name, info in wrappers.items():
            if name not in referenced:
                _remove_from_parent(info.node)
        self.mark_changed()
        return None

    def generic_visit(self, node: Node):
        pass
