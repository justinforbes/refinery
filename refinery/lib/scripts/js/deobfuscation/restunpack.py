"""
Unpack rest-parameter arrays that pack multiple variables into a single parameter.

Some obfuscation transforms replace named function parameters and locals with indexed accesses
on a single rest parameter array:

    function(...stack) { stack.length = N; ... }

This transformer detects the pattern, builds a variable map from collected access keys, and
replaces indexed accesses with fresh named identifiers.
"""
from __future__ import annotations

from typing import NamedTuple

from refinery.lib.scripts import Node, _replace_in_parent, set_body
from refinery.lib.scripts.js.analysis.cache import model_cache
from refinery.lib.scripts.js.analysis.model import SemanticModel
from refinery.lib.scripts.js.deobfuscation.helpers import (
    ScriptLevelTransformer,
    member_key,
    numeric_value,
)
from refinery.lib.scripts.js.model import (
    JsAssignmentExpression,
    JsBlockStatement,
    JsExpressionStatement,
    JsFunctionDeclaration,
    JsFunctionExpression,
    JsIdentifier,
    JsMemberExpression,
    JsNumericLiteral,
    JsRestElement,
    JsScript,
    JsStringLiteral,
    JsUnaryExpression,
    JsVariableDeclaration,
    JsVariableDeclarator,
    JsVarKind,
)
from refinery.lib.scripts.js.numbers import exact_integer, js_number_to_string


_MAX_PARAMETERS = 65535
"""
The largest truncation length this pass will rewrite into a formal parameter list. A larger one does
not describe the pattern, and taking it at face value would spend the rest of the run minting names
for parameters no engine would accept.

The number is not itself an engine limit and should not be read as one: V8 refuses a function of
more than 65525 formal parameters, so a length between that and this bound is still rewritten into a
program the engine will not load. Nor is every larger length a runtime error to begin with — a
non-uint32 length such as `1e300` throws `RangeError`, but any uint32 up to 4294967295 does not.
"""


class _TruncationInfo(NamedTuple):
    param_count: int
    stack_chain: str | None
    length_access: JsMemberExpression


class _NestedFrameAccess(Exception):
    pass


def _extract_truncation(
    stmts: list,
    rest_name: str,
) -> _TruncationInfo | None:
    """
    Find the `.length = N` truncation statement in the function body. Returns the param count
    and the stack chain key (None for simple case where rest param IS the stack). Returns None
    if no truncation pattern is found.

    The object decides which statement is the truncation, and the length is read afterwards: the
    first write to the length of the rest array or of a resolvable chain is the one this rewrite is
    about, and a length it cannot read as a parameter count means the rewrite cannot be done.
    Reading the length first and moving on when it does not answer would let a later, unrelated
    `.length =` stand in for the real one, and the parameter list would then be built from a count
    belonging to something else.
    """
    for stmt in stmts:
        if not isinstance(stmt, JsExpressionStatement):
            continue
        expr = stmt.expression
        if not isinstance(expr, JsAssignmentExpression) or expr.operator != '=':
            continue
        lhs = expr.left
        if not isinstance(lhs, JsMemberExpression):
            continue
        if lhs.computed:
            continue
        if not isinstance(lhs.property, JsIdentifier) or lhs.property.name != 'length':
            continue
        obj = lhs.object
        if isinstance(obj, JsIdentifier) and obj.name == rest_name:
            chain = None
        elif isinstance(obj, JsMemberExpression):
            chain = member_key(obj)
            if chain is None:
                continue
        else:
            continue
        rhs = expr.right
        n = None if rhs is None else numeric_value(rhs)
        length = None if n is None else exact_integer(n)
        if length is None or not (0 <= length <= _MAX_PARAMETERS):
            return None
        return _TruncationInfo(length, chain, lhs)
    return None


def _collect_accesses_simple(
    body: JsBlockStatement,
    rest_name: str,
    length_access: JsMemberExpression,
) -> dict[str, list[JsMemberExpression]] | None:
    """
    Collect all `restParam[key]` and `restParam.key` accesses in the immediate function body
    (not descending into nested functions). Returns a map from string key to list of AST nodes.
    Returns None if the rest param is used in a way that prevents demasking.

    Only the one `.length` member that *length_access* names is skipped, because it is the one the
    rewrite removes. Any other mention of the rest array's length is an ordinary read of a binding
    the rewrite is about to delete, and answering it with a rewritten body would leave a reference
    to a name that no longer exists.
    """
    accesses: dict[str, list[JsMemberExpression]] = {}
    if not _walk_collect_simple(body, rest_name, accesses, length_access):
        return None
    return accesses


def _walk_collect_simple(
    node: Node,
    rest_name: str,
    accesses: dict[str, list[JsMemberExpression]],
    length_access: JsMemberExpression,
) -> bool:
    for child in node.children():
        if isinstance(child, (JsFunctionExpression, JsFunctionDeclaration)):
            continue
        if isinstance(child, JsMemberExpression):
            obj = child.object
            if isinstance(obj, JsIdentifier) and obj.name == rest_name:
                if child is length_access:
                    continue
                key = _extract_access_key(child)
                if key is None:
                    return False
                accesses.setdefault(key, []).append(child)
                continue
        if isinstance(child, JsIdentifier) and child.name == rest_name:
            parent = child.parent
            if isinstance(parent, JsMemberExpression) and parent.object is child:
                continue
            return False
        if not _walk_collect_simple(child, rest_name, accesses, length_access):
            return False
    return True


def _collect_accesses_frame(
    body: JsBlockStatement,
    stack_chain: str,
) -> dict[str, list[JsMemberExpression]] | None:
    """
    Collect all accesses to the frame-qualified stack chain. Returns None if any access exists
    inside a nested function (closure capture prevents demasking).
    """
    accesses: dict[str, list[JsMemberExpression]] = {}
    try:
        _walk_collect_frame(body, stack_chain, accesses, depth=0)
    except _NestedFrameAccess:
        return None
    return accesses


def _walk_collect_frame(
    node: Node,
    stack_chain: str,
    accesses: dict[str, list[JsMemberExpression]],
    depth: int,
) -> None:
    for child in node.children():
        if isinstance(child, (JsFunctionExpression, JsFunctionDeclaration)):
            _walk_collect_frame(child, stack_chain, accesses, depth + 1)
            continue
        if isinstance(child, JsMemberExpression):
            obj = child.object
            if isinstance(obj, JsMemberExpression):
                chain = member_key(obj)
                if chain == stack_chain:
                    key = _extract_access_key(child)
                    if key is not None:
                        if depth > 0:
                            raise _NestedFrameAccess
                        accesses.setdefault(key, []).append(child)
                        continue
        _walk_collect_frame(child, stack_chain, accesses, depth)


def _numeric_key(value: float) -> str | None:
    """
    The property name a Number indexes, or `None` when it names no slot this pass rewrites. The
    spelling is `Number.prototype.toString` and not `str` of a Python integer, because the two part
    ways exactly where a double stops determining its own digits: `s[2 ** 60]` reads the property
    `'1152921504606847000'`, which is not what the exact value spells.
    """
    if exact_integer(value) is None:
        return None
    return js_number_to_string(value)


def _extract_access_key(node: JsMemberExpression) -> str | None:
    """
    Extract the key from a stack access expression. Returns a string representation of the key
    or None if the key cannot be statically resolved.
    """
    if node.computed:
        prop = node.property
        if isinstance(prop, JsNumericLiteral):
            return _numeric_key(prop.value)
        if isinstance(prop, JsStringLiteral):
            return prop.value
        if (
            isinstance(prop, JsUnaryExpression)
            and prop.operator == '-'
            and isinstance(prop.operand, JsNumericLiteral)
        ):
            return _numeric_key(-prop.operand.value)
        return None
    if isinstance(node.property, JsIdentifier):
        if node.property.name == 'length':
            return None
        return node.property.name
    return None


def _mentioned_names(node: Node) -> set[str]:
    """
    Every identifier name the subtree mentions. A name this pass introduces has to avoid all of them
    and not only the declared ones: one that matches a local the body declares shadows it, and one
    that matches a name the body reads from an enclosing scope captures it.
    """
    return {child.name for child in node.walk() if isinstance(child, JsIdentifier)}


def _generate_names(
    param_count: int,
    keys: set[str],
    taken: set[str],
) -> tuple[dict[str, str], list[str]]:
    """
    Fresh identifier names for the stack keys, together with the whole parameter list they are drawn
    from. A key naming an index below *param_count* is that parameter and takes its name from its
    position, so that two keys can never land on one name; every other key names a local.
    """
    used = set(taken)
    params: list[str] = []
    candidate = 0
    while len(params) < param_count:
        name = F'p{candidate}'
        candidate += 1
        if name not in used:
            used.add(name)
            params.append(name)
    mapping: dict[str, str] = {}
    candidate = 0
    for key in sorted(keys, key=_sort_key):
        if key.isdigit() and int(key) < param_count:
            mapping[key] = params[int(key)]
            continue
        while True:
            name = F'v{candidate}'
            candidate += 1
            if name not in used:
                break
        used.add(name)
        mapping[key] = name
    return mapping, params


def _sort_key(key: str) -> tuple[int, int | str]:
    try:
        n = int(key)
        return (0, n)
    except ValueError:
        return (1, key)


def _remove_truncation(body: JsBlockStatement, length_access: JsMemberExpression) -> None:
    """
    Remove the truncation statement from the function body. It is found by the member node the match
    recorded rather than by re-running the match: a body may hold more than one `.length =` and only
    the one that was read as the parameter count may be dropped.
    """
    stmts = body.body
    for i, stmt in enumerate(stmts):
        if not isinstance(stmt, JsExpressionStatement):
            continue
        expr = stmt.expression
        if isinstance(expr, JsAssignmentExpression) and expr.left is length_access:
            stmts.pop(i)
            return


class JsRestArrayUnpacking(ScriptLevelTransformer):
    """
    Unpack rest-param arrays back into named identifiers. Detects functions where all parameters
    and locals are packed into a single rest parameter accessed by index, and replaces indexed
    accesses with fresh named variables.
    """

    def _process_script(self, node: JsScript) -> None:
        count = 0
        model = model_cache(self, node).model
        for fn_node in node.walk():
            if not isinstance(fn_node, (JsFunctionExpression, JsFunctionDeclaration)):
                continue
            if self._demask_function(fn_node, model):
                count += 1
        if count > 0:
            self.mark_changed()

    def _demask_function(
        self,
        fn: JsFunctionExpression | JsFunctionDeclaration,
        model: SemanticModel,
    ) -> bool:
        if len(fn.params) != 1:
            return False
        param = fn.params[0]
        if not isinstance(param, JsRestElement):
            return False
        if not isinstance(param.argument, JsIdentifier):
            return False
        binding = model.binding_of(param.argument)
        if binding is None or binding.captured or model.reflection_can_reach(binding):
            return False
        rest_name = param.argument.name
        if fn.body is None or not isinstance(fn.body, JsBlockStatement):
            return False
        if not fn.body.body:
            return False
        result = _extract_truncation(fn.body.body, rest_name)
        if result is None:
            return False
        param_count, stack_chain, length_access = result
        if stack_chain is None:
            accesses = _collect_accesses_simple(fn.body, rest_name, length_access)
        else:
            accesses = _collect_accesses_frame(fn.body, stack_chain)
        if accesses is None:
            return False
        if param_count > 0 and not any(str(i) in accesses for i in range(param_count)):
            return False
        if not accesses:
            _remove_truncation(fn.body, length_access)
            fn.params.clear()
            return True
        taken = _mentioned_names(fn.body)
        mapping, params = _generate_names(param_count, set(accesses.keys()), taken)
        for key, nodes in accesses.items():
            name = mapping[key]
            for access_node in nodes:
                replacement = JsIdentifier(name=name)
                _replace_in_parent(access_node, replacement)
        _remove_truncation(fn.body, length_access)
        fn.params.clear()
        for name in params:
            fn.params.append(JsIdentifier(name=name))
        if stack_chain is None:
            self._add_local_declarations(fn.body, mapping, param_count)
        return True

    def _add_local_declarations(
        self,
        body: JsBlockStatement,
        mapping: dict[str, str],
        param_count: int,
    ) -> None:
        """
        Insert `var` declarations for local variables (keys that aren't parameters).
        """
        locals_: list[str] = []
        for key, name in mapping.items():
            try:
                idx = int(key)
                if 0 <= idx < param_count:
                    continue
            except ValueError:
                pass
            locals_.append(name)
        if not locals_:
            return
        declarators = [
            JsVariableDeclarator(id=JsIdentifier(name=n), init=None)
            for n in locals_
        ]
        decl = JsVariableDeclaration(declarations=declarators, kind=JsVarKind.VAR)
        for d in declarators:
            d.parent = decl
            if d.id is not None:
                d.id.parent = d
        set_body(body, [decl, *body.body])
