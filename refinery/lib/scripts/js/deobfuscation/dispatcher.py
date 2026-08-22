"""
The dispatcher obfuscation wraps function bodies into a central routing function that uses a string
keyed lookup table and a global payload array for argument passing. This transformer detects the
pattern structurally (no reliance on variable names), extracts the original functions, rewrites all
call sites, and removes the dispatcher scaffolding.
"""
from __future__ import annotations

from dataclasses import dataclass

from refinery.lib.scripts import (
    Node,
    _remove_from_parent,
    _replace_in_parent,
)
from refinery.lib.scripts.js.analysis.cache import model_cache
from refinery.lib.scripts.js.deobfuscation.helpers import (
    ScopeProcessingTransformer,
    access_key,
    binding_has_references,
    make_undefined_expression,
    property_key,
    remove_declarator,
)
from refinery.lib.scripts.js.model import (
    JsArrayExpression,
    JsArrayPattern,
    JsAssignmentExpression,
    JsBinaryExpression,
    JsBlockStatement,
    JsCallExpression,
    JsExpressionStatement,
    JsFunctionDeclaration,
    JsFunctionExpression,
    JsIdentifier,
    JsIfStatement,
    JsLogicalExpression,
    JsMemberExpression,
    JsNewExpression,
    JsNullLiteral,
    JsObjectExpression,
    JsProperty,
    JsReturnStatement,
    JsScript,
    JsSequenceExpression,
    JsStringLiteral,
    JsVariableDeclaration,
    JsVariableDeclarator,
)


@dataclass
class _DispatcherInfo:
    """
    All structurally-extracted metadata about a single dispatcher function.
    """
    decl: JsFunctionDeclaration
    dispatcher_id: str
    fns_map: dict[str, JsFunctionExpression]
    fns_declarator: JsVariableDeclarator
    payload_id: str
    wrap_key: str | None
    cache_id: str | None


@dataclass
class _DispatchSite:
    """
    One dispatch this pass can read: the expression it is written as, the name of the function it
    selects, and the arguments that function is reached with. *arguments* is `None` where the site
    names the function rather than calling it, which the wrapped-reference form does.
    """
    node: Node
    key: str
    arguments: list | None

    def replacement(self) -> Node:
        if self.arguments is None:
            return JsIdentifier(name=self.key)
        return JsCallExpression(
            callee=JsIdentifier(name=self.key),
            arguments=self.arguments,
        )


def _extract_fns_table(
    body: list,
) -> tuple[JsVariableDeclarator, dict[str, JsFunctionExpression]] | None:
    """
    Finds a declaration of the form

        var fns = { ... }

    where every property value is a zero-parameter
    `refinery.lib.scripts.js.model.JsFunctionExpression`. Returns the declarator node and a map from
    string key to function.
    """
    for stmt in body:
        if not isinstance(stmt, JsVariableDeclaration):
            continue
        for decl in stmt.declarations:
            if not isinstance(decl, JsVariableDeclarator):
                continue
            if not isinstance(decl.init, JsObjectExpression):
                continue
            obj = decl.init
            if not obj.properties:
                continue
            fns: dict[str, JsFunctionExpression] = {}
            ok = True
            for prop in obj.properties:
                if not isinstance(prop, JsProperty):
                    ok = False
                    break
                key = property_key(prop)
                if key is None:
                    ok = False
                    break
                if not isinstance(prop.value, JsFunctionExpression):
                    ok = False
                    break
                if prop.value.params:
                    ok = False
                    break
                fns[key] = prop.value
            if ok and fns:
                return decl, fns
    return None


def _find_payload_id(body: list, second_param: str) -> str | None:
    """
    Find the payload-init guard:

        if (p1 === "...") { payload = []; }

    and return the payload identifier name. The guard compares the function's second parameter to a
    string literal and assigns an empty array to the payload variable.
    """
    for stmt in body:
        if not isinstance(stmt, JsIfStatement):
            continue
        test = stmt.test
        if not isinstance(test, JsBinaryExpression) or test.operator != '===':
            continue
        if not (
            isinstance(test.left, JsIdentifier)
            and test.left.name == second_param
            and isinstance(test.right, JsStringLiteral)
        ):
            continue
        cons = stmt.consequent
        if isinstance(cons, JsBlockStatement) and len(cons.body) == 1:
            cons = cons.body[0]
        if not isinstance(cons, JsExpressionStatement):
            continue
        expr = cons.expression
        if not isinstance(expr, JsAssignmentExpression) or expr.operator != '=':
            continue
        if isinstance(expr.left, JsIdentifier) and isinstance(expr.right, JsArrayExpression):
            if not expr.right.elements:
                return expr.left.name
    return None


def _find_wrap_key(body: list, third_param: str) -> str | None:
    """
    Find the return-type wrapper:

        if (p2 === "...") { return { "wrapKey": output }; }

    and return the wrapper property name.
    """
    for stmt in body:
        if not isinstance(stmt, JsIfStatement):
            continue
        test = stmt.test
        if not isinstance(test, JsBinaryExpression) or test.operator != '===':
            continue
        if not (
            isinstance(test.left, JsIdentifier)
            and test.left.name == third_param
            and isinstance(test.right, JsStringLiteral)
        ):
            continue
        cons = stmt.consequent
        if isinstance(cons, JsBlockStatement) and len(cons.body) == 1:
            inner = cons.body[0]
        else:
            inner = cons
        if not isinstance(inner, JsReturnStatement):
            continue
        ret_val = inner.argument
        if not isinstance(ret_val, JsObjectExpression):
            continue
        if len(ret_val.properties) != 1:
            continue
        prop = ret_val.properties[0]
        if isinstance(prop, JsProperty):
            key = property_key(prop)
            if key is not None:
                return key
    return None


def _find_cache_id(body: list, first_param: str) -> str | None:
    """
    Find the cache variable from the create-flag branch. Looks for an `if` whose body contains a
    logical-or assignment like

        cache[p0] || (cache[p0] = ...)

    Returns the cache identifier.
    """
    for stmt in body:
        if not isinstance(stmt, JsIfStatement):
            continue
        for node in stmt.walk():
            if not isinstance(node, JsMemberExpression):
                continue
            if (
                isinstance(node.object, JsIdentifier)
                and isinstance(node.property, JsIdentifier)
                and node.property.name == first_param
                and node.computed
            ):
                parent = node.parent
                if isinstance(parent, JsLogicalExpression) and parent.operator == '||':
                    return node.object.name
    return None


def _detect_dispatcher(func: JsFunctionDeclaration) -> _DispatcherInfo | None:
    """
    Structurally detect whether `func` is a dispatcher function. Returns the extracted metadata or
    `None` if the function does not match the pattern.
    """
    if not isinstance(func.id, JsIdentifier):
        return None
    if not isinstance(func.body, JsBlockStatement):
        return None
    if len(func.params) < 3:
        return None
    p0 = func.params[0]
    p1 = func.params[1]
    p2 = func.params[2]
    if (
        not isinstance(p0, JsIdentifier)
        or not isinstance(p1, JsIdentifier)
        or not isinstance(p2, JsIdentifier)
    ):
        return None
    first_param: str = p0.name
    second_param: str = p1.name
    third_param: str = p2.name
    body = func.body.body
    result = _extract_fns_table(body)
    if result is None:
        return None
    fns_declarator, fns_map = result
    payload_id = _find_payload_id(body, second_param)
    if payload_id is None:
        return None
    wrap_key = _find_wrap_key(body, third_param)
    cache_id = _find_cache_id(body, first_param)
    return _DispatcherInfo(
        decl=func,
        dispatcher_id=func.id.name,
        fns_map=fns_map,
        fns_declarator=fns_declarator,
        payload_id=payload_id,
        wrap_key=wrap_key,
        cache_id=cache_id,
    )


def _extract_params(
    fn: JsFunctionExpression,
    payload_id: str,
) -> list[JsIdentifier] | None:
    """
    Extract parameter names from the leading payload destructuring statement:

        var [a, b] = payload;

    Returns the parameter identifiers or `None` if the pattern is not found.
    """
    if not isinstance(fn.body, JsBlockStatement) or not fn.body.body:
        return []
    first = fn.body.body[0]
    if not isinstance(first, JsVariableDeclaration):
        return []
    for decl in first.declarations:
        if not isinstance(decl, JsVariableDeclarator):
            continue
        if not isinstance(decl.id, JsArrayPattern):
            continue
        if not isinstance(decl.init, JsIdentifier):
            continue
        if decl.init.name != payload_id:
            continue
        params: list[JsIdentifier] = []
        for elem in decl.id.elements:
            if not isinstance(elem, JsIdentifier):
                return None
            params.append(JsIdentifier(name=elem.name))
        return params
    return []


def _build_extracted_function(
    key: str,
    fn: JsFunctionExpression,
    payload_id: str,
) -> JsFunctionDeclaration | None:
    """
    Convert a dispatcher function-table entry into a standalone
    `refinery.lib.scripts.js.model.JsFunctionDeclaration` of the same kind. Extracts parameters from
    the payload destructuring and removes that statement.

    The kind is carried rather than defaulted: an entry that was `async` returns a promise and one
    that was a generator returns an iterator, and a declaration that dropped either would compute a
    different value — a `yield` left in a body no longer marked `*` does not even parse.
    """
    params = _extract_params(fn, payload_id)
    if params is None:
        return None
    body = fn.body
    if not isinstance(body, JsBlockStatement):
        return None
    new_body_stmts = list(body.body)
    if new_body_stmts and params:
        first = new_body_stmts[0]
        if isinstance(first, JsVariableDeclaration):
            remaining = [
                d for d in first.declarations
                if not (
                    isinstance(d, JsVariableDeclarator)
                    and isinstance(d.id, JsArrayPattern)
                    and isinstance(d.init, JsIdentifier)
                    and d.init.name == payload_id
                )
            ]
            if not remaining:
                new_body_stmts = new_body_stmts[1:]
            else:
                first.declarations = remaining
    new_body = JsBlockStatement(body=new_body_stmts)
    decl = JsFunctionDeclaration(
        id=JsIdentifier(name=key),
        params=list(params),
        body=new_body,
        generator=fn.generator,
        is_async=fn.is_async,
    )
    return decl


def _is_object_create_null(node: Node) -> bool:
    """
    Check if *node* is `Object.create(null)`.
    """
    if not isinstance(node, JsCallExpression):
        return False
    if len(node.arguments) != 1 or not isinstance(node.arguments[0], JsNullLiteral):
        return False
    callee = node.callee
    if not isinstance(callee, JsMemberExpression):
        return False
    if not isinstance(callee.object, JsIdentifier) or callee.object.name != 'Object':
        return False
    prop = callee.property
    if isinstance(prop, JsStringLiteral):
        return prop.value == 'create'
    if isinstance(prop, JsIdentifier) and not callee.computed:
        return prop.name == 'create'
    return False


class JsDispatcherUnwrapper(ScopeProcessingTransformer):
    """
    Detect and unwrap a dispatcher pattern. For each dispatcher found, extract the wrapped
    functions, rewrite call sites, and remove the dispatcher scaffolding.
    """

    def __init__(self):
        super().__init__()
        self._root: JsScript | None = None

    def visit_JsScript(self, node: JsScript):
        self._root = node
        return super().visit_JsScript(node)

    def _process_scope_body(self, scope: Node, body: list) -> None:
        for func in list(body):
            if not isinstance(func, JsFunctionDeclaration):
                continue
            info = _detect_dispatcher(func)
            if info is None:
                continue
            self._unwrap_dispatcher(scope, body, info)

    def _unwrap_dispatcher(
        self,
        scope: Node,
        body: list,
        info: _DispatcherInfo,
    ) -> None:
        plan = self._plan_call_sites(scope, info)
        if not self._plan_covers_every_reference(info, plan):
            return
        extracted: dict[str, JsFunctionDeclaration] = {}
        for key, fn in info.fns_map.items():
            decl = _build_extracted_function(key, fn, info.payload_id)
            if decl is None:
                return
            extracted[key] = decl
        for site in plan:
            _replace_in_parent(site.node, site.replacement())
        insert_idx = body.index(info.decl)
        body.remove(info.decl)
        for i, (key, decl) in enumerate(extracted.items()):
            decl.parent = scope
            body.insert(insert_idx + i, decl)
        self.mark_changed()
        self._remove_boilerplate(scope, body, info)

    def _plan_call_sites(
        self,
        scope: Node,
        info: _DispatcherInfo,
    ) -> list[_DispatchSite]:
        """
        Every dispatch through *info* that this pass can read, with nothing replaced yet. Whether
        the dispatcher may be removed at all is a question about the whole set, so the set has to
        exist before the first replacement does.

        A site nested inside another is dropped: the outer replacement takes the inner one with it,
        and replacing a node that is no longer in the tree puts the new one nowhere.
        """
        planned: list[_DispatchSite] = []
        for node in list(scope.walk()):
            if isinstance(node, JsSequenceExpression):
                site = self._read_direct_call(node, info)
            elif isinstance(node, JsMemberExpression):
                site = self._read_wrapped_ref(node, info)
            elif isinstance(node, JsCallExpression):
                site = self._read_bare_call(node, info)
            else:
                continue
            if site is not None:
                planned.append(site)
        nested = {
            id(inner)
            for site in planned
            for inner in site.node.walk()
            if inner is not site.node
        }
        return [site for site in planned if id(site.node) not in nested]

    def _plan_covers_every_reference(
        self,
        info: _DispatcherInfo,
        plan: list[_DispatchSite],
    ) -> bool:
        """
        Whether *plan* replaces every reference to the dispatcher, so that removing its declaration
        leaves nothing naming it. A dispatch this pass cannot read is one it is right to leave
        alone, and what it leaves alone still calls the function it is leaving, so removing the
        declaration anyway hands back a file that throws a `ReferenceError` where the original ran.

        The whole unwrap is refused rather than the removal alone, because extraction is not
        something a surviving dispatcher can share a table with: `_build_extracted_function` reuses
        the statement nodes of the entry it extracts and strips the payload destructuring out of
        them in place, so a dispatcher left standing beside the extracted functions calls bodies
        that no longer read the payload it writes.

        The dispatcher's own body is excluded rather than counted, since it goes with the
        declaration. Everything else is asked of the model, so a same-named binding in another
        scope is not mistaken for a use of this one.
        """
        assert self._root is not None
        if not isinstance(info.decl.id, JsIdentifier):
            return False
        model = model_cache(self, self._root).model
        binding = model.binding_of(info.decl.id)
        rewritten = {id(node) for site in plan for node in site.node.walk()}
        return not binding_has_references(
            model,
            binding,
            exclude=info.decl,
            exclude_ids=rewritten,
        )

    def _read_direct_call(
        self,
        seq: JsSequenceExpression,
        info: _DispatcherInfo,
    ) -> _DispatchSite | None:
        """
        The sequence expression dispatch call *seq* is:

            (payload = [args], dispatcher("key"))  ->  key(args)

        Also reads the wrapped variant where the return value is unwrapped via a member access
        on the wrap key:

            (payload = [args], dispatcher("key", s, wrapFlag)["wk"])
        """
        if len(seq.expressions) != 2:
            return None
        assign, second = seq.expressions
        if not isinstance(assign, JsAssignmentExpression):
            return None
        if assign.operator != '=':
            return None
        if not isinstance(assign.left, JsIdentifier) or assign.left.name != info.payload_id:
            return None
        if not isinstance(assign.right, JsArrayExpression):
            return None
        dispatch_call = self._unwrap_dispatch_call(second, info)
        if dispatch_call is None:
            return None
        if not dispatch_call.arguments:
            return None
        key_arg = dispatch_call.arguments[0]
        if not isinstance(key_arg, JsStringLiteral):
            return None
        if key_arg.value not in info.fns_map:
            return None
        args = [
            make_undefined_expression() if e is None else e
            for e in assign.right.elements
        ]
        return _DispatchSite(seq, key_arg.value, args)

    @staticmethod
    def _unwrap_dispatch_call(
        node: Node,
        info: _DispatcherInfo,
    ) -> JsCallExpression | JsNewExpression | None:
        """
        Extract a dispatcher call from *node*, which may be a bare call or a member access of
        the form:

            dispatcher(...)["wrapKey"]

        Returns the call node or `None`.
        """
        call = node
        if isinstance(node, JsMemberExpression) and info.wrap_key is not None:
            if access_key(node) == info.wrap_key:
                call = node.object
        if not isinstance(call, (JsCallExpression, JsNewExpression)):
            return None
        if not isinstance(call.callee, JsIdentifier):
            return None
        if call.callee.name != info.dispatcher_id:
            return None
        return call

    def _read_wrapped_ref(
        self,
        member: JsMemberExpression,
        info: _DispatcherInfo,
    ) -> _DispatchSite | None:
        """
        The new-expression dispatch with wrap key access *member* is, which names the function it
        selects rather than calling it:

            new dispatcher("key", s2, s3)["wrapKey"]  ->  key
        """
        if info.wrap_key is None:
            return None
        if access_key(member) != info.wrap_key:
            return None
        new_expr = member.object
        if not isinstance(new_expr, JsNewExpression):
            return None
        if not isinstance(new_expr.callee, JsIdentifier):
            return None
        if new_expr.callee.name != info.dispatcher_id:
            return None
        if not new_expr.arguments:
            return None
        key_arg = new_expr.arguments[0]
        if not isinstance(key_arg, JsStringLiteral):
            return None
        if key_arg.value not in info.fns_map:
            return None
        return _DispatchSite(member, key_arg.value, None)

    def _read_bare_call(
        self,
        call: JsCallExpression,
        info: _DispatcherInfo,
    ) -> _DispatchSite | None:
        """
        The bare `dispatcher("key")` call *call* is. These occur without a preceding payload
        assignment, when the dispatched function takes no arguments.
        """
        if not isinstance(call.callee, JsIdentifier):
            return None
        if call.callee.name != info.dispatcher_id:
            return None
        if not call.arguments:
            return None
        key_arg = call.arguments[0]
        if not isinstance(key_arg, JsStringLiteral):
            return None
        if key_arg.value not in info.fns_map:
            return None
        return _DispatchSite(call, key_arg.value, [])

    def _remove_boilerplate(self, scope: Node, body: list, info: _DispatcherInfo) -> None:
        """
        Remove dispatcher-related boilerplate declarations from the scope body.
        """
        assert self._root is not None
        model = model_cache(self, self._root).model
        to_remove = []
        for stmt in list(body):
            if isinstance(stmt, JsVariableDeclaration):
                for decl in stmt.declarations:
                    if not isinstance(decl, JsVariableDeclarator):
                        continue
                    if not isinstance(decl.id, JsIdentifier):
                        continue
                    if decl.id.name == info.payload_id and decl.init is None:
                        remove_declarator(decl)
                        break
                    if info.cache_id and decl.id.name == info.cache_id:
                        if decl.init is not None and _is_object_create_null(decl.init):
                            remove_declarator(decl)
                            break
            elif isinstance(stmt, JsFunctionDeclaration):
                if (
                    isinstance(stmt.id, JsIdentifier)
                    and isinstance(stmt.body, JsBlockStatement)
                    and not stmt.body.body
                    and not stmt.params
                ):
                    binding = model.binding_of(stmt.id)
                    if not binding_has_references(model, binding, exclude=stmt):
                        to_remove.append(stmt)
        for stmt in to_remove:
            _remove_from_parent(stmt)
