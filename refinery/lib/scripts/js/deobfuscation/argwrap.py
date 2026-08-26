"""
The obfuscator converts statement sequences into calls to a self-disabling no-op function whose
arguments carry all side effects. This transformer detects the pattern structurally, expands the
calls that reach such a function back into individual statements, and removes the declaration where
the expansion left nothing that reads it.

Which calls reach it is a question about the binding the wrapper declares and never about the name
it answers to, and every part of that question is settled before anything is rewritten:
`_the_calls_that_reach` holds the rule.
"""
from __future__ import annotations

from typing import NamedTuple

from refinery.lib.scripts import Expression, Node, _remove_from_parent, _replace_in_parent
from refinery.lib.scripts.js.analysis.cache import model_cache
from refinery.lib.scripts.js.analysis.model import SemanticModel
from refinery.lib.scripts.js.deobfuscation.helpers import ScriptLevelTransformer
from refinery.lib.scripts.js.model import (
    FUNCTION_NODES,
    JsAssignmentExpression,
    JsBlockStatement,
    JsCallExpression,
    JsExportNamedDeclaration,
    JsExpressionStatement,
    JsForInStatement,
    JsForOfStatement,
    JsFunctionDeclaration,
    JsFunctionExpression,
    JsIdentifier,
    JsNumericLiteral,
    JsScript,
    JsSequenceExpression,
    JsSpreadElement,
    JsSwitchCase,
    JsUnaryExpression,
    JsVariableDeclarator,
    wraps_return,
)


class _Wrapper(NamedTuple):
    """
    A self-disabling wrapper declaration and every call that reaches it.

    The calls are the nodes themselves rather than a question asked again later, because the
    rewrite moves them: the declaration goes exactly when every one of them was expanded, and that
    is a question about the calls this list named before anything moved.
    """
    declaration: JsFunctionDeclaration
    calls: list[JsCallExpression]


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
    if wraps_return(rhs):
        return False
    if rhs.params:
        return False
    if isinstance(rhs.body, JsBlockStatement) and rhs.body.body:
        return False
    return True


def _names_an_export_list_reads(root: Node) -> set[str]:
    """
    The names the export lists of *root* read, less any list that names the file to read them from.

    `export { W }` is a read of `W` that `refinery.lib.scripts.js.analysis.model.is_use_position`
    does not record, so the model reports the binding as one nothing outside its own declaration
    names. Only a list carrying a source, `export { W } from './m.js'`, names something across the
    module boundary and nothing local at all.
    """
    names: set[str] = set()
    for node in root.walk():
        if not isinstance(node, JsExportNamedDeclaration) or node.source is not None:
            continue
        for specifier in node.specifiers:
            if isinstance(local := specifier.local, JsIdentifier):
                names.add(local.name)
    return names


def _stands_as_a_statement_of_a_function_or_the_script(node: JsFunctionDeclaration) -> bool:
    """
    Whether *node* is a statement of the script or of a function body, which is the only placement
    the whole of a declaration can be read from.

    A block-scoped function declaration is one thing in strict code, where it is scoped to the
    block, and another in sloppy code, where the enclosing scope holds the name from the start and
    reaching the block is what puts the function in it; a labeled or exported declaration sits in a
    slot `_remove_from_parent` cannot take it from. A block is a function body when the node holding
    it is a function and that block is its body, which no static block or catch clause is.
    """
    parent = node.parent
    if isinstance(parent, JsScript):
        return True
    if not isinstance(parent, JsBlockStatement):
        return False
    owner = parent.parent
    return isinstance(owner, FUNCTION_NODES) and owner.body is parent


def _a_bare_var_declarator_outside_a_loop_head(declared: Node) -> bool:
    """
    Whether *declared* declares a `var` with no initializer outside a loop head, which is the one
    further declaration of a wrapper's name that says nothing about what the name holds.

    An initializer is a declaration and not a write, and the store a `for-in` or `for-of` head makes
    is neither, so the model reports no reference for either of them. A binding carrying one is a
    binding something puts a value into that the expansion would never have seen.
    """
    declarator = declared.parent
    if not isinstance(declarator, JsVariableDeclarator) or declarator.init is not None:
        return False
    declaration = declarator.parent
    if declaration is None:
        return False
    return not isinstance(declaration.parent, (JsForInStatement, JsForOfStatement))


def _the_calls_that_reach(
    model: SemanticModel,
    declaration: JsFunctionDeclaration,
) -> list[JsCallExpression] | None:
    """
    Every call that reaches the wrapper *declaration* declares, or `None` where the expansion is not
    equivalent for the binding it declares at all.

    Expanding a call to the statements its arguments carry is equivalent only where the name holds
    this wrapper when the call runs and nothing observes that the call then disabled it. That is
    three questions about the binding:

    - it is declared once, here, but for a bare `var` of the same name, which stores nothing;
    - every write to it is the self-disabling assignment inside this declaration, so nothing else
      ever puts a value into the name;
    - every reference to it, including one a `with` body resolves at run time, is the callee of a
      call, so that no read of the name can tell the wrapper from what it left behind.

    A reference that is not an identifier stands in for an access made through an object aliasing
    the binding, which is not a name this pass can follow.
    """
    assert declaration.id is not None
    binding = model.binding_of(declaration.id)
    if binding is None:
        return None
    if sum(1 for name in binding.declarations if name is declaration.id) != 1:
        return None
    for name in binding.declarations:
        if name is not declaration.id and not _a_bare_var_declarator_outside_a_loop_head(name):
            return None
    for write in (*binding.writes, *binding.indefinite_writes):
        if not isinstance(write, JsIdentifier) or not write.is_descendant_of(declaration):
            return None
    calls: list[JsCallExpression] = []
    for reference in (*binding.reads, *binding.dynamic_refs):
        if not isinstance(reference, JsIdentifier):
            return None
        call = reference.parent
        if not isinstance(call, JsCallExpression) or call.callee is not reference:
            return None
        calls.append(call)
    return calls


def _admitted_wrappers(model: SemanticModel, root: JsScript) -> list[_Wrapper]:
    """
    The self-disabling wrappers of *root* the pass may expand, each with the calls that reach it.

    A wrapper whose call answers a promise or a generator object is refused: expanding one call site
    takes the body that disables the wrapper with it, so a call left standing beside an expanded one
    answers a promise where the input answered `undefined`. This is decided per declaration and no
    longer poisons a plain wrapper elsewhere that happens to answer to the same name.
    """
    shaped = [
        node for node in root.walk()
        if isinstance(node, JsFunctionDeclaration) and _is_expression_wrapper(node)
    ]
    if not shaped:
        return []
    exported = _names_an_export_list_reads(root)
    wrappers: list[_Wrapper] = []
    for node in shaped:
        assert node.id is not None
        if wraps_return(node):
            continue
        if not _stands_as_a_statement_of_a_function_or_the_script(node):
            continue
        if node.id.name in exported:
            continue
        calls = _the_calls_that_reach(model, node)
        if calls is None:
            continue
        wrappers.append(_Wrapper(node, calls))
    return wrappers


class JsAssignmentsAsFunctionArgs(ScriptLevelTransformer):
    """
    Detect self-disabling wrapper functions and expand the call sites that reach one: a call in
    statement position becomes the individual argument statements, and a call embedded in a larger
    expression becomes the equivalent comma sequence in place, so evaluation order is preserved.

    Which wrapper a call site reaches, and which of them may be expanded at all, is decided by
    `_admitted_wrappers` before the first call is rewritten.
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

    def _expand(self, call: JsCallExpression) -> bool:
        """
        Expand one call to the statements or the sequence it is worth, and report whether it was.

        A spread argument is neither a statement nor a sequence operand, and a statement its own
        list does not hold cannot be spliced into; either way the call stays, and the wrapper it
        reaches stays with it.
        """
        if any(isinstance(arg, JsSpreadElement) for arg in call.arguments):
            return False
        parent = call.parent
        if (
            isinstance(parent, JsExpressionStatement)
            and isinstance(block := parent.parent, (JsBlockStatement, JsScript, JsSwitchCase))
        ):
            body = block.body
            try:
                index = body.index(parent)
            except ValueError:
                return False
            statements = [JsExpressionStatement(expression=arg) for arg in call.arguments]
            body[index:index + 1] = statements
            for statement in statements:
                statement.parent = block
            return True
        _replace_in_parent(call, self._sequence_lowering(call.arguments))
        return True

    def _process_script(self, node: JsScript):
        wrappers = _admitted_wrappers(model_cache(self, node).model, node)
        if not wrappers:
            return
        reached = {id(call) for wrapper in wrappers for call in wrapper.calls}
        expanded: set[int] = set()
        for ast_node in list(node.walk()):
            if not isinstance(ast_node, JsCallExpression) or id(ast_node) not in reached:
                continue
            if self._expand(ast_node):
                expanded.add(id(ast_node))
        if not expanded:
            return
        self.mark_changed()
        for wrapper in wrappers:
            if all(id(call) in expanded for call in wrapper.calls):
                _remove_from_parent(wrapper.declaration)
