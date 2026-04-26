"""
Resolve the string-array rotation pattern produced by popular JavaScript obfuscators.

The obfuscator extracts all string literals into a single array, wraps access through an accessor
function and scrambles the array via a rotation IIFE that push/shifts until a checksum computed
from parseInt of the array's own elements matches a target constant. This transformer detects that
three-part pattern, simulates the rotation in Python, resolves every accessor call to its string
literal, and removes the dead definitions.
"""
from __future__ import annotations

from refinery.lib.scripts import (
    Node,
    Transformer,
    _remove_from_parent,
    _replace_in_parent,
)
from refinery.lib.scripts.js.deobfuscation.helpers import (
    BINARY_OPS,
    js_parse_int,
    make_string_literal,
    string_value,
)
from refinery.lib.scripts.js.model import (
    JsArrayExpression,
    JsAssignmentExpression,
    JsBinaryExpression,
    JsCallExpression,
    JsExpressionStatement,
    JsFunctionDeclaration,
    JsFunctionExpression,
    JsIdentifier,
    JsNumericLiteral,
    JsParenthesizedExpression,
    JsScript,
    JsUnaryExpression,
    JsVariableDeclaration,
    JsVariableDeclarator,
    JsWhileStatement,
)

from typing import NamedTuple, Sequence


class ArrayFunction(NamedTuple):
    """
    Result of detecting the array-holder function pattern.
    """
    node: JsFunctionDeclaration
    name: str
    strings: list[str]


class AccessorFunction(NamedTuple):
    """
    Result of detecting the accessor function pattern.
    """
    node: JsFunctionDeclaration
    name: str
    base_offset: int


class RotationIIFE(NamedTuple):
    """
    Result of detecting the rotation IIFE pattern.
    """
    node: JsExpressionStatement
    target: int
    body: Sequence[Node]


class ChecksumInfo(NamedTuple):
    """
    Result of extracting the checksum expression from the rotation IIFE.
    """
    node: Node
    local_accessor: str


def _find_array_function(body: Sequence[Node]) -> ArrayFunction | None:
    """
    Detect the array-holder function pattern:

        function NAME() {
          var x = ['str0', 'str1', ...];
          NAME = function() { return x; };
          return NAME();
        }

    Returns (node, function_name, initial_string_list) or None.
    """
    for statement in body:
        if not isinstance(statement, JsFunctionDeclaration):
            continue
        if statement.id is None or statement.body is None:
            continue
        name = statement.id.name
        statements = statement.body.body
        if len(statements) < 2:
            continue
        array_literal: list[str] | None = None
        has_self_reassignment = False
        for s in statements:
            if isinstance(s, JsVariableDeclaration):
                for decl in s.declarations:
                    if (
                        isinstance(decl, JsVariableDeclarator)
                        and isinstance(decl.init, JsArrayExpression)
                    ):
                        elements = []
                        for element in decl.init.elements:
                            if (sv := string_value(element)) is None:
                                break
                            elements.append(sv)
                        else:
                            if elements:
                                array_literal = elements
            elif (
                isinstance(s, JsExpressionStatement)
                and isinstance(assign := s.expression, JsAssignmentExpression)
                and isinstance(assign.left, JsIdentifier)
                and assign.left.name == name
            ):
                has_self_reassignment = True
        if array_literal is not None and has_self_reassignment:
            return ArrayFunction(statement, name, array_literal)
    return None


def _find_accessor_function(
    body: Sequence[Node],
    array_fn_name: str,
) -> AccessorFunction | None:
    """
    Detect the accessor function pattern:

        function NAME(param, _unused) {
          param = param - BASE_OFFSET;
          var v = ARRAY_FN();
          var r = v[param];
          return r;
        }

    Returns (node, function_name, base_offset) or None.
    """
    for stmt in body:
        if not isinstance(stmt, JsFunctionDeclaration):
            continue
        if stmt.id is None or stmt.body is None:
            continue
        if len(stmt.params) != 2:
            continue
        fn_name = stmt.id.name
        first_param = stmt.params[0]
        if not isinstance(first_param, JsIdentifier):
            continue
        param_name = first_param.name
        base_offset: int | None = None
        calls_array_fn = False
        for s in stmt.body.body:
            if isinstance(s, JsExpressionStatement) and isinstance(s.expression, JsAssignmentExpression):
                assign = s.expression
                if (
                    isinstance(assign.left, JsIdentifier)
                    and assign.left.name == param_name
                    and isinstance(assign.right, JsBinaryExpression)
                    and assign.right.operator == '-'
                    and isinstance(assign.right.left, JsIdentifier)
                    and assign.right.left.name == param_name
                    and isinstance(assign.right.right, JsNumericLiteral)
                ):
                    base_offset = int(assign.right.right.value)
            elif isinstance(s, JsVariableDeclaration):
                for decl in s.declarations:
                    if isinstance(decl, JsVariableDeclarator) and isinstance(decl.init, JsCallExpression):
                        if isinstance(decl.init.callee, JsIdentifier) and decl.init.callee.name == array_fn_name:
                            calls_array_fn = True
        if base_offset is not None and calls_array_fn:
            return AccessorFunction(stmt, fn_name, base_offset)
    return None


def _find_rotation_iife(
    body: Sequence[Node],
    array_fn_name: str,
) -> RotationIIFE | None:
    """
    Detect the rotation IIFE pattern:

        (function(getArray, target) {
          var arr = getArray();
          while (true) { try { ... parseInt ... push(shift) } catch { push(shift) } }
        })(ARRAY_FN, TARGET_NUMBER);

    Returns (statement_node, target_checksum, iife_body_statements) or None.
    """
    for stmt in body:
        if not isinstance(stmt, JsExpressionStatement):
            continue
        expr = stmt.expression
        if isinstance(expr, JsParenthesizedExpression):
            expr = expr.expression
        call = expr
        if not isinstance(call, JsCallExpression):
            continue
        if not isinstance(call.callee, JsFunctionExpression):
            continue
        if len(call.arguments) != 2:
            continue
        first_arg = call.arguments[0]
        second_arg = call.arguments[1]
        if not (isinstance(first_arg, JsIdentifier) and first_arg.name == array_fn_name):
            continue
        if not isinstance(second_arg, JsNumericLiteral):
            continue
        fn_body = call.callee.body
        if fn_body is None:
            continue
        has_while = False
        for s in fn_body.body:
            if isinstance(s, JsWhileStatement):
                has_while = True
                break
        if has_while:
            return RotationIIFE(stmt, int(second_arg.value), fn_body.body)
    return None


class _EvalError(Exception):
    pass


def _extract_checksum_expression(
    iife_body: Sequence[Node],
    accessor_name: str,
) -> ChecksumInfo | None:
    """
    Extract the checksum expression AST node and the local accessor alias from the rotation
    IIFE body statements. Returns (checksum_expression_node, local_accessor_name) or None.
    """
    local_accessor = accessor_name
    for s in iife_body:
        if isinstance(s, JsVariableDeclaration):
            for decl in s.declarations:
                if (
                    isinstance(decl, JsVariableDeclarator)
                    and isinstance(decl.id, JsIdentifier)
                    and isinstance(decl.init, JsIdentifier)
                    and decl.init.name == accessor_name
                ):
                    local_accessor = decl.id.name
    checksum_node: Node | None = None
    for s in iife_body:
        if isinstance(s, JsWhileStatement) and s.body is not None:
            for ws in s.walk():
                if isinstance(ws, JsVariableDeclaration):
                    for decl in ws.declarations:
                        if isinstance(decl, JsVariableDeclarator) and decl.init is not None:
                            checksum_node = decl.init
                            break
                    if checksum_node is not None:
                        break
            break
    if checksum_node is None:
        return None
    return ChecksumInfo(checksum_node, local_accessor)


def _eval_checksum(
    node: Node,
    local_accessor: str,
    strings: list[str],
    base_offset: int,
) -> float:
    """
    Evaluate a checksum expression against the current array state. Handles the arithmetic
    operators (+, -, *, /), unary negation, parentheses, parseInt calls on accessor lookups,
    and numeric literals. Raises _EvalError on any unrecognized pattern.
    """
    if isinstance(node, JsNumericLiteral):
        return float(node.value)
    if isinstance(node, JsParenthesizedExpression) and node.expression:
        return _eval_checksum(node.expression, local_accessor, strings, base_offset)
    if isinstance(node, JsUnaryExpression) and node.operator == '-' and node.operand:
        return -_eval_checksum(node.operand, local_accessor, strings, base_offset)
    if isinstance(node, JsUnaryExpression) and node.operator == '+' and node.operand:
        return _eval_checksum(node.operand, local_accessor, strings, base_offset)
    if isinstance(node, JsBinaryExpression) and node.left and node.right:
        left = _eval_checksum(node.left, local_accessor, strings, base_offset)
        right = _eval_checksum(node.right, local_accessor, strings, base_offset)
        fn = BINARY_OPS.get(node.operator)
        if fn is not None:
            if node.operator == '/' and right == 0:
                raise _EvalError
            return fn(left, right)
        raise _EvalError
    if isinstance(node, JsCallExpression) and isinstance(node.callee, JsIdentifier):
        if node.callee.name == 'parseInt' and len(node.arguments) >= 1:
            inner = node.arguments[0]
            if isinstance(inner, JsCallExpression) and isinstance(inner.callee, JsIdentifier):
                if inner.callee.name == local_accessor and len(inner.arguments) >= 1:
                    arg = inner.arguments[0]
                    if isinstance(arg, JsNumericLiteral):
                        idx = int(arg.value) - base_offset
                        if 0 <= idx < len(strings):
                            result = js_parse_int(strings[idx])
                            if result is None:
                                raise _EvalError
                            return float(result)
            raise _EvalError
    raise _EvalError


def _simulate_rotation(
    strings: list[str],
    base_offset: int,
    checksum_node: Node,
    local_accessor: str,
    target: int,
) -> list[str] | None:
    """
    Simulate the array rotation loop. For each rotation position, evaluate the checksum
    expression against the current array state. Stop when the checksum matches the target,
    or bail after len(strings) attempts.
    """
    arr = list(strings)
    n = len(arr)
    for _ in range(n):
        try:
            checksum = _eval_checksum(checksum_node, local_accessor, arr, base_offset)
            if int(checksum) == target:
                return arr
        except _EvalError:
            pass
        arr.append(arr.pop(0))
    return None


def _collect_accessor_aliases(body: Sequence[Node], accessor_name: str) -> set[str]:
    """
    Collect all variable names that are directly assigned the accessor function identifier,
    walking the entire AST. These aliases are used at the top level (e.g. var _0xcbb5cc = _0x4914)
    and inside function bodies (e.g. var _0x4bad70 = _0x4914).
    """
    aliases: set[str] = set()
    for stmt in body:
        for node in stmt.walk():
            if (
                isinstance(node, JsVariableDeclarator)
                and isinstance(node.id, JsIdentifier)
                and isinstance(node.init, JsIdentifier)
                and node.init.name == accessor_name
            ):
                aliases.add(node.id.name)
    return aliases


def _replace_accessor_calls(root: Node, aliases: set[str], lookup: dict[int, str]):
    """
    Walk the entire AST and replace accessor calls with resolved string literals.
    """
    for node in list(root.walk()):
        if not isinstance(node, JsCallExpression):
            continue
        if not isinstance(node.callee, JsIdentifier):
            continue
        if node.callee.name not in aliases:
            continue
        if len(node.arguments) < 1:
            continue
        arg = node.arguments[0]
        if not isinstance(arg, JsNumericLiteral):
            continue
        idx = int(arg.value)
        value = lookup.get(idx)
        if value is None:
            continue
        replacement = make_string_literal(value)
        _replace_in_parent(node, replacement)


class JsStringArrayResolver(Transformer):

    def visit_JsScript(self, node: JsScript):
        body = node.body
        array = _find_array_function(body)
        if array is None:
            return None
        accessor = _find_accessor_function(body, array.name)
        if accessor is None:
            return None
        rotation = _find_rotation_iife(body, array.name)
        if rotation is None:
            return None
        checksum = _extract_checksum_expression(rotation.body, accessor.name)
        if checksum is None:
            return None
        resolved = _simulate_rotation(
            array.strings,
            accessor.base_offset,
            checksum.node,
            checksum.local_accessor,
            rotation.target,
        )
        if resolved is None:
            return None
        aliases = _collect_accessor_aliases(body, accessor.name)
        aliases.add(accessor.name)
        lookup = {i + accessor.base_offset: s for i, s in enumerate(resolved)}
        _replace_accessor_calls(node, aliases, lookup)
        _remove_from_parent(array.node)
        _remove_from_parent(accessor.node)
        _remove_from_parent(rotation.node)
        self.mark_changed()
        return None

    def generic_visit(self, node: Node):
        pass
