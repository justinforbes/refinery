"""
JavaScript syntax normalization transforms.
"""
from __future__ import annotations

from refinery.lib.scripts import Transformer
from refinery.lib.scripts.js.deobfuscation.helpers import (
    BINARY_OPS,
    RELATIONAL_OPS,
    is_literal,
    is_nullish,
    is_statically_evaluable,
    is_truthy,
    is_valid_identifier,
    make_numeric_literal,
    make_string_literal,
    numeric_value,
    string_value,
    unescape_string_raw,
)
from refinery.lib.scripts.js.model import (
    JsArrayExpression,
    JsBinaryExpression,
    JsBooleanLiteral,
    JsConditionalExpression,
    JsIdentifier,
    JsLogicalExpression,
    JsMemberExpression,
    JsNullLiteral,
    JsNumericLiteral,
    JsParenthesizedExpression,
    JsSequenceExpression,
    JsStringLiteral,
    JsUnaryExpression,
)


class JsSimplifications(Transformer):

    def visit_JsBinaryExpression(self, node: JsBinaryExpression):
        self.generic_visit(node)
        if node.left is None or node.right is None:
            return None
        op = node.operator
        left_str = string_value(node.left)
        right_str = string_value(node.right)
        if op == '+' and left_str is not None and right_str is not None:
            return make_string_literal(left_str + right_str)
        left_num = numeric_value(node.left)
        right_num = numeric_value(node.right)
        if left_num is not None and right_num is not None:
            fn = BINARY_OPS.get(op)
            if fn is not None:
                try:
                    result = fn(left_num, right_num)
                except (ZeroDivisionError, ValueError, OverflowError):
                    return None
                if isinstance(result, float) and (
                    result != result or result == float('inf') or result == float('-inf')
                ):
                    return None
                return make_numeric_literal(result)
            if op == '>>>':
                try:
                    left_i = int(left_num) & 0xFFFFFFFF
                    shift = int(right_num) & 0x1F
                    result = (left_i >> shift) & 0xFFFFFFFF
                except (ValueError, OverflowError):
                    return None
                return make_numeric_literal(result)
        if op in ('===', '!==', '==', '!='):
            equal: bool | None = None
            if left_str is not None and right_str is not None:
                equal = left_str == right_str
            elif left_num is not None and right_num is not None:
                equal = left_num == right_num
            elif (
                isinstance(node.left, JsBooleanLiteral)
                and isinstance(node.right, JsBooleanLiteral)
            ):
                equal = node.left.value == node.right.value
            elif (
                isinstance(node.left, JsNullLiteral)
                and isinstance(node.right, JsNullLiteral)
            ):
                equal = True
            if equal is not None:
                return JsBooleanLiteral(value=equal if op in ('===', '==') else not equal)
        if op in RELATIONAL_OPS:
            if left_num is not None and right_num is not None:
                return JsBooleanLiteral(value=RELATIONAL_OPS[op](left_num, right_num))
            if left_str is not None and right_str is not None:
                return JsBooleanLiteral(value=RELATIONAL_OPS[op](left_str, right_str))
        return None

    def visit_JsConditionalExpression(self, node: JsConditionalExpression):
        self.generic_visit(node)
        if node.test is None or not is_statically_evaluable(node.test):
            return None
        truthy = is_truthy(node.test)
        if truthy is None:
            return None
        return node.consequent if truthy else node.alternate

    def visit_JsParenthesizedExpression(self, node: JsParenthesizedExpression):
        self.generic_visit(node)
        inner = node.expression
        if inner is None:
            return None
        if is_literal(inner):
            return inner
        if isinstance(inner, JsSequenceExpression) and inner.expressions:
            if all(is_literal(e) for e in inner.expressions):
                return inner.expressions[-1]
        return None

    def visit_JsMemberExpression(self, node: JsMemberExpression):
        self.generic_visit(node)
        if node.computed and node.object is not None and node.property is not None:
            if (
                isinstance(node.object, JsArrayExpression)
                and isinstance(node.property, JsNumericLiteral)
            ):
                idx = node.property.value
                elements = node.object.elements
                if (
                    isinstance(idx, int) and 0 <= idx < len(elements)
                    and all(e is not None and is_literal(e) for e in elements)
                ):
                    return elements[idx]
            prop_str = string_value(node.property)
            if prop_str is not None and is_valid_identifier(prop_str):
                node.computed = False
                node.property = JsIdentifier(name=prop_str)
                self.mark_changed()
                return None
        return None

    def visit_JsUnaryExpression(self, node: JsUnaryExpression):
        self.generic_visit(node)
        if node.operand is None:
            return None
        op = node.operator
        if op == '!' and is_statically_evaluable(node.operand):
            truthy = is_truthy(node.operand)
            if truthy is not None:
                return JsBooleanLiteral(value=not truthy)
        if op == '-' and isinstance(node.operand, JsNumericLiteral):
            return make_numeric_literal(-node.operand.value)
        if op == '+' and isinstance(node.operand, JsNumericLiteral):
            return node.operand
        if op == '~' and isinstance(node.operand, JsNumericLiteral):
            try:
                v = int(node.operand.value) & 0xFFFFFFFF
                v = ~v & 0xFFFFFFFF
                if v >= 0x80000000:
                    v -= 0x100000000
                return make_numeric_literal(v)
            except (ValueError, OverflowError):
                pass
        if op == 'typeof' and is_literal(node.operand):
            if isinstance(node.operand, JsNumericLiteral):
                return make_string_literal('number')
            if isinstance(node.operand, JsStringLiteral):
                return make_string_literal('string')
            if isinstance(node.operand, JsBooleanLiteral):
                return make_string_literal('boolean')
        if op == 'void' and isinstance(node.operand, JsNumericLiteral):
            if node.operand.value == 0:
                return JsIdentifier(name='undefined')
        return None

    def visit_JsStringLiteral(self, node: JsStringLiteral):
        result = unescape_string_raw(node.raw)
        if result is not None:
            node.raw = result
            self.mark_changed()
        return None

    def visit_JsLogicalExpression(self, node: JsLogicalExpression):
        self.generic_visit(node)
        if node.left is None or node.right is None:
            return None
        if not is_statically_evaluable(node.left):
            return None
        op = node.operator
        if op == '??':
            if is_nullish(node.left):
                return node.right
            return node.left
        truthy = is_truthy(node.left)
        if truthy is None:
            return None
        if op == '&&':
            return node.right if truthy else node.left
        if op == '||':
            return node.left if truthy else node.right
        return None
