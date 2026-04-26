"""
Shared utilities for JavaScript deobfuscation transforms.
"""
from __future__ import annotations

import operator
import re

from typing import Callable, TYPE_CHECKING

from refinery.lib.scripts import Expression, Node
from refinery.lib.scripts.js.model import (
    JsBooleanLiteral,
    JsIdentifier,
    JsNullLiteral,
    JsNumericLiteral,
    JsStringLiteral,
)
from refinery.lib.scripts.js.token import FUTURE_RESERVED, KEYWORDS

if  TYPE_CHECKING:
    from typing import TypeGuard

SIMPLE_IDENTIFIER = re.compile(r'^[a-zA-Z_$][a-zA-Z_$0-9]*$')

JS_RESERVED = frozenset(set(KEYWORDS) | FUTURE_RESERVED | {'undefined'})

BINARY_OPS: dict[str, Callable] = {
    '+'  : operator.add,
    '-'  : operator.sub,
    '*'  : operator.mul,
    '/'  : operator.truediv,
    '%'  : operator.mod,
    '**' : operator.pow,
    '|'  : operator.or_,
    '&'  : operator.and_,
    '^'  : operator.xor,
    '<<' : operator.lshift,
    '>>' : operator.rshift,
}

RELATIONAL_OPS: dict[str, Callable] = {
    '<' : operator.lt,
    '>' : operator.gt,
    '<=': operator.le,
    '>=': operator.ge,
}


def escape_js_string(value: str, quote: str = "'") -> str:
    """
    Escape a string for use in a JavaScript string literal. Returns the escaped body without
    surrounding quotes. Backslash is escaped first to avoid double-escaping.
    """
    escaped = value.replace('\\', '\\\\').replace(quote, F'\\{quote}')
    escaped = escaped.replace('\n', '\\n').replace('\r', '\\r')
    escaped = escaped.replace('\t', '\\t').replace('\0', '\\0')
    return escaped


def string_value(node: Expression | None) -> str | None:
    if isinstance(node, JsStringLiteral):
        return node.value
    return None


def make_string_literal(value: str) -> JsStringLiteral:
    escaped = escape_js_string(value)
    raw = F"'{escaped}'"
    return JsStringLiteral(value=value, raw=raw)


def numeric_value(node: Expression) -> int | float | None:
    if isinstance(node, JsNumericLiteral):
        return node.value
    return None


def make_numeric_literal(value: int | float) -> JsNumericLiteral:
    if isinstance(value, float):
        if value == int(value) and not (value == 0.0 and str(value).startswith('-')):
            raw = str(int(value))
        else:
            raw = str(value)
    else:
        raw = str(value)
    return JsNumericLiteral(value=value, raw=raw)


def is_literal(node: Node) -> TypeGuard[JsStringLiteral | JsNumericLiteral | JsBooleanLiteral | JsNullLiteral]:
    return isinstance(node, (
        JsStringLiteral, JsNumericLiteral, JsBooleanLiteral, JsNullLiteral,
    ))


def is_valid_identifier(name: str) -> bool:
    return bool(SIMPLE_IDENTIFIER.match(name)) and name not in JS_RESERVED


def is_simple_expression(node: Node) -> bool:
    """
    Check whether a node is a side-effect-free leaf expression: a literal value or an identifier.
    """
    return is_literal(node) or isinstance(node, JsIdentifier)


def is_truthy(node: Node) -> bool | None:
    """
    Return the JavaScript truthiness of a literal node, or ``None`` when the value cannot be
    determined statically.
    """
    if isinstance(node, JsBooleanLiteral):
        return node.value
    if isinstance(node, JsNumericLiteral):
        # return correct value for NaN
        return (v := node.value) != 0 and v == v
    if isinstance(node, JsStringLiteral):
        return bool(node.value)
    if isinstance(node, JsNullLiteral):
        return False
    if isinstance(node, JsIdentifier) and node.name == 'undefined':
        return False
    return None


def is_statically_evaluable(node: Node) -> bool:
    """
    Return whether the node can be evaluated to a known truthiness at transform time. This
    includes all literal types and the ``undefined`` identifier.
    """
    return is_literal(node) or (isinstance(node, JsIdentifier) and node.name == 'undefined')


def is_nullish(node: Node) -> bool:
    """
    Return whether the node is statically known to be ``null`` or ``undefined``.
    """
    if isinstance(node, JsNullLiteral):
        return True
    if isinstance(node, JsIdentifier) and node.name == 'undefined':
        return True
    return False


_HEX_ESCAPE = re.compile(r'\\x([0-9A-Fa-f]{2})|\\u00([0-9A-Fa-f]{2})')


def unescape_string_raw(raw: str) -> str | None:
    """
    Replace printable ``\\xHH`` and ``\\u00HH`` escapes in a raw string literal with their
    literal characters. Returns the rewritten raw string, or ``None`` if nothing changed.
    The quote character and backslash are never unescaped.
    """
    if len(raw) < 2:
        return None
    quote = raw[0]

    def _replace(m: re.Match) -> str:
        code = int(m.group(1) or m.group(2), 16)
        ch = chr(code)
        if 0x20 <= code <= 0x7E and ch != quote and ch != '\\':
            return ch
        return m.group(0)

    result = raw[0] + _HEX_ESCAPE.sub(_replace, raw[1:-1]) + raw[-1]
    return result if result != raw else None


_LEADING_DIGITS = re.compile(r'^[+-]?\d+')


def js_parse_int(s: str) -> int | None:
    """
    Replicate the semantics of JavaScript's parseInt(s, 10) for the subset used by the
    obfuscator's checksum: extract leading decimal digits, ignore trailing non-digit
    characters. Returns None when no leading digits are found (JS would return NaN).
    """
    m = _LEADING_DIGITS.match(s.strip())
    if m is None:
        return None
    return int(m.group())
