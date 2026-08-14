"""
Shared utilities for JavaScript deobfuscation transforms, and the runtime value domain they and the
interpreter agree on: what a JavaScript value is (`Value`, `JS_NULL`, `JsBuffer`), the ECMA-262
conversions between values (`to_number`, `to_string`, `to_boolean`, `js_typeof`), the operator tables
over them (`UNARY_OPS`, `BINARY_OPS`), the own-property read on them (`read_data_property`), and the
two bridges to the syntax tree (`extract_literal_value`, `value_to_node`).

That domain lives below the interpreter rather than inside it, because a static fold and an emulated
execution must answer an operator the same way — if `~NaN` were implemented once for each, only one of
the two would be wrong at a time and nothing would say which. The interpreter is a consumer here like
any transform is.
"""
from __future__ import annotations

import math
import operator
import re

from collections import Counter
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

if TYPE_CHECKING:
    from typing import TypeAlias

    from refinery.lib.scripts.js.model import JsArrowFunctionExpression as _Arrow
    from refinery.lib.scripts.js.model import JsFunctionDeclaration as _FuncDecl
    from refinery.lib.scripts.js.model import JsFunctionExpression as _FuncExpr

    LiteralValue: TypeAlias = str | int | float | bool | list | dict | None
    Value: TypeAlias = str | float | bool | list | dict | _FuncDecl | _FuncExpr | _Arrow | None

from refinery.lib.scripts import (
    Expression,
    Node,
    Statement,
    Transformer,
    _clone_node,
    _compute_children,
    _remove_from_parent,
    _replace_in_parent,
    set_body,
)
from refinery.lib.scripts.js.analysis.cache import model_cache
from refinery.lib.scripts.js.analysis.effects import side_effect_free
from refinery.lib.scripts.js.analysis.model import (
    FUNCTION_NODES,
    Binding,
    Role,
    SemanticModel,
    build_semantic_model,
    is_use_position,
    reference_role,
)
from refinery.lib.scripts.js.model import (
    JsArrayExpression,
    JsArrowFunctionExpression,
    JsAssignmentExpression,
    JsBinaryExpression,
    JsBlockStatement,
    JsBooleanLiteral,
    JsCallExpression,
    JsClassDeclaration,
    JsClassExpression,
    JsConditionalExpression,
    JsForInStatement,
    JsForOfStatement,
    JsFunctionDeclaration,
    JsFunctionExpression,
    JsIdentifier,
    JsLogicalExpression,
    JsMemberExpression,
    JsNullLiteral,
    JsNumericLiteral,
    JsObjectExpression,
    JsParenthesizedExpression,
    JsProperty,
    JsReturnStatement,
    JsScript,
    JsSequenceExpression,
    JsStaticBlock,
    JsStringLiteral,
    JsTaggedTemplateExpression,
    JsThisExpression,
    JsUnaryExpression,
    JsVariableDeclaration,
    JsVariableDeclarator,
    JsVarKind,
    JsWhileStatement,
    strip_parens,
)
from refinery.lib.scripts.js.numbers import (
    is_negative_zero,
    js_number_to_string,
    js_string_to_number,
    to_js_number,
)
from refinery.lib.scripts.js.token import FUTURE_RESERVED, KEYWORDS
from refinery.lib.scripts.js.utf16 import code_units

SIMPLE_IDENTIFIER = re.compile(r'^[a-zA-Z_$][a-zA-Z_$0-9]*$')

JS_RESERVED = frozenset(set(KEYWORDS) | FUTURE_RESERVED | {'undefined'})

GLOBAL_OBJECT_ALIASES: frozenset[str] = frozenset({'globalThis', 'global', 'window', 'self'})
VOID_LITERAL_OPERANDS = (JsNumericLiteral, JsStringLiteral, JsBooleanLiteral, JsNullLiteral)

OBJECT_PROTOTYPE_MEMBERS = frozenset({
    '__defineGetter__',
    '__defineSetter__',
    '__lookupGetter__',
    '__lookupSetter__',
    '__proto__',
    'constructor',
    'hasOwnProperty',
    'isPrototypeOf',
    'propertyIsEnumerable',
    'toLocaleString',
    'toString',
    'valueOf',
})
"""
The members every plain object inherits from `Object.prototype`. An access of one of these names on
an object that does not own it resolves through the prototype rather than to `undefined`, so a fold
that treats an absent own-property as `undefined` must leave these intact.
"""

STRING_PROTOTYPE_METHODS = frozenset({
    'anchor', 'at', 'big', 'blink', 'bold', 'charAt', 'charCodeAt', 'codePointAt', 'concat',
    'endsWith', 'fixed', 'fontcolor', 'fontsize', 'includes', 'indexOf', 'isWellFormed', 'italics',
    'lastIndexOf', 'link', 'localeCompare', 'match', 'matchAll', 'normalize', 'padEnd', 'padStart',
    'repeat', 'replace', 'replaceAll', 'search', 'slice', 'small', 'split', 'startsWith', 'strike',
    'sub', 'substr', 'substring', 'sup', 'toLocaleLowerCase', 'toLocaleUpperCase', 'toLowerCase',
    'toString', 'toUpperCase', 'toWellFormed', 'trim', 'trimEnd', 'trimLeft', 'trimRight',
    'trimStart', 'valueOf',
})

ARRAY_PROTOTYPE_METHODS = frozenset({
    'at', 'concat', 'copyWithin', 'entries', 'every', 'fill', 'filter', 'find', 'findIndex',
    'findLast', 'findLastIndex', 'flat', 'flatMap', 'forEach', 'includes', 'indexOf', 'join',
    'keys', 'lastIndexOf', 'map', 'pop', 'push', 'reduce', 'reduceRight', 'reverse', 'shift',
    'slice', 'some', 'sort', 'splice', 'toLocaleString', 'toReversed', 'toSorted', 'toSpliced',
    'toString', 'unshift', 'values', 'with',
})
"""
The callable members of `String.prototype` and `Array.prototype`, enumerated from a real engine
rather than from the subset this package implements. Reading one of these names yields the method
itself, so a reader that answers `undefined` for the ones we cannot evaluate would contradict
`typeof`; membership and evaluability are separate questions.
"""

SEQUENCE_DATA_PROPERTIES = frozenset({'length'})
"""
The non-callable inherited properties of a string or array. `length` is the only one, which is why
it must never be reached through a method registry: `'abc'.length` is the number `3` and
`'abc'.length()` is a `TypeError`, whereas a registry entry would answer `3` to both.
"""

PROTOTYPE_CHAIN_PROPERTIES = frozenset({'__proto__', 'constructor'})
"""
The two properties that expose the prototype chain itself. Both exist on every value, so answering
`undefined` for them is wrong, but modelling them would hand out the `Function` constructor that
`[].constructor.constructor('...')()` reflection depends on. They are therefore left unevaluated.
"""


class _JsNull:
    """
    Singleton sentinel for the JavaScript `null` value. The interpreter uses Python `None` for
    `undefined` (the value of missing/absent things), so a distinct object is required to keep `null`
    and `undefined` apart where JavaScript treats them differently: `Number(null)` is `0` but
    `Number(undefined)` is `NaN`, `typeof null` is `'object'`, `null === undefined` is `false`, and
    `String(null)` is `'null'`.
    """
    __slots__ = ()

    def __repr__(self) -> str:
        return 'JS_NULL'


JS_NULL = _JsNull()

GLOBAL_VALUE_NAMES: dict[str, Value] = {
    'undefined': None,
    'NaN': float('nan'),
    'Infinity': float('inf'),
}
"""
The three global names that carry a value no literal spells, so that an operand written with one of
them holds a value `extract_literal_value` cannot report and a caller that wants it looks here.

What makes reading them safe is *not* the specification. ES5 made them non-writable and
non-configurable, but ES3 did not — its own Annex E lists the change as an intentional
incompatibility — and ES3 is what Windows Script Host runs, which is the dialect of the `.js`, `.wsf`
and `.hta` droppers this tool exists for. Measured under `cscript`, JScript 11.0: `undefined =
'CLOBBERED'` sticks, and `typeof undefined` becomes `'string'` afterwards. Clobbering `undefined` is
a live obfuscation technique precisely because it breaks a naive `=== undefined` check.

What makes reading them safe is the binding analysis. A program that assigns one of these names at
top level creates an `IMPLICIT_GLOBAL` binding the model records, and `denoted_value` — the only
reader that may consult this table — refuses any name the model resolves. All three are also
ordinary identifiers as far as scoping goes: a parameter, a `let` or a `var` of the same name shadows
them, and that is the same refusal. Reading the table without asking the model is a bug.
"""

PROTO_KEY = '__proto__'
"""
The one property key whose plain spelling in an object literal does not denote an own property.
`{__proto__: v}` and `{'__proto__': v}` install `v` as the prototype and leave the object with no own
property at all, whereas `{['__proto__']: v}` — and `JSON.parse` — create an ordinary own property of
that name. A runtime object modelled as a Python dict holds only own data properties, so a
`__proto__` entry in such a dict can only have come from one of the latter two, and may only be
rendered back as the computed form.
"""


class JsBuffer(list):
    """
    Thin wrapper around `list` to distinguish a Node.js Buffer (byte array) from a plain JS Array in
    the interpreter's type-based method dispatch. It lives beside `JS_NULL` because both are members
    of the interpreter's value domain that a plain Python type cannot express, and every consumer of
    that domain — most importantly `value_to_node`, which must not render a Buffer as an array
    literal — has to be able to tell them apart.
    """
    pass


def canonical_array_index(key: str) -> int | None:
    """
    The integer index *key* denotes as an array index, or `None` when it is not one. JavaScript treats
    a property key as an index only when it is the canonical decimal spelling of a non-negative
    integer, so `'1'` indexes but `'+1'`, `'01'`, `'1.0'`, `' 1 '`, `'1_0'`, and `'0x1'` are ordinary
    property names that resolve to `undefined`. Python's `int` accepts every one of those spellings,
    and `str.isdigit` additionally accepts non-ASCII digits such as `'²'`, so neither is usable alone.
    """
    if not key or not all(c in '0123456789' for c in key):
        return None
    index = int(key)
    if str(index) != key:
        return None
    return index


class MemberRead(Enum):
    """
    What reading a key off a value found. `FOUND` carries the value the read answers with. `ABSENT`
    is an index past the end of a string or array: the value holds no such slot, so the read is
    `undefined` unless the prototype chain supplies one. `NOT_DATA` is every other key, which is a
    name the chain has to be consulted about before anything can be said.

    The two ways of not finding a value are kept apart because the callers part company on them. An
    emulated execution answers `undefined` for an index past the end, having no reason to doubt the
    chain of a value it is holding; a fold has to leave that read standing, because the file it is
    rewriting may install an index on `String.prototype` before it runs. Reporting both as one
    outcome would force the caller that cares to ask `canonical_array_index` a second time, which is
    the duplication this function exists to remove.
    """
    FOUND = auto()
    ABSENT = auto()
    NOT_DATA = auto()


def read_data_property(obj: Value, key: str) -> tuple[MemberRead, Value]:
    """
    Read *key* off *obj* as far as the value itself decides it: the own data properties of a string,
    an array, and a plain object, which are `length`, a canonical index, and a present key. Every
    other key is `NOT_DATA` — a method name, an inherited name, a missing key of an object, or any
    key at all of a value with no own slots to read, such as a number.

    The read is the *own* half of a property access and answers nothing about the prototype chain,
    which is where the outcome is decided for a `NOT_DATA` key and is why this takes no model. What
    it does answer, it answers alone: `length` and an index within range are own properties of a
    string or array, so no prototype can be consulted for them and none can shadow them.

    *obj* must hold a string the way JavaScript does, as UTF-16 code units — the form the lexer gives
    a literal and the builtin registry gives a produced string. A string of code points read here
    counts an astral character once where JavaScript counts it twice, and answers `length` and every
    index after it one too low.
    """
    if isinstance(obj, (str, list)):
        if key in SEQUENCE_DATA_PROPERTIES:
            return MemberRead.FOUND, len(obj)
        index = canonical_array_index(key)
        if index is not None:
            if 0 <= index < len(obj):
                return MemberRead.FOUND, obj[index]
            return MemberRead.ABSENT, None
    elif isinstance(obj, dict) and key in obj:
        return MemberRead.FOUND, obj[key]
    return MemberRead.NOT_DATA, None


def utf16_code_units(text: str) -> list[str]:
    """
    Split *text* into its UTF-16 code units, which is what JavaScript indexing and `split('')` operate
    on. Python strings are sequences of code points, so a character outside the BMP is one Python
    character but two JavaScript ones: `'\U0001F600'.split('')` has length 2 in JS, and each half is a
    lone surrogate. Iterating the Python string directly would under-count it.
    """
    units: list[str] = []
    for char in text:
        units.extend(code_units(ord(char)))
    return units


def _to_int32(v: int | float) -> int:
    """
    Replicate the ECMA-262 ToInt32 abstract operation: `NaN`, `+Infinity`, and `-Infinity` all
    coerce to `0`, finite floats truncate towards zero, the result is taken mod 2^32 and
    sign-extended to the int32 range.
    """
    if isinstance(v, float):
        if v != v or v == float('inf') or v == float('-inf'):
            return 0
        v = int(v) if v >= 0 else -int(-v)
    v = v & 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _to_uint32(v: int | float) -> int:
    """
    Replicate the ECMA-262 ToUint32 abstract operation.
    """
    if isinstance(v, float):
        if v != v or v == float('inf') or v == float('-inf'):
            return 0
        v = int(v) if v >= 0 else -int(-v)
    return v & 0xFFFFFFFF


def to_boolean(value: Value) -> bool:
    """
    Apply the ECMA-262 ToBoolean abstract operation. This is the value-domain counterpart of the
    AST-node `is_truthy`; the two must agree on which values are falsy (`undefined`, `null`, `0`,
    `NaN`, `''`) so that interpreted and statically-folded conditionals stay consistent.
    """
    if value is None or value is JS_NULL:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0 and value == value
    if isinstance(value, str):
        return len(value) > 0
    if isinstance(value, list):
        return True
    if isinstance(value, dict):
        return True
    if isinstance(value, (JsFunctionDeclaration, JsFunctionExpression, JsArrowFunctionExpression)):
        return True
    return False


def to_number(value: Value) -> float:
    """
    Apply the ECMA-262 ToNumber abstract operation, which is a dispatch on the type of a value. The
    string case is the only one with a grammar behind it, and that grammar is `js_string_to_number`
    in the Number domain, where the reading of a Number belongs; asking Python's `float` here
    instead would answer a different question, its own grammar being wider than the language's in
    several places at once.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return to_js_number(value)
    if isinstance(value, str):
        return js_string_to_number(value)
    if value is JS_NULL:
        return 0.0
    if isinstance(value, list):
        return to_number(to_string(value))
    return float('nan')


def to_string(value: Value) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return 'undefined'
    if value is JS_NULL:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return js_number_to_string(value)
    if isinstance(value, list):
        return ','.join(_array_element_string(v) for v in value)
    return '[object Object]'


def _array_element_string(value: Value) -> str:
    """
    Stringify an array element for `Array.prototype.toString` / `join`. JavaScript renders `null` and
    `undefined` elements as the empty string (e.g. `[1, null, 2].toString()` is `'1,,2'`), unlike a
    top-level `String(null)` which is `'null'`.
    """
    if value is None or value is JS_NULL:
        return ''
    return to_string(value)


def _to_int(value: Value) -> int:
    n = to_number(value)
    if n != n or math.isinf(n):
        return 0
    return int(n)


def js_typeof(value: Value) -> str:
    """
    Apply the `typeof` operator to a value. Total over the domain, and the reason `typeof null` is
    `'object'` falls out of the ordering rather than being stated: `JS_NULL` is not any of the
    primitive types tested for, so it reaches the same answer every object does.
    """
    if value is None:
        return 'undefined'
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, (int, float)):
        return 'number'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, (JsFunctionDeclaration, JsFunctionExpression, JsArrowFunctionExpression)):
        return 'function'
    return 'object'


def _js_div(a: int | float, b: int | float) -> int | float:
    if b == 0:
        if a == 0 or a != a:
            return float('nan')
        negative = (a < 0) != (math.copysign(1.0, b) < 0)
        return float('-inf') if negative else float('inf')
    return a / b


def _js_mul(a: int | float, b: int | float) -> int | float:
    """
    Multiply two JavaScript numbers, preserving the IEEE-754 sign of a zero product: a product of
    magnitude zero is negative zero exactly when the operands have opposite signs. Python integer
    multiplication cannot represent `-0`, so `0 * -5` would otherwise silently lose the sign.
    """
    result = a * b
    if result == 0 and (math.copysign(1.0, a) < 0) != (math.copysign(1.0, b) < 0):
        return -0.0
    return result


def _js_mod(a: int | float, b: int | float) -> int | float:
    if b == 0 or a != a or b != b:
        return float('nan')
    if a == float('inf') or a == float('-inf'):
        return float('nan')
    if b == float('inf') or b == float('-inf'):
        return a
    return math.fmod(a, b)


def _js_pow(base: int | float, exp: int | float) -> float:
    """
    Replicate JavaScript exponentiation (`**` / `Math.pow`). JavaScript numbers are IEEE-754 doubles,
    so this diverges from Python in cases that matter: `anything ** 0` is `1` (even `NaN ** 0`); a base
    of `1` or `-1` with an infinite exponent is `NaN` (Python: `1.0`); a negative base with a
    non-integer exponent is a complex number in Python (JS: `NaN`); a zero base with a negative
    exponent is `Infinity` (with the sign rule for `-0`); and a magnitude beyond the double range is
    `Infinity`, whereas Python's arbitrary-precision `int ** int` returns an exact bignum.

    An infinite exponent is decided from the base's *magnitude* alone and never from its sign, which
    is why it is answered before the negative-base rule rather than folded into it. Raising `-2` to
    `Infinity` is `Infinity` and raising `-0.5` to it is `0`, where reading an infinite exponent as
    a non-integer one would call both `NaN`.

    Both operands are coerced here and not only at `eval_binary_op`, because this is the one
    operator that answers an unbounded amount of work when handed Python integers, and it is
    reachable through `BINARY_OPS` without passing that function.
    """
    base = to_js_number(base)
    exp = to_js_number(exp)
    inf = float('inf')
    if exp == 0:
        return 1.0
    if base != base or exp != exp:
        return float('nan')
    if exp in (inf, -inf):
        magnitude = abs(base)
        if magnitude == 1:
            return float('nan')
        return inf if (magnitude > 1) == (exp == inf) else 0.0
    is_int_exp = exp == int(exp)
    if base == 0 and exp < 0:
        if is_int_exp and int(exp) % 2 != 0 and math.copysign(1.0, base) < 0:
            return -inf
        return inf
    if base < 0 and base != -inf and not is_int_exp:
        return float('nan')
    try:
        result = base ** exp
    except OverflowError:
        return -inf if (base < 0 and is_int_exp and int(exp) % 2 != 0) else inf
    except (ValueError, ZeroDivisionError):
        return float('nan')
    return result


BINARY_OPS: dict[str, Callable] = {
    '+'  : operator.add,
    '-'  : operator.sub,
    '*'  : _js_mul,
    '/'  : _js_div,
    '%'  : _js_mod,
    '**' : _js_pow,
    '|'  : lambda a, b: float(_to_int32(a) | _to_int32(b)),
    '&'  : lambda a, b: float(_to_int32(a) & _to_int32(b)),
    '^'  : lambda a, b: float(_to_int32(a) ^ _to_int32(b)),
    '<<' : lambda a, b: float(_to_int32(_to_int32(a) << (_to_int32(b) & 0x1F))),
    '>>' : lambda a, b: float(_to_int32(a) >> (_to_int32(b) & 0x1F)),
}
"""
JavaScript's binary operators over two *numbers*. Every entry assumes both operands have already been coerced,
which is why `+` is `operator.add` here: on numbers that is what `+` means, but on a value that may be a string
or an object `+` is a different operator entirely — it applies ToPrimitive to both sides and concatenates when
either is a string. A caller holding uncoerced values must not reach this table for `+`; the interpreter routes
every operator through `JsInterpreter._apply_binary`, which resolves the string cases before delegating here.
"""

UNARY_OPS: dict[str, Callable[[Value], Value]] = {
    '-'     : lambda v: -to_number(v),
    '+'     : to_number,
    '~'     : lambda v: float(_to_int32(~_to_int(v))),
    '!'     : lambda v: not to_boolean(v),
    'void'  : lambda v: None,
    'typeof': js_typeof,
}
"""
JavaScript's unary operators that are functions of their operand's *value* alone. Each is total over
the value domain — the coercions answer for every value rather than refusing any — so a caller holding
a value needs no per-operator guard, and a caller holding a syntax tree needs only to obtain the value.

`delete` is absent because it is not one of these: its result depends on the operand's *reference*
rather than its value, and evaluating it changes the object it names. A pass that wants to fold a
`delete` has to reason about that object, which is a question about the program and not about a value,
so the absence is what keeps this table from being asked it.

`void` is a member despite discarding its operand, because discarding a value is still a function of
it. The caller remains responsible for evaluating the operand: `void f()` is `undefined` and calls `f`,
and this table only supplies the first half.
"""

RELATIONAL_OPS: dict[str, Callable] = {
    '<' : operator.lt,
    '>' : operator.gt,
    '<=': operator.le,
    '>=': operator.ge,
}

LOGICAL_ASSIGNMENT_OPS = frozenset({'&&=', '||=', '??='})
"""
The assignment operators that short-circuit. Unlike every other compound assignment, these evaluate their
right operand only when the target's existing value does not already decide the result, and perform no store
when it does — a distinction JavaScript makes observable through a setter or a frozen object. They therefore
have no entry in `BINARY_OPS`, whose members are total functions of both operands.
"""


def eval_binary_op(op: str, left: float, right: float) -> float | bool | None:
    """
    Evaluate a JavaScript binary operator on two numeric operands. Returns the result value, or
    `None` when the operator is unknown or the computation overflows/divides by zero. Handles
    arithmetic, bitwise, relational, equality, and the unsigned right shift `>>>`.

    Both operands are coerced here so that every operator below sees the Number the caller meant and
    not whichever Python type happened to carry it. That is a normalization and not the bound on the
    work: `refinery.lib.scripts.js.deobfuscation.stringarray` reaches `BINARY_OPS` without passing
    through this function, so the operator that can be asked for unbounded work — `**`, where Python
    integers build a number of half a billion digits for what a double answers in one operation —
    coerces for itself as well.
    """
    left = to_js_number(left)
    right = to_js_number(right)
    if op in ('===', '=='):
        return left == right
    if op in ('!==', '!='):
        return left != right
    rel = RELATIONAL_OPS.get(op)
    if rel is not None:
        return rel(left, right)
    if op == '>>>':
        a = _to_uint32(left)
        b = _to_uint32(right) & 0x1F
        return float((a >> b) & 0xFFFFFFFF)
    fn = BINARY_OPS.get(op)
    if fn is None:
        return None
    try:
        return fn(left, right)
    except (ZeroDivisionError, OverflowError, ValueError):
        return None


_SURROGATE_PAIR = re.compile('[\ud800-\udbff][\udc00-\udfff]')


def _escape_residue(m: re.Match[str]):
    cp = ord(m.group())
    if cp > 0xFF:
        return F'\\u{cp:04X}'
    return F'\\x{cp:02x}'


def _combine_surrogate_pair(m: re.Match[str]) -> str:
    high, low = m.group()
    return chr(0x10000 + ((ord(high) - 0xD800) << 10) + (ord(low) - 0xDC00))


def spell_astral_characters(value: str) -> str:
    """
    Write the pairs of code units that name a character above the basic plane as that character. A
    value is held as the code units a JavaScript string is made of, which is what a program asking
    about its length or its halves has to be answered from; but a file is written in characters, so
    printing the units back would spell an emoji as two escapes nobody wrote.

    A surrogate standing alone is left alone. It names no character, so there is nothing to write it
    as, and an escape is the only spelling a file has for it.
    """
    return _SURROGATE_PAIR.sub(_combine_surrogate_pair, value)


def code_points(value: str) -> list[str]:
    """
    The code points of a string, each kept as the code units that spell it. A JavaScript string is
    indexed by code unit but iterated by code point — a `for ... of`, a spread, `Array.from` — so a
    well-formed surrogate pair is one element here, and every other code unit is its own. A lone
    surrogate stands as a code point of its own, because a string may hold one and the split may not
    invent the partner it lacks.
    """
    points: list[str] = []
    index = 0
    length = len(value)
    while index < length:
        step = 2 if _SURROGATE_PAIR.match(value, index) else 1
        points.append(value[index:index + step])
        index += step
    return points


def escape_js_string(value: str, quote: str = "'") -> str:
    """
    Escape a string for use in a JavaScript string literal. Returns the escaped body without
    surrounding quotes. Backslash is escaped first to avoid double-escaping. Control characters
    not covered by named escapes are emitted as `\\xHH`; a surrogate that names no character on its
    own as `\\uXXXX`, and a pair of them as the character they name.

    A NUL is `\\0`, except where a digit stands behind it: `\\0` followed by `0` through `7` is one
    legacy octal escape and would swallow the digit into a different character, and followed by `8`
    or `9` it is an escape strict code refuses. A NUL a digit follows is spelled `\\x00` instead, so
    the character behind it stays the character the value held.
    """
    value = spell_astral_characters(value)
    value = value.replace('\\', r'\\')
    value = value.replace('\n', r'\n')
    value = value.replace('\r', r'\r')
    value = value.replace('\t', r'\t')
    value = re.sub(r'\x00(?=[0-9])', r'\\x00', value)
    value = value.replace('\0', r'\0')
    value = value.replace(quote, F'\\{quote}')
    return re.sub(r'[\x01-\x1f\ud800-\udfff]', _escape_residue, value)


def escape_js_template_text(value: str) -> str:
    """
    Escape a string so that it spells itself inside a template literal. Three characters end a run
    of template text rather than standing in it — the backtick that closes the literal, the `${`
    that opens a hole, and the backslash that would eat what follows it.

    A line feed stands as itself, because a template is the one literal that may span lines. A
    carriage return does not: every line terminator sequence a template is written with denotes a
    line feed, so a return written into the text would come back as one and the string would not
    be the string. A lone surrogate has to be spelled too, for the same reason a string spells one
    — there is no encoding of the file that carries it.
    """
    value = spell_astral_characters(value)
    value = value.replace('\\', r'\\')
    value = value.replace('\r', r'\r')
    value = value.replace('`', r'\`')
    value = value.replace('${', r'\${')
    return re.sub(r'[\x00-\x08\x0b-\x1f\ud800-\udfff]', _escape_residue, value)


def string_value(node: Expression | None) -> str | None:
    """
    The text a literal denotes, where it is a literal that denotes one. A literal the source never
    closed is not, and answering with the text it would have denoted is how a fold repairs it: the
    text goes into a fresh literal that carries the closing quote nobody wrote, and a file that no
    engine reads comes back as a program that runs.
    """
    if isinstance(node, JsStringLiteral) and node.terminated:
        return node.value
    return None


def property_key(prop: JsProperty) -> str | None:
    """
    Extract the string key from a property node. Handles both string-literal keys and plain
    identifier keys. Returns `None` for computed keys.
    """
    if prop.computed:
        return None
    if isinstance(prop.key, JsStringLiteral):
        return prop.key.value
    if isinstance(prop.key, JsIdentifier):
        return prop.key.name
    return None


def access_key(node: JsMemberExpression) -> str | None:
    """
    Extract the string key from a member-access expression. Handles both computed (`obj['key']`)
    and dot (`obj.key`) accesses.
    """
    if node.computed:
        return string_value(node.property)
    if isinstance(node.property, JsIdentifier):
        return node.property.name
    return None


def make_string_literal(value: str) -> JsStringLiteral:
    escaped = escape_js_string(value)
    raw = F"'{escaped}'"
    return JsStringLiteral(value=value, raw=raw)


def numeric_value(node: Expression) -> float | None:
    if isinstance(node, JsNumericLiteral):
        return node.value
    return None


def make_numeric_literal(value: int | float) -> JsNumericLiteral | None:
    """
    Spell a Number as a literal, or refuse with `None` when it has none. `NaN` is the only Number
    without one, and a caller that can produce it must spell it through `value_to_node`.

    The infinities do have one. ECMA-262 defines the mathematical value of a decimal literal and then
    rounds it to the nearest Number, and a value too large to round to a finite one rounds to the
    infinity — so `1e999` is a numeric literal denoting `+Infinity` exactly as `1` is one denoting
    one. That matters because the alternative spelling, the identifier `Infinity`, is an ordinary
    global binding that the program being deobfuscated may have rebound, whereas a literal denotes
    its value in every scope.

    The spelling is otherwise `Number.prototype.toString`, with one deliberate deviation: that
    algorithm reads the mathematical value, so it prints negative zero as `0`, but a literal `0`
    denotes *positive* zero and the two are distinguishable — `1 / -0` is `-Infinity`. Negative zero
    is therefore spelled `-0`. That spelling, like `-1e999` and the one every other negative value
    gets, is a negation applied to a literal rather than a literal, so the node binds like the unary
    operator it starts with. That is a fact about the spelling, which
    `refinery.lib.scripts.js.precedence` therefore reads from the `raw` rather than from the class.
    """
    value = to_js_number(value)
    if value != value:
        return None
    if value == float('inf'):
        return JsNumericLiteral(value=value, raw='1e999')
    if value == float('-inf'):
        return JsNumericLiteral(value=value, raw='-1e999')
    if is_negative_zero(value):
        return JsNumericLiteral(value=value, raw='-0')
    return JsNumericLiteral(value=value, raw=js_number_to_string(value))


def make_undefined_expression() -> JsUnaryExpression:
    """
    The expression that spells `undefined`. That value has no literal, and the global name that
    denotes it is an ordinary binding which any scope may rebind, so it is written as an operator
    applied to a literal instead: `void 0` denotes it wherever it stands.
    """
    return JsUnaryExpression(operator='void', operand=JsNumericLiteral(value=0, raw='0'))


def make_nan_expression() -> JsBinaryExpression:
    """
    The expression that spells `NaN`, which has no literal either, for the same reason and by the
    same means: `0 / 0`.
    """
    return JsBinaryExpression(
        operator='/',
        left=JsNumericLiteral(value=0, raw='0'),
        right=JsNumericLiteral(value=0, raw='0'),
    )


def denotes_nan(node: Node) -> bool:
    """
    Whether *node* is the expression `make_nan_expression` builds. A zero literal divided by a zero
    literal is `NaN` however either zero happens to be written, so the test reads the two values
    rather than the text.
    """
    return (
        isinstance(node, JsBinaryExpression)
        and node.operator == '/'
        and isinstance(node.left, JsNumericLiteral)
        and isinstance(node.right, JsNumericLiteral)
        and node.left.value == 0
        and node.right.value == 0
    )


def extract_literal_value(node: Node) -> tuple[bool, LiteralValue]:
    """
    Extract a Python value from a literal AST node. Returns `(True, value)` on success or
    `(False, None)` when the node is not a recognized literal form. Handles string, numeric,
    boolean, null literals, `void expr`, negative numerics, `!0`/`!1`, `0 / 0`, and array
    expressions where all elements are themselves literals.

    The two forms that are operator expressions rather than literals, `void 0` and `0 / 0`, are here
    because they are what `undefined` and `NaN` have instead of a literal: an expression built from
    an operator, which no scope can rebind, rather than from one of the global names, which any
    scope can. Recognizing them is what lets those two values survive a round trip through the tree,
    and this must stay paired with `value_to_node`, its declared inverse.
    """
    if isinstance(node, JsStringLiteral):
        return (True, node.value) if node.terminated else (False, None)
    if isinstance(node, JsNumericLiteral):
        return True, node.value
    if isinstance(node, JsBooleanLiteral):
        return True, node.value
    if isinstance(node, JsNullLiteral):
        return True, JS_NULL
    if isinstance(node, JsUnaryExpression):
        if node.operator == 'void' and isinstance(node.operand, VOID_LITERAL_OPERANDS):
            return True, None
        if node.operator == '-' and isinstance(node.operand, JsNumericLiteral):
            return True, -node.operand.value
        if node.operator == '+' and isinstance(node.operand, JsNumericLiteral):
            return True, node.operand.value
        if node.operator == '!' and isinstance(node.operand, JsNumericLiteral):
            return True, not bool(node.operand.value)
    if denotes_nan(node):
        return True, float('nan')
    if isinstance(node, JsArrayExpression):
        items: list[LiteralValue] = []
        for el in node.elements:
            if el is None:
                return False, None
            ok, val = extract_literal_value(el)
            if not ok:
                return False, None
            items.append(val)
        return True, items
    return False, None


def value_to_node(value: object) -> Expression | None:
    """
    Convert a Python value to the corresponding AST literal node, or `None` when the value has no
    literal form that denotes it faithfully. Refusing is always sound — the caller leaves the original
    expression in place — whereas rendering an approximation silently changes what the program means,
    so every case here either round-trips exactly or returns `None`.

    A number is spelled by `make_numeric_literal` whatever its sign, so that a negative one is a single
    literal carrying its sign in the `raw` and not a negation applied to its magnitude. Those two nodes
    synthesize to the same text, which is what let the second spelling go unnoticed, but only the first
    is a `JsNumericLiteral`: the fold that reads an operand with `numeric_value` sees a number in one
    and nothing in the other.

    `NaN` and `undefined` are the only values this returns a compound node for, because they are the
    only ones no literal denotes. Neither is spelled with the global name that names it. Those names
    are ordinary bindings, and this function does not know the scope it is writing into: a fold that
    happens under `function (NaN) { … }` would otherwise emit text meaning the parameter.
    """
    if isinstance(value, str):
        return make_string_literal(value)
    if isinstance(value, bool):
        return JsBooleanLiteral(value=value)
    if isinstance(value, (int, float)):
        number = to_js_number(value)
        if number != number:
            return make_nan_expression()
        return make_numeric_literal(number)
    if isinstance(value, JsBuffer):
        return None
    if isinstance(value, list):
        elements: list[Expression | None] = []
        for item in value:
            el = value_to_node(item)
            if el is None:
                return None
            elements.append(el)
        return JsArrayExpression(elements=elements)
    if isinstance(value, dict):
        properties = []
        for k, v in value.items():
            if not isinstance(k, str):
                return None
            val_node = value_to_node(v)
            if val_node is None:
                return None
            properties.append(JsProperty(
                key=make_string_literal(k),
                value=val_node,
                computed=k == PROTO_KEY,
            ))
        return JsObjectExpression(properties=properties)
    if value is JS_NULL:
        return JsNullLiteral()
    if value is None:
        return make_undefined_expression()
    return None


def is_literal(node: Node) -> bool:
    """
    Whether *node* is a constant expression whose value the tree carries in full — the test a pass
    applies before cloning it to another position. `void 0` and `0 / 0` count for the same reason
    `extract_literal_value` reads them: they are what `undefined` and `NaN` have instead of a
    literal, and an operator applied to literals is as constant as a literal is.
    """
    if isinstance(node, JsStringLiteral):
        return node.terminated
    if isinstance(node, (JsNumericLiteral, JsBooleanLiteral, JsNullLiteral)):
        return True
    if isinstance(node, JsUnaryExpression):
        if node.operator == 'void' and isinstance(node.operand, VOID_LITERAL_OPERANDS):
            return True
        if node.operator == '-' and isinstance(node.operand, JsNumericLiteral):
            return True
    return denotes_nan(node)


def member_key(node: JsMemberExpression) -> str | None:
    """
    Flatten a chain of property accesses into a dot-separated key string. Handles both dot
    notation and computed access with string-literal keys. Returns `None` if the chain contains
    a dynamic computed access that cannot be resolved to a static key.
    """
    parts: list[str] = []
    cursor: Expression | None = node
    while isinstance(cursor, JsMemberExpression):
        key = access_key(cursor)
        if key is None:
            return None
        parts.append(key)
        cursor = cursor.object
    if not isinstance(cursor, JsIdentifier):
        return None
    parts.append(cursor.name)
    parts.reverse()
    return '.'.join(parts)


def is_while_true(node: JsWhileStatement) -> bool:
    """
    Check whether the while-loop condition is `true`, `!![]`, or `!0` — the forms the
    obfuscator uses for infinite loops.
    """
    test = node.test
    if isinstance(test, JsBooleanLiteral) and test.value is True:
        return True
    if not isinstance(test, JsUnaryExpression) or test.operator != '!':
        return False
    inner = test.operand
    if isinstance(inner, JsNumericLiteral) and inner.value == 0:
        return True
    if isinstance(inner, JsUnaryExpression) and inner.operator == '!':
        return True
    return False


def is_valid_identifier(name: str) -> bool:
    return bool(SIMPLE_IDENTIFIER.match(name)) and name not in JS_RESERVED


def is_valid_property_key(name: str) -> bool:
    return bool(SIMPLE_IDENTIFIER.match(name))


def is_simple_expression(node: Node) -> bool:
    """
    Check whether a node is a side-effect-free leaf expression: a literal value, an identifier, or
    a unary operator applied to a literal (e.g. `-42`).
    """
    if is_literal(node) or isinstance(node, JsIdentifier):
        return True
    if isinstance(node, JsUnaryExpression) and node.operand is not None:
        return is_literal(node.operand)
    return False


def is_write_target(node: JsIdentifier) -> bool:
    """
    Return whether this identifier is a write target: the left-hand side of an assignment
    expression, or the iteration variable of a `for-in` / `for-of` statement.
    """
    p = node.parent
    if isinstance(p, JsAssignmentExpression) and p.left is node:
        return True
    if isinstance(p, (JsForInStatement, JsForOfStatement)) and p.left is node:
        return True
    return False


def is_binding_site(node: JsIdentifier) -> bool:
    """
    Return whether this identifier is in a binding position (variable declarator id or function
    declaration name) rather than a reference/read position.
    """
    p = node.parent
    if isinstance(p, JsVariableDeclarator) and p.id is node:
        return True
    if isinstance(p, JsFunctionDeclaration) and p.id is node:
        return True
    return False


def is_reference(node: JsIdentifier) -> bool:
    """
    Return whether this identifier is in a true variable reference position: not a binding site,
    not a non-computed member property, and not a non-computed object-literal key.
    """
    p = node.parent
    if p is None:
        return False
    if isinstance(p, JsVariableDeclarator) and p.id is node:
        return False
    if isinstance(p, JsFunctionDeclaration) and p.id is node:
        return False
    if isinstance(p, JsMemberExpression) and p.property is node and not p.computed:
        return False
    if isinstance(p, JsProperty) and p.key is node and not p.computed:
        return False
    return True


def name_is_unbound(node: JsIdentifier, model: SemanticModel) -> bool:
    """
    Whether nothing in the program can have given *node*'s name a meaning of its own, so that it still
    denotes whatever the host supplies under that name. This is the question behind every table of
    well-known names the tool keeps — the values in `GLOBAL_VALUE_NAMES`, the built-ins in
    `BUILTIN_REGISTRY` — because what such a table records is a fact about the *host*, and whether the
    name still reaches the host is a fact about the *scope*.

    There are three ways a name can mean something else, and `resolve` reports only the first:

    - a declaration binds it, anywhere from a parameter or a `catch` clause to an assignment at top
      level, which the model records as an `IMPLICIT_GLOBAL`
    - the lookup crosses a `with` body, where the object may carry a property of that name and reading
      it may even run a getter. `resolve` answers `None` here as well, so `read_has_dynamic_effect` is
      what separates the two cases
    - a direct `eval` declared it, which no reference records at all;
      `free_name_reachable_by_direct_eval` reports the positions that could see such a binding
    """
    return (
        model.resolve(node) is None
        and not model.read_has_dynamic_effect(node)
        and not model.free_name_reachable_by_direct_eval(node)
    )


def names_global_value(node: JsIdentifier, model: SemanticModel) -> bool:
    """
    Whether *node* is one of `GLOBAL_VALUE_NAMES` still denoting its value. Every reader of those
    names has to come through here.
    """
    return node.name in GLOBAL_VALUE_NAMES and name_is_unbound(node, model)


def denoted_value(node: Node | None, model: SemanticModel) -> tuple[bool, Value]:
    """
    The value *node* denotes, as `(True, value)`, or `(False, None)` when nothing decides it. This is
    `extract_literal_value` widened by the two things a literal cannot express: a name that still
    denotes one of the global values, and an operator standing in front of either.
    """
    node = strip_parens(node)
    if node is None:
        return False, None
    if isinstance(node, JsIdentifier):
        if not names_global_value(node, model):
            return False, None
        return True, GLOBAL_VALUE_NAMES[node.name]
    if isinstance(node, JsUnaryExpression):
        apply = UNARY_OPS.get(node.operator)
        if apply is None:
            return False, None
        known, value = denoted_value(node.operand, model)
        return (True, apply(value)) if known else (False, None)
    return extract_literal_value(node)


def allocated_object_type(node: Node | None) -> str | None:
    """
    The `typeof` of the object *node* allocates, when it is a form that always evaluates to a freshly
    created one — `None` otherwise. Such a node has no value this module can extract: an object or
    function expression denotes an identity no literal reproduces, and an array whose elements are
    not themselves literals is the same. Its *type* is nevertheless fixed by the syntax alone, and so
    is its truthiness, since every object is truthy — which is why `!{}`, `typeof {}` and
    `if ([f()])` are all answerable from this one fact.

    Deciding an operand from its allocation says nothing about whether evaluating it is free of
    effects; `[f()]` allocates an array and calls `f`. A caller that discards the operand has to ask
    that separately.
    """
    node = strip_parens(node)
    if isinstance(node, (JsObjectExpression, JsArrayExpression)):
        return 'object'
    if isinstance(node, (JsFunctionExpression, JsArrowFunctionExpression, JsClassExpression)):
        return 'function'
    return None


def is_truthy(node: Node, model: SemanticModel) -> bool | None:
    """
    The JavaScript truthiness of *node*, or `None` when nothing decides it. The AST-node counterpart
    of the value-domain `to_boolean`, which it answers by asking wherever a value is known: the two
    used to agree by inspection, and now agree by construction.

    An allocation is the one case with no value to ask about, and it needs none — every object is
    truthy. This does not gate that on the allocation being effect-free, because deciding truthiness
    does not by itself discard the operand; the caller that goes on to drop it is the one that has to
    keep its effects.
    """
    known, value = denoted_value(node, model)
    if known:
        return to_boolean(value)
    return True if allocated_object_type(node) is not None else None


def is_nullish(node: Node, model: SemanticModel) -> bool | None:
    """
    Whether *node* denotes `null` or `undefined` — the two values `??` treats as absent — or `None`
    when the value it denotes is not decided. A caller has to tell that third answer from `False`:
    `a ?? b` keeps `a` when `a` is known not to be nullish, and must be left alone when nothing is
    known about it at all.
    """
    known, value = denoted_value(node, model)
    if not known:
        return None
    return value is None or value is JS_NULL


def get_body(node: Node) -> list[Statement] | None:
    """
    Return the statement body list of a node if it has one (JsScript or JsBlockStatement).
    """
    if isinstance(node, (JsScript, JsBlockStatement)):
        return node.body
    return None


def remove_declarator(declarator: JsVariableDeclarator) -> None:
    """
    Remove a `refinery.lib.scripts.js.model.JsVariableDeclarator` from its parent
    `refinery.lib.scripts.js.model.JsVariableDeclaration`. If the declaration has no remaining
    declarators afterward, remove it from the body as well.
    """
    var_decl = declarator.parent
    _remove_from_parent(declarator)
    if isinstance(var_decl, JsVariableDeclaration) and not var_decl.declarations:
        _remove_from_parent(var_decl)


def extract_identifier_params(params: list) -> list[str] | None:
    """
    Extract plain identifier names from a function's parameter list. Returns `None` if any parameter
    is not a simple `refinery.lib.scripts.js.model.JsIdentifier` (e.g. destructuring or rest
    patterns).
    """
    names: list[str] = []
    for p in params:
        if not isinstance(p, JsIdentifier):
            return None
        names.append(p.name)
    return names


def is_closed_expression(node: Node, allowed_names: set[str]) -> bool:
    """
    Check whether every leaf in the expression tree is either a literal or an identifier whose
    name is in *allowed_names*. This ensures the expression has no free variables.
    """
    children = list(node.children())
    if not children:
        if isinstance(node, JsIdentifier):
            return node.name in allowed_names
        return is_simple_expression(node)
    return all(is_closed_expression(child, allowed_names) for child in children)


def _collect_unconditional_identifiers(expr: Node) -> list[str]:
    """
    Walk *expr* in evaluation order, descending only into children that are unconditionally
    evaluated (not short-circuit branches or ternary arms). Return the identifier names encountered
    in evaluation order.
    """
    names: list[str] = []
    stack: list[Node] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, JsIdentifier):
            names.append(node.name)
            continue
        if isinstance(node, (JsBinaryExpression, JsAssignmentExpression)):
            children: list[Node] = [c for c in (node.left, node.right) if c is not None]
        elif isinstance(node, JsUnaryExpression):
            children = [node.operand] if node.operand is not None else []
        elif isinstance(node, JsLogicalExpression):
            children = [node.left] if node.left is not None else []
        elif isinstance(node, JsConditionalExpression):
            children = [node.test] if node.test is not None else []
        elif isinstance(node, JsSequenceExpression):
            children = list(node.expressions)
        elif isinstance(node, JsMemberExpression):
            children = [node.object] if node.object is not None else []
            if node.computed and node.property is not None:
                children.append(node.property)
        else:
            continue
        for child in reversed(children):
            stack.append(child)
    return names


def _param_written(expr: Node, param_names: set[str]) -> bool:
    """
    Whether any of *param_names* occurs at a write position — an assignment, compound-assignment, or
    update target — within *expr*. Such a parameter is not read-only, so substituting the call
    argument for it would place the argument at a write target: assigning to a value (`(11 = 'x')`)
    or, for an lvalue argument, mutating the caller's binding. A wrapper with a written parameter is
    therefore not a pure function of its arguments and must not be inlined by substitution.
    """
    return any(
        isinstance(node, JsIdentifier)
        and node.name in param_names
        and reference_role(node) is not Role.READ
        for node in expr.walk()
    )


def is_safe_iife_inline(
    expr: Node,
    param_names: Sequence[str],
    call_args: Sequence[Node],
    call_pure: Callable[..., bool] | None = None,
    read_effect: Callable[[Node], bool] | None = None,
    call_established: Callable[..., bool] | None = None,
) -> bool:
    """
    Verify that substituting IIFE arguments into the body expression preserves evaluation semantics.
    An argument used more than once must be a simple, identity-stable expression — a literal or a bare
    identifier: duplicating a fresh array/object/function literal (or a call) would split one value into
    distinct copies and break an identity comparison such as `x === x`. An effectful argument must
    additionally be used exactly once, in an unconditionally-evaluated position, and in declaration
    order relative to other effectful arguments, so its side effect is neither dropped, duplicated, nor
    reordered. When *call_pure* is given (an
    `refinery.lib.scripts.js.analysis.effects.EffectModel.is_pure_call`), a call argument it proves pure
    counts as side-effect-free for the ordering rules — but only when *call_established* also certifies
    its callee is in place before the call runs, and, being a call, it is not simple, so it is still not
    duplicated. When *read_effect* is given (a
    `refinery.lib.scripts.js.analysis.model.SemanticModel.read_has_dynamic_effect`), an argument reading
    a bare name through a `with` body's dynamic scope counts as effectful — the read may fire the `with`
    object's getter or throw — so it too must not be dropped or reordered.

    Identity stability under duplication is the *may*-allocate direction and must not be merged with
    `EffectModel._fresh_kind`, which is *must*-allocate: a fresh literal is the thing this refuses to
    duplicate and the thing that predicate admits. The two agree on the syntax and disagree on the verdict,
    which is exactly why sharing one predicate between them would be wrong.
    """
    if _param_written(expr, set(param_names)):
        return False
    use_counts = Counter(
        n.name for n in expr.walk()
        if isinstance(n, JsIdentifier) and is_use_position(n)
    )
    for i, arg in enumerate(call_args):
        if use_counts[param_names[i]] > 1 and not is_simple_expression(arg):
            return False
    effectful_indices = [
        i for i, arg in enumerate(call_args)
        if not side_effect_free(
            arg, call_pure=call_pure, read_effect=read_effect, call_established=call_established,
        )
    ]
    if not effectful_indices:
        return True
    for i in effectful_indices:
        if use_counts[param_names[i]] != 1:
            return False
    unconditional = _collect_unconditional_identifiers(expr)
    effectful_names = {param_names[i] for i in effectful_indices}
    effectful_in_eval = [n for n in unconditional if n in effectful_names]
    if len(effectful_in_eval) != len(effectful_indices):
        return False
    param_order = {name: i for i, name in enumerate(param_names)}
    prev = -1
    for name in effectful_in_eval:
        idx = param_order[name]
        if idx <= prev:
            return False
        prev = idx
    return True


def substitute_params(
    expression: Node,
    params: Sequence[Node],
    arguments: Sequence[Node],
    transformer: Transformer | None = None,
) -> Node:
    """
    Deep-clone *expression* and replace every reference to one of the function parameters *params* with
    a clone of the positionally corresponding node from *arguments*. Only identifiers the parameter
    actually binds are replaced: a non-computed property key (the `a` in `b.a`) names a property, and a
    function or class nested in *expression* that reintroduces a parameter's name keeps its own
    identifiers rather than the outer parameter's. When *expression* nests no scope, no name under it
    can be rebound, so a parameter's references are exactly the use-position identifiers carrying its
    name and are substituted directly; only when it does nest a scope is a semantic model built to
    resolve each occurrence against the binding it reads. When *transformer* is given, that model is
    taken from its shared analysis cache; otherwise it is built standalone.
    """
    cloned = _clone_node(expression)
    mapping = {
        param.name: argument
        for param, argument in zip(params, arguments)
        if isinstance(param, JsIdentifier)
    }
    if isinstance(expression, JsIdentifier):
        if expression.name in mapping and is_use_position(expression):
            return _clone_node(mapping[expression.name])
        return cloned
    if not _introduces_nested_scope(expression):
        for node in list(cloned.walk()):
            if isinstance(node, JsIdentifier) and node.name in mapping and is_use_position(node):
                _substitute_use_position(node, _clone_node(mapping[node.name]))
        return cloned
    root = expression
    while root.parent is not None:
        root = root.parent
    assert isinstance(root, JsScript)
    if transformer is None:
        model = build_semantic_model(root)
    else:
        model = model_cache(transformer, root).model
    bindings = {
        param.name: model.binding_of(param)
        for param in params
        if isinstance(param, JsIdentifier)
    }
    for original, clone in zip(list(expression.walk()), list(cloned.walk())):
        if not isinstance(original, JsIdentifier) or original.name not in mapping:
            continue
        binding = bindings.get(original.name)
        if binding is None or model.resolve(original) is not binding:
            continue
        if isinstance(clone, JsIdentifier) and clone.name == original.name:
            _substitute_use_position(clone, _clone_node(mapping[original.name]))
    return cloned


def _introduces_nested_scope(node: Node) -> bool:
    """
    Whether the subtree at *node* contains a function or class — a scope in which an enclosing
    function's parameter name could be rebound. When it does not, no identifier under *node* can shadow
    such a parameter, so the parameter's references are exactly the use-position identifiers that carry
    its name.
    """
    return any(
        isinstance(child, (
            JsFunctionExpression,
            JsArrowFunctionExpression,
            JsFunctionDeclaration,
            JsClassExpression,
            JsClassDeclaration,
        ))
        for child in node.walk()
    )


def _substitute_use_position(node: JsIdentifier, replacement: Node) -> None:
    """
    Replace use-position identifier *node* with *replacement*. In an object-literal shorthand (`{a}`),
    one identifier serves as both the property name and the read of the variable, so a plain replacement
    would rename the property — or emit invalid syntax for a non-identifier argument. Keep the name in
    that case: never substitute a non-computed property key, and clear the shorthand flag when replacing
    its value so the property is written out in full. Guarding the key makes the result independent of
    which of the two cloned occurrences is visited first.
    """
    parent = node.parent
    if isinstance(parent, JsProperty) and not parent.computed:
        if parent.key is node:
            return
        if parent.shorthand and parent.value is node:
            parent.shorthand = False
    _replace_in_parent(node, replacement)


def try_inline_trivial_function(
    func: JsFunctionExpression,
    call_args: list,
    *,
    relaxed: bool = False,
    transformer: Transformer | None = None,
) -> Node | None:
    """
    If *func* is a trivial wrapper (single return whose expression uses only the function's
    parameters), substitute call-site arguments into a clone of the return expression. Returns the
    inlined expression or `None` if the function is not a simple wrapper.

    When *relaxed* is False (default), all arguments must be side-effect-free simple expressions.
    When *relaxed* is True, only arguments used more than once in the return expression need to be
    simple (prevents duplicating side effects while allowing complex single-use arguments).

    An async or generator function is never inlined: calling it produces a promise or an iterator, not
    the bare value of its return expression, so substituting the expression in for the call would drop
    that wrapping and change the value's type.
    """
    if func.is_async or func.generator:
        return None
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
    if _param_written(expr, set(param_names)):
        return None
    if relaxed:
        for i, name in enumerate(param_names):
            uses = sum(
                1 for n in expr.walk()
                if isinstance(n, JsIdentifier) and n.name == name and is_use_position(n)
            )
            if uses > 1 and not is_simple_expression(call_args[i]):
                return None
    return substitute_params(expr, func.params, call_args, transformer=transformer)


def walk_scope(root: Node, *, include_root_body: bool = False) -> Iterator[Node]:
    """
    Walk the AST under *root* without descending into nested function bodies. Function boundary
    nodes are yielded (so their identifiers can be inspected) but their subtrees are suppressed.
    Children are visited in source order.

    When *include_root_body* is True and *root* is itself a function, its body IS traversed (only
    inner functions are skipped). This is useful when *root* represents the scope being analyzed.
    """
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, FUNCTION_NODES):
            if not (include_root_body and node is root):
                continue
        cc = _compute_children(node)
        stack.extend(reversed(cc))


def collect_identifier_names(node: Node) -> set[str]:
    """
    Collect the names of all `refinery.lib.scripts.js.model.JsIdentifier` nodes in the subtree
    rooted at *node*.
    """
    return {n.name for n in node.walk() if isinstance(n, JsIdentifier)}


def find_enclosing_body(node: Node) -> list[Statement] | None:
    """
    Walk up parent pointers from *node* to find the body list that directly contains it. Returns the
    `body` attribute of the nearest `refinery.lib.scripts.js.model.JsBlockStatement` or
    `refinery.lib.scripts.js.model.JsScript` ancestor whose body list includes *node* (or an
    ancestor of *node*).
    """
    child = node
    parent = node.parent
    while parent is not None:
        if isinstance(parent, (JsBlockStatement, JsScript)):
            if child in parent.body:
                return parent.body
        child = parent
        parent = parent.parent
    return None


def function_binds_name(func: Node, name: str) -> bool:
    """
    Check if a function creates a local binding for `name` (parameter, function name, or var
    declaration anywhere in its body — excluding nested functions).
    """
    if isinstance(func, JsFunctionDeclaration) and func.id is not None and func.id.name == name:
        return True
    for p in (getattr(func, 'params', None) or []):
        if isinstance(p, JsIdentifier) and p.name == name:
            return True
    body = getattr(func, 'body', None)
    if not isinstance(body, JsBlockStatement):
        return False
    stack: list[Node] = [body]
    while stack:
        node = stack.pop()
        if isinstance(node, FUNCTION_NODES):
            continue
        if isinstance(node, JsVariableDeclaration) and node.kind == JsVarKind.VAR:
            for decl in node.declarations:
                if isinstance(decl, JsVariableDeclarator) and isinstance(decl.id, JsIdentifier):
                    if decl.id.name == name:
                        return True
        for child in node.children():
            stack.append(child)
    return False


def walk_receiver_scope(root: Node) -> Iterator[Node]:
    """
    Yield every node in the subtree at *root* that shares *root*'s `this`/`super` receiver, without
    descending into a nested regular or generator function, which rebinds `this`. Arrow functions are
    descended, since they inherit the receiver lexically. A class rebinds `this` for its method bodies
    and field initializers, but its `extends` clause and any computed member keys are evaluated in the
    enclosing receiver context, so only those parts of a class are descended. *root* itself is always
    yielded and descended, so a method reached directly through *root* is included.
    """
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (JsFunctionExpression, JsFunctionDeclaration)) and node is not root:
            continue
        if isinstance(node, (JsClassDeclaration, JsClassExpression)):
            if node.super_class is not None:
                stack.append(node.super_class)
            if node.body is not None:
                for member in node.body.body:
                    if isinstance(member, JsStaticBlock):
                        continue
                    if member.computed and member.key is not None:
                        stack.append(member.key)
            continue
        stack.extend(node.children())


def references_receiver_this(root: Node) -> bool:
    """
    Return whether relocating *root* would change the meaning of a `this` or `super` reference bound
    to its current receiver. Both are receiver-bound: `this` to the call's receiver and `super` to the
    method's home object, and `super` is a syntax error outside a method, so a value that uses either
    cannot be detached from its containing method. The receiver boundary is `walk_receiver_scope`:
    arrow functions inherit both lexically and are traversed; regular and generator functions nested
    below *root* rebind `this` (and cannot name the outer `super`) and are not descended into; a class
    rebinds `this` for its method bodies and field initializers, so only its `extends` clause and
    computed member keys are traversed. An identifier `super` that merely names a property (`x.super`)
    or an object-literal key is not a receiver-bound reference, so it is gated on `is_use_position` and
    does not count.
    """
    return any(
        isinstance(node, JsThisExpression)
        or (isinstance(node, JsIdentifier) and node.name == 'super' and is_use_position(node))
        for node in walk_receiver_scope(root)
    )


def is_receiver_binding_call(member: Node) -> bool:
    """
    Return whether *member* is evaluated in a position where the resulting call binds its object as the
    call's `this` receiver: the callee of a call `m(...)` (including an optional call `m?.(...)`) or the
    tag of a tagged template. Transparent parentheses are seen through; a comma-sequence callee
    `(0, m)()` yields a value rather than a Reference and so detaches the receiver, as does `new m()`
    (a constructor receives a fresh `this`). This is the dual of `references_receiver_this`: detaching
    *member* from its object preserves `this` unless the call site binds a receiver and the callee
    observes one.
    """
    node = member
    parent = node.parent
    while isinstance(parent, JsParenthesizedExpression):
        node, parent = parent, parent.parent
    if isinstance(parent, JsCallExpression) and parent.callee is node:
        return True
    if isinstance(parent, JsTaggedTemplateExpression) and parent.tag is node:
        return True
    return False


def rewrite_receiver_this_to_global(root: Node) -> bool:
    """
    Replace every `this` bound to *root*'s own receiver with a `globalThis` identifier, returning whether
    any replacement was made. The rewrite descends the receiver boundary `walk_receiver_scope` defines —
    through arrow functions and a class's `extends` clause and computed keys, but not into a nested
    regular or generator function, whose `this` is its own — so only *root*'s own `this` is rewritten.
    A caller uses this where *root* is invoked with no receiver, so its `this` is the global object: a
    `Function`-constructed body, or a recognized global-object finder whose `… || this` fallback yields
    the global. The synthesized `globalThis` identifiers stay subject to whatever free-name or shadow
    check the caller applies, so a binding named `globalThis` in scope declines the rewrite at the
    caller's discretion.
    """
    changed = False
    for node in list(walk_receiver_scope(root)):
        if isinstance(node, JsThisExpression):
            _replace_in_parent(node, JsIdentifier(name='globalThis'))
            changed = True
    return changed


def binding_has_references(
    model: SemanticModel,
    binding: Binding | None,
    *,
    exclude: Node | None = None,
    exclude_ids: set[int] | None = None,
) -> bool:
    """
    Whether *binding* is still read or written outside an excluded region. Resolution is
    binding-precise: only references that actually resolve to *binding* count, so a same-named
    variable in another scope never keeps it alive — this subsumes the name-based shadow check that
    `has_remaining_references` performs textually. A `None` binding (a name the model cannot resolve
    to a declaration) is conservatively reported as still referenced. References within the subtree
    of *exclude*, or whose node identity is in *exclude_ids*, are not counted.
    """
    if binding is None:
        return True
    for ref in model.references(binding, exclude=exclude):
        if exclude_ids and id(ref) in exclude_ids:
            continue
        return True
    return False


class BodyProcessingTransformer(Transformer):
    """
    Intermediate base for JS deobfuscation transformers that process the statement list (body) of
    `refinery.lib.scripts.js.model.JsScript` and `refinery.lib.scripts.js.model.JsBlockStatement`
    nodes after visiting children. Subclasses override `_process_body`.
    """

    def visit_JsScript(self, node: JsScript):
        self.generic_visit(node)
        self._process_body(node, node.body)
        return None

    def visit_JsBlockStatement(self, node: JsBlockStatement):
        self.generic_visit(node)
        self._process_body(node, node.body)
        return None

    def _process_body(self, parent: Node, body: list[Statement]) -> None:
        raise NotImplementedError

    def _replace_body(self, parent: Node, replacement: list[Statement]) -> None:
        """
        Replace the body of *parent* with *replacement* through `refinery.lib.scripts.set_body`, so
        the adoption of the new statements and the advance of the tree's mutation counter happen the
        one way every splice performs them, and mark the transformer as changed.
        """
        set_body(parent, list(replacement))
        self.mark_changed()


class ScopeProcessingTransformer(Transformer):
    """
    Base for transforms that process at function-scope boundaries. Visits
    `refinery.lib.scripts.js.model.JsScript` and each function body
    (`refinery.lib.scripts.js.model.JsFunctionDeclaration`,
    `refinery.lib.scripts.js.model.JsFunctionExpression`,
    `refinery.lib.scripts.js.model.JsArrowFunctionExpression`). Subclasses may override either
    `_process_scope` or `_process_scope_body`.
    """

    def visit_JsScript(self, node: JsScript):
        self.generic_visit(node)
        self._process_scope(node)
        return None

    def visit_JsFunctionDeclaration(self, node: JsFunctionDeclaration):
        self.generic_visit(node)
        if isinstance(node.body, JsBlockStatement):
            self._process_scope(node.body)
        return None

    def visit_JsFunctionExpression(self, node: JsFunctionExpression):
        self.generic_visit(node)
        if isinstance(node.body, JsBlockStatement):
            self._process_scope(node.body)
        return None

    def visit_JsArrowFunctionExpression(self, node: JsArrowFunctionExpression):
        self.generic_visit(node)
        if isinstance(node.body, JsBlockStatement):
            self._process_scope(node.body)
        return None

    def _process_scope(self, scope: Node) -> None:
        """
        Receives the raw scope node (`refinery.lib.scripts.js.model.JsScript` or
        `refinery.lib.scripts.js.model.JsBlockStatement`).
        """
        body = get_body(scope)
        if body is not None:
            self._process_scope_body(scope, body)

    def _process_scope_body(self, scope: Node, body: list) -> None:
        """
        Receives the scope node and its `body` list. The `_process_scope` method extracts the body
        and delegates here.
        """
        raise NotImplementedError


class ScriptLevelTransformer(Transformer):
    """
    Base for transforms that process the entire script manually rather than using the recursive
    visitor. Subclasses override `_process_script`.
    """

    def visit_JsScript(self, node: JsScript):
        self._process_script(node)
        return None

    def generic_visit(self, node: Node):
        pass

    def _process_script(self, node: JsScript) -> None:
        raise NotImplementedError
