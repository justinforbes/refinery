"""
The JavaScript Number value domain. A Number is an IEEE-754 double, so the Python type that models
one is `float` and nothing else: `float` arithmetic *is* double arithmetic, whereas Python's `int` is
an arbitrary-precision integer whose arithmetic silently disagrees with the language everywhere past
2^53. Holding a Number as an `int` therefore does not merely spell a value differently, it computes
different answers, which is why the coercion lives here and is applied where a Number enters the tree
rather than at the places that happen to have been noticed.

This module sits beside `model.py` and `token.py` because what a Number is, and how it prints, are
properties of the language rather than of any one pass over a program.
"""
from __future__ import annotations

import math

from decimal import Decimal


def to_js_number(value: int | float) -> float:
    """
    Coerce a Python number to the JavaScript Number it denotes. An integer too large for a double
    becomes an infinity, which is what the language does with a numeric literal of that magnitude —
    `1e400` is `Infinity` in JavaScript, and so is a literal written out with 400 digits.
    """
    try:
        return float(value)
    except OverflowError:
        return float('-inf') if value < 0 else float('inf')


STRING_NUMERIC_TRIM = (
    '\t\n\v\f\r\x20\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006'
    '\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff'
)
"""
The characters a string may be padded with and still name a Number: ECMA-262 WhiteSpace, which is
the space separators together with `U+FEFF`, plus the line terminators. It is spelled out rather
than left to `str.strip`, whose notion of a space is neither a subset nor a superset of this one:
Python takes `U+001C` through `U+001F`, which JavaScript does not, and leaves `U+FEFF`, which
JavaScript takes.
"""


def is_negative_zero(value: float) -> bool:
    """
    Whether *value* is negative zero. It is the one Number that neither `==` nor
    `js_number_to_string` can tell from its positive counterpart, so any code that must preserve it
    has to ask for the sign directly; `1 / -0` is `-Infinity` where `1 / 0` is `Infinity`.
    """
    return value == 0 and math.copysign(1.0, value) < 0


def exact_integer(value: int | float) -> int | None:
    """
    The integer *value* is, or `None` when it is not one. A consumer that needs a Python `int` — an
    index, a count, a radix — must ask this rather than call `int` on the Number, because the domain
    contains `NaN` and the infinities, on which `int` raises rather than answers.
    """
    if not math.isfinite(value):
        return None
    integer = int(value)
    return integer if integer == value else None


def apply_sign(magnitude: float, negative: bool) -> float:
    """
    The Number of the given *magnitude* carrying the sign that *negative* names. Zero decides how
    this has to be written: Python's integers have a single zero, so a sign carried through them is
    lost exactly where JavaScript keeps it, and `Number('-0')` and `parseInt('-0')` are both `-0`.
    """
    return math.copysign(magnitude, -1.0 if negative else 1.0)


def _significant_digits_to_string(value: float) -> str:
    """
    Format a finite, non-zero double as the ECMA-262 Number::toString algorithm would. That
    algorithm is stated over the *shortest* decimal digit string that round-trips to the double,
    which is what Python's `repr` produces; the exact mathematical value would carry digits past the
    ones the double determines, and no engine prints those. This also controls the
    decimal/exponential cutoff (exponential at magnitudes >= 1e21 or < 1e-6) and the exponent format
    (`1e-7`, not Python's `1e-07`).
    """
    negative = value < 0
    decimal = Decimal(repr(abs(value)))
    digits = ''.join(str(digit) for digit in decimal.as_tuple().digits).rstrip('0') or '0'
    count = len(digits)
    point = decimal.adjusted() + 1
    if count <= point <= 21:
        result = digits + '0' * (point - count)
    elif 0 < point <= 21:
        result = digits[:point] + '.' + digits[point:]
    elif -6 < point <= 0:
        result = '0.' + '0' * -point + digits
    else:
        mantissa = digits if count == 1 else digits[0] + '.' + digits[1:]
        exponent = point - 1
        result = F"{mantissa}e{'+' if exponent >= 0 else '-'}{abs(exponent)}"
    return F'-{result}' if negative else result


def js_number_to_string(value: float) -> str:
    """
    Apply `Number.prototype.toString` to a Number. Total over the domain, including the values that
    have no literal spelling: `NaN`, the infinities, and negative zero, which prints as `0` because
    the algorithm reads the mathematical value and the sign of a zero is not part of it.

    A `repr` that ends in `.0` is already the answer: Python writes a double positionally only below
    1e16, so such a value is an integer that the shortest round-tripping digits spell in full, which
    is the same branch of the algorithm `_significant_digits_to_string` would take. Spelling it as
    `int(value)` instead would be wrong past 2^53, where the double's exact value has more digits
    than it determines — `2 ** 60` is `1152921504606847000`, not `1152921504606846976`.
    """
    if value != value:
        return 'NaN'
    if value == float('inf'):
        return 'Infinity'
    if value == float('-inf'):
        return '-Infinity'
    if value == 0:
        return '0'
    plain = repr(value)
    if plain.endswith('.0'):
        return plain[:-2]
    return _significant_digits_to_string(value)
