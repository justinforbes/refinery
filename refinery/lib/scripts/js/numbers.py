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


def _significant_digits_to_string(value: float) -> str:
    """
    Format a finite, non-zero double as the ECMA-262 Number::toString algorithm would. This controls
    the decimal/exponential cutoff (exponential at magnitudes >= 1e21 or < 1e-6) and the exponent
    format (`1e-7`, not Python's `1e-07`).
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
    """
    if value != value:
        return 'NaN'
    if value == float('inf'):
        return 'Infinity'
    if value == float('-inf'):
        return '-Infinity'
    if value == 0:
        return '0'
    if value.is_integer() and abs(value) < 1e21:
        return str(int(value))
    return _significant_digits_to_string(value)
