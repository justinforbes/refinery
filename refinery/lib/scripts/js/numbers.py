"""
The JavaScript Number value domain. A Number is an IEEE-754 double, so the Python type that models
one is `float` and nothing else: `float` arithmetic *is* double arithmetic, whereas Python's `int` is
an arbitrary-precision integer whose arithmetic silently disagrees with the language everywhere past
2^53. Holding a Number as an `int` therefore does not merely spell a value differently, it computes
different answers, which is why the coercion lives here and is applied where a Number enters the tree
rather than at the places that happen to have been noticed.

This module sits beside `model.py` and `token.py` because what a Number is, how it prints, and which
Number a piece of text names are properties of the language rather than of any one pass over a
program. Both directions live here so that neither is written twice: a fold and an emulated execution
that disagreed about `Number('1e3')` would each be reading a grammar of its own.
"""
from __future__ import annotations

import math
import re

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


_STR_DECIMAL_LITERAL = re.compile(
    r'[+-]?(?:'
    r'Infinity'
    r'|[0-9]+\.[0-9]*(?:[eE][+-]?[0-9]+)?'
    r'|\.[0-9]+(?:[eE][+-]?[0-9]+)?'
    r'|[0-9]+(?:[eE][+-]?[0-9]+)?'
    r')'
)
"""
The ECMA-262 StrDecimalLiteral production, transcribed. Three of its properties are the ones a hand
written scan gets wrong. The sign is consumed once, ahead of the alternation, because `Infinity`
belongs to the unsigned production and a sign read inside it would leave `-Infinity` naming nothing.
An exponent that is begun and not finished is not an error but a shorter literal, so `1e+` names the
same Number as `1`; the engine supplies that backtracking. And a decimal digit is `0` through `9`
alone, never `str.isdigit`, which is true of every Unicode digit and of the superscripts.

The alternatives are ordered rather than longest-match, and the order that is correct is the
specification's own: the two that begin with a digit are tried with the fraction first, and a
fraction is never shorter than the integer part it extends.
"""

_NON_DECIMAL_INTEGER = re.compile(r'0[bB][01]+|0[oO][0-7]+|0[xX][0-9a-fA-F]+')
"""
The ECMA-262 NonDecimalIntegerLiteral production, in the form StrNumericLiteral admits it: with the
numeric separator forbidden, and with no sign, since the sign belongs to the decimal alternative.
"""

_NON_DECIMAL_RADIX = {'b': 2, 'o': 8, 'x': 16}


def _read_decimal_literal(text: str) -> tuple[float, int] | None:
    """
    The Number named by the StrDecimalLiteral that *text* begins with, together with the length of
    that literal, or `None` when the text does not begin with one. A reader of a prefix wants only
    the Number; a reader of a whole string wants the length as well, so that it can refuse text with
    anything left over.

    The matched text goes to Python's `float`, which reads it identically, because the grammar has
    already excluded every spelling on which the two disagree: `inf`, `nan`, the numeric separator,
    and the Unicode digits outside `0` through `9`. `Infinity` is the one alternative that never
    reaches `float`, the language and Python spelling that value differently. The digits are not
    truncated first — `float` is linear in them, and shortening them would move the rounding that
    makes `5e-324` and `1e309` come out at all.
    """
    match = _STR_DECIMAL_LITERAL.match(text)
    if match is None:
        return None
    literal = match[0]
    negative = literal[0] == '-'
    if literal[0] in '+-':
        literal = literal[1:]
    magnitude = float('inf') if literal == 'Infinity' else float(literal)
    return apply_sign(magnitude, negative), match.end()


def _read_non_decimal_integer(text: str) -> float | None:
    """
    The Number named by *text* when the whole of it is a NonDecimalIntegerLiteral, and `None`
    otherwise. Because this alternative carries neither a sign nor a numeric separator,
    `Number('-0x10')` and `Number('0x1_0')` are both `NaN`, where the same text is a perfectly good
    numeric literal in source and a perfectly good expression at runtime.
    """
    if _NON_DECIMAL_INTEGER.fullmatch(text) is None:
        return None
    return to_js_number(int(text[2:], _NON_DECIMAL_RADIX[text[1].lower()]))


def js_string_to_number(text: str) -> float:
    """
    Apply the ECMA-262 StringToNumber abstract operation, which is the Number that `Number(string)`
    answers. The whole of the string, once its padding is removed, has to be a StrNumericLiteral: a
    literal the text merely begins with is not an answer here, so `Number('1x')` is `NaN` where
    `parseFloat('1x')` is `1`. Padding and nothing else names zero.
    """
    text = text.strip(STRING_NUMERIC_TRIM)
    if not text:
        return 0.0
    integer = _read_non_decimal_integer(text)
    if integer is not None:
        return integer
    read = _read_decimal_literal(text)
    if read is None:
        return float('nan')
    value, length = read
    return value if length == len(text) else float('nan')


def js_parse_float(text: str) -> float:
    """
    Apply the `parseFloat` global function to a string. It reads the longest prefix of the text that
    is a StrDecimalLiteral and disregards the rest, so it answers a Number for text that only begins
    as one, and `NaN` only for text that does not begin as one at all. It reads no
    NonDecimalIntegerLiteral, which is why `parseFloat('0x10')` is `0`, stopped at the `x`.
    """
    read = _read_decimal_literal(text.lstrip(STRING_NUMERIC_TRIM))
    if read is None:
        return float('nan')
    return read[0]


_MAX_DIGITS_IN_A_DOUBLE = {
    radix: math.ceil(1024 / math.log2(radix)) for radix in range(2, 37)
}
"""
Per radix, the digit count past which a written-out integer is certainly outside the double range
and so denotes an infinity. Answering from the count rather than from the value keeps `parseInt`
total: building the integer first would make it hostage to CPython's limit on converting a long
decimal string, which raises rather than returning the `Infinity` the language specifies.
"""


def js_parse_int(text: str, radix: int = 0) -> float | None:
    """
    Replicate the semantics of JavaScript's `parseInt(string, radix)`. Strips leading whitespace,
    handles an optional `+`/`-` sign, and skips a leading `0x`/`0X` prefix for radix 16. Parses
    leading characters valid for the given radix (2-36) and stops at the first invalid one. Returns
    `None` when no valid digits are found (JS would return `NaN`).

    A radix of `0` is the language's "not supplied" — `parseInt(s)` and `parseInt(s, 0)` are the
    same call — and it is not a synonym for 10: an unsupplied radix reads a `0x` prefix as selecting
    base 16, so `parseInt('0x1f')` is `31`. A caller that defaults the radix to 10 instead answers
    `0` for that string, having stopped at the `x`.

    The digits are accumulated exactly and only then coerced, because `parseInt` reads the whole
    digit string before producing a Number: enough digits and the result is `Infinity`, and a digit
    string past 2^53 names the nearest double rather than itself. Leading zeros are dropped as they
    are read and the count is answered as soon as it passes the bound, so a digit string of any
    length costs what its significant prefix costs rather than what it is.
    """
    text = text.strip(STRING_NUMERIC_TRIM)
    if not text:
        return None
    negative = False
    if text[0] in '+-':
        negative = text[0] == '-'
        text = text[1:]
    hex_prefixed = len(text) >= 2 and text[0] == '0' and text[1] in 'xX'
    if radix == 0:
        radix = 16 if hex_prefixed else 10
    if not (2 <= radix <= 36):
        return None
    if radix == 16 and hex_prefixed:
        text = text[2:]
    limit = _MAX_DIGITS_IN_A_DOUBLE[radix]
    digits: list[str] = []
    scanned = False
    for ch in text:
        if '0' <= ch <= '9':
            if ord(ch) - ord('0') >= radix:
                break
        elif 'a' <= ch <= 'z' or 'A' <= ch <= 'Z':
            if ord(ch.lower()) - ord('a') + 10 >= radix:
                break
        else:
            break
        scanned = True
        if digits or ch != '0':
            digits.append(ch)
        if len(digits) > limit:
            return apply_sign(float('inf'), negative)
    if not scanned:
        return None
    return apply_sign(to_js_number(int(''.join(digits) or '0', radix)), negative)


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
