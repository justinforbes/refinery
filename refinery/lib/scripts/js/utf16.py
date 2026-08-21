from __future__ import annotations

import re


_ABOVE_THE_BASIC_PLANE = re.compile('[\U00010000-\U0010FFFF]')
_SURROGATE_PAIR = re.compile('[\ud800-\udbff][\udc00-\udfff]')


def code_units(value: int) -> str:
    """
    The UTF-16 code units a code point is written with, held as the characters Python spells those
    units with. A JavaScript string is a sequence of code units and a Python string a sequence of
    code points, so a value above the basic plane is two characters here and not one: `\\u{1F600}`
    and `\\uD83D\\uDE00` are one string written two ways, and it is the pair of surrogates they
    share. Naming the code point instead would make them two strings, and would answer one short
    for every question a program asks about the length of that string or the units inside it.
    """
    if value <= 0xFFFF:
        return chr(value)
    value -= 0x10000
    return chr(0xD800 + (value >> 10)) + chr(0xDC00 + (value & 0x3FF))


def to_code_units(text: str) -> str:
    """
    The string *text* holds, rewritten so every character is one UTF-16 code unit: a code point
    above the basic plane becomes the surrogate pair that spells it. A code unit is left
    unchanged, so a string already in this form, one holding surrogate pairs or a lone surrogate,
    passes through untouched, which makes applying this at a value's every point of entry safe to
    repeat.
    """
    return _ABOVE_THE_BASIC_PLANE.sub(lambda m: code_units(ord(m.group())), text)


def from_code_units(text: str) -> str:
    """
    The characters *text* spells, undoing `to_code_units`: a pair of surrogates becomes the one code
    point it encodes. A string held as code units cannot be written to a file as it stands, because
    a surrogate is not a character any encoding spells, and whatever holds such a string has to spell
    it out again before anything reads it as text.

    A lone surrogate is left standing, since no pairing rule may invent the partner it lacks, and a
    string with none in it comes back unchanged.
    """
    return _SURROGATE_PAIR.sub(
        lambda match: chr(
            0x10000
            + ((ord(match.group()[0]) - 0xD800) << 10)
            + (ord(match.group()[1]) - 0xDC00)
        ),
        text,
    )
