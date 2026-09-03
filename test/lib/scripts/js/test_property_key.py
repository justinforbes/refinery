"""
An object literal names a property with an IdentifierName, a string literal, a numeric literal, or a
bracketed expression, and a class body names an element with one of those or with a private name. An
IdentifierName is wider than a name the code around it could refer to, every reserved word spelling
one, which is what makes `x = { if: 1 };` a program; a numeral carries its own kind, so `x = { 1n:
1 };` names its property with the BigInt `1n` and not with a name whose text reads `1n`.

A punctuator spells none of these, and a key or a class element written with one is an early error;
so, outside a class body, is a private name, which the object grammar has no room for. A file that
breaks the rule is no program, and the parser reads it with the repair recorded, so
`refinery.lib.scripts.is_well_formed` answers `False` for the tree — the domain every fidelity law is
stated over — while the text still comes back as it went in.

SECURITY: the snippets here are hand-authored and benign, and `node --check` only parses them, never
running one. Nothing from `samples` may ever be fed to this.
"""
from __future__ import annotations

import unittest

from test import TestBase
from test.lib.scripts.js.analysis.differential import (
    node_executable,
    node_reads_as_a_program,
)
from test.lib.scripts.js.ledger import printed, well_formed

from refinery.lib.scripts.js.model import JsBigIntLiteral, JsNumericLiteral, JsProperty
from refinery.lib.scripts.js.parser import JsParser


#: Each property key against whether the file that writes it is a program. An identifier name, a
#: reserved word, a string, every spelling of a numeral, and a computed key are property names; a
#: punctuator is none, and a private name is one only in a class body. The last two rows carry a
#: punctuator into a method position, in an object literal and in a class body, so that the rule is
#: read over an element name and not over a value position alone.
A_PROPERTY_KEY = {
    'x = { a: 1 };': True,
    'x = { if: 1 };': True,
    "x = { 'a b': 1 };": True,
    'x = { 1: 1 };': True,
    'x = { .5: 1 };': True,
    'x = { 08: 1 };': True,
    'x = { 1_0: 1 };': True,
    'x = { 1n: 1 };': True,
    'x = { [k]: 1 };': True,
    'x = { +: 1 };': False,
    'x = { ,: 1 };': False,
    'x = { ;: 1 };': False,
    'x = { ): 1 };': False,
    'x = { %: 1 };': False,
    'x = { ++: 1 };': False,
    'x = { =>: 1 };': False,
    'x = { @: 1 };': False,
    'x = { #a: 1 };': False,
    'x = { +() {} };': False,
    'class C { +() {} }': False,
}


def _sole_property(source: str) -> JsProperty:
    return [
        node for node in JsParser(source).parse().walk()
        if isinstance(node, JsProperty)
    ][0]


def _canonical(text: str) -> str:
    return ''.join(text.split())


class TestAPropertyKeyIsSpelledByAName(TestBase):
    def test_a_key_that_spells_no_property_name_is_not_a_well_formed_program(self):
        self.assertEqual(
            {source: well_formed(source) for source in A_PROPERTY_KEY},
            A_PROPERTY_KEY,
        )

    def test_a_numeral_in_a_key_position_is_the_numeral_it_is_anywhere_else(self):
        """
        `x = 1n;` is read as the BigInt numeral it spells and so is the key of `x = { 1n: 1 };`,
        where the decimal beside it stays the Number it is. A key carries the kind of the numeral
        written there and is no name whose text happens to read `1n`, a spelling no identifier may
        hold and a property no object has.
        """
        self.assertEqual(
            (
                type(_sole_property('x = { 1n: 1 };').key),
                type(_sole_property('x = { 1: 1 };').key),
            ),
            (JsBigIntLiteral, JsNumericLiteral),
        )

    def test_a_key_the_rule_refuses_still_prints_back_as_it_was_written(self):
        """
        The refusal is recorded on the tree and not paid for in text: a file read with the repair
        recorded still prints back as it was written, layout aside, so nothing an analyst was handed
        goes missing when the tree stops being called a program.
        """
        self.assertEqual(
            {source: _canonical(printed(source)) for source in A_PROPERTY_KEY},
            {source: _canonical(source) for source in A_PROPERTY_KEY},
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestTheEngineReadsEachKeyAsTheCorpusRecords(TestBase):
    """
    The corpus is what carries the rule, so an engine re-measures every key of it: a row that went
    stale would leave the law above asserting the wrong thing about a file Node reads the other way.
    """

    def test_node_reads_each_key_as_the_corpus_records_it(self):
        self.assertEqual(
            {source: node_reads_as_a_program(source) for source in A_PROPERTY_KEY},
            A_PROPERTY_KEY,
        )
