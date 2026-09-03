"""
An update operator, `++` or `--`, writes its operand back, so §13.4 requires that operand to be a
reference: a name, a member access, or — in the sloppy code a script is until a directive makes it
strict — a call. A parenthesis wraps a reference only where what it holds is one, and a single `?.`
anywhere in the member-and-call chain makes the whole an optional expression the language forbids
an update from writing to. A value made on the spot — a function, an arrow, a class, a literal, an
array, `this`, a `new`, a sequence — is no reference, and an update written against one is an early
error the grammar refuses.

A file that breaks the rule is no program, and the parser reads it with the repair recorded, so
`refinery.lib.scripts.is_well_formed` answers `False` for the tree — the domain every fidelity law
is stated over. A caller told the tree is well formed would otherwise compare text that is not a
program against the file it came from.

SECURITY: the snippets here are hand-authored and benign, and `node --check` only parses them,
never running one. Nothing from `samples` may ever be fed to this.
"""
from __future__ import annotations

import functools
import subprocess
import tempfile
import unittest

from pathlib import Path

from test import TestBase
from test.lib.scripts.js.analysis.differential import node_executable
from test.lib.scripts.js.ledger import well_formed


#: Each expression an update is written against, mapped to whether it is a reference the operator
#: may write back to. A name, a member access and a call are references; a call because a script is
#: sloppy code until a directive makes it strict, where the same operand would be refused. An
#: optional link anywhere in a chain, a value made on the spot, a literal, `this`, a bare `new` and
#: a sequence are not, and a parenthesis is one exactly when what it wraps is.
AN_UPDATE_OPERAND = {
    'a': True,
    'a.b': True,
    'a[b]': True,
    'a.b.c': True,
    'f()': True,
    'f().b': True,
    'a.b()': True,
    '(a)': True,
    '(a.b)': True,
    'new a().b': True,
    'a?.b': False,
    'a?.b.c': False,
    'a.b?.c': False,
    'this': False,
    '1': False,
    '[a]': False,
    '(a, b)': False,
    'new a': False,
    'a => {}': False,
    'function () {}': False,
    'class {}': False,
    '(a?.b)': False,
}


#: Each operand written as both a postfix and a prefix update, mapped to whether the file is a
#: program: an update is a program exactly when its operand is a reference, whichever side the
#: operator stands. The two rows the ledger entry this law retires was quantified over are the
#: arrow and the function expression, kept here verbatim as the witnesses they were.
A_FILE_WRITING_AN_UPDATE = {
    **{
        source: reference
        for operand, reference in AN_UPDATE_OPERAND.items()
        for source in (F'{operand}++;', F'++{operand};')
    },
    'f = a => {}++': False,
    'f = function () {}++': False,
}


@functools.lru_cache(maxsize=None)
def _node_reads_as_a_program(source: str) -> bool:
    """
    Whether `node --check` reads *source* as a program, parsing it as a script and never running it.
    An update against a non-reference is a syntax error the check reports, so this is the engine
    answering the same question `well_formed` answers.
    """
    executable = node_executable()
    assert executable is not None
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'snippet.js'
        path.write_text(source, encoding='utf8')
        completed = subprocess.run(
            [executable, '--check', str(path)],
            capture_output=True,
            text=True,
        )
    return completed.returncode == 0


class TestAnUpdateWritesBackToAReference(TestBase):
    def test_an_update_against_something_that_is_no_reference_is_no_program(self):
        self.assertEqual(
            {source: well_formed(source) for source in A_FILE_WRITING_AN_UPDATE},
            A_FILE_WRITING_AN_UPDATE,
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestTheEngineReadsEachUpdateAsTheCorpusRecords(TestBase):
    """
    The corpus is what carries the rule, so an engine re-measures every row of it: one that went
    stale would leave the law above asserting the wrong thing about a file Node reads the other way.
    """

    def test_node_reads_each_update_as_the_corpus_records_it(self):
        self.assertEqual(
            {source: _node_reads_as_a_program(source) for source in A_FILE_WRITING_AN_UPDATE},
            A_FILE_WRITING_AN_UPDATE,
        )
