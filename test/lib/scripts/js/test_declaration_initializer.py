"""
A variable declaration is written with an initializer, and what may go without one is decided by
three things together: the keyword the declaration opens with, whether it binds a name or a pattern,
and whether it stands in a for-in or for-of head.

`var` and `let` bind a bare name and leave it undefined, so a name under either may go without a
value. `const` may not: the binding it makes can never be assigned again, so a declaration that gave
it no value would leave it unreadable for the whole of its life. A destructuring target may not go
without one under any keyword, a pattern having nothing to take apart when there is nothing on the
right. A for-in or for-of head is the one position that lifts the requirement whole, because it hands
the binding its value on every pass; the first clause of a C-style head is an ordinary declaration
and both rules reach it unchanged.

A declaration that breaks the rule is no program, and the parser reads it with the repair recorded,
so `refinery.lib.scripts.is_well_formed` answers `False` for the tree — the domain every fidelity law
is stated over — while the text still comes back as it went in.

SECURITY: the snippets here are hand-authored and benign, and `node --check` only parses them, never
running one. Nothing from `samples` may ever be fed to this.
"""
from __future__ import annotations

import functools
import subprocess
import tempfile
import unittest

from pathlib import Path

from test import TestBase
from test.lib.scripts.js.analysis.differential import node_executable
from test.lib.scripts.js.ledger import printed, well_formed


#: A declaration standing as a statement, mapped to whether it is a well-formed program. The nine
#: cells are the three keywords against a name and the two patterns; only `var x;` and `let x;` may
#: go without an initializer.
A_DECLARATION_AS_A_STATEMENT = {
    'var x;': True,
    'let x;': True,
    'const x;': False,
    'var [a];': False,
    'var {a};': False,
    'let [a];': False,
    'let {a};': False,
    'const [a];': False,
    'const {a};': False,
}


def _in_the_iterating_heads() -> dict[str, bool]:
    return {
        F'for ({keyword} {target} {word} o) {{}}': True
        for keyword in ('var', 'let', 'const')
        for target in ('x', '[a]', '{a}')
        for word in ('in', 'of')
    }


#: The same nine cells in each of the three heads a declaration stands in. The eighteen files of the
#: two heads that iterate are programs whatever their keyword and target; the nine of the C-style
#: head are the statement rule again, its first clause being an ordinary declaration.
A_DECLARATION_IN_A_FOR_HEAD = {
    **_in_the_iterating_heads(),
    'for (var x;;) {}': True,
    'for (let x;;) {}': True,
    'for (const x;;) {}': False,
    'for (var [a];;) {}': False,
    'for (var {a};;) {}': False,
    'for (let [a];;) {}': False,
    'for (let {a};;) {}': False,
    'for (const [a];;) {}': False,
    'for (const {a};;) {}': False,
}

A_DECLARATION_WHEREVER_IT_STANDS = {
    **A_DECLARATION_AS_A_STATEMENT,
    **A_DECLARATION_IN_A_FOR_HEAD,
}


@functools.lru_cache(maxsize=None)
def _node_reads_as_a_program(source: str) -> bool:
    """
    Whether `node --check` reads *source* as a program, parsing it as a script and never running it.
    A missing initializer is a syntax error the check reports, so this is the engine answering the
    same question `well_formed` answers.
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


def _canonical(text: str) -> str:
    return ''.join(text.split())


class TestADeclarationIsWrittenWithAnInitializerAsItsKeywordAndTargetDecide(TestBase):
    def test_a_statement_declaration_may_omit_the_initializer_only_for_a_named_var_or_let(self):
        self.assertEqual(
            {source: well_formed(source) for source in A_DECLARATION_AS_A_STATEMENT},
            A_DECLARATION_AS_A_STATEMENT,
        )

    def test_a_for_head_lifts_the_requirement_only_where_it_iterates(self):
        self.assertEqual(
            {source: well_formed(source) for source in A_DECLARATION_IN_A_FOR_HEAD},
            A_DECLARATION_IN_A_FOR_HEAD,
        )

    def test_a_declaration_the_rule_refuses_still_prints_back_as_it_was_written(self):
        """
        The refusal is recorded on the tree and not paid for in text: a file read with the repair
        recorded still prints back as it was written, layout aside, so nothing an analyst was handed
        goes missing when the tree stops being called a program.
        """
        self.assertEqual(
            {source: _canonical(printed(source)) for source in A_DECLARATION_WHEREVER_IT_STANDS},
            {source: _canonical(source) for source in A_DECLARATION_WHEREVER_IT_STANDS},
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestTheEngineReadsEachDeclarationAsTheCorpusRecords(TestBase):
    """
    The corpus is what carries the rule, so an engine re-measures every cell of it: a row that went
    stale would leave the law above asserting the wrong thing about a file Node reads the other way.
    """

    def test_node_reads_each_declaration_as_the_corpus_records_it(self):
        self.assertEqual(
            {
                source: _node_reads_as_a_program(source)
                for source in A_DECLARATION_WHEREVER_IT_STANDS
            },
            A_DECLARATION_WHEREVER_IT_STANDS,
        )
