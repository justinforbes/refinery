"""
The with-clause a module declaration names its module with, and where the parser reads one.

An import attribute clause — `with { type: 'json' }` — stands on any declaration that names a
module: a default, namespace, or named import, and a re-export written with a `from`
(`export { a } from`, `export *`, `export * as`). It is the same clause in every one of these
positions and it names attributes of the module, not bindings the program refers to, so its keys
are read in no use position and a rename must leave them alone.

The clause has meaning only as the tail of a `from`, so a declaration that writes one with no
module specifier for it to follow is no program. A declaration that writes none carries none,
which is what keeps the clause additive rather than something the printer supplies.

Node is the oracle for whether a shape is a program: each declaration below imports a module that
is not there, so a host that reads the declaration reaches the linker and fails to resolve it,
where a host that cannot read it fails in the parser instead. The two are told apart by the error
Node reports — a missing module against a `SyntaxError`.

SECURITY: every snippet below is written by this file and Node runs only those. Nothing from
`samples` may ever be handed to the engine.
"""
from __future__ import annotations

import unittest

from test import TestBase
from test.lib.scripts.js.analysis.differential import behavior, node_executable
from test.lib.scripts.js.ledger import printed, well_formed

from refinery.lib.scripts.js.analysis.model import is_use_position
from refinery.lib.scripts.js.model import JsIdentifier
from refinery.lib.scripts.js.parser import JsParser


#: One import attribute clause in each declaration that names a module: the three import forms and
#: the three re-export forms. Every row spells the clause `with { type: 'json' }`, so a program that
#: reads the key `type` as a binding is one this corpus catches.
A_MODULE_DECLARATION_CARRYING_ATTRIBUTES = (
    "import d from './x.json' with { type: 'json' };",
    "import * as ns from './x.json' with { type: 'json' };",
    "import { a } from './x.json' with { type: 'json' };",
    "export { a as b } from './x.json' with { type: 'json' };",
    "export * from './x.json' with { type: 'json' };",
    "export * as ns from './x.json' with { type: 'json' };",
)

#: A clause of more than one attribute, so that the comma between them is shown to survive the print
#: as much as the single attribute does.
A_CLAUSE_OF_MORE_THAN_ONE_ATTRIBUTE = (
    "export * from './x.json' with { type: 'json', hint: 'eager' };"
)

#: A re-export written with no clause, which has to come back with none: the clause appears only
#: where the file wrote one, and never because a re-export carries a source.
A_RE_EXPORT_CARRYING_NO_CLAUSE = (
    "export { a } from 'm';",
    "export * from 'm';",
    "export * as ns from 'm';",
)

#: A clause written where no `from` precedes it. A with-clause is the tail of a module specifier and
#: nothing else, so each of these is a declaration the language refuses.
A_CLAUSE_WITH_NO_FROM_TO_FOLLOW = (
    "import x with { type: 'json' };",
    "export * with { type: 'json' };",
    "export { default as j } with { type: 'json' };",
)


def _reads_a_binding_named(source: str, name: str) -> int:
    """
    How many identifiers named *name* in *source* stand where a binding is read or written. An
    attribute key names a property of the module and reads nothing, so a clause whose key is *name*
    contributes none.
    """
    tree = JsParser(source).parse()
    return sum(
        is_use_position(node) for node in tree.walk()
        if isinstance(node, JsIdentifier) and node.name == name
    )


class TestAModuleDeclarationCarriesTheAttributesItWasWrittenWith(TestBase):

    def test_each_clause_is_a_well_formed_program_that_reads_no_binding(self):
        rows = A_MODULE_DECLARATION_CARRYING_ATTRIBUTES
        self.assertEqual(
            {source: (well_formed(source), _reads_a_binding_named(source, 'type')) for source in rows},
            {source: (True, 0) for source in rows},
        )

    def test_each_clause_prints_back_exactly_as_it_was_written(self):
        rows = A_MODULE_DECLARATION_CARRYING_ATTRIBUTES
        self.assertEqual(
            {source: printed(source) for source in rows},
            {source: source for source in rows},
        )

    def test_a_clause_of_more_than_one_attribute_survives_the_print(self):
        source = A_CLAUSE_OF_MORE_THAN_ONE_ATTRIBUTE
        self.assertEqual(
            (
                well_formed(source),
                printed(source),
                _reads_a_binding_named(source, 'type'),
                _reads_a_binding_named(source, 'hint'),
            ),
            (True, source, 0, 0),
        )

    def test_a_re_export_with_no_clause_prints_none(self):
        rows = A_RE_EXPORT_CARRYING_NO_CLAUSE
        self.assertEqual(
            {source: printed(source) for source in rows},
            {source: source for source in rows},
        )

    def test_a_clause_with_no_from_to_follow_is_no_program(self):
        rows = A_CLAUSE_WITH_NO_FROM_TO_FOLLOW
        self.assertEqual(
            {source: well_formed(source) for source in rows},
            {source: False for source in rows},
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestNodeReadsEachClauseAndRefusesOneWithNoFrom(TestBase):

    def test_node_parses_each_clause_and_refuses_a_clause_with_no_from(self):
        """
        Node fails at the link for every declaration that carries a clause, the module `./x.json`
        being absent, and fails in the parser for every clause that no `from` precedes. The missing
        module is reported as an `ERR_MODULE_NOT_FOUND`, which `behavior` buckets as `ERROR`, and the
        parse failure as a `SyntaxError`.
        """
        carried = A_MODULE_DECLARATION_CARRYING_ATTRIBUTES + (A_CLAUSE_OF_MORE_THAN_ONE_ATTRIBUTE,)
        no_from = A_CLAUSE_WITH_NO_FROM_TO_FOLLOW
        self.assertEqual(
            {source: behavior(source, module=True)[1] for source in carried + no_from},
            {
                **{source: 'ERROR' for source in carried},
                **{source: 'SyntaxError' for source in no_from},
            },
        )
