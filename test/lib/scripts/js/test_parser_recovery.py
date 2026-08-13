from __future__ import annotations

from typing import NamedTuple

from test import TestBase

from refinery.lib.scripts import is_well_formed
from refinery.lib.scripts.js.model import JsErrorNode
from refinery.lib.scripts.js.parser import JsParser
from refinery.lib.scripts.js.synth import JsSynthesizer
from refinery.units.sinks.ppjscript import ppjscript


_HEAD = 'const registry = {};\nconst names = Object.keys(registry);\n'
"""
The two statements every source below starts with. A recovery that swallowed what stood before the
position it could not read would show up in the printed text rather than hide in a one-token file,
and the printer writes these two back character for character, which is why every expectation here
is spelled as this text followed by the tail the broken construct contributes.
"""


class Stop(NamedTuple):
    """
    A source that stops where the language still requires something, together with the text that
    would have finished it. `cut` is what the tool is handed and `whole` is the file it was cut
    from, so the text the cut took is the whole of the difference between the two. Node rejects
    every `cut` here and accepts every `whole`.
    """
    opened: str
    closing: str

    @property
    def cut(self) -> str:
        return F'{_HEAD}{self.opened}'

    @property
    def whole(self) -> str:
        return F'{_HEAD}{self.opened}{self.closing}'


_STOPS = {
    'var_with_no_name': Stop(
        'var', ' handler = registry;\n'),
    'const_with_no_name': Stop(
        'const', ' handler = registry;\n'),
    'member_access_with_no_name': Stop(
        'registry.', 'alpha;\n'),
    'optional_member_access_with_no_name': Stop(
        'registry?.', 'alpha;\n'),
    'chained_member_access_with_no_name': Stop(
        'registry.alpha.', 'beta;\n'),
    'new_meta_property_with_no_name': Stop(
        'function build() { new.', 'target; }\n'),
    'object_getter_with_no_name': Stop(
        'const o = { get', ' alpha() { return 1; } };\n'),
    'object_setter_with_no_name': Stop(
        'const o = { set', ' alpha(v) { registry.v = v; } };\n'),
    'object_async_method_with_no_name': Stop(
        'const o = { async', ' alpha() {} };\n'),
    'catch_parameter_with_no_name': Stop(
        'try { names.pop(); } catch (', 'err) { registry.e = err; }\n'),
    'label_with_no_statement': Stop(
        'outer:', ' while (names.pop()) break outer;\n'),
    'super_member_with_no_name': Stop(
        'class Foo extends Object { m() { super.', 'toString(); } }\n'),
    'class_heritage_with_no_name': Stop(
        'class Foo extends', ' Object {}\n'),
    'export_default_with_no_value': Stop(
        'export default', ' registry;\n'),
    'computed_member_with_no_key': Stop(
        'registry[', "'alpha'];\n"),
    'spread_with_no_argument': Stop(
        'const o = { ...', 'registry };\n'),
    'for_head_with_no_binding': Stop(
        'for (const', ' x of names) registry[x] = 1;\n'),
    'switch_case_with_no_test': Stop(
        'switch (names.length) { case', ' 1: break; }\n'),
    'arrow_with_no_body': Stop(
        'const f = () =>', ' registry;\n'),
}


_MALFORMED = {
    'doubled_dot': 'registry..alpha;\n',
    'dot_before_a_semicolon': 'registry.;\n',
    'dot_before_a_bracket': 'registry.];\n',
    'dot_before_a_string': "registry.'alpha';\n",
    'new_dot_before_a_string': "function f() { new.'target'; }\n",
    'label_before_a_closing_brace': 'function f() { outer: }\n',
}
"""
Sources that stop nowhere and are simply wrong: a name position holding something that is not a
name. Node rejects each of them, and unlike a cut file no completion turns one into a program, so
what the tool does with them is pinned on its own.
"""


class TestParserRecoveryAlwaysPrints(TestBase):
    """
    `refinery.lib.scripts.js.parser.JsParser` never raises, so every question about a broken file is
    answered by the tree it recovers. The tree may hold a
    `refinery.lib.scripts.js.model.JsErrorNode`, which keeps the source it stands for, but it may
    not hold a shape the model calls unspellable: `refinery.lib.scripts.Synthesizer.visit` refuses
    those, and a recovery that builds one turns a file that merely fails to parse into a crash of
    the tools that print it. What is pinned here is that no such shape is reached, that the text
    comes back, and that the tree says it is not a program.
    """

    def _print(self, source: str) -> str:
        return JsSynthesizer().convert(JsParser(source).parse())

    def _errors(self, source: str) -> list[tuple[str, str]]:
        return [
            (node.text, node.message)
            for node in JsParser(source).parse().walk_in_order()
            if isinstance(node, JsErrorNode)
        ]

    def test_a_source_that_stops_mid_construct_prints_the_text_it_read(self):
        expected = {
            'var_with_no_name': 'var ;',
            'const_with_no_name': 'const ;',
            'member_access_with_no_name': 'registry.;',
            'optional_member_access_with_no_name': 'registry?.;',
            'chained_member_access_with_no_name': 'registry.alpha.;',
            'new_meta_property_with_no_name': 'function build() {\n  new.;\n}',
            'object_getter_with_no_name': 'const o = { get () {} };',
            'object_setter_with_no_name': 'const o = { set () {} };',
            'object_async_method_with_no_name': 'const o = { async () {} };',
            'catch_parameter_with_no_name': 'try {\n  names.pop();\n} catch () {}',
            'label_with_no_statement': 'outer: ',
            'super_member_with_no_name': 'class Foo extends Object {\n  m() {\n    super.;\n  }\n}',
            'class_heritage_with_no_name': 'class Foo extends  {}',
            'export_default_with_no_value': 'export default ;',
            'computed_member_with_no_key': 'registry[];',
            'spread_with_no_argument': 'const o = { ... };',
            'for_head_with_no_binding': 'for (const ; ; ) {\n  \n}',
            'switch_case_with_no_test': 'switch (names.length) {\n  case :\n}',
            'arrow_with_no_body': 'const f = () => ;',
        }
        for name, tail in expected.items():
            with self.subTest(name):
                self.assertEqual(self._print(_STOPS[name].cut), F'{_HEAD}{tail}')

    def test_a_source_that_stops_mid_construct_is_not_a_well_formed_program(self):
        for name, stop in _STOPS.items():
            with self.subTest(name):
                self.assertEqual(is_well_formed(JsParser(stop.cut).parse()), False)

    def test_the_position_a_source_stops_at_is_kept_as_an_error_node(self):
        expected = {
            'var_with_no_name': [('', 'expected a name')],
            'const_with_no_name': [('', 'expected a name')],
            'member_access_with_no_name': [('', 'expected a property name')],
            'optional_member_access_with_no_name': [('', 'expected a property name')],
            'chained_member_access_with_no_name': [('', 'expected a property name')],
            'new_meta_property_with_no_name': [('', 'expected a property name')],
            'object_getter_with_no_name': [('', 'expected a name')],
            'object_setter_with_no_name': [('', 'expected a name')],
            'object_async_method_with_no_name': [('', 'expected a name')],
            'catch_parameter_with_no_name': [('', 'expected a name')],
            'label_with_no_statement': [('', 'unexpected token')],
            'super_member_with_no_name': [('', 'expected a property name')],
            'class_heritage_with_no_name': [('', 'unexpected token')],
            'export_default_with_no_value': [('', 'unexpected token')],
            'computed_member_with_no_key': [('', 'unexpected token')],
            'spread_with_no_argument': [('', 'unexpected token')],
            'for_head_with_no_binding': [
                ('', 'expected a name'),
                ('', 'unexpected token'),
                ('', 'unexpected token'),
                ('', 'unexpected token'),
            ],
            'switch_case_with_no_test': [('', 'unexpected token')],
            'arrow_with_no_body': [('', 'unexpected token')],
        }
        for name, errors in expected.items():
            with self.subTest(name):
                self.assertEqual(self._errors(_STOPS[name].cut), errors)

    def test_writing_the_text_the_cut_took_yields_a_program_that_prints(self):
        expected = {
            'var_with_no_name': 'var handler = registry;',
            'const_with_no_name': 'const handler = registry;',
            'member_access_with_no_name': 'registry.alpha;',
            'optional_member_access_with_no_name': 'registry?.alpha;',
            'chained_member_access_with_no_name': 'registry.alpha.beta;',
            'new_meta_property_with_no_name': 'function build() {\n  new.target;\n}',
            'object_getter_with_no_name': 'const o = { get alpha() {\n  return 1;\n} };',
            'object_setter_with_no_name': 'const o = { set alpha(v) {\n  registry.v = v;\n} };',
            'object_async_method_with_no_name': 'const o = { async alpha() {} };',
            'catch_parameter_with_no_name':
                'try {\n  names.pop();\n} catch (err) {\n  registry.e = err;\n}',
            'label_with_no_statement': 'outer: while (names.pop()) {\n  break outer;\n}',
            'super_member_with_no_name':
                'class Foo extends Object {\n  m() {\n    super.toString();\n  }\n}',
            'class_heritage_with_no_name': 'class Foo extends Object {}',
            'export_default_with_no_value': 'export default registry;',
            'computed_member_with_no_key': "registry['alpha'];",
            'spread_with_no_argument': 'const o = { ...registry };',
            'for_head_with_no_binding': 'for (const x of names) {\n  registry[x] = 1;\n}',
            'switch_case_with_no_test': 'switch (names.length) {\n  case 1:\n    break;\n}',
            'arrow_with_no_body': 'const f = () => registry;',
        }
        for name, tail in expected.items():
            with self.subTest(name):
                whole = _STOPS[name].whole
                self.assertEqual(is_well_formed(JsParser(whole).parse()), True)
                self.assertEqual(self._print(whole), F'{_HEAD}{tail}')

    def test_the_printer_reads_back_what_it_printed_for_a_source_that_stops_mid_construct(self):
        """
        The output of a recovery is itself input to the tool, and an error node prints as the text
        it kept rather than as a form any parser agreed to read, so the second pass reads something
        else than the first built. What may not happen is a refusal on the second pass: the tool
        would then fail on a file it wrote itself. Only the label keeps its spelling; the rest drift
        by the character the recovery could not attribute, and the drift is pinned rather than
        described.
        """
        expected = {
            'var_with_no_name': 'var ;;',
            'const_with_no_name': 'const ;;',
            'member_access_with_no_name': 'registry.;;',
            'optional_member_access_with_no_name': 'registry?.;;',
            'chained_member_access_with_no_name': 'registry.alpha.;;',
            'new_meta_property_with_no_name': 'function build() {\n  new.;;\n}',
            'object_getter_with_no_name': 'const o = { get() {} };',
            'object_setter_with_no_name': 'const o = { set() {} };',
            'object_async_method_with_no_name': 'const o = { async() {} };',
            'catch_parameter_with_no_name': 'try {\n  names.pop();\n} catch ()) {}',
            'label_with_no_statement': 'outer: ',
            'super_member_with_no_name':
                'class Foo extends Object {\n  m() {\n    super.;;\n  }\n}',
            'class_heritage_with_no_name': 'class Foo extends {} {}',
            'export_default_with_no_value': 'export default ;;',
            'computed_member_with_no_key': 'registry[]];',
            'spread_with_no_argument': 'const o = { ...}, ; };',
            'for_head_with_no_binding': 'for (const ;; ); }) {\n  \n}',
            'switch_case_with_no_test': 'switch (names.length) {\n  case ::\n}',
            'arrow_with_no_body': 'const f = () => ;;',
        }
        for name, tail in expected.items():
            with self.subTest(name):
                once = self._print(_STOPS[name].cut)
                self.assertEqual(self._print(once), F'{_HEAD}{tail}')

    def test_ppjscript_prints_a_source_that_stops_mid_construct(self):
        expected = {
            'var_with_no_name': 'var ;',
            'const_with_no_name': 'const ;',
            'member_access_with_no_name': 'registry.;',
            'optional_member_access_with_no_name': 'registry?.;',
            'chained_member_access_with_no_name': 'registry.alpha.;',
            'new_meta_property_with_no_name': 'function build() {\n    new.;\n}',
            'object_getter_with_no_name': 'const o = { get () {} };',
            'object_setter_with_no_name': 'const o = { set () {} };',
            'object_async_method_with_no_name': 'const o = { async () {} };',
            'catch_parameter_with_no_name': 'try {\n    names.pop();\n} catch () {}',
            'label_with_no_statement': 'outer: ',
            'super_member_with_no_name':
                'class Foo extends Object {\n    m() {\n        super.;\n    }\n}',
            'class_heritage_with_no_name': 'class Foo extends  {}',
            'export_default_with_no_value': 'export default ;',
            'computed_member_with_no_key': 'registry[];',
            'spread_with_no_argument': 'const o = { ... };',
            'for_head_with_no_binding': 'for (const ; ; ) {\n    \n}',
            'switch_case_with_no_test': 'switch (names.length) {\n    case :\n}',
            'arrow_with_no_body': 'const f = () => ;',
        }
        for name, tail in expected.items():
            with self.subTest(name):
                printed = _STOPS[name].cut.encode('utf8') | ppjscript() | str
                self.assertEqual(printed, F'{_HEAD}{tail}')


class TestParserRecoveryOverAMalformedNamePosition(TestBase):
    """
    The same law where a name position holds a token that is not a name. Nothing was cut off here,
    so the parser has text to keep, and what it kept is what comes back.
    """

    def _print(self, source: str) -> str:
        return JsSynthesizer().convert(JsParser(source).parse())

    def test_a_malformed_name_position_prints_the_text_it_read(self):
        expected = {
            'doubled_dot': 'registry..;\nalpha;',
            'dot_before_a_semicolon': 'registry.;;',
            'dot_before_a_bracket': 'registry.];',
            'dot_before_a_string': "registry.'alpha';",
            'new_dot_before_a_string': "function f() {\n  new.'target';\n}",
            'label_before_a_closing_brace': 'function f() {\n  outer: }\n}',
        }
        for name, tail in expected.items():
            with self.subTest(name):
                self.assertEqual(self._print(F'{_HEAD}{_MALFORMED[name]}'), F'{_HEAD}{tail}')

    def test_a_malformed_name_position_is_not_a_well_formed_program(self):
        for name, tail in _MALFORMED.items():
            with self.subTest(name):
                self.assertEqual(is_well_formed(JsParser(F'{_HEAD}{tail}').parse()), False)

    def test_the_token_that_was_not_a_name_is_kept_as_an_error_node(self):
        expected = {
            'doubled_dot': [('.', 'expected a property name')],
            'dot_before_a_semicolon': [(';', 'expected a property name')],
            'dot_before_a_bracket': [(']', 'expected a property name')],
            'dot_before_a_string': [("'alpha'", 'expected a property name')],
            'new_dot_before_a_string': [("'target'", 'expected a property name')],
            'label_before_a_closing_brace': [('}', 'unexpected token')],
        }
        for name, errors in expected.items():
            with self.subTest(name):
                script = JsParser(F'{_HEAD}{_MALFORMED[name]}').parse()
                self.assertEqual([
                    (node.text, node.message)
                    for node in script.walk_in_order()
                    if isinstance(node, JsErrorNode)
                ], errors)

    def test_ppjscript_prints_a_malformed_name_position(self):
        expected = {
            'doubled_dot': 'registry..;\nalpha;',
            'dot_before_a_semicolon': 'registry.;;',
            'dot_before_a_bracket': 'registry.];',
            'dot_before_a_string': "registry.'alpha';",
            'new_dot_before_a_string': "function f() {\n    new.'target';\n}",
            'label_before_a_closing_brace': 'function f() {\n    outer: }\n}',
        }
        for name, tail in expected.items():
            with self.subTest(name):
                printed = F'{_HEAD}{_MALFORMED[name]}'.encode('utf8') | ppjscript() | str
                self.assertEqual(printed, F'{_HEAD}{tail}')


#: Module declarations that stop before the specifier they read from. A module host refuses every
#: one of them: `new vm.SourceTextModule` under `node --experimental-vm-modules` answers
#: `SyntaxError: Unexpected end of input`, and `Unexpected token ','` for `import a,`.
MODULE_DECLARATIONS_CUT_BEFORE_THEIR_SPECIFIER = (
    'import',
    'import a',
    'import a from',
    'import a,',
    'import a, {',
    'import a, { b',
    'import {',
    'import { a',
    'import { a as b } from',
    'import * as ns from',
    'export *',
    'export * from',
)

#: Module declarations that name the module they read from, which the same host accepts. It answers
#: for `export { a as b };` alone, and for a name rather than for a spelling: `Export 'a' is not
#: defined in module`, where `var a; export { a as b };` is accepted.
MODULE_DECLARATIONS_THAT_NAME_THEIR_SPECIFIER = (
    'import a from "m";',
    'import "m";',
    'import { a as b } from "m";',
    'import * as ns from "m";',
    'import a, { b } from "m";',
    'import a from "m" with { type: "json" };',
    'export * from "m";',
    'export * as ns from "m";',
    'export { a as b } from "m";',
    'export { a as b };',
)


class TestAModuleDeclarationIsReadOnlyWhereItsSpecifierIsWritten(TestBase):
    """
    A module declaration names the module it reads from with a string literal, and a file that
    stops before that literal was written holds no declaration at all. Such a span is kept as a
    `refinery.lib.scripts.js.model.JsErrorNode` reading the text verbatim, so the tree reports that
    the file is not a program and what is printed for it is what was handed over.

    This was an entry of this ledger until the parser stopped answering such a file with a
    declaration carrying a specifier that no text spells, and it stays as the regression test that
    entry became.
    """

    @staticmethod
    def _well_formed(source: str) -> bool:
        return is_well_formed(JsParser(source).parse())

    @staticmethod
    def _printed(source: str) -> str:
        return JsSynthesizer().convert(JsParser(source).parse())

    def test_a_declaration_cut_before_its_specifier_is_not_a_well_formed_program(self):
        sources = MODULE_DECLARATIONS_CUT_BEFORE_THEIR_SPECIFIER
        self.assertEqual(
            {source: self._well_formed(source) for source in sources},
            {source: False for source in sources},
        )

    def test_a_declaration_cut_before_its_specifier_is_printed_as_the_source_wrote_it(self):
        sources = MODULE_DECLARATIONS_CUT_BEFORE_THEIR_SPECIFIER
        self.assertEqual(
            {source: self._printed(source) for source in sources},
            {source: source for source in sources},
        )

    def test_printing_a_declaration_cut_before_its_specifier_twice_writes_the_source_again(self):
        sources = MODULE_DECLARATIONS_CUT_BEFORE_THEIR_SPECIFIER
        self.assertEqual(
            {source: self._printed(self._printed(source)) for source in sources},
            {source: source for source in sources},
        )

    def test_a_declaration_that_names_its_specifier_is_a_program_printed_as_it_was_written(self):
        sources = MODULE_DECLARATIONS_THAT_NAME_THEIR_SPECIFIER
        self.assertEqual(
            {source: (self._well_formed(source), self._printed(source)) for source in sources},
            {source: (True, source) for source in sources},
        )
