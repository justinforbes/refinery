from __future__ import annotations

import unittest

from refinery.lib.scripts.ps1.dotnet import Ps1TypeName, parse_type_name


class TestPs1TypeNameGrammar(unittest.TestCase):

    def test_a_plain_name_is_its_own_definition(self):
        parsed = parse_type_name('System.Int32')
        self.assertEqual(parsed, Ps1TypeName('System.Int32'))
        self.assertEqual(parsed.definition, 'System.Int32')
        self.assertEqual(str(parsed), 'System.Int32')

    def test_the_grammar_does_not_resolve_names(self):
        for name in ('int', 'Int32', 'System.Int32', 'ref', 'Collections.ArrayList'):
            parsed = parse_type_name(name)
            self.assertIsNotNone(parsed)
            self.assertEqual(parsed.name, name)
            self.assertEqual(parsed.arity, 0)

    def test_source_and_reflection_form_agree(self):
        source = parse_type_name('System.Collections.Generic.List[System.Int32]')
        reflect = parse_type_name('System.Collections.Generic.List`1[[System.Int32]]')
        self.assertIsNotNone(source)
        self.assertIsNotNone(reflect)
        self.assertEqual(source.definition, 'System.Collections.Generic.List`1')
        self.assertEqual(source.definition, reflect.definition)
        self.assertEqual(source.arguments, reflect.arguments)
        self.assertEqual(source.arguments, (Ps1TypeName('System.Int32'),))

    def test_an_open_generic_has_arity_without_arguments(self):
        parsed = parse_type_name('System.Collections.Generic.List`1')
        self.assertEqual(parsed.name, 'System.Collections.Generic.List')
        self.assertEqual(parsed.arity, 1)
        self.assertEqual(parsed.arguments, ())
        self.assertEqual(parsed.definition, 'System.Collections.Generic.List`1')
        self.assertEqual(str(parsed), 'System.Collections.Generic.List`1')

    def test_arity_is_inferred_from_the_argument_count(self):
        parsed = parse_type_name('Collections.Generic.Dictionary[string, int]')
        self.assertEqual(parsed.arity, 2)
        self.assertEqual(parsed.definition, 'Collections.Generic.Dictionary`2')
        self.assertEqual(parsed.arguments, (Ps1TypeName('string'), Ps1TypeName('int')))

    def test_an_arity_marker_contradicting_the_arguments_is_rejected(self):
        self.assertIsNone(parse_type_name('Collections.Generic.List`2[int]'))
        self.assertIsNone(parse_type_name('Collections.Generic.Dictionary`2[int]'))

    def test_generic_arguments_nest(self):
        parsed = parse_type_name('Dictionary[string, List[Byte[]]]')
        self.assertEqual(parsed.arity, 2)
        inner = parsed.arguments[1]
        self.assertEqual(inner.name, 'List')
        self.assertEqual(inner.arity, 1)
        self.assertEqual(inner.arguments[0], Ps1TypeName('Byte', ranks=(1,)))

    def test_assembly_qualification_is_captured_and_not_confused_with_arguments(self):
        parsed = parse_type_name(
            'System.Collections.Generic.List`1'
            '[[System.Int32, mscorlib, Version=4.0.0.0, Culture=neutral,'
            ' PublicKeyToken=b77a5c561934e089]], mscorlib, Version=4.0.0.0'
        )
        self.assertEqual(parsed.definition, 'System.Collections.Generic.List`1')
        self.assertEqual(parsed.assembly, 'mscorlib, Version=4.0.0.0')
        self.assertEqual(len(parsed.arguments), 1)
        argument = parsed.arguments[0]
        self.assertEqual(argument.name, 'System.Int32')
        self.assertEqual(
            argument.assembly,
            'mscorlib, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089',
        )

    def test_array_suffixes_are_recorded_in_source_order(self):
        cases = {
            'int[]'      : (1,),          # noqa
            'int[,]'     : (2,),          # noqa
            'int[,,]'    : (3,),          # noqa
            'int[][]'    : (1, 1),        # noqa
            'int[,][]'   : (2, 1),        # noqa
        }
        for text, ranks in cases.items():
            parsed = parse_type_name(text)
            self.assertIsNotNone(parsed, text)
            self.assertEqual(parsed.name, 'int', text)
            self.assertEqual(parsed.ranks, ranks, text)
            self.assertTrue(parsed.is_array, text)
            self.assertEqual(str(parsed), text, text)

    def test_a_generic_type_can_be_an_array(self):
        parsed = parse_type_name('Collections.Generic.List[int][]')
        self.assertEqual(parsed.definition, 'Collections.Generic.List`1')
        self.assertEqual(parsed.arguments, (Ps1TypeName('int'),))
        self.assertEqual(parsed.ranks, (1,))

    def test_a_byref_parameter_type_is_recognized(self):
        parsed = parse_type_name('System.Int32&')
        self.assertEqual(parsed.name, 'System.Int32')
        self.assertTrue(parsed.byref)
        self.assertEqual(parsed.definition, 'System.Int32')
        self.assertEqual(str(parsed), 'System.Int32&')

    def test_an_array_passed_by_reference_carries_both_suffixes(self):
        parsed = parse_type_name('System.Byte[]&')
        self.assertEqual(parsed.name, 'System.Byte')
        self.assertEqual(parsed.ranks, (1,))
        self.assertTrue(parsed.byref)

    def test_pointer_suffixes_are_counted(self):
        parsed = parse_type_name('System.Void**')
        self.assertEqual(parsed.name, 'System.Void')
        self.assertEqual(parsed.pointers, 2)
        self.assertFalse(parsed.byref)

    def test_suffixes_out_of_runtime_order_are_rejected(self):
        for text in ('System.Int32*[]', 'System.Int32&[]', 'System.Int32&*', 'System.Int32&&'):
            self.assertIsNone(parse_type_name(text), text)

    def test_a_nested_type_keeps_its_declaring_type(self):
        parsed = parse_type_name('System.Environment+SpecialFolder')
        self.assertEqual(parsed.name, 'System.Environment+SpecialFolder')
        self.assertEqual(parsed.definition, 'System.Environment+SpecialFolder')

    def test_whitespace_between_name_parts_is_insignificant(self):
        spaced = parse_type_name('  System . Collections . Generic . List [ int ] ')
        self.assertEqual(spaced, parse_type_name('System.Collections.Generic.List[int]'))

    def test_names_that_are_not_type_names_are_rejected(self):
        for text in (
            '',
            '   ',
            '1',
            '1Type',
            'q[1]',
            'System.Int32[',
            'System.Int32]',
            'List[int',
            'List[int]]',
            'List[]extra',
            'List[,int]',
            'System.Int32, ',
            '[System.Int32]',
            'System..Int32',
            'System.',
            '.Int32',
            'Foo`',
            'Foo`x',
        ):
            self.assertIsNone(parse_type_name(text), text)

    def test_the_source_form_round_trips(self):
        for text in (
            'System.Int32',
            'int[]',
            'Byte[,]',
            'System.Collections.Generic.List`1',
            'System.Collections.Generic.List[System.Int32]',
            'Dictionary[String, List[Byte[]]][]',
            'System.Environment+SpecialFolder',
            'System.Int32&',
            'System.Byte[]&',
            'System.Void**',
        ):
            parsed = parse_type_name(text)
            self.assertIsNotNone(parsed, text)
            self.assertEqual(str(parsed), text, text)
            self.assertEqual(parse_type_name(str(parsed)), parsed, text)

    def test_a_parsed_name_is_hashable(self):
        names = {parse_type_name('List[int]'), parse_type_name('Collections.Generic.List[int]')}
        self.assertEqual(len(names), 2)
        self.assertIn(parse_type_name('List[ int ]'), names)


if __name__ == '__main__':
    unittest.main()
