from __future__ import annotations

import unittest

from refinery.lib.scripts.ps1 import data


class TestPs1MetadataReader(unittest.TestCase):
    """
    These exercise the reader against the shipped metadata rather than a fixture: the data is the
    genuine collected surface, and the point is that the reader's resolution agrees with it.
    """

    def test_the_schema_version_is_the_one_the_reader_expects(self):
        self.assertEqual(data._META['schema']['version'], data.SCHEMA_VERSION)
        self.assertTrue(data._META['authoritative'])

    def test_accelerators_resolve_to_their_full_type(self):
        cases = {
            'int'         : 'System.Int32',            # noqa
            'string'      : 'System.String',           # noqa
            'ref'         : 'System.Management.Automation.PSReference',
            'hashtable'   : 'System.Collections.Hashtable',
            'ipaddress'   : 'System.Net.IPAddress',    # noqa
            'scriptblock' : 'System.Management.Automation.ScriptBlock',
        }
        for accelerator, full in cases.items():
            self.assertEqual(data.resolve_type(accelerator), full, accelerator)

    def test_a_qualified_name_resolves_without_the_system_prefix(self):
        self.assertEqual(data.resolve_type('Net.WebClient'), 'System.Net.WebClient')
        self.assertEqual(data.resolve_type('System.Net.WebClient'), 'System.Net.WebClient')

    def test_a_mixed_case_accelerator_resolves(self):
        # The accelerator table is keyed in the casing PowerShell reports, which is mixed; resolving
        # one has to be case-insensitive or every capital-lettered accelerator silently fails.
        self.assertEqual(
            data.resolve_type('CimSession'), 'Microsoft.Management.Infrastructure.CimSession')
        self.assertEqual(
            data.resolve_type('X509Certificate'),
            'System.Security.Cryptography.X509Certificates.X509Certificate',
        )

    def test_a_generic_name_resolves_to_its_open_definition(self):
        self.assertEqual(
            data.resolve_type('Collections.Generic.List[int]'),
            'System.Collections.Generic.List`1',
        )
        self.assertEqual(
            data.resolve_type('System.Collections.Generic.Dictionary[string, object]'),
            'System.Collections.Generic.Dictionary`2',
        )

    def test_a_name_that_is_not_a_type_does_not_resolve(self):
        self.assertIsNone(data.resolve_type('NotARealType'))
        self.assertIsNone(data.resolve_type('q[1]'))
        self.assertIsNone(data.resolve_type(''))

    def test_canonical_type_is_resolve_type(self):
        self.assertEqual(data.canonical_type('int'), data.resolve_type('int'))

    def test_an_explicit_interface_member_is_present_with_its_type(self):
        # System.Array's Count comes from ICollection and is invisible to a bare GetProperties.
        members = data.type_members('array')
        self.assertIsNotNone(members)
        self.assertIn('Count', members)
        self.assertEqual(members['Count']['type'], 'System.Int32')
        self.assertEqual(members['Count']['source'], 'reflection')

    def test_an_extended_type_system_member_is_marked_as_such(self):
        # Process.Path is a ScriptProperty from types.ps1xml; reflection never reports it, and a
        # later purity gate must be able to tell it from a plain property.
        members = data.type_members('System.Diagnostics.Process')
        self.assertIsNotNone(members)
        self.assertEqual(members['Path']['kind'], 'ets_script_property')
        self.assertEqual(members['Path']['source'], 'ets')

    def test_member_order_is_present_for_a_type_with_an_instance(self):
        order = data.member_order('System.String')
        self.assertIsNotNone(order)
        self.assertIn('Length', order)

    def test_member_order_is_absent_for_a_type_without_an_instance(self):
        # SecurityProtocolType has no instance expression, so its Get-Member order was not observed.
        self.assertIsNone(data.member_order('System.Net.SecurityProtocolType'))

    def test_a_command_carries_its_module_and_common_flag(self):
        gci = data.command('Get-ChildItem')
        self.assertIsNotNone(gci)
        self.assertEqual(gci['module'], 'Microsoft.PowerShell.Management')
        self.assertIn('Path', gci['parameters'])
        self.assertFalse(gci['parameters']['Path']['common'])
        self.assertTrue(gci['parameters']['ErrorAction']['common'])

    def test_a_command_resolves_case_insensitively(self):
        # PowerShell resolves command names without regard to case, and an obfuscated script may
        # write any casing; the lookup must not depend on the collected PascalCase key.
        self.assertIsNotNone(data.command('get-childitem'))
        self.assertEqual(data.command('GET-CHILDITEM'), data.command('Get-ChildItem'))

    def test_an_unknown_command_is_none(self):
        self.assertIsNone(data.command('Definitely-NotACommand'))


class TestPs1MetadataViews(unittest.TestCase):
    """
    The historical view tables the deobfuscation transforms still read must stay well-formed.
    """

    def test_the_views_are_populated(self):
        self.assertGreater(len(data.TYPE_MEMBERS), 500)
        self.assertGreater(len(data.KNOWN_CMDLETS), 1000)
        self.assertGreater(len(data.ALL_PARAMETER_NAMES), 1000)

    def test_a_type_member_view_is_lowercased_and_canonical(self):
        members = data.MEMBER_LOOKUP['system.string']
        self.assertEqual(members['substring'], 'Substring')

    def test_common_parameters_are_excluded_from_the_parameter_views(self):
        # ALL_PARAMETER_NAMES drives a command-independent casing rewrite; the common parameters
        # are deliberately kept out of it.
        self.assertNotIn('erroraction', data.ALL_PARAMETER_NAMES)
        self.assertNotIn('verbose', data.ALL_PARAMETER_NAMES)

    def test_an_enum_view_lists_its_values_not_its_object_methods(self):
        members = data.TYPE_MEMBERS['system.net.securityprotocoltype']
        self.assertIn('Tls12', members)
        self.assertNotIn('GetType', members)

    def test_property_types_resolve_a_member_result_type(self):
        self.assertEqual(data.PROPERTY_TYPES[('system.array', 'length')], 'system.int32')
        self.assertEqual(data.PROPERTY_TYPES[('system.array', 'count')], 'system.int32')

    def test_module_provided_cim_aliases_are_present(self):
        # A pristine Get-Alias drops these module short forms; they are supplied so that a sample
        # using gcim/icim is still de-aliased to its (collected) target cmdlet.
        self.assertEqual(data.KNOWN_ALIAS['gcim'], 'Get-CimInstance')
        self.assertEqual(data.KNOWN_ALIAS['icim'], 'Invoke-CimMethod')
        self.assertIn('get-ciminstance', data.KNOWN_CMDLETS)

    def test_the_scope_only_pscmdlet_variable_type_is_present(self):
        self.assertEqual(
            data.VARIABLE_TYPES['pscmdlet'], 'system.management.automation.psscriptcmdlet')


if __name__ == '__main__':
    unittest.main()
