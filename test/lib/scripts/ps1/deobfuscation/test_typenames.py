from __future__ import annotations

from inspect import cleandoc

from test.lib.scripts.ps1.deobfuscation import TestPs1

from refinery.lib.scripts.ps1.deobfuscation import Ps1TypeSystemSimplifications


class TestPs1TypeSystemSimplifications(TestPs1):

    def test_get_member_index_name_resolved(self):
        result = self._deobfuscate('($ExecutionContext | Get-Member)[6].Name')
        self.assertIn('InvokeCommand', result)
        self.assertNotIn('[6]', result)

    def test_get_member_index_unknown_type_preserved(self):
        result = self._deobfuscate('($unknown | Get-Member)[6].Name')
        self.assertIn('[6].Name', result)

    def test_get_member_index_out_of_range_preserved(self):
        result = self._deobfuscate('($ExecutionContext | Get-Member)[999].Name')
        self.assertIn('[999].Name', result)

    def test_name_on_string_literal_stripped(self):
        result = self._deobfuscate("$x.('GetCmdlets'.Name)('*w-*ct')")
        self.assertNotIn('.Name', result)
        self.assertIn('New-Object', result)


class TestPs1MemberFoldsFollowTheWriteAReadObserves(TestPs1):
    """
    Which write a name's type is taken from, at the pass that spells members out. A name is not a
    variable: the same spelling in two bodies is two of them, and a read outside the body that
    writes it observes nothing that body did.
    """

    def test_a_function_bodys_write_does_not_fold_a_get_member_index_outside_it(self):
        """
        Nothing says `f` was ever called, so the top-level `$q` holds `$null` and `Get-Member` over
        `$null` is an error rather than a member list.
        """
        self._assertUnchanged(cleandoc("""
            function f {
              $q = New-Object Net.WebClient
            }
            ($q | Get-Member)[0].Name
        """), Ps1TypeSystemSimplifications)

    def test_each_function_body_spells_its_members_from_its_own_write(self):
        result = self._apply(cleandoc("""
            function f {
              $q = New-Object Net.WebClient
              $q.downloadstring('u')
            }
            function g {
              $q = New-Object Text.StringBuilder
              $q.tostring()
            }
        """), Ps1TypeSystemSimplifications)
        self.assertEqual(result, cleandoc("""
            function f {
              $q = New-Object Net.WebClient
              $q.DownloadString('u')
            }
            function g {
              $q = New-Object Text.StringBuilder
              $q.ToString()
            }
        """))

    def test_a_store_through_a_member_leaves_the_member_spelling_resolvable(self):
        result = self._apply(cleandoc("""
            $q = New-Object Net.WebClient
            $q.Proxy = $null
            $q.downloadstring('u')
        """), Ps1TypeSystemSimplifications)
        self.assertEqual(result, cleandoc("""
            $q = New-Object Net.WebClient
            $q.Proxy = $null
            $q.DownloadString('u')
        """))

    def test_a_write_and_a_read_inside_one_subexpression_resolve(self):
        result = self._apply(
            "$($q = New-Object Net.WebClient; $q.downloadstring('u'))",
            Ps1TypeSystemSimplifications,
        )
        self.assertEqual(result, cleandoc("""
            $($q = New-Object Net.WebClient
            $q.DownloadString('u'))
        """))

    def test_a_get_member_index_before_every_write_of_the_name_is_left_alone(self):
        """
        The name holds what it held before the script ran, and `Get-Member` over that `$null` is an
        error rather than a member list, so no index into it names a member.
        """
        self._assertUnchanged(cleandoc("""
            ($q | Get-Member)[0].Name
            $q = New-Object Net.WebClient
        """), Ps1TypeSystemSimplifications)

    def test_a_read_at_the_top_of_a_loop_body_is_left_alone(self):
        self._assertUnchanged(cleandoc("""
            while ($c) {
              $q.downloadstring('u')
              $q = New-Object Net.WebClient
            }
        """), Ps1TypeSystemSimplifications)

    def test_a_loop_body_read_resolves_once_a_write_before_the_loop_reaches_it(self):
        result = self._apply(cleandoc("""
            $q = New-Object Net.WebClient
            while ($c) {
              $q.downloadstring('u')
              $q = New-Object Net.WebClient
            }
        """), Ps1TypeSystemSimplifications)
        self.assertEqual(result, cleandoc("""
            $q = New-Object Net.WebClient
            while ($c) {
              $q.DownloadString('u')
              $q = New-Object Net.WebClient
            }
        """))

    def test_a_write_on_one_branch_only_does_not_fold_the_read_after_the_branch(self):
        self._assertUnchanged(cleandoc("""
            if ($c) {
              $q = New-Object Net.WebClient
            }
            ($q | Get-Member)[0].Name
        """), Ps1TypeSystemSimplifications)

    def test_a_write_before_the_branch_folds_the_read_after_it(self):
        result = self._apply(cleandoc("""
            $q = New-Object Net.WebClient
            if ($c) {
              $q = New-Object Net.WebClient
            }
            ($q | Get-Member)[0].Name
        """), Ps1TypeSystemSimplifications)
        self.assertEqual(result, cleandoc("""
            $q = New-Object Net.WebClient
            if ($c) {
              $q = New-Object Net.WebClient
            }
            'CancelAsync'
        """))

    def test_a_qualified_write_inside_a_body_does_not_fold_a_top_level_read(self):
        self._assertUnchanged(cleandoc("""
            function f {
              $script:q = New-Object Net.WebClient
            }
            ($q | Get-Member)[0].Name
        """), Ps1TypeSystemSimplifications)

    def test_a_qualified_write_folds_the_reads_that_follow_it_in_its_own_body(self):
        result = self._apply(cleandoc("""
            function f {
              $script:q = New-Object Net.WebClient
              ($q | Get-Member)[0].Name
            }
        """), Ps1TypeSystemSimplifications)
        self.assertEqual(result, cleandoc("""
            function f {
              $script:q = New-Object Net.WebClient
              'CancelAsync'
            }
        """))

    def test_a_read_between_two_writes_of_different_types_reads_the_first(self):
        result = self._apply(cleandoc("""
            $q = New-Object Net.WebClient
            $q.downloadstring('u')
            $q = New-Object Text.StringBuilder
            $q.tostring()
        """), Ps1TypeSystemSimplifications)
        self.assertEqual(result, cleandoc("""
            $q = New-Object Net.WebClient
            $q.DownloadString('u')
            $q = New-Object Text.StringBuilder
            $q.ToString()
        """))

    def test_a_read_two_writes_of_different_types_reach_is_left_alone(self):
        self._assertUnchanged(cleandoc("""
            $q = New-Object Net.WebClient
            if ($c) {
              $q = New-Object Text.StringBuilder
            }
            $q.tostring()
        """), Ps1TypeSystemSimplifications)

    def test_an_unattributable_write_before_the_read_leaves_the_member_alone(self):
        self._assertUnchanged(cleandoc("""
            $q = New-Object Net.WebClient
            Invoke-Expression $code
            $q.downloadstring('u')
        """), Ps1TypeSystemSimplifications)

    def test_an_unattributable_write_after_the_read_leaves_the_member_resolvable(self):
        result = self._apply(cleandoc("""
            $q = New-Object Net.WebClient
            $q.downloadstring('u')
            Invoke-Expression $code
        """), Ps1TypeSystemSimplifications)
        self.assertEqual(result, cleandoc("""
            $q = New-Object Net.WebClient
            $q.DownloadString('u')
            Invoke-Expression $code
        """))

    def test_a_read_through_a_scope_qualifier_leaves_the_member_alone(self):
        self._assertUnchanged(cleandoc("""
            $global:q = New-Object Net.WebClient
            $global:q.downloadstring('u')
        """), Ps1TypeSystemSimplifications)

    def test_a_foreach_over_a_string_spells_a_string_member(self):
        """
        `Substring` is a member of the string itself; the characters `foreach` is often assumed to
        yield have no such member at all.
        """
        result = self._apply(cleandoc("""
            foreach ($s in 'abc') {
              $s.substring(0, 1)
            }
        """), Ps1TypeSystemSimplifications)
        self.assertEqual(result, cleandoc("""
            foreach ($s in 'abc') {
              $s.Substring(0, 1)
            }
        """))
