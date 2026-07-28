from .. import TestUnitBase


class TestUniversalDeobfuscator(TestUnitBase):
    """
    The universal unit selects a backend by parse quality and then has to hand that backend the
    options the caller gave it. Only the switches are pinned here; what each pipeline does with them
    is its own suite's subject.
    """

    def test_the_powershell_backend_is_selected_and_strips_console_output(self):
        unit = self.load()
        self.assertEqual(bytes(b"'TVqQAAMA'\nWrite-Host 'go'" | unit), b"Write-Host 'go'")

    def test_the_output_switch_reaches_the_powershell_backend(self):
        # Regression: the unit called the backend with no options at all, so the PowerShell default
        # of deleting a statement that only writes to the console was unconditional here and had no
        # counterpart to `ps1 -k`. An analyst reaching for the universal unit — the natural first
        # move when the language is unknown — lost payload strings with no switch and no warning.
        unit = self.load(keep_output=True)
        self.assertEqual(
            bytes(b"'TVqQAAMA'\nWrite-Host 'go'" | unit), b"'TVqQAAMA'\nWrite-Host 'go'")

    def test_a_backend_with_no_such_notion_is_not_handed_the_switch(self):
        # JavaScript and VBA pipelines take no such keyword, so passing one would raise rather than
        # be ignored. Both settings have to reach them unchanged.
        for keep in (False, True):
            with self.subTest(keep_output=keep):
                unit = self.load(keep_output=keep)
                self.assertEqual(bytes(b'var x = 1; console.log(x);' | unit), b'console.log(1);')
                self.assertEqual(
                    bytes(b'Sub Main()\nMsgBox "hi"\nEnd Sub' | unit),
                    b'Sub Main()\n  MsgBox "hi"\nEnd Sub')
