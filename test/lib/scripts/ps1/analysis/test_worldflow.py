from __future__ import annotations

from test import TestBase

from refinery.lib.scripts import set_body
from refinery.lib.scripts.ps1.analysis.cache import Ps1ModelCache
from refinery.lib.scripts.ps1.parser import Ps1Parser


class TestPs1WorldReachIsBoundToTheTreeItMeasured(TestBase):

    def test_a_positional_grant_is_withdrawn_once_the_tree_changes(self):
        script = Ps1Parser('$Null = [Math]::Sqrt(144)\nInvoke-Expression $c').parse()
        reach = Ps1ModelCache(script).world_reach
        read = script.body[0]
        self.assertFalse(reach.closed_for_the_whole_run)
        self.assertTrue(reach.closed_at(read))
        set_body(script, [*script.body, Ps1Parser('Write-Host done').parse().body[0]])
        self.assertFalse(reach.closed_at(read))
