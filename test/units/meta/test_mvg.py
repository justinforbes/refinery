#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from . import TestMetaBase
from refinery.lib.loader import load_pipeline as L


class TestMetaVarGlobal(TestMetaBase):

    def test_mvg_01(self):
        pl = L('emit FOO [| nop [| put x BAR | mvg ]| cca var:x ]')
        self.assertEqual(pl(), B'FOOBAR')

    def test_mvg_02(self):
        pl = L('emit FOO [| nop [[| put x BAR | mvg ]| nop ]| cfmt {}{x} ]')
        self.assertEqual(pl(), B'FOO{x}')

    def test_mvg_03(self):
        pl = L('emit FOO [| nop [[| put x BAR | mvg -t ]| nop ]| cfmt {}{x} ]')
        self.assertEqual(pl(), B'FOOBAR')

    def test_mvg_04(self):
        pl = L('emit FOO [| rex . [| put x | mvg ]| cfmt {}{x} ]')
        self.assertEqual(pl(), B'FOO{x}')

    def test_scope_cannot_be_increased(self):
        pl = L('emit s: [| put x alpha [| nop [| put x beta | mvg ]| nop ]| cfmt {x} ]')
        self.assertEqual(pl(), B'alpha')
