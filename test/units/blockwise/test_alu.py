#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .. import TestUnitBase


class TestALU(TestUnitBase):

    def test_index_starts_at_zero(self):
        unit = self.load("B+K")
        self.assertEqual(bytes(bytes(5) | unit), bytes(range(5)))

    def test_real_world_01(self):
        data = bytes.fromhex('7C737376545F0808244368244668652724684643275F2424054B')
        goal = b'https'b'://45.41.204'b'.150:443\0'
        unit = self.load('(41*(B-75))%127')
        self.assertEqual(data | unit | bytes, goal)

    def test_signed_unsigned_methods(self):
        data = bytes.fromhex(
            '1D 83 46 28 43 76 56 55 56 E3 8D BF F0 54 AA A9 1D DB A6 21 AE 00 AB 1D'
            '7D 9F 1E 28 C1 5D 48 A8 9D 1E D6 92 83 E9 7C 01 AB 55 9A 99 1E 20 04 8E')
        goal = bytes.fromhex(
            '48 83 EC 28 E8 23 00 00 00 48 8D 15 F0 FF FF FF 48 8D 0D 21 04 00 00 48'
            '2B CA 48 83 C1 F7 48 03 C8 48 83 C4 28 E9 D6 01 00 00 CC CC 48 8B 04 24')
        unit = self.load('B@S', seed=14120, prologue='U(I(S<<9)/3)', precision=4)
        self.assertEqual(data | unit | bytearray, goal)


class TestALUAgainstOtherUnits(TestUnitBase):

    def setUp(self):
        super().setUp()
        self.buffer = self.generate_random_buffer(1024)
        self.arg = 'BAADF00D'

    def test_against_add(self):
        bop = self.load('B + A', self.arg)
        add = self.ldu('add', self.arg)
        self.assertEqual(add(self.buffer), bop(self.buffer))

    def test_against_sub(self):
        sub = self.ldu('sub', self.arg)
        bop = self.load('B - A', self.arg)
        self.assertEqual(sub(self.buffer), bop(self.buffer))

    def test_against_xor_01(self):
        xor = self.ldu('xor', self.arg)
        bop = self.load('B ^ A', self.arg)
        self.assertEqual(xor(self.buffer), bop(self.buffer))

    def test_against_xor_02(self):
        xor = self.ldu('xor', self.arg)
        bop = self.load('(~B & A) | (B & ~A)', self.arg)
        self.assertEqual(xor(self.buffer), bop(self.buffer))

    def test_against_xor_03(self):
        xor = self.ldu('xor', self.arg)
        bop = self.load('(A | B) & ~(B & A)', self.arg)
        self.assertEqual(xor(self.buffer), bop(self.buffer))

    def test_against_shl(self):
        shl = self.ldu('shl', '3')
        bop = self.load('B << 3')
        self.assertEqual(shl(self.buffer), bop(self.buffer))

    def test_against_shr(self):
        shr = self.ldu('shr', '3')
        bop = self.load('B >> 3')
        self.assertEqual(shr(self.buffer), bop(self.buffer))

    def test_against_ror(self):
        ror = self.ldu('rotr', '3')
        bop = self.load('(B >> 3) | (B << 5)')
        self.assertEqual(ror(self.buffer), bop(self.buffer))
