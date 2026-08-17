from ... import TestUnitBase


class TestMicrosoftKeyBlob(TestUnitBase):

    def test_plaintext_des(self):
        wish = self.generate_random_buffer(0x54)
        data = bytes.fromhex(
            '08'        # PLAINTEXTBLOB
            '02'        # version2
            '0000'      # reserved
            '01660000'  # CALG_DES
            '54000000'  # size of key
        ) + wish
        unit = self.load()
        self.assertEqual(unit(data), wish)

    def test_simpleblob_aes(self):
        wish = self.generate_random_buffer(0x100)
        data = bytes.fromhex(
            '01'
            '02'
            '0000'
            '10660000'
            '00A40000'
        ) + wish
        unit = self.load()
        self.assertEqual(unit(data), wish)

    def test_blob_structure_is_reported_as_type_and_algorithm(self):
        data = bytes.fromhex(
            '08'
            '02'
            '0000'
            '01660000'
            '54000000'
        ) + self.generate_random_buffer(0x54)
        result, = data | self.load() | []
        self.assertEqual(str(result['type']), 'PLAINTEXTKEYBLOB')
        self.assertEqual(str(result['algorithm']), 'CALG_DES')
