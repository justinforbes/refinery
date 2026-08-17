from .. import TestUnitBase
from . import DER_SEQUENCE


class TestCarveDER(TestUnitBase):

    def test_carve_simple_sequence(self):
        unit = self.load()
        # DER-encoded SEQUENCE containing two INTEGERs: 1 and 2
        der = DER_SEQUENCE
        data = b'\x00\x00' + der + b'\x00\x00'
        results = data | unit | []
        self.assertListEqual(results, [der])

    def test_no_der_found(self):
        unit = self.load()
        data = b'Just plain text without any DER sequences'
        results = data | unit | []
        self.assertEqual(len(results), 0)

    def test_skip_null_length(self):
        unit = self.load()
        # 0x30 followed by 0x00 should be skipped
        data = b'\x30\x00rest of data'
        results = data | unit | []
        self.assertEqual(len(results), 0)

    def test_reports_the_span_of_the_carved_sequence(self):
        for padding in (1, 3, 37, 200):
            with self.subTest(padding=padding):
                data = bytes(padding) + DER_SEQUENCE
                result, = data | self.load() | []
                start, end = result['start'], result['end']
                self.assertEqual((start, end), (padding, padding + len(DER_SEQUENCE)))
                self.assertEqual(data[start:end], result)

    def test_reports_the_position_of_each_of_two_sequences(self):
        data = bytes(10) + DER_SEQUENCE + bytes(5) + DER_SEQUENCE
        results = data | self.load() | []
        self.assertEqual(len(results), 2)
        self.assertListEqual(
            [result['start'] for result in results],
            [10, 10 + len(DER_SEQUENCE) + 5])
