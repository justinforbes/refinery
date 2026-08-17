from .. import TestUnitBase
from . import SEVENZIP_ARCHIVE


class TestCarve7Zip(TestUnitBase):

    def test_01(self):
        from refinery import carve_7z as unit

        data = SEVENZIP_ARCHIVE
        for prefix in (
            unit.HEADER_SIGNATURE + self.generate_random_buffer(20),
            unit.HEADER_SIGNATURE * 3,
            self.generate_random_buffer(64) + unit.HEADER_SIGNATURE + b'AAAA'
        ):
            for pf in (2, 4, 90, 200, 2048):
                blob = prefix + data + self.generate_random_buffer(pf)
                self.assertEqual(blob | unit | bytearray, data)

    def test_reports_the_span_of_the_carved_archive(self):
        for padding in (1, 3, 37, 200):
            with self.subTest(padding=padding):
                data = bytes(padding) + SEVENZIP_ARCHIVE
                result, = data | self.load() | []
                start, end = result['start'], result['end']
                self.assertEqual((start, end), (padding, padding + len(SEVENZIP_ARCHIVE)))
                self.assertEqual(bytes(data[start:end]), bytes(result))

    def test_reports_the_position_of_each_of_two_archives(self):
        data = bytes(10) + SEVENZIP_ARCHIVE + bytes(5) + SEVENZIP_ARCHIVE
        results = data | self.load() | []
        self.assertEqual(len(results), 2)
        self.assertListEqual(
            [result['start'] for result in results],
            [10, 10 + len(SEVENZIP_ARCHIVE) + 5])
