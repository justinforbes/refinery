from .. import TestUnitBase
from . import RTF_DOCUMENT


class TestCarveRTF(TestUnitBase):

    def test_carve_simple_rtf(self):
        unit = self.load()
        rtf = b'{\\rtf1\\ansi Hello World}'
        data = b'GARBAGE' + rtf + b'MORE GARBAGE'
        results = data | unit | []
        self.assertEqual(len(results), 1)
        self.assertEqual(bytes(results[0]), rtf)

    def test_carve_nested_rtf(self):
        unit = self.load()
        rtf = b'{\\rtf1\\ansi {\\b Bold} text}'
        data = b'PREFIX' + rtf + b'SUFFIX'
        results = data | unit | []
        self.assertEqual(len(results), 1)
        self.assertEqual(bytes(results[0]), rtf)

    def test_no_rtf_found(self):
        unit = self.load()
        data = b'This is just plain text with no RTF content.'
        results = data | unit | []
        self.assertEqual(len(results), 0)

    def test_multiple_rtf_documents(self):
        unit = self.load()
        rtf1 = b'{\\rtf1 first}'
        rtf2 = b'{\\rtf1 second}'
        data = rtf1 + b'BETWEEN' + rtf2
        results = data | unit | []
        self.assertEqual(len(results), 2)

    def test_reports_the_span_of_the_carved_document(self):
        for padding in (1, 3, 37, 200):
            with self.subTest(padding=padding):
                data = bytes(padding) + RTF_DOCUMENT
                result, = data | self.load() | []
                start, end = result['start'], result['end']
                self.assertEqual((start, end), (padding, padding + len(RTF_DOCUMENT)))
                self.assertEqual(bytes(data[start:end]), bytes(result))

    def test_reports_the_position_of_each_of_two_documents(self):
        data = bytes(10) + RTF_DOCUMENT + bytes(5) + RTF_DOCUMENT
        results = data | self.load() | []
        self.assertEqual(len(results), 2)
        self.assertListEqual(
            [result['start'] for result in results],
            [10, 10 + len(RTF_DOCUMENT) + 5])
