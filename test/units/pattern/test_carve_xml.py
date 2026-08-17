from .. import TestUnitBase
from . import XML_DOCUMENT


class TestCarveXML(TestUnitBase):

    def test_wikipedia_unicode_example(self):
        xstr = '<?xml version="1.0" encoding="UTF-8"?><俄语 լեզու="ռուսերեն">данные</俄语>'
        unit = self.load()
        norm = xstr.encode(unit.codec)
        for encoding in ['UTF8', 'UTF-16LE']:
            xbin = xstr.encode(encoding)
            data = self.generate_random_buffer(200) + xbin + self.generate_random_buffer(100)
            self.assertEqual(unit(data), norm)

    def test_reports_the_span_of_the_carved_document(self):
        for padding in (1, 3, 37, 200):
            with self.subTest(padding=padding):
                data = bytes(padding) + XML_DOCUMENT
                result, = data | self.load() | []
                start, end = result['start'], result['end']
                self.assertEqual((start, end), (padding, padding + len(XML_DOCUMENT)))
                self.assertEqual(bytes(data[start:end]), bytes(result))

    def test_reports_the_span_of_the_encoded_source(self):
        """
        The unit re-encodes the document it finds, so the reported span describes the region in the
        input rather than the size of the output chunk.
        """
        source = XML_DOCUMENT.decode().encode('utf-16le')
        data = bytes(40) + source
        result, = data | self.load() | []
        self.assertEqual((result['start'], result['end']), (40, 40 + len(source)))
        self.assertEqual(bytes(result), XML_DOCUMENT)

    def test_reports_the_position_of_each_of_two_documents(self):
        data = bytes(10) + XML_DOCUMENT + bytes(5) + XML_DOCUMENT
        results = data | self.load() | []
        self.assertEqual(len(results), 2)
        self.assertListEqual(
            [result['start'] for result in results],
            [10, 10 + len(XML_DOCUMENT) + 5])
