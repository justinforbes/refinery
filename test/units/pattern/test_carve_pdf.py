from .. import TestUnitBase
from . import PDF_DOCUMENT


class TestCarvePdf(TestUnitBase):

    def test_single_pdf(self):
        pdf = b'%PDF-1.4\n1 0 obj\n<< >>\nendobj\n%%EOF\n'
        data = b'junk' + pdf + b'more junk'
        unit = self.load()
        results = data | unit | []
        self.assertEqual(len(results), 1)
        self.assertEqual(bytes(results[0]), pdf)

    def test_multiple_pdfs(self):
        pdf1 = b'%PDF-1.4\nsome content\n%%EOF\n'
        pdf2 = b'%PDF-1.7\nother content\n%%EOF\n'
        data = b'header' + pdf1 + b'middle' + pdf2 + b'trailer'
        unit = self.load()
        results = data | unit | []
        self.assertEqual(len(results), 2)
        self.assertEqual(bytes(results[0]), pdf1)
        self.assertEqual(bytes(results[1]), pdf2)

    def test_no_pdf(self):
        data = b'This is not a PDF file at all.'
        unit = self.load()
        results = data | unit | []
        self.assertEqual(len(results), 0)

    def test_pdf_at_start(self):
        pdf = b'%PDF-1.5\ncontent here\n%%EOF'
        unit = self.load()
        results = pdf | unit | []
        self.assertEqual(len(results), 1)
        self.assertEqual(bytes(results[0]), pdf)

    def test_pdf_without_eof(self):
        data = b'%PDF-1.4\nsome content without trailer'
        unit = self.load()
        results = data | unit | []
        self.assertEqual(len(results), 0)

    def test_reports_the_span_of_the_carved_document(self):
        for padding in (1, 3, 37, 200):
            with self.subTest(padding=padding):
                data = bytes(padding) + PDF_DOCUMENT
                result, = data | self.load() | []
                start, end = result['start'], result['end']
                self.assertEqual((start, end), (padding, padding + len(PDF_DOCUMENT)))
                self.assertEqual(bytes(data[start:end]), bytes(result))

    def test_reports_the_position_of_each_of_two_documents(self):
        data = bytes(10) + PDF_DOCUMENT + bytes(5) + PDF_DOCUMENT
        results = data | self.load() | []
        self.assertEqual(len(results), 2)
        self.assertListEqual(
            [result['start'] for result in results],
            [10, 10 + len(PDF_DOCUMENT) + 5])

    def test_eof_with_crlf(self):
        pdf = b'%PDF-1.4\ncontent\n%%EOF\r\n'
        data = b'junk' + pdf + b'more'
        unit = self.load()
        results = data | unit | []
        self.assertEqual(len(results), 1)
        self.assertEqual(bytes(results[0]), pdf)
