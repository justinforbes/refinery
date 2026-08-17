import io
import tarfile

from .. import TestUnitBase
from . import TAR_ARCHIVE


def _make_tar(name: str = 'test.txt', content: bytes = b'hello world') -> bytes:
    buf = io.BytesIO()
    with tarfile.open(mode='w', fileobj=buf) as t:
        info = tarfile.TarInfo(name=name)
        info.size = len(content)
        t.addfile(info, io.BytesIO(content))
    return buf.getvalue()


class TestCarveTar(TestUnitBase):

    def test_carve_single_tar(self):
        tar = _make_tar()
        # carve_tar uses data.find(b'ustar', offset) > 0 so tar must not start at byte 0
        data = b'\x00' * 32 + tar + b'\xFF' * 64
        unit = self.load()
        result = data | unit | []
        self.assertEqual(len(result), 1)
        # Verify the carved result is a valid tar
        with tarfile.open(mode='r', fileobj=io.BytesIO(bytes(result[0]))) as t:
            members = t.getnames()
        self.assertIn('test.txt', members)

    def test_carve_no_tar(self):
        data = b'\x00' + self.generate_random_buffer(512)
        unit = self.load()
        result = data | unit | []
        self.assertEqual(len(result), 0)

    def test_reports_the_span_of_the_carved_archive(self):
        for padding in (1, 3, 37, 200):
            with self.subTest(padding=padding):
                data = bytes(padding) + TAR_ARCHIVE
                result, = data | self.load() | []
                start, end = result['start'], result['end']
                self.assertEqual((start, end), (padding, padding + len(TAR_ARCHIVE)))
                self.assertEqual(bytes(data[start:end]), bytes(result))

    def test_concatenated_archives_are_carved_as_a_single_stream(self):
        """
        A tar archive ends in zero padding and `tarfile` reads concatenated archives as one stream,
        so two adjacent archives are reported as a single item starting at the first one.
        """
        data = bytes(10) + TAR_ARCHIVE + bytes(5) + TAR_ARCHIVE
        result, = data | self.load() | []
        self.assertEqual(result['start'], 10)
        self.assertGreater(len(result), len(TAR_ARCHIVE))
