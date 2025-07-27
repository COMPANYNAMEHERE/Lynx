import unittest
from pathlib import Path
from lynx.download import is_url
try:
    from lynx.upscale import pick_model
except Exception:  # missing heavy deps
    pick_model = None  # type: ignore
from lynx.models import _sha256_of_file

class TestHelpers(unittest.TestCase):
    def test_is_url(self):
        self.assertTrue(is_url('https://example.com/video'))
        self.assertFalse(is_url('/local/path.mp4'))

    def test_pick_model(self):
        if pick_model is None:
            self.skipTest('RealESRGAN dependencies missing')
        self.assertEqual(pick_model(1.5), ('RealESRGAN_x2plus.pth', 2))
        self.assertEqual(pick_model(4.0), ('RealESRGAN_x4plus.pth', 4))

    def test_sha256_of_file(self):
        tmp = Path('tests/tmp.txt')
        tmp.write_text('hello')
        try:
            self.assertEqual(_sha256_of_file(tmp), '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824')
        finally:
            tmp.unlink()

if __name__ == '__main__':
    unittest.main()
