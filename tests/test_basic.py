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

    def test_patch_torchvision(self):
        from lynx.compat import patch_torchvision
        import sys
        try:
            import torchvision.transforms._functional_tensor  # type: ignore
        except Exception:
            self.skipTest("torchvision not installed")

        sys.modules.pop('torchvision.transforms.functional_tensor', None)
        patch_torchvision()
        self.assertIn('torchvision.transforms.functional_tensor', sys.modules)

    def test_yt_download_env_restore(self):
        from lynx import download as dl
        from unittest import mock
        import os
        import shutil

        orig_tmp = os.environ.get('TMP')
        orig_temp = os.environ.get('TEMP')
        dld = Path('tests/dld')
        tmp = Path('tests/tmpdir')
        dld.mkdir(parents=True, exist_ok=True)
        tmp.mkdir(parents=True, exist_ok=True)

        class Dummy:
            def __init__(self, opts):
                self.opts = opts
            def __enter__(self):
                self.f = dld / 'file.mkv'
                self.f.write_text('x')
                return self
            def __exit__(self, *a):
                pass
            def extract_info(self, url, download=True):
                return {'title': 'file', 'id': 'id', 'ext': 'mkv'}
            def prepare_filename(self, info):
                return str(dld / 'file')

        with mock.patch('yt_dlp.YoutubeDL', Dummy):
            path = dl.yt_download('https://x', dld, tmp)
            self.assertTrue(path.exists())

        self.assertEqual(os.environ.get('TMP'), orig_tmp)
        self.assertEqual(os.environ.get('TEMP'), orig_temp)

        shutil.rmtree(dld)
        shutil.rmtree(tmp)

if __name__ == '__main__':
    unittest.main()
