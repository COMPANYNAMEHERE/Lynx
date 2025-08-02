import unittest
from pathlib import Path
from lynx.download import is_url
try:
    from lynx.upscale import pick_model_by_quality
except Exception:  # missing heavy deps
    pick_model_by_quality = None  # type: ignore
from lynx.models import _sha256_of_file

class TestHelpers(unittest.TestCase):
    def test_is_url(self):
        self.assertTrue(is_url('https://example.com/video'))
        self.assertFalse(is_url('/local/path.mp4'))

    def test_pick_model_by_quality(self):
        if pick_model_by_quality is None:
            self.skipTest('RealESRGAN dependencies missing')
        self.assertEqual(pick_model_by_quality('quick'), ('realesr-general-x4v3.pth', 4))
        self.assertEqual(pick_model_by_quality('normal'), ('Swin2SR_ClassicalSR_X4_64.pth', 4))

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

    def test_detect_gpu_info(self):
        from lynx.env import detect_gpu_info
        from unittest import mock

        with mock.patch('shutil.which') as m_which:
            m_which.side_effect = lambda c: '/usr/bin/nvidia-smi' if c == 'nvidia-smi' else None
            with mock.patch('subprocess.check_output', return_value='GPU 0: Fake'):
                self.assertEqual(detect_gpu_info(), 'GPU 0: Fake')

        with mock.patch('shutil.which', return_value=None):
            self.assertIsNone(detect_gpu_info())

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
            path, title = dl.yt_download('https://x', dld, tmp)
            self.assertTrue(path.exists())
            self.assertEqual(title, 'file')

        self.assertEqual(os.environ.get('TMP'), orig_tmp)
        self.assertEqual(os.environ.get('TEMP'), orig_temp)

        shutil.rmtree(dld)
        shutil.rmtree(tmp)

    def test_cli_parse_args(self):
        from lynx.cli import parse_args

        args = parse_args(["in.mp4", "-o", "out.mp4", "--width", "1280", "--quality", "better"])
        self.assertEqual(args.input, "in.mp4")
        self.assertEqual(args.output, "out.mp4")
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.quality, "better")

if __name__ == '__main__':
    unittest.main()
