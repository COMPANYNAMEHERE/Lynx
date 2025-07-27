"""Simple self-test runner for Lynx."""
import unittest

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover('tests')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
