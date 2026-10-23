# Licensed under the Apache License, Version 2.0; see LICENSE.
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from reproduce import prepare_output


class ReproductionOutputTests(unittest.TestCase):
    def test_repository_root_is_not_an_output_directory(self):
        with self.assertRaises(ValueError):
            prepare_output(ROOT, ROOT)

    def test_source_directory_is_not_an_output_directory(self):
        with self.assertRaises(ValueError):
            prepare_output(ROOT, ROOT / "analysis")

    def test_nonempty_external_directory_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "unrelated.txt").write_text("preserve")
            with self.assertRaises(ValueError):
                prepare_output(ROOT, path)
            self.assertEqual((path / "unrelated.txt").read_text(), "preserve")

    def test_managed_directory_can_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            prepare_output(ROOT, path)
            (path / "generated.txt").write_text("example")
            prepare_output(ROOT, path)
            self.assertTrue((path / ".genrm-reproduction.json").exists())


if __name__ == "__main__":
    unittest.main()
