# Licensed under the Apache License, Version 2.0; see LICENSE.
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from reproduce import prepare_output
import build_empirical_supplement as supplement


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

    def test_all_baseline_observations_fit_the_actual_plot_axes(self):
        # Exercise the real panel selection/axis calculation against retained
        # source tables; only drawing is replaced. The initial Base Thinking
        # own-proxy rating is below -100 and must stay visible.
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(supplement, "chart") as draw, patch.object(supplement, "new_canvas"):
                supplement.baseline_figure(Path(directory) / "baseline.pdf", ROOT / "results/raw")
        self.assertEqual(draw.call_count, 5)
        for call in draw.call_args_list:
            _, _, title, curves, xlim, ylim, *_ = call.args
            for curve in curves:
                for x, y in curve["points"]:
                    self.assertLessEqual(xlim[0], x, title)
                    self.assertLessEqual(x, xlim[1], title)
                    self.assertLessEqual(ylim[0], y, title)
                    self.assertLessEqual(y, ylim[1], title)


if __name__ == "__main__":
    unittest.main()
