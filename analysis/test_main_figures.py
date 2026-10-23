#!/usr/bin/env python3
"""Fast CPU-only checks for the source-preserving main-figure builder."""

import tempfile
import unittest
from pathlib import Path

import build_main_figures as figures


class SourcePreservingFigures(unittest.TestCase):
    def test_prefix_selection_is_exact_and_chronological(self):
        table = {"ans_4b_4b_20": 2, "ans_4b_14b_0": 99,
                 "ans_4b_4b_0": 0, "ans_4b_4b_10": 1}
        self.assertEqual(figures.select(table, "ans_4b_4b"), [
            (0, "ans_4b_4b_0", 0), (10, "ans_4b_4b_10", 1), (20, "ans_4b_4b_20", 2)])

    def test_duplicate_player_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            path.write_text("player,rating\nans_4b_4b_0,1\nans_4b_4b_0,2\n")
            with self.assertRaisesRegex(ValueError, "Duplicate player ID"):
                figures.ratings(path)

    def test_nonfinite_ratings_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nonfinite.csv"
            path.write_text("player,rating\nans_4b_4b_0,nan\n")
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                figures.ratings(path)

    def test_missing_gold_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "proxy.csv").write_text("player,rating\nans_4b_4b_0,1\nans_4b_4b_10,2\n")
            (folder / "gold.csv").write_text("player,rating\nans_4b_4b_0,7\n")
            with self.assertRaisesRegex(ValueError, "Missing Gold rows"):
                figures.joined(folder, "proxy", "ans_4b_4b", "test", "#000000")

    def test_exact_id_join_does_not_substitute_a_related_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "proxy.csv").write_text("player,rating\nans_4b_4b_10,2\nans_4b_4b_0,1\nans_4b_14b_0,99\n")
            (folder / "gold.csv").write_text("player,rating\nans_4b_4b_10,20\nans_4b_4b_0,10\nans_4b_14b_0,999\n")
            result = figures.joined(folder, "proxy", "ans_4b_4b", "test", "#000000")
            self.assertEqual(result["points"], [(1, 10, 0), (2, 20, 10)])

    def test_plot_bounds_retain_negative_exported_coordinates_and_every_point(self):
        values = [-.0014444727206219, .0082420004058446, .887]
        (low, high), ticks = figures.nice_bounds(values)
        self.assertLess(low, min(values))
        self.assertGreater(high, max(values))
        self.assertTrue(all(low <= value <= high for value in ticks))


if __name__ == "__main__":
    unittest.main()
