# Licensed under the Apache License, Version 2.0; see LICENSE.
import csv
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import fit_size_scaling as scaling


class SizeScalingTests(unittest.TestCase):
    def test_exact_matched_configuration_count(self):
        rows = scaling.configurations()
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({r["prefix"] for r in rows}), 16)
        self.assertEqual(sum(r["mode"] == "Thinking" for r in rows), 8)

    def test_descriptive_coefficients_match_retained_reference(self):
        rows = scaling.configurations()
        with (ROOT / "results/reference/global_log_scaling_fits.csv").open(newline="") as handle:
            reference = list(csv.DictReader(handle))
        for expected in reference:
            actual = scaling.fit(rows, expected["metric"])
            for name, value in expected.items():
                if name != "metric":
                    self.assertAlmostEqual(float(actual[name]), float(value), places=7,
                                           msg=f"{expected['metric']}/{name}")


if __name__ == "__main__":
    unittest.main()
