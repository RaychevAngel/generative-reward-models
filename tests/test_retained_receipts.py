# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Recount published retained evidence without original text or model calls."""

import csv
import importlib.util
import json
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("receipt_summary", ROOT / "analysis/summarize_receipts.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class RetainedReceiptTests(unittest.TestCase):
    def test_audit_counts_and_wilson_intervals(self):
        report = summary.summarize()
        pooled = report["separate_qwen_judge_audit"]
        self.assertEqual((pooled["answer_only_wins"], pooled["comparisons"]), (42, 80))
        expected = {"4b_policy_4b_judge": (17, 40, (0.285, 0.578)),
                    "4b_policy_14b_judge": (25, 40, (0.470, 0.758))}
        self.assertEqual([round(v, 3) for v in pooled["wilson_95_z1_96"]], [0.417, 0.631])
        for config, (wins, count, interval) in expected.items():
            row = report["by_config"][config]
            self.assertEqual((row["answer_only_wins"], row["comparisons"]), (wins, count))
            self.assertEqual(tuple(round(v, 3) for v in row["wilson_95_z1_96"]), interval)
        self.assertEqual(report["unique_prompt_clusters"], 76)
        self.assertEqual(report["prompt_clusters_shared_across_configurations"], 4)
        self.assertEqual(report["prompt_cluster_multiplicities"], {1: 72, 2: 4})

    def test_text_free_audit_schema_and_order_mapping(self):
        rows = summary.read_csv("independent_judge_receipt.csv")
        self.assertEqual(list(rows[0]), ["row", "config", "answer_player", "thinking_player",
                                         "mode_presented_a", "mode_presented_b", "verdict",
                                         "winning_mode", "prompt_cluster_id"])
        self.assertEqual([int(row["row"]) for row in rows], list(range(1, 81)))
        for row in rows:
            self.assertEqual({row["mode_presented_a"], row["mode_presented_b"]},
                             {"answer_only", "thinking"})
            self.assertIn(row["verdict"], {"A", "B"})
            expected = row["mode_presented_a" if row["verdict"] == "A" else "mode_presented_b"]
            self.assertEqual(row["winning_mode"], expected)
            self.assertRegex(row["prompt_cluster_id"], r"^p\d{3}$")
            self.assertRegex(row["answer_player"], r"^ans_4b_(4b|14b)_\d+$")
            self.assertRegex(row["thinking_player"], r"^think_4b_(4b|14b)_\d+$")
        inputs = json.loads((summary.RECEIPTS / "manifest.json").read_text())["inputs"]
        for source in inputs:
            self.assertEqual(set(source), {"name", "sha256"})
            self.assertNotIn("/", source["name"])
            self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")

    def test_retained_gap_table_and_positive_interval_counts(self):
        rows = summary.read_csv("elo_bootstrap_gaps.csv")
        self.assertEqual(len(rows), 16)
        for metric in ["peak", "top5"]:
            selected = [row for row in rows if row["metric"] == metric]
            self.assertEqual(len(selected), 8)
            self.assertEqual(sum(float(row["ci_low"]) > 0 for row in selected), 5)
            self.assertEqual(len({(row["policy"], row["judge"]) for row in selected}), 8)
        for row in rows:
            self.assertAlmostEqual(float(row["answer"]) - float(row["thinking"]),
                                   float(row["gap_answer_minus_thinking"]), places=9)
            self.assertLessEqual(float(row["ci_low"]), float(row["ci_high"]))
            self.assertTrue(0 <= float(row["answer_win_bootstrap_fraction"]) <= 1)

    def test_diagnostics_are_only_categorical_and_numeric_aggregates(self):
        rows = summary.read_csv("generation_diagnostics_aggregate.csv")
        self.assertEqual(len(rows), 4)
        self.assertEqual(list(rows[0]), ["config", "mode", "player", "rows", "mean_words",
                                         "median_words", "mean_nonempty_lines",
                                         "mean_distinct_4gram_ratio", "mean_duplicate_line_rate",
                                         "markdown_bold_rate", "top_first_line_share"])
        for row in rows:
            self.assertEqual(int(row["rows"]), 779)
            self.assertIn(row["mode"], {"Answer-Only", "Thinking"})
            for key in list(row)[3:]:
                self.assertTrue(math.isfinite(float(row[key])))


if __name__ == "__main__":
    unittest.main()
