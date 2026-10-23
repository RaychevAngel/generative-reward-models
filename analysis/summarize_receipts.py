#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Summarize retained empirical receipts, without model calls or refitting."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "results/receipts"


def wilson_interval(wins, count, z=1.96):
    proportion = wins / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    radius = z * math.sqrt((proportion * (1 - proportion)
                            + z * z / (4 * count)) / count) / denominator
    return [center - radius, center + radius]


def read_csv(name):
    with (RECEIPTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize():
    manifest = json.loads((RECEIPTS / "manifest.json").read_text())
    for record in manifest["files"]:
        path = RECEIPTS / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Receipt hash mismatch: " + record["path"])
        rows = read_csv(record["path"])
        if len(rows) != record["rows"] or list(rows[0]) != record["columns"]:
            raise ValueError("Receipt schema/count mismatch: " + record["path"])
    audit = read_csv("independent_judge_receipt.csv")
    by_config = defaultdict(list)
    clusters = defaultdict(set)
    for row in audit:
        by_config[row["config"]].append(row)
        clusters[row["prompt_cluster_id"]].add(row["config"])

    def counts(rows):
        wins = sum(row["winning_mode"] == "answer_only" for row in rows)
        return {"answer_only_wins": wins, "comparisons": len(rows),
                "win_fraction": wins / len(rows),
                "wilson_95_z1_96": wilson_interval(wins, len(rows))}

    gaps = read_csv("elo_bootstrap_gaps.csv")
    by_metric = {}
    for metric in sorted({row["metric"] for row in gaps}):
        rows = [row for row in gaps if row["metric"] == metric]
        by_metric[metric] = {"matched_cells": len(rows),
                             "ci_fully_above_zero": sum(float(row["ci_low"]) > 0 for row in rows)}
    return {
        "separate_qwen_judge_audit": counts(audit),
        "by_config": {config: counts(rows) for config, rows in sorted(by_config.items())},
        "unique_prompt_clusters": len(clusters),
        "prompt_clusters_shared_across_configurations": sum(len(configs) > 1 for configs in clusters.values()),
        "prompt_cluster_multiplicities": dict(sorted(Counter(Counter(
            row["prompt_cluster_id"] for row in audit).values()).items())),
        "retained_bootstrap_gap_rows": len(gaps),
        "retained_bootstrap_by_metric": by_metric,
        "interpretation": "Retained verdict/interval receipts only; row-wise Wilson intervals are not prompt-clustered or independent-training uncertainty; no new judging or bootstrap fits",
    }


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
