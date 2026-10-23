#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Reproduce descriptive log-size OLS fits, not KL-axis functional fits.

This isolates the size-fit portion of the paper's analysis from historical
exploratory KL fits. It uses the 16 matched configurations in one common Gold
arena; it performs no model sampling, judging or training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from summarize_results import summarize


def configurations():
    records = []
    for cell in summarize()["matched_configurations"]:
        for mode in ["ans", "think"]:
            records.append({"prefix": f"{mode}_{cell['policy']}_{cell['judge']}",
                            "mode": "Answer-Only" if mode == "ans" else "Thinking",
                            "policy_b": float(cell["policy"][:-1]),
                            "judge_b": float(cell["judge"][:-1]),
                            "peak_step": cell[mode]["peak_step"],
                            "top5_steps": ";".join(map(str, cell[mode]["top_five_steps"])),
                            "peak_elo": cell[mode]["peak_rating"],
                            "top5_elo": cell[mode]["top_five_mean_rating"]})
    return sorted(records, key=lambda row: row["prefix"])


def fit(rows, metric):
    x, y = [], []
    for row in rows:
        log_p, log_r = math.log(row["policy_b"]), math.log(row["judge_b"])
        thinking = float(row["mode"] == "Thinking")
        x.append([1.0, log_p, log_r, thinking, thinking * log_p, thinking * log_r])
        y.append(row[metric])
    x, y = np.asarray(x), np.asarray(y)
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coefficients
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = 1.0 - float(np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2))
    c = [float(v) for v in coefficients]
    log2 = math.log(2.0)
    return {"metric": metric, "intercept": c[0], "log_policy_answer": c[1],
            "log_judge_answer": c[2], "thinking_offset": c[3],
            "log_policy_thinking_delta": c[4], "log_judge_thinking_delta": c[5],
            "answer_policy_per_doubling": c[1] * log2,
            "answer_judge_per_doubling": c[2] * log2,
            "thinking_policy_per_doubling": (c[1] + c[4]) * log2,
            "thinking_judge_per_doubling": (c[2] + c[5]) * log2,
            "rmse": rmse, "r2": r2, "n_configs": len(rows)}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("build/reproduction/repro_artifacts/size_scaling"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = configurations()
    fits = [fit(rows, metric) for metric in ["peak_elo", "top5_elo"]]
    write_csv(args.output / "global_configuration_summary.csv", rows)
    write_csv(args.output / "global_log_scaling_fits.csv", fits)
    source = Path(__file__).resolve().parents[1] / "results/raw/all_params/gold.csv"
    (args.output / "manifest.json").write_text(json.dumps({
        "source": "results/raw/all_params/gold.csv",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model": "Y = b0 + bP ln(P) + bR ln(R) + bM M + bMP M ln(P) + bMR M ln(R)",
        "mode_indicator": "M=1 for Thinking, 0 for Answer-Only",
        "estimator": "numpy.linalg.lstsq, rcond=None; unweighted ordinary least squares",
        "numpy_version": np.__version__,
        "configurations": len(rows), "coefficients_per_fit": 6,
        "interpretation": "In-sample descriptive fit on a sparse selected-checkpoint grid; no independent-seed or out-of-sample claim",
        "kl_inputs_used": False,
    }, indent=2) + "\n")
    print(json.dumps({"fits": fits, "output": str(args.output), "kl_inputs_used": False}, indent=2))


if __name__ == "__main__":
    main()
