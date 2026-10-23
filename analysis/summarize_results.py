#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Print descriptive matched-setting summaries from the retained common arena."""

import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [("0.6b", "4b"), ("1.7b", "4b"), ("4b", "4b"), ("8b", "4b"),
           ("4b", "0.6b"), ("4b", "1.7b"), ("4b", "8b"), ("4b", "14b")]


def summarize():
    with (ROOT / "results/raw/all_params/gold.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    for policy, judge in CONFIGS:
        item = {"policy": policy, "judge": judge}
        for mode in ["ans", "think"]:
            prefix = f"{mode}_{policy}_{judge}"
            selected = [(int(row["player"].rsplit("_", 1)[1]), float(row["rating"]))
                        for row in rows if row["player"].rsplit("_", 1)[0] == prefix]
            if not selected:
                raise ValueError(f"No retained rows for {prefix}")
            ordered = sorted(v for _, v in selected)
            ranked = sorted(selected, key=lambda point: (-point[1], point[0]))
            item[mode] = {"snapshots": len(selected),
                          "first_step": min(s for s, _ in selected),
                          "last_step": max(s for s, _ in selected),
                          "peak_step": ranked[0][0],
                          "top_five_steps": [s for s, _ in ranked[:5]],
                          "peak_rating": max(ordered),
                          "top_five_mean_rating": mean(ordered[-5:])}
        records.append(item)
    return {"arena": "results/raw/all_params/gold.csv",
            "meaning": "Descriptive observed peak and selected top-five summaries; not significance tests or independent reruns",
            "matched_configurations": records,
            "answer_higher_peak": sum(r["ans"]["peak_rating"] > r["think"]["peak_rating"] for r in records),
            "answer_higher_top_five": sum(r["ans"]["top_five_mean_rating"] > r["think"]["top_five_mean_rating"] for r in records)}


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
