#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Derive text-free receipts from authorized retained results; no inference.

The private inputs are not distributed. This optional preparation command is
separate from CPU reproduction of the public receipts and never uses a Hub.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
METHOD_PROVENANCE = ROOT / "docs/historical-source/bootstrap_elo_ci.provenance.json"
CONFIGS = {"4b_policy_4b_judge", "4b_policy_14b_judge"}
MODES = {"answer_only", "thinking"}
PLAYER = re.compile(r"(?:ans|think)_\d+(?:\.\d+)?b_\d+(?:\.\d+)?b_\d+\Z")
AUDIT_COLUMNS = ["row", "config", "answer_player", "thinking_player",
                 "mode_presented_a", "mode_presented_b", "verdict",
                 "winning_mode", "prompt_cluster_id"]
DIAGNOSTIC_COLUMNS = ["config", "mode", "player", "rows", "mean_words",
                      "median_words", "mean_nonempty_lines",
                      "mean_distinct_4gram_ratio", "mean_duplicate_line_rate",
                      "markdown_bold_rate", "top_first_line_share"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_method_provenance(method_dir):
    """Regenerate only the committed, allow-listed method record; verify both files."""
    record = json.loads(METHOD_PROVENANCE.read_text())
    keys = {"path", "source_name", "source_snapshot_commit", "source_sha256",
            "release_sha256", "status", "adaptation", "unchanged_source_sections_sha256"}
    sections = {"parse_player", "fit_bt", "prefix_metrics", "percentile",
                "numerical_pipeline", "constants"}
    if (set(record) != keys or record["path"] != "docs/historical-source/bootstrap_elo_ci.py"
            or record["source_name"] != "bootstrap_elo_ci.py"
            or not isinstance(record["unchanged_source_sections_sha256"], dict)
            or set(record["unchanged_source_sections_sha256"]) != sections):
        raise ValueError("Unexpected bootstrap-method provenance schema")
    hashes = [record["source_sha256"], record["release_sha256"],
              *record["unchanged_source_sections_sha256"].values()]
    if (not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes)
            or not isinstance(record["source_snapshot_commit"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", record["source_snapshot_commit"])
            or not all(isinstance(record[key], str) and record[key] for key in ["status", "adaptation"])):
        raise ValueError("Invalid bootstrap-method provenance fields")
    if digest(method_dir / record["source_name"]) != record["source_sha256"]:
        raise ValueError("Original bootstrap method does not match the committed source hash")
    if digest(ROOT / record["path"]) != record["release_sha256"]:
        raise ValueError("Released bootstrap method does not match its provenance hash")
    return record


def csv_bytes(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def audit_receipt(rows):
    clusters, receipt = {}, []
    for ordinal, row in enumerate(rows, 1):
        if row["config"] not in CONFIGS:
            raise ValueError(f"Unexpected configuration in audit row {ordinal}")
        a, b = row["response_a_source"], row["response_b_source"]
        if {a, b} != MODES or row["verdict"] not in {"A", "B"}:
            raise ValueError(f"Invalid mode/verdict in audit row {ordinal}")
        if not all(PLAYER.fullmatch(row[key])
                   for key in ["answer_player", "thinking_player"]):
            raise ValueError(f"Unexpected player identifier in audit row {ordinal}")
        prompt = row["prompt"]
        if not isinstance(prompt, str):
            raise ValueError(f"Non-string prompt in audit row {ordinal}")
        # Equality is exact, with no normalization, prompt hash or text export.
        if prompt not in clusters:
            clusters[prompt] = f"p{len(clusters) + 1:03d}"
        receipt.append(dict(zip(AUDIT_COLUMNS, [
            ordinal, row["config"], row["answer_player"], row["thinking_player"],
            a, b, row["verdict"], a if row["verdict"] == "A" else b,
            clusters[prompt],
        ])))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--method-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    method_source = load_method_provenance(args.method_dir)
    source = args.source_dir
    gaps = source / "elo_bootstrap_gaps.csv"
    audit = source / "independent_judge_results.jsonl"
    diagnostics = source / "generation_diagnostics_summary.csv"
    audit_rows = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
    receipt = audit_receipt(audit_rows)
    with diagnostics.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames) != set(DIAGNOSTIC_COLUMNS + ["top_first_line"]):
            raise ValueError("Unexpected diagnostic schema")
        diagnostic_rows = [{key: row[key] for key in DIAGNOSTIC_COLUMNS}
                           for row in reader]
    for row in diagnostic_rows:
        if row["config"] not in CONFIGS or row["mode"] not in {"Answer-Only", "Thinking"}:
            raise ValueError("Unexpected diagnostic configuration/mode")
        if not PLAYER.fullmatch(row["player"]):
            raise ValueError("Unexpected diagnostic player identifier")
        for key in DIAGNOSTIC_COLUMNS[3:]:
            float(row[key])
    if len(receipt) != 80 or len(diagnostic_rows) != 4:
        raise ValueError("Retained source record counts differ from the release")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "elo_bootstrap_gaps.csv": gaps.read_bytes(),
        "independent_judge_receipt.csv": csv_bytes(receipt, AUDIT_COLUMNS),
        "generation_diagnostics_aggregate.csv": csv_bytes(diagnostic_rows, DIAGNOSTIC_COLUMNS),
    }
    records = []
    for name, payload in payloads.items():
        path = output / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace a different receipt: {name}")
        path.write_bytes(payload)
        rows = list(csv.DictReader(io.StringIO(payload.decode())))
        records.append({"path": name, "sha256": digest(path),
                        "rows": len(rows), "columns": list(rows[0])})
    inputs = [gaps, audit, diagnostics,
              args.method_dir / "bootstrap_elo_ci.py",
              args.method_dir / "run_tinker_independent_judge.py"]
    manifest = {
        "receipt_date": "2026-10-21",
        "status": "Existing retained results; no new generation, judging or bootstrap fitting",
        "inputs": [{"name": path.name, "sha256": digest(path)} for path in inputs],
        "method_sources": [method_source],
        "files": records,
        "derivation": {
            "elo_bootstrap_gaps.csv": "Exact byte copy of the retained 16-row result",
            "independent_judge_receipt.csv": "Allow-listed metadata/verdict fields in original row order; winning mode derived from A/B mapping; p001... IDs assigned by first occurrence of exact prompt equality; no prompt text or per-prompt hashes",
            "generation_diagnostics_aggregate.csv": "Retained aggregate fields with the top_first_line text column omitted; no opener text",
        },
        "separate_audit_judge": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "privacy": "No prompts, stories, raw judgments, private source repositories, original row IDs or local source paths are exported",
        "replay_boundary": "Public receipts reproduce counts, Wilson intervals, cluster overlap and reported CI signs; they cannot replay judging, recompute text diagnostics, or refit the match bootstrap without separately authorized original inputs",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"receipt_files": len(records), "audit_rows": len(receipt),
                      "prompt_clusters": len({r['prompt_cluster_id'] for r in receipt}),
                      "raw_text_exported": False}, indent=2))


if __name__ == "__main__":
    main()
