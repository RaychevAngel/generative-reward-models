#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Verify released input/source hashes and Python syntax without model imports."""

import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    tables = json.loads((ROOT / "results/manifest.json").read_text())["files"]
    for record in tables:
        path = ROOT / record["path"]
        require(digest(path) == record["sha256"], f"Hash mismatch: {record['path']}")
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == record["rows"], record["path"])
        require(all(math.isfinite(float(r["rating"])) for r in rows), record["path"])
        if "player" in record["columns"]:
            require(len({r["player"] for r in rows}) == len(rows), record["path"])
        else:
            require(all(math.isfinite(float(r["kl"])) for r in rows), record["path"])
    provenance = json.loads((ROOT / "docs/public-artifacts.json").read_text())
    require(provenance["scope"] == "public research artifacts and CPU analysis", "Unexpected release scope")
    source = provenance["files"]
    for record in source:
        path = ROOT / record["path"]
        require(not path.is_symlink(), "Symlink in artifact manifest")
        require(path.resolve().is_relative_to(ROOT.resolve()), "Artifact path escapes repository")
        require(not record["path"].startswith(("training/", "recipes/")), "Private training path in public manifest")
        require(digest(path) == record["sha256"], f"Artifact mismatch: {record['path']}")
    require(not (ROOT / "training").exists() and not (ROOT / "recipes").exists(), "Private training source must not be present")
    receipts_manifest = ROOT / "results/derived/MANIFEST.json"
    if receipts_manifest.exists():
        receipts = json.loads(receipts_manifest.read_text())["files"]
        for record in receipts:
            require(digest(ROOT / record["path"]) == record["sha256"], record["path"])
    comparison_manifest = ROOT / "results/receipts/manifest.json"
    if comparison_manifest.exists():
        comparison = json.loads(comparison_manifest.read_text())
        for record in comparison["files"]:
            require(digest(comparison_manifest.parent / record["path"]) == record["sha256"],
                    "Comparison receipt mismatch: " + record["path"])
    historical_manifest = ROOT / "results/historical-explorations/manifest.json"
    if historical_manifest.exists():
        historical = json.loads(historical_manifest.read_text())
        for record in historical["files"]:
            require(digest(historical_manifest.parent / record["path"]) == record["sha256"],
                    "Historical image mismatch: " + record["path"])
    paper_manifest_path = ROOT / "paper/RELEASE.json"
    require(paper_manifest_path.exists(), "Missing paper/RELEASE.json")
    if paper_manifest_path.exists():
        paper = json.loads(paper_manifest_path.read_text())
        for record in paper["files"]:
            require(digest(ROOT / "paper" / record["path"]) == record["sha256"], record["path"])
        require(digest(ROOT / "paper/main.pdf") == paper["pdf_sha256"], "Paper PDF mismatch")
    python_files = [p for base in ["analysis", "tests"]
                    for p in (ROOT / base).rglob("*.py")]
    for path in python_files:
        # Some preserved upstream utility regexes predate Python 3.12's escape
        # warnings; retain their original bytes rather than restyling research
        # source. A genuine syntax error still fails this check.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(path.read_text(), filename=path.relative_to(ROOT).as_posix())
    print(json.dumps({"verified_tables": len(tables), "verified_source_files": len(source),
                      "python_files_parsed": len(python_files),
                      "paper_manifest_checked": paper_manifest_path.exists(),
                      "scope": provenance["scope"],
                      "models_loaded": 0, "network_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
