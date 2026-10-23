#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Rebuild numeric-backed paper figures, without network, models or credentials."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def prepare_output(root: Path, output: Path):
    marker = output / ".genrm-reproduction.json"
    within_build = root / "build" in output.parents
    if output == root or root in output.parents and not within_build:
        raise ValueError("Use an external output directory or a directory below this repository's build/.")
    if output.exists() and any(output.iterdir()) and not within_build and not marker.exists():
        raise ValueError("Refusing an existing nonempty external directory without a reproduction marker.")
    output.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"format": 1, "owner": "generative-reward-models-artifacts"}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/reproduction"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    try:
        prepare_output(root, output)
    except ValueError as error:
        parser.error(str(error))
    manifest = json.loads((root / "results/manifest.json").read_text())
    for record in manifest["files"]:
        path = root / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Input hash mismatch: {record['path']}")
        dest = output / "paper/figures/results" / path.relative_to(root / "results/raw")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    env = os.environ.copy()
    # ReportLab honors this for stable PDF metadata; no measured values change.
    env["SOURCE_DATE_EPOCH"] = "1792540800"
    for script in ["build_empirical_supplement.py", "build_main_figures.py"]:
        subprocess.run([sys.executable, str(root / "analysis" / script),
                        "--root", str(output)], check=True, env=env)
    print(json.dumps({"source_tables": len(manifest["files"]),
                      "pdfs": len(list((output / "arxiv/figures").rglob("*.pdf"))),
                      "output": str(output),
                      "models_or_network_required": False}, indent=2))


if __name__ == "__main__":
    main()
