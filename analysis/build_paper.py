#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Compile released paper sources in build/, preserving the distributed PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from reproduce import prepare_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/paper"))
    args = parser.parse_args()
    if shutil.which("latexmk") is None:
        parser.error("Install a TeX distribution with latexmk, pdflatex and bibtex first.")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    try:
        prepare_output(root, output)
    except ValueError as error:
        parser.error(str(error))
    manifest = json.loads((root / "paper/RELEASE.json").read_text())
    for record in manifest["files"]:
        relative = Path(record["path"])
        if relative.as_posix() in {"main.pdf", "README.md"}:
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "paper" / relative, destination)
    subprocess.run(["latexmk", "-pdf", "-silent", "-interaction=nonstopmode",
                    "-halt-on-error", "main.tex"], cwd=output, check=True)
    print(json.dumps({"compiled_pdf": str(output / "main.pdf"),
                      "distributed_pdf_preserved": "paper/main.pdf"}, indent=2))


if __name__ == "__main__":
    main()
