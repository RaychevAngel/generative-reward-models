# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Check method-source preservation and transport safety without running fits."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/receipts/manifest.json"
METHOD = ROOT / "docs/historical-source/bootstrap_elo_ci.py"
PROVENANCE = ROOT / "docs/historical-source/bootstrap_elo_ci.provenance.json"
spec = importlib.util.spec_from_file_location("evidence_extractor", ROOT / "analysis/extract_evidence_receipts.py")
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)


def digest(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


class BootstrapMethodSourceTests(unittest.TestCase):
    def cli(self, *arguments):
        # Isolated, no-site mode cannot import NumPy/SciPy or user-site plugins.
        return subprocess.run([sys.executable, "-I", "-S", str(METHOD), *map(str, arguments)],
                              capture_output=True, text=True, timeout=10)

    def test_original_numerical_source_sections_are_preserved(self):
        manifest = json.loads(MANIFEST.read_text())
        record = next(item for item in manifest["method_sources"]
                      if item["path"] == METHOD.relative_to(ROOT).as_posix())
        self.assertEqual(record, json.loads(PROVENANCE.read_text()))
        source = METHOD.read_text()
        self.assertEqual(digest(source), record["release_sha256"])
        retained = next(item for item in manifest["inputs"]
                        if item["name"] == "bootstrap_elo_ci.py")
        self.assertEqual(record["source_sha256"], retained["sha256"])
        tree = ast.parse(source)
        sections = {}
        for name in ["parse_player", "fit_bt", "prefix_metrics", "percentile"]:
            function = next(node for node in tree.body
                            if isinstance(node, ast.FunctionDef) and node.name == name)
            sections[name] = ast.get_source_segment(source, function)
        sections["numerical_pipeline"] = source[source.index("    players = sorted"):
                                                source.index("    csv_path =")]
        sections["constants"] = source[source.index("PLAYER_RE ="):
                                       source.index("\n\ndef parse_args")]
        self.assertEqual({name: digest(text) for name, text in sections.items()},
                         record["unchanged_source_sections_sha256"])
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                       for alias in node.names)
        self.assertEqual(imports, {"__future__", "argparse", "csv", "math", "re",
                                   "collections", "pathlib", "numpy", "scipy.optimize", "scipy.special"})

    def test_reextraction_validates_and_regenerates_method_provenance(self):
        record = json.loads(PROVENANCE.read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bootstrap_elo_ci.py"
            source.write_text("# original-source fixture, never executed\n")
            record["source_sha256"] = extractor.digest(source)
            provenance = root / "provenance.json"
            provenance.write_text(json.dumps(record))
            with patch.object(extractor, "METHOD_PROVENANCE", provenance):
                first = extractor.load_method_provenance(root)
                self.assertEqual(first, record)
                self.assertEqual(extractor.load_method_provenance(root), first)
                source.write_text("# changed original-source fixture\n")
                with self.assertRaisesRegex(ValueError, "Original bootstrap method"):
                    extractor.load_method_provenance(root)
                source.write_text("# original-source fixture, never executed\n")
                changed = dict(record, release_sha256="0" * 64)
                provenance.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "Released bootstrap method"):
                    extractor.load_method_provenance(root)
                provenance.write_text(json.dumps(dict(record, unapproved_metadata="reject this")))
                with self.assertRaisesRegex(ValueError, "Unexpected bootstrap-method provenance schema"):
                    extractor.load_method_provenance(root)

    def test_help_and_required_paths_need_no_numerical_dependencies(self):
        result = self.cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--matches", result.stdout)
        self.assertIn("--out-dir", result.stdout)
        self.assertNotIn("--repo", result.stdout)
        result = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn("--matches", result.stderr)
        self.assertIn("--out-dir", result.stderr)

    def test_local_rows_preserve_order_and_multiplicity(self):
        method = runpy.run_path(str(METHOD))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "matches.csv"
            source.write_text("winner,loser\nans_4b_4b_0,think_4b_4b_0\n"
                              "think_4b_4b_0,ans_4b_4b_0\nans_4b_4b_0,think_4b_4b_0\n")
            self.assertEqual(method["read_matches"](source), {
                "winner": ["ans_4b_4b_0", "think_4b_4b_0", "ans_4b_4b_0"],
                "loser": ["think_4b_4b_0", "ans_4b_4b_0", "think_4b_4b_0"],
            })

    def test_transport_errors_cannot_overwrite_or_start_a_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "matches.csv"
            source.write_text("winner,loser\nans_4b_4b_0,think_4b_4b_0\n")
            existing = root / "existing"
            existing.mkdir()
            sentinel = existing / "elo_bootstrap_gaps.csv"
            sentinel.write_text("retain this output\n")
            result = self.cli("--matches", source, "--out-dir", existing)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("existing output is never overwritten", result.stderr)
            self.assertEqual(sentinel.read_text(), "retain this output\n")
            new_output = root / "new-output"
            result = self.cli("--matches", source, "--out-dir", new_output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("separately installed NumPy and SciPy", result.stderr)
            self.assertFalse(new_output.exists())
            source.write_text("winner,loser,story\na,b,not an allowed input column\n")
            result = self.cli("--matches", source, "--out-dir", new_output)
            self.assertIn("exactly winner,loser columns", result.stderr)
            self.assertFalse(new_output.exists())
            result = self.cli("--matches", root / "missing.csv", "--out-dir", new_output)
            self.assertIn("existing local CSV", result.stderr)
            self.assertFalse(new_output.exists())


if __name__ == "__main__":
    unittest.main()
