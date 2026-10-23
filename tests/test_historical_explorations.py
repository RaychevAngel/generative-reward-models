# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Verify the exact, allow-listed historical image exports without inference."""

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/historical-explorations"
APPROVED = {
    "ultrainteract-ultrafeedback-proxy-reference.png": (
        "F09ATDEU44E", "2025-08-17",
        "358bd987f2847fb037e21db28ae02921fa3f4871631e3f2922c13e6e894e337b"),
    "single-and-mixed-data-rewarders.png": (
        "F09DGPU6D7G", "2025-09-03",
        "140ef90c9fc82e0e7f94c0487e3fd047108b5a830b58b3ad59e07d87d66d47c2"),
    "distilled-thinking-rewarder-capacity.png": (
        "F09E3QFENAY", "2025-09-07",
        "fb6a13f7554196d1c23b39de95988940d6c2098f3e79b8f8df0f5dbc1049932a"),
}


class HistoricalExplorationTests(unittest.TestCase):
    def test_only_approved_original_images_and_provenance_are_present(self):
        self.assertEqual({p.name for p in ARCHIVE.iterdir()},
                         set(APPROVED) | {"README.md", "manifest.json"})
        manifest = json.loads((ARCHIVE / "manifest.json").read_text())
        self.assertEqual(set(manifest), {"schema_version", "status",
                                         "date_interpretation", "evidence_scope", "files"})
        self.assertEqual({record["path"] for record in manifest["files"]}, set(APPROVED))
        for record in manifest["files"]:
            self.assertEqual(set(record), {"path", "original_artifact_id", "artifact_date",
                                           "sha256", "byte_count", "x_axis", "y_axis"})
            artifact_id, date, expected_hash = APPROVED[record["path"]]
            self.assertEqual((record["original_artifact_id"], record["artifact_date"],
                              record["sha256"]), (artifact_id, date, expected_hash))
            payload = (ARCHIVE / record["path"]).read_bytes()
            self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_hash)
            self.assertEqual(len(payload), record["byte_count"])
            self.assertIn("win rate minus loss rate", record["y_axis"].lower())
        provenance = (ARCHIVE / "manifest.json").read_text()
        self.assertNotIn("https://", provenance)
        self.assertNotIn("/Users/", provenance)
        self.assertNotIn("source_message", provenance)


if __name__ == "__main__":
    unittest.main()
