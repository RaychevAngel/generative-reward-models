# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Exercise the published artifact boundary, independently of private training."""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]

class PublicScopeTests(unittest.TestCase):
    def test_no_training_or_private_inputs(self):
        for name in ['training', 'recipes', '.env', '.source-cache']:
            self.assertFalse((ROOT/name).exists(), name)
        files = json.loads((ROOT/'docs/public-artifacts.json').read_text())['files']
        for record in files:
            path = ROOT/record['path']
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),record['sha256'])
            self.assertNotIn(path.suffix.lower(),['.safetensors','.pt','.pth','.pem','.key','.parquet'])

    def test_paper_and_release_scope_are_explicit(self):
        readme = (ROOT/'README.md').read_text()
        self.assertIn('https://arxiv.org/abs/2610.30765',readme)
        self.assertIn('not included',readme)
        self.assertIn('earlier release scope',readme)
        self.assertIn('no model inference',readme)

    def test_all_tracked_paths_are_in_public_manifest(self):
        result = subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True)
        if result.returncode:
            self.skipTest('Git checkout unavailable')
        tracked = set(filter(None,result.stdout.decode().split('\0')))
        if not tracked:
            self.skipTest('Initial publication has not been staged')
        manifested = {r['path'] for r in json.loads((ROOT/'docs/public-artifacts.json').read_text())['files']}
        self.assertEqual(tracked,manifested|{'docs/public-artifacts.json'})

if __name__=='__main__':
    unittest.main()
