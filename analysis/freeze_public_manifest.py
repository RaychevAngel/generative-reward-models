#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Maintainer-only manifest refresh from an explicitly staged public allowlist.

This never adds files to Git. Review and stage the public selection first.
It refuses tracked paths outside the package's narrow release boundary.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'docs/public-artifacts.json'
ROOT_FILES = {'.gitattributes', '.gitignore', 'README.md', 'CITATION.cff',
              'LICENSE', 'NOTICE', 'pyproject.toml', 'uv.lock', 'Makefile'}
DOC_FILES = {'docs/reproduction.md', 'docs/artifacts.md', 'docs/release-scope.md',
             'docs/verification.md', 'docs/online-dpo-loop.svg', MANIFEST}
PREFIXES = ('analysis/', 'paper/', 'results/', 'docs/historical-source/',
            'docs/images/', 'tests/')
FORBIDDEN_SUFFIXES = {'.safetensors', '.pt', '.pth', '.key', '.pem', '.parquet'}


def main():
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    records = []
    for name in sorted(filter(None, paths)):
        path = ROOT / name
        parts = PurePosixPath(name).parts
        if ('..' in parts or PurePosixPath(name).is_absolute() or
                not (name in ROOT_FILES or name in DOC_FILES or name.startswith(PREFIXES) or
                     name == '.github/workflows/cpu-reproduction.yml') or
                path.suffix.lower() in FORBIDDEN_SUFFIXES or path.is_symlink() or
                not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve())):
            raise SystemExit('Outside the reviewed public boundary: ' + name)
        if name != MANIFEST:
            records.append({'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'bytes': path.stat().st_size})
    if not records:
        raise SystemExit('Stage the reviewed public files before creating their manifest.')
    payload = {'release': 'v2026.10.23-analysis',
               'scope': 'public research artifacts and CPU analysis',
               'excludes': ['private training source and recipes', 'weights', 'raw stories',
                            'credentials', 'private correspondence', 'prior training-source Git history'],
               'manifest_self_hash_excluded': True, 'files': records}
    (ROOT / MANIFEST).write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'manifested_public_files': len(records), 'path': MANIFEST}))


if __name__ == '__main__':
    main()
