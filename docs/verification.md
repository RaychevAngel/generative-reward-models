# Release verification

The current release qualifies **CPU analysis of retained research artifacts**.
It does not include or qualify GPU training code.

## Repeat the checks

From a fresh clone, with Python 3.10 or newer and `uv` available:

```sh
uv sync --frozen
make verify test figures summary size-fits receipts
```

These commands check the public-file allowlist and hashes, numerical table
integrity, receipt recounts, empirical figure reconstruction and descriptive
size fits. Output goes under `build/`; source tables and the distributed
manuscript are unchanged. No model weights, credentials, GPU or private data
are required. Package installation requires access to its package registry;
the installed analysis commands themselves make no model-provider calls.

The [CPU workflow](../.github/workflows/cpu-reproduction.yml) runs the same
checks on Linux with Python 3.10, 3.11 and 3.12. Current execution records are
visible in [GitHub Actions](https://github.com/RaychevAngel/generative-reward-models/actions).
Do not treat an old run from a differently scoped repository as verification
of this release.

## Manuscript identity

[`paper/RELEASE.json`](../paper/RELEASE.json) identifies the distributed PDF
and its sources. `make paper` can compile the supplied TeX using a separately
installed TeX distribution. PDF metadata and binary output may differ across
TeX installations; scientific text and figure content are the relevant
comparison, not a claim of universal binary identity.

## Boundaries

These checks do not rerun model training, regenerate historical stories,
repeat all original judging, refit missing raw comparison rows, or establish
independent-seed uncertainty. See [reproduction](reproduction.md),
[artifacts](artifacts.md) and the individual results guides for what each input
and computation supports.
