# Release scope, provenance and licenses

## Current public scope — 23 October 2026

This repository contains the manuscript, numerical research artifacts, selected
historical chart exports, prompt templates and runnable CPU analysis. The
custom training implementation and historical training launchers are not
included. Model weights, raw stories, credentials and private correspondence
are not redistributed.

An earlier public release included training source. That release is no longer
the contents of this repository. The paper's original code-availability
paragraph reflects the earlier scope; the current README states the change
explicitly. The implementation is retained privately by SynthLabs. We make no
promise of a later source release or access on request. Changing the current
repository does not erase copies or rights granted by earlier distributions.

This public history was created from an explicit artifact allowlist, rather
than by removing files in a later commit from the old training-source history.
The released file identities are recorded in
[`public-artifacts.json`](public-artifacts.json). The distributed manuscript's
identity remains recorded separately in [`paper/RELEASE.json`](../paper/RELEASE.json).

## Analysis provenance

Numerical inputs retain their original hashes and arena distinctions. The
results guides explain the joins, retained comparison receipts, historical
preprocessing qualifications and image-only gaps. Analysis-method source under
[`historical-source/`](historical-source/) is distinct from the private training
engine; its provenance and role are described in the
[receipt guide](../results/receipts/README.md#inspectable-method-source).

## License boundaries

Public analysis code is provided under the Apache License 2.0. Existing
copyright and attribution notices are retained; see [LICENSE](../LICENSE) and
[NOTICE](../NOTICE).

The manuscript and research figures remain attributed scholarly work by Angel
Raychev, Nathan Lile and Rafael Rafailov. The code license does not independently
relicense scholarly text or third-party material. Dataset and model licenses
remain those of their respective distributions. No raw third-party story text
or model weights are redistributed here.

## Verification boundary

The public workflow verifies retained artifact integrity and freshly executes
CPU analyses. It does not execute or qualify the private GPU training system.
Paper results, historical execution and newly repeated analysis are different
evidence; a successful plot rebuild is not a fresh training replication.
