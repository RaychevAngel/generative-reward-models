# CPU analysis

Run `make verify test figures summary size-fits receipts` from the repository root.
The [reproduction guide](../docs/reproduction.md) explains the environment,
outputs and evidence boundaries.

| Script | Purpose |
| --- | --- |
| `verify_release.py` | Check public artifact and paper hashes, numerical table integrity and syntax without importing model frameworks. |
| `reproduce.py` | Copy hash-verified tables into an isolated build tree and invoke both vector figure builders. |
| `build_empirical_supplement.py` | Rebuild chronological/common-arena, retained KL-coordinate, baseline and capacity views. |
| `build_main_figures.py` | Rebuild the numeric-backed main figures and individual matched-setting views using their correct evaluator arenas. |
| `summarize_results.py` | Print the eight matched cells' observed peak and top-five summaries. |
| `fit_size_scaling.py` | Reproduce the 16-cell descriptive log-size OLS fits, independently of KL-coordinate inputs. |
| `summarize_receipts.py` | Recount retained comparisons and intervals without rejudging responses or refitting missing matches. |
| `freeze_public_manifest.py` | Maintainer-only refresh of reviewed staged files; never adds new paths to Git. |
| `build_paper.py` | Compile the manuscript into an isolated build directory without modifying the distributed PDF. |
| `test_main_figures.py` | Check exact prefix selection, unique joins, missing/duplicate/nonfinite rejection and complete plot bounds. |

The builders do not make model calls or infer absent measurements. Their
manifests expose exact input hashes and every retained join; the main-figure
builder also exports all plotted coordinates. Historical KL preprocessing is
qualified explicitly rather than silently treated as raw measurements.

Original plotting/export source is preserved separately under
[`docs/historical-source/`](../docs/historical-source/). It documents provenance
and is not a dependency of this self-contained CPU path. The released analysis
does not rerun earlier exploratory functional fits on an unqualified KL axis.
