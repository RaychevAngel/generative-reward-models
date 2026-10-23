# Retained comparison receipts

These small tables expose the arithmetic behind the paper's matched-setting
uncertainty table and separate-judge audit. They are **existing results**, not
new judging or a rerun of the training experiments.

| File | Contents |
| --- | --- |
| [`elo_bootstrap_gaps.csv`](elo_bootstrap_gaps.csv) | Exact retained 16-row result: eight matched cells, each with peak and top-five Answer-Only-minus-Thinking gaps and percentile intervals. |
| [`independent_judge_receipt.csv`](independent_judge_receipt.csv) | Eighty recorded verdicts, configuration/checkpoint IDs, presented A/B modes, derived winning mode and anonymous prompt clusters. No prompt/story/judgment text. |
| [`generation_diagnostics_aggregate.csv`](generation_diagnostics_aggregate.csv) | Four selected-checkpoint aggregate rows with the original first-line text column omitted. |

Run the self-contained CPU summary with:

```sh
make receipts
```

## Separate-judge audit

The recorded evaluator was `Qwen/Qwen3-235B-A22B-Instruct-2507`, separate from
the study's learned Gold judge, but still in the Qwen family. All 80 verdicts
were A/B choices. The receipt preserves their original row order and the
response-mode mapping, so the following counts and row-wise Wilson intervals
can be recalculated without the copyrighted stories:

| Selected configuration | Answer-Only wins | 95% Wilson interval |
| --- | ---: | ---: |
| 4B policy, 4B judge | 17 / 40 | [0.285, 0.578] |
| 4B policy, 14B judge | 25 / 40 | [0.470, 0.758] |
| Pooled rows | 42 / 80 | [0.417, 0.631] |

Intervals use the retained method's `z=1.96`. These results do not establish
an Answer-Only advantage under the separate evaluator. Checkpoints were
selected retrospectively, not by a fresh held-out selection rule.

`prompt_cluster_id` values `p001` onward are assigned by first occurrence of
**exact prompt equality**, with no normalization. The 80 rows contain 76 unique
prompts: 72 appear once and four appear in both configurations. No prompt text
or per-prompt hash is released. The Wilson calculation is row-wise, not a
prompt-clustered interval, and does not represent uncertainty across independent
training runs. The released verdicts support recounting, not independently
reassessing the hidden stories or reparsing the original model responses.

## Retained match-bootstrap gaps

The CSV contains the original Bradley–Terry refit's point estimates and
200-replicate, match-level bootstrap results. The [retained method](../../docs/historical-source/bootstrap_elo_ci.py) specifies
default seed 13, an `ans_4b_4b_0` anchor, regularization `alpha=0.01`, and bounds
at 2.5 and 97.5. Peak/top-five selection is recomputed within each replicate.
Five of eight intervals are wholly positive for each metric.

The `answer` and `thinking` columns are ratings from that anchored refit; do
not mix their absolute levels with another fitted arena. The last column is
the observed fraction of bootstrap replicates with a positive gap, not a
posterior probability. These intervals condition on the stored sampled matches;
they do not measure prompt-clustered or independent-training uncertainty.

The public CPU summary checks the recorded gaps and interval signs. It does
**not** rerun the bootstrap: the underlying match table is not included here.

### Inspectable method source

The released script adapts only the original method's input/output transport
and validation: it requires an explicit local CSV and a new output directory,
has no dataset-service import or default remote input, and refuses existing
output paths. Its player parser, numerical constants, fitting functions and
complete numerical pipeline (including resampling and checkpoint selection)
are unchanged source sections. Their SHA-256 hashes, the original whole-file
hash and the adapted whole-file hash are recorded in
[`manifest.json`](manifest.json) and checked by a source-equivalence test.
The canonical [method provenance record](../../docs/historical-source/bootstrap_elo_ci.provenance.json)
also supplies this entry during optional receipt re-extraction. The extractor
validates both the caller's original method file and the released method against
their committed hashes before writing outputs; it does not preserve metadata
from an existing output manifest.
This method comes from the retained manuscript snapshot, separately from the
2025 training-source pin.

Inspect the CLI without installing numerical dependencies:

```sh
python3 docs/historical-source/bootstrap_elo_ci.py --help
```

Optional execution requires separately installed NumPy and SciPy, an authorized
local `--matches` CSV with exactly `winner,loser` columns, and an explicit
`--out-dir` path that does not yet exist. Each comparison occupies one row;
preserve player IDs, row order and duplicate-row multiplicity. Summary IDs use
`ans_<policy>_<judge>_<step>` and `think_<policy>_<judge>_<step>`; other players
remain in the fitted arena. The historical method averages the five highest
snapshots, or all available snapshots when a supplied trajectory has fewer
than five. It warns on optimizer non-convergence and still uses the returned
coordinates; per-replicate convergence logs are not part of the receipt.

No fit was run to qualify this source release, and the original match inputs
are not distributed. The script is not called by `make receipts` or figure
reproduction. CI checks source equivalence, help and input safeguards without
calling the optimizer. Source equivalence makes the implemented method
inspectable; it does not establish a fresh reproduction of the sixteen recorded
rows.

## Aggregate generation diagnostics

Each row summarizes 779 stored responses at one selected checkpoint. Those
aggregates are a different sample from the 80-row judge audit. Word/line counts,
distinct-four-gram ratios, duplicate-line rates, Markdown-bold rates and
first-line concentration are surface diagnostics, not direct writing-quality
scores. Only the concentration fraction remains; the common first line itself
and all story/opening text are omitted. Text-level recomputation requires
separately authorized original outputs.

## Provenance and replay boundary

[`manifest.json`](manifest.json) records the source files' whole-file SHA-256
hashes, hashes of the retained method scripts, the public schemas and output
hashes. It exposes no private source paths or repository IDs.
[`extract_evidence_receipts.py`](../../analysis/extract_evidence_receipts.py)
documents and implements the allow-listed transformation for an operator who
already has authorized original inputs. It performs no generation, judging,
downloads or bootstrap fits and is not called by ordinary CPU reproduction.
The original inputs remain undistributed; a source hash identifies an artifact,
but does not substitute for access to it or establish that its judgments were
correct.
