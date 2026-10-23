# Retained numerical results

`raw/` contains the retained numerical CSVs used to reconstruct the paper's
rating and historical KL-axis figures. The [manifest](manifest.json) records every
file's hash, schema and row count. There is no prompt or story text in these
tables.

Two schemas are retained:

- **`player,rating`**: a contestant identifier and its fitted rating in that
  file's evaluator arena. The final underscore-delimited identifier component
  is the policy-training step; preceding components identify the trajectory.
- **`kl,rating,source,label,prefix`**: an exported KL-axis value, corresponding
  rating, evaluator role and trajectory labels. These exports do not contain
  explicit step IDs. The figure generator recovers a step only through a unique
  exact-prefix/rating join and records missing matches. It never invents one.

The historical plotting pipeline includes a Savitzky-Golay smoothing stage;
the retained beta-axis producer uses its smoothed export. These files should
not be described as uniformly raw per-checkpoint KL measurements. Negative
values are retained as stored rather than clipped or explained as ordinary
sampling noise. Current plotting adds no new smoothing. Exact rating joins
recover the retained export pairing, not an independent validation against
the original unsmoothed metric logs.

## Important distinctions

`all_params/gold.csv` supplies one common Gold arena for the eight matched
configurations. Other directories retain their own comparison arenas. Equal
ratings across separate arenas are not calibrated equality; the analysis
does not merge their absolute scales.

The `top_5` summary selects each trajectory's five highest observed Gold
ratings. It is neither improvement from step zero nor a held-out checkpoint
selection policy. Snapshots along a training trajectory are not independent
training replications.

Several original filenames are historical rather than reliable manifests. For
example, `genrm_sizes_answer/kl_grm_sizes_ans.csv` contains only the 0.6B-judge
Answer/Thinking pair, not a complete five-size Answer sweep. The released
scripts select exact prefixes from the contents and do not infer experiments
from a filename alone.

Regenerate figures and inspect all plotted coordinates using
[`docs/reproduction.md`](../docs/reproduction.md). Original comparison-row
evidence and model outputs are not part of these CSVs; this release does not
claim that plotting tables alone can refit every rating or uncertainty interval.

## Comparison and uncertainty receipts

[`receipts/`](receipts/README.md) adds the retained sixteen-row bootstrap-gap
table, eighty text-free separate-judge verdict records with anonymous prompt
clusters, and aggregate generation diagnostics. `make receipts` reproduces
their counts and reported Wilson intervals without models or private text.
The receipt guide distinguishes this arithmetic from replaying the original
judging or match-bootstrap fit.

## Earlier exploratory images

Three [original historical chart exports](historical-explorations/README.md)
preserve data-recipe and distillation explorations. They use net win rate,
not Elo, and are kept separate from the paper's Gold rating arena. The archive
includes byte-level provenance and explicitly separates image preservation
from numerical reconstruction or controlled replication.
