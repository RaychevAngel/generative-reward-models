# Artifacts and access

## Included here

- The released paper and exact figure assets.
- Numerical player/rating and historical KL-axis tables, with no story or prompt text.
- Deterministic CPU figure generators and input/output manifests.
- Retained Elo/uncertainty analysis-method source, with provenance.
- A public-artifact hash manifest, licenses and attribution.

## Inputs a new training run would require

| Input | Purpose | Included? |
| --- | --- | --- |
| Training implementation and resolved recipes | Generative judging, online preference assembly and distributed updates | Not part of the current public package. |
| Qwen3 base models and tokenizers | Policy/reference initialization and judge initialization | No weights bundled; use the relevant Qwen model's distribution and license. |
| Prepared LitBench pairs and split membership | Vote-derived Gold supervision and validation | No third-party text bundled. |
| Gold-labeled proxy-judge examples | Proxy SFT/GRPO targets | No; obtain authorized inputs and record exact revisions and filtering. |
| WritingPrompts-derived policy prompt splits | Online policy generation | No third-party text bundled. |
| Trained Gold and proxy judges | Historical evaluation and preference provision | No; some original artifacts require authorized access and are not externally downloadable through a public link. |
| Per-run policy checkpoints and comparison rows | Fresh judging, rating refits or training continuation | Not bundled. The numerical tables support the released analysis, not regeneration of missing model outputs. |

The public datasets that informed the work include
[SAA-Lab/LitBench-Train](https://huggingface.co/datasets/SAA-Lab/LitBench-Train)
and [euclaise/WritingPrompts_preferences](https://huggingface.co/datasets/euclaise/WritingPrompts_preferences).
Their distribution terms and content rights are separate from this analysis
release. Dataset names alone do not establish exact historical preprocessing,
consumed rows or disjointness. The paper describes the recovered split policy
and the boundaries of that evidence.

## What the numerical tables can and cannot recover

The released CSVs retain model/step identifiers, fitted ratings and historical
KL-axis exports. They support exact plotting, per-configuration summaries and
inspection of retained chronology. They do **not** contain all original
comparison outcomes, prompt identities or training randomness. They therefore
cannot independently recover the original rating fit or a prompt-clustered
bootstrap from scratch. No new identifiers are manufactured to imply otherwise.
The KL-axis exports have an additional provenance boundary: historical
smoothing/export processing is not reversed by reproducing the stored values.
See the [table guide](../results/README.md).

Three retained original figures remain image-only in this release: the static
accuracy comparison, intermediate-judge-checkpoint panel and ancillary
OpenOrca panel. Their images are not silently digitized into purported source
measurements. Most other displayed results have released numerical inputs.

## Reference evaluator

“Gold” means the study's learned 32B reference evaluator, not a human oracle or
an independent annotation service. Proxy and Gold comparisons are separately
fitted rating arenas. A proxy/Gold diagonal is not response-level agreement,
and ratings from different arenas should not be pooled onto one absolute scale.
