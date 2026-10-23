# Reproduce the research artifacts

This package lets you **rebuild the retained numerical evidence on a CPU**.
It does not contain the private training implementation or the inputs required
to repeat model training. See the dated [release scope](release-scope.md).

## CPU figures and descriptive summaries

Requirements: Python 3.10 or newer and [uv](https://docs.astral.sh/uv/). The root
environment contains ReportLab, NumPy and their transitive dependency set. It does
not install Torch, vLLM, Ray or a training stack.

```sh
git clone https://github.com/RaychevAngel/generative-reward-models.git
cd generative-reward-models
uv sync --frozen
make verify test figures summary size-fits receipts
```

Equivalent individual commands:

```sh
uv run --frozen python analysis/verify_release.py
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python analysis/test_main_figures.py
uv run --frozen python analysis/reproduce.py
uv run --frozen python analysis/summarize_results.py
uv run --frozen python analysis/fit_size_scaling.py
```

Only environment installation needs package-registry access. Once installed,
these analysis commands use the repository's retained CSVs and do not contact
GitHub, Hugging Face or a model provider. They perform no generation, judging,
training, weight downloads or external writes.

### Outputs

The reproduction command creates an isolated `build/reproduction/` tree:

| Output | Contents |
| --- | --- |
| `arxiv/figures/restored/` | Seven empirical vector figures: chronological matched-setting, retained KL-axis, baseline and capacity views. |
| `arxiv/figures/replotted/` | Fifteen vector plots, including the seven numeric-backed main-figure replacements and eight individual matched-setting views. |
| `repro_artifacts/empirical_supplement/` | Exact input hashes, configuration summaries and retained KL-export join/missingness records. |
| `repro_artifacts/main_figure_replots/` | Exact source/arena joins, input/output hashes and every plotted coordinate. |

Lines connect retained observations in training chronology. The scripts do not
add new smoothing or fits, interpolate absent checkpoints, convert image pixels into
measurements or treat snapshots as independent training repetitions.
Some KL-axis inputs were already processed by the historical producer; see
the [provenance qualification](../results/README.md). Reproduction preserves
those values and does not certify them as unsmoothed nonnegative divergences.

`summarize_results.py` prints the eight matched cells' observed peak/top-five
statistics from the common Gold arena. It reproduces a descriptive summary,
not a new statistical test. `verify_release.py` checks released numerical
table hashes, unique player IDs, finite values, the public-file manifest and
Python syntax without importing model frameworks.

### Descriptive parameter-size fits

`make size-fits` reproduces the paper's unweighted ordinary least-squares fit
on the 16 matched mode/configuration cells in the common Gold arena. Its six
features are an intercept, log policy size, log judge size, Thinking indicator
and the indicator's interactions with both log sizes. Separate outputs use
the observed peak and selected top-five mean rating.

Coefficients, per-doubling transformations, in-sample RMSE/R-squared and the
16-cell input summary are written under
`build/reproduction/repro_artifacts/size_scaling/`. Tests compare the fit to
the retained coefficient table in `results/reference/`. These are descriptive
summaries of a sparse, checkpoint-selected grid—not independent-seed uncertainty,
out-of-sample predictions, causal gains or compute-optimal allocation rules.
This command uses no KL-axis input and does not rerun functional KL fits.

### Retained comparison receipts

`make receipts` recounts the paper's retained separate-judge audit, recomputes
its row-wise Wilson intervals, verifies anonymous prompt overlap, and summarizes
the signs of the stored bootstrap-gap intervals. See the
[receipt guide](../results/receipts/README.md) for the text-free schemas and
provenance. This requires no model calls and does not rerun the original judging
or bootstrap fits.

## Compile the paper

The `paper/` directory contains the selected release manuscript and referenced
assets. Install a TeX distribution providing `latexmk`, `pdflatex` and `bibtex`,
then run:

```sh
make paper
```

The generated `build/paper/main.pdf` should match the scientific contents of the
download on the [project page](https://angelraychev.com/scaling-laws-for-generative-reward-models/).
PDF metadata can differ across TeX installations. The release record identifies
the distributed PDF by hash; no claim of cross-TeX binary identity is made.
Compilation leaves `paper/main.pdf` and the released source files unchanged,
so their integrity checks remain valid after rebuilding.

## What this release does not reconstruct

The numerical tables are not full original comparison rows, generated stories
or independent training runs. They cannot recreate every original Elo fit,
uncertainty calculation or missing checkpoint. Image-only source gaps are
identified in the [artifact map](artifacts.md). The paper's conclusions remain
bounded by the actual evaluator and observed configurations.

No command above trains or evaluates a language model. Repeating the training
study would require the private implementation, a separately qualified GPU
environment, authorized model/data access and resolved run configurations.
This package makes no promise of access to those materials.
