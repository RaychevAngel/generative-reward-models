# Reconstruction receipts

These are the numeric outputs and portable manifests from the release's CPU
reconstruction. They make the joins, plotted values and size-fit coefficients
inspectable without first installing the tools. They are not new model outputs
or independent experiment replications.

- `empirical_supplement/`: eight-setting summaries, retained KL-coordinate joins
  and missingness/provenance manifest.
- `main_figure_replots/`: every plotted coordinate and the source/arena/line
  semantics used for the main-figure reconstruction.
- `size_scaling/`: the 16-cell input summary, six-coefficient log-size OLS fits
  for each metric, and fit/source metadata. No KL-axis inputs enter these fits.

The [receipt manifest](MANIFEST.json) hashes the selected files. Paths inside
the individual figure manifests are relative to the isolated reproduction tree
created by `make figures`; its `paper/figures/results/` inputs are copied from
this repository's `results/raw/`. Rebuild with `make figures size-fits`.

Floating-point OLS results may differ at rounding precision across NumPy/BLAS
implementations; tests compare them numerically. The original input tables and
the source-selection/coordinate joins are preserved exactly. Historical KL
preprocessing qualifications remain in the [table guide](../README.md).
