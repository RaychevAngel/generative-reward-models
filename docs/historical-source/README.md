# Historical plotting/export provenance

`z_elo_calculation.py` is preserved unchanged from research revision
`b13b2057c6c87ba754b1271513504bc985e034d8`; its hash is recorded in the
[public artifact manifest](../public-artifacts.json).

It is **historical source evidence**, not the public CPU reproduction entry
point. It includes original model/dataset mappings and depends on historical
helpers and model libraries not needed by the current analysis. Use
`make figures` at the repository root for the self-contained reconstruction.
No data, generated story text, credentials or private service helper are
included with this source.

The relevant export path is directly inspectable:

- `_combine_kl_by_prefix()` combines metric exports and step offsets.
- `_plot_steps_vs_kl_series(..., smooth=True)` applies a Savitzky-Golay filter
  before writing `steps_vs_kl_est.csv`.
- `plot_kl_vs_rating()` reads that axis file, indexes `kl_values[step]` and
  exports `kl,rating,source,label,prefix` rows.
- The captured active beta call selects `steps_vs_kl_est.csv`.

Consequently, the retained beta values are historically smoothed plot
coordinates, not raw nonnegative per-update KL measurements. Other retained
KL-table preprocessing has not been fully reconstructed. Today's exact-prefix
rating joins validate the retained row pairing, not the original metric-log
step alignment. Values are preserved; negative coordinates are neither clipped
nor explained as ordinary sampling noise of the nonnegative k3 expression.

The new release's log-size OLS script uses only the common Gold rating table,
not these KL-axis coordinates. It deliberately does not rerun the exploratory
functional KL fits present in older research analysis.
