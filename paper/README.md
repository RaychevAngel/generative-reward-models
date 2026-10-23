# Paper

**Scaling Laws for Generative Reward Models**

Angel Raychev · Nathan Lile · Rafael Rafailov

[arXiv:2610.30765](https://arxiv.org/abs/2610.30765) · [Read the release PDF](main.pdf) ·
[Project page and citation](https://angelraychev.com/scaling-laws-for-generative-reward-models/)

This directory contains the release manuscript and the TeX, bibliography and
referenced figures required to build it. [`RELEASE.json`](RELEASE.json) records
the distributed PDF and source-file hashes.

With `latexmk`, `pdflatex` and `bibtex` installed, run `make paper` from the
repository root. CPU figure regeneration and its numerical provenance are
documented separately in [`docs/reproduction.md`](../docs/reproduction.md).

The manuscript is a research preprint, not a peer-reviewed publication.
The scientific manuscript is unchanged from the selected release. Its original
code-availability paragraph describes an earlier distribution; the current
public package is limited to research artifacts and CPU analysis, as explained
in the dated [release-scope note](../docs/release-scope.md).
