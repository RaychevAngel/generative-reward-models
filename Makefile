.PHONY: install verify test figures summary size-fits receipts paper clean-help

install:
	uv sync --frozen

verify:
	uv run --frozen python analysis/verify_release.py

test:
	uv run --frozen python -m unittest discover -s tests -v
	uv run --frozen python analysis/test_main_figures.py

figures:
	uv run --frozen python analysis/reproduce.py

summary:
	uv run --frozen python analysis/summarize_results.py

size-fits:
	uv run --frozen python analysis/fit_size_scaling.py

receipts:
	uv run --frozen python analysis/summarize_receipts.py

paper:
	python3 analysis/build_paper.py

clean-help:
	@printf '%s\n' 'Generated output is under build/. This target deletes nothing.'
