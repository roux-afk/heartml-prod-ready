.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test check train eda clean

help:  ## Show available commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Create .venv and install all dependencies + git hooks
	poetry install
	poetry run pre-commit install

lint:  ## Run ruff linter
	poetry run ruff check src tests

format:  ## Auto-format code and fix lint issues
	poetry run ruff check --fix src tests
	poetry run ruff format src tests

typecheck:  ## Run mypy
	poetry run mypy src tests

test:  ## Run tests with coverage
	poetry run pytest

check: lint typecheck test  ## Run all checks (same as CI)

train:  ## Train all models and save the best one
	poetry run heart-disease-train

eda:  ## Generate EDA figures
	poetry run heart-disease-eda

clean:  ## Remove caches and generated artifacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov models reports/figures reports/metrics.json
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
