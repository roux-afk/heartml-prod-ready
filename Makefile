.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test check train retrain rollback eda clean

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
	poetry run python -m heart_disease.train

retrain:  ## Validate new data, train a candidate, and deploy only if quality gates pass
	poetry run python -m heart_disease.retrain

rollback:  ## Restore the previous verified registry version
	poetry run python -m heart_disease.retrain --rollback

eda:  ## Generate EDA figures
	poetry run python -m heart_disease.plots

clean:  ## Remove caches and figures; preserve the model registry and split history
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov reports/figures reports/metrics.json
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
