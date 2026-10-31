"""Batch prediction using the version activated by the retraining process."""

import argparse
from pathlib import Path

import pandas as pd

from heart_disease.config import FEATURE_COLUMNS, PROJECT_ROOT
from heart_disease.registry import ActiveModel
from heart_disease.retrain_settings import load_settings


def main(argv: list[str] | None = None) -> None:
    """Read a headered feature CSV and append class predictions and model version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "retraining.toml")
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.input)
    model = ActiveModel(load_settings(args.config).store_dir)
    frame["prediction"] = model.predict(frame[FEATURE_COLUMNS])
    frame["model_version"] = model.version
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
