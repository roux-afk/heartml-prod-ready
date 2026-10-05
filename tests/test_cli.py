import json
from collections.abc import Callable
from pathlib import Path

import joblib
import pandas as pd
import pytest

from heart_disease.data import split_features_target
from heart_disease.plots import plot_age_by_sex, plot_age_distribution
from heart_disease.train import main


def test_train_cli_saves_model_and_metrics(raw_csv: Path, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    main(
        [
            *("--data-path", str(raw_csv)),
            *("--models", "naive_bayes", "logistic_regression"),
            *("--metric", "recall"),
            *("--models-dir", str(models_dir)),
            *("--reports-dir", str(reports_dir)),
        ]
    )

    report = json.loads((reports_dir / "metrics.json").read_text(encoding="utf-8"))
    assert report["selection_metric"] == "recall"
    assert set(report["models"]) == {"naive_bayes", "logistic_regression"}
    assert report["best_model"] in report["models"]
    assert (models_dir / "model.joblib").exists()


def test_saved_model_predicts_on_raw_features(
    raw_csv: Path, tmp_path: Path, clean_df: pd.DataFrame
) -> None:
    main(
        [
            *("--data-path", str(raw_csv)),
            *("--models", "svm"),
            *("--models-dir", str(tmp_path)),
            *("--reports-dir", str(tmp_path)),
        ]
    )

    pipeline = joblib.load(tmp_path / "model.joblib")
    features, _ = split_features_target(clean_df)

    assert len(pipeline.predict(features)) == len(features)


def test_train_cli_rejects_unknown_model() -> None:
    with pytest.raises(SystemExit):
        main(["--models", "unknown"])


@pytest.mark.parametrize("plot_fn", [plot_age_distribution, plot_age_by_sex])
def test_plots_are_saved(
    plot_fn: Callable[[pd.DataFrame, Path], Path], clean_df: pd.DataFrame, tmp_path: Path
) -> None:
    path = plot_fn(clean_df, tmp_path)

    assert path.exists()
    assert path.stat().st_size > 0
