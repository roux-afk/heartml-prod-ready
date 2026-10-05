import pandas as pd
import pytest

from heart_disease.data import split_features_target
from heart_disease.evaluate import evaluate
from heart_disease.models import MODEL_REGISTRY, build_pipeline


def test_build_pipeline_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("does_not_exist")


@pytest.mark.parametrize("model_name", sorted(MODEL_REGISTRY))
def test_pipeline_fits_data_with_missing_values(model_name: str, clean_df: pd.DataFrame) -> None:
    features, target = split_features_target(clean_df)

    pipeline = build_pipeline(model_name).fit(features, target)

    assert set(pipeline.predict(features)) <= {0, 1}


def test_evaluate_returns_metrics_in_unit_range(clean_df: pd.DataFrame) -> None:
    features, target = split_features_target(clean_df)
    pipeline = build_pipeline("logistic_regression").fit(features, target)

    metrics = evaluate(pipeline, features, target)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_evaluate_skips_roc_auc_without_predict_proba(clean_df: pd.DataFrame) -> None:
    features, target = split_features_target(clean_df)
    pipeline = build_pipeline("svm").fit(features, target)

    metrics = evaluate(pipeline, features, target)

    assert "roc_auc" not in metrics
