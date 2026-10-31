"""Classification metrics for fitted pipelines."""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline


def evaluate(
    pipeline: Pipeline, features: pd.DataFrame, target: pd.Series, *, digits: int | None = 4
) -> dict[str, float]:
    """Compute binary classification metrics; ROC-AUC only if the model has predict_proba."""
    predictions = pipeline.predict(features)

    metrics = {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
    }
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)[:, 1]
        metrics["roc_auc"] = roc_auc_score(target, probabilities)

    return {
        name: float(value) if digits is None else round(float(value), digits)
        for name, value in metrics.items()
    }
