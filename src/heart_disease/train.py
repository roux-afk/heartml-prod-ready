"""CLI entry point: train models, compare metrics and persist the best pipeline."""

import argparse
import json
import logging
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from heart_disease.config import MODELS_DIR, RANDOM_STATE, RAW_DATA_PATH, REPORTS_DIR, TEST_SIZE
from heart_disease.data import clean_data, load_raw_data, split_features_target
from heart_disease.evaluate import evaluate
from heart_disease.models import MODEL_REGISTRY, build_pipeline

logger = logging.getLogger(__name__)

SELECTION_METRICS = ("accuracy", "precision", "recall", "f1")

Metrics = dict[str, dict[str, dict[str, float]]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train heart disease classifiers.")
    parser.add_argument("--data-path", type=Path, default=RAW_DATA_PATH, help="Path to raw CSV")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_REGISTRY),
        default=list(MODEL_REGISTRY),
        help="Models to train (default: all)",
    )
    parser.add_argument(
        "--metric",
        choices=SELECTION_METRICS,
        default="f1",
        help="Test metric used to select the best model",
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    return parser.parse_args(argv)


def train(data_path: Path, model_names: list[str]) -> tuple[dict[str, Pipeline], Metrics]:
    """Fit every requested model on the same split and collect train/test metrics."""
    features, target = split_features_target(clean_data(load_raw_data(data_path)))
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    pipelines: dict[str, Pipeline] = {}
    metrics: Metrics = {}
    for name in model_names:
        pipeline = build_pipeline(name).fit(x_train, y_train)
        pipelines[name] = pipeline
        metrics[name] = {
            "train": evaluate(pipeline, x_train, y_train),
            "test": evaluate(pipeline, x_test, y_test),
        }
        logger.info("%-20s test: %s", name, metrics[name]["test"])

    return pipelines, metrics


def save_artifacts(
    best_name: str,
    pipeline: Pipeline,
    metrics: Metrics,
    selection_metric: str,
    models_dir: Path,
    reports_dir: Path,
) -> None:
    """Persist the best pipeline and a JSON report with all metrics."""
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    report = {"best_model": best_name, "selection_metric": selection_metric, "models": metrics}
    report_path = reports_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("Saved model to %s and metrics to %s", model_path, report_path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    pipelines, metrics = train(args.data_path, args.models)
    best_name = max(metrics, key=lambda name: metrics[name]["test"][args.metric])
    logger.info("Best model by test %s: %s", args.metric, best_name)

    save_artifacts(
        best_name, pipelines[best_name], metrics, args.metric, args.models_dir, args.reports_dir
    )


if __name__ == "__main__":
    main()
    