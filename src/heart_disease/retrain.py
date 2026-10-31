"""Automatic local retraining: ingest, validate, evaluate, register and deploy."""

import argparse
import fcntl
import hashlib
import json
import logging
import math
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

from heart_disease.config import (
    BINARY_CLASSES,
    FEATURE_COLUMNS,
    LOG_FORMAT,
    PROJECT_ROOT,
    TARGET_COLUMN,
)
from heart_disease.evaluate import evaluate
from heart_disease.models import build_pipeline
from heart_disease.registry import ModelRegistry, atomic_json, read_json
from heart_disease.retrain_data import InsufficientDataError, load_dataset, split_dataset
from heart_disease.retrain_settings import RetrainSettings, load_settings

logger = logging.getLogger(__name__)
HISTORY_KEYS = {"seen_records", "known_patients", "heldout_patients"}


@contextmanager
def run_lock(root: Path) -> Iterator[None]:
    """Serialize writers; OS releases the advisory lock even after process failure."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".retrain.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_history(root: Path) -> dict[str, list[str]]:
    """Fail closed when split lineage is missing for an already active model."""
    path = root / "history.json"
    if not path.exists():
        if (root / "active.json").exists():
            raise ValueError("Active model has no data lineage")
        return {key: [] for key in HISTORY_KEYS}
    history = read_json(path)
    if set(history) != HISTORY_KEYS or any(
        not isinstance(values, list) or not all(isinstance(item, str) for item in values)
        for values in history.values()
    ):
        raise ValueError("Invalid retraining history")
    return history


def quality_reasons(
    candidate: dict[str, float], incumbent: dict[str, float] | None, settings: RetrainSettings
) -> list[str]:
    """Apply absolute thresholds and a same-holdout champion/challenger comparison."""
    reasons = []
    for name in ("recall", "precision", "f1"):
        value = candidate.get(name, math.nan)
        if not math.isfinite(value) or value < getattr(settings, f"min_{name}"):
            reasons.append(f"Candidate {name} does not meet its threshold")
    if not all(math.isfinite(value) for value in candidate.values()):
        reasons.append("Candidate metrics contain nonfinite values")
    if incumbent is not None:
        if not all(math.isfinite(value) for value in incumbent.values()):
            reasons.append("Incumbent metrics contain nonfinite values")
        for name in ("recall", "f1"):
            if candidate.get(name, -1) < incumbent.get(name, math.inf):
                reasons.append(f"Candidate {name} is worse than incumbent")
        improvement = max(
            candidate.get(name, -1) - incumbent.get(name, 1) for name in ("recall", "f1")
        )
        if improvement < settings.min_improvement:
            reasons.append("No sufficient improvement over incumbent")
    return reasons


def select_candidate(
    development: pd.DataFrame, settings: RetrainSettings
) -> tuple[str, Pipeline, dict[str, list[float]]]:
    """Select on patient-grouped CV, then refit only on development data."""
    features, target = development[FEATURE_COLUMNS], development[TARGET_COLUMN]
    splitter = StratifiedGroupKFold(
        n_splits=settings.cv_folds, shuffle=True, random_state=settings.seed
    )
    splits = list(splitter.split(features, target, development["patient_id"]))
    for train_indices, validation_indices in splits:
        if target.iloc[train_indices].nunique() != len(BINARY_CLASSES) or target.iloc[
            validation_indices
        ].nunique() != len(BINARY_CLASSES):
            raise InsufficientDataError("Every CV fold must contain both classes")
    scores = {}
    for name in settings.models:
        logger.info("Cross-validating %s", name)
        values = cross_val_score(
            build_pipeline(name), features, target, cv=splits, scoring="f1", error_score="raise"
        )
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite cross-validation scores")
        scores[name] = [float(value) for value in values]
    best = max(scores, key=lambda name: float(np.mean(scores[name])))
    return best, build_pipeline(best).fit(features, target), scores


def provenance() -> dict[str, Any]:
    """Record the code content as well as Git revision and exact runtime versions."""
    revision = None
    if shutil.which("git"):
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            revision = result.stdout.strip()
    sources = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    return {
        "git_sha": revision,
        "source_hashes": sources,
        "python": sys.version,
        "dependencies": {
            name: package_version(name)
            for name in ("pandas", "numpy", "scikit-learn", "joblib", "lightgbm", "xgboost")
        },
    }


def _reserve_evaluation(
    root: Path, frame: pd.DataFrame, evaluation: pd.DataFrame, history: dict[str, list[str]]
) -> None:
    atomic_json(
        root / "history.json",
        {
            "seen_records": sorted(set(history["seen_records"]) | set(frame["record_id"])),
            "known_patients": sorted(set(history["known_patients"]) | set(frame["patient_id"])),
            "heldout_patients": sorted(
                set(history["heldout_patients"]) | set(evaluation["patient_id"])
            ),
        },
    )


def _reported_metrics(metrics: dict[str, float] | None) -> dict[str, float | None] | None:
    if metrics is None:
        return None
    return {name: value if math.isfinite(value) else None for name, value in metrics.items()}


def _execute(settings: RetrainSettings, directory: Path, report: dict[str, Any]) -> None:
    report["stage"] = "data_validation"
    frame, manifest = load_dataset(settings)
    history = read_history(settings.store_dir)
    development, evaluation = split_dataset(frame, history, settings)
    report.update(
        {
            "sources": manifest,
            "development_rows": len(development),
            "evaluation_rows": len(evaluation),
            "new_rows": int((~frame["record_id"].isin(history["seen_records"])).sum()),
        }
    )
    development.to_csv(directory / "development.csv", index=False)
    evaluation.to_csv(directory / "evaluation.csv", index=False)
    report["stage"] = "training"
    best, pipeline, scores = select_candidate(development, settings)
    report.update({"best_model": best, "cv_f1": scores})
    joblib.dump(pipeline, directory / "candidate.joblib")
    # Reserve before observing holdout metrics, including for rejected/failed candidates.
    _reserve_evaluation(settings.store_dir, frame, evaluation, history)
    report["stage"] = "evaluation"
    features, target = evaluation[FEATURE_COLUMNS], evaluation[TARGET_COLUMN]
    candidate = evaluate(pipeline, features, target, digits=None)
    registry = ModelRegistry(settings.store_dir)
    active = registry.active()
    incumbent = (
        evaluate(registry.load(active["version"]), features, target, digits=None)
        if active
        else None
    )
    reasons = quality_reasons(candidate, incumbent, settings)
    report.update(
        {
            "candidate_metrics": _reported_metrics(candidate),
            "incumbent_metrics": _reported_metrics(incumbent),
            "incumbent_version": active["version"] if active else None,
            "confusion_matrix": confusion_matrix(
                target, pipeline.predict(features), labels=[0, 1]
            ).tolist(),
            "rejection_reasons": reasons,
        }
    )
    if reasons:
        report["status"] = "rejected"
        return
    report["stage"] = "registration"
    registry.register(report["run_id"], pipeline, report, features)
    report["stage"] = "deployment"
    registry.promote(report["run_id"])
    report.update({"status": "deployed", "active_version": report["run_id"], "stage": "complete"})


def _run_locked(settings: RetrainSettings, trigger: str) -> dict[str, Any]:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]
    directory = settings.store_dir / "runs" / run_id
    directory.mkdir(parents=True)
    serialized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(settings).items()
    }
    report: dict[str, Any] = {
        "run_id": run_id,
        "trigger": trigger,
        "status": "running",
        "settings": serialized,
        "started_at": datetime.now(UTC).isoformat(),
    }
    try:
        report.update(provenance())
        _execute(settings, directory, report)
    except InsufficientDataError as error:
        report.update({"status": "skipped", "reason": str(error)})
    except Exception as error:
        logger.exception("Retraining failed at %s", report.get("stage"))
        report.update({"status": "failed", "reason": f"{type(error).__name__}: {error}"})
    report["finished_at"] = datetime.now(UTC).isoformat()
    atomic_json(directory / "report.json", report)
    atomic_json(settings.store_dir / "last_run.json", report)
    return report


def run_retraining(settings: RetrainSettings, *, trigger: str = "manual") -> dict[str, Any]:
    """Perform one transaction; rejection, lack of data and busy workers are explicit outcomes."""
    try:
        with run_lock(settings.store_dir):
            return _run_locked(settings, trigger)
    except BlockingIOError:
        return {"status": "busy", "reason": "Another retraining or rollback is in progress"}


def main(argv: list[str] | None = None) -> None:
    """Run the same guarded process manually or from the operating-system scheduler."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "retraining.toml")
    parser.add_argument("--trigger", default="manual")
    parser.add_argument(
        "--rollback", action="store_true", help="Restore the previous registry version"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    settings = load_settings(args.config)
    if args.rollback:
        with run_lock(settings.store_dir):
            restored = ModelRegistry(settings.store_dir).rollback()
            report = {"status": "rolled_back", "active_version": restored}
            atomic_json(settings.store_dir / "last_rollback.json", report)
    else:
        report = run_retraining(settings, trigger=args.trigger)
    print(json.dumps(report, indent=2, allow_nan=False))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
