"""Validated configuration for local automatic retraining."""

import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from heart_disease.config import PROJECT_ROOT
from heart_disease.models import MODEL_REGISTRY

MIN_CV_FOLDS = 2


@dataclass(frozen=True)
class RetrainSettings:
    """Paths and demonstration quality thresholds, independent of a fitted model."""

    raw_path: Path = PROJECT_ROOT / "data/raw/cleveland.csv"
    incoming_dir: Path = PROJECT_ROOT / "data/incoming"
    store_dir: Path = PROJECT_ROOT / "models"
    models: tuple[str, ...] = tuple(MODEL_REGISTRY)
    min_new_rows: int = 100
    min_eval_per_class: int = 10
    cv_folds: int = 5
    test_size: float = 0.2
    max_missing: float = 0.05
    min_recall: float = 0.90
    min_precision: float = 0.80
    min_f1: float = 0.85
    min_improvement: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject thresholds that could accidentally disable the quality gate."""
        for name in ("min_new_rows", "min_eval_per_class", "cv_folds", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"Invalid integer setting: {name}")
        if self.cv_folds < MIN_CV_FOLDS:
            raise ValueError("cv_folds must be at least 2")
        for name in (
            "test_size",
            "max_missing",
            "min_recall",
            "min_precision",
            "min_f1",
            "min_improvement",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Invalid numeric setting: {name}")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Setting outside [0, 1]: {name}")
        if not 0 < self.test_size < 1 or self.min_improvement <= 0:
            raise ValueError("test_size must be inside (0, 1); min_improvement must be positive")
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("models must be nonempty and unique")
        if set(self.models) - MODEL_REGISTRY.keys():
            raise ValueError("Unknown model in retraining configuration")


def load_settings(path: Path) -> RetrainSettings:
    """Read TOML; resolve relative paths against the configuration directory."""
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(document) != {"retraining"}:
        raise ValueError("Configuration must contain only [retraining]")
    values: dict[str, Any] = document["retraining"]
    unknown = set(values) - {field.name for field in fields(RetrainSettings)}
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    for name in ("raw_path", "incoming_dir", "store_dir"):
        if name in values:
            values[name] = (path.resolve().parent / values[name]).resolve()
    if "models" in values:
        values["models"] = tuple(values["models"])
    return RetrainSettings(**values)
