"""Immutable local model versions and atomic deployment for Python consumers."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from heart_disease.config import FEATURE_COLUMNS


def atomic_json(path: Path, value: object) -> None:
    """Replace a JSON file atomically on its own filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    """Read an object; invalid persisted state must never silently reset."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def smoke_check(pipeline: Pipeline, features: pd.DataFrame) -> None:
    """Check the fitted input contract, classes, finite outputs and NaN support."""
    if list(features.columns) != FEATURE_COLUMNS or features.empty:
        raise ValueError("Expected a nonempty DataFrame with 13 ordered feature columns")
    sample = features.head(8).copy()
    # Exercise both ordinary input and the imputation saved inside the Pipeline.
    missing = sample.copy()
    missing.iloc[0, 0] = np.nan
    for batch in (sample, missing):
        predictions = np.asarray(pipeline.predict(batch))
        if predictions.shape != (len(batch),) or not np.isin(predictions, [0, 1]).all():
            raise ValueError("Invalid prediction shape or classes")
        if hasattr(pipeline, "predict_proba"):
            probabilities = np.asarray(pipeline.predict_proba(batch))
            valid = (
                probabilities.shape == (len(batch), 2)
                and np.isfinite(probabilities).all()
                and ((probabilities >= 0) & (probabilities <= 1)).all()
                and np.allclose(probabilities.sum(axis=1), 1)
            )
            if not valid:
                raise ValueError("Invalid probabilities")


class ModelRegistry:
    """Register versions, promote a pointer and restore it after failed activation."""

    def __init__(self, root: Path) -> None:
        """Use one persistent directory shared by training and inference."""
        self.root = root
        self.active_path = root / "active.json"

    def version_dir(self, version: str) -> Path:
        """Reject traversal in persisted or user-provided version identifiers."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", version):
            raise ValueError("Invalid model version")
        return self.root / "registry" / version

    def active(self) -> dict[str, Any] | None:
        """Read the current pointer without loading a pickle."""
        return read_json(self.active_path) if self.active_path.exists() else None

    def register(
        self, version: str, pipeline: Pipeline, metadata: dict[str, Any], sample: pd.DataFrame
    ) -> None:
        """Save a new complete version; existing versions cannot be overwritten."""
        destination = self.version_dir(version)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            staging = Path(temporary) / version
            staging.mkdir()
            model_path = staging / "model.joblib"
            joblib.dump(pipeline, model_path)
            sample.head(8).to_csv(staging / "smoke.csv", index=False)
            metadata = {
                **metadata,
                "version": version,
                "features": FEATURE_COLUMNS,
                "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            }
            atomic_json(staging / "metadata.json", metadata)
            smoke_check(cast(Pipeline, joblib.load(model_path)), sample)
            staging.rename(destination)

    def load(self, version: str) -> Pipeline:
        """Verify integrity and schema before loading an artifact from the local registry."""
        directory = self.version_dir(version)
        metadata = read_json(directory / "metadata.json")
        path = directory / "model.joblib"
        if metadata.get("version") != version or metadata.get("features") != FEATURE_COLUMNS:
            raise ValueError("Model metadata does not match the requested contract")
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("sha256"):
            raise ValueError("Model artifact checksum mismatch")
        pipeline = joblib.load(path)
        if not isinstance(pipeline, Pipeline):
            raise ValueError("Registry artifact is not an sklearn Pipeline")
        return pipeline

    def promote(self, version: str) -> None:
        """Stage a version and check a fresh consumer after atomic pointer replacement."""
        sample = pd.read_csv(self.version_dir(version) / "smoke.csv")
        smoke_check(self.load(version), sample)
        previous = self.active()
        pointer = {
            "version": version,
            "previous_version": previous["version"] if previous else None,
        }
        atomic_json(self.active_path, pointer)
        try:
            ActiveModel(self.root).predict(sample)
        except BaseException:
            if previous is None:
                self.active_path.unlink(missing_ok=True)
            else:
                atomic_json(self.active_path, previous)
            raise

    def rollback(self) -> str:
        """Restore the previous verified version and permit future explicit rollbacks."""
        pointer = self.active()
        if pointer is None or pointer.get("previous_version") is None:
            raise ValueError("No previous registered version is available")
        version = str(pointer["previous_version"])
        self.promote(version)
        return version


class ActiveModel:
    """An inference consumer that reloads automatically when active.json changes."""

    def __init__(self, root: Path) -> None:
        """Start without a cached artifact; failures never fall back silently."""
        self.registry = ModelRegistry(root)
        self.version: str | None = None
        self.pipeline: Pipeline | None = None

    def predict(self, features: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Check the active pointer on each call and return one binary label per row."""
        if list(features.columns) != FEATURE_COLUMNS:
            raise ValueError("Feature names and order must match FEATURE_COLUMNS")
        pointer = self.registry.active()
        if pointer is None:
            raise FileNotFoundError("No active model; run heart-disease-retrain first")
        version = str(pointer["version"])
        if self.pipeline is None or self.version != version:
            pipeline = self.registry.load(version)
            smoke_check(pipeline, features)
            self.pipeline, self.version = pipeline, version
        predictions = np.asarray(self.pipeline.predict(features))
        if predictions.shape != (len(features),) or not np.isin(predictions, [0, 1]).all():
            raise ValueError("Invalid model output")
        return predictions
