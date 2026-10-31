"""Ingest labelled local batches and construct patient-disjoint evaluation data."""

import hashlib
import io

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from heart_disease.config import ALL_COLUMNS, BINARY_CLASSES, FEATURE_COLUMNS, TARGET_COLUMN
from heart_disease.retrain_settings import RetrainSettings

CATEGORY_VALUES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}


class InsufficientDataError(ValueError):
    """A valid dataset does not yet contain enough new independent observations."""


def digest(value: str) -> str:
    """Return a stable fingerprint without persisting a source patient identifier."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_numeric(frame: pd.DataFrame, max_missing: float) -> pd.DataFrame:
    """Validate labels before binarization, units-compatible numbers and category codes."""
    numeric: pd.DataFrame = frame[ALL_COLUMNS].apply(pd.to_numeric, errors="raise").astype(float)
    if numeric.empty or np.isinf(numeric.to_numpy()).any():
        raise ValueError("Empty data or infinite values")
    if not numeric[TARGET_COLUMN].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("target must be present and belong to 0..4")
    if numeric[FEATURE_COLUMNS].isna().mean().gt(max_missing).any():
        raise ValueError("Too many missing feature values")
    for column, allowed in CATEGORY_VALUES.items():
        if not numeric[column].dropna().isin(allowed).all():
            raise ValueError(f"Invalid category: {column}")
    for column in ("age", "trestbps", "chol", "thalach"):
        if numeric[column].dropna().le(0).any():
            raise ValueError(f"Expected positive measurement: {column}")
    numeric[TARGET_COLUMN] = (numeric[TARGET_COLUMN] > 0).astype(int)
    return numeric


def _batch(payload: bytes, settings: RetrainSettings, *, legacy: bool) -> pd.DataFrame:
    source = pd.read_csv(io.BytesIO(payload), header=None if legacy else "infer")
    if legacy:
        if source.shape[1] != len(ALL_COLUMNS):
            raise ValueError("Legacy CSV must contain exactly 14 columns")
        source.columns = ALL_COLUMNS
    elif set(source.columns) != {*ALL_COLUMNS, "patient_id"}:
        raise ValueError("Expected patient_id and the 14 Cleveland columns")
    frame = validate_numeric(source, settings.max_missing)
    rows = frame[FEATURE_COLUMNS].to_csv(index=False, header=False, float_format="%.17g")
    frame["record_id"] = [digest(row) for row in rows.splitlines()]
    if legacy:
        frame["patient_id"] = "legacy:" + frame["record_id"]
    else:
        identifiers = source["patient_id"].astype("string").str.strip()
        if identifiers.isna().any() or identifiers.eq("").any():
            raise ValueError("Every incoming record needs a pseudonymous patient_id")
        frame["patient_id"] = identifiers.map(lambda value: digest(f"patient:{value}"))
    return frame


def load_dataset(settings: RetrainSettings) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read historical data plus all complete *.csv batches; reject conflicting labels."""
    paths = [settings.raw_path, *sorted(settings.incoming_dir.glob("*.csv"))]
    batches = []
    manifest = {}
    for index, path in enumerate(paths):
        payload = path.read_bytes()
        batches.append(_batch(payload, settings, legacy=index == 0))
        manifest[str(path)] = hashlib.sha256(payload).hexdigest()
    frame = pd.concat(batches, ignore_index=True)
    if frame.groupby("record_id")[TARGET_COLUMN].nunique().gt(1).any():
        raise ValueError("Identical features have conflicting target labels")
    if frame.groupby("record_id")["patient_id"].nunique().gt(1).any():
        raise ValueError("Identical features have conflicting patient identities; resolve overlaps")
    frame = frame.drop_duplicates("record_id", keep="first").reset_index(drop=True)
    return frame, manifest


def split_dataset(
    frame: pd.DataFrame,
    history: dict[str, list[str]],
    settings: RetrainSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve a fresh holdout; never train on any previously evaluated patient."""
    new_rows = ~frame["record_id"].isin(history.get("seen_records", []))
    if int(new_rows.sum()) < settings.min_new_rows:
        raise InsufficientDataError("Not enough new unique labelled records")
    fresh = frame.loc[~frame["patient_id"].isin(history.get("known_patients", []))]
    if fresh[TARGET_COLUMN].nunique() != len(BINARY_CLASSES) or fresh["patient_id"].nunique() < len(
        BINARY_CLASSES
    ):
        raise InsufficientDataError("Fresh evaluation patients must cover both classes")
    splitter = GroupShuffleSplit(
        n_splits=20, test_size=settings.test_size, random_state=settings.seed
    )
    for _, indices in splitter.split(fresh, groups=fresh["patient_id"]):
        evaluation = fresh.iloc[indices]
        counts = evaluation[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
        if int(counts.min()) < settings.min_eval_per_class:
            continue
        excluded = set(history.get("heldout_patients", [])) | set(evaluation["patient_id"])
        development = frame.loc[~frame["patient_id"].isin(excluded)]
        groups_per_class = development.groupby(TARGET_COLUMN)["patient_id"].nunique()
        if (
            len(groups_per_class) == len(BINARY_CLASSES)
            and int(groups_per_class.min()) >= settings.cv_folds
        ):
            return development.reset_index(drop=True), evaluation.reset_index(drop=True)
    raise InsufficientDataError("Not enough class coverage for independent evaluation and CV")
