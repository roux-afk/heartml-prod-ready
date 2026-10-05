"""Loading and cleaning of the Cleveland heart disease dataset."""

import logging
from pathlib import Path

import pandas as pd

from heart_disease.config import ALL_COLUMNS, FEATURE_COLUMNS, RAW_DATA_PATH, TARGET_COLUMN

logger = logging.getLogger(__name__)


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw headerless CSV and assign column names."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, header=None, names=ALL_COLUMNS)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Binarize the target: 0 = no disease, 1-4 = disease present.

    Missing values are intentionally left untouched: they are imputed
    inside the model pipeline after the train/test split to avoid leakage.
    """
    cleaned = df.copy()
    cleaned[TARGET_COLUMN] = (cleaned[TARGET_COLUMN] > 0).astype(int)
    return cleaned


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix from the target vector."""
    return df[FEATURE_COLUMNS], df[TARGET_COLUMN]