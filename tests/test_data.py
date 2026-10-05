from pathlib import Path

import pandas as pd
import pytest

from heart_disease.config import ALL_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from heart_disease.data import clean_data, load_raw_data, split_features_target


def test_load_raw_data_assigns_column_names(raw_csv: Path) -> None:
    df = load_raw_data(raw_csv)

    assert list(df.columns) == ALL_COLUMNS
    assert len(df) == 60


def test_load_raw_data_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "missing.csv")


def test_clean_data_binarizes_target(raw_csv: Path) -> None:
    raw = load_raw_data(raw_csv)

    cleaned = clean_data(raw)

    assert set(cleaned[TARGET_COLUMN].unique()) == {0, 1}
    assert (cleaned[TARGET_COLUMN] == (raw[TARGET_COLUMN] > 0)).all()


def test_clean_data_does_not_mutate_input(raw_csv: Path) -> None:
    raw = load_raw_data(raw_csv)
    snapshot = raw.copy()

    clean_data(raw)

    pd.testing.assert_frame_equal(raw, snapshot)


def test_clean_data_keeps_missing_values(clean_df: pd.DataFrame) -> None:
    # Imputation happens inside the pipeline after the split, not here.
    assert clean_df[["ca", "thal"]].isna().sum().sum() == 3


def test_split_features_target(clean_df: pd.DataFrame) -> None:
    features, target = split_features_target(clean_df)

    assert list(features.columns) == FEATURE_COLUMNS
    assert target.name == TARGET_COLUMN
    assert len(features) == len(target)
