from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heart_disease.config import FEATURE_COLUMNS
from heart_disease.data import clean_data, load_raw_data

N_ROWS = 60


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    """Small synthetic dataset in the raw Cleveland format: no header, NaNs, target 0-4."""
    rng = np.random.default_rng(seed=42)
    data = pd.DataFrame(
        rng.integers(0, 4, size=(N_ROWS, len(FEATURE_COLUMNS))).astype(float),
        columns=FEATURE_COLUMNS,
    )
    data["age"] = rng.integers(30, 80, size=N_ROWS)
    data["sex"] = rng.integers(0, 2, size=N_ROWS)
    data.loc[[0, 5], "ca"] = np.nan
    data.loc[3, "thal"] = np.nan
    data["target"] = np.tile([0, 1, 2, 3, 4, 0], N_ROWS // 6)

    path = tmp_path / "raw.csv"
    data.to_csv(path, header=False, index=False)
    return path


@pytest.fixture
def clean_df(raw_csv: Path) -> pd.DataFrame:
    return clean_data(load_raw_data(raw_csv))
