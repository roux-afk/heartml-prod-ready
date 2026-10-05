"""Project-wide configuration: paths, dataset schema and training constants."""

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_PATH: Final[Path] = DATA_DIR / "raw" / "cleveland.csv"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"

FEATURE_COLUMNS: Final[list[str]] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COLUMN: Final[str] = "target"
ALL_COLUMNS: Final[list[str]] = [*FEATURE_COLUMNS, TARGET_COLUMN]

TEST_SIZE: Final[float] = 0.2
RANDOM_STATE: Final[int] = 0

LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(name)s: %(message)s"
