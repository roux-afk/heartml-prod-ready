"""Exploratory data analysis plots, saved to disk instead of shown interactively."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from heart_disease.config import FIGURES_DIR, LOG_FORMAT, TARGET_COLUMN
from heart_disease.data import clean_data, load_raw_data

logger = logging.getLogger(__name__)

SEX_LABELS = {0: "female", 1: "male"}


def plot_age_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Count of patients per age, split by target class."""
    grid = sns.catplot(
        kind="count",
        data=df,
        x="age",
        hue=TARGET_COLUMN,
        order=sorted(df["age"].unique()),
        height=5,
        aspect=3,
    )
    grid.figure.suptitle("Variation of age for each target class", y=1.02)
    return _save_figure(grid.figure, output_dir / "age_distribution.png")


def plot_age_by_sex(df: pd.DataFrame, output_dir: Path) -> Path:
    """Mean age per sex, split by target class."""
    labeled = df.assign(sex=df["sex"].map(SEX_LABELS))
    grid = sns.catplot(kind="bar", data=labeled, x="sex", y="age", hue=TARGET_COLUMN)
    grid.figure.suptitle("Distribution of age vs sex with the target class", y=1.02)
    return _save_figure(grid.figure, output_dir / "age_by_sex.png")


def _save_figure(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved figure to %s", path)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    df = clean_data(load_raw_data())

    sns.set_context("paper", font_scale=1.5)
    plot_age_distribution(df, FIGURES_DIR)
    plot_age_by_sex(df, FIGURES_DIR)


if __name__ == "__main__":
    main()