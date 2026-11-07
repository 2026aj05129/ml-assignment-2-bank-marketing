"""Dataset loading and checks for the UCI Bank Marketing assignment."""

from pathlib import Path

import pandas as pd

TARGET = "y"
LEAKAGE_FEATURES = ("duration",)
MINIMUM_ROWS = 500
MINIMUM_FEATURES = 12


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the labeled UCI CSV and separate its predictors from its target."""
    data = pd.read_csv(path, sep=";")

    if len(data) < MINIMUM_ROWS:
        raise ValueError(
            f"Dataset must contain at least {MINIMUM_ROWS} rows; found {len(data)}."
        )
    if TARGET not in data.columns:
        raise ValueError(f"Dataset must contain the target column {TARGET!r}.")

    predictors = data.drop(
        columns=[TARGET, *[name for name in LEAKAGE_FEATURES if name in data.columns]]
    )
    if predictors.shape[1] < MINIMUM_FEATURES:
        raise ValueError(
            f"Dataset must contain at least {MINIMUM_FEATURES} input features "
            f"after removing the target and leakage features; found {predictors.shape[1]}."
        )

    if data[TARGET].isna().any():
        raise ValueError(f"Target column {TARGET!r} contains missing values.")
    if data[TARGET].nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")

    return predictors, data[TARGET]
