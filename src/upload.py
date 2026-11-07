"""Parse and validate labeled Bank Marketing evaluation data."""

import csv
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import TARGET

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000


def read_uploaded_csv(content: bytes) -> pd.DataFrame:
    if not content:
        raise ValueError("The CSV is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("The CSV exceeds the 10 MB upload limit.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Save the CSV using UTF-8 encoding.") from error
    try:
        first_line = text.splitlines()[0]
        delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
        header = next(csv.reader(StringIO(text), delimiter=delimiter, strict=True))
    except (IndexError, StopIteration, csv.Error) as error:
        raise ValueError("The CSV header could not be read.") from error
    if len(set(header)) != len(header):
        raise ValueError("The CSV contains duplicate column names.")
    if any(not name.strip() for name in header):
        raise ValueError("The CSV contains an empty column name.")
    frame = pd.read_csv(BytesIO(content), sep=delimiter, nrows=MAX_ROWS + 1)
    if len(frame) > MAX_ROWS:
        raise ValueError(f"The CSV exceeds the {MAX_ROWS:,}-row limit.")
    return frame


def split_features_target(
    frame: pd.DataFrame, model: Pipeline
) -> tuple[pd.DataFrame, pd.Series]:
    if frame.empty:
        raise ValueError("The uploaded CSV contains no data rows.")
    expected = list(model.feature_names_in_)
    missing = [column for column in [*expected, TARGET] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}.")
    target = frame[TARGET]
    if target.isna().any() or set(target.unique()) != set(model.classes_):
        raise ValueError(
            "The y column must contain both labels 'no' and 'yes', with no "
            "missing or unexpected values. Both classes are required for AUC."
        )
    features = frame.loc[:, expected].copy()
    preprocessor = model.named_steps["preprocessor"]
    numeric_columns = preprocessor.transformers_[0][2]
    categorical_columns = preprocessor.transformers_[1][2]
    for column in numeric_columns:
        try:
            features[column] = pd.to_numeric(features[column], errors="raise")
        except ValueError as error:
            raise ValueError(f"Feature '{column}' must contain numeric values.") from error
        if np.isinf(features[column]).any():
            raise ValueError(f"Feature '{column}' must not contain infinity.")
    for column in categorical_columns:
        features[column] = features[column].map(
            lambda value: str(value) if pd.notna(value) else np.nan
        )
    return features, target


def ignored_columns(frame: pd.DataFrame, model: Pipeline) -> list[str]:
    """Return additional columns, including the excluded duration if present."""
    expected = set(model.feature_names_in_) | {TARGET}
    return [column for column in frame.columns if column not in expected]
