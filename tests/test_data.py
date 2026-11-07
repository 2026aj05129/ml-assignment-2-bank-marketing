"""Tests for assignment dataset loading and validation."""

import pandas as pd
import pytest

from src.data import LEAKAGE_FEATURES, TARGET, load_dataset


def make_dataset(rows: int = 500) -> pd.DataFrame:
    data = {f"feature_{index}": [index] * rows for index in range(12)}
    data["duration"] = [30] * rows
    data[TARGET] = ["yes", "no"] * (rows // 2)
    return pd.DataFrame(data)


def test_load_dataset_removes_target_and_duration(tmp_path):
    source = tmp_path / "bank.csv"
    make_dataset().to_csv(source, sep=";", index=False)

    features, target = load_dataset(source)

    assert len(features) == 500
    assert features.shape[1] == 12
    assert TARGET not in features.columns
    assert not set(LEAKAGE_FEATURES).intersection(features.columns)
    assert target.name == TARGET


def test_load_dataset_rejects_too_few_rows(tmp_path):
    source = tmp_path / "small.csv"
    make_dataset(rows=20).to_csv(source, sep=";", index=False)

    with pytest.raises(ValueError, match="at least 500 rows"):
        load_dataset(source)


def test_load_dataset_rejects_missing_target(tmp_path):
    source = tmp_path / "no-target.csv"
    make_dataset().drop(columns=TARGET).to_csv(source, sep=";", index=False)

    with pytest.raises(ValueError, match="target column"):
        load_dataset(source)
