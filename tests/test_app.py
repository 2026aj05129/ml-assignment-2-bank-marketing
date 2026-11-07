"""Exercise saved pipeline loading and evaluation through the Streamlit app."""

from io import BytesIO
from pathlib import Path
import warnings

import pandas as pd
import pytest
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from streamlit.testing.v1 import AppTest

from src.modeling import MODEL_NAMES

PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_uploaded_holdout_matches_recorded_metrics(monkeypatch, model_name):
    uploaded = BytesIO((PROJECT_DIR / "test_data.csv").read_bytes())
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    monkeypatch.setattr(st, "selectbox", lambda *args, **kwargs: model_name)
    expected = pd.read_csv(PROJECT_DIR / "results" / "metrics.csv").set_index("Model")

    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run(timeout=120)

    assert not app.exception
    assert len(app.metric) == 6
    for metric in app.metric:
        assert float(metric.value) == pytest.approx(
            expected.loc[model_name, metric.label], abs=0.000051
        )
    assert len(app.dataframe) == 2
    assert [tab.label for tab in app.tabs] == [
        "Confusion matrix", "Classification report", "Data preview"
    ]


@pytest.mark.parametrize("fault", ["empty", "no_target", "missing_feature", "single_class", "unknown_label", "invalid_numeric", "infinity"])
def test_invalid_upload_shows_error(monkeypatch, fault):
    frame = pd.read_csv(PROJECT_DIR / "test_data.csv").head(30)
    if fault == "empty":
        frame = frame.iloc[:0]
    elif fault == "no_target":
        frame = frame.drop(columns=["y"])
    elif fault == "missing_feature":
        frame = frame.drop(columns=["age"])
    elif fault == "single_class":
        frame["y"] = "no"
    elif fault == "unknown_label":
        frame.loc[frame.index[0], "y"] = "unknown"
    elif fault == "invalid_numeric":
        frame["age"] = "invalid"
    elif fault == "infinity":
        frame["age"] = frame["age"].astype(float)
        frame.loc[frame.index[0], "age"] = float("inf")
    uploaded = BytesIO(frame.to_csv(index=False).encode())
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run(timeout=120)
    assert not app.exception
    assert len(app.error) == 1
    assert len(app.metric) == 0


def test_readme_metrics_match_results():
    expected = pd.read_csv(PROJECT_DIR / "results" / "metrics.csv").set_index("Model")
    found = set()
    for line in (PROJECT_DIR / "README.md").read_text().splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 7 and cells[0] in MODEL_NAMES:
            assert [float(value) for value in cells[1:]] == [
                round(value, 4) for value in expected.loc[cells[0]]
            ]
            found.add(cells[0])
    assert found == set(MODEL_NAMES)
