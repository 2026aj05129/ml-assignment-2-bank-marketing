"""Exercise saved pipeline loading and evaluation through the Streamlit app."""

from io import BytesIO
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import classification_report, confusion_matrix
from streamlit.testing.v1 import AppTest

from src.modeling import MODEL_NAMES
from src.upload import ignored_columns, read_uploaded_csv, split_features_target

PROJECT_DIR = Path(__file__).resolve().parents[1]


def named_upload(content, name="test_data.csv"):
    upload = BytesIO(content)
    upload.name = name
    return upload


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_uploaded_holdout_matches_recorded_metrics(monkeypatch, model_name):
    uploaded = named_upload((PROJECT_DIR / "test_data.csv").read_bytes())
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    monkeypatch.setattr(st, "selectbox", lambda *args, **kwargs: model_name)
    expected = pd.read_csv(PROJECT_DIR / "results" / "metrics.csv").set_index("Model")

    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run(timeout=120)

    assert not app.exception
    assert len(app.metric) == 6
    assert any("CSV selected: test_data.csv" in item.value for item in app.markdown)
    assert any(
        all(
            text in element.value
            for text in (model_name, "9,043 records", "Actual no: 7,985",
                         "Actual yes: 1,058", "Positive: yes")
        )
        for element in app.markdown
    )
    for metric in app.metric:
        assert float(metric.value) == pytest.approx(
            expected.loc[model_name, metric.label], abs=0.000051
        )
    assert len(app.dataframe) == 2
    assert not app.tabs
    assert {"Performance summary", "Confusion matrix", "Classification report"} <= {
        heading.value for heading in app.subheader
    }
    assert "Data preview" in [expander.label for expander in app.expander]
    frame = pd.read_csv(PROJECT_DIR / "test_data.csv")
    model_filename = model_name.lower().replace(" ", "_").replace("-", "") + ".joblib"
    model = joblib.load(PROJECT_DIR / "model" / model_filename)
    features, target = split_features_target(frame, model)
    predictions = model.predict(features)
    matrix = confusion_matrix(target, predictions, labels=["yes", "no"])
    matrix_html = next(item.value for item in app.markdown if 'class="matrix-grid"' in item.value)
    for code, count, label in (
        ("TP", matrix[0, 0], "True positive"),
        ("FN", matrix[0, 1], "False negative"),
        ("FP", matrix[1, 0], "False positive"),
        ("TN", matrix[1, 1], "True negative"),
    ):
        assert (
            f'<span class="matrix-code">{code}</span>'
            f'<span class="matrix-count">{count:,}</span>'
            f'<span class="matrix-description">{label}</span>'
        ) in matrix_html
    assert [matrix_html.index(f'class="matrix-code">{code}') for code in ("TP", "FN", "FP", "TN")] == sorted(
        matrix_html.index(f'class="matrix-code">{code}') for code in ("TP", "FN", "FP", "TN")
    )
    assert any(
        all(
            item in element.value
            for item in (
                'class="screen-reader-only"',
                f"True positive: {matrix[0, 0]:,}",
                f"false negative: {matrix[0, 1]:,}",
                f"false positive: {matrix[1, 0]:,}",
                f"true negative: {matrix[1, 1]:,}",
            )
        )
        for element in app.markdown
    )
    report = classification_report(
        target, predictions, labels=model.classes_, output_dict=True, zero_division=0
    )
    expected_report = pd.DataFrame(
        {name: values for name, values in report.items() if name != "accuracy"}
    ).transpose().round(4)
    expected_report.index.name = "Class / average"
    expected_report["support"] = expected_report["support"].astype(int)
    pd.testing.assert_frame_equal(
        app.dataframe[0].value, expected_report
    )
    assert "accuracy" not in app.dataframe[0].value.index
    assert app.dataframe[0].value.loc["weighted avg", "support"] == len(target)
    assert app.dataframe[0].proto.height == 0
    assert any("Overall accuracy is shown" in caption.value for caption in app.caption)


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
    uploaded = named_upload(frame.to_csv(index=False).encode())
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


@pytest.mark.parametrize("delimiter", [",", ";"])
def test_upload_accepts_both_csv_formats(delimiter):
    source = pd.read_csv(PROJECT_DIR / "test_data.csv").head(100)
    frame = read_uploaded_csv(source.to_csv(index=False, sep=delimiter).encode())
    model = joblib.load(PROJECT_DIR / "model" / "logistic_regression.joblib")
    features, target = split_features_target(frame, model)
    assert features.shape == (100, 15)
    assert len(target) == 100


@pytest.mark.parametrize(
    "content, error",
    [
        (b"", "empty"),
        (b"\xff", "UTF-8"),
        (b"age,age,y\n1,2,no\n", "duplicate"),
        (b"age,,y\n1,2,no\n", "empty column"),
    ],
)
def test_malformed_csv_is_rejected(content, error):
    with pytest.raises(ValueError, match=error):
        read_uploaded_csv(content)


def test_upload_limits(monkeypatch):
    monkeypatch.setattr("src.upload.MAX_UPLOAD_BYTES", 100)
    with pytest.raises(ValueError, match="10 MB"):
        read_uploaded_csv(b"x" * 101)
    monkeypatch.setattr("src.upload.MAX_UPLOAD_BYTES", 1024)
    monkeypatch.setattr("src.upload.MAX_ROWS", 2)
    with pytest.raises(ValueError, match="2-row"):
        read_uploaded_csv(b"a,b\n1,2\n3,4\n5,6\n")


def test_extra_columns_and_missing_features_are_handled():
    frame = pd.read_csv(PROJECT_DIR / "test_data.csv").head(100)
    frame["duration"] = 20
    frame["extra"] = "unused"
    frame.loc[0, "age"] = np.nan
    frame.loc[0, "job"] = np.nan
    model = joblib.load(PROJECT_DIR / "model" / "logistic_regression.joblib")
    features, target = split_features_target(frame, model)
    assert ignored_columns(frame, model) == ["duration", "extra"]
    assert len(model.predict(features)) == len(target)


def test_landing_page_shows_summary_without_model_loading():
    app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run()
    assert not app.exception
    assert [metric.label for metric in app.metric] == [
        "Dataset rows", "Model inputs", "Classifiers", "Holdout rows"
    ]
    assert len(app.selectbox[0].options) == 5
    assert len(app.sidebar.selectbox) == 0
    assert len(app.get("popover")) == 1
    assert any(item.value == "No CSV selected" for item in app.caption)


def test_selected_filename_is_escaped_even_for_invalid_csv(monkeypatch):
    uploaded = named_upload(b"", 'a"<tag>.csv')
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run()
    assert not app.exception
    assert len(app.error) == 1
    assert any(
        'CSV selected: a&quot;&lt;tag&gt;.csv' in item.value
        for item in app.markdown
    )


@pytest.mark.parametrize("content", [b"", b"\xff", b"age,age,y\n1,2,no\n"])
def test_invalid_csv_bytes_show_error(monkeypatch, content):
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: named_upload(content))
    app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run()
    assert not app.exception
    assert len(app.error) == 1
    assert len(app.metric) == 0
