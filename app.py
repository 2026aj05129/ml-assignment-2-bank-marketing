"""Streamlit evaluation app for ML Assignment 2 (UCI Bank Marketing).

Loads the five fitted pipelines saved by ``train_models.py`` and evaluates
whichever one the user picks on an uploaded, labeled test CSV. The app never
retrains anything: preprocessing and classifiers come straight from the
``model/`` artifacts.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from src.data import LEAKAGE_FEATURES, TARGET
from src.modeling import evaluate_model

PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
METRIC_ORDER = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")


@st.cache_resource(show_spinner="Loading fitted pipeline...")
def load_model(filename: str):
    return joblib.load(MODEL_DIR / filename)


@st.cache_data(show_spinner=False)
def read_uploaded_csv(uploaded_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO

    # test_data.csv is comma-separated; the raw UCI file uses semicolons.
    frame = pd.read_csv(BytesIO(uploaded_bytes))
    if frame.shape[1] == 1:
        frame = pd.read_csv(BytesIO(uploaded_bytes), sep=";")
    return frame


def split_features_target(frame: pd.DataFrame, model):
    expected = list(model.feature_names_in_)
    if frame.empty:
        raise ValueError("The uploaded CSV contains no data rows.")
    if TARGET not in frame.columns:
        raise ValueError(
            f"The uploaded CSV must include the target column `{TARGET}`."
        )
    features = frame.drop(
        columns=[TARGET, *[c for c in LEAKAGE_FEATURES if c in frame.columns]]
    )
    missing = [c for c in expected if c not in features.columns]
    if missing:
        raise ValueError(f"The uploaded CSV is missing feature columns: {missing}")
    target = frame[TARGET]
    if target.isna().any() or set(target.unique()) != set(model.classes_):
        raise ValueError(
            "The target column must contain both labels `no` and `yes`, "
            "without missing or unexpected values. Both classes are needed for AUC."
        )
    features = features.loc[:, expected].copy()
    numeric_columns = model.named_steps["preprocessor"].transformers_[0][2]
    for column in numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="raise")
        if np.isinf(features[column]).any():
            raise ValueError(f"Numeric feature `{column}` must not contain infinity.")
    return features, target


def main() -> None:
    st.set_page_config(page_title="Bank Marketing Classifier Lab", page_icon="🏦")
    st.title("🏦 Bank Marketing — Term Deposit Classifier Lab")
    st.caption(
        "ML Assignment 2 · UCI Bank Marketing dataset · Five pre-trained "
        "classifiers evaluated on your uploaded test split."
    )

    missing_models = [f for f in MODEL_FILES.values() if not (MODEL_DIR / f).is_file()]
    if missing_models:
        st.error(
            "Saved pipelines not found. Run `python train_models.py` first to "
            f"create the `model/` folder. Missing: {missing_models}"
        )
        st.stop()

    with st.sidebar:
        st.header("1 · Upload labeled test data")
        uploaded = st.file_uploader(
            "CSV with all input features and the `y` target column "
            "(for example the repository's `test_data.csv`).",
            type="csv",
        )
        st.header("2 · Pick a classifier")
        model_name = st.selectbox("Model", list(MODEL_FILES))
        st.caption(
            "Pipelines were fitted on an 80% stratified training split; the "
            "`duration` column is excluded to avoid target leakage."
        )

    if uploaded is None:
        st.info(
            "⬅️ Upload a labeled test CSV to begin. Tip: `test_data.csv` in the "
            "repository is the 20% holdout split produced by `train_models.py`."
        )
        st.stop()

    model = load_model(MODEL_FILES[model_name])
    try:
        frame = read_uploaded_csv(uploaded.getvalue())
        features, target = split_features_target(frame, model)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Cannot evaluate this CSV: {error}")
        st.stop()

    st.subheader(f"Evaluation — {model_name}")
    st.write(
        f"Scoring **{len(features):,} rows** with the saved pipeline "
        "(no retraining)."
    )

    metrics = evaluate_model(model, features, target)
    columns = st.columns(3)
    for i, name in enumerate(METRIC_ORDER):
        columns[i % 3].metric(name, f"{metrics[name]:.4f}")

    predictions = model.predict(features)

    tab_cm, tab_report, tab_data = st.tabs(
        ["Confusion matrix", "Classification report", "Data preview"]
    )

    with tab_cm:
        figure, axis = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix(target, predictions, labels=model.classes_),
            display_labels=model.classes_,
        ).plot(ax=axis, cmap="Blues", colorbar=False)
        axis.set_title(f"{model_name} — confusion matrix")
        st.pyplot(figure)
        plt.close(figure)

    with tab_report:
        report = classification_report(target, predictions, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    with tab_data:
        st.dataframe(frame.head(25), use_container_width=True)
        st.caption(f"Showing the first 25 of {len(frame):,} uploaded rows.")


if __name__ == "__main__":
    main()
