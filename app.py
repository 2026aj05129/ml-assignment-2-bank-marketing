"""Evaluate saved Bank Marketing classifiers on a labeled CSV."""

import logging
import pickle
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from src.modeling import evaluate_model
from src.upload import ignored_columns, read_uploaded_csv, split_features_target

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
LOGGER = logging.getLogger(__name__)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
METRIC_HELP = {
    "Accuracy": "Share of all predictions that are correct.",
    "AUC": "How well scores rank subscribers above non-subscribers.",
    "Precision": "Share of predicted subscribers who actually subscribed.",
    "Recall": "Share of actual subscribers correctly identified.",
    "F1": "Harmonic mean of precision and recall.",
    "MCC": "Correlation between predictions and labels, from -1 to 1.",
}


@st.cache_resource(show_spinner="Loading saved model...")
def load_model(filename: str) -> Pipeline:
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        return joblib.load(MODEL_DIR / filename)


def main() -> None:
    st.set_page_config(page_title="Term Deposit Insights", page_icon="📊", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 5rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {
            background: #fffaf1;
            border: 1px solid #ead8bc;
            border-radius: 6px;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.caption("CAMPAIGN INTELLIGENCE / BANK MARKETING")
    st.title("Term Deposit Insights")
    st.write(
        "Investigate how five classification methods identify term-deposit "
        "subscribers in the UCI Bank Marketing dataset."
    )
    st.divider()
    st.subheader("Set up an analysis")
    upload_column, model_column = st.columns([3, 2], gap="large")
    with upload_column:
        st.markdown("**Campaign test file**")
        st.caption("Upload a labeled CSV to calculate performance.")
        uploaded = st.file_uploader("Upload campaign CSV", type="csv")
        sample_path = PROJECT_DIR / "test_data.csv"
        if sample_path.is_file():
            st.download_button(
                "Download example campaign data",
                sample_path.read_bytes(),
                file_name="test_data.csv",
                mime="text/csv",
            )
        else:
            st.error("Sample test data is missing from the project.")
    with model_column:
        st.markdown("**Classification method**")
        st.caption("Choose a saved model; changing it updates the results.")
        model_name = st.selectbox("Select a model", list(MODEL_FILES))
        st.caption("The saved preprocessing and model are used without retraining.")
        with st.expander("File format and data requirements"):
            st.write(
                "Include all 15 trained input columns and the y target. "
                "Both 'no' and 'yes' labels are needed for AUC. "
                "Comma- and semicolon-delimited UTF-8 files are supported "
                "(up to 10 MB and 100,000 rows)."
            )
            st.caption(
                "Missing feature values use the saved imputers. "
                "Additional columns, including duration, are ignored."
            )

    st.divider()
    if uploaded is None:
        st.subheader("Campaign dataset")
        for column, label, value in zip(
            st.columns(4),
            ("Dataset rows", "Model inputs", "Classifiers", "Holdout rows"),
            ("45,211", "15", "5", "9,043"),
            strict=True,
        ):
            column.metric(label, value)
        st.info("Upload campaign data above to see metrics and classification results.")
        st.subheader("Benchmark performance")
        st.caption(
            "Results from the original holdout split, not necessarily from an uploaded file. "
            "The positive class is 'yes'."
        )
        results_path = PROJECT_DIR / "results" / "metrics.csv"
        if results_path.is_file():
            st.dataframe(
                pd.read_csv(results_path).round(4),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.error("Recorded holdout metrics are missing from the project.")
        return

    try:
        model = load_model(MODEL_FILES[model_name])
    except (
        OSError, EOFError, pickle.UnpicklingError, InconsistentVersionWarning,
        ImportError, AttributeError, ValueError,
    ):
        LOGGER.exception("Failed to load saved model %s", model_name)
        st.error(
            "Could not load the saved model. Check that the model file is present "
            "and that Python 3.12 and the pinned dependencies are installed."
        )
        return

    try:
        frame = read_uploaded_csv(uploaded.getvalue())
        features, target = split_features_target(frame, model)
    except (ValueError, pd.errors.ParserError) as error:
        st.error(f"Cannot evaluate this CSV: {error}")
        return

    extra = ignored_columns(frame, model)
    if extra:
        st.info(f"Ignored columns not used by the model: {', '.join(extra)}.")
    try:
        with st.spinner("Evaluating the selected model..."):
            metrics = evaluate_model(model, features, target)
            predictions = model.predict(features)
    except (ValueError, TypeError) as error:
        LOGGER.exception("Failed to evaluate model %s", model_name)
        st.error(f"Evaluation failed. Check the CSV feature values: {error}")
        return

    st.subheader("Performance summary")
    st.write(
        f"**{model_name}** | {len(features):,} rows | "
        f"{int((target == 'no').sum()):,} no | {int((target == 'yes').sum()):,} yes"
    )
    st.caption("Positive class: yes. Metrics use the uploaded labels and saved pipeline.")
    for row in (list(metrics.items())[:3], list(metrics.items())[3:]):
        for column, (name, value) in zip(st.columns(3), row, strict=True):
            column.metric(name, f"{value:.4f}", help=METRIC_HELP[name])

    matrix_tab, report_tab, data_tab = st.tabs(
        ["Confusion matrix", "Classification report", "Data preview"]
    )
    with matrix_tab:
        matrix = confusion_matrix(target, predictions, labels=model.classes_)
        st.caption("Rows are actual labels; columns are predicted labels.")
        st.write(
            f"**Correct no:** {matrix[0, 0]:,} | **Correct yes:** {matrix[1, 1]:,} | "
            f"**False positives:** {matrix[0, 1]:,} | **False negatives:** {matrix[1, 0]:,}"
        )
        figure, axis = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(matrix, display_labels=model.classes_        ).plot(ax=axis, cmap="YlOrBr", colorbar=False)
        st.pyplot(figure, use_container_width=False)
        plt.close(figure)
    with report_tab:
        report = classification_report(
            target, predictions, labels=model.classes_, output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
    with data_tab:
        st.caption(f"Showing the first {min(25, len(frame)):,} of {len(frame):,} rows.")
        st.dataframe(frame.head(25), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
