"""Evaluate saved Bank Marketing classifiers on a labeled CSV."""

import logging
import pickle
from html import escape
from io import BytesIO
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import classification_report, confusion_matrix
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


def make_confusion_chart(matrix):
    """Draw a two-by-two matrix with actual classes as rows."""
    figure, axis = plt.subplots(figsize=(6.4, 2.45))
    figure.patch.set_facecolor("#fffcf6")
    axis.set_facecolor("#fffcf6")
    axis.text(0.13, 0.99, "ACTUAL", fontsize=9, fontweight="bold", color="#66584d")
    axis.text(0.62, 0.99, "PREDICTED", fontsize=9, fontweight="bold", color="#66584d")
    for col, label in enumerate(("YES", "NO")):
        axis.text(0.47 + col * 0.37, 0.88, label, ha="center",
                  fontsize=10, fontweight="bold", color="#3c3028")
    names = (("True positive", "False negative"), ("False positive", "True negative"))
    abbreviations = (("TP", "FN"), ("FP", "TN"))
    for row, label in enumerate(("YES", "NO")):
        y = 0.48 - row * 0.42
        axis.text(0.13, y + 0.16, label, va="center", fontsize=10,
                  fontweight="bold", color="#3c3028")
        for col in range(2):
            x = 0.30 + col * 0.37
            correct = row == col
            axis.add_patch(FancyBboxPatch(
                (x, y), 0.34, 0.34, boxstyle="round,pad=0.005,rounding_size=0.025",
                facecolor="#e8f2ef" if correct else "#fff2e2",
                edgecolor="#a4c9bd" if correct else "#e5bd87", linewidth=1,
            ))
            axis.text(x + 0.025, y + 0.27, abbreviations[row][col],
                      fontsize=9, fontweight="bold",
                      color="#216b5d" if correct else "#955414")
            axis.text(x + 0.025, y + 0.15, f"{matrix[row, col]:,}",
                      fontsize=16, fontweight="bold", color="#243b36", va="center")
            axis.text(
                x + 0.025, y + 0.03, names[row][col], fontsize=8.5, color="#51443a",
            )
    axis.set(xlim=(0, 1.05), ylim=(0, 1.05))
    axis.axis("off")
    figure.tight_layout(pad=0.2)
    return figure


def main() -> None:
    st.set_page_config(
        page_title="Term Deposit Insights", page_icon="📊",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        [data-testid="stVerticalBlock"] {gap: 0.6rem;}
        [data-testid="stMetric"] {
            background: #fffaf1;
            border: 1px solid #ead8bc;
            border-radius: 6px;
            padding: 0.6rem;
        }
        .evaluation-summary {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.4rem;
        }
        .evaluation-summary span {
            background: #f4ede2;
            border: 1px solid #e8dac7;
            border-radius: 999px;
            padding: 0.2rem 0.65rem;
            color: #4b3b30;
            font-size: 0.85rem;
        }
        .evaluation-summary .active-model {
            background: #e4f2ef;
            border-color: #b6d9d1;
            color: #1c5f58;
            font-weight: 600;
        }
        .upload-status {
            color: #806f62;
            font-size: 0.875rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .screen-reader-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Term Deposit Insights")
    st.caption("UCI Bank Marketing | Select a model and upload labeled campaign data")
    model_column, upload_column, sample_column = st.columns([3, 3, 2], gap="medium")
    with model_column:
        st.caption("Classification method")
        model_name = st.selectbox(
            "Classification method", list(MODEL_FILES), label_visibility="collapsed"
        )
    with upload_column:
        st.caption("Evaluation data")
        with st.popover("Upload or change CSV", use_container_width=True):
            uploaded = st.file_uploader("Upload campaign CSV", type="csv")
            st.caption(
                "Include all 15 trained inputs and both target labels, no and yes. "
                "UTF-8 comma- or semicolon-delimited CSV, up to 10 MB and "
                "100,000 rows. Missing feature values use the saved imputers; "
                "extra columns including duration are ignored."
            )
        if uploaded is None:
            st.caption("No CSV selected")
        else:
            filename = escape(uploaded.name, quote=True)
            st.markdown(
                f'<div class="upload-status" title="{filename}">CSV selected: {filename}</div>',
                unsafe_allow_html=True,
            )
    with sample_column:
        st.caption("Example data")
        sample_path = PROJECT_DIR / "test_data.csv"
        if sample_path.is_file():
            st.download_button(
                "Download sample CSV",
                sample_path.read_bytes(),
                file_name="test_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.error("Sample test data is missing from the project.")
    if uploaded is None:
        st.subheader("Campaign dataset")
        for column, label, value in zip(
            st.columns(4),
            ("Dataset rows", "Model inputs", "Classifiers", "Holdout rows"),
            ("45,211", "15", "5", "9,043"),
            strict=True,
        ):
            column.metric(label, value)
        st.info("Select 'Upload or change CSV' above to evaluate the selected model.")
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

    st.markdown(
        '<div class="evaluation-summary">'
        f'<span class="active-model">{model_name}</span>'
        f"<span>{len(features):,} records</span>"
        f"<span>Actual no: {int((target == 'no').sum()):,}</span>"
        f"<span>Actual yes: {int((target == 'yes').sum()):,}</span>"
        "<span>Positive: yes</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Performance summary")
    for column, (name, value) in zip(st.columns(6), metrics.items(), strict=True):
        column.metric(name, f"{value:.4f}", help=METRIC_HELP[name])
    matrix_column, report_column = st.columns(2, gap="large")
    with matrix_column:
        st.subheader("Confusion matrix", anchor="confusion-matrix")
        matrix = confusion_matrix(target, predictions, labels=["yes", "no"])
        figure = make_confusion_chart(matrix)
        image = BytesIO()
        figure.savefig(image, format="png", dpi=100, bbox_inches="tight")
        plt.close(figure)
        st.image(image.getvalue(), use_column_width=True)
        st.markdown(
            '<div class="screen-reader-only">Confusion matrix. '
            "Rows are actual classes; columns are predicted classes. "
            f"True positive: {matrix[0, 0]:,}; false negative: {matrix[0, 1]:,}; "
            f"false positive: {matrix[1, 0]:,}; true negative: {matrix[1, 1]:,}. "
            "The positive class is yes.</div>",
            unsafe_allow_html=True,
        )
    with report_column:
        st.subheader("Classification report", anchor="classification-report")
        report = classification_report(
            target, predictions, labels=model.classes_, output_dict=True, zero_division=0
        )
        report_table = pd.DataFrame(
            {name: values for name, values in report.items() if name != "accuracy"}
        ).transpose().round(4)
        report_table.index.name = "Class / average"
        report_table["support"] = report_table["support"].astype(int)
        st.dataframe(
            report_table,
            use_container_width=True,
            column_config={
                "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
                "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
                "f1-score": st.column_config.NumberColumn("F1-score", format="%.4f"),
                "support": st.column_config.NumberColumn("Support", format="%d"),
            },
        )
        st.caption(
            "Support is the number of actual examples in each class. "
            "Overall accuracy is shown in the performance summary."
        )

    with st.expander("Data preview"):
        st.caption(f"Showing the first {min(25, len(frame)):,} of {len(frame):,} rows.")
        st.dataframe(frame.head(25), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
