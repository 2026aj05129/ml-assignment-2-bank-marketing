"""Evaluate saved Bank Marketing classifiers on a labeled CSV."""

import logging
import pickle
from io import BytesIO
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
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


def main() -> None:
    st.set_page_config(
        page_title="Term Deposit Insights", page_icon="📊",
        layout="wide", initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 4rem; padding-bottom: 2rem;}
        [data-testid="stMetric"] {
            background: #fffaf1;
            border: 1px solid #ead8bc;
            border-radius: 6px;
            padding: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.header("Campaign workspace")
        st.caption("Evaluate saved models on labeled test data.")
        model_name = st.selectbox("Classification method", list(MODEL_FILES))
        uploaded = st.file_uploader("Upload campaign CSV", type="csv")
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
        st.caption("Saved models are evaluated without retraining.")

    st.title("Term Deposit Insights")
    st.caption("UCI Bank Marketing | Model evaluation dashboard")
    if uploaded is None:
        st.subheader("Campaign dataset")
        for column, label, value in zip(
            st.columns(4),
            ("Dataset rows", "Model inputs", "Classifiers", "Holdout rows"),
            ("45,211", "15", "5", "9,043"),
            strict=True,
        ):
            column.metric(label, value)
        st.info("Upload a labeled CSV using the left panel to evaluate a model.")
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

    st.write(
        f"**{model_name}** | {len(features):,} rows | "
        f"{int((target == 'no').sum()):,} no | {int((target == 'yes').sum()):,} yes"
    )
    st.subheader("Performance summary")
    for column, (name, value) in zip(st.columns(6), metrics.items(), strict=True):
        column.metric(name, f"{value:.4f}", help=METRIC_HELP[name])
    st.caption("Positive class: yes. Metrics use the uploaded labels and saved pipeline.")

    matrix_column, report_column = st.columns(2, gap="large")
    with matrix_column:
        st.subheader("Confusion matrix", anchor="confusion-matrix")
        matrix = confusion_matrix(target, predictions, labels=model.classes_)
        st.caption("Actual labels run down the rows; predictions run across the columns.")
        figure, axis = plt.subplots(figsize=(4.2, 2.8))
        figure.patch.set_facecolor("#fffcf6")
        axis.set_facecolor("#fffcf6")
        descriptions = (
            ("Correct no", "False positive"),
            ("False negative", "Correct yes"),
        )
        colors = (("#e6f0e2", "#fbe8d5"), ("#fbe8d5", "#e6f0e2"))
        for row in range(2):
            for col in range(2):
                axis.add_patch(
                    plt.Rectangle((col, row), 1, 1, facecolor=colors[row][col],
                                  edgecolor="white", linewidth=4)
                )
                axis.text(col + 0.5, row + 0.39, f"{matrix[row, col]:,}",
                          ha="center", va="center", fontsize=18, fontweight="bold",
                          color="#332316")
                axis.text(col + 0.5, row + 0.68, descriptions[row][col],
                          ha="center", va="center", fontsize=10, color="#574332")
        axis.set(xlim=(0, 2), ylim=(2, 0), aspect="equal")
        axis.set_xticks([0.5, 1.5], ["No", "Yes"])
        axis.set_yticks([0.5, 1.5], ["No", "Yes"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        figure.tight_layout()
        image = BytesIO()
        figure.savefig(image, format="png", dpi=100, bbox_inches="tight")
        plt.close(figure)
        st.image(image.getvalue(), width=270, caption="Confusion matrix: actual vs predicted")
        st.caption(
            f"Correct no: {matrix[0, 0]:,}; false positive: {matrix[0, 1]:,}; "
            f"false negative: {matrix[1, 0]:,}; correct yes: {matrix[1, 1]:,}."
        )
    with report_column:
        st.subheader("Classification report", anchor="classification-report")
        st.caption("Precision, recall, F1 and support for each class and overall.")
        report = classification_report(
            target, predictions, labels=model.classes_, output_dict=True, zero_division=0
        )
        st.dataframe(
            pd.DataFrame(report).transpose().round(4),
            use_container_width=True,
            height=230,
        )
        st.caption("Support is the number of actual examples in each class.")

    with st.expander("Data preview"):
        st.caption(f"Showing the first {min(25, len(frame)):,} of {len(frame):,} rows.")
        st.dataframe(frame.head(25), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
