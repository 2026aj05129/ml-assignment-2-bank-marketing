"""Preprocessing, classifier definitions, and held-out metric calculation."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


MODEL_NAMES = (
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbors",
    "Gaussian Naive Bayes",
    "Support Vector Machine",
)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing fitted only on training data when used in a pipeline."""
    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0,
    )


def build_models(features: pd.DataFrame) -> dict[str, Pipeline]:
    """Return all required classifiers with a shared preprocessing design."""
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=min(11, len(features)), n_jobs=-1
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="linear"),
    }
    return {
        name: Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(features)),
                ("classifier", estimator),
            ]
        )
        for name, estimator in estimators.items()
    }


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    """Calculate the six assignment metrics for binary or multiclass labels."""
    is_binary = len(classes) == 2
    average = "binary" if is_binary else "macro"
    positive_label: Any = classes[1] if is_binary else None
    metrics_args: dict[str, Any] = {"average": average, "zero_division": 0}
    if is_binary:
        metrics_args["pos_label"] = positive_label

    if is_binary:
        auc = roc_auc_score(np.asarray(y_true) == positive_label, y_score)
    else:
        auc = roc_auc_score(
            y_true,
            y_score,
            labels=classes,
            multi_class="ovr",
            average="macro",
        )

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(auc),
        "Precision": float(precision_score(y_true, y_pred, **metrics_args)),
        "Recall": float(recall_score(y_true, y_pred, **metrics_args)),
        "F1": float(f1_score(y_true, y_pred, **metrics_args)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def get_auc_scores(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return decision scores or class probabilities for ROC AUC."""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)
        if scores.shape[1] == 2:
            return scores[:, 1]
        return scores

    scores = model.decision_function(features)
    if np.ndim(scores) == 1:
        return scores
    return np.asarray(scores)


def evaluate_model(
    model: Pipeline, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    """Predict and calculate assignment metrics using a fitted pipeline."""
    predictions = model.predict(features)
    scores = get_auc_scores(model, features)
    return calculate_metrics(target, predictions, scores, model.classes_)
