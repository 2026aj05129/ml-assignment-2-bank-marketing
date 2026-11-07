"""Tests for assignment metric calculations and classifier setup."""

import numpy as np
import pandas as pd

from src.modeling import MODEL_NAMES, build_models, calculate_metrics


def test_build_models_returns_five_required_pipelines():
    features = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4],
            "category": ["a", "b", "a", "b"],
        }
    )

    models = build_models(features)

    assert tuple(models) == MODEL_NAMES
    assert all("preprocessor" in model.named_steps for model in models.values())


def test_calculate_binary_metrics_uses_second_class_as_positive():
    actual = np.array(["no", "no", "yes", "yes"])
    predicted = np.array(["no", "yes", "yes", "yes"])
    scores = np.array([0.1, 0.6, 0.8, 0.9])

    result = calculate_metrics(
        actual, predicted, scores, np.array(["no", "yes"])
    )

    assert result["Accuracy"] == 0.75
    assert result["Precision"] == 2 / 3
    assert result["Recall"] == 1.0
    assert result["F1"] == 0.8
    assert result["AUC"] == 1.0
    assert -1.0 <= result["MCC"] <= 1.0
