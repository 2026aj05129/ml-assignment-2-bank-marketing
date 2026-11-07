"""Train the five assignment classifiers and evaluate them on one holdout set."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import TARGET, load_dataset
from src.modeling import build_models, evaluate_model

PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "bank-full.csv"
MODEL_DIR = PROJECT_DIR / "model"
RESULTS_DIR = PROJECT_DIR / "results"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"


def main() -> None:
    features, target = load_dataset(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    models = build_models(x_train)
    rows = []

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        rows.append({"Model": name, **metrics})
        filename = name.lower().replace(" ", "_").replace("-", "")
        joblib.dump(model, MODEL_DIR / f"{filename}.joblib")
        print(
            f"  Accuracy={metrics['Accuracy']:.3f}, "
            f"AUC={metrics['AUC']:.3f}, F1={metrics['F1']:.3f}, "
            f"MCC={metrics['MCC']:.3f}"
        )

    results = pd.DataFrame(rows).set_index("Model")
    results.to_csv(RESULTS_DIR / "metrics.csv", float_format="%.6f")
    pd.concat([x_test, y_test.rename(TARGET)], axis=1).to_csv(
        TEST_DATA_PATH, index=False
    )
    print(f"\nMetrics saved to {RESULTS_DIR / 'metrics.csv'}")
    print(f"Labeled test data saved to {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
