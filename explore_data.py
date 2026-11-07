"""Print a small, reproducible summary of the downloaded assignment dataset."""

from pathlib import Path

from src.data import load_dataset

DATA_PATH = Path(__file__).parent / "data" / "bank-full.csv"


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download and extract UCI Bank "
            "Marketing, then place bank-full.csv in the data/ directory."
        )

    features, target = load_dataset(DATA_PATH)
    print(f"Rows: {len(features):,}")
    print(f"Predictor columns: {features.shape[1]}")
    print(f"Target: {target.name}")
    print("\nTarget counts:")
    print(target.value_counts(dropna=False).to_string())
    print("\nTarget proportions:")
    print(target.value_counts(normalize=True, dropna=False).round(3).to_string())
    print("\nPredictor data types:")
    print(features.dtypes.value_counts().to_string())
    print("\nMissing values by predictor:")
    print(features.isna().sum().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
