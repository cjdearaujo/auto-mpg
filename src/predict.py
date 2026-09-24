"""
Prediction application for the Auto MPG model.

Usage as a library:
    from predict import predict_from_dataframe
    preds = predict_from_dataframe(df)

Usage as a script:
    python src/predict.py path/to/input.csv
Prints one predicted mpg value per input row.
"""

import sys

import joblib
import numpy as np
import pandas as pd

from data import FEATURE_COLUMNS

MODEL_PATH = "models/model.joblib"


def load_model(model_path: str = MODEL_PATH):
    return joblib.load(model_path)


def predict_from_dataframe(df: pd.DataFrame, model=None) -> np.ndarray:
    """
    Predict mpg for each row of `df`.

    Raises a ValueError with a clear message if any required feature column
    is missing from `df`. This is the check exercised by the "missing
    feature is rejected" automated test.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required feature column(s): {missing}. "
            f"Required columns are: {FEATURE_COLUMNS}"
        )

    if model is None:
        model = load_model()

    predictions = model.predict(df[FEATURE_COLUMNS])
    return np.asarray(predictions, dtype=float)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/input.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    df = pd.read_csv(input_path)
    preds = predict_from_dataframe(df)
    for i, p in enumerate(preds):
        print(f"row {i}: predicted mpg = {p:.2f}")


if __name__ == "__main__":
    main()
