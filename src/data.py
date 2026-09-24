"""
Data loading and validation for the Auto MPG regression pipeline.

Dataset: Auto MPG (city-cycle fuel consumption for 398 cars, 1970-1982).
Source: originally from Quinlan, R. (1993), Auto MPG, UCI Machine Learning
Repository (https://archive.ics.uci.edu/dataset/9/auto+mpg). This project
uses the commonly-used "seaborn-data" mirror of that dataset.

The dataset is included directly in this repository (data/mpg.csv, ~21 KB,
well under the 5 MB limit) rather than downloaded at runtime, so the
pipeline has no dependency on an external site being reachable.

Task: regression. Predict "mpg" (miles per gallon, continuous) from engine
and vehicle specifications.
"""

import os
import sys
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mpg.csv")

TARGET_COLUMN = "mpg"

NUMERIC_FEATURES = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
]

CATEGORICAL_FEATURES = ["origin"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# "name" (car name) is dropped: it is a near-unique identifier, not a
# predictive feature, and is intentionally excluded from REQUIRED_COLUMNS.
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the dataset from the local CSV file included in this repository."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: could not load dataset from {path}: {exc}")
        sys.exit(1)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    Fail loudly (non-zero exit) if any expected feature column or the target
    column is missing, or if the target itself has missing values.

    Note: 'horsepower' is allowed to contain a small number of missing
    values -- that is handled by imputation in the training pipeline, not
    treated as a validation failure. Only *structural* problems (missing
    columns, missing target values, empty dataset) are hard failures here.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"FATAL: dataset is missing required column(s): {missing_cols}")
        sys.exit(1)

    if len(df) == 0:
        print("FATAL: dataset has zero rows after download.")
        sys.exit(1)

    if df[TARGET_COLUMN].isnull().any():
        print(f"FATAL: target column '{TARGET_COLUMN}' contains null values.")
        sys.exit(1)


def load_and_validate(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the dataset and validate it. Exits non-zero on any failure."""
    df = load_raw_data(path)
    validate_columns(df)
    print(f"Data OK: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


if __name__ == "__main__":
    load_and_validate()
