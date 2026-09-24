"""
Train a baseline (DummyRegressor) and a candidate model (RandomForestRegressor)
for the Auto MPG regression task, compare them with a quality gate, and save
the trained model + a metrics report.

Exits with a non-zero status code if:
  - the dataset fails validation (see src/data.py), or
  - the candidate model does not beat the baseline by the required margin.

--- FAILURE DEMO A (model qity gate) ---
To deliberately weaken the model for the "failed quality gate" demonstration,
edit ONE of the constants in the "MODEL CONFIG" block below, e.g.:
    N_ESTIMATORS = 200
    MAX_DEPTH = None
or:
    TRAIN_SUBSET_FRACTION = 0.05
Commit and push. The gate will fail and no package will be published.
Afterwards, restore the original values before demonstrating Failure B.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_and_validate,
)

# ---------------------------------------------------------------------------
# Reproducibility / evaluation configuration - keep unchanged across the
# failure/recovery demonstrations, per assignment instructions.
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
METRIC_NAME = "MAE"  # Mean Absolute Error, in mpg. Lower is better.
MARGIN = 1.0  # mpg. Candidate must beat baseline MAE by at least this much.

# ---------------------------------------------------------------------------
# MODEL CONFIG - this is the block to edit for Failure Demo A.
# ---------------------------------------------------------------------------
N_ESTIMATORS = 200
MAX_DEPTH = None
TRAIN_SUBSET_FRACTION = 1.0  # 1.0 = use all training data

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("regressor", model)])


def main() -> None:
    df = load_and_validate()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    if TRAIN_SUBSET_FRACTION < 1.0:
        X_train = X_train.sample(frac=TRAIN_SUBSET_FRACTION, random_state=RANDOM_STATE)
        y_train = y_train.loc[X_train.index]
        print(
            f"NOTE: training on a {TRAIN_SUBSET_FRACTION:.0%} subset "
            f"({len(X_train)} rows) of the training data."
        )

    # --- Baseline ---
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_val)
    baseline_mae = mean_absolute_error(y_val, baseline_pred)

    # --- Candidate ---
    model = build_pipeline()
    model.fit(X_train, y_train)
    candidate_pred = model.predict(X_val)
    candidate_mae = mean_absolute_error(y_val, candidate_pred)

    # --- Quality gate: error metric, lower is better ---
    # require: candidate_mae <= baseline_mae - MARGIN
    required_max_mae = baseline_mae - MARGIN
    gate_passed = bool(candidate_mae <= required_max_mae)

    print(f"Baseline {METRIC_NAME}: {baseline_mae:.4f} mpg")
    print(f"Candidate {METRIC_NAME}: {candidate_mae:.4f} mpg")
    print(f"Required margin: {MARGIN:.4f} mpg (candidate must be <= {required_max_mae:.4f})")
    print(f"Quality gate: {'PASSED' if gate_passed else 'FAILED'}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    metrics_report = {
        "metric": METRIC_NAME,
        "direction": "lower_is_better",
        "baseline_score": round(float(baseline_mae), 4),
        "model_score": round(float(candidate_mae), 4),
        "margin": MARGIN,
        "required_threshold": round(float(required_max_mae), 4),
        "gate_passed": gate_passed,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "train_subset_fraction": TRAIN_SUBSET_FRACTION,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"Wrote metrics report to {METRICS_PATH}")

    # Always save the trained model artifact for inspection, but only a
    # process that exits 0 lets the workflow proceed to testing/packaging.
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    if not gate_passed:
        print("FATAL: candidate model failed to meet the quality gate.")
        sys.exit(1)


if __name__ == "__main__":
    main()
