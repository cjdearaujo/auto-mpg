"""
Automated tests for the Auto MPG pipeline.

These tests assume that src/train.py has already been run in the same job
(so models/model.joblib exists) - this is why training and testing must run
in the same GitHub Actions job.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import MODEL_PATH, load_model, predict_from_dataframe  # noqa: E402
from data import FEATURE_COLUMNS  # noqa: E402

VALID_SAMPLE = {
    "cylinders": 4,
    "displacement": 140.0,
    "horsepower": 90.0,
    "weight": 2264,
    "acceleration": 15.5,
    "model_year": 71,
    "origin": "usa",
}


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), (
        f"Expected trained model at {MODEL_PATH}. "
        "Did the training step run before the tests?"
    )


def test_saved_model_can_be_loaded():
    model = load_model()
    assert model is not None
    # A fitted sklearn Pipeline exposes .predict
    assert hasattr(model, "predict")


def test_valid_sample_produces_expected_shape_and_type():
    df = pd.DataFrame([VALID_SAMPLE])
    preds = predict_from_dataframe(df)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (1,)
    assert np.issubdtype(preds.dtype, np.floating)
    # Sanity: predicted mpg should be a plausible positive number.
    assert 0 < preds[0] < 100


def test_multiple_valid_samples_produce_matching_row_count():
    df = pd.DataFrame([VALID_SAMPLE, VALID_SAMPLE, VALID_SAMPLE])
    preds = predict_from_dataframe(df)
    assert preds.shape == (3,)


def test_missing_required_feature_is_rejected():
    incomplete = {k: v for k, v in VALID_SAMPLE.items() if k != "horsepower"}
    df = pd.DataFrame([incomplete])
    with pytest.raises(ValueError, match="missing required feature"):
        predict_from_dataframe(df)


def test_missing_feature_error_names_the_column():
    incomplete = {k: v for k, v in VALID_SAMPLE.items() if k != "weight"}
    df = pd.DataFrame([incomplete])
    with pytest.raises(ValueError, match="weight"):
        predict_from_dataframe(df)
