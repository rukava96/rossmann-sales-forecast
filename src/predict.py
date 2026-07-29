"""Inference entry point for generating Kaggle submission predictions."""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_FILE, STORE_PATH, SUBMISSION_PATH, TEST_PATH, TRAIN_PATH
from src.features import (
    add_historical_store_aggregates,
    build_features,
    clean_feature_names,
)
from src.models import predict_ensemble

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model_bundle() -> dict:
    """Load the persisted ensemble and feature column list."""
    return joblib.load(MODEL_FILE)


def build_test_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, store_df: pd.DataFrame
) -> pd.DataFrame:
    """Engineer features for the test set using train-only aggregates."""
    train_engineered = build_features(train_df, store_df)
    test_engineered = build_features(test_df, store_df)

    train_open = train_engineered[
        (train_engineered["Open"] == 1) & (train_engineered["Sales"] > 0)
    ]
    (test_engineered,) = add_historical_store_aggregates(train_open, test_engineered)
    return test_engineered


def run_inference_pipeline() -> None:
    """Execute the end-to-end inference pipeline and write a submission file."""
    logger.info("Loading raw data")
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False, parse_dates=["Date"])
    test_df = pd.read_csv(TEST_PATH, low_memory=False, parse_dates=["Date"])
    store_df = pd.read_csv(STORE_PATH, low_memory=False)

    logger.info("Building test features")
    test_engineered = build_test_features(train_df, test_df, store_df)

    bundle = load_model_bundle()
    models = bundle["models"]
    feature_columns = bundle["feature_columns"]

    x_test = clean_feature_names(test_engineered[feature_columns])
    open_mask = (test_engineered["Open"] == 1).to_numpy()

    predictions = np.zeros(len(x_test))
    predictions[open_mask] = predict_ensemble(models, x_test.loc[open_mask])

    ids = (
        test_engineered["Id"].to_numpy()
        if "Id" in test_engineered.columns
        else np.arange(1, len(predictions) + 1)
    )
    submission = pd.DataFrame({"Id": ids, "Sales": predictions})
    submission.to_csv(SUBMISSION_PATH, index=False)
    logger.info("Saved submission to %s", SUBMISSION_PATH)


if __name__ == "__main__":
    run_inference_pipeline()
