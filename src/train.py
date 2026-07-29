"""Training entry point for the Rossmann Store Sales forecasting pipeline."""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_IMPORTANCE_FILE,
    MODEL_FILE,
    STORE_PATH,
    TRAIN_PATH,
    VALIDATION_SPLIT_QUANTILE,
)
from src.features import (
    add_historical_store_aggregates,
    build_features,
    clean_feature_names,
)
from src.models import (
    feature_importance_table,
    predict_ensemble,
    rmspe,
    train_lgbm_ensemble,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw training sales and store metadata from disk."""
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False, parse_dates=["Date"])
    store_df = pd.read_csv(STORE_PATH, low_memory=False)
    return train_df, store_df


def time_based_split(
    df: pd.DataFrame, quantile: float = VALIDATION_SPLIT_QUANTILE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train/validation partitions by date."""
    df = df.sort_values("Date").reset_index(drop=True)
    split_date = df["Date"].quantile(quantile)

    train_part = df[df["Date"] <= split_date].copy()
    valid_part = df[df["Date"] > split_date].copy()

    train_part = train_part[(train_part["Open"] == 1) & (train_part["Sales"] > 0)].copy()
    return train_part, valid_part


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of model feature columns."""
    return [c for c in df.columns if c not in ("Sales", "Date")]


def run_training_pipeline() -> None:
    """Execute the end-to-end training pipeline and persist artefacts."""
    logger.info("Loading raw data")
    train_raw, store_raw = load_raw_data()

    logger.info("Building leakage-free features")
    engineered = build_features(train_raw, store_raw)

    logger.info("Splitting data by date")
    train_part, valid_part = time_based_split(engineered)

    logger.info("Attaching historical store aggregates")
    train_open_for_agg = train_part
    train_part, valid_part = add_historical_store_aggregates(
        train_open_for_agg, train_part, valid_part
    )

    feature_columns = get_feature_columns(train_part)
    x_train = clean_feature_names(train_part[feature_columns])
    x_valid = clean_feature_names(valid_part[feature_columns])

    y_train_log = np.log1p(train_part["Sales"])
    valid_open_mask = (valid_part["Open"] == 1).to_numpy()
    y_valid_log = np.log1p(valid_part.loc[valid_open_mask, "Sales"])
    x_valid_open = x_valid.loc[valid_open_mask]

    logger.info("Training LightGBM ensemble")
    models = train_lgbm_ensemble(x_train, y_train_log, x_valid_open, y_valid_log)

    preds_valid = np.zeros(len(x_valid))
    preds_valid[valid_open_mask] = predict_ensemble(models, x_valid_open)
    score = rmspe(valid_part["Sales"].to_numpy(), preds_valid)
    logger.info("Validation RMSPE: %.6f", score)

    importance_df = feature_importance_table(models, list(x_train.columns))
    importance_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
    logger.info("Saved feature importance to %s", FEATURE_IMPORTANCE_FILE)

    joblib.dump(
        {"models": models, "feature_columns": feature_columns},
        MODEL_FILE,
    )
    logger.info("Saved model ensemble to %s", MODEL_FILE)


if __name__ == "__main__":
    run_training_pipeline()
