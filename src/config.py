"""Project-wide configuration and path constants.

This module centralises filesystem paths, model artefact locations, and
high-level constants shared across the feature engineering, training and
inference pipelines.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
STORE_PATH = DATA_DIR / "store.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
SUBMISSION_PATH = DATA_DIR / "submission.csv"

MODEL_FILE = MODEL_DIR / "lgbm_ensemble.joblib"
FEATURE_IMPORTANCE_FILE = DATA_DIR / "feature_importance.csv"

MODEL_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)

FORECAST_HORIZON_WEEKS = 6
METRIC_NAME = "RMSPE"
VALIDATION_SPLIT_QUANTILE = 0.8

ENSEMBLE_SEEDS = (42, 123, 456, 789, 1010)

LGBM_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "verbosity": -1,
}

EARLY_STOPPING_ROUNDS = 50

CATEGORICAL_COLUMNS = (
    "StoreType",
    "Assortment",
    "StateHoliday",
    "PromoStoreType",
    "PromoAssortment",
)

LEAKY_OR_USELESS_COLUMNS = ("Unnamed 0", "Customers", "PromoInterval")
