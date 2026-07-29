"""Model definitions and evaluation metrics for the Rossmann pipeline.

The final model is a bagging ensemble of LightGBM regressors trained on a
log1p-transformed target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping

from src.config import EARLY_STOPPING_ROUNDS, ENSEMBLE_SEEDS, LGBM_PARAMS


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Root Mean Squared Percentage Error.

    Rows where ``y_true`` is zero are excluded.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    y_true, y_pred = y_true[mask], y_pred[mask]
    return float(np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))


def rmspe_lgb_eval(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
    """LightGBM-compatible RMSPE evaluation metric on log1p-scale inputs."""
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.maximum(np.expm1(y_pred), 0)
    score = rmspe(y_true_orig, y_pred_orig)
    return "rmspe", score, False


def train_lgbm_ensemble(
    x_train: pd.DataFrame,
    y_train_log: pd.Series,
    x_valid: pd.DataFrame,
    y_valid_log: pd.Series,
    params: dict | None = None,
    seeds: tuple[int, ...] = ENSEMBLE_SEEDS,
) -> list[LGBMRegressor]:
    """Train a bagging ensemble of LightGBM regressors."""
    base_params = dict(params or LGBM_PARAMS)
    models: list[LGBMRegressor] = []
    for seed in seeds:
        seed_params = dict(base_params, random_state=seed)
        model = LGBMRegressor(**seed_params)
        model.fit(
            x_train,
            y_train_log,
            eval_set=[(x_valid, y_valid_log)],
            eval_metric=rmspe_lgb_eval,
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS)],
        )
        models.append(model)
    return models


def predict_ensemble(models: list[LGBMRegressor], x: pd.DataFrame) -> np.ndarray:
    """Average predictions from an ensemble of fitted models."""
    preds = np.zeros(len(x), dtype=float)
    for model in models:
        preds += np.expm1(model.predict(x))
    preds /= len(models)
    return np.maximum(preds, 0.0)


def feature_importance_table(
    models: list[LGBMRegressor], feature_names: list[str]
) -> pd.DataFrame:
    """Build an averaged feature importance table across ensemble members."""
    avg_importance = np.mean(
        [model.feature_importances_ for model in models], axis=0
    )
    table = pd.DataFrame({"feature": feature_names, "importance": avg_importance})
    table = table.sort_values("importance", ascending=False).reset_index(drop=True)
    table["importance_pct"] = table["importance"] / table["importance"].sum() * 100
    return table
