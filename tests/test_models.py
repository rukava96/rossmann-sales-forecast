"""Unit tests for src.models, primarily the RMSPE metric implementation."""

from __future__ import annotations

import numpy as np

from src.models import rmspe, rmspe_lgb_eval


def test_rmspe_perfect_prediction_is_zero() -> None:
    """RMSPE must be exactly zero when predictions match the target."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])

    assert rmspe(y_true, y_pred) == 0.0


def test_rmspe_ignores_zero_sales_rows() -> None:
    """Rows with zero true sales must be excluded from the RMSPE computation."""
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([9999.0, 100.0])

    assert rmspe(y_true, y_pred) == 0.0


def test_rmspe_known_relative_error() -> None:
    """A 10% relative error on every row should yield RMSPE close to 0.1."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = y_true * 1.1

    assert abs(rmspe(y_true, y_pred) - 0.1) < 1e-9


def test_rmspe_lgb_eval_returns_expected_tuple_shape() -> None:
    """The LightGBM eval callback must return (name, value, is_higher_better)."""
    y_true_log = np.log1p(np.array([100.0, 200.0]))
    y_pred_log = np.log1p(np.array([110.0, 190.0]))

    name, value, is_higher_better = rmspe_lgb_eval(y_true_log, y_pred_log)

    assert name == "rmspe"
    assert value >= 0.0
    assert is_higher_better is False
