"""Unit tests for src.features, focused on leakage-safety guarantees."""

from __future__ import annotations

import pandas as pd

from src.features import add_historical_store_aggregates, clean_feature_names


def _sample_train_open() -> pd.DataFrame:
    """Build a tiny synthetic open-store training frame for aggregate tests."""
    return pd.DataFrame(
        {
            "Store": [1, 1, 2, 2],
            "DayOfWeek": [1, 2, 1, 2],
            "Sales": [100.0, 200.0, 300.0, 400.0],
        }
    )


def test_historical_aggregates_do_not_use_future_rows() -> None:
    """Aggregates must be computed only from the supplied train_open frame."""
    train_open = _sample_train_open()
    future_frame = pd.DataFrame({"Store": [1], "DayOfWeek": [1]})

    (result,) = add_historical_store_aggregates(train_open, future_frame)

    assert result.loc[0, "store_mean_sales"] == 150.0
    assert result.loc[0, "store_dow_mean_sales"] == 100.0


def test_historical_aggregates_fill_unknown_store_with_global_mean() -> None:
    """Stores absent from the training aggregate fall back to the global mean.

    This prevents NaNs from appearing for unseen stores.
    """
    train_open = _sample_train_open()
    unseen_store_frame = pd.DataFrame({"Store": [999], "DayOfWeek": [1]})

    (result,) = add_historical_store_aggregates(train_open, unseen_store_frame)

    expected_global_mean = train_open["Sales"].mean()
    assert result.loc[0, "store_mean_sales"] == expected_global_mean
    assert not result["store_mean_sales"].isna().any()


def test_clean_feature_names_strips_special_characters() -> None:
    """LightGBM-incompatible characters must be removed from column names."""
    df = pd.DataFrame({"a b-c": [1], "d/e": [2]})

    cleaned = clean_feature_names(df)

    assert list(cleaned.columns) == ["abc", "de"]


def test_no_sales_lag_or_rolling_columns_are_produced() -> None:
    """Regression guard against reintroducing leaky lag/rolling Sales features."""
    train_open = _sample_train_open()
    future_frame = pd.DataFrame({"Store": [1], "DayOfWeek": [1]})

    (result,) = add_historical_store_aggregates(train_open, future_frame)

    leak_prefixes = ("lag_", "rolling_mean_", "rolling_std_", "sales_minus_rolling_mean_")
    assert not any(col.startswith(p) for col in result.columns for p in leak_prefixes)
