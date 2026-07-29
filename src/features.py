"""Feature engineering utilities for the Rossmann Store Sales pipeline.

All transformations in this module are leakage-safe: no feature is derived
from the current-day target (``Sales``) or from any information that would
be unavailable at prediction time on the Kaggle test set. In particular,
this module intentionally avoids lag and rolling statistics computed on
``Sales`` -- earlier iterations of this project used ``rolling().mean()``
without an explicit ``shift(1)``, which included the current day in the
averaging window and leaked the target into the features. That leakage
produced a train/validation RMSPE of roughly 0.03 while the true Kaggle
score was 0.23-0.26. The pipeline below removes lag/rolling sales features
entirely and instead relies on historical, leakage-free aggregates
(per-store and per-store-per-weekday mean sales) computed only on the
training partition.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_COLUMNS, LEAKY_OR_USELESS_COLUMNS


def merge_store_info(sales_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """Merge sales records with static store metadata.

    Args:
        sales_df: Train or test dataframe containing a ``Store`` column.
        store_df: Store metadata dataframe (``store.csv``).

    Returns:
        A dataframe with store attributes joined onto ``sales_df``.
    """
    return sales_df.merge(store_df, on="Store", how="left")


def clean_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leaky or redundant columns and coerce basic dtypes.

    Args:
        df: Raw merged dataframe.

    Returns:
        A copy of ``df`` without leaky/unused columns.
    """
    df = df.copy()
    columns_to_drop = [c for c in LEAKY_OR_USELESS_COLUMNS if c in df.columns]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features derived purely from the ``Date`` column.

    Args:
        df: Dataframe containing a datetime ``Date`` column.

    Returns:
        A copy of ``df`` with additional calendar columns.
    """
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Quarter"] = df["Date"].dt.quarter
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)
    return df


def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features describing nearby competitor stores.

    Args:
        df: Dataframe with ``CompetitionOpenSinceYear`` /
            ``CompetitionOpenSinceMonth`` and a ``Date`` column.

    Returns:
        A copy of ``df`` with competition-related indicators.
    """
    df = df.copy()
    has_competition_since = (
        df["CompetitionOpenSinceYear"].notna()
        & df["CompetitionOpenSinceMonth"].notna()
    )
    df["is_competition_open_known"] = has_competition_since.astype(int)

    competition_open_date = pd.to_datetime(
        {
            "year": df["CompetitionOpenSinceYear"].fillna(df["Date"].dt.year),
            "month": df["CompetitionOpenSinceMonth"].fillna(df["Date"].dt.month),
            "day": 1,
        },
        errors="coerce",
    )
    months_since_open = (
        (df["Date"].dt.year - competition_open_date.dt.year) * 12
        + (df["Date"].dt.month - competition_open_date.dt.month)
    )
    df["months_since_competition_open"] = np.where(
        has_competition_since, months_since_open.clip(lower=0), 0.0
    )
    return df


def add_promo2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features describing the recurring ``Promo2`` campaign.

    Args:
        df: Dataframe with ``Promo2SinceYear`` / ``Promo2SinceWeek`` and a
            ``Date`` column.

    Returns:
        A copy of ``df`` with Promo2-related indicators.
    """
    df = df.copy()
    has_promo2_since = df["Promo2SinceYear"].notna() & df["Promo2SinceWeek"].notna()
    df["is_promo2_known"] = has_promo2_since.astype(int)

    promo2_start = pd.to_datetime(
        df["Promo2SinceYear"].fillna(df["Date"].dt.year).astype(int).astype(str)
        + "-W"
        + df["Promo2SinceWeek"].fillna(1).astype(int).astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )
    weeks_since_promo2 = (df["Date"] - promo2_start).dt.days / 7.0
    df["months_since_promo2"] = np.where(
        has_promo2_since, weeks_since_promo2.clip(lower=0) / 4.345, 0.0
    )
    return df


def add_promo_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between ``Promo`` and other categoricals.

    Args:
        df: Dataframe with ``Promo``, ``DayOfWeek``, ``StoreType`` and
            ``Assortment`` columns.

    Returns:
        A copy of ``df`` with promo interaction columns.
    """
    df = df.copy()
    df["PromoDayOfWeek"] = df["Promo"] * df["DayOfWeek"]
    df["PromoStoreType"] = df["Promo"].astype(str) + df["StoreType"].astype(str)
    df["PromoAssortment"] = df["Promo"].astype(str) + df["Assortment"].astype(str)
    return df


def add_holiday_neighbor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous/next day holiday indicators.

    Args:
        df: Dataframe sorted or sortable by ``Date``, containing
            ``StateHoliday`` and ``SchoolHoliday`` columns.

    Returns:
        A copy of ``df`` sorted by ``Date`` with holiday neighbor columns.
    """
    df = df.sort_values("Date").copy()

    is_state_holiday = (df["StateHoliday"] != "0").astype(int)
    df["is_state_holiday"] = is_state_holiday
    df["tomorrow_state_holiday"] = is_state_holiday.shift(-1).fillna(0).astype(int)
    df["yesterday_state_holiday"] = is_state_holiday.shift(1).fillna(0).astype(int)

    is_school_holiday = df["SchoolHoliday"].astype(int)
    df["is_school_holiday"] = is_school_holiday
    df["tomorrow_school_holiday"] = is_school_holiday.shift(-1).fillna(0).astype(int)
    df["yesterday_school_holiday"] = is_school_holiday.shift(1).fillna(0).astype(int)

    return df


def add_historical_store_aggregates(
    train_open_df: pd.DataFrame,
    *frames: pd.DataFrame,
) -> list[pd.DataFrame]:
    """Attach leakage-free historical mean-sales aggregates.

    Args:
        train_open_df: Training partition filtered to ``Open == 1`` and
            ``Sales > 0``, used to compute the aggregates.
        *frames: Dataframes (train/validation/test) that should receive the
            joined aggregate columns.

    Returns:
        Dataframes with ``store_mean_sales`` and ``store_dow_mean_sales``.
    """
    store_mean = (
        train_open_df.groupby("Store")["Sales"].mean().rename("store_mean_sales")
    )
    store_dow_mean = (
        train_open_df.groupby(["Store", "DayOfWeek"])["Sales"]
        .mean()
        .rename("store_dow_mean_sales")
    )
    global_mean = train_open_df["Sales"].mean()

    result = []
    for frame in frames:
        merged = frame.merge(store_mean, on="Store", how="left")
        merged = merged.merge(store_dow_mean, on=["Store", "DayOfWeek"], how="left")
        merged["store_mean_sales"] = merged["store_mean_sales"].fillna(global_mean)
        merged["store_dow_mean_sales"] = merged["store_dow_mean_sales"].fillna(
            global_mean
        )
        result.append(merged)
    return result


def encode_categoricals(
    df: pd.DataFrame, categorical_columns: tuple[str, ...] = CATEGORICAL_COLUMNS
) -> pd.DataFrame:
    """Cast known categorical columns to the pandas ``category`` dtype.

    Args:
        df: Dataframe to encode.
        categorical_columns: Column names to convert, if present.

    Returns:
        A copy of ``df`` with categorical dtypes applied.
    """
    df = df.copy()
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitise column names for gradient boosting libraries.

    Args:
        df: Dataframe whose columns should be sanitised.

    Returns:
        A copy of ``df`` with cleaned column names.
    """
    df = df.copy()
    df.columns = [re.sub(r"[^0-9a-zA-Z_]", "", str(col)) for col in df.columns]
    return df


def build_features(sales_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """Run the full leakage-free feature engineering pipeline.

    Args:
        sales_df: Raw train or test dataframe.
        store_df: Store metadata dataframe.

    Returns:
        A fully engineered dataframe (без sales-лагов/роллингов).
    """
    df = merge_store_info(sales_df, store_df)
    df = clean_basic_columns(df)
    df = add_time_features(df)
    df = add_competition_features(df)
    df = add_promo2_features(df)
    df = add_promo_interactions(df)
    df = add_holiday_neighbor_features(df)
    df = encode_categoricals(df)
    return df
