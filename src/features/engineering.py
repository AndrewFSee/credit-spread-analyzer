"""
Feature engineering for the Credit Spread Analysis & Prediction Platform.

Transforms raw market-data DataFrames into model-ready feature matrices by
adding lags, rolling statistics, momentum signals, yield-curve shape features,
cross-asset ratios, z-scores, and forward-return targets.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_lagged_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Add lagged versions of selected columns.

    Parameters
    ----------
    df:
        Input DataFrame (index must be a DatetimeIndex).
    columns:
        Column names to lag.
    lags:
        List of integer lag periods.

    Returns
    -------
    pd.DataFrame
        DataFrame with new ``{col}_lag{n}`` columns appended.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            logger.warning("Column %s not found – skipping lags.", col)
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_stats(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """Add rolling mean, standard deviation, minimum, and maximum.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to compute rolling statistics on.
    windows:
        Rolling window sizes in days.

    Returns
    -------
    pd.DataFrame
        DataFrame with new rolling-stat columns appended.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            logger.warning("Column %s not found – skipping rolling stats.", col)
            continue
        for w in windows:
            r = df[col].rolling(window=w, min_periods=w)
            df[f"{col}_rmean{w}"] = r.mean()
            df[f"{col}_rstd{w}"] = r.std()
            df[f"{col}_rmin{w}"] = r.min()
            df[f"{col}_rmax{w}"] = r.max()
    return df


def add_momentum_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """Add rate-of-change (momentum) features.

    ``momentum = (x_t - x_{t-w}) / x_{t-w}``

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to compute momentum on.
    windows:
        Look-back periods.

    Returns
    -------
    pd.DataFrame
        DataFrame with new ``{col}_mom{w}`` columns appended.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            logger.warning("Column %s not found – skipping momentum.", col)
            continue
        for w in windows:
            past = df[col].shift(w)
            df[f"{col}_mom{w}"] = (df[col] - past) / past.replace(0, np.nan)
    return df


def add_yield_curve_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive yield-curve slope and curvature features.

    Expects columns ``t10y2y`` (10y–2y spread), ``fed_funds``, and optionally
    ``t10y3m`` (10y–3m spread) in *df*.  Missing inputs are handled gracefully.

    Parameters
    ----------
    df:
        Input DataFrame containing Treasury spread columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with yield-curve feature columns appended.
    """
    df = df.copy()

    # Slope: 10y-2y is already a spread; also compute changes
    if "t10y2y" in df.columns:
        df["yc_slope"] = df["t10y2y"]
        df["yc_slope_chg1"] = df["t10y2y"].diff(1)
        df["yc_slope_chg5"] = df["t10y2y"].diff(5)
        df["yc_slope_chg20"] = df["t10y2y"].diff(20)

    # Curvature: BBB – HY spread (rough proxy)
    if "bbb_spread" in df.columns and "hy_spread" in df.columns:
        df["yc_curvature"] = df["hy_spread"] - df["bbb_spread"]

    # Level: Fed Funds rate as proxy for the short end
    if "fed_funds" in df.columns:
        df["rate_level"] = df["fed_funds"]
        df["rate_level_chg20"] = df["fed_funds"].diff(20)

    return df


def add_cross_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-asset ratio features.

    Parameters
    ----------
    df:
        Input DataFrame containing spread and price columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with ratio columns appended.
    """
    df = df.copy()

    # HY / IG spread ratio
    if "hy_spread" in df.columns and "ig_spread" in df.columns:
        df["hy_ig_ratio"] = df["hy_spread"] / df["ig_spread"].replace(0, np.nan)

    # BBB / IG ratio
    if "bbb_spread" in df.columns and "ig_spread" in df.columns:
        df["bbb_ig_ratio"] = df["bbb_spread"] / df["ig_spread"].replace(0, np.nan)

    # HY spread per unit of VIX (stress-adjusted)
    if "hy_spread" in df.columns and "vix" in df.columns:
        df["hy_per_vix"] = df["hy_spread"] / df["vix"].replace(0, np.nan)

    # Gold / crude ratio (risk-off proxy)
    if "gold" in df.columns and "crude_oil" in df.columns:
        df["gold_crude_ratio"] = df["gold"] / df["crude_oil"].replace(0, np.nan)

    return df


def add_zscore_features(
    df: pd.DataFrame,
    columns: list[str],
    window: int = 60,
) -> pd.DataFrame:
    """Add rolling z-score features.

    ``z = (x - rolling_mean) / rolling_std``

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to z-score.
    window:
        Rolling window length.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``{col}_z{window}`` columns appended.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            logger.warning("Column %s not found – skipping z-score.", col)
            continue
        r = df[col].rolling(window=window, min_periods=window)
        mu = r.mean()
        sigma = r.std()
        df[f"{col}_z{window}"] = (df[col] - mu) / sigma.replace(0, np.nan)
    return df


def create_targets(
    df: pd.DataFrame,
    target_col: str,
    horizons: list[int],
) -> pd.DataFrame:
    """Create forward-return and direction targets.

    For each horizon *h*, adds:

    * ``target_{h}d_return`` – forward log-return over *h* days.
    * ``target_{h}d_up`` – binary flag (1 if return > 0, else 0).

    Parameters
    ----------
    df:
        Input DataFrame containing *target_col*.
    target_col:
        Column to compute forward returns for (e.g. ``"hy_spread"``).
    horizons:
        List of forward horizons in days.

    Returns
    -------
    pd.DataFrame
        DataFrame with target columns appended (NaN at the tail).
    """
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in DataFrame.")
    for h in horizons:
        fwd_return = np.log(df[target_col].shift(-h) / df[target_col])
        df[f"target_{h}d_return"] = fwd_return
        df[f"target_{h}d_up"] = (fwd_return > 0).astype(int)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    target_horizon: int = 5,
    spread_cols: Optional[list[str]] = None,
    macro_cols: Optional[list[str]] = None,
    lags: Optional[list[int]] = None,
    windows: Optional[list[int]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a clean, NaN-free feature matrix and target DataFrame.

    Applies the full feature-engineering pipeline and then drops rows that
    contain any NaN values.

    Parameters
    ----------
    df:
        Raw merged data DataFrame.
    target_horizon:
        Forward horizon in days for target creation.
    spread_cols:
        Credit-spread columns to feature-engineer.  Defaults to
        ``["hy_spread", "ig_spread", "bbb_spread"]``.
    macro_cols:
        Macro / rate columns to feature-engineer.  Defaults to
        ``["t10y2y", "fed_funds", "vix"]``.
    lags:
        Lag periods.  Defaults to ``[1, 5, 10, 20]``.
    windows:
        Rolling window sizes.  Defaults to ``[5, 10, 20, 60]``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(X, y)`` where *X* contains feature columns and *y* contains the
        target columns aligned on the same (NaN-free) index.
    """
    if spread_cols is None:
        spread_cols = [c for c in ["hy_spread", "ig_spread", "bbb_spread"] if c in df.columns]
    if macro_cols is None:
        macro_cols = [c for c in ["t10y2y", "fed_funds", "vix", "dxy"] if c in df.columns]
    if lags is None:
        lags = [1, 5, 10, 20]
    if windows is None:
        windows = [5, 10, 20, 60]

    all_feature_cols = spread_cols + macro_cols
    feat_cols_in_df = [c for c in all_feature_cols if c in df.columns]

    df = add_lagged_features(df, feat_cols_in_df, lags)
    df = add_rolling_stats(df, feat_cols_in_df, windows)
    df = add_momentum_features(df, feat_cols_in_df, windows)
    df = add_yield_curve_features(df)
    df = add_cross_ratios(df)
    df = add_zscore_features(df, feat_cols_in_df, window=60)

    target_col = spread_cols[0] if spread_cols else feat_cols_in_df[0]
    df = create_targets(df, target_col, horizons=[target_horizon])

    target_cols = [c for c in df.columns if c.startswith("target_")]
    meta_cols = feat_cols_in_df  # keep raw columns too for reference

    # Drop rows with any NaN
    df_clean = df.dropna()

    feature_exclude = set(target_cols)
    X = df_clean[[c for c in df_clean.columns if c not in feature_exclude]]
    y = df_clean[target_cols]

    logger.info(
        "Feature matrix shape: X=%s  y=%s (dropped %d rows with NaN)",
        X.shape,
        y.shape,
        len(df) - len(df_clean),
    )
    return X, y
