"""
Feature engineering for the Credit Spread Analysis & Prediction Platform.

Transforms raw market-data DataFrames into model-ready feature matrices.

The default feature set (:func:`build_signal_features`) is deliberately small
and stationary – changes, z-scores, and volatilities rather than raw levels –
because walk-forward testing showed that level-based features (lags and
rolling min/max of spread levels) let tree models fit regime-specific noise
that does not generalise out of sample.  Every feature at date *t* uses only
information published by the close of *t*.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    FRED_DAILY_PUBLICATION_LAG,
    HY_FEATURE_SPREAD_PREFERENCE,
    MIN_SPREAD_HISTORY,
    PRIMARY_SPREAD,
)

logger = logging.getLogger(__name__)

# Daily FRED columns that are published one business day after their date.
_FRED_DAILY_COLUMNS = (
    "baa_spread", "aaa_spread", "hy_spread", "ig_spread", "bbb_spread",
    "t10y2y", "t10y3m", "dgs10", "tbill_3m", "fed_funds",
)
_SPREAD_PREFERENCE = ("baa_spread", "hy_spread", "bbb_spread", "ig_spread", "aaa_spread")


def _concat_new(df: pd.DataFrame, new: dict[str, pd.Series]) -> pd.DataFrame:
    """Append many columns at once (avoids DataFrame fragmentation)."""
    if not new:
        return df.copy()
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def add_lagged_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Add lagged versions of selected columns as ``{col}_lag{n}``."""
    new: dict[str, pd.Series] = {}
    for col in columns:
        if col not in df.columns:
            logger.warning("Column %s not found – skipping lags.", col)
            continue
        for lag in lags:
            new[f"{col}_lag{lag}"] = df[col].shift(lag)
    return _concat_new(df, new)


def add_rolling_stats(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """Add trailing rolling mean, standard deviation, minimum, and maximum."""
    new: dict[str, pd.Series] = {}
    for col in columns:
        if col not in df.columns:
            logger.warning("Column %s not found – skipping rolling stats.", col)
            continue
        for w in windows:
            r = df[col].rolling(window=w, min_periods=w)
            new[f"{col}_rmean{w}"] = r.mean()
            new[f"{col}_rstd{w}"] = r.std()
            new[f"{col}_rmin{w}"] = r.min()
            new[f"{col}_rmax{w}"] = r.max()
    return _concat_new(df, new)


def add_momentum_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """Add rate-of-change features ``{col}_mom{w} = (x_t - x_{t-w}) / x_{t-w}``."""
    new: dict[str, pd.Series] = {}
    for col in columns:
        if col not in df.columns:
            logger.warning("Column %s not found – skipping momentum.", col)
            continue
        for w in windows:
            past = df[col].shift(w)
            new[f"{col}_mom{w}"] = (df[col] - past) / past.replace(0, np.nan)
    return _concat_new(df, new)


def add_yield_curve_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive yield-curve slope features from ``t10y2y``, ``t10y3m`` and ``fed_funds``."""
    new: dict[str, pd.Series] = {}
    if "t10y2y" in df.columns:
        new["yc_slope"] = df["t10y2y"]
        new["yc_slope_chg1"] = df["t10y2y"].diff(1)
        new["yc_slope_chg5"] = df["t10y2y"].diff(5)
        new["yc_slope_chg20"] = df["t10y2y"].diff(20)
    if "t10y3m" in df.columns:
        new["yc_slope_3m"] = df["t10y3m"]
        new["yc_slope_3m_chg60"] = df["t10y3m"].diff(60)
    if "fed_funds" in df.columns:
        new["rate_level"] = df["fed_funds"]
        new["rate_level_chg20"] = df["fed_funds"].diff(20)
    return _concat_new(df, new)


def add_cross_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-asset spread / ratio features."""
    new: dict[str, pd.Series] = {}
    if "hy_spread" in df.columns and "ig_spread" in df.columns:
        new["hy_ig_ratio"] = df["hy_spread"] / df["ig_spread"].replace(0, np.nan)
    if "bbb_spread" in df.columns and "ig_spread" in df.columns:
        new["bbb_ig_ratio"] = df["bbb_spread"] / df["ig_spread"].replace(0, np.nan)
    if "baa_spread" in df.columns and "aaa_spread" in df.columns:
        new["quality_spread"] = df["baa_spread"] - df["aaa_spread"]
    if "hy_spread" in df.columns and "vix" in df.columns:
        new["hy_per_vix"] = df["hy_spread"] / df["vix"].replace(0, np.nan)
    if "gold" in df.columns and "crude_oil" in df.columns:
        # Front-month crude traded below zero in April 2020.
        new["gold_crude_ratio"] = df["gold"] / df["crude_oil"].where(df["crude_oil"] > 0)
    return _concat_new(df, new)


def add_zscore_features(
    df: pd.DataFrame,
    columns: list[str],
    window: int = 60,
) -> pd.DataFrame:
    """Add trailing z-scores ``{col}_z{window} = (x - rolling_mean) / rolling_std``."""
    new: dict[str, pd.Series] = {}
    for col in columns:
        if col not in df.columns:
            logger.warning("Column %s not found – skipping z-score.", col)
            continue
        new[f"{col}_z{window}"] = rolling_zscore(df[col], window)
    return _concat_new(df, new)


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """Trailing z-score of *s* over *window* observations."""
    r = s.rolling(window=window, min_periods=window)
    return (s - r.mean()) / r.std().replace(0, np.nan)


def apply_publication_lag(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    lag: int = FRED_DAILY_PUBLICATION_LAG,
) -> pd.DataFrame:
    """Shift daily FRED columns so each row only holds values already published.

    FRED posts daily series (Moody's yields, Treasury rates, ICE OAS) one
    business day after their observation date, so a signal computed at the
    close of *t* can only use the value dated *t-1*.
    """
    if columns is None:
        columns = [c for c in _FRED_DAILY_COLUMNS if c in df.columns]
    out = df.copy()
    if lag:
        out[columns] = out[columns].shift(lag)
    return out


def create_targets(
    df: pd.DataFrame,
    target_col: str,
    horizons: list[int],
) -> pd.DataFrame:
    """Create forward-looking targets for each horizon *h*.

    * ``target_{h}d_change`` – change in *target_col* over the next *h* rows
      (basis points for spread columns).
    * ``target_{h}d_return`` – forward log-change over *h* rows.
    * ``target_{h}d_up`` – 1 if the forward change is positive, else 0
      (NaN where the forward value is not yet known).
    """
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in DataFrame.")
    x = df[target_col]
    for h in horizons:
        fwd = x.shift(-h)
        change = fwd - x
        df[f"target_{h}d_change"] = change
        df[f"target_{h}d_return"] = np.log(fwd / x)
        df[f"target_{h}d_up"] = (change > 0).astype(float).where(change.notna())
    return df


def _equity_price(df: pd.DataFrame) -> Optional[pd.Series]:
    """Equity price level, reconstructed from returns when no price column exists."""
    for col in ("sp500", "spy"):
        if col in df.columns:
            return df[col]
    for col in ("sp500_return", "spy_return"):
        if col in df.columns:
            return (1 + df[col].fillna(0)).cumprod()
    return None


def _equity_returns(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in ("sp500_return", "spy_return"):
        if col in df.columns:
            return df[col]
    price = _equity_price(df)
    return None if price is None else price.pct_change(fill_method=None)


def build_signal_features(
    df: pd.DataFrame,
    spread_col: Optional[str] = None,
    publication_lag: int = FRED_DAILY_PUBLICATION_LAG,
) -> pd.DataFrame:
    """Build the core stationary feature set used by the forecasting models.

    Features whose inputs are missing from *df* are skipped, so the function
    works on partial datasets (e.g. synthetic test data).

    Parameters
    ----------
    df:
        Trading-day DataFrame as produced by :func:`src.data.fetcher.fetch_all_data`.
    spread_col:
        Credit spread driving the spread features.  Defaults to the first
        available of ``baa_spread``, ``hy_spread``, ``bbb_spread``, ``ig_spread``.
    publication_lag:
        Rows by which daily FRED inputs are shifted (see :func:`apply_publication_lag`).

    Returns
    -------
    pd.DataFrame
        Feature DataFrame on the same index as *df*.
    """
    spread_col = spread_col or default_spread_column(df)
    lagged = apply_publication_lag(df, lag=publication_lag)
    f: dict[str, pd.Series] = {}

    s = lagged[spread_col]
    for w in (5, 20, 60):
        f[f"spread_chg_{w}"] = s.diff(w)
    f["spread_z60"] = rolling_zscore(s, 60)
    f["spread_z252"] = rolling_zscore(s, 252)
    f["spread_vol60"] = s.diff().rolling(60).std()

    if spread_col == "baa_spread" and "aaa_spread" in lagged.columns:
        f["quality_spread_chg_20"] = (lagged["baa_spread"] - lagged["aaa_spread"]).diff(20)
    if "dgs10" in lagged.columns:
        f["y10_chg_5"] = lagged["dgs10"].diff(5)
        f["y10_chg_20"] = lagged["dgs10"].diff(20)
    if "t10y3m" in lagged.columns:
        f["t10y3m_chg_60"] = lagged["t10y3m"].diff(60)

    price = _equity_price(df)
    if price is not None:
        for w in (5, 20, 60):
            f[f"eq_ret_{w}"] = np.log(price / price.shift(w))
        f["eq_drawdown_252"] = price / price.rolling(252).max() - 1
    rets = _equity_returns(df)
    if rets is not None:
        f["eq_vol_20"] = rets.rolling(20).std() * np.sqrt(252)

    if "vix" in df.columns:
        v = df["vix"]
        f["vix_chg_5"] = v.diff(5)
        f["vix_chg_20"] = v.diff(20)
        f["vix_z252"] = rolling_zscore(v, 252)
        if "eq_vol_20" in f:
            f["vol_risk_premium"] = v / 100 - f["eq_vol_20"]

    if "nfci" in df.columns:
        f["nfci_chg_20"] = df["nfci"].diff(20)
    if "claims" in df.columns:
        f["claims_growth_60"] = np.log(df["claims"]).diff(60)

    return pd.DataFrame(f, index=df.index)


def default_spread_column(df: pd.DataFrame) -> str:
    """Return the preferred available spread column (``PRIMARY_SPREAD`` first)."""
    for col in (PRIMARY_SPREAD, *_SPREAD_PREFERENCE):
        if col in df.columns and df[col].notna().any():
            return col
    raise ValueError("No credit spread column found in DataFrame.")


def spread_columns_with_history(df: pd.DataFrame, min_obs: int = MIN_SPREAD_HISTORY) -> list[str]:
    """Spread columns with at least *min_obs* observations, in preference order.

    FRED serves only ~3 years of the ICE OAS series, so those columns are only
    usable for modelling when a longer licensed history has been spliced in
    (see :func:`src.data.fetcher.load_local_spread_history`).
    """
    ordered = [PRIMARY_SPREAD, *(c for c in _SPREAD_PREFERENCE if c != PRIMARY_SPREAD)]
    return [c for c in ordered if c in df.columns and int(df[c].notna().sum()) >= min_obs]


def hy_feature_spread_column(df: pd.DataFrame, min_obs: int = MIN_SPREAD_HISTORY) -> str:
    """Spread used for the high-yield model's core features.

    Prefers the ICE high-yield OAS when a long history is available – it beat
    the Moody's Baa proxy in both walk-forward periods – and otherwise falls
    back to :func:`default_spread_column`.
    """
    for col in HY_FEATURE_SPREAD_PREFERENCE:
        if col in df.columns and int(df[col].notna().sum()) >= min_obs:
            return col
    return default_spread_column(df)


def horizon_from_target_name(name: object) -> Optional[int]:
    """Rows until a ``target_{h}d_*`` label is fully known.

    Targets that start after an execution lag are named
    ``target_{h}d_lag{L}_*`` and need ``h + L`` rows.
    """
    m = re.match(r"target_(\d+)d_(?:lag(\d+)_)?", str(name))
    if not m:
        return None
    return int(m.group(1)) + int(m.group(2) or 0)


def _finalise(X: pd.DataFrame, y: pd.DataFrame, dropna: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Drop features that are entirely missing (e.g. inputs absent in this dataset window).
    X = X.loc[:, X.notna().any()]
    if dropna:
        mask = X.notna().all(axis=1) & y.notna().all(axis=1)
        dropped = int((~mask).sum())
        X, y = X.loc[mask], y.loc[mask]
        logger.info(
            "Feature matrix shape: X=%s  y=%s (dropped %d rows with NaN)", X.shape, y.shape, dropped
        )
    return X, y


def regime_features(regime_probs: pd.DataFrame) -> pd.DataFrame:
    """Turn ordered regime probabilities into features.

    ``regime_probs`` must hold ``regime_prob_0 … regime_prob_{k-1}`` from a
    causal source such as
    :func:`src.models.regime.walk_forward_regime_probabilities`.  Returns the
    probability of the most stressed regime and the expected regime level.
    """
    cols = sorted(c for c in regime_probs.columns if c.startswith("regime_prob_"))
    if not cols:
        raise ValueError("regime_probs must contain regime_prob_* columns.")
    probs = regime_probs[cols]
    levels = np.arange(len(cols), dtype=float)
    return pd.DataFrame({
        "regime_p_stress": probs[cols[-1]],
        "regime_level": probs.mul(levels, axis=1).sum(axis=1, min_count=len(cols)),
    }, index=regime_probs.index)


def build_hy_features(df: pd.DataFrame, name: str = "hyg_xs") -> pd.DataFrame:
    """Momentum, volatility and drawdown of a hedged high-yield excess-return index.

    Expects ``{name}_return`` and ``{name}_index`` columns (see
    :func:`src.data.fetcher.add_hy_excess_returns`).  Adds changes in the
    Gilchrist–Zakrajšek spread when available.
    """
    f: dict[str, pd.Series] = {}
    ret_col, idx_col = f"{name}_return", f"{name}_index"
    if ret_col in df.columns and idx_col in df.columns:
        idx = df[idx_col]
        for w in (5, 20, 60):
            f[f"hy_xs_ret_{w}"] = np.log(idx / idx.shift(w))
        f["hy_xs_vol_20"] = df[ret_col].rolling(20).std() * np.sqrt(252)
        f["hy_xs_drawdown_252"] = idx / idx.rolling(252).max() - 1
    if "gz_spread" in df.columns:
        f["gz_spread_chg_60"] = df["gz_spread"].diff(60)
    if "ebp" in df.columns:
        f["ebp_z756"] = rolling_zscore(df["ebp"], 756)
    return pd.DataFrame(f, index=df.index)


def create_return_targets(
    returns: pd.Series,
    horizon: int,
    execution_lag: int = 1,
) -> pd.DataFrame:
    """Forward compounded return targets that start after an execution lag.

    The label at *t* compounds the returns of rows ``t+1+lag … t+h+lag``: a
    position decided at the close of *t* is opened at the close of *t+lag*.

    Returns
    -------
    pd.DataFrame
        ``target_{h}d_lag{lag}_xs_return`` (bps) and ``target_{h}d_lag{lag}_xs_up``.
    """
    log_ret = np.log1p(returns)
    window = log_ret.rolling(horizon, min_periods=horizon).sum().shift(-(horizon + execution_lag))
    fwd = np.expm1(window) * 1e4
    prefix = f"target_{horizon}d_lag{execution_lag}_xs"
    return pd.DataFrame({
        f"{prefix}_return": fwd,
        f"{prefix}_up": (fwd > 0).astype(float).where(fwd.notna()),
    }, index=returns.index)


def build_hy_feature_matrix(
    df: pd.DataFrame,
    target_horizon: int = 5,
    hy_name: str = "hyg_xs",
    execution_lag: int = 1,
    publication_lag: int = FRED_DAILY_PUBLICATION_LAG,
    regime_probs: Optional[pd.DataFrame] = None,
    dropna: bool = True,
    spread_col: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features and targets for forecasting tradable high-yield excess returns.

    The target is the duration-hedged excess return of *hy_name* (default:
    HYG vs IEI) over *target_horizon* days, starting *execution_lag* days after
    the signal.  Features are the core signal features plus
    :func:`build_hy_features`.

    Mutual-fund based proxies (``hy_fund_xs``) have stale prices, which makes
    their returns look predictable in ways no investor can trade; use the ETF
    pair for evaluation.

    *spread_col* selects the credit spread behind the core features; it
    defaults to :func:`hy_feature_spread_column`.
    """
    spread_col = spread_col or hy_feature_spread_column(df)
    X = pd.concat([
        build_signal_features(df, spread_col=spread_col, publication_lag=publication_lag),
        build_hy_features(df, hy_name),
    ], axis=1)
    if regime_probs is not None:
        X = X.join(regime_features(regime_probs))
    ret_col = f"{hy_name}_return"
    if ret_col not in df.columns:
        raise ValueError(f"Column '{ret_col}' not found – run add_hy_excess_returns first.")
    y = create_return_targets(df[ret_col], target_horizon, execution_lag)
    return _finalise(X, y, dropna)


def build_feature_matrix(
    df: pd.DataFrame,
    target_horizon: int = 5,
    target_col: Optional[str] = None,
    publication_lag: int = FRED_DAILY_PUBLICATION_LAG,
    dropna: bool = True,
    regime_probs: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the feature matrix *X* and target DataFrame *y*.

    Targets are measured from the as-dated spread at *t* to *t + h*, while
    features only use data published by *t*, so there is no overlap between
    the information in *X* and the period the target covers.

    Parameters
    ----------
    df:
        Raw trading-day DataFrame.
    target_horizon:
        Forward horizon in trading days.
    target_col:
        Spread to forecast.  Defaults to :func:`default_spread_column`.
    publication_lag:
        Publication lag applied to daily FRED inputs.
    dropna:
        Drop rows with any missing feature or target.  With ``False`` rows are
        kept (useful for producing live signals where the target is unknown).
    regime_probs:
        Optional causal regime probabilities to add as features (see
        :func:`regime_features`).  Walk-forward tests found no improvement from
        them, so they are off by default.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(X, y)`` aligned on the same index.  *y* holds
        ``target_{h}d_change``, ``target_{h}d_return`` and ``target_{h}d_up``.
    """
    target_col = target_col or default_spread_column(df)
    X = build_signal_features(df, spread_col=target_col, publication_lag=publication_lag)
    if regime_probs is not None:
        X = X.join(regime_features(regime_probs))
    y = create_targets(df[[target_col]], target_col, horizons=[target_horizon]).drop(columns=[target_col])
    return _finalise(X, y, dropna)
