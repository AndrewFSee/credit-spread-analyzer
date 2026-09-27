"""
Leading-indicator backtesting for the Credit Spread Analysis & Prediction Platform.

Uses credit-spread signals to set the equity weight of an equity / cash
portfolio, then evaluates the result against buy-and-hold.  Signals can be
all-or-nothing (spread widening, spread z-score) or fractional (volatility
targeting scaled by the probability of a credit-stress regime).

Timing convention: a signal dated *t* uses only data published by the close
of *t* (daily FRED series are shifted by their one-day publication lag).  With
``execution_lag=1`` the position changes at the close of *t + 1*, so the
first return it earns is the one from *t + 1* to *t + 2*.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import EXPOSURE_METHOD, EXPOSURE_TARGET_VOL, FRED_DAILY_PUBLICATION_LAG
from src.features.engineering import rolling_zscore

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def compute_spread_signal(
    df: pd.DataFrame,
    spread_col: str = "baa_spread",
    widen_threshold: float = 50.0,
    lookback_days: int = 20,
    publication_lag: int = FRED_DAILY_PUBLICATION_LAG,
) -> pd.Series:
    """Binary defensive signal based on sharp spread widening.

    The signal is 1 (go defensive) when the spread has widened by more than
    *widen_threshold* basis points over the prior *lookback_days* days.

    Parameters
    ----------
    df:
        Input DataFrame containing *spread_col* in basis points.
    spread_col:
        Column name of the credit spread series.
    widen_threshold:
        Basis-point widening threshold to trigger a defensive signal.
    lookback_days:
        Number of days to measure the spread change over.
    publication_lag:
        Rows by which the spread is shifted to reflect its publication delay.

    Returns
    -------
    pd.Series
        Signal (0 = risk-on, 1 = defensive, NaN during warm-up) aligned with *df*.
    """
    if spread_col not in df.columns:
        raise ValueError(f"Column '{spread_col}' not found in DataFrame.")

    spread_change = df[spread_col].shift(publication_lag).diff(lookback_days)
    signal = (spread_change > widen_threshold).astype(float).where(spread_change.notna())
    signal.name = "spread_signal"
    return signal


def compute_zscore_signal(
    df: pd.DataFrame,
    spread_col: str = "baa_spread",
    window: int = 252,
    enter_threshold: float = 0.5,
    exit_threshold: Optional[float] = 0.0,
    publication_lag: int = FRED_DAILY_PUBLICATION_LAG,
) -> pd.Series:
    """Defensive signal from the spread's trailing z-score, with hysteresis.

    Go defensive when the z-score rises above *enter_threshold*; return to
    risk-on only once it falls below *exit_threshold*.  The gap between the
    two thresholds suppresses whipsaw trades.  ``exit_threshold=None`` uses a
    single threshold.

    Returns
    -------
    pd.Series
        Signal (0 = risk-on, 1 = defensive, NaN during warm-up) aligned with *df*.
    """
    if spread_col not in df.columns:
        raise ValueError(f"Column '{spread_col}' not found in DataFrame.")
    if exit_threshold is None:
        exit_threshold = enter_threshold
    if exit_threshold > enter_threshold:
        raise ValueError("exit_threshold must not exceed enter_threshold.")

    z = rolling_zscore(df[spread_col].shift(publication_lag), window)
    out = np.full(len(z), np.nan)
    state = 0.0
    for i, value in enumerate(z.to_numpy()):
        if np.isnan(value):
            continue
        if state == 0.0 and value > enter_threshold:
            state = 1.0
        elif state == 1.0 and value < exit_threshold:
            state = 0.0
        out[i] = state
    return pd.Series(out, index=df.index, name="zscore_signal")


def compute_vol_target_weight(
    df: pd.DataFrame,
    equity_col: str = "spy_return",
    target_vol: float = EXPOSURE_TARGET_VOL,
    span: int = 20,
) -> pd.Series:
    """Equity weight that targets *target_vol* annualised volatility (capped at 1).

    Uses an exponentially weighted estimate of recent equity volatility, known
    at each close.
    """
    if equity_col not in df.columns:
        raise ValueError(f"Column '{equity_col}' not found in DataFrame.")
    vol = df[equity_col].ewm(span=span, min_periods=span).std() * np.sqrt(TRADING_DAYS)
    return (target_vol / vol.replace(0, np.nan)).clip(upper=1.0).rename("vol_target_weight")


def compute_regime_weight(
    df: pd.DataFrame,
    spread_col: str = "baa_spread",
    n_states: int = 3,
    regime_probs: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Equity weight ``1 − P(most stressed credit regime)``.

    Probabilities come from :func:`src.models.regime.real_time_regime_probabilities`
    (HMM refitted yearly on past data, forward-filtered) unless *regime_probs*
    is supplied.
    """
    if regime_probs is None:
        from src.models.regime import real_time_regime_probabilities

        regime_probs = real_time_regime_probabilities(df, spread_col=spread_col, n_states=n_states)
    stress_col = sorted(c for c in regime_probs.columns if c.startswith("regime_prob_"))[-1]
    return (1.0 - regime_probs[stress_col]).reindex(df.index).rename("regime_weight")


EXPOSURE_METHODS = ("vol_regime", "vol", "regime", "zscore", "widening")


def compute_exposure_weight(
    df: pd.DataFrame,
    method: str = EXPOSURE_METHOD,
    spread_col: str = "baa_spread",
    equity_col: str = "spy_return",
    target_vol: float = EXPOSURE_TARGET_VOL,
    vol_span: int = 20,
    n_states: int = 3,
    regime_probs: Optional[pd.DataFrame] = None,
    zscore_window: int = 252,
    enter_threshold: float = 0.5,
    exit_threshold: Optional[float] = 0.0,
    widen_threshold: float = 50.0,
    lookback_days: int = 20,
) -> pd.Series:
    """Target equity weight (0 … 1) for each date.

    Methods
    -------
    ``vol_regime`` (default)
        Volatility target × (1 − P(credit stress regime)).  The only overlay
        that both raised the Sharpe ratio and kept the maximum drawdown under
        20% in the 2000–2018 and 2019–2026 walk-forward tests.
    ``vol``
        Volatility target only.
    ``regime``
        1 − P(credit stress regime).
    ``zscore`` / ``widening``
        All-or-nothing versions of :func:`compute_zscore_signal` and
        :func:`compute_spread_signal`.
    """
    if method == "vol_regime":
        return (
            compute_vol_target_weight(df, equity_col, target_vol, vol_span)
            * compute_regime_weight(df, spread_col, n_states, regime_probs)
        ).rename("weight")
    if method == "vol":
        return compute_vol_target_weight(df, equity_col, target_vol, vol_span).rename("weight")
    if method == "regime":
        return compute_regime_weight(df, spread_col, n_states, regime_probs).rename("weight")
    if method == "zscore":
        signal = compute_zscore_signal(
            df, spread_col=spread_col, window=zscore_window,
            enter_threshold=enter_threshold, exit_threshold=exit_threshold,
        )
        return (1.0 - signal).rename("weight")
    if method == "widening":
        signal = compute_spread_signal(
            df, spread_col=spread_col, widen_threshold=widen_threshold, lookback_days=lookback_days,
        )
        return (1.0 - signal).rename("weight")
    raise ValueError(f"Unknown method '{method}'. Choose from {EXPOSURE_METHODS}.")


def _cash_returns(df: pd.DataFrame, rf_col: Optional[str], risk_free_rate: float) -> pd.Series:
    """Daily cash return; uses the (lagged) T-bill yield column when available."""
    if rf_col and rf_col in df.columns:
        rate = df[rf_col].shift(1).ffill().fillna(risk_free_rate * 100)
        return rate / 100.0 / TRADING_DAYS
    return pd.Series(risk_free_rate / TRADING_DAYS, index=df.index)


def apply_rebalance_band(weight: pd.Series, band: float) -> pd.Series:
    """Hold the current weight until the target moves by more than *band*.

    Moves to exactly 0 or 1 are always executed so that all-or-nothing
    signals are unaffected.  NaN targets keep the current weight.
    """
    if band <= 0:
        return weight
    out = np.full(len(weight), np.nan)
    current = np.nan
    for i, target in enumerate(weight.to_numpy(dtype=float)):
        if not np.isnan(target):
            to_bound = target in (0.0, 1.0) and target != current
            if np.isnan(current) or abs(target - current) > band or to_bound:
                current = target
        out[i] = current
    return pd.Series(out, index=weight.index, name=weight.name)


def backtest_allocation(
    df: pd.DataFrame,
    weight: pd.Series,
    equity_col: str = "spy_return",
    risk_free_rate: float = 0.02,
    rf_col: Optional[str] = "tbill_3m",
    execution_lag: int = 1,
    cost_bps: float = 5.0,
    rebalance_band: float = 0.0,
) -> pd.DataFrame:
    """Simulate a portfolio holding *weight* in equities and the rest in cash.

    Parameters
    ----------
    df:
        DataFrame containing *equity_col* (daily **simple** returns).
    weight:
        Target equity weight dated when it is decided (0 … 1).  The weight
        in force on day *t* is the target from ``1 + execution_lag`` rows
        earlier; days where it is unknown are dropped.
    equity_col:
        Column name for daily equity returns.
    risk_free_rate:
        Annualised cash rate used when *rf_col* is unavailable.
    rf_col:
        Column with the annualised T-bill yield in percent.
    execution_lag:
        Extra days between the decision date and the trade (see module docstring).
    cost_bps:
        Cost in basis points per unit of turnover (a full switch costs *cost_bps*).
    rebalance_band:
        Minimum weight change that triggers a trade (see :func:`apply_rebalance_band`).

    Returns
    -------
    pd.DataFrame
        Columns ``weight``, ``signal`` (defensive share ``1 − weight``),
        ``equity_return``, ``cash_return``, ``turnover``, ``strategy_return``,
        ``bh_cumulative`` and ``strategy_cumulative``.
    """
    if equity_col not in df.columns:
        raise ValueError(f"Column '{equity_col}' not found in DataFrame.")

    target = weight.reindex(df.index).clip(0.0, 1.0)
    bt = pd.DataFrame(index=df.index)
    bt["weight"] = apply_rebalance_band(target, rebalance_band).shift(1 + execution_lag)
    bt["equity_return"] = df[equity_col]
    bt["cash_return"] = _cash_returns(df, rf_col, risk_free_rate)
    bt = bt.dropna(subset=["weight", "equity_return"])
    bt["signal"] = 1.0 - bt["weight"]

    bt["turnover"] = bt["weight"].diff().abs().fillna(0.0)
    bt["strategy_return"] = (
        bt["weight"] * bt["equity_return"]
        + (1.0 - bt["weight"]) * bt["cash_return"]
        - bt["turnover"] * cost_bps / 1e4
    )
    bt["bh_cumulative"] = (1 + bt["equity_return"]).cumprod()
    bt["strategy_cumulative"] = (1 + bt["strategy_return"]).cumprod()
    return bt


def backtest_strategy(
    df: pd.DataFrame,
    signal: pd.Series,
    equity_col: str = "spy_return",
    risk_free_rate: float = 0.02,
    rf_col: Optional[str] = "tbill_3m",
    execution_lag: int = 1,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Simulate an all-or-nothing strategy: cash while *signal* is 1, equities while 0.

    Thin wrapper around :func:`backtest_allocation` with ``weight = 1 − signal``;
    the ``signal`` column is returned as integers.
    """
    bt = backtest_allocation(
        df, 1.0 - signal.reindex(df.index), equity_col=equity_col, risk_free_rate=risk_free_rate,
        rf_col=rf_col, execution_lag=execution_lag, cost_bps=cost_bps,
    )
    bt["signal"] = bt["signal"].round().astype(int)
    return bt


def _performance(returns: pd.Series, cash: pd.Series) -> dict[str, float]:
    if returns.empty:
        return {k: float("nan") for k in ("cagr", "vol", "sharpe", "max_drawdown", "total_return")}
    wealth = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    excess = returns - cash
    vol = float(returns.std() * np.sqrt(TRADING_DAYS))
    ex_std = float(excess.std())
    return {
        "cagr": float(wealth.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan"),
        "vol": vol,
        "sharpe": float(excess.mean() / ex_std * np.sqrt(TRADING_DAYS)) if ex_std > 0 else 0.0,
        "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
        "total_return": float(wealth.iloc[-1] - 1),
    }


def compute_backtest_metrics(backtest_df: pd.DataFrame) -> dict[str, float]:
    """Compute strategy and buy-and-hold performance metrics.

    ``sharpe`` is the annualised Sharpe ratio of returns in excess of cash.
    ``annualised_return`` is the geometric (CAGR) return.  ``switches_per_year``
    is the annual turnover (one full switch = 1).  Metrics prefixed ``bh_``
    describe buy-and-hold over the same period.
    """
    ret = backtest_df["strategy_return"]
    cash = backtest_df["cash_return"] if "cash_return" in backtest_df else pd.Series(0.0, index=ret.index)
    weight = backtest_df["weight"] if "weight" in backtest_df else 1.0 - backtest_df["signal"]
    strat = _performance(ret, cash)
    bh = _performance(backtest_df["equity_return"], cash)
    years = len(ret) / TRADING_DAYS
    turnover = float(weight.diff().abs().sum() / years) if years > 0 else float("nan")

    return {
        "sharpe": strat["sharpe"],
        "max_drawdown": strat["max_drawdown"],
        "win_rate": float((ret > 0).mean()),
        "total_return": strat["total_return"],
        "avg_daily_return": float(ret.mean()),
        "annualised_return": strat["cagr"],
        "annualised_volatility": strat["vol"],
        "avg_return_defensive": float(ret[weight < 1].mean()),
        "avg_return_risk_on": float(ret[weight >= 1].mean()),
        "avg_equity_weight": float(weight.mean()),
        "time_defensive": float(1.0 - weight.mean()),
        "switches_per_year": turnover,
        "turnover_per_year": turnover,
        "bh_sharpe": bh["sharpe"],
        "bh_max_drawdown": bh["max_drawdown"],
        "bh_annualised_return": bh["cagr"],
        "bh_annualised_volatility": bh["vol"],
        "bh_total_return": bh["total_return"],
    }


def run_full_backtest(
    df: pd.DataFrame,
    spread_col: str = "baa_spread",
    equity_col: str = "spy_return",
    method: str = EXPOSURE_METHOD,
    widen_threshold: float = 50.0,
    lookback_days: int = 20,
    zscore_window: int = 252,
    enter_threshold: float = 0.5,
    exit_threshold: Optional[float] = 0.0,
    target_vol: float = EXPOSURE_TARGET_VOL,
    vol_span: int = 20,
    n_states: int = 3,
    regime_probs: Optional[pd.DataFrame] = None,
    rebalance_band: Optional[float] = None,
    risk_free_rate: float = 0.02,
    execution_lag: int = 1,
    cost_bps: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Orchestrate the full exposure-signal backtest pipeline.

    Parameters
    ----------
    method:
        One of :data:`EXPOSURE_METHODS` (see :func:`compute_exposure_weight`).
    equity_col:
        Equity return column; falls back to ``sp500_return`` when missing.
    regime_probs:
        Pre-computed walk-forward regime probabilities (saves ~40 s of HMM
        fitting for the regime-based methods).
    rebalance_band:
        Defaults to 0.10 for fractional methods and 0 for all-or-nothing ones.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, float]]
        ``(backtest_df, metrics)``
    """
    if equity_col not in df.columns and "sp500_return" in df.columns:
        equity_col = "sp500_return"
    if method not in EXPOSURE_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {EXPOSURE_METHODS}.")

    weight = compute_exposure_weight(
        df, method=method, spread_col=spread_col, equity_col=equity_col,
        target_vol=target_vol, vol_span=vol_span, n_states=n_states, regime_probs=regime_probs,
        zscore_window=zscore_window, enter_threshold=enter_threshold, exit_threshold=exit_threshold,
        widen_threshold=widen_threshold, lookback_days=lookback_days,
    )
    if rebalance_band is None:
        rebalance_band = 0.0 if method in ("zscore", "widening") else 0.10

    backtest_df = backtest_allocation(
        df,
        weight=weight,
        equity_col=equity_col,
        risk_free_rate=risk_free_rate,
        execution_lag=execution_lag,
        cost_bps=cost_bps,
        rebalance_band=rebalance_band,
    )
    metrics = compute_backtest_metrics(backtest_df)

    logger.info(
        "Backtest (%s) – Sharpe: %.2f (B&H %.2f) | MaxDD: %.1f%% (B&H %.1f%%)",
        method,
        metrics["sharpe"],
        metrics["bh_sharpe"],
        metrics["max_drawdown"] * 100,
        metrics["bh_max_drawdown"] * 100,
    )
    return backtest_df, metrics
