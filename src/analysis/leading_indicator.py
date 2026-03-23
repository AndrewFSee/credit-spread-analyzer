"""
Leading-indicator backtesting for the Credit Spread Analysis & Prediction Platform.

Uses credit-spread signals to rotate between risk-on and risk-off positions,
then evaluates the resulting strategy with standard performance metrics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_spread_signal(
    df: pd.DataFrame,
    spread_col: str = "hy_spread",
    widen_threshold: float = 50.0,
    lookback_days: int = 20,
) -> pd.Series:
    """Compute a binary defensive signal based on spread widening.

    The signal is 1 (go defensive) when the spread has widened by more than
    *widen_threshold* basis points over the prior *lookback_days* days.

    Parameters
    ----------
    df:
        Input DataFrame containing *spread_col*.
    spread_col:
        Column name of the credit spread series.
    widen_threshold:
        Basis-point widening threshold to trigger a defensive signal.
    lookback_days:
        Number of days to measure the spread change over.

    Returns
    -------
    pd.Series
        Binary signal Series (0 = risk-on, 1 = defensive) aligned with *df*.
    """
    if spread_col not in df.columns:
        raise ValueError(f"Column '{spread_col}' not found in DataFrame.")

    spread_change = df[spread_col].diff(lookback_days)
    signal = (spread_change > widen_threshold).astype(int)
    signal.name = "spread_signal"
    return signal


def backtest_strategy(
    df: pd.DataFrame,
    signal: pd.Series,
    equity_col: str = "sp500_return",
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """Simulate a credit-spread-based tactical allocation strategy.

    When *signal* == 1 the strategy holds cash (daily risk-free rate).
    When *signal* == 0 the strategy is fully invested in equities.

    Parameters
    ----------
    df:
        DataFrame containing *equity_col* (daily log-returns).
    signal:
        Binary signal Series aligned with *df* (1 = defensive).
    equity_col:
        Column name for daily equity log-returns.
    risk_free_rate:
        Annualised risk-free rate used to compute daily cash return.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ``signal``, ``equity_return``, ``strategy_return``,
        ``bh_cumulative``, ``strategy_cumulative``.
    """
    if equity_col not in df.columns:
        raise ValueError(f"Column '{equity_col}' not found in DataFrame.")

    daily_rf = risk_free_rate / 252.0

    bt = pd.DataFrame(index=df.index)
    bt["signal"] = signal.reindex(df.index).fillna(0).astype(int)
    bt["equity_return"] = df[equity_col].fillna(0.0)

    # Strategy: risk-free when signal==1, equity when signal==0
    bt["strategy_return"] = np.where(
        bt["signal"] == 1,
        daily_rf,
        bt["equity_return"],
    )

    bt["bh_cumulative"] = (1 + bt["equity_return"]).cumprod()
    bt["strategy_cumulative"] = (1 + bt["strategy_return"]).cumprod()

    return bt


def compute_backtest_metrics(backtest_df: pd.DataFrame) -> dict[str, float]:
    """Compute performance metrics from a backtest DataFrame.

    Parameters
    ----------
    backtest_df:
        DataFrame produced by :func:`backtest_strategy`.

    Returns
    -------
    dict[str, float]
        Dictionary with ``sharpe``, ``max_drawdown``, ``win_rate``,
        ``total_return``, ``avg_daily_return``, ``annualised_return``,
        ``annualised_volatility`` keys.
    """
    ret = backtest_df["strategy_return"]
    cum = backtest_df["strategy_cumulative"]

    # Annualised metrics (252 trading days)
    ann_return = float((1 + ret.mean()) ** 252 - 1)
    ann_vol = float(ret.std() * np.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max.replace(0, np.nan)
    max_dd = float(drawdown.min())

    # Win rate
    win_rate = float((ret > 0).mean())
    total_return = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0.0

    # Per-regime avg return (defensive vs risk-on)
    defensive_avg = float(ret[backtest_df["signal"] == 1].mean())
    risk_on_avg = float(ret[backtest_df["signal"] == 0].mean())

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_return": total_return,
        "avg_daily_return": float(ret.mean()),
        "annualised_return": ann_return,
        "annualised_volatility": ann_vol,
        "avg_return_defensive": defensive_avg,
        "avg_return_risk_on": risk_on_avg,
    }


def run_full_backtest(
    df: pd.DataFrame,
    spread_col: str = "hy_spread",
    equity_col: str = "sp500_return",
    widen_threshold: float = 50.0,
    lookback_days: int = 20,
    risk_free_rate: float = 0.02,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Orchestrate the full spread-signal backtest pipeline.

    Parameters
    ----------
    df:
        Raw DataFrame with spread and equity-return columns.
    spread_col:
        Credit-spread column to generate the signal from.
    equity_col:
        Equity log-return column.
    widen_threshold:
        Basis-point widening threshold.
    lookback_days:
        Signal look-back period.
    risk_free_rate:
        Annualised risk-free rate.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, float]]
        ``(backtest_df, metrics)``
    """
    signal = compute_spread_signal(
        df,
        spread_col=spread_col,
        widen_threshold=widen_threshold,
        lookback_days=lookback_days,
    )
    backtest_df = backtest_strategy(
        df,
        signal=signal,
        equity_col=equity_col,
        risk_free_rate=risk_free_rate,
    )
    metrics = compute_backtest_metrics(backtest_df)

    logger.info(
        "Backtest complete – Sharpe: %.2f | MaxDD: %.2f%% | Total return: %.2f%%",
        metrics["sharpe"],
        metrics["max_drawdown"] * 100,
        metrics["total_return"] * 100,
    )
    return backtest_df, metrics
