"""Analysis sub-package: leading indicator backtesting."""

from src.analysis.leading_indicator import (
    compute_spread_signal,
    backtest_strategy,
    compute_backtest_metrics,
    run_full_backtest,
)

__all__ = [
    "compute_spread_signal",
    "backtest_strategy",
    "compute_backtest_metrics",
    "run_full_backtest",
]
