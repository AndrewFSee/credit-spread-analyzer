"""
Smoke tests for src/visualization/plots.py.

The dashboard renders every chart through these functions, so each must build
a figure from typical inputs with readable labels.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _spreads(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({
        "baa_spread": 200 + np.cumsum(rng.normal(0, 2, n)),
        "aaa_spread": 100 + np.cumsum(rng.normal(0, 1, n)),
    }, index=idx)


class TestPlots(unittest.TestCase):
    """Every Plotly chart builds and uses readable names."""

    def test_spread_history_uses_readable_names(self) -> None:
        from src.visualization.plots import plot_spread_history

        fig = plot_spread_history(_spreads())
        self.assertEqual([t.name for t in fig.data], ["Moody's Baa – 10y", "Moody's Aaa – 10y"])

    def test_regime_overlay_uses_ordinal_colours(self) -> None:
        from src.visualization.plots import ORDINAL_BLUES, plot_regime_overlay

        df = _spreads()
        regimes = np.repeat([0, 1, 2], len(df) // 3)
        fig = plot_regime_overlay(df, regimes)
        self.assertEqual([t.marker.color for t in fig.data], ORDINAL_BLUES[3])
        self.assertTrue(fig.data[0].name.endswith("(calmest)"))

    def test_backtest_plot_has_log_ticks_and_weight_panel(self) -> None:
        from src.visualization.plots import plot_backtest_results

        idx = pd.bdate_range("2020-01-01", periods=500)
        bt = pd.DataFrame({
            "strategy_cumulative": np.linspace(1, 3, 500),
            "bh_cumulative": np.linspace(1, 4, 500),
            "weight": np.where(np.arange(500) % 100 < 50, 1.0, 0.4),
        }, index=idx)
        fig = plot_backtest_results(bt)
        self.assertEqual(fig.layout.yaxis.type, "log")
        self.assertIn("$1", fig.layout.yaxis.ticktext)
        self.assertEqual(fig.data[2].name, "Equity weight")
        # Old backtests without a weight column still plot (weight = 1 - signal).
        legacy = bt.drop(columns="weight").assign(signal=0)
        self.assertEqual(float(plot_backtest_results(legacy).data[2].y.max()), 1.0)

    def test_feature_importance_is_sorted_and_labelled(self) -> None:
        from src.visualization.plots import plot_feature_importance

        imp = pd.Series({"eq_ret_20": 0.2, "vix_chg_5": 0.5, "unknown_feature": 0.3})
        fig = plot_feature_importance(imp)
        bar = fig.data[0]
        self.assertEqual(list(bar.y), ["S&P 500 return, 20d", "unknown_feature", "VIX change, 5d"])  # ascending → largest on top
        self.assertAlmostEqual(float(sum(bar.x)), 1.0)

    def test_forecast_and_stress_charts(self) -> None:
        from src.visualization.plots import plot_forecast_vs_actual, plot_stress_probability

        idx = pd.bdate_range("2024-01-01", periods=50)
        fig = plot_forecast_vs_actual(np.zeros(50), np.ones(50), index=idx, y_title="Spread change (bps)")
        self.assertEqual([t.name for t in fig.data], ["Actual", "Forecast"])
        self.assertEqual(fig.layout.yaxis.title.text, "Spread change (bps)")

        probs = pd.Series(np.linspace(0, 1, 50), index=idx)
        fig = plot_stress_probability(probs)
        self.assertEqual(fig.layout.yaxis.tickformat, ".0%")

    def test_series_label_fallback(self) -> None:
        from src.visualization.plots import series_label

        self.assertEqual(series_label("hy_spread"), "ICE BofA High Yield OAS")
        self.assertEqual(series_label("spread_z252"), "Spread z-score, 1y")
        self.assertEqual(series_label("something_new"), "something_new")


if __name__ == "__main__":
    unittest.main()
