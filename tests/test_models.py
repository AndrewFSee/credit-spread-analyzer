"""
Tests for regime detection, ML models, and backtest analysis.

Uses small synthetic datasets to keep tests fast and dependency-free.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _make_spread_df(n: int = 300) -> pd.DataFrame:
    """Create a small synthetic DataFrame for model testing."""
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame(
        {
            "hy_spread": 400 + np.cumsum(rng.normal(0, 5, n)),
            "ig_spread": 120 + np.cumsum(rng.normal(0, 2, n)),
            "vix": 18 + np.abs(rng.normal(0, 4, n)),
            "sp500_return": rng.normal(0.0004, 0.01, n),
        },
        index=idx,
    )


# ============================================================
# Regime detection tests
# ============================================================

class TestFitHMM(unittest.TestCase):
    """Tests for fit_hmm()."""

    def test_fit_hmm_returns_labels(self) -> None:
        """fit_hmm + label_regimes should return an integer array with n_states unique values."""
        try:
            from hmmlearn.hmm import GaussianHMM  # noqa: F401 – check availability
        except ImportError:
            self.skipTest("hmmlearn not installed")

        from src.models.regime import fit_hmm, label_regimes

        df = _make_spread_df(200)
        data = df[["hy_spread"]].values
        model = fit_hmm(data, n_states=3)
        labels = label_regimes(model, data, model_type="hmm")

        self.assertEqual(len(labels), len(data))
        unique_labels = np.unique(labels)
        self.assertLessEqual(len(unique_labels), 3)

    def test_fit_hmm_transition_matrix_rows_sum_to_one(self) -> None:
        """Each row of the transition matrix must sum to 1."""
        try:
            from hmmlearn.hmm import GaussianHMM  # noqa: F401
        except ImportError:
            self.skipTest("hmmlearn not installed")

        from src.models.regime import fit_hmm, get_transition_matrix

        df = _make_spread_df(200)
        data = df[["hy_spread"]].values
        model = fit_hmm(data, n_states=3)
        trans = get_transition_matrix(model, model_type="hmm")

        row_sums = trans.values.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-6)


# ============================================================
# ML model tests
# ============================================================

class TestTrainAndEvaluateXGBoost(unittest.TestCase):
    """Tests for train_and_evaluate() with XGBoost."""

    def test_train_and_evaluate_xgboost_returns_metrics(self) -> None:
        """train_and_evaluate should return a dict with model and metrics keys."""
        try:
            import xgboost  # noqa: F401
        except ImportError:
            self.skipTest("xgboost not installed")

        from src.features.engineering import build_feature_matrix
        from src.models.ml_models import train_and_evaluate

        df = _make_spread_df(300)
        X, y = build_feature_matrix(df, target_horizon=5)
        target_col = next(c for c in y.columns if "return" in c)

        result = train_and_evaluate(
            X, y[target_col], model_type="xgboost", task="regression", n_splits=3
        )

        self.assertIn("model", result)
        self.assertIn("mean_metrics", result)
        self.assertIn("rmse", result["mean_metrics"])
        self.assertIn("directional_accuracy", result["mean_metrics"])
        # RMSE should be a non-negative finite number
        self.assertTrue(np.isfinite(result["mean_metrics"]["rmse"]))
        self.assertGreaterEqual(result["mean_metrics"]["rmse"], 0)

    def test_train_and_evaluate_feature_importance_length(self) -> None:
        """Feature importance Series length must match number of input features."""
        try:
            import xgboost  # noqa: F401
        except ImportError:
            self.skipTest("xgboost not installed")

        from src.features.engineering import build_feature_matrix
        from src.models.ml_models import train_and_evaluate

        df = _make_spread_df(300)
        X, y = build_feature_matrix(df, target_horizon=5)
        target_col = next(c for c in y.columns if "return" in c)

        result = train_and_evaluate(X, y[target_col], model_type="xgboost", n_splits=2)
        self.assertEqual(len(result["feature_importance"]), X.shape[1])


class TestComputeMetrics(unittest.TestCase):
    """Tests for compute_metrics()."""

    def test_compute_metrics_regression(self) -> None:
        """RMSE and MAE should be finite and non-negative for regression task."""
        from src.models.ml_models import compute_metrics

        rng = np.random.default_rng(42)
        y_true = rng.normal(0, 1, 100)
        y_pred = y_true + rng.normal(0, 0.1, 100)

        metrics = compute_metrics(y_true, y_pred, task="regression")

        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertGreaterEqual(metrics["rmse"], 0)
        self.assertGreaterEqual(metrics["mae"], 0)
        self.assertTrue(np.isfinite(metrics["rmse"]))

    def test_compute_metrics_perfect_prediction_zero_rmse(self) -> None:
        """Perfect predictions should yield RMSE = 0."""
        from src.models.ml_models import compute_metrics

        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = compute_metrics(y, y, task="regression")
        self.assertAlmostEqual(metrics["rmse"], 0.0, places=10)

    def test_compute_signal_sharpe_positive(self) -> None:
        """A perfectly directional predictor should yield a positive Sharpe."""
        from src.models.ml_models import compute_signal_sharpe

        rng = np.random.default_rng(0)
        y_true = rng.normal(0, 0.01, 252)
        # Perfect signal: predict sign correctly
        y_pred = np.sign(y_true) * 0.5

        sharpe = compute_signal_sharpe(y_true, y_pred)
        self.assertGreater(sharpe, 0)


# ============================================================
# Backtest tests
# ============================================================

class TestBacktestStrategy(unittest.TestCase):
    """Tests for backtest_strategy()."""

    def test_backtest_strategy_returns_correct_columns(self) -> None:
        """backtest_strategy must return the required column set."""
        from src.analysis.leading_indicator import backtest_strategy, compute_spread_signal

        df = _make_spread_df(252)
        signal = compute_spread_signal(df, spread_col="hy_spread")
        bt = backtest_strategy(df, signal, equity_col="sp500_return")

        required_cols = {"signal", "equity_return", "strategy_return",
                         "bh_cumulative", "strategy_cumulative"}
        self.assertTrue(required_cols.issubset(set(bt.columns)))

    def test_backtest_strategy_cumulative_ends_positive(self) -> None:
        """Cumulative return series must stay positive and finish above zero."""
        from src.analysis.leading_indicator import backtest_strategy, compute_spread_signal

        df = _make_spread_df(252)
        signal = compute_spread_signal(df, spread_col="hy_spread")
        bt = backtest_strategy(df, signal, equity_col="sp500_return")

        # cumprod of (1 + daily_return) is always positive for typical daily returns
        self.assertTrue((bt["strategy_cumulative"] > 0).all())
        self.assertTrue((bt["bh_cumulative"] > 0).all())
        # Both series start at 1 + first_daily_return, which should be close to 1
        self.assertAlmostEqual(bt["strategy_cumulative"].iloc[0],
                               1.0 + bt["strategy_return"].iloc[0], places=10)
        self.assertAlmostEqual(bt["bh_cumulative"].iloc[0],
                               1.0 + bt["equity_return"].iloc[0], places=10)

    def test_compute_backtest_metrics_keys(self) -> None:
        """compute_backtest_metrics must return all expected metric keys."""
        from src.analysis.leading_indicator import (
            backtest_strategy,
            compute_backtest_metrics,
            compute_spread_signal,
        )

        df = _make_spread_df(252)
        signal = compute_spread_signal(df, spread_col="hy_spread")
        bt = backtest_strategy(df, signal, equity_col="sp500_return")
        metrics = compute_backtest_metrics(bt)

        expected_keys = {"sharpe", "max_drawdown", "win_rate", "total_return"}
        self.assertTrue(expected_keys.issubset(metrics.keys()))


if __name__ == "__main__":
    unittest.main()
