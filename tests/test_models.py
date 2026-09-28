"""
Tests for regime detection, ML / DL models, statistical tests, and backtests.

Uses small synthetic datasets to keep tests fast and dependency-free.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


def _make_spread_df(n: int = 600) -> pd.DataFrame:
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


def _regime_data(seed: int = 0) -> np.ndarray:
    """Two clearly separated regimes: low spreads first, then high spreads."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(100, 5, 150), rng.normal(300, 5, 150)]).reshape(-1, 1)


def _skip_without(module: str):
    try:
        __import__(module)
        return lambda f: f
    except ImportError:
        return unittest.skip(f"{module} not installed")


# ============================================================
# Regime detection tests
# ============================================================

@_skip_without("hmmlearn")
class TestFitHMM(unittest.TestCase):
    """Tests for fit_hmm()."""

    def test_fit_hmm_returns_labels(self) -> None:
        """fit_hmm + label_regimes should return one label per observation."""
        from src.models.regime import fit_hmm, label_regimes

        df = _make_spread_df(200)
        data = df[["hy_spread"]].values
        model = fit_hmm(data, n_states=3)
        labels = label_regimes(model, data, model_type="hmm")

        self.assertEqual(len(labels), len(data))
        self.assertLessEqual(len(np.unique(labels)), 3)

    def test_fit_hmm_transition_matrix_rows_sum_to_one(self) -> None:
        """Each row of the transition matrix must sum to 1."""
        from src.models.regime import fit_hmm, get_transition_matrix

        df = _make_spread_df(200)
        data = df[["hy_spread"]].values
        model = fit_hmm(data, n_states=3)
        trans = get_transition_matrix(model, model_type="hmm")

        np.testing.assert_allclose(trans.values.sum(axis=1), np.ones(3), atol=1e-6)

    def test_restarts_keep_best_likelihood(self) -> None:
        """With several restarts the returned fit is at least as good as each single run."""
        from src.models.regime import fit_hmm

        data = _make_spread_df(300)[["hy_spread"]].values
        best = fit_hmm(data, n_states=3, n_init=3).score(data)
        singles = [fit_hmm(data, n_states=3, n_init=1, random_state=42 + i).score(data) for i in range(3)]
        self.assertAlmostEqual(best, max(singles), places=6)

    def test_labels_are_ordered_by_mean(self) -> None:
        """Regime 0 must be the low-spread regime regardless of the fitted state order."""
        from src.models.regime import fit_hmm, fit_gmm, label_regimes

        data = _regime_data()
        for model, kind in ((fit_hmm(data, n_states=2), "hmm"), (fit_gmm(data, n_components=2), "gmm")):
            labels = label_regimes(model, data, model_type=kind)
            self.assertEqual(labels[0], 0, kind)
            self.assertEqual(labels[-1], 1, kind)

    def test_transition_matrix_follows_ordered_labels(self) -> None:
        """The transition matrix is permuted consistently with the relabelling."""
        from src.models.regime import fit_hmm, get_transition_matrix

        data = _regime_data()
        model = fit_hmm(data, n_states=2)
        raw = model.transmat_
        order = model.state_order_
        trans = get_transition_matrix(model).values
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(trans[order[i], order[j]], raw[i, j])

    def test_filtered_probabilities_are_causal(self) -> None:
        """Filtered probabilities at t must not change when later data changes."""
        from src.models.regime import filtered_regime_probabilities, fit_hmm

        data = _regime_data()
        model = fit_hmm(data, n_states=2)
        probs = filtered_regime_probabilities(model, data)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)

        altered = data.copy()
        altered[200:] = 100.0
        probs_altered = filtered_regime_probabilities(model, altered)
        np.testing.assert_allclose(probs[:200], probs_altered[:200])
        # And they identify the regimes: early rows are regime 0, late rows regime 1.
        self.assertGreater(probs[10, 0], 0.9)
        self.assertGreater(probs[-10, 1], 0.9)


@_skip_without("hmmlearn")
class TestWalkForwardRegimes(unittest.TestCase):
    """Tests for walk_forward_regime_probabilities()."""

    def _series(self) -> pd.Series:
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2010-01-01", periods=900)
        levels = np.concatenate([rng.normal(100, 3, 400), rng.normal(250, 5, 200), rng.normal(110, 3, 300)])
        return pd.Series(levels, index=idx)

    def test_probabilities_are_causal(self) -> None:
        from src.models.regime import walk_forward_regime_probabilities

        s = self._series()
        probs = walk_forward_regime_probabilities(s, n_states=2, min_train=300, n_init=2)
        self.assertTrue(probs.loc[: s.index[299]].isna().all().all())
        valid = probs.dropna()
        np.testing.assert_allclose(valid.sum(axis=1), 1.0, atol=1e-9)

        cutoff = pd.Timestamp("2012-01-01")
        altered = s.copy()
        altered[altered.index >= cutoff] = 400.0
        probs_altered = walk_forward_regime_probabilities(altered, n_states=2, min_train=300, n_init=2)
        pd.testing.assert_frame_equal(probs.loc[: cutoff - pd.DateOffset(days=1)],
                                      probs_altered.loc[: cutoff - pd.DateOffset(days=1)])

    def test_detects_stress_episode(self) -> None:
        from src.models.regime import walk_forward_regime_probabilities

        s = self._series()
        probs = walk_forward_regime_probabilities(s, n_states=2, start="2011-01-03", min_train=300, n_init=2)
        # The high-spread block (rows 400-599) is in 2011-2012 and is fitted from 2012 on.
        stress = probs["regime_prob_1"]
        self.assertGreater(stress.iloc[450:600].mean(), 0.9)
        self.assertLess(stress.iloc[-100:].mean(), 0.1)

    def test_real_time_helper_applies_publication_lag(self) -> None:
        from src.models import regime

        df = pd.DataFrame({"baa_spread": self._series()})
        captured = {}

        def fake(series, **kwargs):
            captured["series"] = series
            return pd.DataFrame(index=series.index)

        with patch.object(regime, "walk_forward_regime_probabilities", fake):
            regime.real_time_regime_probabilities(df, n_states=2)
        pd.testing.assert_series_equal(captured["series"], df["baa_spread"].shift(1))
        with self.assertRaises(ValueError):
            regime.real_time_regime_probabilities(df, spread_col="nope")


class TestRegimeStats(unittest.TestCase):
    """Tests for compute_regime_stats()."""

    def test_spread_change_not_computed_across_regime_gaps(self) -> None:
        from src.models.regime import compute_regime_stats

        df = pd.DataFrame({"baa_spread": [100.0, 101.0, 500.0, 102.0, 103.0]})
        regimes = np.array([0, 0, 1, 0, 0])
        stats = compute_regime_stats(df, regimes, spread_col="baa_spread")
        # Regime 0 changes on the full series: +1 (row 1), -398 (row 3), +1 (row 4)
        self.assertAlmostEqual(stats.loc[0, "mean_spread_change"], (1 - 398 + 1) / 3)


# ============================================================
# ML model tests
# ============================================================

@_skip_without("xgboost")
class TestTrainAndEvaluateXGBoost(unittest.TestCase):
    """Tests for train_and_evaluate() with XGBoost."""

    def test_train_and_evaluate_xgboost_returns_metrics(self) -> None:
        """train_and_evaluate should return a dict with model and metrics keys."""
        from src.features.engineering import build_feature_matrix
        from src.models.ml_models import train_and_evaluate

        df = _make_spread_df()
        X, y = build_feature_matrix(df, target_horizon=5)

        result = train_and_evaluate(
            X, y["target_5d_change"], model_type="xgboost", task="regression", n_splits=3
        )

        for key in ("model", "mean_metrics", "oof_predictions", "feature_importance"):
            self.assertIn(key, result)
        for key in ("rmse", "oos_r2", "ic", "directional_accuracy", "signal_sharpe"):
            self.assertIn(key, result["mean_metrics"])
        self.assertTrue(np.isfinite(result["mean_metrics"]["rmse"]))
        self.assertGreaterEqual(result["mean_metrics"]["rmse"], 0)

    def test_train_and_evaluate_feature_importance_length(self) -> None:
        """Feature importance Series length must match number of input features."""
        from src.features.engineering import build_feature_matrix
        from src.models.ml_models import train_and_evaluate

        df = _make_spread_df()
        X, y = build_feature_matrix(df, target_horizon=5)
        target_col = next(c for c in y.columns if "return" in c)

        result = train_and_evaluate(X, y[target_col], model_type="xgboost", n_splits=2)
        self.assertEqual(len(result["feature_importance"]), X.shape[1])


class TestModelVariety(unittest.TestCase):
    """Every model type trains and predicts in both tasks it supports."""

    def setUp(self) -> None:
        from src.features.engineering import build_feature_matrix

        self.X, self.y = build_feature_matrix(_make_spread_df(), target_horizon=5)

    def test_all_regression_models(self) -> None:
        from src.models.ml_models import MODEL_TYPES, make_model

        for model_type in MODEL_TYPES:
            if model_type in ("xgboost", "lightgbm"):
                try:
                    __import__(model_type)
                except ImportError:
                    continue
            with self.subTest(model_type=model_type):
                model = make_model(model_type, "regression")
                model.fit(self.X, self.y["target_5d_change"])
                preds = model.predict(self.X)
                self.assertEqual(preds.shape, (len(self.X),))
                self.assertTrue(np.isfinite(preds).all())

    def test_classification_probabilities(self) -> None:
        from src.models.ml_models import make_model

        for model_type in ("ridge", "composite", "random_forest"):
            with self.subTest(model_type=model_type):
                model = make_model(model_type, "classification")
                model.fit(self.X, self.y["target_5d_up"])
                proba = model.predict_proba(self.X)[:, 1]
                self.assertTrue(((proba >= 0) & (proba <= 1)).all())

    def test_ensemble_rejects_classification(self) -> None:
        from src.models.ml_models import make_model

        with self.assertRaises(ValueError):
            make_model("ensemble", "classification")


class TestCompositeFactorModel(unittest.TestCase):
    """Tests for the sign-constrained composite model."""

    def test_signs_are_fixed_and_slope_non_negative(self) -> None:
        from src.models.ml_models import CompositeFactorModel

        rng = np.random.default_rng(0)
        X = pd.DataFrame({"spread_chg_20": rng.normal(size=500), "eq_ret_20": rng.normal(size=500)})

        # Target consistent with the priors → positive slope
        y_good = X["spread_chg_20"] - X["eq_ret_20"] + rng.normal(scale=0.1, size=500)
        model = CompositeFactorModel().fit(X, y_good)
        self.assertGreater(model.beta_, 0)
        self.assertGreater(np.corrcoef(model.predict(X), y_good)[0, 1], 0.9)

        # Target opposite to the priors → slope floored at zero (no sign flip)
        model = CompositeFactorModel().fit(X, -y_good)
        self.assertEqual(model.beta_, 0.0)
        np.testing.assert_array_equal(model.predict(X), 0.0)

    def test_missing_factor_columns_raise(self) -> None:
        from src.models.ml_models import CompositeFactorModel

        with self.assertRaises(ValueError):
            CompositeFactorModel().fit(pd.DataFrame({"x": [1.0, 2.0]}), [0.0, 1.0])


class _RecordingModel:
    """Test double that records the training index it was fitted on."""

    calls: list[pd.Index] = []

    def fit(self, X, y):
        _RecordingModel.calls.append(X.index)
        return self

    def predict(self, X):
        return np.zeros(len(X))


class TestWalkForward(unittest.TestCase):
    """Tests for walk_forward_predict()."""

    def test_training_window_is_purged(self) -> None:
        """Training rows end `gap` rows before each test period starts."""
        from src.models import ml_models

        idx = pd.bdate_range("2015-01-01", "2018-12-31")
        X = pd.DataFrame({"a": np.arange(len(idx), dtype=float)}, index=idx)
        y = pd.Series(np.ones(len(idx)), index=idx, name="target_20d_change")

        _RecordingModel.calls = []
        with patch.object(ml_models, "make_model", lambda *a, **k: _RecordingModel()):
            preds = ml_models.walk_forward_predict(X, y, start="2017-01-01", min_train=10)

        self.assertEqual(len(_RecordingModel.calls), 2)  # 2017 and 2018
        for train_index, year in zip(_RecordingModel.calls, (2017, 2018)):
            first_test = idx.get_loc(idx[idx.year == year][0])
            self.assertEqual(idx.get_loc(train_index[-1]), first_test - 20 - 1)
        self.assertTrue(preds.loc[:"2016-12-31"].isna().all())
        self.assertTrue(preds.loc["2017-01-01":].notna().all())

    def test_gap_inferred_from_target_name(self) -> None:
        from src.models.ml_models import _resolve_gap

        self.assertEqual(_resolve_gap(pd.Series(name="target_5d_up", dtype=float), None), 5)
        self.assertEqual(_resolve_gap(pd.Series(name="whatever", dtype=float), None), 0)
        self.assertEqual(_resolve_gap(pd.Series(name="target_5d_up", dtype=float), 3), 3)


class TestComputeMetrics(unittest.TestCase):
    """Tests for compute_metrics()."""

    def test_compute_metrics_regression(self) -> None:
        """RMSE and MAE should be finite and non-negative for regression task."""
        from src.models.ml_models import compute_metrics

        rng = np.random.default_rng(42)
        y_true = rng.normal(0, 1, 100)
        y_pred = y_true + rng.normal(0, 0.1, 100)

        metrics = compute_metrics(y_true, y_pred, task="regression")

        self.assertGreaterEqual(metrics["rmse"], 0)
        self.assertGreaterEqual(metrics["mae"], 0)
        self.assertTrue(np.isfinite(metrics["rmse"]))
        self.assertGreater(metrics["oos_r2"], 0.9)
        self.assertGreater(metrics["ic"], 0.9)

    def test_compute_metrics_perfect_prediction_zero_rmse(self) -> None:
        """Perfect predictions should yield RMSE = 0."""
        from src.models.ml_models import compute_metrics

        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = compute_metrics(y, y, task="regression")
        self.assertAlmostEqual(metrics["rmse"], 0.0, places=10)
        self.assertAlmostEqual(metrics["oos_r2"], 1.0)

    def test_zero_forecast_has_zero_skill(self) -> None:
        from src.models.ml_models import compute_metrics

        y = np.array([1.0, -2.0, 3.0])
        self.assertAlmostEqual(compute_metrics(y, np.zeros(3))["oos_r2"], 0.0)

    def test_classification_brier_skill(self) -> None:
        from src.models.ml_models import compute_metrics

        y = np.array([0, 1, 0, 1])
        perfect = compute_metrics(y, np.array([0.0, 1.0, 0.0, 1.0]), task="classification")
        base = compute_metrics(y, np.full(4, 0.5), task="classification")
        self.assertAlmostEqual(perfect["brier_skill"], 1.0)
        self.assertAlmostEqual(base["brier_skill"], 0.0)
        self.assertAlmostEqual(perfect["roc_auc"], 1.0)

    def test_compute_signal_sharpe_positive(self) -> None:
        """A perfectly directional predictor should yield a positive Sharpe."""
        from src.models.ml_models import compute_signal_sharpe

        rng = np.random.default_rng(0)
        y_true = rng.normal(0, 0.01, 252)
        y_pred = np.sign(y_true) * 0.5

        self.assertGreater(compute_signal_sharpe(y_true, y_pred), 0)

    def test_signal_sharpe_uses_non_overlapping_periods(self) -> None:
        """With horizon h only every h-th observation counts and annualisation is sqrt(252/h)."""
        from src.models.ml_models import compute_signal_sharpe

        rng = np.random.default_rng(1)
        y_true = rng.normal(0.001, 0.01, 1000)
        y_pred = np.ones(1000)
        sub = y_true[::5]
        expected = sub.mean() / sub.std() * np.sqrt(252 / 5)
        self.assertAlmostEqual(compute_signal_sharpe(y_true, y_pred, horizon=5), expected)


# ============================================================
# Deep-learning tests
# ============================================================

@_skip_without("torch")
class TestDeepLearning(unittest.TestCase):
    """Tests for src/models/dl_models.py."""

    def test_dataset_label_is_last_step(self) -> None:
        from src.models.dl_models import CreditSpreadDataset

        X = np.arange(10, dtype=np.float32).reshape(-1, 1)
        y = np.arange(10, dtype=np.float32) * 10
        ds = CreditSpreadDataset(X, y, seq_len=3)
        self.assertEqual(len(ds), 8)
        x_seq, target = ds[0]
        self.assertEqual(x_seq[-1, 0].item(), 2.0)
        self.assertEqual(target.item(), 20.0)

    def test_train_dl_model_scales_on_training_rows_only(self) -> None:
        from src.models.dl_models import train_dl_model

        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3)).astype(np.float32)
        X[170:] += 100.0  # a shift confined to the validation period
        y = rng.normal(size=200).astype(np.float32)

        result = train_dl_model(X, y, seq_len=5, epochs=2, val_fraction=0.15, gap=5, hidden_size=8)
        # val_start = 170, train_end = 165: the scaler must not see the shifted rows.
        self.assertLess(np.abs(result["scaler"].mean_).max(), 1.0)
        self.assertEqual(result["val_start"], 170)
        self.assertEqual(len(result["predictions"]), 30)
        self.assertIn("oos_r2", result["metrics"])

    def test_transformer_runs(self) -> None:
        from src.models.dl_models import train_dl_model

        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 4)).astype(np.float32)
        y = rng.normal(size=120).astype(np.float32)
        result = train_dl_model(X, y, model_type="transformer", seq_len=5, epochs=1, hidden_size=8)
        self.assertTrue(np.isfinite(result["predictions"]).all())


# ============================================================
# Statistical tests
# ============================================================

class TestStatistical(unittest.TestCase):
    """Tests for src/models/statistical.py."""

    def test_granger_detects_lagged_driver_in_differences(self) -> None:
        from src.models.statistical import run_granger_causality

        rng = np.random.default_rng(0)
        x = np.cumsum(rng.normal(size=600))
        y = np.zeros(600)
        y[1:] = x[:-1]  # y's changes follow x's changes one day later
        y += np.cumsum(rng.normal(scale=0.1, size=600))
        df = pd.DataFrame({"y": y, "x": x})

        pvals = run_granger_causality(df, caused="y", causing="x", maxlag=2)
        self.assertLess(pvals[1], 0.01)

    def test_var_irf_and_fevd(self) -> None:
        from src.models.statistical import compute_irf, compute_variance_decomposition, fit_var_model

        rng = np.random.default_rng(0)
        changes = np.zeros((400, 2))
        for t in range(1, 400):  # AR(1) dynamics in the differences
            changes[t] = 0.5 * changes[t - 1] + [0.0, 0.4 * changes[t - 1, 0]] + rng.normal(size=2)
        df = pd.DataFrame(
            np.cumsum(changes, axis=0), columns=["a", "b"], index=pd.bdate_range("2020-01-01", periods=400)
        )
        result = fit_var_model(df, columns=["a", "b"], maxlags=3)
        self.assertGreaterEqual(result.k_ar, 1)
        self.assertEqual(result.nobs, 399 - result.k_ar)  # fitted on differences
        irf = compute_irf(result, periods=5)
        self.assertEqual(irf.irfs.shape, (6, 2, 2))
        compute_variance_decomposition(result, periods=5)

    def test_granger_rejects_unknown_transform(self) -> None:
        from src.models.statistical import run_granger_causality

        df = pd.DataFrame({"y": np.arange(100.0), "x": np.arange(100.0)})
        with self.assertRaises(ValueError):
            run_granger_causality(df, "y", "x", maxlag=2, transform="log")


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

        required_cols = {"signal", "equity_return", "cash_return", "strategy_return",
                         "bh_cumulative", "strategy_cumulative"}
        self.assertTrue(required_cols.issubset(set(bt.columns)))

    def test_backtest_strategy_cumulative_ends_positive(self) -> None:
        """Cumulative return series must stay positive."""
        from src.analysis.leading_indicator import backtest_strategy, compute_spread_signal

        df = _make_spread_df(252)
        signal = compute_spread_signal(df, spread_col="hy_spread")
        bt = backtest_strategy(df, signal, equity_col="sp500_return")

        self.assertTrue((bt["strategy_cumulative"] > 0).all())
        self.assertTrue((bt["bh_cumulative"] > 0).all())
        self.assertAlmostEqual(bt["strategy_cumulative"].iloc[0],
                               1.0 + bt["strategy_return"].iloc[0], places=10)
        self.assertAlmostEqual(bt["bh_cumulative"].iloc[0],
                               1.0 + bt["equity_return"].iloc[0], places=10)

    def test_signal_is_applied_after_execution_lag(self) -> None:
        """A signal dated t with execution_lag=1 first affects the return on t+2."""
        from src.analysis.leading_indicator import backtest_strategy

        idx = pd.bdate_range("2020-01-01", periods=10)
        df = pd.DataFrame({"eq": np.full(10, 0.01)}, index=idx)
        signal = pd.Series(0.0, index=idx)
        signal.iloc[3] = 1.0

        bt = backtest_strategy(df, signal, equity_col="eq", rf_col=None, risk_free_rate=0.0,
                               execution_lag=1, cost_bps=0.0)
        defensive_days = bt.index[bt["signal"] == 1]
        self.assertEqual(list(defensive_days), [idx[5]])
        self.assertEqual(bt.loc[idx[5], "strategy_return"], 0.0)
        self.assertEqual(bt.loc[idx[4], "strategy_return"], 0.01)

    def test_costs_and_cash_rate(self) -> None:
        """Switching costs are charged per switch and cash earns the lagged T-bill yield."""
        from src.analysis.leading_indicator import backtest_strategy

        idx = pd.bdate_range("2020-01-01", periods=6)
        df = pd.DataFrame({"eq": np.zeros(6), "tbill_3m": np.full(6, 2.52)}, index=idx)
        signal = pd.Series([0, 1, 1, 1, 1, 1], index=idx, dtype=float)
        bt = backtest_strategy(df, signal, equity_col="eq", execution_lag=0, cost_bps=10.0)
        # Position switches on idx[2]; cash return 2.52% / 252 = 1bp per day.
        self.assertAlmostEqual(bt.loc[idx[2], "strategy_return"], 0.0001 - 0.001)
        self.assertAlmostEqual(bt.loc[idx[3], "strategy_return"], 0.0001)

    def test_spread_signal_threshold_in_bps(self) -> None:
        """A 60 bp widening over 20 days triggers the default 50 bp signal (after the publication lag)."""
        from src.analysis.leading_indicator import compute_spread_signal

        idx = pd.bdate_range("2020-01-01", periods=40)
        spread = pd.Series(300.0, index=idx)
        spread.iloc[30:] = 360.0
        signal = compute_spread_signal(pd.DataFrame({"baa_spread": spread}))
        self.assertEqual(signal.iloc[30], 0.0)  # value dated t is not yet published
        self.assertEqual(signal.iloc[31], 1.0)
        self.assertTrue(signal.iloc[:21].isna().all())

    def test_zscore_signal_hysteresis(self) -> None:
        from src.analysis.leading_indicator import compute_zscore_signal

        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2020-01-01", periods=400)
        spread = pd.Series(200 + rng.normal(0, 1, 400), index=idx)
        spread.iloc[300:320] += 20   # stress episode
        spread.iloc[320:] += 0.2     # back near normal but slightly elevated
        df = pd.DataFrame({"baa_spread": spread})

        sig = compute_zscore_signal(df, window=252, enter_threshold=2.0, exit_threshold=-5.0)
        single = compute_zscore_signal(df, window=252, enter_threshold=2.0, exit_threshold=None)
        self.assertEqual(sig.iloc[310], 1.0)
        # With a very low exit threshold the signal stays on after the episode ends.
        self.assertEqual(sig.iloc[-1], 1.0)
        self.assertLess(single.iloc[330:].mean(), 1.0)
        with self.assertRaises(ValueError):
            compute_zscore_signal(df, enter_threshold=0.0, exit_threshold=1.0)

    def test_fractional_allocation_and_rebalance_band(self) -> None:
        from src.analysis.leading_indicator import apply_rebalance_band, backtest_allocation

        target = pd.Series([np.nan, 1.0, 0.95, 0.85, 0.8, 0.0, 0.05, 1.0])
        banded = apply_rebalance_band(target, 0.10)
        np.testing.assert_allclose(banded.values[1:], [1.0, 1.0, 0.85, 0.85, 0.0, 0.0, 1.0])
        self.assertTrue(np.isnan(banded.iloc[0]))

        idx = pd.bdate_range("2020-01-01", periods=5)
        df = pd.DataFrame({"eq": [0.0, 0.0, 0.02, 0.02, 0.02]}, index=idx)
        weight = pd.Series([0.5, 0.5, 0.25, 0.25, 0.25], index=idx)
        bt = backtest_allocation(df, weight, equity_col="eq", rf_col=None, risk_free_rate=0.0,
                                 execution_lag=0, cost_bps=10.0)
        self.assertAlmostEqual(bt.loc[idx[2], "strategy_return"], 0.5 * 0.02)
        self.assertAlmostEqual(bt.loc[idx[3], "strategy_return"], 0.25 * 0.02 - 0.25 * 0.001)
        self.assertAlmostEqual(bt.loc[idx[3], "signal"], 0.75)

    def test_vol_target_weight(self) -> None:
        from src.analysis.leading_indicator import compute_vol_target_weight

        idx = pd.bdate_range("2020-01-01", periods=200)
        rng = np.random.default_rng(0)
        calm = rng.normal(0, 0.005, 100)      # ~8% annualised
        wild = rng.normal(0, 0.03, 100)       # ~48% annualised
        df = pd.DataFrame({"spy_return": np.concatenate([calm, wild])}, index=idx)
        w = compute_vol_target_weight(df, target_vol=0.15, span=20)
        self.assertTrue(w.iloc[:19].isna().all())
        self.assertAlmostEqual(w.iloc[90], 1.0)
        self.assertLess(w.iloc[-1], 0.5)

    def test_vol_regime_weight_uses_supplied_probabilities(self) -> None:
        from src.analysis.leading_indicator import compute_exposure_weight

        df = _make_spread_df(300).rename(columns={"hy_spread": "baa_spread", "sp500_return": "spy_return"})
        probs = pd.DataFrame({"regime_prob_0": 0.6, "regime_prob_1": 0.4}, index=df.index)
        w = compute_exposure_weight(df, method="vol_regime", regime_probs=probs, target_vol=10.0)
        # target_vol is so high that the vol leg is capped at 1: weight = 1 - P(stress)
        np.testing.assert_allclose(w.dropna().values, 0.6)
        with self.assertRaises(ValueError):
            compute_exposure_weight(df, method="nope")

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

        expected_keys = {"sharpe", "max_drawdown", "win_rate", "total_return",
                         "bh_sharpe", "bh_max_drawdown", "time_defensive", "switches_per_year"}
        self.assertTrue(expected_keys.issubset(metrics.keys()))

    def test_run_full_backtest_methods(self) -> None:
        from src.analysis.leading_indicator import run_full_backtest

        df = _make_spread_df(400).rename(columns={"hy_spread": "baa_spread"})
        probs = pd.DataFrame({"regime_prob_0": 0.8, "regime_prob_1": 0.2}, index=df.index)
        for method in ("zscore", "widening", "vol", "regime", "vol_regime"):
            bt, metrics = run_full_backtest(df, method=method, regime_probs=probs)
            self.assertGreater(len(bt), 0)
            self.assertTrue(np.isfinite(metrics["sharpe"]))
        with self.assertRaises(ValueError):
            run_full_backtest(df, method="magic")


if __name__ == "__main__":
    unittest.main()
