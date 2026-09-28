"""
Tests for src/features/engineering.py.

Uses synthetic DataFrames so no real market data is needed.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _make_df(n: int = 200) -> pd.DataFrame:
    """Create a synthetic DataFrame that mirrors the expected raw data schema."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame(
        {
            "hy_spread": 400 + np.cumsum(rng.normal(0, 5, n)),
            "ig_spread": 120 + np.cumsum(rng.normal(0, 2, n)),
            "bbb_spread": 200 + np.cumsum(rng.normal(0, 3, n)),
            "t10y2y": rng.normal(1.0, 0.8, n),
            "fed_funds": np.clip(2 + np.cumsum(rng.normal(0, 0.02, n)), 0, 8),
            "vix": 18 + np.abs(rng.normal(0, 4, n)),
            "sp500_return": rng.normal(0.0004, 0.01, n),
        },
        index=idx,
    )


def _make_full_df(n: int = 600) -> pd.DataFrame:
    """Synthetic frame with the long-history columns used by the default pipeline."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2000-01-03", periods=n)
    baa = 250 + np.cumsum(rng.normal(0, 2, n))
    return pd.DataFrame(
        {
            "baa_spread": baa,
            "aaa_spread": 0.5 * baa + rng.normal(0, 2, n),
            "dgs10": 4 + np.cumsum(rng.normal(0, 0.03, n)),
            "t10y3m": rng.normal(1.0, 0.5, n),
            "sp500": 1000 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)),
            "vix": 18 + np.abs(rng.normal(0, 4, n)),
            "nfci": rng.normal(0, 0.3, n),
            "claims": 300_000 + rng.normal(0, 10_000, n),
        },
        index=idx,
    )


class TestAddLaggedFeatures(unittest.TestCase):
    """Tests for add_lagged_features()."""

    def test_add_lagged_features_shape(self) -> None:
        """Output should have original cols + len(cols) * len(lags) new columns."""
        from src.features.engineering import add_lagged_features

        df = _make_df(100)
        cols = ["hy_spread", "ig_spread"]
        lags = [1, 5]
        result = add_lagged_features(df, cols, lags)

        expected_extra = len(cols) * len(lags)
        self.assertEqual(result.shape[1], df.shape[1] + expected_extra)

    def test_add_lagged_features_values_shifted(self) -> None:
        """Lag-1 of a column should equal the column shifted by 1."""
        from src.features.engineering import add_lagged_features

        df = _make_df(50)
        result = add_lagged_features(df, ["hy_spread"], [1])
        pd.testing.assert_series_equal(
            result["hy_spread_lag1"].iloc[1:],
            result["hy_spread"].shift(1).iloc[1:],
            check_names=False,
        )

    def test_add_lagged_features_missing_column_is_skipped(self) -> None:
        """Non-existent column names should be silently skipped."""
        from src.features.engineering import add_lagged_features

        df = _make_df(50)
        result = add_lagged_features(df, ["nonexistent_col"], [1, 2])
        # Shape should be unchanged
        self.assertEqual(result.shape, df.shape)


class TestAddRollingStats(unittest.TestCase):
    """Tests for add_rolling_stats()."""

    def test_add_rolling_stats_no_future_leakage(self) -> None:
        """Rolling statistics must only use the window ending at t."""
        from src.features.engineering import add_rolling_stats

        df = _make_df(200)
        result = add_rolling_stats(df, ["hy_spread"], [20])

        for t in range(25, 30):
            manual_mean = df["hy_spread"].iloc[t - 19 : t + 1].mean()
            self.assertAlmostEqual(
                result["hy_spread_rmean20"].iloc[t], manual_mean, places=8
            )

    def test_add_rolling_stats_columns_created(self) -> None:
        """The four stat suffixes (rmean, rstd, rmin, rmax) must all be present."""
        from src.features.engineering import add_rolling_stats

        df = _make_df(100)
        result = add_rolling_stats(df, ["hy_spread"], [10])
        for suffix in ["rmean10", "rstd10", "rmin10", "rmax10"]:
            self.assertIn(f"hy_spread_{suffix}", result.columns)


class TestBuildFeatureMatrix(unittest.TestCase):
    """Tests for build_feature_matrix()."""

    def test_build_feature_matrix_no_nan_in_output(self) -> None:
        """X and y returned by build_feature_matrix must have zero NaN values."""
        from src.features.engineering import build_feature_matrix

        df = _make_df(400)
        X, y = build_feature_matrix(df, target_horizon=5)
        self.assertGreater(len(X), 0)
        self.assertEqual(X.isnull().sum().sum(), 0, "X contains NaN values")
        self.assertEqual(y.isnull().sum().sum(), 0, "y contains NaN values")

    def test_build_feature_matrix_index_aligned(self) -> None:
        """X and y must share the same index."""
        from src.features.engineering import build_feature_matrix

        df = _make_df(400)
        X, y = build_feature_matrix(df, target_horizon=5)
        pd.testing.assert_index_equal(X.index, y.index)

    def test_build_feature_matrix_has_target_column(self) -> None:
        """y should contain the change, return and direction targets."""
        from src.features.engineering import build_feature_matrix

        df = _make_df(400)
        _, y = build_feature_matrix(df, target_horizon=5)
        self.assertEqual(
            set(y.columns), {"target_5d_change", "target_5d_return", "target_5d_up"}
        )

    def test_build_feature_matrix_survives_sparse_columns(self) -> None:
        """A column that only exists for part of the sample must not wipe out all rows."""
        from src.features.engineering import build_feature_matrix

        df = _make_full_df(600)
        df["hy_spread"] = np.nan
        df.iloc[-50:, df.columns.get_loc("hy_spread")] = 300.0  # e.g. ICE data, last 50 days only
        X, _ = build_feature_matrix(df, target_horizon=5)
        self.assertGreater(len(X), 250)

    def test_build_feature_matrix_prefers_baa_spread(self) -> None:
        """The long-history Baa spread is the default target."""
        from src.features.engineering import build_feature_matrix

        df = _make_full_df(600)
        X, y = build_feature_matrix(df, target_horizon=5)
        expected = df["baa_spread"].shift(-5) - df["baa_spread"]
        pd.testing.assert_series_equal(
            y["target_5d_change"], expected.loc[y.index], check_names=False
        )

    def test_features_do_not_use_future_data(self) -> None:
        """Changing data after date t must not change any feature at or before t."""
        from src.features.engineering import build_signal_features

        df = _make_full_df(600)
        cutoff = df.index[450]
        base = build_signal_features(df)

        perturbed = df.copy()
        perturbed.loc[perturbed.index > cutoff] *= 1.5
        after = build_signal_features(perturbed)

        pd.testing.assert_frame_equal(base.loc[:cutoff], after.loc[:cutoff])

    def test_fred_inputs_respect_publication_lag(self) -> None:
        """Features at t must not use the Baa spread dated t (published at t+1)."""
        from src.features.engineering import build_signal_features

        df = _make_full_df(600)
        t = df.index[500]
        base = build_signal_features(df)

        shocked = df.copy()
        shocked.loc[t, "baa_spread"] += 100.0
        after = build_signal_features(shocked)

        pd.testing.assert_series_equal(base.loc[t], after.loc[t])
        self.assertNotAlmostEqual(
            base.loc[df.index[501], "spread_chg_5"], after.loc[df.index[501], "spread_chg_5"]
        )

    def test_equity_features_from_returns_only(self) -> None:
        """Equity features are derived from returns when no price column exists."""
        from src.features.engineering import build_signal_features

        feats = build_signal_features(_make_df(400))
        for col in ("eq_ret_20", "eq_drawdown_252", "eq_vol_20"):
            self.assertIn(col, feats.columns)


class TestCreateTargets(unittest.TestCase):
    """Tests for create_targets()."""

    def test_create_targets_correct_horizon(self) -> None:
        """Forward return at horizon h should equal log(x[t+h] / x[t])."""
        from src.features.engineering import create_targets

        df = _make_df(100)
        result = create_targets(df, target_col="hy_spread", horizons=[5])

        expected = np.log(df["hy_spread"].iloc[15] / df["hy_spread"].iloc[10])
        actual = result["target_5d_return"].iloc[10]
        self.assertAlmostEqual(actual, expected, places=10)

    def test_create_targets_change_in_units(self) -> None:
        """The change target is the plain difference x[t+h] - x[t]."""
        from src.features.engineering import create_targets

        df = _make_df(100)
        result = create_targets(df, target_col="hy_spread", horizons=[5])
        expected = df["hy_spread"].iloc[15] - df["hy_spread"].iloc[10]
        self.assertAlmostEqual(result["target_5d_change"].iloc[10], expected, places=10)

    def test_create_targets_binary_column(self) -> None:
        """Binary up column must contain only 0 and 1."""
        from src.features.engineering import create_targets

        df = _make_df(100)
        result = create_targets(df, target_col="hy_spread", horizons=[5])
        unique = set(result["target_5d_up"].dropna().unique())
        self.assertTrue(unique.issubset({0, 1}))

    def test_create_targets_tail_is_nan(self) -> None:
        """The last h rows of every target column must be NaN (not a false 0 label)."""
        from src.features.engineering import create_targets

        df = _make_df(50)
        result = create_targets(df, target_col="hy_spread", horizons=[3])
        for col in ("target_3d_return", "target_3d_change", "target_3d_up"):
            self.assertTrue(result[col].iloc[-3:].isna().all(), col)


class TestHighYieldFeatures(unittest.TestCase):
    """Tests for the tradable high-yield target and features."""

    def _df(self) -> pd.DataFrame:
        from src.data.fetcher import add_hy_excess_returns

        df = _make_full_df(600)
        rng = np.random.default_rng(7)
        df["hyg_return"] = rng.normal(0.0002, 0.004, len(df))
        df["iei_return"] = rng.normal(0.0001, 0.002, len(df))
        df["gz_spread"] = 150 + np.cumsum(rng.normal(0, 1, len(df)))
        return add_hy_excess_returns(df, pairs={"hyg_xs": ("hyg", "iei", 0.85)})

    def test_return_target_starts_after_execution_lag(self) -> None:
        from src.features.engineering import create_return_targets

        r = pd.Series([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
        y = create_return_targets(r, horizon=2, execution_lag=1)
        # label at t=0 compounds rows 2 and 3
        self.assertAlmostEqual(y["target_2d_lag1_xs_return"].iloc[0], ((1.02 * 1.03) - 1) * 1e4)
        self.assertEqual(y["target_2d_lag1_xs_up"].iloc[0], 1.0)
        self.assertTrue(y.iloc[-3:].isna().all().all())

    def test_hy_feature_matrix(self) -> None:
        from src.features.engineering import build_hy_feature_matrix

        X, y = build_hy_feature_matrix(self._df(), target_horizon=5)
        self.assertGreater(len(X), 100)
        for col in ("hy_xs_ret_20", "hy_xs_vol_20", "hy_xs_drawdown_252", "gz_spread_chg_60", "eq_ret_20"):
            self.assertIn(col, X.columns)
        self.assertEqual(list(y.columns), ["target_5d_lag1_xs_return", "target_5d_lag1_xs_up"])
        self.assertEqual(X.isna().sum().sum(), 0)

    def test_hy_features_do_not_use_future_data(self) -> None:
        from src.features.engineering import build_hy_features

        df = self._df()
        cutoff = df.index[400]
        base = build_hy_features(df)
        changed = df.copy()
        changed.loc[changed.index > cutoff, ["hyg_xs_return", "gz_spread"]] *= 3
        changed.loc[changed.index > cutoff, "hyg_xs_index"] *= 0.5
        pd.testing.assert_frame_equal(base.loc[:cutoff], build_hy_features(changed).loc[:cutoff])

    def test_missing_hy_columns_raise(self) -> None:
        from src.features.engineering import build_hy_feature_matrix

        with self.assertRaises(ValueError):
            build_hy_feature_matrix(_make_full_df(300))

    def test_regime_features_join(self) -> None:
        from src.features.engineering import build_feature_matrix

        df = _make_full_df(600)
        probs = pd.DataFrame({
            "regime_prob_0": 0.7, "regime_prob_1": 0.2, "regime_prob_2": 0.1,
        }, index=df.index)
        X, _ = build_feature_matrix(df, regime_probs=probs)
        self.assertAlmostEqual(X["regime_p_stress"].iloc[-1], 0.1)
        self.assertAlmostEqual(X["regime_level"].iloc[-1], 0.2 + 0.2)


class TestSpreadSelection(unittest.TestCase):
    """Tests for history-aware spread selection."""

    def _df(self, hy_obs: int) -> pd.DataFrame:
        df = _make_full_df(2000)
        df["hy_spread"] = np.nan
        if hy_obs:
            df.iloc[-hy_obs:, df.columns.get_loc("hy_spread")] = 400.0
        return df

    def test_short_ice_history_is_not_usable(self) -> None:
        """FRED's ~3-year ICE window (about 750 rows) must not be picked for modelling."""
        from src.features.engineering import hy_feature_spread_column, spread_columns_with_history

        df = self._df(750)
        self.assertNotIn("hy_spread", spread_columns_with_history(df))
        self.assertEqual(hy_feature_spread_column(df), "baa_spread")

    def test_long_ice_history_is_preferred_for_hy_features(self) -> None:
        from src.features.engineering import hy_feature_spread_column, spread_columns_with_history

        df = self._df(1800)
        self.assertIn("hy_spread", spread_columns_with_history(df))
        self.assertEqual(spread_columns_with_history(df)[0], "baa_spread")  # primary stays first
        self.assertEqual(hy_feature_spread_column(df), "hy_spread")

    def test_hy_matrix_uses_the_selected_spread(self) -> None:
        from src.data.fetcher import add_hy_excess_returns
        from src.features.engineering import build_hy_feature_matrix

        rng = np.random.default_rng(5)
        df = self._df(1800)
        df["hyg_return"] = rng.normal(0.0002, 0.004, len(df))
        df["iei_return"] = rng.normal(0.0001, 0.002, len(df))
        df = add_hy_excess_returns(df, pairs={"hyg_xs": ("hyg", "iei", 0.85)})
        X_auto, _ = build_hy_feature_matrix(df, target_horizon=5)
        X_baa, _ = build_hy_feature_matrix(df, target_horizon=5, spread_col="baa_spread")
        # the ICE history wins automatically, and the two feature sets differ
        self.assertFalse(X_auto["spread_chg_20"].equals(X_baa["spread_chg_20"]))
        self.assertIn("quality_spread_chg_20", X_baa.columns)


class TestHelpers(unittest.TestCase):
    """Tests for small helpers."""

    def test_horizon_from_target_name(self) -> None:
        from src.features.engineering import horizon_from_target_name

        self.assertEqual(horizon_from_target_name("target_20d_change"), 20)
        self.assertEqual(horizon_from_target_name("target_5d_lag1_xs_return"), 6)
        self.assertIsNone(horizon_from_target_name("spread"))

    def test_gold_crude_ratio_ignores_negative_oil(self) -> None:
        from src.features.engineering import add_cross_ratios

        df = pd.DataFrame({"gold": [1700.0, 1700.0], "crude_oil": [20.0, -37.6]})
        out = add_cross_ratios(df)
        self.assertTrue(np.isnan(out["gold_crude_ratio"].iloc[1]))


if __name__ == "__main__":
    unittest.main()
