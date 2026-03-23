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
        """Rolling statistics must not use future data (shift check)."""
        from src.features.engineering import add_rolling_stats

        df = _make_df(200)
        result = add_rolling_stats(df, ["hy_spread"], [20])

        # At index t, rmean20 must equal the mean of rows [t-19 … t]
        # Equivalently, it must not be higher/lower than possible
        # A simple proxy: mean at t should equal mean of the window ending at t
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

        df = _make_df(300)
        X, y = build_feature_matrix(df, target_horizon=5)
        self.assertEqual(X.isnull().sum().sum(), 0, "X contains NaN values")
        self.assertEqual(y.isnull().sum().sum(), 0, "y contains NaN values")

    def test_build_feature_matrix_index_aligned(self) -> None:
        """X and y must share the same index."""
        from src.features.engineering import build_feature_matrix

        df = _make_df(300)
        X, y = build_feature_matrix(df, target_horizon=5)
        pd.testing.assert_index_equal(X.index, y.index)

    def test_build_feature_matrix_has_target_column(self) -> None:
        """y should contain a column with 'target' in its name."""
        from src.features.engineering import build_feature_matrix

        df = _make_df(300)
        _, y = build_feature_matrix(df, target_horizon=5)
        self.assertTrue(any("target" in c for c in y.columns))


class TestCreateTargets(unittest.TestCase):
    """Tests for create_targets()."""

    def test_create_targets_correct_horizon(self) -> None:
        """Forward return at horizon h should equal log(x[t+h] / x[t])."""
        from src.features.engineering import create_targets

        df = _make_df(100)
        result = create_targets(df, target_col="hy_spread", horizons=[5])

        # Spot-check at row 10
        expected = np.log(df["hy_spread"].iloc[15] / df["hy_spread"].iloc[10])
        actual = result["target_5d_return"].iloc[10]
        self.assertAlmostEqual(actual, expected, places=10)

    def test_create_targets_binary_column(self) -> None:
        """Binary up column must contain only 0 and 1."""
        from src.features.engineering import create_targets

        df = _make_df(100)
        result = create_targets(df, target_col="hy_spread", horizons=[5])
        unique = set(result["target_5d_up"].dropna().unique())
        self.assertTrue(unique.issubset({0, 1}))

    def test_create_targets_tail_is_nan(self) -> None:
        """The last h rows of the target column must be NaN."""
        from src.features.engineering import create_targets

        df = _make_df(50)
        result = create_targets(df, target_col="hy_spread", horizons=[3])
        self.assertTrue(result["target_3d_return"].iloc[-3:].isna().all())


if __name__ == "__main__":
    unittest.main()
