"""
Tests for src/data/fetcher.py.

All external API calls are mocked so the tests run without credentials.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


class TestFetchFredData(unittest.TestCase):
    """Tests for fetch_fred_data()."""

    @patch("src.data.fetcher.Fred")
    def test_fetch_fred_data_returns_dataframe(self, MockFred: MagicMock) -> None:
        """fetch_fred_data should return a DataFrame with expected columns."""
        from src.data.fetcher import _FRED_SERIES, fetch_fred_data

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        fake_series = pd.Series(np.random.rand(10), index=idx)

        mock_fred_instance = MockFred.return_value
        mock_fred_instance.get_series.return_value = fake_series

        result = fetch_fred_data(
            api_key="TEST_KEY",
            start_date="2020-01-01",
            end_date="2020-01-10",
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result.columns), 0)
        self.assertEqual(mock_fred_instance.get_series.call_count, len(_FRED_SERIES))

    @patch("src.data.fetcher.Fred")
    def test_fetch_fred_data_handles_individual_series_failure(
        self, MockFred: MagicMock
    ) -> None:
        """Failures on individual series should be logged and skipped, not raised."""
        from src.data.fetcher import _FRED_SERIES, fetch_fred_data

        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        good_series = pd.Series(np.ones(5), index=idx)
        failing_id = next(iter(_FRED_SERIES.values()))

        def get_series(series_id, **_kwargs):
            if series_id == failing_id:
                raise Exception("API error")
            return good_series

        MockFred.return_value.get_series.side_effect = get_series

        result = fetch_fred_data("TEST_KEY", "2020-01-01", "2020-01-05")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[1], len(_FRED_SERIES) - 1)

    @patch("src.data.fetcher._fetch_fred_csv")
    def test_fetch_fred_data_without_key_uses_csv_and_converts_spreads(
        self, mock_csv: MagicMock
    ) -> None:
        """Without an API key the public CSV endpoint is used; spreads become bps."""
        from src.data.fetcher import fetch_fred_data

        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        mock_csv.return_value = pd.Series([3.5, 3.6, 3.7], index=idx)

        result = fetch_fred_data(
            "", "2020-01-01", "2020-01-03", series={"baa_spread": "BAA10Y", "dgs10": "DGS10"}
        )
        self.assertEqual(mock_csv.call_count, 2)
        np.testing.assert_allclose(result["baa_spread"].values, [350.0, 360.0, 370.0])
        np.testing.assert_allclose(result["dgs10"].values, [3.5, 3.6, 3.7])


class TestFetchYahooData(unittest.TestCase):
    """Tests for fetch_yahoo_data()."""

    @patch("src.data.fetcher.yf")
    def test_fetch_yahoo_data_returns_dataframe(self, mock_yf: MagicMock) -> None:
        """fetch_yahoo_data should return a DataFrame with price and return columns."""
        from src.data.fetcher import fetch_yahoo_data

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        fake_hist = pd.DataFrame({"Close": np.linspace(100, 110, 10)}, index=idx)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = fake_hist
        mock_yf.Ticker.return_value = mock_ticker

        result = fetch_yahoo_data(start_date="2020-01-01", end_date="2020-01-10")

        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result.columns), 0)
        return_cols = [c for c in result.columns if c.endswith("_return")]
        self.assertGreater(len(return_cols), 0)
        # Simple (not log) returns
        self.assertAlmostEqual(result["sp500_return"].iloc[1], fake_hist["Close"].pct_change().iloc[1])

    @patch("src.data.fetcher.yf")
    def test_fetch_yahoo_data_handles_empty_response(self, mock_yf: MagicMock) -> None:
        """Empty ticker responses should be skipped gracefully."""
        from src.data.fetcher import fetch_yahoo_data

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        result = fetch_yahoo_data(start_date="2020-01-01", end_date="2020-01-10")
        self.assertIsInstance(result, pd.DataFrame)


class TestAlignToTradingDays(unittest.TestCase):
    """Tests for align_to_trading_days()."""

    def _yahoo(self) -> pd.DataFrame:
        from src.data.fetcher import add_returns

        idx = pd.bdate_range("2020-01-01", "2020-03-31")
        return add_returns(pd.DataFrame({"sp500": np.linspace(3000, 3300, len(idx))}, index=idx), ["sp500"])

    def test_no_weekend_rows_and_returns_not_duplicated(self) -> None:
        """FRED calendar-day series must not add weekend rows or copy returns forward."""
        from src.data.fetcher import align_to_trading_days

        daily = pd.date_range("2020-01-01", "2020-03-31", freq="D")
        fred = pd.DataFrame({"fed_funds": np.ones(len(daily))}, index=daily)
        out = align_to_trading_days(fred, self._yahoo(), release_lags={})

        self.assertEqual(int((out.index.dayofweek >= 5).sum()), 0)
        self.assertEqual(out["sp500_return"].iloc[1:].duplicated().sum(), 0)

    def test_monthly_series_only_visible_after_release(self) -> None:
        """A value dated 2020-01-01 with a 45-day release lag appears from 2020-02-17."""
        from src.data.fetcher import align_to_trading_days

        fred = pd.DataFrame(
            {"cpi": [100.0, 101.0]}, index=pd.to_datetime(["2020-01-01", "2020-02-01"])
        )
        out = align_to_trading_days(fred, self._yahoo(), release_lags={"cpi": 45})

        self.assertTrue(out.loc[:"2020-02-14", "cpi"].isna().all())
        # 2020-02-15 is a Saturday, so the first trading day with the value is Monday.
        self.assertEqual(out.loc["2020-02-17", "cpi"], 100.0)
        self.assertEqual(out.loc["2020-03-17", "cpi"], 101.0)

    def test_value_dated_on_holiday_rolls_forward(self) -> None:
        """Observations dated on non-trading days are carried to the next trading day."""
        from src.data.fetcher import align_to_trading_days

        fred = pd.DataFrame({"nfci": [0.5]}, index=pd.to_datetime(["2020-01-04"]))  # Saturday
        out = align_to_trading_days(fred, self._yahoo(), release_lags={})
        self.assertEqual(out.loc["2020-01-06", "nfci"], 0.5)
        self.assertTrue(np.isnan(out.loc["2020-01-03", "nfci"]))


class TestExternalSources(unittest.TestCase):
    """Tests for external CSVs, local spread histories and HY excess returns."""

    @patch("src.data.fetcher._http_get")
    def test_external_csv_downloaded_once_and_scaled(self, mock_get: MagicMock) -> None:
        from src.data.fetcher import fetch_external_csv_data

        mock_get.return_value = b"date,gz_spread,ebp\n1/1/2020,1.5,0.2\n2/1/2020,1.7,-0.1\n"
        url = "https://example.org/ebp.csv"
        out = fetch_external_csv_data(
            "2020-01-01", "2020-12-31",
            sources={"gz_spread": (url, "gz_spread", 100.0), "ebp": (url, "ebp", 100.0)},
        )
        self.assertEqual(mock_get.call_count, 1)
        np.testing.assert_allclose(out["gz_spread"].values, [150.0, 170.0])
        np.testing.assert_allclose(out["ebp"].values, [20.0, -10.0])
        self.assertEqual(out.index[1], pd.Timestamp("2020-02-01"))

    @patch("src.data.fetcher._http_get", side_effect=OSError("offline"))
    def test_external_csv_failure_returns_empty(self, _mock_get: MagicMock) -> None:
        from src.data.fetcher import fetch_external_csv_data

        out = fetch_external_csv_data("2020-01-01", "2020-12-31",
                                      sources={"x": ("https://example.org/x.csv", "x", 1.0)})
        self.assertTrue(out.empty)

    def test_local_history_units_and_splice(self) -> None:
        """Percent files become bps; FRED values win on the overlap."""
        from src.data.fetcher import load_local_spread_history, splice_spread_history

        with tempfile.TemporaryDirectory() as tmp_dir:
            pd.DataFrame({
                "DATE": ["2019-12-30", "2019-12-31", "2020-01-02"],
                "BAMLH0A0HYM2": [3.50, 3.55, 9.99],
            }).to_csv(Path(tmp_dir) / "hy_spread.csv", index=False)
            history = load_local_spread_history(Path(tmp_dir))

        np.testing.assert_allclose(history["hy_spread"].values, [350.0, 355.0, 999.0])

        fred = pd.DataFrame(
            {"hy_spread": [360.0, 365.0], "dgs10": [1.9, 1.8]},
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        )
        out = splice_spread_history(fred, history)
        self.assertEqual(list(out.index), list(pd.to_datetime(
            ["2019-12-30", "2019-12-31", "2020-01-02", "2020-01-03"])))
        np.testing.assert_allclose(out["hy_spread"].values, [350.0, 355.0, 360.0, 365.0])
        self.assertTrue(np.isnan(out.loc["2019-12-30", "dgs10"]))

    def test_missing_history_dir_is_empty(self) -> None:
        from src.data.fetcher import load_local_spread_history

        self.assertTrue(load_local_spread_history(Path("does/not/exist")).empty)

    def test_hy_excess_returns(self) -> None:
        from src.data.fetcher import add_hy_excess_returns

        idx = pd.bdate_range("2020-01-01", periods=4)
        df = pd.DataFrame({
            "hyg_return": [np.nan, 0.01, -0.02, 0.005],
            "iei_return": [np.nan, 0.002, 0.004, 0.0],
        }, index=idx)
        out = add_hy_excess_returns(df, pairs={"hyg_xs": ("hyg", "iei", 0.5)})
        np.testing.assert_allclose(out["hyg_xs_return"].values[1:], [0.009, -0.022, 0.005])
        self.assertTrue(np.isnan(out["hyg_xs_index"].iloc[0]))
        self.assertAlmostEqual(out["hyg_xs_index"].iloc[-1], 1.009 * 0.978 * 1.005)
        # Pairs whose inputs are missing are skipped
        self.assertNotIn("x_return", add_hy_excess_returns(df, pairs={"x": ("a", "b", 1.0)}).columns)


class TestFetchAllData(unittest.TestCase):
    """Tests for fetch_all_data()."""

    def _make_fred_df(self) -> pd.DataFrame:
        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        return pd.DataFrame({"hy_spread": np.random.rand(20) * 400 + 300}, index=idx)

    def _make_yahoo_df(self) -> pd.DataFrame:
        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        return pd.DataFrame(
            {"sp500": np.cumprod(1 + np.random.randn(20) * 0.01) * 3000,
             "sp500_return": np.random.randn(20) * 0.01},
            index=idx,
        )

    @patch("src.data.fetcher.fetch_external_csv_data", return_value=pd.DataFrame())
    @patch("src.data.fetcher.fetch_yahoo_data")
    @patch("src.data.fetcher.fetch_fred_data")
    def test_fetch_all_data_merges_correctly(
        self, mock_fred: MagicMock, mock_yahoo: MagicMock, _mock_ext: MagicMock
    ) -> None:
        """fetch_all_data should merge FRED and Yahoo DataFrames on the date index."""
        from src.data.fetcher import fetch_all_data

        mock_fred.return_value = self._make_fred_df()
        mock_yahoo.return_value = self._make_yahoo_df()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = fetch_all_data(
                start_date="2020-01-01",
                end_date="2020-01-20",
                api_key="TEST_KEY",
                cache_dir=Path(tmp_dir),
                history_dir=Path(tmp_dir) / "none",
            )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("hy_spread", result.columns)
        self.assertIn("sp500", result.columns)

    @patch("src.data.fetcher.fetch_external_csv_data", return_value=pd.DataFrame())
    @patch("src.data.fetcher.fetch_yahoo_data")
    @patch("src.data.fetcher.fetch_fred_data")
    def test_fetch_all_data_uses_cache_when_available(
        self, mock_fred: MagicMock, mock_yahoo: MagicMock, _mock_ext: MagicMock
    ) -> None:
        """Second call should load from cache without hitting the APIs."""
        from src.data.fetcher import fetch_all_data

        mock_fred.return_value = self._make_fred_df()
        mock_yahoo.return_value = self._make_yahoo_df()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            fetch_all_data(
                "2020-01-01", "2020-01-20", api_key="TEST_KEY", cache_dir=tmp_path,
                history_dir=tmp_path / "none",
            )
            first_call_count = mock_fred.call_count

            fetch_all_data(
                "2020-01-01", "2020-01-20", api_key="TEST_KEY", cache_dir=tmp_path,
                history_dir=tmp_path / "none",
            )
            self.assertEqual(mock_fred.call_count, first_call_count)

    @patch("src.data.fetcher.fetch_external_csv_data", return_value=pd.DataFrame())
    @patch("src.data.fetcher.fetch_yahoo_data")
    @patch("src.data.fetcher.fetch_fred_data")
    def test_outdated_cache_is_rebuilt(
        self, mock_fred: MagicMock, mock_yahoo: MagicMock, _mock_ext: MagicMock
    ) -> None:
        """A cache written without the current schema version is refreshed."""
        from src.data.fetcher import CACHE_SCHEMA_VERSION, fetch_all_data

        mock_fred.return_value = self._make_fred_df()
        mock_yahoo.return_value = self._make_yahoo_df()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cache_file = tmp_path / "market_data_2020-01-01_2020-01-20.parquet"
            pd.DataFrame({"old_column": [1.0]}, index=pd.to_datetime(["2020-01-01"])).to_parquet(cache_file)

            result = fetch_all_data(
                "2020-01-01", "2020-01-20", api_key="TEST_KEY", cache_dir=tmp_path,
                history_dir=tmp_path / "none",
            )
            self.assertEqual(mock_fred.call_count, 1)
            self.assertIn("hy_spread", result.columns)
            self.assertEqual(pd.read_parquet(cache_file).attrs.get("schema_version"), CACHE_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
