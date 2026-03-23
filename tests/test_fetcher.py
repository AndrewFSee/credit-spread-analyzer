"""
Tests for src/data/fetcher.py.

All external API calls are mocked so the tests run without credentials.
"""

from __future__ import annotations

import io
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

        # Build a fake series for each FRED code
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
        # At least some columns should be returned
        self.assertGreater(len(result.columns), 0)
        # get_series should have been called once per series
        self.assertEqual(mock_fred_instance.get_series.call_count, len(_FRED_SERIES))

    @patch("src.data.fetcher.Fred")
    def test_fetch_fred_data_handles_individual_series_failure(
        self, MockFred: MagicMock
    ) -> None:
        """Failures on individual series should be logged and skipped, not raised."""
        from src.data.fetcher import fetch_fred_data

        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        good_series = pd.Series(np.ones(5), index=idx)

        mock_fred_instance = MockFred.return_value
        # First call raises, subsequent calls succeed
        mock_fred_instance.get_series.side_effect = [
            Exception("API error"),
            good_series,
            good_series,
            good_series,
            good_series,
            good_series,
            good_series,
            good_series,
            good_series,
        ]

        result = fetch_fred_data("TEST_KEY", "2020-01-01", "2020-01-05")
        # Should not raise; should have partial data
        self.assertIsInstance(result, pd.DataFrame)


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
        # Return columns should also be present
        return_cols = [c for c in result.columns if c.endswith("_return")]
        self.assertGreater(len(return_cols), 0)

    @patch("src.data.fetcher.yf")
    def test_fetch_yahoo_data_handles_empty_response(self, mock_yf: MagicMock) -> None:
        """Empty ticker responses should be skipped gracefully."""
        from src.data.fetcher import fetch_yahoo_data

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()  # empty
        mock_yf.Ticker.return_value = mock_ticker

        result = fetch_yahoo_data(start_date="2020-01-01", end_date="2020-01-10")
        self.assertIsInstance(result, pd.DataFrame)


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

    @patch("src.data.fetcher.fetch_yahoo_data")
    @patch("src.data.fetcher.fetch_fred_data")
    def test_fetch_all_data_merges_correctly(
        self, mock_fred: MagicMock, mock_yahoo: MagicMock, tmp_path: Path
    ) -> None:
        """fetch_all_data should merge FRED and Yahoo DataFrames on the date index."""
        from src.data.fetcher import fetch_all_data

        mock_fred.return_value = self._make_fred_df()
        mock_yahoo.return_value = self._make_yahoo_df()

        result = fetch_all_data(
            start_date="2020-01-01",
            end_date="2020-01-20",
            api_key="TEST_KEY",
            cache_dir=tmp_path,
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("hy_spread", result.columns)
        self.assertIn("sp500", result.columns)

    @patch("src.data.fetcher.fetch_yahoo_data")
    @patch("src.data.fetcher.fetch_fred_data")
    def test_fetch_all_data_uses_cache_when_available(
        self, mock_fred: MagicMock, mock_yahoo: MagicMock, tmp_path: Path
    ) -> None:
        """Second call should load from cache without hitting the APIs."""
        from src.data.fetcher import fetch_all_data

        mock_fred.return_value = self._make_fred_df()
        mock_yahoo.return_value = self._make_yahoo_df()

        # First call: fetches from APIs and writes cache
        fetch_all_data(
            "2020-01-01", "2020-01-20", api_key="TEST_KEY", cache_dir=tmp_path
        )
        first_call_count = mock_fred.call_count

        # Second call: should load from cache
        fetch_all_data(
            "2020-01-01", "2020-01-20", api_key="TEST_KEY", cache_dir=tmp_path
        )
        # fred should NOT have been called again
        self.assertEqual(mock_fred.call_count, first_call_count)


# Allow running with unittest.main() directly
if __name__ == "__main__":
    unittest.main()
