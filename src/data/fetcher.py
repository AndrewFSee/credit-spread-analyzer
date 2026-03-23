"""
Data fetching module for the Credit Spread Analysis & Prediction Platform.

Provides functions to pull data from FRED (via fredapi) and Yahoo Finance
(via yfinance), merge them into a single daily DataFrame, and cache to
Parquet for fast subsequent loads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from fredapi import Fred  # type: ignore
except ImportError:
    Fred = None  # type: ignore

try:
    import yfinance as yf  # type: ignore
except ImportError:
    yf = None  # type: ignore

logger = logging.getLogger(__name__)

# Default series / tickers (can be overridden via arguments)
_FRED_SERIES: dict[str, str] = {
    "hy_spread": "BAMLH0A0HYM2",
    "ig_spread": "BAMLC0A0CM",
    "bbb_spread": "BAMLC0A4CBBB",
    "t10y2y": "T10Y2Y",
    "fed_funds": "DFF",
    "dxy": "DTWEXBGS",
    "cpi": "CPIAUCSL",
    "unrate": "UNRATE",
    "ted_rate": "TEDRATE",
}

_YAHOO_TICKERS: dict[str, str] = {
    "sp500": "^GSPC",
    "vix": "^VIX",
    "move": "^MOVE",
    "crude_oil": "CL=F",
    "gold": "GC=F",
}


def fetch_fred_data(
    api_key: str,
    start_date: str,
    end_date: str,
    series: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Fetch FRED time-series data and return a combined DataFrame.

    Each series is fetched individually; failures are logged and skipped so
    that a partial outage does not abort the entire download.

    Parameters
    ----------
    api_key:
        FRED API key (obtain from https://fred.stlouisfed.org/docs/api/api_key.html).
    start_date:
        ISO-format start date string, e.g. ``"2000-01-01"``.
    end_date:
        ISO-format end date string, e.g. ``"2024-12-31"``.
    series:
        Mapping of ``{column_name: fred_series_id}``.  Defaults to the module-level
        ``_FRED_SERIES`` mapping.

    Returns
    -------
    pd.DataFrame
        Daily DataFrame indexed by date, one column per series.
    """
    if Fred is None:
        raise ImportError("fredapi is required: pip install fredapi")

    if series is None:
        series = _FRED_SERIES

    fred = Fred(api_key=api_key)
    frames: dict[str, pd.Series] = {}

    for name, series_id in series.items():
        try:
            logger.info("Fetching FRED series %s (%s) …", name, series_id)
            s = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            s.name = name
            frames[name] = s
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch FRED series %s: %s", series_id, exc)

    if not frames:
        logger.warning("No FRED data fetched – returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.concat(frames.values(), axis=1)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def fetch_yahoo_data(
    start_date: str,
    end_date: str,
    tickers: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Fetch adjusted-close prices from Yahoo Finance.

    Parameters
    ----------
    start_date:
        ISO-format start date string.
    end_date:
        ISO-format end date string.
    tickers:
        Mapping of ``{column_name: yahoo_ticker_symbol}``.  Defaults to the
        module-level ``_YAHOO_TICKERS`` mapping.

    Returns
    -------
    pd.DataFrame
        Daily DataFrame indexed by date with price and return columns.
    """
    if yf is None:
        raise ImportError("yfinance is required: pip install yfinance")

    if tickers is None:
        tickers = _YAHOO_TICKERS

    frames: dict[str, pd.Series] = {}

    for name, ticker_sym in tickers.items():
        try:
            logger.info("Fetching Yahoo Finance ticker %s (%s) …", name, ticker_sym)
            ticker_obj = yf.Ticker(ticker_sym)
            hist = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
            if hist.empty:
                logger.warning("No data returned for ticker %s.", ticker_sym)
                continue
            price_series = hist["Close"].rename(name)
            price_series.index = pd.to_datetime(price_series.index).tz_localize(None)
            frames[name] = price_series
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch Yahoo ticker %s: %s", ticker_sym, exc)

    if not frames:
        logger.warning("No Yahoo Finance data fetched – returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.concat(frames.values(), axis=1)
    df.sort_index(inplace=True)

    # Compute daily log-returns
    for col in list(frames.keys()):
        df[f"{col}_return"] = np.log(df[col] / df[col].shift(1))

    return df


def fetch_all_data(
    start_date: str,
    end_date: str,
    api_key: str = "",
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Master data-fetch function: pulls FRED + Yahoo, merges, caches to Parquet.

    If a cached Parquet file already exists in *cache_dir* and *force_refresh*
    is ``False``, the cache is loaded instead of hitting the APIs.

    Parameters
    ----------
    start_date:
        ISO-format start date string.
    end_date:
        ISO-format end date string.
    api_key:
        FRED API key.  Can be empty if using a cached file.
    cache_dir:
        Directory to store / read the Parquet cache file.  Defaults to
        ``./data``.
    force_refresh:
        If ``True``, always re-fetch from APIs and overwrite the cache.

    Returns
    -------
    pd.DataFrame
        Merged daily DataFrame with all FRED and Yahoo columns.
    """
    if cache_dir is None:
        cache_dir = Path("data")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"market_data_{start_date}_{end_date}.parquet"

    if cache_file.exists() and not force_refresh:
        logger.info("Loading cached data from %s", cache_file)
        return pd.read_parquet(cache_file)

    logger.info("Fetching fresh data (start=%s, end=%s) …", start_date, end_date)

    fred_df = pd.DataFrame()
    if api_key:
        fred_df = fetch_fred_data(api_key, start_date, end_date)
    else:
        logger.warning("No FRED API key provided – skipping FRED data.")

    yahoo_df = fetch_yahoo_data(start_date, end_date)

    # Merge on date index
    if fred_df.empty and yahoo_df.empty:
        logger.error("Both FRED and Yahoo data are empty.")
        return pd.DataFrame()

    if fred_df.empty:
        merged = yahoo_df
    elif yahoo_df.empty:
        merged = fred_df
    else:
        merged = fred_df.merge(yahoo_df, left_index=True, right_index=True, how="outer")

    merged.sort_index(inplace=True)

    # Forward-fill gaps up to 5 business days (handles weekends + holidays)
    merged = merged.ffill(limit=5)

    logger.info("Saving merged data (%d rows × %d cols) to %s", *merged.shape, cache_file)
    merged.to_parquet(cache_file)

    return merged
