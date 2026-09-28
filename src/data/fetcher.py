"""
Data fetching module for the Credit Spread Analysis & Prediction Platform.

Provides functions to pull data from FRED (via fredapi, or the public CSV
endpoint when no API key is available) and Yahoo Finance (via yfinance),
align them on a trading-day calendar without look-ahead, and cache the
result to Parquet for fast subsequent loads.
"""

from __future__ import annotations

import io
import logging
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import (
    EXTERNAL_CSV_SERIES,
    FRED_SERIES,
    HY_EXCESS_RETURN_PAIRS,
    RELEASE_LAG_DAYS,
    SPREAD_COLUMNS,
    SPREAD_HISTORY_DIR,
    YAHOO_TICKERS,
)

try:
    from fredapi import Fred  # type: ignore
except ImportError:
    Fred = None  # type: ignore

try:
    import yfinance as yf  # type: ignore
except ImportError:
    yf = None  # type: ignore

logger = logging.getLogger(__name__)

_FRED_SERIES: dict[str, str] = FRED_SERIES
_YAHOO_TICKERS: dict[str, str] = YAHOO_TICKERS
_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# Maximum number of trading days a value is carried forward: daily series only
# bridge holidays, while weekly / monthly series persist until the next release.
_DAILY_FFILL_LIMIT = 5
_LOW_FREQ_FFILL_LIMIT = 70
_EQUITY_CALENDAR_COLUMNS = ("sp500", "spy")
_USER_AGENT = "Mozilla/5.0 (compatible; credit-spread-analyzer)"
# Bump whenever the columns produced by fetch_all_data change; older caches are rebuilt.
CACHE_SCHEMA_VERSION = 2


def _http_get(url: str, timeout: int = 60) -> bytes:
    """Download *url*; some hosts (e.g. federalreserve.gov) reject Python's default user agent."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 – configured https URLs
        return resp.read()


def _fetch_fred_csv(series_id: str, start_date: str, end_date: str) -> pd.Series:
    """Download one FRED series from the public CSV endpoint (no API key)."""
    raw = _http_get(_FRED_CSV_URL.format(series_id=series_id))
    s = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
    return s.loc[start_date:end_date].astype(float)


def fetch_fred_data(
    api_key: str,
    start_date: str,
    end_date: str,
    series: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Fetch FRED time-series data and return a combined DataFrame.

    Each series is fetched individually; failures are logged and skipped so
    that a partial outage does not abort the entire download.  Spread series
    listed in ``SPREAD_COLUMNS`` are converted from percent to basis points.

    Parameters
    ----------
    api_key:
        FRED API key.  If empty, the public ``fredgraph.csv`` endpoint is used.
    start_date:
        ISO-format start date string, e.g. ``"2000-01-01"``.
    end_date:
        ISO-format end date string, e.g. ``"2024-12-31"``.
    series:
        Mapping of ``{column_name: fred_series_id}``.  Defaults to
        ``config.settings.FRED_SERIES``.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by observation date, one column per series.
    """
    if series is None:
        series = _FRED_SERIES

    fred = None
    if api_key:
        if Fred is None:
            raise ImportError("fredapi is required when an API key is given: pip install fredapi")
        fred = Fred(api_key=api_key)

    frames: dict[str, pd.Series] = {}
    for name, series_id in series.items():
        try:
            logger.info("Fetching FRED series %s (%s) …", name, series_id)
            if fred is not None:
                s = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            else:
                s = _fetch_fred_csv(series_id, start_date, end_date)
            s = s.astype(float)
            if name in SPREAD_COLUMNS:
                s = s * 100.0  # percent → bps
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


def fetch_external_csv_data(
    start_date: str,
    end_date: str,
    sources: Optional[dict[str, tuple[str, str, float]]] = None,
) -> pd.DataFrame:
    """Fetch public CSV series hosted outside FRED (e.g. the Fed's GZ spread).

    Parameters
    ----------
    sources:
        ``{column: (url, source_column, multiplier)}``.  Defaults to
        ``config.settings.EXTERNAL_CSV_SERIES``.  Each URL is downloaded once.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by observation date; failed sources are skipped.
    """
    if sources is None:
        sources = EXTERNAL_CSV_SERIES

    downloads: dict[str, Optional[pd.DataFrame]] = {}
    frames: dict[str, pd.Series] = {}
    for name, (url, source_col, multiplier) in sources.items():
        if url not in downloads:
            try:
                logger.info("Fetching external CSV %s …", url)
                table = pd.read_csv(io.BytesIO(_http_get(url)))
                table.index = pd.to_datetime(table.iloc[:, 0])
                downloads[url] = table
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not fetch %s: %s", url, exc)
                downloads[url] = None
        table = downloads[url]
        if table is None or source_col not in table.columns:
            continue
        series = pd.to_numeric(table[source_col], errors="coerce") * multiplier
        frames[name] = series.sort_index().loc[start_date:end_date].rename(name)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames.values(), axis=1).sort_index()


def load_local_spread_history(
    directory: Optional[Path] = None,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load user-supplied spread histories (e.g. a licensed ICE HY OAS file).

    Looks for ``<column>.csv`` in *directory* for every spread column.  Each
    file needs a date in the first column and the spread in the second.  Series
    whose median is below 30 are treated as percentages and converted to bps.

    Returns
    -------
    pd.DataFrame
        One column per file found (empty if none).
    """
    directory = Path(directory) if directory is not None else SPREAD_HISTORY_DIR
    columns = columns or SPREAD_COLUMNS
    if not directory.is_dir():
        return pd.DataFrame()

    frames: dict[str, pd.Series] = {}
    for col in columns:
        path = directory / f"{col}.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        values = pd.to_numeric(table.iloc[:, 1], errors="coerce").to_numpy()
        s = pd.Series(values, index=pd.to_datetime(table.iloc[:, 0]), name=col).dropna().sort_index()
        s = s[~s.index.duplicated(keep="last")]
        if s.empty:
            continue
        if s.median() < 30:
            s = s * 100.0  # percent → bps
        logger.info("Loaded local history for %s: %d rows (%s → %s)",
                    col, len(s), s.index.min().date(), s.index.max().date())
        frames[col] = s
    return pd.DataFrame(frames)


def splice_spread_history(fred_df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Extend FRED spread columns with *history*; FRED values win where both exist.

    Logs the median absolute difference over the overlap so that mismatched
    sources (different index versions or units) are easy to spot.
    """
    if history.empty:
        return fred_df
    out = fred_df.copy()
    for col in history.columns:
        hist = history[col].dropna()
        if col in out.columns and out[col].notna().any():
            fred = out[col].dropna()
            overlap = fred.index.intersection(hist.index)
            if len(overlap):
                diff = float((fred.loc[overlap] - hist.loc[overlap]).abs().median())
                level = logging.WARNING if diff > 10 else logging.INFO
                logger.log(level, "%s: median |FRED - local| over %d overlapping days = %.1f bps",
                           col, len(overlap), diff)
            combined = fred.combine_first(hist)
        else:
            combined = hist
        out = out.reindex(out.index.union(combined.index))
        out[col] = combined.reindex(out.index)
    return out.sort_index()


def add_hy_excess_returns(
    df: pd.DataFrame,
    pairs: Optional[dict[str, tuple[str, str, float]]] = None,
) -> pd.DataFrame:
    """Add duration-hedged high-yield excess returns and their cumulative index.

    For each ``name: (hy_col, tsy_col, hedge_ratio)`` this appends
    ``{name}_return = r_hy - hedge_ratio * r_tsy`` and ``{name}_index``, the
    compounded excess-return index (it falls when high-yield spreads widen).
    """
    if pairs is None:
        pairs = HY_EXCESS_RETURN_PAIRS
    new: dict[str, pd.Series] = {}
    for name, (hy_col, tsy_col, ratio) in pairs.items():
        hy_ret, tsy_ret = f"{hy_col}_return", f"{tsy_col}_return"
        if hy_ret not in df.columns or tsy_ret not in df.columns:
            continue
        xs = df[hy_ret] - ratio * df[tsy_ret]
        started = xs.notna().cumsum() > 0
        new[f"{name}_return"] = xs
        new[f"{name}_index"] = (1 + xs.fillna(0.0)).cumprod().where(started)
    if not new:
        return df
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


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
        Mapping of ``{column_name: yahoo_ticker_symbol}``.  Defaults to
        ``config.settings.YAHOO_TICKERS``.

    Returns
    -------
    pd.DataFrame
        Daily DataFrame indexed by date with price and simple-return columns.
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
            idx = pd.to_datetime(price_series.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            price_series.index = idx.normalize()
            frames[name] = price_series
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch Yahoo ticker %s: %s", ticker_sym, exc)

    if not frames:
        logger.warning("No Yahoo Finance data fetched – returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.concat(frames.values(), axis=1)
    df.sort_index(inplace=True)
    return add_returns(df, list(frames.keys()))


def add_returns(df: pd.DataFrame, price_cols: list[str]) -> pd.DataFrame:
    """Append ``{col}_return`` simple daily returns for each price column.

    Returns are computed over consecutive valid prices only, so a missing
    price produces a missing return rather than a stale or duplicated one.
    """
    df = df.copy()
    for col in price_cols:
        if col in df.columns:
            df[f"{col}_return"] = df[col].pct_change(fill_method=None)
    return df


def align_to_trading_days(
    fred_df: pd.DataFrame,
    yahoo_df: pd.DataFrame,
    release_lags: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """Merge FRED and Yahoo data on a trading-day calendar without look-ahead.

    * The calendar is the set of days with an equity-index price (falls back to
      weekdays when no equity data is available).
    * Low-frequency FRED series are re-dated to their public release date using
      *release_lags* and carried forward until the next release.
    * Daily series only carry forward across short gaps (holidays).
    * Returns are recomputed on the aligned prices, never forward-filled.
    """
    if release_lags is None:
        release_lags = RELEASE_LAG_DAYS

    price_cols = [c for c in yahoo_df.columns if not c.endswith("_return")]
    equity_cols = [c for c in _EQUITY_CALENDAR_COLUMNS if c in price_cols]
    if equity_cols:
        calendar = yahoo_df.index[yahoo_df[equity_cols].notna().any(axis=1)]
    elif price_cols:
        calendar = yahoo_df.index
    else:
        calendar = pd.bdate_range(fred_df.index.min(), fred_df.index.max())
    calendar = pd.DatetimeIndex(calendar).unique().sort_values()

    def _align(s: pd.Series, limit: int) -> pd.Series:
        # Reindex onto the union so that values dated on non-trading days roll
        # forward to the next trading day, then keep only trading days.
        s = s.dropna()
        union = calendar.union(s.index)
        return s.reindex(union).ffill(limit=limit).reindex(calendar)

    columns: dict[str, pd.Series] = {}
    for col in fred_df.columns:
        s = fred_df[col].dropna()
        lag = release_lags.get(col)
        if lag:
            s.index = s.index + pd.to_timedelta(lag, unit="D")
        columns[col] = _align(s, _LOW_FREQ_FFILL_LIMIT if lag else _DAILY_FFILL_LIMIT)

    for col in price_cols:
        columns[col] = _align(yahoo_df[col], _DAILY_FFILL_LIMIT)

    merged = pd.DataFrame(columns, index=calendar)
    merged.index.name = "date"
    return add_hy_excess_returns(add_returns(merged, price_cols))


def fetch_all_data(
    start_date: str,
    end_date: str,
    api_key: str = "",
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
    history_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Master data-fetch function: pulls FRED, external CSVs and Yahoo, aligns, caches.

    If a cached Parquet file already exists in *cache_dir* and *force_refresh*
    is ``False``, the cache is loaded instead of hitting the APIs.  Re-run with
    ``force_refresh=True`` after adding files to *history_dir*.

    Parameters
    ----------
    start_date:
        ISO-format start date string.
    end_date:
        ISO-format end date string.
    api_key:
        FRED API key.  Optional – the public CSV endpoint is used without one.
    cache_dir:
        Directory to store / read the Parquet cache file.  Defaults to
        ``./data``.
    force_refresh:
        If ``True``, always re-fetch from APIs and overwrite the cache.
    history_dir:
        Directory with user-supplied spread histories (see
        :func:`load_local_spread_history`).  Defaults to ``data/external``.

    Returns
    -------
    pd.DataFrame
        Trading-day DataFrame with all columns.
    """
    if cache_dir is None:
        cache_dir = Path("data")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"market_data_{start_date}_{end_date}.parquet"

    if cache_file.exists() and not force_refresh:
        cached = pd.read_parquet(cache_file)
        if cached.attrs.get("schema_version") == CACHE_SCHEMA_VERSION:
            logger.info("Loading cached data from %s", cache_file)
            return cached
        logger.info("Cache %s was written by an older version – refreshing.", cache_file)

    logger.info("Fetching fresh data (start=%s, end=%s) …", start_date, end_date)

    fred_df = fetch_fred_data(api_key, start_date, end_date)
    history = load_local_spread_history(history_dir)
    if not history.empty:
        fred_df = splice_spread_history(fred_df, history.loc[start_date:end_date])
    external = fetch_external_csv_data(start_date, end_date)
    if not external.empty:
        fred_df = external if fred_df.empty else fred_df.join(external, how="outer")
    yahoo_df = fetch_yahoo_data(start_date, end_date)

    if fred_df.empty and yahoo_df.empty:
        logger.error("Both FRED and Yahoo data are empty.")
        return pd.DataFrame()

    merged = align_to_trading_days(fred_df, yahoo_df)

    logger.info("Saving merged data (%d rows × %d cols) to %s", *merged.shape, cache_file)
    merged.attrs["schema_version"] = CACHE_SCHEMA_VERSION
    merged.to_parquet(cache_file)

    return merged
