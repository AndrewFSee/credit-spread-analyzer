"""
Configuration settings for the Credit Spread Analysis & Prediction Platform.

All tuneable parameters, API keys, series identifiers, and model hyper-parameters
live here so that notebooks and scripts never hard-code values.
"""

import os
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
# Optional: without a key, FRED series are downloaded from the public
# ``fredgraph.csv`` endpoint instead of the API.
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

# ---------------------------------------------------------------------------
# Date range defaults
# ---------------------------------------------------------------------------
DEFAULT_START_DATE: str = "1990-01-01"
DEFAULT_END_DATE: str = datetime.today().strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# FRED series to download
# Keys are human-readable names used as DataFrame column names.
# ---------------------------------------------------------------------------
FRED_SERIES: dict[str, str] = {
    # Long-history credit spreads (Moody's seasoned yield minus 10y Treasury), 1986+.
    "baa_spread": "BAA10Y",
    "aaa_spread": "AAA10Y",
    # ICE BofA option-adjusted spreads.  FRED only serves the most recent
    # ~3 years of these series, so they are for monitoring, not model training.
    "hy_spread": "BAMLH0A0HYM2",
    "ig_spread": "BAMLC0A0CM",
    "bbb_spread": "BAMLC0A4CBBB",
    # Rates
    "t10y2y": "T10Y2Y",                 # 10-Year minus 2-Year Treasury spread
    "t10y3m": "T10Y3M",                 # 10-Year minus 3-Month Treasury spread
    "dgs10": "DGS10",                   # 10-Year Treasury yield
    "tbill_3m": "DGS3MO",               # 3-Month Treasury yield (cash return in backtests)
    "fed_funds": "DFF",                 # Federal Funds Effective Rate
    # Macro / financial conditions (lower frequency, published with a delay)
    "dxy": "DTWEXBGS",                  # Broad USD index (2006+)
    "nfci": "NFCI",                     # Chicago Fed National Financial Conditions Index (weekly)
    "claims": "ICSA",                   # Initial jobless claims (weekly)
    "cpi": "CPIAUCSL",                  # Consumer Price Index (monthly)
    "unrate": "UNRATE",                 # Civilian Unemployment Rate (monthly)
}

# FRED reports these in percentage points; the fetcher converts them to basis points.
SPREAD_COLUMNS: list[str] = ["baa_spread", "aaa_spread", "hy_spread", "ig_spread", "bbb_spread"]

# Calendar days between a FRED observation date and its public release.  Series
# listed here are re-dated to their release date before being aligned to the
# trading calendar, so a row never contains data that was not yet published.
RELEASE_LAG_DAYS: dict[str, int] = {
    "cpi": 45,       # month M (dated M-01) is released mid-month M+1
    "unrate": 38,    # month M is released on the first Friday of M+1
    "nfci": 5,       # week ending Friday is released the following Wednesday
    "claims": 5,     # week ending Saturday is released the following Thursday
    "dxy": 7,        # H.10 daily rates are published weekly
    "gz_spread": 75,  # Fed GZ spread for month M (dated M-01) is posted during M+2
    "ebp": 75,
}

# Public CSV sources outside FRED: {column: (url, source column, multiplier)}.
GZ_CSV_URL: str = "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv"
EXTERNAL_CSV_SERIES: dict[str, tuple[str, str, float]] = {
    # Gilchrist–Zakrajšek corporate bond spread and excess bond premium (monthly, 1973+), percent → bps
    "gz_spread": (GZ_CSV_URL, "gz_spread", 100.0),
    "ebp": (GZ_CSV_URL, "ebp", 100.0),
}

# Daily FRED series appear one business day after their observation date.
# Features built from them are shifted by this many rows.
FRED_DAILY_PUBLICATION_LAG: int = 1

# ---------------------------------------------------------------------------
# Yahoo Finance tickers
# ---------------------------------------------------------------------------
YAHOO_TICKERS: dict[str, str] = {
    "sp500": "^GSPC",
    "spy": "SPY",            # dividend-adjusted, used as the total-return equity leg
    "vix": "^VIX",
    "move": "^MOVE",
    "crude_oil": "CL=F",
    "gold": "GC=F",
    "hyg": "HYG",            # high-yield ETF (2007+)
    "iei": "IEI",            # 3-7y Treasury ETF (2007+), duration hedge for HYG
    "ief": "IEF",
    "hy_fund": "VWEHX",      # Vanguard High-Yield Corporate fund NAV (1985+)
    "tsy_fund": "VFITX",     # Vanguard Intermediate-Term Treasury fund NAV (1991+)
}

# Duration-hedged high-yield excess returns: {name: (hy price column, treasury price column, hedge ratio)}.
# Hedge ratios are approximate duration ratios (HY ≈ 4y; IEI ≈ 4.5y; VFITX ≈ 5y).
HY_EXCESS_RETURN_PAIRS: dict[str, tuple[str, str, float]] = {
    "hy_fund_xs": ("hy_fund", "tsy_fund", 0.8),
    "hyg_xs": ("hyg", "iei", 0.85),
}

# Optional licensed spread histories supplied by the user, one CSV per column
# (e.g. data/external/hy_spread.csv with columns: date, value).  They are
# spliced behind the FRED data, which takes precedence where both exist.
SPREAD_HISTORY_DIR: Path = Path(__file__).parent.parent / "data" / "external"

# ---------------------------------------------------------------------------
# Feature engineering parameters
# ---------------------------------------------------------------------------
PRIMARY_SPREAD: str = "baa_spread"
# Minimum observations before a spread column is usable for modelling: FRED
# serves only ~3 years (~750 rows) of the ICE OAS series, so they qualify only
# when a longer licensed history has been spliced in via SPREAD_HISTORY_DIR.
MIN_SPREAD_HISTORY: int = 1500
# Spread behind the tradable high-yield model's core features: the ICE HY OAS
# beat the Baa proxy in both walk-forward periods when a long history exists.
HY_FEATURE_SPREAD_PREFERENCE: tuple[str, ...] = ("hy_spread", "baa_spread")
FEATURE_LAGS: list[int] = [1, 5, 10, 20]
ROLLING_WINDOWS: list[int] = [5, 10, 20, 60]
TARGET_HORIZON: int = 5  # Trading days ahead

# Start of the untouched out-of-sample period used for final model evaluation.
HOLDOUT_START: str = "2019-01-01"

# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------
# Tree models are kept shallow and heavily regularised: with ~250 independent
# 20-day observations per decade, deeper trees memorise crisis episodes.
MODEL_PARAMS: dict[str, dict] = {
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 2,
        "learning_rate": 0.02,
        "subsample": 0.7,
        "colsample_bytree": 0.5,
        "min_child_weight": 250,
        "reg_lambda": 10.0,
        "random_state": 42,
        "n_jobs": -1,
    },
    "lightgbm": {
        "n_estimators": 200,
        "max_depth": 2,
        "num_leaves": 4,
        "learning_rate": 0.02,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.5,
        "min_child_samples": 250,
        "reg_lambda": 10.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 4,
        "min_samples_leaf": 250,
        "max_features": 0.5,
        "random_state": 42,
        "n_jobs": -1,
    },
}

# Model recommended for each forecast horizon, based on walk-forward results
# (see reports/signal_evaluation.md).
RECOMMENDED_MODEL: dict[int, str] = {5: "ensemble", 20: "composite"}
# Tradable high-yield target (HYG vs IEI excess return, 5 days, next-day execution).
RECOMMENDED_MODEL_HY: str = "ensemble"

# Default equity-exposure overlay (see src/analysis/leading_indicator.py).
EXPOSURE_METHOD: str = "vol_regime"
EXPOSURE_TARGET_VOL: float = 0.15

# ---------------------------------------------------------------------------
# Hidden Markov Model parameters
# ---------------------------------------------------------------------------
HMM_N_STATES: int = 3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = Path(__file__).parent.parent / "data"
MODELS_DIR: Path = Path(__file__).parent.parent / "models" / "saved"
