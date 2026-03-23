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
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

# ---------------------------------------------------------------------------
# Date range defaults
# ---------------------------------------------------------------------------
DEFAULT_START_DATE: str = "2000-01-01"
DEFAULT_END_DATE: str = datetime.today().strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# FRED series to download
# Keys are human-readable names used as DataFrame column names.
# ---------------------------------------------------------------------------
FRED_SERIES: dict[str, str] = {
    "hy_spread": "BAMLH0A0HYM2",       # ICE BofA US High-Yield OAS
    "ig_spread": "BAMLC0A0CM",          # ICE BofA US Corporate OAS
    "bbb_spread": "BAMLC0A4CBBB",       # ICE BofA BBB US Corporate OAS
    "t10y2y": "T10Y2Y",                 # 10-Year minus 2-Year Treasury spread
    "fed_funds": "DFF",                 # Federal Funds Effective Rate
    "dxy": "DTWEXBGS",                  # Broad USD index
    "cpi": "CPIAUCSL",                  # Consumer Price Index (All Urban)
    "unrate": "UNRATE",                 # Civilian Unemployment Rate
    "ted_rate": "TEDRATE",              # TED spread (discontinued but historical)
}

# ---------------------------------------------------------------------------
# Yahoo Finance tickers
# ---------------------------------------------------------------------------
YAHOO_TICKERS: dict[str, str] = {
    "sp500": "^GSPC",
    "vix": "^VIX",
    "move": "^MOVE",
    "crude_oil": "CL=F",
    "gold": "GC=F",
}

# ---------------------------------------------------------------------------
# Feature engineering parameters
# ---------------------------------------------------------------------------
FEATURE_LAGS: list[int] = [1, 5, 10, 20]
ROLLING_WINDOWS: list[int] = [5, 10, 20, 60]
TARGET_HORIZON: int = 5  # Business days ahead

# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------
MODEL_PARAMS: dict[str, dict] = {
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    },
    "lightgbm": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_leaf": 10,
        "max_features": 0.5,
        "random_state": 42,
        "n_jobs": -1,
    },
}

# ---------------------------------------------------------------------------
# Hidden Markov Model parameters
# ---------------------------------------------------------------------------
HMM_N_STATES: int = 3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = Path(__file__).parent.parent / "data"
MODELS_DIR: Path = Path(__file__).parent.parent / "models" / "saved"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
