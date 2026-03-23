# Credit Spread Analysis & Prediction Platform

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

A modular, end-to-end platform for analysing credit spreads (HY, IG, BBB) as
leading indicators of economic stress and equity-market regimes.  It combines
macro-economic data from FRED with market data from Yahoo Finance, applies
statistical and machine-learning models, and surfaces insights through an
interactive Streamlit dashboard and five Jupyter notebooks.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Data Sources                         │
│  FRED API (spreads, rates, macro)  Yahoo Finance (equity│
│         prices, VIX, MOVE, commodities)                 │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               src/data/fetcher.py                       │
│    fetch_fred_data → fetch_yahoo_data → merge → cache   │
│                  (Parquet cache on disk)                 │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│            src/features/engineering.py                  │
│   Lags │ Rolling stats │ Momentum │ Z-scores │ Targets  │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌────────────┐  ┌─────────────┐  ┌──────────────────────┐
│ Regime     │  │ Statistical │  │ ML / DL Models       │
│ Detection  │  │ Models      │  │ XGBoost │ LightGBM   │
│ HMM / GMM  │  │ VAR │ IRF   │  │ RandomForest │ LSTM  │
│            │  │ Granger     │  │ Transformer          │
└─────┬──────┘  └──────┬──────┘  └──────────┬───────────┘
      │                │                     │
      └────────────────┴─────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           src/analysis/leading_indicator.py             │
│    Signal generation │ Backtest │ Performance metrics   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│   src/visualization/plots.py  │  src/dashboard/app.py   │
│   Plotly / Matplotlib charts  │  Streamlit dashboard    │
└─────────────────────────────────────────────────────────┘
```

---

## Features

- **Multi-source data ingestion** — FRED API (9 series) + Yahoo Finance (5 tickers) with Parquet caching
- **Rich feature engineering** — lags, rolling statistics, momentum, yield-curve shape, cross-asset ratios, z-scores
- **Regime detection** — Gaussian HMM and GMM with transition matrices and regime-conditional statistics
- **Statistical modelling** — Granger causality, Vector Autoregression, IRF, FEVD, Johansen cointegration
- **ML forecasting** — XGBoost, LightGBM, Random Forest with time-series cross-validation, SHAP explainability
- **Deep learning** — LSTM and Transformer encoder models with early stopping
- **Signal backtesting** — spread-widening signal back-tested against S&P 500 with Sharpe / max-drawdown metrics
- **Interactive dashboard** — 5-tab Streamlit app (Overview, Regimes, Leading Indicator, Forecasting, Correlations)
- **CLI scripts** — `download_data.py`, `train_models.py`, `run_dashboard.py`

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/credit-spread-analyzer.git
cd credit-spread-analyzer

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) install the package in development mode
pip install -e .

# 5. Set your FRED API key (get one at https://fred.stlouisfed.org/docs/api/api_key.html)
export FRED_API_KEY="your_api_key_here"
```

---

## Usage

### Download data

```bash
python scripts/download_data.py \
    --start-date 2000-01-01 \
    --end-date 2024-12-31 \
    --api-key $FRED_API_KEY \
    --output-dir data/
```

### Train a model

```bash
python scripts/train_models.py \
    --data-path data/market_data_2000-01-01_2024-12-31.parquet \
    --model-type xgboost \
    --target-horizon 5 \
    --output-dir models/saved/
```

### Launch the Streamlit dashboard

```bash
python scripts/run_dashboard.py
# or directly:
streamlit run src/dashboard/app.py
```

---

## Module Descriptions

| Module | Description |
|--------|-------------|
| `config/settings.py` | Central configuration: API keys, FRED/Yahoo series, feature/model parameters |
| `src/data/fetcher.py` | FRED + Yahoo Finance data fetching with Parquet caching |
| `src/features/engineering.py` | Feature engineering pipeline (lags, rolling stats, targets) |
| `src/models/regime.py` | HMM and GMM regime detection |
| `src/models/statistical.py` | Granger causality, VAR, IRF, FEVD, Johansen cointegration |
| `src/models/ml_models.py` | XGBoost / LightGBM / Random Forest with CV and SHAP |
| `src/models/dl_models.py` | LSTM and Transformer with PyTorch training loop |
| `src/analysis/leading_indicator.py` | Spread-signal generation and strategy backtesting |
| `src/visualization/plots.py` | Plotly / Matplotlib charting functions |
| `src/dashboard/app.py` | 5-tab Streamlit dashboard |
| `scripts/download_data.py` | CLI: fetch and cache market data |
| `scripts/train_models.py` | CLI: train and save ML models |
| `scripts/run_dashboard.py` | CLI: launch Streamlit dashboard |

---

## Data Sources

| Source | Series / Ticker | Description |
|--------|----------------|-------------|
| [FRED](https://fred.stlouisfed.org) | `BAMLH0A0HYM2` | ICE BofA US High-Yield OAS |
| FRED | `BAMLC0A0CM` | ICE BofA US Corporate OAS (IG) |
| FRED | `BAMLC0A4CBBB` | ICE BofA BBB US Corporate OAS |
| FRED | `T10Y2Y` | 10-Year minus 2-Year Treasury Spread |
| FRED | `DFF` | Federal Funds Effective Rate |
| FRED | `DTWEXBGS` | Broad USD Index (DTWEXBGS) |
| FRED | `CPIAUCSL` | Consumer Price Index |
| FRED | `UNRATE` | Unemployment Rate |
| FRED | `TEDRATE` | TED Spread (historical) |
| [Yahoo Finance](https://finance.yahoo.com) | `^GSPC` | S&P 500 Index |
| Yahoo Finance | `^VIX` | CBOE Volatility Index |
| Yahoo Finance | `^MOVE` | ICE BofA MOVE Index (bond vol) |
| Yahoo Finance | `CL=F` | Crude Oil Futures |
| Yahoo Finance | `GC=F` | Gold Futures |

---

## Configuration

All parameters are in `config/settings.py`:

```python
FRED_API_KEY       # Set via FRED_API_KEY env var
DEFAULT_START_DATE # Default: "2000-01-01"
DEFAULT_END_DATE   # Default: today
FRED_SERIES        # Dict mapping column names to FRED series IDs
YAHOO_TICKERS      # Dict mapping column names to Yahoo ticker symbols
FEATURE_LAGS       # [1, 5, 10, 20]
ROLLING_WINDOWS    # [5, 10, 20, 60]
TARGET_HORIZON     # 5 (business days)
MODEL_PARAMS       # Dict with xgboost / lightgbm / random_forest sub-dicts
HMM_N_STATES       # 3
DATA_DIR           # ./data
MODELS_DIR         # ./models/saved
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_features.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Notebooks

| Notebook | Topic |
|----------|-------|
| `01_data_exploration.ipynb` | Data loading, statistics, distributions, correlations |
| `02_regime_detection.ipynb` | HMM / GMM fitting, transition matrices, regime stats |
| `03_granger_causality.ipynb` | Granger tests, VAR, IRF, FEVD, Johansen cointegration |
| `04_ml_forecasting.ipynb` | Feature matrix, XGBoost/LightGBM/RF training, SHAP |
| `05_deep_learning.ipynb` | LSTM and Transformer training and comparison |

---

## Future Improvements

- **Real-time data** — integrate WebSocket feeds for intraday spread updates
- **Regime-switching forecasts** — combine HMM regime labels with ML predictions
- **Portfolio optimisation** — extend backtest to multi-asset allocation with Kelly sizing
- **Alternative data** — sentiment from news/earnings calls via NLP
- **Hyperparameter search** — Optuna integration for automated model tuning
- **MLflow / W&B tracking** — experiment tracking and model registry
- **Docker deployment** — containerised dashboard for cloud hosting
- **Alert system** — email / Slack notifications on regime shifts

---

## License

MIT – see [LICENSE](LICENSE) for details.
