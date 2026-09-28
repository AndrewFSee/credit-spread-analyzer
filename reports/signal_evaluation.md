# Signal evaluation

Data: `market_data_1990-01-01_2026-09-15.parquet` (1990-01-02 → 2026-09-14).  
Development period: 2000-01-01 → 2018-12-31 (2011-01-01 → for HYG); holdout: 2019-01-01 → 2026-09-14.

All predictions are out-of-sample: models are refitted yearly on data whose labels were fully realised before the prediction year.

* `oos_r2` – 1 − SSE / SSE(zero forecast); > 0 means the forecast beats "no change".
* `ic` – Spearman rank correlation of forecast and outcome.
* `hit_rate` – share of correct direction calls.
* `signal_sharpe` – annualised Sharpe of trading the sign of the forecast, non-overlapping periods.
* `years_ic>0` / `years_auc>0.5` – calendar years in which the signal had the right sign.
* `brier_skill` – improvement in Brier score over the expanding historical base rate.

# 1. Moody's Baa – 10y spread forecasts

Target: change in `baa_spread` (bps) over the next *h* trading days.

## 5-day spread change – regression

| period | model | n | oos_r2 | ic | hit_rate | signal_sharpe | years_ic>0 |
|---|---|---|---|---|---|---|---|
| development | composite | 4779 | 0.112 | 0.262 | 0.544 | 1.811 | 16/19 |
| development | ridge | 4779 | 0.110 | 0.250 | 0.543 | 1.221 | 16/19 |
| development | lightgbm | 4779 | 0.086 | 0.231 | 0.535 | 1.266 | 14/19 |
| development | xgboost | 4779 | 0.085 | 0.228 | 0.531 | 1.352 | 14/19 |
| development | random_forest | 4779 | 0.082 | 0.216 | 0.527 | 1.329 | 16/19 |
| development | ensemble | 4779 | 0.113 | 0.261 | 0.546 | 1.404 | 16/19 |
| holdout | composite | 1930 | 0.068 | 0.159 | 0.534 | 1.019 | 8/8 |
| holdout | ridge | 1930 | 0.009 | 0.208 | 0.566 | 1.479 | 8/8 |
| holdout | lightgbm | 1930 | 0.032 | 0.173 | 0.538 | 0.994 | 8/8 |
| holdout | xgboost | 1930 | 0.028 | 0.168 | 0.537 | 1.027 | 8/8 |
| holdout | random_forest | 1930 | 0.021 | 0.167 | 0.539 | 1.010 | 7/8 |
| holdout | ensemble | 1930 | 0.046 | 0.215 | 0.548 | 1.106 | 8/8 |

## 5-day spread widening (up/down) – classification

| period | model | n | base_rate | auc | brier_skill | years_auc>0.5 |
|---|---|---|---|---|---|---|
| development | composite | 4779 | 0.447 | 0.621 | 0.046 | 17/19 |
| development | ridge | 4779 | 0.447 | 0.600 | 0.029 | 17/19 |
| development | lightgbm | 4779 | 0.447 | 0.609 | 0.037 | 18/19 |
| holdout | composite | 1930 | 0.407 | 0.578 | 0.013 | 7/8 |
| holdout | ridge | 1930 | 0.407 | 0.578 | 0.001 | 6/8 |
| holdout | lightgbm | 1930 | 0.407 | 0.573 | 0.016 | 6/8 |

LSTM (holdout, h=5, single seed – results vary with the seed): oos_r2 0.102, ic 0.186, hit_rate 0.548, signal_sharpe 1.19

## 20-day spread change – regression

| period | model | n | oos_r2 | ic | hit_rate | signal_sharpe | years_ic>0 |
|---|---|---|---|---|---|---|---|
| development | composite | 4779 | 0.115 | 0.275 | 0.581 | 0.936 | 13/19 |
| development | ridge | 4779 | 0.020 | 0.195 | 0.539 | 0.465 | 13/19 |
| development | lightgbm | 4779 | 0.077 | 0.235 | 0.555 | 0.661 | 14/19 |
| development | xgboost | 4779 | 0.072 | 0.233 | 0.556 | 0.596 | 14/19 |
| development | random_forest | 4779 | 0.072 | 0.208 | 0.555 | 0.595 | 11/19 |
| development | ensemble | 4779 | 0.074 | 0.234 | 0.552 | 0.486 | 14/19 |
| holdout | composite | 1915 | 0.001 | 0.131 | 0.539 | 0.442 | 5/8 |
| holdout | ridge | 1915 | -0.336 | 0.178 | 0.557 | -0.207 | 6/8 |
| holdout | lightgbm | 1915 | -0.127 | 0.146 | 0.542 | 0.165 | 6/8 |
| holdout | xgboost | 1915 | -0.118 | 0.154 | 0.546 | 0.165 | 6/8 |
| holdout | random_forest | 1915 | -0.088 | 0.167 | 0.552 | 0.259 | 6/8 |
| holdout | ensemble | 1915 | -0.182 | 0.160 | 0.548 | -0.056 | 6/8 |

## 20-day spread widening (up/down) – classification

| period | model | n | base_rate | auc | brier_skill | years_auc>0.5 |
|---|---|---|---|---|---|---|
| development | composite | 4779 | 0.476 | 0.613 | 0.048 | 12/19 |
| development | ridge | 4779 | 0.476 | 0.536 | -0.086 | 13/19 |
| development | lightgbm | 4779 | 0.476 | 0.553 | 0.004 | 12/19 |
| holdout | composite | 1915 | 0.403 | 0.555 | -0.000 | 5/8 |
| holdout | ridge | 1915 | 0.403 | 0.530 | -0.059 | 5/8 |
| holdout | lightgbm | 1915 | 0.403 | 0.531 | -0.011 | 4/8 |

LSTM (holdout, h=20, single seed – results vary with the seed): oos_r2 -0.022, ic 0.074, hit_rate 0.544, signal_sharpe -0.23

# 2. How much of the spread forecast is tradable?

Correlation between each series' daily move on day *t + k* and the SPY return on day *t*.  Non-zero values at k ≥ 1 mean the series reacts to equity moves with a delay (stale or smoothed pricing), which makes it partly predictable without being tradable.

| series | from | k=0 | k=1 | k=2 | k=3 |
|---|---|---|---|---|---|
| Moody's Baa – 10y (Δ) | 1993-02-01 | -0.189 | -0.099 | -0.085 | -0.075 |
| ICE HY OAS (Δ) | 1997-01-02 | -0.425 | -0.243 | -0.099 | -0.054 |
| ICE BBB OAS (Δ) | 2023-09-27 | -0.435 | -0.274 | -0.023 | -0.019 |
| Vanguard HY fund excess return | 1993-02-01 | 0.383 | 0.179 | 0.124 | 0.071 |
| HYG excess return | 2007-04-12 | 0.717 | 0.000 | 0.015 | -0.021 |

Rank IC of the Baa ensemble's 5-day forecast (holdout walk-forward) against other targets (the ICE columns start when FRED's own window does, unless a longer history was spliced in):

| target | from | n | ic |
|---|---|---|---|
| Moody's Baa – 10y (the training target) | 2019-01-02 | 1930 | 0.215 |
| ICE BofA HY OAS | 2019-01-02 | 1930 | 0.075 |
| ICE BofA BBB OAS | 2023-09-26 | 739 | 0.196 |
| ICE BofA IG OAS | 2023-09-26 | 739 | 0.164 |
| HYG vs IEI excess return (sign flipped) | 2019-01-02 | 1929 | 0.043 |
| HYG vs IEI excess return (sign flipped) | 2019-01-02 | 1929 | 0.043 |

# 3. Tradable high yield: HYG vs IEI excess return

Target: 5-day return of HYG minus 0.85 × IEI (bps), starting the day after the signal (`target_5d_lag1_xs_return`).  Features: the core set plus HY excess-return momentum, volatility and drawdown, and Gilchrist–Zakrajšek spread changes.  The composite uses the same economic signs, reversed for a return target.

| period | model | n | oos_r2 | ic | hit_rate | signal_sharpe | years_ic>0 |
|---|---|---|---|---|---|---|---|
| development | composite | 2012 | -0.025 | -0.125 | 0.475 | -0.323 | 0/8 |
| development | ridge | 2012 | -0.239 | 0.146 | 0.519 | 0.550 | 7/8 |
| development | lightgbm | 2012 | 0.005 | 0.056 | 0.501 | 0.637 | 8/8 |
| development | ensemble | 2012 | -0.003 | 0.152 | 0.528 | 0.727 | 8/8 |
| holdout | composite | 1929 | -0.002 | -0.055 | 0.517 | -0.228 | 2/8 |
| holdout | ridge | 1929 | -0.060 | 0.165 | 0.552 | 1.124 | 8/8 |
| holdout | lightgbm | 1929 | -0.006 | 0.134 | 0.571 | 0.788 | 7/8 |
| holdout | ensemble | 1929 | 0.024 | 0.177 | 0.554 | 1.034 | 8/8 |

# 4. Regime-conditional forecasts

Walk-forward HMM probabilities (3 states on the Baa spread, refitted yearly on past data, forward-filtered) added as features (`regime_p_stress`, `regime_level`).  Rows are restricted to dates where the probabilities exist, so the baseline differs slightly from section 1.

| period | model | n | oos_r2 | ic | hit_rate | signal_sharpe | years_ic>0 |
|---|---|---|---|---|---|---|---|
| development | Baa 5d change · ensemble without regimes | 4779 | 0.111 | 0.251 | 0.545 | 1.422 | 16/19 |
| holdout | Baa 5d change · ensemble without regimes | 1930 | 0.061 | 0.220 | 0.550 | 1.056 | 8/8 |
| development | Baa 5d change · ensemble with regimes | 4779 | 0.110 | 0.244 | 0.546 | 1.441 | 15/19 |
| holdout | Baa 5d change · ensemble with regimes | 1930 | 0.057 | 0.227 | 0.556 | 1.262 | 8/8 |
| development | HYG 5d excess return · ensemble without regimes | 2012 | -0.003 | 0.152 | 0.528 | 0.727 | 8/8 |
| holdout | HYG 5d excess return · ensemble without regimes | 1929 | 0.024 | 0.177 | 0.554 | 1.034 | 8/8 |
| development | HYG 5d excess return · ensemble with regimes | 2012 | -0.116 | 0.141 | 0.523 | 0.490 | 8/8 |
| holdout | HYG 5d excess return · ensemble with regimes | 1929 | 0.024 | 0.174 | 0.556 | 0.991 | 8/8 |

# 5. Equity exposure overlays (SPY vs 3-month T-bills)

Weights use data published by each close and trade at the next close.  Fractional overlays only rebalance when the target moves by more than 10 percentage points; costs are 5 bp per unit of turnover.  These overlays were compared on both periods at the same time, so the holdout is not untouched for this particular choice.

| period | overlay | cagr | vol | sharpe | max_dd | avg_equity | turnover/yr |
|---|---|---|---|---|---|---|---|
| development | vol target 15% × (1 − P(stress)) | 6.0% | 9.0% | 0.510 | -14.9% | 54.6% | 0.946 |
| development | vol target 15% only | 5.1% | 13.0% | 0.323 | -42.2% | 86.1% | 1.972 |
| development | 1 − P(stress) only | 6.4% | 10.9% | 0.471 | -19.3% | 59.4% | 0.469 |
| development | z-score in/out (enter 0.5 / exit 0) | 7.0% | 10.8% | 0.532 | -23.6% | 60.2% | 1.318 |
| development | 20d widening > 50bp in/out | 5.5% | 17.7% | 0.300 | -50.2% | 98.4% | 1.266 |
| development | buy & hold SPY | 4.8% | 19.2% | 0.254 | -55.2% | 100.0% | 0.000 |
| holdout | vol target 15% × (1 − P(stress)) | 13.4% | 13.0% | 0.815 | -18.1% | 84.5% | 1.663 |
| holdout | vol target 15% only | 14.0% | 13.3% | 0.839 | -18.2% | 87.6% | 1.893 |
| holdout | 1 − P(stress) only | 15.2% | 18.7% | 0.696 | -33.7% | 99.3% | 0.260 |
| holdout | z-score in/out (enter 0.5 / exit 0) | 8.4% | 11.3% | 0.519 | -20.8% | 61.4% | 3.516 |
| holdout | 20d widening > 50bp in/out | 16.7% | 17.1% | 0.820 | -24.5% | 98.8% | 0.260 |
| holdout | buy & hold SPY | 17.3% | 19.3% | 0.777 | -33.7% | 100.0% | 0.000 |

Sharpe ratio by sub-period:

| overlay | 2000–2004 | 2005–2009 | 2010–2014 | 2015–2018 | 2019–2022 | 2023–2026 |
|---|---|---|---|---|---|---|
| vol target 15% × (1 − P(stress)) | 0.488 | 0.344 | 0.705 | 0.521 | 0.687 | 0.974 |
| vol target 15% only | -0.304 | 0.101 | 1.128 | 0.510 | 0.736 | 0.962 |
| 1 − P(stress) only | 0.358 | 0.350 | 0.629 | 0.514 | 0.462 | 1.093 |
| z-score in/out (enter 0.5 / exit 0) | 0.302 | 0.642 | 0.527 | 0.836 | 0.287 | 0.797 |
| 20d widening > 50bp in/out | -0.198 | 0.146 | 1.003 | 0.514 | 0.631 | 1.093 |
| buy & hold SPY | -0.144 | 0.019 | 0.975 | 0.514 | 0.608 | 1.093 |

# 6. Long ICE high-yield history vs the Moody's Baa proxy

`hy_spread` holds 7472 observations (1996-12-31 → 2026-09-14), so a licensed history has been spliced in behind FRED's three-year window.  Both spreads are compared on the dates where each has walk-forward predictions.

## As a forecast target

| horizon | period | model | n | oos_r2 | ic | hit_rate | signal_sharpe | years_ic>0 |
|---|---|---|---|---|---|---|---|---|
| 5d | development | hy_spread · composite | 4279 | 0.066 | 0.148 | 0.540 | 1.231 | 8/17 |
| 5d | development | hy_spread · ridge | 4279 | 0.079 | 0.203 | 0.546 | 1.159 | 14/17 |
| 5d | development | hy_spread · lightgbm | 4279 | 0.145 | 0.223 | 0.555 | 1.438 | 13/17 |
| 5d | development | hy_spread · ensemble | 4279 | 0.140 | 0.235 | 0.562 | 1.421 | 14/17 |
| 5d | development | baa_spread · composite | 4279 | 0.128 | 0.274 | 0.545 | 1.897 | 14/17 |
| 5d | development | baa_spread · ridge | 4279 | 0.128 | 0.267 | 0.543 | 1.292 | 14/17 |
| 5d | development | baa_spread · lightgbm | 4279 | 0.107 | 0.258 | 0.538 | 1.384 | 14/17 |
| 5d | development | baa_spread · ensemble | 4279 | 0.133 | 0.286 | 0.549 | 1.533 | 15/17 |
| 5d | holdout | hy_spread · composite | 1930 | 0.007 | -0.004 | 0.511 | 0.128 | 3/8 |
| 5d | holdout | hy_spread · ridge | 1930 | 0.029 | 0.149 | 0.563 | 0.917 | 6/8 |
| 5d | holdout | hy_spread · lightgbm | 1930 | -0.010 | 0.147 | 0.569 | 0.770 | 8/8 |
| 5d | holdout | hy_spread · ensemble | 1930 | 0.026 | 0.155 | 0.564 | 0.824 | 8/8 |
| 5d | holdout | baa_spread · composite | 1930 | 0.068 | 0.159 | 0.534 | 1.019 | 8/8 |
| 5d | holdout | baa_spread · ridge | 1930 | 0.009 | 0.208 | 0.566 | 1.479 | 8/8 |
| 5d | holdout | baa_spread · lightgbm | 1930 | 0.032 | 0.173 | 0.538 | 0.994 | 8/8 |
| 5d | holdout | baa_spread · ensemble | 1930 | 0.046 | 0.215 | 0.548 | 1.106 | 8/8 |
| 20d | development | hy_spread · composite | 4279 | 0.044 | 0.118 | 0.538 | 0.710 | 7/17 |
| 20d | development | hy_spread · ridge | 4279 | -0.041 | 0.165 | 0.532 | 0.681 | 11/17 |
| 20d | development | hy_spread · lightgbm | 4279 | 0.093 | 0.219 | 0.545 | 0.733 | 13/17 |
| 20d | development | hy_spread · ensemble | 4279 | 0.080 | 0.219 | 0.543 | 0.810 | 10/17 |
| 20d | development | baa_spread · composite | 4279 | 0.127 | 0.281 | 0.581 | 0.966 | 12/17 |
| 20d | development | baa_spread · ridge | 4279 | 0.026 | 0.200 | 0.535 | 0.418 | 12/17 |
| 20d | development | baa_spread · lightgbm | 4279 | 0.080 | 0.230 | 0.551 | 0.646 | 12/17 |
| 20d | development | baa_spread · ensemble | 4279 | 0.075 | 0.234 | 0.548 | 0.423 | 13/17 |
| 20d | holdout | hy_spread · composite | 1915 | -0.027 | -0.044 | 0.500 | -0.192 | 3/8 |
| 20d | holdout | hy_spread · ridge | 1915 | -0.450 | 0.046 | 0.502 | 0.032 | 3/8 |
| 20d | holdout | hy_spread · lightgbm | 1915 | -0.207 | 0.036 | 0.528 | -0.200 | 3/8 |
| 20d | holdout | hy_spread · ensemble | 1915 | -0.276 | 0.042 | 0.514 | 0.102 | 3/8 |
| 20d | holdout | baa_spread · composite | 1915 | 0.001 | 0.131 | 0.539 | 0.442 | 5/8 |
| 20d | holdout | baa_spread · ridge | 1915 | -0.336 | 0.178 | 0.557 | -0.207 | 6/8 |
| 20d | holdout | baa_spread · lightgbm | 1915 | -0.127 | 0.146 | 0.542 | 0.165 | 6/8 |
| 20d | holdout | baa_spread · ensemble | 1915 | -0.182 | 0.160 | 0.548 | -0.056 | 6/8 |

## As the feature spread for the tradable HYG model

| period | model | n | oos_r2 | ic | hit_rate | signal_sharpe | years_ic>0 |
|---|---|---|---|---|---|---|---|
| development | features on hy_spread · ridge | 2012 | -0.239 | 0.146 | 0.519 | 0.550 | 7/8 |
| holdout | features on hy_spread · ridge | 1929 | -0.060 | 0.165 | 0.552 | 1.124 | 8/8 |
| development | features on hy_spread · ensemble | 2012 | -0.003 | 0.152 | 0.528 | 0.727 | 8/8 |
| holdout | features on hy_spread · ensemble | 1929 | 0.024 | 0.177 | 0.554 | 1.034 | 8/8 |
| development | features on baa_spread · ridge | 2012 | -0.149 | 0.109 | 0.515 | 0.706 | 7/8 |
| holdout | features on baa_spread · ridge | 1929 | -0.210 | 0.151 | 0.548 | 0.551 | 6/8 |
| development | features on baa_spread · ensemble | 2012 | 0.008 | 0.129 | 0.528 | 0.899 | 8/8 |
| holdout | features on baa_spread · ensemble | 1929 | -0.018 | 0.163 | 0.561 | 0.674 | 8/8 |

## As the driver of the exposure overlay (from 2001, after the HY regime warm-up)

| period | spread | overlay | cagr | sharpe | max_dd | avg_equity | turnover/yr | bh_sharpe |
|---|---|---|---|---|---|---|---|---|
| development | hy_spread | vol target 15% × (1 − P(stress)) | 8.0% | 0.666 | -14.9% | 70.5% | 1.019 | 0.311 |
| holdout | hy_spread | vol target 15% × (1 − P(stress)) | 12.6% | 0.761 | -18.9% | 85.2% | 1.748 | 0.777 |
| development | hy_spread | 1 − P(stress) only | 8.8% | 0.633 | -19.0% | 76.8% | 0.166 | 0.311 |
| holdout | hy_spread | 1 − P(stress) only | 13.0% | 0.622 | -31.2% | 94.4% | 0.254 | 0.777 |
| development | hy_spread | z-score in/out (enter 0.5 / exit 0) | 7.1% | 0.558 | -30.0% | 65.6% | 1.781 | 0.311 |
| holdout | hy_spread | z-score in/out (enter 0.5 / exit 0) | 10.0% | 0.673 | -12.8% | 68.6% | 4.037 | 0.777 |
| development | baa_spread | vol target 15% × (1 − P(stress)) | 6.0% | 0.533 | -14.9% | 57.2% | 0.886 | 0.311 |
| holdout | baa_spread | vol target 15% × (1 − P(stress)) | 13.4% | 0.815 | -18.1% | 84.5% | 1.663 | 0.777 |
| development | baa_spread | 1 − P(stress) only | 6.5% | 0.507 | -19.3% | 61.9% | 0.385 | 0.311 |
| holdout | baa_spread | 1 − P(stress) only | 15.2% | 0.696 | -33.7% | 99.3% | 0.260 | 0.777 |
| development | baa_spread | z-score in/out (enter 0.5 / exit 0) | 6.9% | 0.549 | -23.6% | 62.3% | 1.336 | 0.311 |
| holdout | baa_spread | z-score in/out (enter 0.5 / exit 0) | 8.4% | 0.519 | -20.8% | 61.4% | 3.516 | 0.777 |
