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
| ICE HY OAS (Δ) | 2023-10-02 | -0.622 | -0.142 | 0.008 | 0.069 |
| ICE BBB OAS (Δ) | 2023-10-02 | -0.436 | -0.274 | -0.023 | -0.020 |
| Vanguard HY fund excess return | 1993-02-01 | 0.383 | 0.179 | 0.124 | 0.071 |
| HYG excess return | 2007-04-12 | 0.717 | 0.000 | 0.015 | -0.021 |

Rank IC of the Baa ensemble's 5-day forecast (holdout walk-forward) against other targets (the ICE columns start when FRED's own window does, unless a longer history was spliced in):

| target | from | n | ic |
|---|---|---|---|
| Moody's Baa – 10y (the training target) | 2023-09-29 | 736 | 0.167 |
| ICE BofA HY OAS | 2023-09-29 | 736 | 0.091 |
| ICE BofA BBB OAS | 2023-09-29 | 736 | 0.188 |
| ICE BofA IG OAS | 2023-09-29 | 736 | 0.156 |
| HYG vs IEI excess return (sign flipped) | 2023-09-29 | 735 | 0.053 |
| HYG vs IEI excess return (sign flipped) | 2019-01-02 | 1929 | 0.043 |

# 3. Tradable high yield: HYG vs IEI excess return

Target: 5-day return of HYG minus 0.85 × IEI (bps), starting the day after the signal (`target_5d_lag1_xs_return`).  Features: the core set plus HY excess-return momentum, volatility and drawdown, and Gilchrist–Zakrajšek spread changes.  The composite uses the same economic signs, reversed for a return target.

| period | model | n | oos_r2 | ic | hit_rate | signal_sharpe | years_ic>0 |
|---|---|---|---|---|---|---|---|
| development | composite | 2012 | -0.025 | -0.124 | 0.473 | -0.411 | 1/8 |
| development | ridge | 2012 | -0.149 | 0.109 | 0.515 | 0.706 | 7/8 |
| development | lightgbm | 2012 | 0.009 | 0.082 | 0.516 | 0.629 | 6/8 |
| development | ensemble | 2012 | 0.008 | 0.129 | 0.525 | 0.899 | 8/8 |
| holdout | composite | 1929 | -0.002 | -0.059 | 0.510 | -0.288 | 2/8 |
| holdout | ridge | 1929 | -0.210 | 0.151 | 0.548 | 0.551 | 6/8 |
| holdout | lightgbm | 1929 | -0.013 | 0.088 | 0.554 | 0.462 | 7/8 |
| holdout | ensemble | 1929 | -0.018 | 0.163 | 0.559 | 0.674 | 8/8 |

# 4. Regime-conditional forecasts

Walk-forward HMM probabilities (3 states on the Baa spread, refitted yearly on past data, forward-filtered) added as features (`regime_p_stress`, `regime_level`).  Rows are restricted to dates where the probabilities exist, so the baseline differs slightly from section 1.

| period | model | n | oos_r2 | ic | hit_rate | signal_sharpe | years_ic>0 |
|---|---|---|---|---|---|---|---|
| development | Baa 5d change · ensemble without regimes | 4779 | 0.111 | 0.251 | 0.545 | 1.422 | 16/19 |
| holdout | Baa 5d change · ensemble without regimes | 1930 | 0.061 | 0.220 | 0.550 | 1.056 | 8/8 |
| development | Baa 5d change · ensemble with regimes | 4779 | 0.110 | 0.244 | 0.546 | 1.441 | 15/19 |
| holdout | Baa 5d change · ensemble with regimes | 1930 | 0.057 | 0.227 | 0.556 | 1.262 | 8/8 |
| development | HYG 5d excess return · ensemble without regimes | 2012 | 0.008 | 0.129 | 0.525 | 0.899 | 8/8 |
| holdout | HYG 5d excess return · ensemble without regimes | 1929 | -0.018 | 0.163 | 0.559 | 0.674 | 8/8 |
| development | HYG 5d excess return · ensemble with regimes | 2012 | -0.056 | 0.127 | 0.517 | 0.674 | 8/8 |
| holdout | HYG 5d excess return · ensemble with regimes | 1929 | -0.018 | 0.160 | 0.556 | 0.597 | 8/8 |

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
