"""
Walk-forward evaluation of every forecasting model and exposure signal.

Produces out-of-sample metrics for two periods:

* **development** – used to choose models and rules, and
* **holdout** – ``HOLDOUT_START`` onward.

Models are refitted every year on all data whose labels were realised before
that year (purged by the forecast horizon plus any execution lag).

Usage
-----
    python scripts/evaluate_signals.py --data-path data/market_data_1990-01-01_2026-09-15.parquet

Options
-------
    --data-path     Parquet file from download_data.py (required)
    --horizons      Spread-forecast horizons in trading days (default: 5 20)
    --spread-col    Spread to model (default: the configured primary spread)
    --output        Markdown report path (default: reports/signal_evaluation.md)
    --with-dl       Also evaluate the LSTM on the holdout (slower)

When a long ICE high-yield history has been spliced in (see
SPREAD_HISTORY_DIR), a final section compares it with the Moody's Baa proxy.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from typing import Optional  # noqa: E402

from config.settings import HOLDOUT_START  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_signals")

DEV_START = "2000-01-01"
HY_DEV_START = "2011-01-01"  # HYG starts in 2007 and features need a year of history
HY_MIN_TRAIN = 500
REGRESSION_MODELS = ["composite", "ridge", "lightgbm", "xgboost", "random_forest", "ensemble"]
HY_MODELS = ["composite", "ridge", "lightgbm", "ensemble"]
CLASSIFICATION_MODELS = ["composite", "ridge", "lightgbm"]
SCALED_TREE_MODELS = {"lightgbm", "xgboost", "random_forest"}
EXPOSURE_LABELS = {
    "vol_regime": "vol target 15% × (1 − P(stress))",
    "vol": "vol target 15% only",
    "regime": "1 − P(stress) only",
    "zscore": "z-score in/out (enter 0.5 / exit 0)",
    "widening": "20d widening > 50bp in/out",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--spread-col", default=None, help="Spread column to model")
    parser.add_argument("--output", default="reports/signal_evaluation.md")
    parser.add_argument("--with-dl", action="store_true")
    return parser.parse_args()


def _periods(index: pd.DatetimeIndex, dev_start: str = DEV_START) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    holdout = pd.Timestamp(HOLDOUT_START)
    return {
        "development": (pd.Timestamp(dev_start), holdout - pd.to_timedelta(1, unit="D")),
        "holdout": (holdout, index.max()),
    }


def _yearly_share(pred: pd.Series, y: pd.Series, fn) -> str:
    d = pd.concat([pred.rename("p"), y.rename("y")], axis=1).dropna()
    vals = [fn(g.y.values, g.p.values) for _, g in d.groupby(d.index.year)]
    vals = [v for v in vals if np.isfinite(v)]
    good = sum(v > 0 for v in vals)
    return f"{good}/{len(vals)}"


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b)[0]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")


def _regression_row(period: str, model: str, pred: pd.Series, y: pd.Series, horizon: int) -> dict:
    from src.models.ml_models import compute_metrics

    ok = pred.notna() & y.notna()
    m = compute_metrics(y[ok].values, pred[ok].values, horizon=horizon)
    return {
        "period": period, "model": model, "n": int(ok.sum()),
        "oos_r2": m["oos_r2"], "ic": m["ic"], "hit_rate": m["directional_accuracy"],
        "signal_sharpe": m["signal_sharpe"],
        "years_ic>0": _yearly_share(pred[ok], y[ok], _spearman),
    }


def evaluate_forecasts(df: pd.DataFrame, horizon: int, spread_col: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import roc_auc_score

    from src.features.engineering import build_feature_matrix
    from src.models.ml_models import DEFAULT_SCALE_COL, compute_metrics, walk_forward_predict

    X, y = build_feature_matrix(df, target_horizon=horizon, target_col=spread_col)
    y_chg = y[f"target_{horizon}d_change"]
    y_up = y[f"target_{horizon}d_up"]

    reg_rows, cls_rows = [], []
    for name, (start, end) in _periods(X.index).items():
        mask = (X.index >= start) & (X.index <= end)
        for model_type in REGRESSION_MODELS:
            scale_col = DEFAULT_SCALE_COL if model_type in SCALED_TREE_MODELS else None
            pred = walk_forward_predict(
                X, y_chg, model_type=model_type, start=str(start.date()), end=str(end.date()),
                gap=horizon, scale_col=scale_col,
            )
            row = _regression_row(name, model_type, pred[mask], y_chg[mask], horizon)
            reg_rows.append(row)
            print(f"  baa h={horizon} {name:11s} {model_type:13s} R2={row['oos_r2']:+.3f} IC={row['ic']:+.3f}")

        yb = y_up[mask]
        base = y_up.expanding().mean().shift(horizon)[mask]  # climatology known at the time
        for model_type in CLASSIFICATION_MODELS:
            prob = walk_forward_predict(
                X, y_up, model_type=model_type, task="classification",
                start=str(start.date()), end=str(end.date()), gap=horizon,
            )[mask]
            ok = prob.notna() & base.notna()
            m = compute_metrics(yb[ok].values, prob[ok].values, task="classification")
            ref = float(np.mean((base[ok].values - yb[ok].values) ** 2))
            cls_rows.append({
                "period": name, "model": model_type, "n": int(ok.sum()),
                "base_rate": m["base_rate"], "auc": m["roc_auc"],
                "brier_skill": 1 - m["brier"] / ref if ref > 0 else np.nan,
                "years_auc>0.5": _yearly_share(
                    prob, yb,
                    lambda a, b: roc_auc_score(a, b) - 0.5 if len(np.unique(a)) > 1 else np.nan,
                ),
            })
    return pd.DataFrame(reg_rows), pd.DataFrame(cls_rows)


def evaluate_hy(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Tradable high-yield excess return (HYG vs IEI), next-day execution."""
    from src.features.engineering import build_hy_feature_matrix
    from src.models.ml_models import DEFAULT_SCALE_COL, RETURN_COMPOSITE_WEIGHTS, walk_forward_predict

    X, y = build_hy_feature_matrix(df, target_horizon=horizon)
    target = y[f"target_{horizon}d_lag1_xs_return"]
    rows = []
    for name, (start, end) in _periods(X.index, HY_DEV_START).items():
        mask = (X.index >= start) & (X.index <= end)
        for model_type in HY_MODELS:
            pred = walk_forward_predict(
                X, target, model_type=model_type, start=str(start.date()), end=str(end.date()),
                scale_col=DEFAULT_SCALE_COL if model_type in SCALED_TREE_MODELS else None,
                params={"weights": RETURN_COMPOSITE_WEIGHTS} if model_type == "composite" else None,
                min_train=HY_MIN_TRAIN,
            )
            row = _regression_row(name, model_type, pred[mask], target[mask], horizon)
            rows.append(row)
            print(f"  hyg h={horizon} {name:11s} {model_type:13s} R2={row['oos_r2']:+.3f} IC={row['ic']:+.3f}")
    return pd.DataFrame(rows)


def evaluate_transfer(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Do Baa-spread forecasts carry over to market-based spreads and to HYG?"""
    from src.features.engineering import build_feature_matrix, create_return_targets
    from src.models.ml_models import walk_forward_predict

    X, y = build_feature_matrix(df, target_horizon=horizon)
    pred = walk_forward_predict(X, y[f"target_{horizon}d_change"], model_type="ensemble", start=HOLDOUT_START)
    hyg = -create_return_targets(df["hyg_xs_return"], horizon)[f"target_{horizon}d_lag1_xs_return"]

    ice_start = df["hy_spread"].first_valid_index() if "hy_spread" in df.columns else None
    first = ice_start if ice_start is not None else pd.Timestamp(HOLDOUT_START)
    targets = [("Moody's Baa – 10y (the training target)", df["baa_spread"].shift(-horizon) - df["baa_spread"], first)]
    for col, label in (("hy_spread", "ICE BofA HY OAS"), ("bbb_spread", "ICE BofA BBB OAS"), ("ig_spread", "ICE BofA IG OAS")):
        if col in df.columns and ice_start is not None:
            targets.append((label, df[col].shift(-horizon) - df[col], ice_start))
    targets.append(("HYG vs IEI excess return (sign flipped)", hyg, first))
    if ice_start is not None:
        targets.append(("HYG vs IEI excess return (sign flipped)", hyg, pd.Timestamp(HOLDOUT_START)))

    rows = []
    for label, target, start in targets:
        d = pd.concat([pred.rename("p"), target.rename("y")], axis=1).dropna().loc[start:]
        rows.append({"target": label, "from": str(d.index.min().date()), "n": len(d),
                     "ic": _spearman(d.p.values, d.y.values)})
    return pd.DataFrame(rows)


def staleness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation of each series' daily change with the equity return k days earlier."""
    rows = []
    series = {
        "Moody's Baa – 10y (Δ)": df["baa_spread"].diff(),
        "ICE HY OAS (Δ)": df["hy_spread"].diff() if "hy_spread" in df else None,
        "ICE BBB OAS (Δ)": df["bbb_spread"].diff() if "bbb_spread" in df else None,
        "Vanguard HY fund excess return": df.get("hy_fund_xs_return"),
        "HYG excess return": df.get("hyg_xs_return"),
    }
    for label, s in series.items():
        if s is None:
            continue
        d = pd.concat([s.rename("move"), df["spy_return"].rename("equity")], axis=1).dropna()
        row = {"series": label, "from": str(d.index.min().date())}
        for k in range(4):
            row[f"k={k}"] = float(d["move"].shift(-k).corr(d["equity"]))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_regime_features(df: pd.DataFrame, regime_probs: pd.DataFrame) -> pd.DataFrame:
    """Ablation: do walk-forward regime probabilities improve the recommended models?"""
    from src.features.engineering import build_feature_matrix, build_hy_feature_matrix
    from src.models.ml_models import walk_forward_predict

    rows = []
    specs = [
        ("Baa 5d change", build_feature_matrix, "target_5d_change", DEV_START, 756),
        ("HYG 5d excess return", build_hy_feature_matrix, "target_5d_lag1_xs_return", HY_DEV_START, HY_MIN_TRAIN),
    ]
    for label, builder, target_col, dev_start, min_train in specs:
        X_base, y = builder(df, target_horizon=5)
        X_reg, _ = builder(df, target_horizon=5, regime_probs=regime_probs)
        common = X_base.index.intersection(X_reg.index)
        for variant, X in (("without regimes", X_base.loc[common]), ("with regimes", X_reg.loc[common])):
            target = y.loc[common, target_col]
            for name, (start, end) in _periods(common, dev_start).items():
                mask = (common >= start) & (common <= end)
                pred = walk_forward_predict(X, target, model_type="ensemble", start=str(start.date()),
                                            end=str(end.date()), min_train=min_train)
                row = _regression_row(name, f"{label} · ensemble {variant}", pred[mask], target[mask], 5)
                rows.append(row)
                print(f"  regimes {label} {variant} {name}: R2={row['oos_r2']:+.3f} IC={row['ic']:+.3f}")
    return pd.DataFrame(rows)


def evaluate_exposure(df: pd.DataFrame, regime_probs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.analysis.leading_indicator import compute_backtest_metrics, run_full_backtest

    equity_col = "spy_return" if "spy_return" in df.columns else "sp500_return"
    backtests = {
        method: run_full_backtest(df, equity_col=equity_col, method=method, regime_probs=regime_probs)[0]
        for method in EXPOSURE_LABELS
    }
    rows = []
    for name, (start, end) in _periods(df.index).items():
        for method, bt in backtests.items():
            m = compute_backtest_metrics(bt.loc[start:end])
            rows.append({
                "period": name, "overlay": EXPOSURE_LABELS[method],
                "cagr": m["annualised_return"], "vol": m["annualised_volatility"],
                "sharpe": m["sharpe"], "max_dd": m["max_drawdown"],
                "avg_equity": m["avg_equity_weight"], "turnover/yr": m["turnover_per_year"],
            })
        rows.append({
            "period": name, "overlay": "buy & hold SPY",
            "cagr": m["bh_annualised_return"], "vol": m["bh_annualised_volatility"],
            "sharpe": m["bh_sharpe"], "max_dd": m["bh_max_drawdown"],
            "avg_equity": 1.0, "turnover/yr": 0.0,
        })

    sub_periods = [("2000", "2004"), ("2005", "2009"), ("2010", "2014"), ("2015", "2018"), ("2019", "2022"), ("2023", "2026")]
    sub = {}
    for method, bt in backtests.items():
        sub[EXPOSURE_LABELS[method]] = {
            f"{a}–{b}": compute_backtest_metrics(bt.loc[a:b])["sharpe"] for a, b in sub_periods
        }
    any_bt = next(iter(backtests.values()))
    sub["buy & hold SPY"] = {f"{a}–{b}": compute_backtest_metrics(any_bt.loc[a:b])["bh_sharpe"] for a, b in sub_periods}
    sub_df = pd.DataFrame(sub).T.reset_index().rename(columns={"index": "overlay"})
    return pd.DataFrame(rows), sub_df


def compare_spreads(df: pd.DataFrame, horizons: tuple[int, ...] = (5, 20)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ICE high-yield OAS vs the Moody's Baa proxy, on the dates both can predict."""
    from src.features.engineering import build_feature_matrix, build_hy_feature_matrix
    from src.models.ml_models import DEFAULT_SCALE_COL, walk_forward_predict

    models = ["composite", "ridge", "lightgbm", "ensemble"]
    target_rows = []
    for horizon in horizons:
        predictions, targets = {}, {}
        for spread in ("hy_spread", "baa_spread"):
            X, y = build_feature_matrix(df, target_horizon=horizon, target_col=spread)
            targets[spread] = y[f"target_{horizon}d_change"]
            predictions[spread] = {
                mt: walk_forward_predict(
                    X, targets[spread], model_type=mt, start=DEV_START,
                    scale_col=DEFAULT_SCALE_COL if mt in SCALED_TREE_MODELS else None,
                )
                for mt in models
            }
        common = predictions["hy_spread"]["ensemble"].dropna().index.intersection(
            predictions["baa_spread"]["ensemble"].dropna().index
        )
        for name, (start, end) in _periods(common).items():
            mask = (common >= start) & (common <= end)
            for spread in ("hy_spread", "baa_spread"):
                for mt in models:
                    pred = predictions[spread][mt].reindex(common)[mask]
                    row = _regression_row(name, f"{spread} · {mt}", pred, targets[spread].reindex(common)[mask], horizon)
                    row["horizon"] = f"{horizon}d"
                    target_rows.append(row)
        print(f"  spread comparison h={horizon}: {len(common)} common dates")

    hy_rows = []
    for spread in ("hy_spread", "baa_spread"):
        X, y = build_hy_feature_matrix(df, target_horizon=5, spread_col=spread)
        target = y["target_5d_lag1_xs_return"]
        for mt in ("ridge", "ensemble"):
            pred = walk_forward_predict(X, target, model_type=mt, start=HY_DEV_START, min_train=HY_MIN_TRAIN)
            for name, (start, end) in _periods(X.index, HY_DEV_START).items():
                mask = (X.index >= start) & (X.index <= end)
                hy_rows.append(_regression_row(name, f"features on {spread} · {mt}", pred[mask], target[mask], 5))
    columns = ["horizon", "period", "model", "n", "oos_r2", "ic", "hit_rate", "signal_sharpe", "years_ic>0"]
    return pd.DataFrame(target_rows)[columns], pd.DataFrame(hy_rows)


def compare_overlay_spreads(df: pd.DataFrame, start: str = "2001-01-01") -> pd.DataFrame:
    """Exposure overlays driven by HY-OAS regimes vs Baa regimes."""
    from src.analysis.leading_indicator import compute_backtest_metrics, run_full_backtest
    from src.models.regime import real_time_regime_probabilities

    equity_col = "spy_return" if "spy_return" in df.columns else "sp500_return"
    rows = []
    for spread in ("hy_spread", "baa_spread"):
        probs = real_time_regime_probabilities(df, spread_col=spread)
        for method in ("vol_regime", "regime", "zscore"):
            bt = run_full_backtest(df, spread_col=spread, equity_col=equity_col, method=method, regime_probs=probs)[0]
            bt = bt.loc[start:]
            for name, (period_start, period_end) in _periods(bt.index, start).items():
                m = compute_backtest_metrics(bt.loc[period_start:period_end])
                rows.append({
                    "period": name, "spread": spread, "overlay": EXPOSURE_LABELS[method],
                    "cagr": m["annualised_return"], "sharpe": m["sharpe"], "max_dd": m["max_drawdown"],
                    "avg_equity": m["avg_equity_weight"], "turnover/yr": m["turnover_per_year"],
                    "bh_sharpe": m["bh_sharpe"],
                })
    return pd.DataFrame(rows)


def evaluate_lstm(df: pd.DataFrame, horizon: int) -> dict[str, float]:
    """Train on data before the holdout (early stopping on its last 15%), test on the holdout."""
    from src.features.engineering import build_feature_matrix
    from src.models.dl_models import evaluate_dl_model, train_dl_model

    seq_len = 20
    X, y = build_feature_matrix(df, target_horizon=horizon)
    y_chg = y[f"target_{horizon}d_change"]
    first_test = int(np.argmax(X.index >= pd.Timestamp(HOLDOUT_START)))
    train_stop = first_test - horizon
    result = train_dl_model(
        X.values[:train_stop], y_chg.values[:train_stop], model_type="lstm",
        seq_len=seq_len, gap=horizon, epochs=40,
    )
    ctx = first_test - seq_len + 1
    _, metrics = evaluate_dl_model(
        result["model"], X.values[ctx:], y_chg.values[ctx:], seq_len=seq_len,
        scaler=result["scaler"], y_scale=result["y_scale"], horizon=horizon,
    )
    return metrics


def _fmt(v: object) -> str:
    if isinstance(v, (float, np.floating)):
        return "–" if not np.isfinite(v) else f"{v:.3f}"
    return str(v)


def to_markdown(df: pd.DataFrame, pct_cols: tuple[str, ...] = ()) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.1%}" if c in pct_cols and isinstance(v, float) else _fmt(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.data_path)

    from src.features.engineering import default_spread_column, spread_columns_with_history

    spread_col = args.spread_col or default_spread_column(df)
    spread_label = {"baa_spread": "Moody's Baa – 10y", "hy_spread": "ICE BofA HY OAS"}.get(spread_col, spread_col)
    long_spreads = spread_columns_with_history(df)
    periods = _periods(df.index)
    hy_periods = _periods(df.index, HY_DEV_START)
    out = [
        "# Signal evaluation",
        "",
        f"Data: `{Path(args.data_path).name}` ({df.index.min().date()} → {df.index.max().date()}).  ",
        f"Development period: {periods['development'][0].date()} → {periods['development'][1].date()} "
        f"({hy_periods['development'][0].date()} → for HYG); "
        f"holdout: {periods['holdout'][0].date()} → {periods['holdout'][1].date()}.",
        "",
        "All predictions are out-of-sample: models are refitted yearly on data whose labels were fully realised "
        "before the prediction year.",
        "",
        "* `oos_r2` – 1 − SSE / SSE(zero forecast); > 0 means the forecast beats \"no change\".",
        "* `ic` – Spearman rank correlation of forecast and outcome.",
        "* `hit_rate` – share of correct direction calls.",
        "* `signal_sharpe` – annualised Sharpe of trading the sign of the forecast, non-overlapping periods.",
        "* `years_ic>0` / `years_auc>0.5` – calendar years in which the signal had the right sign.",
        "* `brier_skill` – improvement in Brier score over the expanding historical base rate.",
        "",
        f"# 1. {spread_label} spread forecasts",
        "",
        f"Target: change in `{spread_col}` (bps) over the next *h* trading days.",
        "",
    ]
    for h in args.horizons:
        print(f"Evaluating {spread_label} forecasts, horizon {h} …")
        reg, cls = evaluate_forecasts(df, h, spread_col=spread_col)
        out += [f"## {h}-day spread change – regression", "", to_markdown(reg), ""]
        out += [f"## {h}-day spread widening (up/down) – classification", "", to_markdown(cls), ""]
        if args.with_dl:
            print(f"Evaluating LSTM, horizon {h} …")
            m = evaluate_lstm(df, h)
            out += [
                f"LSTM (holdout, h={h}, single seed – results vary with the seed): oos_r2 {m['oos_r2']:.3f}, "
                f"ic {m['ic']:.3f}, hit_rate {m['directional_accuracy']:.3f}, signal_sharpe {m['signal_sharpe']:.2f}",
                "",
            ]

    print("Evaluating index staleness and transfer …")
    out += [
        "# 2. How much of the spread forecast is tradable?",
        "",
        "Correlation between each series' daily move on day *t + k* and the SPY return on day *t*.  "
        "Non-zero values at k ≥ 1 mean the series reacts to equity moves with a delay "
        "(stale or smoothed pricing), which makes it partly predictable without being tradable.",
        "",
        to_markdown(staleness_table(df)),
        "",
        "Rank IC of the Baa ensemble's 5-day forecast (holdout walk-forward) against other targets "
        "(the ICE columns start when FRED's own window does, unless a longer history was spliced in):",
        "",
        to_markdown(evaluate_transfer(df)),
        "",
    ]

    print("Evaluating tradable high-yield forecasts …")
    out += [
        "# 3. Tradable high yield: HYG vs IEI excess return",
        "",
        "Target: 5-day return of HYG minus 0.85 × IEI (bps), starting the day after the signal "
        "(`target_5d_lag1_xs_return`).  Features: the core set plus HY excess-return momentum, volatility and "
        "drawdown, and Gilchrist–Zakrajšek spread changes.  The composite uses the same economic signs, reversed "
        "for a return target.",
        "",
        to_markdown(evaluate_hy(df)),
        "",
    ]

    print("Computing walk-forward regime probabilities …")
    from src.models.regime import real_time_regime_probabilities

    regime_probs = real_time_regime_probabilities(df)

    print("Evaluating regime features …")
    out += [
        "# 4. Regime-conditional forecasts",
        "",
        "Walk-forward HMM probabilities (3 states on the Baa spread, refitted yearly on past data, forward-filtered) "
        "added as features (`regime_p_stress`, `regime_level`).  Rows are restricted to dates where the "
        "probabilities exist, so the baseline differs slightly from section 1.",
        "",
        to_markdown(evaluate_regime_features(df, regime_probs)),
        "",
    ]

    print("Evaluating exposure overlays …")
    exposure, sub = evaluate_exposure(df, regime_probs)
    out += [
        "# 5. Equity exposure overlays (SPY vs 3-month T-bills)",
        "",
        "Weights use data published by each close and trade at the next close.  Fractional overlays only "
        "rebalance when the target moves by more than 10 percentage points; costs are 5 bp per unit of turnover.  "
        "These overlays were compared on both periods at the same time, so the holdout is not untouched for this "
        "particular choice.",
        "",
        to_markdown(exposure, pct_cols=("cagr", "vol", "max_dd", "avg_equity")),
        "",
        "Sharpe ratio by sub-period:",
        "",
        to_markdown(sub),
        "",
    ]

    if "hy_spread" in long_spreads:
        print("Comparing the long ICE high-yield history with the Baa proxy …")
        target_cmp, hy_feature_cmp = compare_spreads(df, tuple(args.horizons))
        overlay_cmp = compare_overlay_spreads(df)
        out += [
            "# 6. Long ICE high-yield history vs the Moody's Baa proxy",
            "",
            f"`hy_spread` holds {int(df['hy_spread'].notna().sum())} observations "
            f"({df['hy_spread'].first_valid_index().date()} → {df['hy_spread'].last_valid_index().date()}), so a "
            "licensed history has been spliced in behind FRED's three-year window.  Both spreads are compared on "
            "the dates where each has walk-forward predictions.",
            "",
            "## As a forecast target",
            "",
            to_markdown(target_cmp),
            "",
            "## As the feature spread for the tradable HYG model",
            "",
            to_markdown(hy_feature_cmp),
            "",
            "## As the driver of the exposure overlay (from 2001, after the HY regime warm-up)",
            "",
            to_markdown(overlay_cmp, pct_cols=("cagr", "max_dd", "avg_equity")),
            "",
        ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(out), encoding="utf-8")
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
