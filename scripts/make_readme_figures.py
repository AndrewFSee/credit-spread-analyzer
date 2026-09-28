"""
Render the README figures (light and dark variants) from the project's own models.

Usage
-----
    python scripts/make_readme_figures.py --data-path data/market_data_1990-01-01_2026-09-15.parquet

Outputs PNGs to docs/images/.  Only public data is used (Moody's Baa – 10y,
SPY, T-bills and the HY fund / ETF proxies), so anyone can regenerate them
without a licensed spread history.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.settings import HOLDOUT_START  # noqa: E402

OUT_DIR = Path(__file__).parent.parent / "docs" / "images"
RECESSIONS = [("1990-07-01", "1991-03-31"), ("2001-03-01", "2001-11-30"),
              ("2007-12-01", "2009-06-30"), ("2020-02-01", "2020-04-30")]

# Validated palette (dataviz reference instance): slots 1-3 plus chart chrome.
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "band": "#efeee9",
        "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "bench": "#898781",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "band": "#262624",
        "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "bench": "#898781",
    },
}
LINE_W = 1.5
FONT = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render README figures.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def _style_axis(ax: plt.Axes, t: dict, y_grid: bool = True) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=t["muted"], labelsize=9.5, length=0, pad=6)
    if y_grid:
        ax.grid(axis="y", color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)


def _header(fig: plt.Figure, t: dict, title: str, subtitle: str, source: str) -> None:
    fig.text(0.045, 0.955, title, color=t["ink"], fontsize=15.5, fontweight="semibold", va="top")
    fig.text(0.045, 0.895, subtitle, color=t["ink2"], fontsize=10.5, va="top")
    fig.text(0.045, 0.025, source, color=t["muted"], fontsize=8.5, va="bottom")


def _recession_bands(ax: plt.Axes, t: dict, start: pd.Timestamp) -> None:
    for a, b in RECESSIONS:
        if pd.Timestamp(b) >= start:
            ax.axvspan(max(pd.Timestamp(a), start), pd.Timestamp(b), color=t["band"], lw=0, zorder=0)


def _save(fig: plt.Figure, out_dir: Path, name: str, mode: str) -> Path:
    path = out_dir / f"{name}-{mode}.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _date_axis(ax: plt.Axes, years: int = 5) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(years))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr

    from src.analysis.leading_indicator import compute_backtest_metrics, run_full_backtest
    from src.features.engineering import build_feature_matrix
    from src.models.ml_models import walk_forward_predict
    from src.models.regime import real_time_regime_probabilities

    print("Real-time regime probabilities (yearly HMM refits) …")
    probs = real_time_regime_probabilities(df, spread_col="baa_spread")

    print("Exposure-overlay backtest …")
    bt, _ = run_full_backtest(df, spread_col="baa_spread", regime_probs=probs)
    bt = bt.loc["2000-01-01":].copy()
    bt["strategy_cumulative"] = (1 + bt["strategy_return"]).cumprod()
    bt["bh_cumulative"] = (1 + bt["equity_return"]).cumprod()
    stats = compute_backtest_metrics(bt)

    print("Walk-forward 5-day forecasts …")
    X, y = build_feature_matrix(df, target_horizon=5, target_col="baa_spread")
    target = y["target_5d_change"]
    pred = walk_forward_predict(X, target, model_type="ensemble", start="2000-01-01")
    d = pd.concat([pred.rename("p"), target.rename("y")], axis=1).dropna()
    yearly_ic = d.groupby(d.index.year).apply(lambda g: spearmanr(g.p, g.y)[0])

    print("Lead–lag of each series against the prior day's S&P 500 move …")
    series = {
        "Moody's Baa index": -df["baa_spread"].diff(),   # sign flipped: spreads fall when stocks rise
        "Vanguard HY fund NAV": df["hy_fund_xs_return"],
        "HYG (exchange-traded)": df["hyg_xs_return"],
    }
    lead_lag = {}
    for name, s in series.items():
        x = pd.concat([s.rename("move"), df["spy_return"].rename("equity")], axis=1).dropna()
        lead_lag[name] = [x["move"].shift(-k).corr(x["equity"]) for k in range(4)]

    return {"probs": probs, "bt": bt, "stats": stats, "yearly_ic": yearly_ic, "lead_lag": lead_lag}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_stress_monitor(df: pd.DataFrame, data: dict, t: dict) -> plt.Figure:
    start = pd.Timestamp("1994-01-01")
    spread = df["baa_spread"].loc[start:]
    p_stress = data["probs"].iloc[:, -1].loc[start:] * 100

    fig = plt.figure(figsize=(11, 5.6), facecolor=t["surface"])
    gs = fig.add_gridspec(2, 1, height_ratios=[2.1, 1], hspace=0.12, left=0.075, right=0.975, top=0.80, bottom=0.12)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for ax in (ax1, ax2):
        _style_axis(ax, t)
        _recession_bands(ax, t, start)

    ax1.plot(spread.index, spread, color=t["s1"], lw=LINE_W, solid_joinstyle="round")
    ax1.set_ylabel("Spread (bps)", color=t["muted"], fontsize=9.5)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(100))
    ax1.set_ylim(100, 650)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.spines["bottom"].set_visible(False)

    for label, date in [("2008 financial crisis", "2008-12-01"), ("2020 COVID shock", "2020-03-20")]:
        window = spread.loc[pd.Timestamp(date) - pd.Timedelta(days=120):pd.Timestamp(date) + pd.Timedelta(days=120)]
        peak_date, peak = window.idxmax(), window.max()
        ax1.annotate(f"{label}\n{peak:.0f} bps", xy=(peak_date, peak), xytext=(12, -6),
                     textcoords="offset points", color=t["ink2"], fontsize=9, va="top")
    ax1.text(pd.Timestamp("2001-07-01"), 630, "Recessions", color=t["muted"], fontsize=8.5, ha="center", va="top")

    ax2.fill_between(p_stress.index, 0, p_stress, color=t["s2"], alpha=0.14, lw=0)
    ax2.plot(p_stress.index, p_stress, color=t["s2"], lw=1.1)
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(50))
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax2.set_ylabel("P(stress)", color=t["muted"], fontsize=9.5)
    _date_axis(ax2, 4)
    ax2.set_xlim(start, spread.index.max())

    _header(
        fig, t,
        "Detecting credit stress in real time",
        "Moody's Baa – 10y Treasury spread (top) and the model's probability of a credit-stress regime (bottom).\n"
        "The regime model is refitted each January on past data only, so every point is what it would have said that day.",
        "Source: FRED (Moody's via Federal Reserve H.15). Shaded: NBER recessions. Probabilities start once three years of history exist.",
    )
    return fig


def fig_exposure_overlay(data: dict, t: dict) -> plt.Figure:
    bt, s = data["bt"], data["stats"]
    strat_dd = bt["strategy_cumulative"] / bt["strategy_cumulative"].cummax() - 1
    bh_dd = bt["bh_cumulative"] / bt["bh_cumulative"].cummax() - 1

    fig = plt.figure(figsize=(11, 5.9), facecolor=t["surface"])
    gs = fig.add_gridspec(2, 1, height_ratios=[1.7, 1], hspace=0.14, left=0.075, right=0.975, top=0.79, bottom=0.11)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for ax in (ax1, ax2):
        _style_axis(ax, t)
        ax.axvline(pd.Timestamp(HOLDOUT_START), color=t["axis"], lw=0.9, zorder=1)

    strat_end, bench_end = bt["strategy_cumulative"].iloc[-1], bt["bh_cumulative"].iloc[-1]
    strat_label = (f"Credit-aware overlay   \\$1 → \\${strat_end:.1f}  ·  Sharpe {s['sharpe']:.2f}  ·  "
                   f"max drawdown {s['max_drawdown']:.0%}")
    bench_label = (f"Buy & hold S&P 500    \\$1 → \\${bench_end:.1f}  ·  Sharpe {s['bh_sharpe']:.2f}  ·  "
                   f"max drawdown {s['bh_max_drawdown']:.0%}")
    ax1.plot(bt.index, bt["bh_cumulative"], color=t["bench"], lw=1.2, label=bench_label)
    ax1.plot(bt.index, bt["strategy_cumulative"], color=t["s1"], lw=LINE_W, label=strat_label)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_locator(mticker.FixedLocator([0.5, 1, 2, 4, 8]))
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"\\${v:g}"))
    ax1.yaxis.set_minor_locator(mticker.NullLocator())
    ax1.set_ylabel("Growth of $1 (log)", color=t["muted"], fontsize=9.5)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.spines["bottom"].set_visible(False)
    legend = ax1.legend(loc="upper left", frameon=False, fontsize=9, handlelength=1.6, labelcolor=t["ink2"])
    for line in legend.get_lines():
        line.set_linewidth(2.2)
    ax1.text(pd.Timestamp(HOLDOUT_START) + pd.Timedelta(days=60), 0.55, "Holdout →", color=t["muted"], fontsize=8.5)

    ax2.fill_between(bh_dd.index, 0, bh_dd * 100, color=t["bench"], alpha=0.16, lw=0)
    ax2.plot(bh_dd.index, bh_dd * 100, color=t["bench"], lw=1.0)
    ax2.fill_between(strat_dd.index, 0, strat_dd * 100, color=t["s1"], alpha=0.14, lw=0)
    ax2.plot(strat_dd.index, strat_dd * 100, color=t["s1"], lw=1.2)
    ax2.set_ylim(-60, 2)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax2.set_ylabel("Drawdown", color=t["muted"], fontsize=9.5)
    _date_axis(ax2, 4)
    ax2.set_xlim(bt.index.min(), bt.index.max())

    _header(
        fig, t,
        "Nearly the S&P 500's return, with a third of the drawdown",
        "Equity weight = volatility target × (1 − real-time probability of credit stress); the rest sits in T-bills.\n"
        "Trades execute the day after each signal with 5 bp costs. It lags in calm bull markets and wins in crises.",
        f"Backtest {bt.index.min():%b %Y} – {bt.index.max():%b %Y}, SPY total return vs 3-month T-bills. "
        "Vertical line: start of the 2019+ holdout. Details and caveats: reports/signal_evaluation.md.",
    )
    return fig


def fig_forecast_skill(data: dict, t: dict) -> plt.Figure:
    ic = data["yearly_ic"]
    years = ic.index.values
    holdout_year = pd.Timestamp(HOLDOUT_START).year

    fig = plt.figure(figsize=(11, 4.9), facecolor=t["surface"])
    ax = fig.add_axes([0.075, 0.13, 0.9, 0.62])
    _style_axis(ax, t)
    ax.axvspan(holdout_year - 0.5, years.max() + 0.5, color=t["band"], lw=0, zorder=0)
    ax.bar(years, ic.values, width=0.56, color=t["s1"], zorder=2)
    ax.axhline(0, color=t["axis"], lw=0.9, zorder=3)
    ax.set_xlim(years.min() - 0.7, years.max() + 0.7)
    ax.set_ylim(min(-0.1, ic.min() - 0.05), max(0.5, ic.max() + 0.08))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_ylabel("Rank correlation (IC)", color=t["muted"], fontsize=9.5)

    dev, hold = ic[ic.index < holdout_year], ic[ic.index >= holdout_year]
    ax.text(years.min() - 0.3, ax.get_ylim()[1] * 0.93,
            f"Development: mean IC {dev.mean():.2f}, positive in {(dev > 0).sum()} of {len(dev)} years",
            color=t["ink2"], fontsize=9.5, va="top")
    ax.text(holdout_year - 0.3, ax.get_ylim()[1] * 0.93,
            f"Holdout: mean IC {hold.mean():.2f},\npositive in {(hold > 0).sum()} of {len(hold)} years",
            color=t["ink2"], fontsize=9.5, va="top")

    _header(
        fig, t,
        "Forecast skill that holds up out of sample, year after year",
        "Rank correlation between the 5-day spread forecast and what actually happened, by calendar year.\n"
        "Every forecast comes from a model refitted each January on data available at the time.",
        "Ridge + LightGBM ensemble forecasting the Moody's Baa – 10y spread. Shaded: holdout years, never used to choose models.",
    )
    return fig


def fig_stale_pricing(data: dict, t: dict) -> plt.Figure:
    lead_lag = data["lead_lag"]
    colors = [t["s1"], t["s2"], t["s3"]]
    lags = np.arange(4)

    fig = plt.figure(figsize=(11, 4.9), facecolor=t["surface"])
    ax = fig.add_axes([0.075, 0.13, 0.9, 0.62])
    _style_axis(ax, t)
    for (name, values), color in zip(lead_lag.items(), colors, strict=True):
        ax.plot(lags, values, color=color, lw=LINE_W, label=name, zorder=3)
        ax.scatter(lags, values, s=42, color=color, edgecolor=t["surface"], linewidth=2, zorder=4)
    hyg, fund = lead_lag["HYG (exchange-traded)"], lead_lag["Vanguard HY fund NAV"]
    ax.annotate("ETF: fully priced in on day one", xy=(1, hyg[1]), xytext=(14, -10), textcoords="offset points",
                color=t["ink2"], fontsize=9.5, va="top")
    ax.annotate("Fund NAV and index still catching up", xy=(1, fund[1]), xytext=(14, 14), textcoords="offset points",
                color=t["ink2"], fontsize=9.5, va="bottom")
    ax.axhline(0, color=t["axis"], lw=0.9, zorder=2)
    ax.set_xticks(lags, ["Same day", "Next day", "Day 2", "Day 3"])
    ax.set_xlim(-0.15, 3.15)
    ax.set_ylim(-0.2, 0.8)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_ylabel("Correlation with the S&P 500 move", color=t["muted"], fontsize=9.5)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=t["ink2"])
    for line in legend.get_lines():
        line.set_linewidth(2.2)

    _header(
        fig, t,
        "Catching a hidden bias: stale prices look predictable",
        "How strongly each credit series moves with the S&P 500 on the same day and the days after.\n"
        "Index and fund prices keep catching up for days; the exchange-traded ETF reacts at once, so it is the honest test.",
        "Daily data since 1993 (HYG since 2007). Spread changes are sign-flipped so that all three series rise with stocks.",
    )
    return fig


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.data_path)
    data = compute(df)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT, "axes.unicode_minus": True})
    for mode, t in THEMES.items():
        figures = {
            "stress-monitor": fig_stress_monitor(df, data, t),
            "exposure-overlay": fig_exposure_overlay(data, t),
            "forecast-skill": fig_forecast_skill(data, t),
            "stale-pricing": fig_stale_pricing(data, t),
        }
        for name, fig in figures.items():
            print("wrote", _save(fig, out_dir, name, mode))


if __name__ == "__main__":
    main()
