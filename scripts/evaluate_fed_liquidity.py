"""
Does Fed liquidity predict credit spreads?

Tests whether changes in Fed net liquidity (total assets − Treasury General
Account − overnight reverse repo) lead credit-spread changes, on a weekly
Friday panel with no look-ahead:

* the H.4.1 balance sheet is dated from the Friday after its Thursday release
  (RELEASE_LAG_DAYS in config/settings.py);
* control variables use only spread values published by Friday's close.

Sections
--------
1. Same-period correlation (the endogeneity problem).
2. Predictive regressions with HAC t-stats, with and without controls,
   plus sub-period robustness and quintile sorts.
3. Event study of major Fed actions (src/analysis/fed_liquidity.py) against
   a no-intervention baseline of comparable widening episodes.

Spreads are included only when they have at least MIN_SPREAD_HISTORY
observations: FRED serves about three years of the ICE OAS series, so HY OAS
appears only when a longer licensed history has been spliced in.  CCC is
reported descriptively when present.

Usage
-----
    python scripts/evaluate_fed_liquidity.py --data-path data/market_data_1990-01-01_2026-10-18.parquet

Options
-------
    --data-path     Parquet file from fetch_all_data / download_data.py (required)
    --output        Markdown report path (default: reports/fed_liquidity.md)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from config.settings import FRED_DAILY_PUBLICATION_LAG, MIN_SPREAD_HISTORY  # noqa: E402
from src.analysis.fed_liquidity import FED_ACTIONS, event_study, liquidity_snapshot, net_liquidity  # noqa: E402

SPREADS = {"hy_spread": "ICE BofA HY OAS", "baa_spread": "Moody's Baa – 10y"}
HORIZONS = (4, 13)                 # weeks
LIQUIDITY_SIGNALS = ("net_pct_4w", "net_pct_13w")
PANEL_START = "2003-06-01"         # reverse repo and three years of control history
STRESS_PERCENTILE = 0.95           # baseline: 20-day widening in the top 5% of history
EPISODE_GAP = 60                   # trading days between baseline episodes
ACTION_EXCLUSION = 20              # trading days around a Fed action excluded from the baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", default="reports/fed_liquidity.md")
    return parser.parse_args()


def to_markdown(df: pd.DataFrame, decimals: int = 2) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else f"{v:+.{decimals}f}" if str(c).startswith(("coef", "t", "corr"))
                             else f"{v:.{decimals}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def weekly_panel(df: pd.DataFrame, spreads: list[str]) -> pd.DataFrame:
    """Friday panel: liquidity as released, spreads as published, forward outcomes."""
    liq = net_liquidity(df)
    fridays = df.index[df.index.dayofweek == 4]
    w = liq.reindex(fridays).ffill()
    for k in (4, 13):
        w[f"net_chg_{k}w"] = w["net"].diff(k)
        w[f"net_pct_{k}w"] = w["net"].pct_change(k, fill_method=None) * 100
    equity = "spy_return" if "spy_return" in df.columns else "sp500_return"
    if equity in df.columns:
        w["spy_ret_4w"] = np.log((1 + df[equity].fillna(0)).cumprod()).reindex(fridays).diff(4)
    if "vix" in df.columns:
        w["vix_chg_4w"] = df["vix"].reindex(fridays).diff(4)
    for col in spreads:
        s = df[col].reindex(fridays)                                       # as-dated: base for outcomes
        known = df[col].shift(FRED_DAILY_PUBLICATION_LAG).reindex(fridays)  # published by Friday's close
        w[f"{col}_known"] = known
        w[f"{col}_chg_4w_known"] = known.diff(4)
        w[f"{col}_level_z"] = (known - known.rolling(156).mean()) / known.rolling(156).std()
        for h in HORIZONS:
            w[f"{col}_fwd_{h}w"] = s.shift(-h) - s
    if "hyg_xs_index" in df.columns:
        hyg = np.log(df["hyg_xs_index"].reindex(fridays))
        for h in HORIZONS:
            w[f"hyg_fwd_{h}w"] = (hyg.shift(-h) - hyg) * 1e4
    return w.loc[PANEL_START:]


def controls(w: pd.DataFrame, spread: str) -> list[str]:
    cols = [f"{spread}_level_z", f"{spread}_chg_4w_known", "spy_ret_4w", "vix_chg_4w"]
    return [c for c in cols if c in w.columns]


def hac(w: pd.DataFrame, y: str, x: list[str], lags: int) -> tuple[float, float, int]:
    d = w[[y, *x]].dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sm.OLS(d[y], sm.add_constant(d[x])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.params[x[0]]), float(r.tvalues[x[0]]), len(d)


def regression_table(w: pd.DataFrame, spreads: list[str]) -> pd.DataFrame:
    rows = []
    targets = [(f"{s}_fwd_{h}w", h, s, SPREADS[s] + " change (bp)") for s in spreads for h in HORIZONS]
    if "hyg_fwd_13w" in w.columns:
        base = spreads[0]
        targets += [(f"hyg_fwd_{h}w", h, base, "HYG excess return (bp)") for h in HORIZONS]
    for y, h, ctrl_spread, label in targets:
        for sig in LIQUIDITY_SIGNALS:
            for with_ctrl in (False, True):
                x = [sig] + (controls(w, ctrl_spread) if with_ctrl else [])
                coef, t, n = hac(w, y, x, h)
                rows.append({"outcome": f"{label}, next {h}w", "liquidity signal": sig.replace("net_pct_", "% change ")
                             .replace("w", " wk"), "controls": "yes" if with_ctrl else "no",
                             "coef per +1%": coef, "t (HAC)": t, "weeks": n})
    return pd.DataFrame(rows)


def robustness_table(w: pd.DataFrame, spreads: list[str]) -> pd.DataFrame:
    crisis = (((w.index >= "2008-06-01") & (w.index < "2010-01-01"))
              | ((w.index >= "2020-01-01") & (w.index < "2021-01-01")))
    masks = {"2003–2018": w.index < "2019-01-01", "2019 onward": w.index >= "2019-01-01",
             "excluding 2008–09 and 2020": ~crisis}
    rows = []
    for s in spreads:
        for name, mask in masks.items():
            coef, t, n = hac(w[mask], f"{s}_fwd_13w", ["net_pct_13w", *controls(w, s)], 13)
            rows.append({"spread": SPREADS[s], "period": name, "coef per +1%": coef, "t (HAC)": t, "weeks": n,
                         "spread_col": s})
    return pd.DataFrame(rows)


def quintile_table(w: pd.DataFrame, spreads: list[str]) -> pd.DataFrame:
    q = pd.qcut(w["net_pct_13w"], 5, labels=["Q1 drained most", "Q2", "Q3", "Q4", "Q5 added most"])
    agg = {"weeks": ("net_pct_13w", "count"), "avg 13w liquidity growth %": ("net_pct_13w", "mean")}
    for s in spreads:
        agg[f"{SPREADS[s]} next 13w (bp)"] = (f"{s}_fwd_13w", "mean")
    if "hyg_fwd_13w" in w.columns:
        agg["HYG excess next 13w (bp)"] = ("hyg_fwd_13w", "mean")
    out = w.groupby(q, observed=True).agg(**agg).reset_index().rename(columns={"net_pct_13w": "quintile"})
    out["weeks"] = out["weeks"].astype(int)
    return out


def baseline_episodes(spread: pd.Series) -> pd.DataFrame:
    """Large 20-day widenings with no Fed action nearby: what happened next without intervention."""
    s = spread.dropna()
    prior = s - s.shift(20)
    threshold = float(prior.quantile(STRESS_PERCENTILE))
    action_pos = [int(s.index.searchsorted(pd.Timestamp(a.date))) for a in FED_ACTIONS]
    rows, last = [], -EPISODE_GAP
    for i in np.flatnonzero((prior >= threshold).to_numpy()):
        if i - last < EPISODE_GAP or i + 20 >= len(s):
            continue
        if any(abs(i - p) <= ACTION_EXCLUSION for p in action_pos):
            continue
        rows.append({"date": s.index[i].date(), "prior_20d": float(prior.iloc[i]),
                     "next_20d": float(s.iloc[i + 20] - s.iloc[i])})
        last = i
    out = pd.DataFrame(rows)
    out.attrs["threshold"] = threshold
    return out


def event_section(df: pd.DataFrame, col: str) -> tuple[list[str], dict[str, float]]:
    s = df[col].dropna()
    events = event_study(s, horizons=(20, 60))
    for h in (20, 60):
        events[f"rank_{h}d"] = events[f"rank_{h}d"] * 100
    table = events.rename(columns={"spread": "level", "prior_20d": "prior 20d", "next_20d": "next 20d",
                                   "rank_20d": "rank 20d (%)", "next_60d": "next 60d", "rank_60d": "rank 60d (%)"})
    base = baseline_episodes(s)
    backstops = events[(events["type"] == "ease") & (events["prior_20d"] >= base.attrs["threshold"])]
    lines = [f"### {SPREADS[col]}", "", to_markdown(table, decimals=0), ""]
    changes = (s.shift(-20) - s).dropna()
    lines += [f"Typical 20-day change: median {changes.median():+.0f} bp; "
              f"10th / 90th percentile {changes.quantile(0.1):+.0f} / {changes.quantile(0.9):+.0f} bp.", ""]
    if not base.empty:
        lines += [
            f"**Baseline without intervention:** {len(base)} episodes where the spread had widened by at least "
            f"{base.attrs['threshold']:.0f} bp over 20 days (top {1 - STRESS_PERCENTILE:.0%}) with no Fed action "
            f"within {ACTION_EXCLUSION} trading days. Next 20 days: median {base['next_20d'].median():+.0f} bp, "
            f"{(base['next_20d'] < 0).mean():.0%} tightened.",
        ]
        if not backstops.empty:
            lines += [
                f"Easing actions taken after a comparable widening ({len(backstops)}): next 20 days median "
                f"{backstops['next_20d'].median():+.0f} bp, {(backstops['next_20d'] < 0).mean():.0%} tightened.",
            ]
        lines.append("")
    summary = {"baseline_median": float(base["next_20d"].median()) if not base.empty else float("nan"),
               "backstop_median": float(backstops["next_20d"].median()) if not backstops.empty else float("nan"),
               "backstops": len(backstops)}
    return lines, summary


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.data_path)
    spreads = [c for c in SPREADS if c in df.columns and int(df[c].notna().sum()) >= MIN_SPREAD_HISTORY]
    if not spreads:
        raise SystemExit("No spread with enough history in the dataset.")
    liq = net_liquidity(df)
    snap = liquidity_snapshot(liq, weeks=13)
    w = weekly_panel(df, spreads)
    base = spreads[0]

    out = [
        "# Fed liquidity and credit spreads",
        "",
        f"Data: `{Path(args.data_path).name}` ({df.index.min().date()} → {df.index.max().date()}); weekly Friday panel "
        f"from {w.index.min().date()} ({len(w)} weeks). Spreads tested: {', '.join(SPREADS[s] for s in spreads)}"
        + ("; HYG excess return (HYG − 0.85 × IEI)." if "hyg_fwd_13w" in w.columns else "."),
        "",
        "Net liquidity = Fed total assets (WALCL) − Treasury General Account (WTREGEN) − overnight reverse repo "
        "(RRPONTSYD), in $ billions. The H.4.1 balance sheet is dated from the Friday after its Thursday release; "
        "control variables use only spread values published by Friday's close.",
        "",
        f"**Latest ({snap['as_of']:%Y-%m-%d}):** net liquidity ${snap['net']:,.0f}bn (assets {snap['assets']:,.0f} − "
        f"TGA {snap['tga']:,.0f} − RRP {snap['rrp']:,.0f}); 13-week change {snap['net_change']:+,.0f}bn "
        f"({snap['net_change_pct']:+.1f}%).",
        "",
        "## 1. Same-period correlation",
        "",
    ]
    same = pd.concat([w["net_pct_13w"], w[f"{base}_known"].diff(13)], axis=1).dropna()
    corr = float(same.corr().iloc[0, 1])
    out += [
        f"Correlation of 13-week liquidity growth with the 13-week change in {SPREADS[base]} over the same weeks: "
        f"**{corr:+.2f}**. A positive number means the Fed adds liquidity *while* spreads widen: it responds to "
        "stress, so a raw correlation cannot show that liquidity calms spreads.",
        "",
        "## 2. Does liquidity growth predict future spread changes?",
        "",
        "OLS on weekly data with HAC (Newey–West) t-stats. A negative coefficient on spread changes means more "
        "liquidity is followed by tightening; a positive coefficient on HYG excess returns means it is followed by "
        "better returns. Controls: spread z-score vs its 3-year history, 4-week spread change, 4-week S&P 500 "
        "return, 4-week VIX change. |t| above 2 would count as evidence.",
        "",
        to_markdown(regressions := regression_table(w, spreads)),
        "",
        "### Robustness: next-13-week spread change on 13-week liquidity growth, with controls",
        "",
        to_markdown((robust := robustness_table(w, spreads)).drop(columns="spread_col")),
        "",
        "### By quintile of 13-week liquidity growth",
        "",
        to_markdown(quintile_table(w, spreads), decimals=1),
        "",
        "## 3. Event study: spread moves after major Fed actions",
        "",
        "Changes in bp over trading days after the announcement (the first trading day on or after it). "
        "*Rank* places the next-20-day change among every 20-day change in the history: 0% is the sharpest "
        "tightening on record, 50% is typical.",
        "",
    ]
    event_summary = {}
    for col in spreads:
        lines, event_summary[col] = event_section(df, col)
        out += lines

    if "oas_ccc" in df.columns and df["oas_ccc"].notna().sum() > 100:
        ccc = df["oas_ccc"].reindex(w.index)
        c = pd.DataFrame({"liq": w["net_pct_13w"], "fwd": ccc.shift(-13) - ccc}).dropna()
        out += [
            "## CCC (descriptive only)",
            "",
            f"ICE BofA CCC & lower OAS is available from {df['oas_ccc'].dropna().index.min().date()} "
            f"({len(c)} weeks with a 13-week outcome), which is too short for a test. Correlation of 13-week liquidity "
            f"growth with the next 13-week CCC change: {c.corr().iloc[0, 1]:+.2f}.",
            "",
        ]

    max_t = float(regressions["t (HAC)"].abs().max())
    flips = [SPREADS[sp] for sp, g in robust.groupby("spread_col") if g["coef per +1%"].gt(0).nunique() > 1]
    predictive = max_t >= 2 and not flips
    out += ["## Conclusion", ""]
    out.append(
        f"* **Liquidity quantity as a predictor: {'supported' if predictive else 'not supported'}.** The largest |t| "
        f"across {len(regressions)} regressions is {max_t:.2f}"
        + (f", and the sign flips across sub-periods for {', '.join(flips)}" if flips else "")
        + (". Net liquidity is therefore not used as a model feature." if not predictive
           else ". Consider testing it as a model feature in the walk-forward evaluation.")
    )
    for col, e in event_summary.items():
        if e["backstops"]:
            out.append(
                f"* **Backstops in stress ({SPREADS[col]}):** {e['backstops']} easing actions came after a top-5% "
                f"widening; the median next-20-day change was {e['backstop_median']:+.0f} bp against "
                f"{e['baseline_median']:+.0f} bp for comparable widenings without Fed action. The sample is small, "
                "and the announcements are discrete policy decisions, not a measurable series to forecast with."
            )
    out += ["", "The dashboard's Fed Liquidity tab shows net liquidity and these events as context.", ""]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(out), encoding="utf-8")
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
