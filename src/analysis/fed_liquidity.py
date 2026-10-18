"""
Fed liquidity monitor.

Net liquidity = Fed total assets − Treasury General Account − overnight reverse
repo: the part of the Fed's balance sheet that is actually available to
markets.  All outputs are in $ billions.

What the walk-forward tests found (scripts/evaluate_fed_liquidity.py,
reports/fed_liquidity.md): week-to-week changes in net liquidity do **not**
predict credit spreads, because the Fed adds liquidity while spreads are
already widening.  Crisis *backstop announcements* (March 2009, March 2020,
the 2020 corporate credit facilities) were followed by unusually sharp
tightening.  The monitor is therefore context, not a model input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class FedAction:
    date: str
    label: str
    kind: str  # "ease" (liquidity / backstop) or "tighten"


FED_ACTIONS: tuple[FedAction, ...] = (
    FedAction("2008-11-25", "QE1 announced (MBS / agency purchases)", "ease"),
    FedAction("2009-03-18", "QE1 expanded ($1.15T incl. Treasuries)", "ease"),
    FedAction("2010-11-03", "QE2 ($600bn Treasuries)", "ease"),
    FedAction("2012-09-13", "QE3 (open-ended MBS)", "ease"),
    FedAction("2013-05-22", "Taper tantrum (Bernanke testimony)", "tighten"),
    FedAction("2017-06-14", "QT plan announced", "tighten"),
    FedAction("2019-10-11", "Bill purchases after repo spike", "ease"),
    FedAction("2020-03-15", "QE + rate cut to zero", "ease"),
    FedAction("2020-03-23", "Unlimited QE + corporate credit facilities", "ease"),
    FedAction("2020-04-09", "Facilities extended to high yield / fallen angels", "ease"),
    FedAction("2021-11-03", "Taper announced", "tighten"),
    FedAction("2022-05-04", "QT announced (from June 2022)", "tighten"),
    FedAction("2023-03-12", "Bank Term Funding Program after SVB", "ease"),
)


def net_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    """Columns ``assets``, ``tga``, ``rrp`` and ``net`` in $ billions on *df*'s calendar.

    Reverse-repo days with no reading (before the facility was used) count as zero.
    """
    missing = [c for c in ("fed_assets", "tga") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Fed liquidity columns: {missing}")
    out = pd.DataFrame(index=df.index)
    out["assets"] = df["fed_assets"] / 1000.0
    out["tga"] = df["tga"] / 1000.0
    rrp = df["on_rrp"] if "on_rrp" in df.columns else pd.Series(0.0, index=df.index)
    out["rrp"] = rrp.ffill(limit=5).fillna(0.0)
    out = out.dropna(subset=["assets", "tga"])
    out["net"] = out["assets"] - out["tga"] - out["rrp"]
    return out


def liquidity_snapshot(liq: pd.DataFrame, weeks: int = 13) -> dict[str, float]:
    """Latest levels and their change over *weeks* (5 trading days per week)."""
    lookback = weeks * 5
    now = liq.iloc[-1]
    then = liq.iloc[-1 - lookback] if len(liq) > lookback else liq.iloc[0]
    return {
        "as_of": liq.index[-1],
        "net": float(now["net"]),
        "net_change": float(now["net"] - then["net"]),
        "net_change_pct": float((now["net"] / then["net"] - 1) * 100),
        "assets": float(now["assets"]),
        "assets_change": float(now["assets"] - then["assets"]),
        "tga": float(now["tga"]),
        "tga_change": float(now["tga"] - then["tga"]),
        "rrp": float(now["rrp"]),
        "rrp_change": float(now["rrp"] - then["rrp"]),
    }


def fed_actions(start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> list[FedAction]:
    """Fed actions dated within [*start*, *end*]."""
    return [a for a in FED_ACTIONS
            if (start is None or pd.Timestamp(a.date) >= start) and (end is None or pd.Timestamp(a.date) <= end)]


def event_study(spread: pd.Series, actions: Optional[list[FedAction]] = None,
                horizons: tuple[int, ...] = (20, 60)) -> pd.DataFrame:
    """Spread change before and after each Fed action, in trading days.

    ``rank_<h>d`` is where the post-event change falls among all *h*-day changes
    (0% = the most tightening on record in this data).
    """
    s = spread.dropna()
    actions = list(FED_ACTIONS) if actions is None else actions
    rows = []
    for action in actions:
        t = pd.Timestamp(action.date)
        if s.empty or t < s.index[0] or t > s.index[-1]:
            continue
        i = int(s.index.searchsorted(t))
        row = {"date": s.index[i].date(), "action": action.label, "type": action.kind,
               "spread": float(s.iloc[i]),
               "prior_20d": float(s.iloc[i] - s.iloc[max(i - 20, 0)])}
        for h in horizons:
            if i + h < len(s):
                change = float(s.iloc[i + h] - s.iloc[i])
                all_changes = (s.shift(-h) - s).dropna()
                row[f"next_{h}d"] = change
                row[f"rank_{h}d"] = float((all_changes <= change).mean())
            else:
                row[f"next_{h}d"] = float("nan")
                row[f"rank_{h}d"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
