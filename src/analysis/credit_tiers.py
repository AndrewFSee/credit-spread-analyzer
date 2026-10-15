"""
Credit-tier monitor: ICE BofA option-adjusted spreads by rating tier.

Stress usually shows up first in the weakest borrowers (CCC), then spreads
up the quality ladder.  These helpers summarise where each tier sits, how
far apart the tiers are, and how that changed over a recent window.

All spreads are in basis points.  FRED serves only ~3 years of these series;
a longer licensed history for any tier can be spliced in via
``data/external/<column>.csv`` (see ``config.settings.CREDIT_TIERS``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config.settings import CREDIT_TIERS

TRADING_DAYS_PER_WEEK = 5


def tier_frame(df: pd.DataFrame, tiers: Optional[dict[str, str]] = None) -> pd.DataFrame:
    """Spreads with one column per rating tier (safest first), for tiers that have data."""
    tiers = tiers or CREDIT_TIERS
    cols = {name: df[col] for name, col in tiers.items() if col in df.columns and df[col].notna().any()}
    if not cols:
        raise ValueError("No credit-tier spread columns found in DataFrame.")
    return pd.DataFrame(cols).dropna(how="all")


def as_of(tiers: pd.DataFrame) -> pd.Timestamp:
    """Latest date on which every available tier has a value."""
    complete = tiers.dropna()
    return complete.index[-1] if not complete.empty else tiers.index[-1]


def _value_before(s: pd.Series, date: pd.Timestamp, lookback: int) -> float:
    """Value *lookback* observations before *date* (NaN if the history is too short)."""
    s = s.loc[:date].dropna()
    return float(s.iloc[-1 - lookback]) if len(s) > lookback else np.nan


def tier_snapshot(tiers: pd.DataFrame, lookback: int = 4 * TRADING_DAYS_PER_WEEK) -> pd.DataFrame:
    """One row per tier: current spread, change over *lookback* trading days, and history context.

    ``percentile`` is the share of days in the available history with a spread
    at or below today's (1.0 = the widest on record in this data).
    """
    date = as_of(tiers)
    rows = {}
    for tier in tiers.columns:
        s = tiers[tier].loc[:date].dropna()
        now = float(s.iloc[-1])
        rows[tier] = {
            "spread": now,
            "change": now - _value_before(s, date, lookback),
            "average": float(s.mean()),
            "low": float(s.min()),
            "high": float(s.max()),
            "percentile": float((s <= now).mean()),
            "history_start": s.index[0],
        }
    out = pd.DataFrame(rows).T
    numeric = ["spread", "change", "average", "low", "high", "percentile"]
    out[numeric] = out[numeric].astype(float)
    out.attrs["as_of"] = date
    return out


def tier_gaps(tiers: pd.DataFrame, date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Lower-triangular matrix of spread differences: row tier minus (safer) column tier."""
    date = date or as_of(tiers)
    levels = tiers.loc[:date].ffill().iloc[-1]
    names = list(tiers.columns)
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for i, riskier in enumerate(names):
        for safer in names[:i]:
            out.loc[riskier, safer] = levels[riskier] - levels[safer]
    return out


def tier_gap_changes(tiers: pd.DataFrame, lookback: int = 4 * TRADING_DAYS_PER_WEEK) -> pd.DataFrame:
    """Change in each tier-to-tier gap over the last *lookback* trading days."""
    date = as_of(tiers)
    complete = tiers.loc[:date].dropna()
    if len(complete) <= lookback:
        return tier_gaps(tiers, date) * np.nan
    earlier = complete.index[-1 - lookback]
    return tier_gaps(tiers, date) - tier_gaps(tiers, earlier)


def decompression(tiers: pd.DataFrame, riskier: str = "CCC", safer: str = "B") -> pd.Series:
    """Gap between two tiers over time; a rising CCC – B gap means the weakest credits are cracking first."""
    if riskier not in tiers or safer not in tiers:
        raise ValueError(f"Need both '{riskier}' and '{safer}' tiers.")
    return (tiers[riskier] - tiers[safer]).dropna().rename(f"{riskier} – {safer}")


def widening_breadth(snapshot: pd.DataFrame, threshold_bp: float = 10.0) -> tuple[int, int]:
    """How many tiers widened by more than *threshold_bp* over the snapshot window, out of how many."""
    changes = snapshot["change"].dropna()
    return int((changes > threshold_bp).sum()), int(len(changes))
