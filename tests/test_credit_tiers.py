"""Tests for src/analysis/credit_tiers.py (synthetic data, no network)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _df(n: int = 60) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-01", periods=n)
    base = {"oas_aaa": 40, "oas_aa": 55, "oas_a": 70, "bbb_spread": 100, "oas_bb": 190, "oas_b": 310, "oas_ccc": 900}
    df = pd.DataFrame({c: np.full(n, float(v)) for c, v in base.items()}, index=idx)
    k = min(20, n)
    df.iloc[-k:, df.columns.get_loc("oas_ccc")] = np.linspace(910, 1100, k)  # CCC widens in the last 4 weeks
    df["dgs10"] = 4.0  # unrelated column is ignored
    return df


class TestCreditTiers(unittest.TestCase):

    def test_tier_frame_orders_safest_first_and_skips_missing(self) -> None:
        from src.analysis.credit_tiers import tier_frame

        df = _df().drop(columns="oas_aa")
        tiers = tier_frame(df)
        self.assertEqual(list(tiers.columns), ["AAA", "A", "BBB", "BB", "B", "CCC"])
        with self.assertRaises(ValueError):
            tier_frame(pd.DataFrame({"x": [1.0]}))

    def test_snapshot_change_and_percentile(self) -> None:
        from src.analysis.credit_tiers import tier_frame, tier_snapshot

        snap = tier_snapshot(tier_frame(_df()), lookback=20)
        self.assertAlmostEqual(snap.loc["CCC", "spread"], 1100)
        self.assertAlmostEqual(snap.loc["CCC", "change"], 1100 - 900)   # 20 rows earlier CCC was 900
        self.assertAlmostEqual(snap.loc["CCC", "percentile"], 1.0)      # widest in the data
        self.assertAlmostEqual(snap.loc["BB", "change"], 0.0)
        self.assertEqual(snap.attrs["as_of"], _df().index[-1])

    def test_gap_matrix_is_lower_triangular(self) -> None:
        from src.analysis.credit_tiers import tier_frame, tier_gap_changes, tier_gaps

        tiers = tier_frame(_df())
        gaps = tier_gaps(tiers)
        self.assertAlmostEqual(gaps.loc["CCC", "B"], 1100 - 310)
        self.assertAlmostEqual(gaps.loc["BB", "AAA"], 190 - 40)
        self.assertTrue(np.isnan(gaps.loc["AAA", "CCC"]))
        self.assertTrue(np.isnan(gaps.loc["B", "B"]))
        changes = tier_gap_changes(tiers, lookback=20)
        self.assertAlmostEqual(changes.loc["CCC", "AAA"], 200)
        self.assertAlmostEqual(changes.loc["B", "BB"], 0)

    def test_decompression_and_breadth(self) -> None:
        from src.analysis.credit_tiers import decompression, tier_frame, tier_snapshot, widening_breadth

        tiers = tier_frame(_df())
        gap = decompression(tiers)
        self.assertAlmostEqual(gap.iloc[0], 900 - 310)
        self.assertAlmostEqual(gap.iloc[-1], 1100 - 310)
        self.assertEqual(widening_breadth(tier_snapshot(tiers)), (1, 7))
        with self.assertRaises(ValueError):
            decompression(tiers.drop(columns="B"))

    def test_short_history_gives_nan_changes(self) -> None:
        from src.analysis.credit_tiers import tier_frame, tier_gap_changes, tier_snapshot

        tiers = tier_frame(_df(10))
        self.assertTrue(np.isnan(tier_snapshot(tiers, lookback=20).loc["CCC", "change"]))
        self.assertTrue(tier_gap_changes(tiers, lookback=20).isna().all().all())


if __name__ == "__main__":
    unittest.main()
