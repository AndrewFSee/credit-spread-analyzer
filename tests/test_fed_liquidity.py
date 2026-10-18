"""Tests for src/analysis/fed_liquidity.py (synthetic data, no network)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _df(n: int = 120) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "fed_assets": np.linspace(4_000_000, 7_000_000, n),   # $ millions
        "tga": np.full(n, 500_000.0),                          # $ millions
        "on_rrp": [np.nan] * 10 + [200.0] * (n - 10),          # $ billions; no reading at first
    }, index=idx)


class TestFedLiquidity(unittest.TestCase):

    def test_net_liquidity_units_and_missing_rrp(self) -> None:
        from src.analysis.fed_liquidity import net_liquidity

        liq = net_liquidity(_df())
        self.assertAlmostEqual(liq["assets"].iloc[0], 4000.0)
        self.assertAlmostEqual(liq["rrp"].iloc[0], 0.0)          # missing early RRP counts as zero
        self.assertAlmostEqual(liq["net"].iloc[0], 4000.0 - 500.0)
        self.assertAlmostEqual(liq["net"].iloc[-1], 7000.0 - 500.0 - 200.0)
        with self.assertRaises(ValueError):
            net_liquidity(pd.DataFrame({"tga": [1.0]}))

    def test_snapshot_changes(self) -> None:
        from src.analysis.fed_liquidity import liquidity_snapshot, net_liquidity

        liq = net_liquidity(_df())
        snap = liquidity_snapshot(liq, weeks=4)
        self.assertAlmostEqual(snap["net_change"], liq["net"].iloc[-1] - liq["net"].iloc[-21])
        self.assertAlmostEqual(snap["tga_change"], 0.0)
        self.assertGreater(snap["net_change_pct"], 0)

    def test_event_study(self) -> None:
        from src.analysis.fed_liquidity import FedAction, event_study

        idx = pd.bdate_range("2020-01-01", periods=200)
        spread = pd.Series(500.0, index=idx)
        spread.iloc[100:] = np.linspace(500, 300, 100)          # tightens after the action
        out = event_study(spread, [FedAction(str(idx[100].date()), "test", "ease"),
                                   FedAction("1990-01-01", "before data", "ease")], horizons=(20,))
        self.assertEqual(len(out), 1)                            # the out-of-range action is skipped
        self.assertAlmostEqual(out.loc[0, "next_20d"], spread.iloc[120] - spread.iloc[100])
        self.assertAlmostEqual(out.loc[0, "prior_20d"], 0.0)
        self.assertLess(out.loc[0, "rank_20d"], 0.5)

    def test_action_filter(self) -> None:
        from src.analysis.fed_liquidity import FED_ACTIONS, fed_actions

        recent = fed_actions(pd.Timestamp("2020-01-01"))
        self.assertTrue(all(pd.Timestamp(a.date) >= pd.Timestamp("2020-01-01") for a in recent))
        self.assertLess(len(recent), len(FED_ACTIONS))
        self.assertTrue({a.kind for a in FED_ACTIONS} <= {"ease", "tighten"})


if __name__ == "__main__":
    unittest.main()
