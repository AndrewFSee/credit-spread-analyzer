# Fed liquidity and credit spreads

Data: `market_data_1990-01-01_2026-10-18.parquet` (1990-01-02 → 2026-10-16); weekly Friday panel from 2003-06-06 (1178 weeks). Spreads tested: Moody's Baa – 10y; HYG excess return (HYG − 0.85 × IEI).

Net liquidity = Fed total assets (WALCL) − Treasury General Account (WTREGEN) − overnight reverse repo (RRPONTSYD), in $ billions. The H.4.1 balance sheet is dated from the Friday after its Thursday release; control variables use only spread values published by Friday's close.

**Latest (2026-10-16):** net liquidity $5,872bn (assets 6,756 − TGA 883 − RRP 0); 13-week change -89bn (-1.5%).

## 1. Same-period correlation

Correlation of 13-week liquidity growth with the 13-week change in Moody's Baa – 10y over the same weeks: **+0.57**. A positive number means the Fed adds liquidity *while* spreads widen: it responds to stress, so a raw correlation cannot show that liquidity calms spreads.

## 2. Does liquidity growth predict future spread changes?

OLS on weekly data with HAC (Newey–West) t-stats. A negative coefficient on spread changes means more liquidity is followed by tightening; a positive coefficient on HYG excess returns means it is followed by better returns. Controls: spread z-score vs its 3-year history, 4-week spread change, 4-week S&P 500 return, 4-week VIX change. |t| above 2 would count as evidence.

| outcome | liquidity signal | controls | coef per +1% | t (HAC) | weeks |
|---|---|---|---|---|---|
| Moody's Baa – 10y change (bp), next 4w | % change 4 wk | no | +0.43 | +0.97 | 1174 |
| Moody's Baa – 10y change (bp), next 4w | % change 4 wk | yes | +0.24 | +0.81 | 1174 |
| Moody's Baa – 10y change (bp), next 4w | % change 13 wk | no | -0.01 | -0.05 | 1174 |
| Moody's Baa – 10y change (bp), next 4w | % change 13 wk | yes | -0.02 | -0.18 | 1174 |
| Moody's Baa – 10y change (bp), next 13w | % change 4 wk | no | -0.09 | -0.11 | 1165 |
| Moody's Baa – 10y change (bp), next 13w | % change 4 wk | yes | -0.40 | -0.65 | 1165 |
| Moody's Baa – 10y change (bp), next 13w | % change 13 wk | no | -0.22 | -1.20 | 1165 |
| Moody's Baa – 10y change (bp), next 13w | % change 13 wk | yes | -0.23 | -1.02 | 1165 |
| HYG excess return (bp), next 4w | % change 4 wk | no | -3.09 | -0.59 | 980 |
| HYG excess return (bp), next 4w | % change 4 wk | yes | -3.46 | -0.84 | 980 |
| HYG excess return (bp), next 4w | % change 13 wk | no | +1.23 | +0.35 | 980 |
| HYG excess return (bp), next 4w | % change 13 wk | yes | +0.97 | +0.30 | 980 |
| HYG excess return (bp), next 13w | % change 4 wk | no | +5.12 | +1.00 | 971 |
| HYG excess return (bp), next 13w | % change 4 wk | yes | +8.86 | +1.83 | 971 |
| HYG excess return (bp), next 13w | % change 13 wk | no | +1.65 | +1.00 | 971 |
| HYG excess return (bp), next 13w | % change 13 wk | yes | +2.25 | +0.75 | 971 |

### Robustness: next-13-week spread change on 13-week liquidity growth, with controls

| spread | period | coef per +1% | t (HAC) | weeks |
|---|---|---|---|---|
| Moody's Baa – 10y | 2003–2018 | -0.40 | -1.21 | 786 |
| Moody's Baa – 10y | 2019 onward | -0.38 | -1.52 | 379 |
| Moody's Baa – 10y | excluding 2008–09 and 2020 | +0.39 | +0.72 | 1038 |

### By quintile of 13-week liquidity growth

| quintile | weeks | avg 13w liquidity growth % | Moody's Baa – 10y next 13w (bp) | HYG excess next 13w (bp) |
|---|---|---|---|---|
| Q1 drained most | 236 | -4.2 | -9.3 | 191.6 |
| Q2 | 235 | -0.8 | -1.4 | 123.2 |
| Q3 | 236 | 0.9 | 2.7 | -27.2 |
| Q4 | 235 | 2.8 | 7.7 | -141.9 |
| Q5 added most | 236 | 15.7 | -6.0 | 75.9 |

## 3. Event study: spread moves after major Fed actions

Changes in bp over trading days after the announcement (the first trading day on or after it). *Rank* places the next-20-day change among every 20-day change in the history: 0% is the sharpest tightening on record, 50% is typical.

### Moody's Baa – 10y

| date | action | type | level | prior 20d | next 20d | rank 20d (%) | next 60d | rank 60d (%) |
|---|---|---|---|---|---|---|---|---|
| 2008-11-25 | QE1 announced (MBS / agency purchases) | ease | 601 | 42 | -11 | 21 | -78 | 1 |
| 2009-03-18 | QE1 expanded ($1.15T incl. Treasuries) | ease | 588 | 67 | -36 | 2 | -209 | 0 |
| 2010-11-03 | QE2 ($600bn Treasuries) | ease | 316 | 3 | -17 | 11 | -48 | 4 |
| 2012-09-13 | QE3 (open-ended MBS) | ease | 321 | -6 | -34 | 2 | -26 | 15 |
| 2013-05-22 | Taper tantrum (Bernanke testimony) | tighten | 278 | -2 | 15 | 88 | -8 | 40 |
| 2017-06-14 | QT plan announced | tighten | 219 | -5 | -10 | 24 | -2 | 53 |
| 2019-10-11 | Bill purchases after repo spike | ease | 222 | 1 | -11 | 20 | -17 | 26 |
| 2020-03-16 | QE + rate cut to zero | ease | 338 | 134 | -2 | 47 | -52 | 3 |
| 2020-03-23 | Unlimited QE + corporate credit facilities | ease | 431 | 218 | -102 | 0 | -144 | 1 |
| 2020-04-09 | Facilities extended to high yield / fallen angels | ease | 361 | 41 | -24 | 6 | -80 | 1 |
| 2021-11-03 | Taper announced | tighten | 170 | -14 | 10 | 81 | 23 | 83 |
| 2022-05-04 | QT announced (from June 2022) | tighten | 208 | 31 | 2 | 61 | 30 | 88 |
| 2023-03-13 | Bank Term Funding Program after SVB | ease | 214 | 29 | -9 | 24 | -8 | 40 |

Typical 20-day change: median -1 bp; 10th / 90th percentile -18 / +18 bp.

**Baseline without intervention:** 29 episodes where the spread had widened by at least 27 bp over 20 days (top 5%) with no Fed action within 20 trading days. Next 20 days: median +1 bp, 48% tightened.
Easing actions taken after a comparable widening (6): next 20 days median -18 bp, 100% tightened.

## CCC (descriptive only)

ICE BofA CCC & lower OAS is available from 2023-10-17 (138 weeks with a 13-week outcome), which is too short for a test. Correlation of 13-week liquidity growth with the next 13-week CCC change: -0.01.

## Conclusion

* **Liquidity quantity as a predictor: not supported.** The largest |t| across 16 regressions is 1.83, and the sign flips across sub-periods for Moody's Baa – 10y. Net liquidity is therefore not used as a model feature.
* **Backstops in stress (Moody's Baa – 10y):** 6 easing actions came after a top-5% widening; the median next-20-day change was -18 bp against +1 bp for comparable widenings without Fed action. The sample is small, and the announcements are discrete policy decisions, not a measurable series to forecast with.

The dashboard's Fed Liquidity tab shows net liquidity and these events as context.
