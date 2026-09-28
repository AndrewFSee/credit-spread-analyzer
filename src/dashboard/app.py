"""
Streamlit dashboard for the Credit Spread Analysis & Prediction Platform.

Run from the project root with:
    python scripts/run_dashboard.py
or:
    python -m streamlit run src/dashboard/app.py

(``python -m`` works even when pip's Scripts folder, which holds the
``streamlit`` executable, is not on PATH.)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from project root when run directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import (
    DATA_DIR,
    DEFAULT_START_DATE,
    EXPOSURE_TARGET_VOL,
    FRED_API_KEY,
    HOLDOUT_START,
    RECOMMENDED_MODEL,
    RECOMMENDED_MODEL_HY,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Spread Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SPREAD_LABELS = {
    "baa_spread": "Moody's Baa – 10y",
    "aaa_spread": "Moody's Aaa – 10y",
    "hy_spread": "ICE HY OAS",
    "ig_spread": "ICE IG OAS",
    "bbb_spread": "ICE BBB OAS",
}
MODEL_CHOICES = ["ensemble", "composite", "ridge", "lightgbm", "xgboost", "random_forest"]
TARGET_BAA = "Credit spread change"
TARGET_HYG = "HYG excess return (tradable)"
EXPOSURE_CHOICES = {
    "vol_regime": "Vol target × (1 − P(credit stress))",
    "vol": "Vol target only",
    "regime": "1 − P(credit stress)",
    "zscore": "Spread z-score in/out",
    "widening": "Spread widening in/out",
}
REGIME_STATES_REALTIME = 3  # the exposure overlay was validated with three regimes

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

start_date = st.sidebar.date_input("Start date", value=pd.Timestamp(DEFAULT_START_DATE))
end_date = st.sidebar.date_input("End date", value=pd.Timestamp("today"))
forecast_target = st.sidebar.radio(
    "Forecast target",
    [TARGET_BAA, TARGET_HYG],
    help="Credit spread: Moody's Baa – 10y index (partly stale pricing). "
         "HYG: 5-day return of HYG minus 0.85 × IEI, entered the day after the signal.",
)
is_hyg = forecast_target == TARGET_HYG
horizon = st.sidebar.selectbox(
    "Forecast horizon (trading days)", options=[5] if is_hyg else [5, 20], index=0,
)
default_model = RECOMMENDED_MODEL_HY if is_hyg else RECOMMENDED_MODEL.get(horizon, "composite")
model_type = st.sidebar.selectbox(
    "Forecast model",
    options=MODEL_CHOICES,
    index=MODEL_CHOICES.index(default_model),
    help="Default is the model with the best out-of-sample record for this target and horizon.",
)
n_regimes = st.sidebar.slider("Number of Regimes", min_value=2, max_value=4, value=3)
fred_api_key = st.sidebar.text_input("FRED API Key (optional)", value=FRED_API_KEY, type="password")

st.sidebar.markdown("---")
refresh = st.sidebar.button("🔄 Refresh Data")

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


def _synthetic_demo_data(start: str, end: str) -> pd.DataFrame:
    """Generate synthetic demo data for UI testing."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(start=start, end=end)
    n = len(idx)
    df = pd.DataFrame(index=idx)
    df["baa_spread"] = np.clip(220 + np.cumsum(rng.normal(0, 2, n)), 80, 700)
    df["aaa_spread"] = np.clip(0.55 * df["baa_spread"] + rng.normal(0, 3, n), 30, 400)
    df["t10y2y"] = rng.normal(1.0, 0.8, n)
    df["dgs10"] = np.clip(4 + np.cumsum(rng.normal(0, 0.03, n)), 0.5, 9)
    df["tbill_3m"] = np.clip(2 + np.cumsum(rng.normal(0, 0.02, n)), 0, 8)
    df["vix"] = 18 + np.abs(np.cumsum(rng.normal(0, 0.5, n)) % 30)
    df["sp500"] = 1000 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    df["sp500_return"] = df["sp500"].pct_change()
    df["fed_funds"] = np.clip(2 + np.cumsum(rng.normal(0, 0.02, n)), 0, 8)
    return df.dropna()


@st.cache_data(show_spinner="Loading market data …", ttl=3600)
def load_data(start: str, end: str, api_key: str) -> tuple[pd.DataFrame, bool]:
    """Fetch and cache market data.  Returns ``(df, is_synthetic)``."""
    try:
        from src.data.fetcher import fetch_all_data

        df = fetch_all_data(start_date=start, end_date=end, api_key=api_key, cache_dir=DATA_DIR)
        if df.empty:
            raise ValueError("no data returned")
        return df, False
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Data fetch failed: {exc}. Using synthetic demo data.")
        return _synthetic_demo_data(start, end), True


@st.cache_data(show_spinner="Running walk-forward forecasts …")
def run_forecasts(
    df: pd.DataFrame, horizon: int, model_type: str, hyg: bool = False, spread_col: str | None = None
) -> dict:
    """Walk-forward out-of-sample forecasts plus a live forecast."""
    from src.features.engineering import build_feature_matrix, build_hy_feature_matrix
    from src.models.ml_models import (
        DEFAULT_SCALE_COL,
        RETURN_COMPOSITE_WEIGHTS,
        TREE_MODELS,
        compute_metrics,
        feature_importance,
        make_model,
        walk_forward_predict,
    )

    if hyg:
        def builder(frame, **kwargs):
            return build_hy_feature_matrix(frame, spread_col=spread_col, **kwargs)

        target_col, years_needed, min_train = f"target_{horizon}d_lag1_xs_return", 2, 500
        params = {"weights": RETURN_COMPOSITE_WEIGHTS} if model_type == "composite" else None
    else:
        def builder(frame, **kwargs):
            return build_feature_matrix(frame, target_col=spread_col, **kwargs)

        target_col, years_needed, min_train = f"target_{horizon}d_change", 5, 756
        params = None
    X, y = builder(df, target_horizon=horizon)
    target = y[target_col]
    scale_col = DEFAULT_SCALE_COL if model_type in TREE_MODELS else None
    first_year = X.index.min() + pd.DateOffset(years=years_needed)
    start = str(pd.Timestamp(year=first_year.year + 1, month=1, day=1).date())
    oos = walk_forward_predict(
        X, target, model_type=model_type, start=start, scale_col=scale_col, params=params, min_train=min_train,
    )
    ok = oos.notna()

    periods = {}
    for name, mask in {
        f"{start[:4]}–{int(HOLDOUT_START[:4]) - 1}": ok & (X.index < HOLDOUT_START),
        f"{HOLDOUT_START[:4]}–now (holdout)": ok & (X.index >= HOLDOUT_START),
    }.items():
        if mask.sum() > horizon * 10:
            periods[name] = compute_metrics(target[mask].values, oos[mask].values, horizon=horizon)

    model = make_model(model_type, "regression", params=params, scale_col=scale_col)
    model.fit(X, target)
    X_live, _ = builder(df, target_horizon=horizon, dropna=False)
    X_live = X_live[X.columns].dropna()
    live = pd.Series(model.predict(X_live.tail(60)), index=X_live.tail(60).index)

    return {
        "oos": oos[ok],
        "actual": target[ok],
        "periods": periods,
        "live": live,
        "importance": feature_importance(model, list(X.columns)),
        "model": model,
        "X": X,
    }


@st.cache_data(show_spinner="Fitting HMM …")
def fit_regimes(values: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit the HMM once per dataset / state count: labels, filtered probabilities, transitions."""
    from src.models.regime import filtered_regime_probabilities, fit_hmm, get_transition_matrix, label_regimes

    model = fit_hmm(values, n_states=n_states)
    return (
        label_regimes(model, values, model_type="hmm"),
        filtered_regime_probabilities(model, values),
        get_transition_matrix(model, model_type="hmm"),
    )


@st.cache_data(show_spinner="Fitting real-time regime model (yearly refits) …")
def realtime_regimes(df: pd.DataFrame, spread_col: str) -> pd.DataFrame:
    """Walk-forward HMM regime probabilities (no look-ahead)."""
    from src.models.regime import real_time_regime_probabilities

    return real_time_regime_probabilities(df, spread_col=spread_col, n_states=REGIME_STATES_REALTIME)


if refresh:
    st.cache_data.clear()

df, is_synthetic = load_data(str(start_date), str(end_date), fred_api_key)

try:
    from src.features.engineering import default_spread_column, spread_columns_with_history

    usable_spreads = spread_columns_with_history(df) or [default_spread_column(df)]
except ValueError:
    st.error("No credit spread column is available in the loaded data.")
    st.stop()

primary = st.sidebar.selectbox(
    "Credit spread",
    options=usable_spreads,
    format_func=lambda c: SPREAD_LABELS.get(c, c),
    help="Spread used for regimes, the exposure overlay and the spread forecast.  Only series with enough "
         "history are listed: FRED serves ~3 years of the ICE OAS series unless a longer licensed history "
         "has been added to data/external/.",
)

equity_return_col = "spy_return" if "spy_return" in df.columns else "sp500_return"

if is_synthetic:
    st.info("Showing synthetic demo data – results are illustrative only.")

# ---------------------------------------------------------------------------
# Main content tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview",
    "🔵 Regime Analysis",
    "🔗 Leading Indicator",
    "🤖 Forecasting",
    "🌡️ Correlation Monitor",
])

# ============================================================
# TAB 1 – Overview
# ============================================================
with tab1:
    st.header("Market Overview")

    spread_cols_available = [c for c in SPREAD_LABELS if c in df.columns and df[c].notna().any()]
    cards = spread_cols_available + (["vix"] if "vix" in df.columns else [])
    cols = st.columns(len(cards))
    for i, col in enumerate(cards):
        s = df[col].dropna()
        if len(s) < 2:
            continue
        if col == "vix":
            cols[i].metric("VIX", f"{s.iloc[-1]:.1f}", f"{s.iloc[-1] - s.iloc[-2]:+.1f}")
            continue
        chg_20 = s.iloc[-1] - s.iloc[-21] if len(s) > 21 else np.nan
        pct = float((s <= s.iloc[-1]).mean() * 100)
        cols[i].metric(
            label=SPREAD_LABELS[col],
            value=f"{s.iloc[-1]:.0f} bps",
            delta=f"{chg_20:+.0f} bps (20d)",
            delta_color="inverse",
            help=f"{pct:.0f}th percentile since {s.index[0].date()}; last value {s.index[-1].date()}.",
        )

    ice_cols = [c for c in ("hy_spread", "ig_spread", "bbb_spread") if c in spread_cols_available]
    if ice_cols:
        st.caption(
            "ICE BofA OAS series are only available from FRED for roughly the last three years, "
            "so models use the Moody's Baa – 10y spread, which has history back to 1986."
        )

    st.divider()
    try:
        from src.visualization.plots import plot_spread_history

        fig = plot_spread_history(df, spread_cols=spread_cols_available)
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not render spread history: {exc}")

    with st.expander("Raw data preview"):
        st.dataframe(df.tail(50))

# ============================================================
# TAB 2 – Regime Analysis
# ============================================================
with tab2:
    st.header("Regime Detection")
    st.caption(
        f"Gaussian HMM on {SPREAD_LABELS.get(primary, primary)}. Regimes are ordered from calmest (0) "
        "to most stressed. The coloured history uses the full sample (hindsight); the current-regime "
        "probabilities below are filtered and use only data available at each date."
    )

    try:
        from src.models.regime import compute_regime_stats
        from src.visualization.plots import plot_regime_overlay

        hmm_frame = df[[primary]].dropna()
        regimes, probs, trans = fit_regimes(hmm_frame.values, n_regimes)

        fig = plot_regime_overlay(hmm_frame, regimes, spread_col=primary)
        st.plotly_chart(fig, width="stretch")

        rt = realtime_regimes(df, primary).dropna()
        if not rt.empty:
            st.subheader("Real-time probability of the credit-stress regime")
            st.caption(
                "Three-state HMM refitted every January on past data only and run forward day by day – "
                "this is what the model would have shown at the time.  It drives the default exposure overlay."
            )
            st.area_chart(rt.iloc[:, -1].rename("P(stress)"))
            st.metric(f"P(stress) on {rt.index[-1].date()}", f"{rt.iloc[-1, -1]:.0%}")

        current = pd.Series(probs[-1], index=[f"Regime {i}" for i in range(n_regimes)])
        st.subheader(f"Full-sample model: regime probabilities on {hmm_frame.index[-1].date()}")
        st.bar_chart(current)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Transition Matrix")
            st.dataframe(trans.style.format("{:.3f}").background_gradient(cmap="Blues"))

        with col_b:
            st.subheader("Regime Statistics")
            stats_df = compute_regime_stats(
                df.loc[hmm_frame.index], regimes, equity_col=equity_return_col, spread_col=primary
            )
            st.dataframe(stats_df.style.format("{:.4f}"))
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Regime analysis failed: {exc}")

# ============================================================
# TAB 3 – Leading Indicator
# ============================================================
with tab3:
    st.header("Leading Indicator Analysis")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Granger Causality")
        st.caption("Tests on daily changes (spreads, VIX, curve) and returns (equities).")
        try:
            from src.models.statistical import run_granger_causality

            stationary = pd.DataFrame({
                "Δ spread": df[primary].diff(),
                "Δ VIX": df["vix"].diff() if "vix" in df.columns else np.nan,
                "Δ 10y–2y": df["t10y2y"].diff() if "t10y2y" in df.columns else np.nan,
                "equity return": df[equity_return_col] if equity_return_col in df.columns else np.nan,
            })
            granger_pairs = [
                ("Δ spread", "equity return"),
                ("equity return", "Δ spread"),
                ("Δ spread", "Δ VIX"),
                ("Δ spread", "Δ 10y–2y"),
            ]
            granger_rows = []
            for caused, causing in granger_pairs:
                if stationary[[caused, causing]].dropna().shape[0] < 100:
                    continue
                try:
                    pvals = run_granger_causality(stationary, caused, causing, maxlag=5, transform="none")
                except ValueError:
                    continue
                min_p = min(pvals.values())
                granger_rows.append({
                    "Caused": caused,
                    "Causing": causing,
                    "Min p-value": round(min_p, 4),
                    "Significant (5%)": "✓" if min_p < 0.05 else "✗",
                })
            if granger_rows:
                st.dataframe(pd.DataFrame(granger_rows), hide_index=True)
            else:
                st.info("Not enough data for Granger tests.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Granger causality failed: {exc}")

    with col_right:
        st.subheader("Backtest: Credit-Aware Equity Exposure")
        method = st.selectbox(
            "Exposure overlay", list(EXPOSURE_CHOICES), format_func=EXPOSURE_CHOICES.get,
            help="Default: equity weight = min(1, target vol / recent vol) × (1 − real-time P(credit stress)).",
        )
        with st.expander("Overlay settings", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            target_vol = c1.number_input("Target vol", value=EXPOSURE_TARGET_VOL, step=0.01, format="%.2f")
            enter_z = c2.number_input("Enter z >", value=0.5, step=0.25)
            exit_z = c3.number_input("Exit z <", value=0.0, step=0.25)
            widen_bp = c4.number_input("Widening (bps)", value=50.0, step=5.0)
        try:
            from src.analysis.leading_indicator import run_full_backtest
            from src.visualization.plots import plot_backtest_results

            if equity_return_col in df.columns:
                regime_probs = realtime_regimes(df, primary) if method in ("vol_regime", "regime") else None
                with st.spinner("Running backtest …"):
                    bt_df, bt_metrics = run_full_backtest(
                        df, spread_col=primary, equity_col=equity_return_col, method=method,
                        target_vol=target_vol, regime_probs=regime_probs,
                        enter_threshold=enter_z, exit_threshold=min(exit_z, enter_z), widen_threshold=widen_bp,
                    )

                fig = plot_backtest_results(bt_df)
                st.plotly_chart(fig, width="stretch")

                m_cols = st.columns(4)
                m_cols[0].metric("Sharpe", f"{bt_metrics['sharpe']:.2f}",
                                 f"{bt_metrics['sharpe'] - bt_metrics['bh_sharpe']:+.2f} vs B&H")
                m_cols[1].metric("Max DD", f"{bt_metrics['max_drawdown']*100:.1f}%",
                                 f"B&H {bt_metrics['bh_max_drawdown']*100:.1f}%", delta_color="off")
                m_cols[2].metric("CAGR", f"{bt_metrics['annualised_return']*100:.1f}%",
                                 f"B&H {bt_metrics['bh_annualised_return']*100:.1f}%", delta_color="off")
                m_cols[3].metric("Avg equity weight", f"{bt_metrics['avg_equity_weight']*100:.0f}%",
                                 f"{bt_metrics['turnover_per_year']:.1f} turnover/yr", delta_color="off")
                st.markdown(f"**Equity weight in force on {bt_df.index[-1].date()}: {bt_df['weight'].iloc[-1]:.0%}**")
                st.caption(
                    "Weights use data published by each close and trade at the next close; fractional overlays "
                    "rebalance only on moves above 10 points; costs are 5 bp per unit of turnover and cash earns "
                    "the 3-month T-bill rate.  In walk-forward tests the default overlay raised Sharpe and kept the "
                    "maximum drawdown under 20% in both 2000–2018 and 2019–2026; the z-score rule lagged in 2019–2026."
                )
            else:
                st.info("No equity return column available for the backtest.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Backtest failed: {exc}")

# ============================================================
# TAB 4 – Forecasting
# ============================================================
with tab4:
    if is_hyg:
        st.header(f"{horizon}-day HYG Excess Return Forecast ({model_type})")
        st.caption(
            f"Target: return of HYG minus 0.85 × IEI over {horizon} trading days, starting the day after the "
            "forecast (bps).  Tradable, but the edge is small: in walk-forward tests the rank IC was about 0.13–0.16 "
            "while the size of moves was not predictable.  History shown is out-of-sample (yearly refits)."
        )
    else:
        st.header(f"{horizon}-day Spread Forecast ({model_type})")
        st.caption(
            f"Target: change in {SPREAD_LABELS.get(primary, primary)} over the next {horizon} trading days (bps). "
            "This index reacts to equity moves with a lag, so part of its predictability is not tradable.  "
            "All history shown is out-of-sample: the model is refitted every January on data available then."
        )

    try:
        from src.models.ml_models import TREE_MODELS, compute_shap_values
        from src.visualization.plots import plot_forecast_vs_actual, plot_shap_summary

        if is_hyg and "hyg_xs_return" not in df.columns:
            raise ValueError("HYG / IEI data is not available in the loaded dataset.")
        fc = run_forecasts(df, horizon, model_type, hyg=is_hyg, spread_col=primary)

        latest_date = fc["live"].index[-1]
        latest = float(fc["live"].iloc[-1])
        if is_hyg:
            direction = "HY outperforms Treasuries" if latest > 0 else "HY underperforms Treasuries"
        else:
            direction = "widening" if latest > 0 else "tightening"
        st.metric(
            f"Forecast from {latest_date.date()}",
            f"{latest:+.1f} bps",
            direction,
            delta_color="normal" if is_hyg else "inverse",
        )

        if fc["periods"]:
            rows = {
                name: {
                    "OOS R² vs no-change": m["oos_r2"],
                    "Rank IC": m["ic"],
                    "Hit rate": m["directional_accuracy"],
                    "Signal Sharpe": m["signal_sharpe"],
                }
                for name, m in fc["periods"].items()
            }
            st.dataframe(pd.DataFrame(rows).T.style.format("{:.3f}"))

        col_pred, col_imp = st.columns(2)
        with col_pred:
            recent = fc["oos"].index >= fc["oos"].index.max() - pd.DateOffset(years=2)
            fig_pred = plot_forecast_vs_actual(
                fc["actual"][recent].values,
                fc["oos"][recent].values,
                title="Out-of-sample forecast vs actual (last 2 years)",
                index=fc["oos"].index[recent],
            )
            st.plotly_chart(fig_pred, width="stretch")

        with col_imp:
            shown = False
            if model_type in TREE_MODELS:
                try:
                    shap_vals = compute_shap_values(fc["model"], fc["X"], model_type=model_type)
                    st.plotly_chart(plot_shap_summary(shap_vals, fc["X"]), width="stretch")
                    shown = True
                except Exception as shap_exc:  # noqa: BLE001
                    st.info(f"SHAP computation unavailable: {shap_exc}")
            if not shown:
                st.subheader("Feature importance")
                st.bar_chart(fc["importance"].head(20))
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Forecasting failed: {exc}")

# ============================================================
# TAB 5 – Correlation Monitor
# ============================================================
with tab5:
    st.header("Correlation Monitor")

    window_size = st.slider("Window (most recent days)", min_value=20, max_value=252, value=60, step=10)

    try:
        from src.visualization.plots import plot_correlation_heatmap

        change_view = pd.DataFrame(index=df.index)
        for col in [primary, "aaa_spread", "hy_spread", "vix", "dgs10", "t10y2y"]:
            if col in df.columns:
                change_view[f"Δ {col}"] = df[col].diff()
        for col in ["spy_return", "sp500_return", "hyg_return", "ief_return", "gold_return"]:
            if col in df.columns:
                change_view[col] = df[col]
        options = list(change_view.columns)
        selected_cols = st.multiselect(
            "Select series (daily changes / returns)", options=options, default=options[:min(8, len(options))]
        )
        if selected_cols:
            fig = plot_correlation_heatmap(change_view[selected_cols].dropna(how="all"), window=window_size)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Select at least one column.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Correlation monitor failed: {exc}")
