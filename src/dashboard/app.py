"""
Streamlit dashboard for the Credit Spread Analysis & Prediction Platform.

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from project root when run directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Spread Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

start_date = st.sidebar.date_input("Start date", value=pd.Timestamp("2005-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.Timestamp("today"))
model_type = st.sidebar.selectbox(
    "ML Model", options=["xgboost", "lightgbm", "random_forest"], index=0
)
n_regimes = st.sidebar.slider("Number of Regimes", min_value=2, max_value=4, value=3)
fred_api_key = st.sidebar.text_input("FRED API Key (optional)", type="password")

st.sidebar.markdown("---")
refresh = st.sidebar.button("🔄 Refresh Data")

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading market data …", ttl=3600)
def load_data(start: str, end: str, api_key: str) -> pd.DataFrame:
    """Fetch and cache market data."""
    try:
        from src.data.fetcher import fetch_all_data  # type: ignore

        return fetch_all_data(
            start_date=start,
            end_date=end,
            api_key=api_key,
            cache_dir=Path("data"),
        )
    except Exception as exc:
        st.warning(f"Data fetch failed: {exc}. Using synthetic demo data.")
        return _synthetic_demo_data(start, end)


def _synthetic_demo_data(start: str, end: str) -> pd.DataFrame:
    """Generate synthetic demo data for UI testing."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(start=start, end=end)
    n = len(idx)
    df = pd.DataFrame(index=idx)
    df["hy_spread"] = 400 + np.cumsum(rng.normal(0, 5, n))
    df["ig_spread"] = 120 + np.cumsum(rng.normal(0, 2, n))
    df["bbb_spread"] = 200 + np.cumsum(rng.normal(0, 3, n))
    df["t10y2y"] = rng.normal(1.0, 0.8, n)
    df["vix"] = 18 + np.abs(np.cumsum(rng.normal(0, 0.5, n)) % 30)
    df["sp500"] = 1000 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    df["sp500_return"] = np.log(df["sp500"] / df["sp500"].shift(1))
    df["fed_funds"] = np.clip(2 + np.cumsum(rng.normal(0, 0.02, n)), 0, 8)
    return df.ffill().dropna()


if refresh:
    st.cache_data.clear()

df = load_data(str(start_date), str(end_date), fred_api_key)

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

    # Metric cards
    spread_cols_available = [c for c in ["hy_spread", "ig_spread", "bbb_spread"] if c in df.columns]
    if spread_cols_available:
        cols = st.columns(len(spread_cols_available) + 2)
        for i, col in enumerate(spread_cols_available):
            current = df[col].iloc[-1]
            prev = df[col].iloc[-2] if len(df) > 1 else current
            pct = float(df[col].rank(pct=True).iloc[-1] * 100)
            cols[i].metric(
                label=col.replace("_", " ").upper(),
                value=f"{current:.0f} bps",
                delta=f"{current - prev:+.1f} bps",
            )
        # VIX
        if "vix" in df.columns:
            cols[-2].metric("VIX", f"{df['vix'].iloc[-1]:.1f}", f"{df['vix'].iloc[-1] - df['vix'].iloc[-2]:+.1f}")

        # Regime badge (placeholder until HMM is run)
        cols[-1].metric("Data Points", f"{len(df):,}")

    st.divider()
    try:
        from src.visualization.plots import plot_spread_history  # type: ignore

        fig = plot_spread_history(df, spread_cols=spread_cols_available)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render spread history: {exc}")

    with st.expander("Raw data preview"):
        st.dataframe(df.tail(50))

# ============================================================
# TAB 2 – Regime Analysis
# ============================================================
with tab2:
    st.header("Regime Detection")

    try:
        from src.models.regime import (  # type: ignore
            compute_regime_stats,
            fit_hmm,
            get_transition_matrix,
            label_regimes,
        )
        from src.visualization.plots import plot_regime_overlay  # type: ignore

        hmm_col = "hy_spread"
        if hmm_col in df.columns:
            hmm_data = df[[hmm_col]].dropna().values

            with st.spinner("Fitting HMM …"):
                hmm_model = fit_hmm(hmm_data, n_states=n_regimes)
                regimes = label_regimes(hmm_model, hmm_data, model_type="hmm")

            # Align regime labels back to df index
            valid_idx = df[[hmm_col]].dropna().index
            regime_series = pd.Series(regimes, index=valid_idx, name="regime")

            fig = plot_regime_overlay(df.loc[valid_idx], regimes, spread_col=hmm_col)
            st.plotly_chart(fig, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Transition Matrix")
                trans = get_transition_matrix(hmm_model, model_type="hmm")
                st.dataframe(trans.style.format("{:.3f}").background_gradient(cmap="Blues"))

            with col_b:
                st.subheader("Regime Statistics")
                if "sp500_return" in df.columns:
                    stats_df = compute_regime_stats(
                        df.loc[valid_idx], regimes, equity_col="sp500_return", spread_col=hmm_col
                    )
                    st.dataframe(stats_df.style.format("{:.4f}"))
                else:
                    st.info("sp500_return column not found – skipping regime stats.")
        else:
            st.info(f"Column '{hmm_col}' not found – cannot run regime detection.")
    except Exception as exc:
        st.warning(f"Regime analysis failed: {exc}")

# ============================================================
# TAB 3 – Leading Indicator
# ============================================================
with tab3:
    st.header("Leading Indicator Analysis")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Granger Causality")
        try:
            from src.models.statistical import run_granger_causality  # type: ignore

            granger_pairs = [
                ("hy_spread", "vix"),
                ("hy_spread", "t10y2y"),
                ("sp500_return", "hy_spread"),
            ]
            granger_rows = []
            for caused, causing in granger_pairs:
                if caused in df.columns and causing in df.columns:
                    try:
                        pvals = run_granger_causality(df.dropna(), caused, causing, maxlag=5)
                        min_p = min(pvals.values())
                        granger_rows.append({
                            "Caused": caused,
                            "Causing": causing,
                            "Min p-value": round(min_p, 4),
                            "Significant (5%)": "✓" if min_p < 0.05 else "✗",
                        })
                    except Exception:
                        pass
            if granger_rows:
                st.dataframe(pd.DataFrame(granger_rows), hide_index=True)
            else:
                st.info("Not enough data for Granger tests.")
        except Exception as exc:
            st.warning(f"Granger causality failed: {exc}")

    with col_right:
        st.subheader("Backtest: Spread Signal Strategy")
        try:
            from src.analysis.leading_indicator import run_full_backtest  # type: ignore
            from src.visualization.plots import plot_backtest_results  # type: ignore

            if "sp500_return" in df.columns and "hy_spread" in df.columns:
                with st.spinner("Running backtest …"):
                    bt_df, bt_metrics = run_full_backtest(df)

                fig = plot_backtest_results(bt_df)
                st.plotly_chart(fig, use_container_width=True)

                m_cols = st.columns(4)
                m_cols[0].metric("Sharpe", f"{bt_metrics['sharpe']:.2f}")
                m_cols[1].metric("Max DD", f"{bt_metrics['max_drawdown']*100:.1f}%")
                m_cols[2].metric("Win Rate", f"{bt_metrics['win_rate']*100:.1f}%")
                m_cols[3].metric("Total Return", f"{bt_metrics['total_return']*100:.1f}%")
            else:
                st.info("sp500_return or hy_spread not available for backtest.")
        except Exception as exc:
            st.warning(f"Backtest failed: {exc}")

# ============================================================
# TAB 4 – Forecasting
# ============================================================
with tab4:
    st.header(f"ML Forecasting ({model_type})")

    try:
        from src.features.engineering import build_feature_matrix  # type: ignore
        from src.models.ml_models import compute_shap_values, train_and_evaluate  # type: ignore
        from src.visualization.plots import plot_forecast_vs_actual, plot_shap_summary  # type: ignore

        if "hy_spread" in df.columns:
            with st.spinner("Building features and training model …"):
                X, y = build_feature_matrix(df, target_horizon=5)
                target_col = [c for c in y.columns if "return" in c][0]
                result = train_and_evaluate(X, y[target_col], model_type=model_type, n_splits=3)

            # Metrics
            mean_m = result["mean_metrics"]
            m_cols = st.columns(4)
            m_cols[0].metric("RMSE", f"{mean_m.get('rmse', float('nan')):.4f}")
            m_cols[1].metric("MAE", f"{mean_m.get('mae', float('nan')):.4f}")
            m_cols[2].metric("Dir. Accuracy", f"{mean_m.get('directional_accuracy', float('nan'))*100:.1f}%")
            m_cols[3].metric("Signal Sharpe", f"{mean_m.get('signal_sharpe', float('nan')):.2f}")

            col_pred, col_shap = st.columns(2)
            with col_pred:
                oof = result["oof_predictions"]
                valid_mask = ~np.isnan(oof)
                fig_pred = plot_forecast_vs_actual(
                    y[target_col].values[valid_mask], oof[valid_mask]
                )
                st.plotly_chart(fig_pred, use_container_width=True)

            with col_shap:
                try:
                    shap_vals = compute_shap_values(result["model"], X, model_type=model_type)
                    fig_shap = plot_shap_summary(shap_vals, X)
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as shap_exc:
                    st.info(f"SHAP computation unavailable: {shap_exc}")
                    fi = result["feature_importance"].head(20)
                    st.bar_chart(fi)
        else:
            st.info("hy_spread column not found – cannot run forecasting.")
    except Exception as exc:
        st.warning(f"Forecasting failed: {exc}")

# ============================================================
# TAB 5 – Correlation Monitor
# ============================================================
with tab5:
    st.header("Correlation Monitor")

    window_size = st.slider("Rolling window (days)", min_value=20, max_value=252, value=60, step=10)

    try:
        from src.visualization.plots import plot_correlation_heatmap  # type: ignore

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect(
            "Select columns",
            options=numeric_cols,
            default=numeric_cols[:min(8, len(numeric_cols))],
        )
        if selected_cols:
            fig = plot_correlation_heatmap(df[selected_cols], window=window_size)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one column.")
    except Exception as exc:
        st.warning(f"Correlation monitor failed: {exc}")
