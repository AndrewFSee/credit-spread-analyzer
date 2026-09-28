"""
Visualization module for the Credit Spread Analysis & Prediction Platform.

All chart functions accept a ``use_plotly`` flag.  When ``True`` (default) they
return a ``plotly.graph_objects.Figure``; when ``False`` they return a
``matplotlib.figure.Figure``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Colour-vision-safe categorical palette, used in this fixed order (validated
# for adjacent-pair separation); a neutral grey marks benchmarks.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BENCHMARK_COLOR = "#898781"
# Ordinal single-hue ramp for ordered categories such as calm → stressed regimes.
ORDINAL_BLUES = {2: ["#86b6ef", "#184f95"], 3: ["#86b6ef", "#3987e5", "#184f95"],
                 4: ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]}
RECESSION_FILL = "rgba(137, 135, 129, 0.14)"

SERIES_LABELS = {
    "baa_spread": "Moody's Baa – 10y",
    "aaa_spread": "Moody's Aaa – 10y",
    "hy_spread": "ICE BofA High Yield OAS",
    "ig_spread": "ICE BofA IG OAS",
    "bbb_spread": "ICE BofA BBB OAS",
}
FEATURE_LABELS = {
    "spread_chg_5": "Spread change, 5d", "spread_chg_20": "Spread change, 20d",
    "spread_chg_60": "Spread change, 60d", "spread_z60": "Spread z-score, 3m",
    "spread_z252": "Spread z-score, 1y", "spread_vol60": "Spread volatility, 3m",
    "quality_spread_chg_20": "Baa – Aaa change, 20d", "y10_chg_5": "10y Treasury change, 5d",
    "y10_chg_20": "10y Treasury change, 20d", "t10y3m_chg_60": "Curve (10y – 3m) change, 3m",
    "eq_ret_5": "S&P 500 return, 5d", "eq_ret_20": "S&P 500 return, 20d", "eq_ret_60": "S&P 500 return, 3m",
    "eq_drawdown_252": "S&P 500 drawdown from 1y high", "eq_vol_20": "S&P 500 volatility, 1m",
    "vix_chg_5": "VIX change, 5d", "vix_chg_20": "VIX change, 20d", "vix_z252": "VIX z-score, 1y",
    "vol_risk_premium": "VIX minus realised volatility", "nfci_chg_20": "Financial conditions (NFCI), 20d",
    "claims_growth_60": "Jobless claims growth, 3m", "hy_xs_ret_5": "HY excess return, 5d",
    "hy_xs_ret_20": "HY excess return, 20d", "hy_xs_ret_60": "HY excess return, 3m",
    "hy_xs_vol_20": "HY excess-return volatility", "hy_xs_drawdown_252": "HY drawdown from 1y high",
    "gz_spread_chg_60": "GZ spread change, 3m", "ebp_z756": "Excess bond premium z-score",
    "regime_p_stress": "P(credit-stress regime)", "regime_level": "Expected regime",
}


def series_label(name: str) -> str:
    """Readable name for a data column or model feature."""
    return SERIES_LABELS.get(name) or FEATURE_LABELS.get(name) or name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recession_bands(fig: Any, **kwargs: Any) -> Any:
    """Add NBER recession shading to a Plotly figure."""
    recessions = [
        ("1990-07-01", "1991-03-31"),
        ("2001-03-01", "2001-11-30"),
        ("2007-12-01", "2009-06-30"),
        ("2020-02-01", "2020-04-30"),
    ]
    for start, end in recessions:
        fig.add_vrect(x0=start, x1=end, fillcolor=RECESSION_FILL, layer="below", line_width=0, **kwargs)
    return fig


def _legend_top() -> dict:
    return dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)


# ---------------------------------------------------------------------------
# Public chart functions
# ---------------------------------------------------------------------------

def plot_spread_history(
    df: pd.DataFrame,
    spread_cols: Optional[list[str]] = None,
    use_plotly: bool = True,
) -> Any:
    """Plot time-series of credit spreads with recession shading.

    Parameters
    ----------
    df:
        DataFrame with DatetimeIndex and spread columns.
    spread_cols:
        Column names to plot.  Auto-detected if ``None``.
    use_plotly:
        Return a Plotly figure when ``True``, Matplotlib figure when ``False``.

    Returns
    -------
    Figure object.
    """
    if spread_cols is None:
        spread_cols = [
            c for c in ["baa_spread", "aaa_spread", "hy_spread", "ig_spread", "bbb_spread"] if c in df.columns
        ]
    if not spread_cols:
        raise ValueError("No spread columns found in DataFrame.")

    if use_plotly:
        import plotly.graph_objects as go  # type: ignore

        fig = go.Figure()
        for i, col in enumerate(spread_cols):
            s = df[col].dropna()
            fig.add_trace(go.Scatter(
                x=s.index, y=s, mode="lines", name=series_label(col),
                line=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], width=1.6),
                hovertemplate="%{y:.0f} bps",
            ))
        fig = _recession_bands(fig)
        fig.update_layout(
            title="Credit spread history (shaded: NBER recessions)",
            yaxis_title="Spread (bps)",
            hovermode="x unified",
            legend=_legend_top(),
            template="plotly_white",
            margin=dict(t=90),
        )
        return fig
    else:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(14, 5))
        for i, col in enumerate(spread_cols):
            ax.plot(df.index, df[col], label=series_label(col), color=SERIES_COLORS[i % len(SERIES_COLORS)])
        ax.set_title("Credit spread history")
        ax.set_xlabel("Date")
        ax.set_ylabel("Spread (bps)")
        ax.legend()
        fig.tight_layout()
        return fig


def plot_regime_overlay(
    df: pd.DataFrame,
    regimes: np.ndarray,
    spread_col: str = "baa_spread",
    use_plotly: bool = True,
) -> Any:
    """Plot spread time-series coloured by regime.

    Parameters
    ----------
    df:
        DataFrame with DatetimeIndex.
    regimes:
        Integer regime-label array aligned with *df*.
    spread_col:
        Spread column to plot.
    use_plotly:
        Return Plotly figure if ``True``.

    Returns
    -------
    Figure object.
    """
    if spread_col not in df.columns:
        raise ValueError(f"Column '{spread_col}' not found.")

    unique_regimes = sorted(np.unique(regimes))
    colours = ORDINAL_BLUES.get(len(unique_regimes), SERIES_COLORS)

    if use_plotly:
        import plotly.graph_objects as go  # type: ignore

        suffix = {unique_regimes[0]: " (calmest)", unique_regimes[-1]: " (most stressed)"}
        fig = go.Figure()
        for i, r in enumerate(unique_regimes):
            mask = regimes == r
            idx = df.index[mask]
            vals = df[spread_col].values[mask]
            fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=vals,
                    mode="markers",
                    marker=dict(color=colours[i % len(colours)], size=4),
                    name=f"Regime {r}{suffix.get(r, '') if len(unique_regimes) > 1 else ''}",
                    hovertemplate="%{y:.0f} bps",
                )
            )
        fig.update_layout(
            title=f"{series_label(spread_col)} coloured by regime",
            yaxis_title="Spread (bps)",
            legend=dict(**_legend_top(), itemsizing="constant"),
            template="plotly_white",
            margin=dict(t=90),
        )
        return fig
    else:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(14, 5))
        for i, r in enumerate(unique_regimes):
            mask = regimes == r
            ax.scatter(df.index[mask], df[spread_col].values[mask], s=4, label=f"Regime {r}",
                       color=colours[i % len(colours)])
        ax.set_title(f"{series_label(spread_col)} coloured by regime")
        ax.legend()
        fig.tight_layout()
        return fig


def plot_stress_probability(probability: pd.Series, title: str = "Real-time probability of credit stress") -> Any:
    """Line + light wash of a 0–1 probability series, with NBER recession shading."""
    import plotly.graph_objects as go  # type: ignore

    p = probability.dropna()
    fig = go.Figure(go.Scatter(
        x=p.index, y=p, mode="lines", name="P(stress)",
        line=dict(color=SERIES_COLORS[1], width=1.4),
        fill="tozeroy", fillcolor="rgba(235, 104, 52, 0.14)",
        hovertemplate="%{y:.0%}<extra></extra>",
    ))
    fig = _recession_bands(fig)
    fig.update_layout(
        title=title, yaxis=dict(range=[0, 1.05], tickformat=".0%", tickvals=[0, 0.5, 1]),
        showlegend=False, hovermode="x unified", template="plotly_white", height=300,
        margin=dict(t=60, b=30),
    )
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    window: int = 60,
    use_plotly: bool = True,
) -> Any:
    """Plot the correlation matrix over the most recent *window* rows.

    Parameters
    ----------
    df:
        DataFrame with numeric columns.
    window:
        Number of most recent observations used.
    use_plotly:
        Return Plotly figure if ``True``.

    Returns
    -------
    Figure object.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.tail(window).corr()

    if use_plotly:
        import plotly.graph_objects as go  # type: ignore

        z = corr.values
        labels = list(numeric_df.columns)
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                colorscale="RdBu",
                zmid=0,
                text=np.round(z, 2),
                texttemplate="%{text}",
            )
        )
        fig.update_layout(title=f"Correlation Heatmap (last {window} days)", template="plotly_white")
        return fig
    else:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
        ax.set_title(f"Correlation Heatmap (last {window} days)")
        fig.tight_layout()
        return fig


def plot_impulse_response(
    irf_results: Any,
    use_plotly: bool = True,
) -> Any:
    """Plot Impulse Response Functions.

    Parameters
    ----------
    irf_results:
        ``statsmodels`` IRAnalysis object returned by :func:`compute_irf`.
    use_plotly:
        Return Plotly figure if ``True``.

    Returns
    -------
    Figure object.
    """
    irfs: np.ndarray = irf_results.irfs  # shape (periods, k, k)
    periods = irfs.shape[0]
    k = irfs.shape[1]
    var_names: list[str] = list(irf_results.model.names)
    x_axis = list(range(periods))

    if use_plotly:
        import plotly.graph_objects as go  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore

        fig = make_subplots(rows=k, cols=k, subplot_titles=[
            f"{var_names[j]} → {var_names[i]}" for i in range(k) for j in range(k)
        ])
        for i in range(k):
            for j in range(k):
                fig.add_trace(
                    go.Scatter(x=x_axis, y=irfs[:, i, j], mode="lines", showlegend=False),
                    row=i + 1,
                    col=j + 1,
                )
        fig.update_layout(title="Impulse Response Functions", template="plotly_white")
        return fig
    else:
        import matplotlib.pyplot as plt  # type: ignore

        fig, axes = plt.subplots(k, k, figsize=(4 * k, 3 * k))
        if k == 1:
            axes = np.array([[axes]])
        for i in range(k):
            for j in range(k):
                axes[i, j].plot(x_axis, irfs[:, i, j])
                axes[i, j].axhline(0, color="k", linewidth=0.5, linestyle="--")
                axes[i, j].set_title(f"{var_names[j]} → {var_names[i]}", fontsize=8)
        fig.suptitle("Impulse Response Functions")
        fig.tight_layout()
        return fig


def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 12,
    title: str = "What drives the forecast",
    x_title: str = "Share of total importance",
) -> Any:
    """Horizontal bar chart of the *top_n* features, largest first, with readable names."""
    import plotly.graph_objects as go  # type: ignore

    imp = importance.abs()
    imp = (imp / imp.sum()).sort_values(ascending=False).head(top_n)[::-1]
    fig = go.Figure(go.Bar(
        x=imp.values, y=[series_label(n) for n in imp.index], orientation="h",
        marker=dict(color=SERIES_COLORS[0]), width=0.6,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title=title, xaxis_title=x_title, xaxis_tickformat=".0%",
        template="plotly_white", showlegend=False, margin=dict(l=10),
    )
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    use_plotly: bool = True,
    max_display: int = 20,
) -> Any:
    """Plot a SHAP feature-importance bar chart.

    Parameters
    ----------
    shap_values:
        SHAP values array of shape ``(n_samples, n_features)``.
    X:
        Feature DataFrame (column names used as labels).
    use_plotly:
        Return Plotly figure if ``True``.
    max_display:
        Maximum number of features to display.

    Returns
    -------
    Figure object.
    """
    feature_names = list(X.columns)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:max_display]
    top_names = [feature_names[i] for i in sorted_idx]
    top_values = mean_abs_shap[sorted_idx]

    if use_plotly:
        return plot_feature_importance(
            pd.Series(top_values, index=top_names), top_n=max_display,
            title="What drives the forecast (SHAP)", x_title="Share of mean |SHAP value|",
        )
    else:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(8, max_display * 0.4 + 1))
        ax.barh([series_label(n) for n in top_names[::-1]], top_values[::-1], color=SERIES_COLORS[0])
        ax.set_xlabel("|SHAP value|")
        ax.set_title("SHAP Feature Importance")
        fig.tight_layout()
        return fig


def plot_forecast_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    use_plotly: bool = True,
    title: str = "Forecast vs Actual",
    index: Optional[Any] = None,
    y_title: str = "bps",
) -> Any:
    """Scatter and line plot comparing predictions to actual values.

    Parameters
    ----------
    y_true:
        Ground-truth values.
    y_pred:
        Model predictions.
    use_plotly:
        Return Plotly figure if ``True``.
    title:
        Chart title.
    index:
        Optional x-axis values (e.g. dates); defaults to the sample number.
    y_title:
        Y-axis title (units of the target).

    Returns
    -------
    Figure object.
    """
    idx = np.arange(len(y_true)) if index is None else index

    if use_plotly:
        import plotly.graph_objects as go  # type: ignore

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx, y=y_true, mode="lines", name="Actual",
                                 line=dict(color=BENCHMARK_COLOR, width=1.2), hovertemplate="%{y:.1f}"))
        fig.add_trace(go.Scatter(x=idx, y=y_pred, mode="lines", name="Forecast",
                                 line=dict(color=SERIES_COLORS[0], width=2), hovertemplate="%{y:.1f}"))
        fig.update_layout(
            title=title,
            xaxis_title="Sample" if index is None else None,
            yaxis_title=y_title,
            hovermode="x unified",
            legend=_legend_top(),
            template="plotly_white",
            margin=dict(t=90),
        )
        return fig
    else:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(idx, y_true, label="Actual", color=BENCHMARK_COLOR, linewidth=1)
        ax.plot(idx, y_pred, label="Forecast", color=SERIES_COLORS[0], linewidth=1.6)
        ax.set_ylabel(y_title)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        return fig


def plot_backtest_results(
    backtest_df: pd.DataFrame,
    use_plotly: bool = True,
) -> Any:
    """Plot strategy vs buy-and-hold growth of $1 with the equity weight underneath.

    Parameters
    ----------
    backtest_df:
        DataFrame produced by :func:`backtest_allocation` / :func:`backtest_strategy`.
    use_plotly:
        Return Plotly figure if ``True``.

    Returns
    -------
    Figure object.
    """
    weight = backtest_df["weight"] if "weight" in backtest_df else 1 - backtest_df["signal"]
    strategy, bench = backtest_df["strategy_cumulative"], backtest_df["bh_cumulative"]

    if use_plotly:
        import plotly.graph_objects as go  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=bench.index, y=bench, mode="lines", name="Buy & hold",
                                 line=dict(color=BENCHMARK_COLOR, width=1.4),
                                 hovertemplate="$%{y:.2f}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=strategy.index, y=strategy, mode="lines", name="Strategy",
                                 line=dict(color=SERIES_COLORS[0], width=2),
                                 hovertemplate="$%{y:.2f}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=weight.index, y=weight, mode="lines", name="Equity weight",
                                 line=dict(color=SERIES_COLORS[0], width=1, shape="hv"),
                                 fill="tozeroy", fillcolor="rgba(42, 120, 214, 0.18)",
                                 hovertemplate="%{y:.0%}", showlegend=False), row=2, col=1)

        lo, hi = float(min(strategy.min(), bench.min())), float(max(strategy.max(), bench.max()))
        ticks = [v for v in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64) if lo / 1.5 <= v <= hi * 1.5]
        fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"${v:g}" for v in ticks],
                         title_text="Growth of $1", row=1, col=1)
        fig.update_yaxes(range=[0, 1.05], tickvals=[0, 0.5, 1], tickformat=".0%",
                         title_text="Equity weight", row=2, col=1)
        fig.update_layout(
            title="Strategy vs buy & hold",
            hovermode="x unified",
            legend=_legend_top(),
            template="plotly_white",
            margin=dict(t=90),
        )
        return fig
    else:
        import matplotlib.pyplot as plt  # type: ignore

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
        ax1.plot(bench.index, bench, label="Buy & hold", color=BENCHMARK_COLOR, linewidth=1.2)
        ax1.plot(strategy.index, strategy, label="Strategy", color=SERIES_COLORS[0], linewidth=1.6)
        ax1.set_yscale("log")
        ax1.set_ylabel("Growth of $1")
        ax1.set_title("Strategy vs buy & hold")
        ax1.legend()
        ax2.fill_between(weight.index, weight, step="post", color=SERIES_COLORS[0], alpha=0.25)
        ax2.set_ylabel("Equity weight")
        ax2.set_ylim(0, 1.05)
        fig.tight_layout()
        return fig
