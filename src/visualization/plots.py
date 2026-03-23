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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recession_bands(fig: Any) -> Any:
    """Add approximate US recession shading to a Plotly figure."""
    import plotly.graph_objects as go  # type: ignore

    recessions = [
        ("2001-03-01", "2001-11-01"),
        ("2007-12-01", "2009-06-01"),
        ("2020-02-01", "2020-04-01"),
    ]
    for start, end in recessions:
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="grey",
            opacity=0.15,
            layer="below",
            line_width=0,
        )
    return fig


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
        spread_cols = [c for c in ["hy_spread", "ig_spread", "bbb_spread"] if c in df.columns]
    if not spread_cols:
        raise ValueError("No spread columns found in DataFrame.")

    if use_plotly:
        import plotly.graph_objects as go  # type: ignore

        fig = go.Figure()
        for col in spread_cols:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
        fig = _recession_bands(fig)
        fig.update_layout(
            title="Credit Spread History",
            xaxis_title="Date",
            yaxis_title="Spread (bps)",
            hovermode="x unified",
            template="plotly_white",
        )
        return fig
    else:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(14, 5))
        for col in spread_cols:
            ax.plot(df.index, df[col], label=col)
        ax.set_title("Credit Spread History")
        ax.set_xlabel("Date")
        ax.set_ylabel("Spread (bps)")
        ax.legend()
        fig.tight_layout()
        return fig


def plot_regime_overlay(
    df: pd.DataFrame,
    regimes: np.ndarray,
    spread_col: str = "hy_spread",
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

    colours = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    unique_regimes = sorted(np.unique(regimes))

    if use_plotly:
        import plotly.graph_objects as go  # type: ignore

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
                    name=f"Regime {r}",
                )
            )
        fig.update_layout(
            title=f"{spread_col} Coloured by Regime",
            xaxis_title="Date",
            yaxis_title="Spread (bps)",
            template="plotly_white",
        )
        return fig
    else:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(14, 5))
        for i, r in enumerate(unique_regimes):
            mask = regimes == r
            ax.scatter(df.index[mask], df[spread_col].values[mask], s=4, label=f"Regime {r}")
        ax.set_title(f"{spread_col} Coloured by Regime")
        ax.legend()
        fig.tight_layout()
        return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    window: int = 60,
    use_plotly: bool = True,
) -> Any:
    """Plot a rolling correlation heatmap.

    Parameters
    ----------
    df:
        DataFrame with numeric columns.
    window:
        Rolling window for correlation computation.
    use_plotly:
        Return Plotly figure if ``True``.

    Returns
    -------
    Figure object.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.rolling(window).corr().iloc[-len(numeric_df.columns) :]

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
        fig.update_layout(title=f"Correlation Heatmap (rolling {window}d)", template="plotly_white")
        return fig
    else:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore

        full_corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(full_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
        ax.set_title(f"Correlation Heatmap (rolling {window}d)")
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
        import plotly.graph_objects as go  # type: ignore

        fig = go.Figure(
            go.Bar(x=top_values[::-1], y=top_names[::-1], orientation="h")
        )
        fig.update_layout(
            title="SHAP Feature Importance (mean |SHAP value|)",
            xaxis_title="|SHAP value|",
            template="plotly_white",
        )
        return fig
    else:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(8, max_display * 0.4 + 1))
        ax.barh(top_names[::-1], top_values[::-1])
        ax.set_xlabel("|SHAP value|")
        ax.set_title("SHAP Feature Importance")
        fig.tight_layout()
        return fig


def plot_forecast_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    use_plotly: bool = True,
    title: str = "Forecast vs Actual",
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

    Returns
    -------
    Figure object.
    """
    idx = np.arange(len(y_true))

    if use_plotly:
        import plotly.graph_objects as go  # type: ignore

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx, y=y_true, mode="lines", name="Actual"))
        fig.add_trace(go.Scatter(x=idx, y=y_pred, mode="lines", name="Predicted", line=dict(dash="dash")))
        fig.update_layout(title=title, xaxis_title="Sample", yaxis_title="Value", template="plotly_white")
        return fig
    else:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(idx, y_true, label="Actual")
        ax.plot(idx, y_pred, label="Predicted", linestyle="--")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        return fig


def plot_backtest_results(
    backtest_df: pd.DataFrame,
    use_plotly: bool = True,
) -> Any:
    """Plot strategy vs buy-and-hold equity curves with signal overlay.

    Parameters
    ----------
    backtest_df:
        DataFrame produced by :func:`backtest_strategy`.
    use_plotly:
        Return Plotly figure if ``True``.

    Returns
    -------
    Figure object.
    """
    if use_plotly:
        import plotly.graph_objects as go  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25])
        fig.add_trace(
            go.Scatter(
                x=backtest_df.index,
                y=backtest_df["strategy_cumulative"],
                mode="lines",
                name="Strategy",
                line=dict(color="#2196F3"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=backtest_df.index,
                y=backtest_df["bh_cumulative"],
                mode="lines",
                name="Buy & Hold",
                line=dict(color="#FF5722", dash="dash"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=backtest_df.index,
                y=backtest_df["signal"],
                name="Signal (1=Defensive)",
                marker_color="rgba(128,128,128,0.4)",
            ),
            row=2, col=1,
        )
        fig.update_layout(
            title="Backtest: Strategy vs Buy & Hold",
            template="plotly_white",
            hovermode="x unified",
        )
        return fig
    else:
        import matplotlib.pyplot as plt  # type: ignore

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        ax1.plot(backtest_df.index, backtest_df["strategy_cumulative"], label="Strategy")
        ax1.plot(backtest_df.index, backtest_df["bh_cumulative"], label="Buy & Hold", linestyle="--")
        ax1.set_title("Backtest: Strategy vs Buy & Hold")
        ax1.legend()
        ax2.bar(backtest_df.index, backtest_df["signal"], color="grey", alpha=0.5, label="Signal")
        ax2.set_ylabel("Signal")
        ax2.legend()
        fig.tight_layout()
        return fig
