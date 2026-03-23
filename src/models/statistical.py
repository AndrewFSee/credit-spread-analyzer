"""
Statistical time-series models for the Credit Spread Analysis & Prediction Platform.

Implements Granger causality tests, Vector Autoregression (VAR), Impulse
Response Functions (IRF), Forecast Error Variance Decomposition (FEVD), and
Johansen cointegration tests.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_granger_causality(
    df: pd.DataFrame,
    caused: str,
    causing: str,
    maxlag: int = 10,
) -> dict[int, float]:
    """Run Granger causality test: does *causing* Granger-cause *caused*?

    Parameters
    ----------
    df:
        DataFrame containing both columns.
    caused:
        Name of the dependent (caused) variable column.
    causing:
        Name of the predictor (causing) variable column.
    maxlag:
        Maximum number of lags to test.

    Returns
    -------
    dict[int, float]
        Mapping of ``{lag: p_value}`` for each lag from 1 to *maxlag*.
    """
    from statsmodels.tsa.stattools import grangercausalitytests  # type: ignore

    subset = df[[caused, causing]].dropna()
    if len(subset) < maxlag * 3:
        raise ValueError(
            f"Insufficient observations ({len(subset)}) for maxlag={maxlag}."
        )

    results = grangercausalitytests(subset, maxlag=maxlag, verbose=False)
    p_values: dict[int, float] = {}
    for lag, res in results.items():
        # Use F-test p-value (first test in the tuple)
        p_values[lag] = float(res[0]["ssr_ftest"][1])

    return p_values


def fit_var_model(
    df: pd.DataFrame,
    columns: list[str],
    maxlags: int = 10,
) -> Any:
    """Fit a Vector Autoregression model and select lag order by AIC.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to include in the VAR.
    maxlags:
        Maximum lag order to consider.

    Returns
    -------
    statsmodels VARResultsWrapper
        Fitted VAR result object.
    """
    from statsmodels.tsa.vector_ar.var_model import VAR  # type: ignore

    subset = df[columns].dropna()
    model = VAR(subset)
    result = model.fit(maxlags=maxlags, ic="aic", verbose=False)
    logger.info("VAR fitted: selected %d lags (AIC=%.4f)", result.k_ar, result.aic)
    return result


def compute_irf(
    var_result: Any,
    periods: int = 20,
) -> Any:
    """Compute Impulse Response Functions from a fitted VAR.

    Parameters
    ----------
    var_result:
        Fitted VAR result (``statsmodels`` VARResultsWrapper).
    periods:
        Number of periods to compute responses for.

    Returns
    -------
    statsmodels IRAnalysis
        IRF object with ``.irfs`` attribute (shape: periods × k × k).
    """
    irf = var_result.irf(periods=periods)
    logger.info("IRF computed for %d periods, %d variables.", periods, var_result.k_ar)
    return irf


def compute_variance_decomposition(
    var_result: Any,
    periods: int = 20,
) -> Any:
    """Compute Forecast Error Variance Decomposition (FEVD).

    Parameters
    ----------
    var_result:
        Fitted VAR result.
    periods:
        Number of periods for decomposition.

    Returns
    -------
    statsmodels FEVD
        FEVD object.
    """
    fevd = var_result.fevd(periods=periods)
    logger.info("FEVD computed for %d periods.", periods)
    return fevd


def run_johansen_cointegration(
    df: pd.DataFrame,
    columns: list[str],
    det_order: int = 0,
    k_ar_diff: int = 2,
) -> Any:
    """Run the Johansen cointegration test.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to test for cointegration.
    det_order:
        Deterministic terms: -1 = no constant, 0 = constant, 1 = linear trend.
    k_ar_diff:
        Number of lagged differences in the VECM.

    Returns
    -------
    statsmodels JohansenTestResult
        Result object with trace/eigen statistics and critical values.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen  # type: ignore

    subset = df[columns].dropna()
    result = coint_johansen(subset.values, det_order=det_order, k_ar_diff=k_ar_diff)
    logger.info("Johansen test completed for %d variables.", len(columns))
    return result


def summarize_granger(
    granger_results: dict[int, float],
    caused: str = "Y",
    causing: str = "X",
    significance: float = 0.05,
) -> str:
    """Return a human-readable summary of Granger causality results.

    Parameters
    ----------
    granger_results:
        Dict returned by :func:`run_granger_causality`.
    caused:
        Name of the caused variable (for display).
    causing:
        Name of the causing variable (for display).
    significance:
        P-value threshold for significance.

    Returns
    -------
    str
        Formatted summary string.
    """
    lines = [
        f"Granger Causality: does '{causing}' → '{caused}'?",
        f"{'Lag':>5}  {'p-value':>10}  {'Significant':>12}",
        "-" * 32,
    ]
    for lag in sorted(granger_results.keys()):
        p = granger_results[lag]
        sig = "✓" if p < significance else "✗"
        lines.append(f"{lag:>5}  {p:>10.4f}  {sig:>12}")

    sig_lags = [lag for lag, p in granger_results.items() if p < significance]
    if sig_lags:
        lines.append(f"\nSignificant at lags: {sig_lags}")
    else:
        lines.append(f"\nNo significant lags at α={significance}.")

    summary = "\n".join(lines)
    print(summary)
    return summary
