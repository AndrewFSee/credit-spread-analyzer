"""
Regime detection models for the Credit Spread Analysis & Prediction Platform.

Implements Hidden Markov Model (HMM) and Gaussian Mixture Model (GMM) based
regime detection, with utilities for labelling observations, computing
regime-conditional statistics, and extracting transition matrices.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fit_hmm(
    data: np.ndarray,
    n_states: int = 3,
    n_iter: int = 200,
    random_state: int = 42,
) -> "hmmlearn.hmm.GaussianHMM":  # type: ignore[name-defined]
    """Fit a Gaussian HMM to the supplied data.

    Parameters
    ----------
    data:
        2-D array of shape ``(n_observations, n_features)``.
    n_states:
        Number of hidden states (regimes).
    n_iter:
        Maximum EM iterations.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    hmmlearn.hmm.GaussianHMM
        Fitted model.
    """
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore
    except ImportError as exc:
        raise ImportError("hmmlearn is required: pip install hmmlearn") from exc

    data_2d = np.asarray(data, dtype=float)
    if data_2d.ndim == 1:
        data_2d = data_2d.reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(data_2d)
    logger.info("HMM fitted: %d states, log-likelihood=%.2f", n_states, model.score(data_2d))
    return model


def fit_gmm(
    data: np.ndarray,
    n_components: int = 3,
    random_state: int = 42,
) -> "sklearn.mixture.GaussianMixture":  # type: ignore[name-defined]
    """Fit a Gaussian Mixture Model to the supplied data.

    Parameters
    ----------
    data:
        2-D array of shape ``(n_observations, n_features)``.
    n_components:
        Number of mixture components.
    random_state:
        Random seed.

    Returns
    -------
    sklearn.mixture.GaussianMixture
        Fitted model.
    """
    from sklearn.mixture import GaussianMixture  # type: ignore

    data_2d = np.asarray(data, dtype=float)
    if data_2d.ndim == 1:
        data_2d = data_2d.reshape(-1, 1)

    model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        n_init=5,
    )
    model.fit(data_2d)
    logger.info("GMM fitted: %d components, BIC=%.2f", n_components, model.bic(data_2d))
    return model


def label_regimes(
    model: object,
    data: np.ndarray,
    model_type: str = "hmm",
) -> np.ndarray:
    """Assign regime labels to observations using a fitted model.

    Parameters
    ----------
    model:
        Fitted HMM or GMM model.
    data:
        Array of shape ``(n_observations,)`` or ``(n_observations, n_features)``.
    model_type:
        Either ``"hmm"`` or ``"gmm"``.

    Returns
    -------
    np.ndarray
        Integer array of regime labels, shape ``(n_observations,)``.
    """
    data_2d = np.asarray(data, dtype=float)
    if data_2d.ndim == 1:
        data_2d = data_2d.reshape(-1, 1)

    if model_type == "hmm":
        labels: np.ndarray = model.predict(data_2d)  # type: ignore[union-attr]
    elif model_type == "gmm":
        labels = model.predict(data_2d)  # type: ignore[union-attr]
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'hmm' or 'gmm'.")

    return labels


def compute_regime_stats(
    df: pd.DataFrame,
    regimes: np.ndarray,
    equity_col: str = "sp500_return",
    spread_col: str = "hy_spread",
) -> pd.DataFrame:
    """Compute per-regime descriptive statistics.

    Parameters
    ----------
    df:
        DataFrame aligned with *regimes*.
    regimes:
        Integer array of regime labels.
    equity_col:
        Column name for equity returns.
    spread_col:
        Column name for the credit spread level.

    Returns
    -------
    pd.DataFrame
        Table with regime as index and stats as columns.
    """
    df = df.copy()
    df["_regime"] = regimes

    stats_rows = []
    for regime_id in sorted(np.unique(regimes)):
        mask = df["_regime"] == regime_id
        subset = df.loc[mask]
        row: dict[str, float | int] = {"regime": int(regime_id), "count": int(mask.sum())}

        if equity_col in df.columns:
            row["mean_equity_return"] = float(subset[equity_col].mean())
            row["equity_volatility"] = float(subset[equity_col].std())

        if spread_col in df.columns:
            row["mean_spread"] = float(subset[spread_col].mean())
            spread_chg = subset[spread_col].diff()
            row["mean_spread_change"] = float(spread_chg.mean())
            row["spread_volatility"] = float(spread_chg.std())

        stats_rows.append(row)

    return pd.DataFrame(stats_rows).set_index("regime")


def get_transition_matrix(
    model: object,
    model_type: str = "hmm",
) -> pd.DataFrame:
    """Extract the regime transition probability matrix.

    Parameters
    ----------
    model:
        Fitted HMM or GMM model.
    model_type:
        ``"hmm"`` returns the model's ``transmat_`` attribute.  ``"gmm"`` does
        not have a transition matrix; a uniform matrix is returned instead.

    Returns
    -------
    pd.DataFrame
        Square DataFrame of transition probabilities.
    """
    if model_type == "hmm":
        mat: np.ndarray = model.transmat_  # type: ignore[union-attr]
        n = mat.shape[0]
        labels = [f"State {i}" for i in range(n)]
        return pd.DataFrame(mat, index=labels, columns=labels)
    elif model_type == "gmm":
        n = model.n_components  # type: ignore[union-attr]
        mat = np.full((n, n), 1.0 / n)
        labels = [f"Component {i}" for i in range(n)]
        logger.info("GMM has no transition matrix – returning uniform matrix.")
        return pd.DataFrame(mat, index=labels, columns=labels)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'.")
