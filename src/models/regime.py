"""
Regime detection models for the Credit Spread Analysis & Prediction Platform.

Implements Hidden Markov Model (HMM) and Gaussian Mixture Model (GMM) based
regime detection, with utilities for labelling observations, computing
regime-conditional statistics, and extracting transition matrices.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import FRED_DAILY_PUBLICATION_LAG

logger = logging.getLogger(__name__)


def _state_order(means: np.ndarray) -> np.ndarray:
    """Rank states by the mean of the first feature (0 = lowest, e.g. tightest spreads)."""
    return np.argsort(np.argsort(np.asarray(means)[:, 0]))


def _relabel(model: object, raw_labels: np.ndarray) -> np.ndarray:
    order = getattr(model, "state_order_", None)
    return raw_labels if order is None else order[raw_labels]


def fit_hmm(
    data: np.ndarray,
    n_states: int = 3,
    n_iter: int = 200,
    random_state: int = 42,
    n_init: int = 10,
) -> "hmmlearn.hmm.GaussianHMM":  # type: ignore[name-defined]
    """Fit a Gaussian HMM to the supplied data.

    EM only finds a local optimum; on spread levels a single run can converge
    to a degenerate solution with two identical states that alternate every
    day.  The model is therefore fitted *n_init* times from different seeds
    and the fit with the highest log-likelihood is kept.

    Parameters
    ----------
    data:
        2-D array of shape ``(n_observations, n_features)``.
    n_states:
        Number of hidden states (regimes).
    n_iter:
        Maximum EM iterations.
    random_state:
        Random seed for reproducibility (seeds ``random_state … random_state + n_init - 1`` are used).
    n_init:
        Number of random restarts.

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

    best_model, best_score = None, -np.inf
    for i in range(max(n_init, 1)):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state + i,
        )
        try:
            model.fit(data_2d)
            score = model.score(data_2d)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.debug("HMM restart %d failed: %s", i, exc)
            continue
        if np.isfinite(score) and score > best_score:
            best_model, best_score = model, score

    if best_model is None:
        raise RuntimeError("HMM fitting failed for every restart.")
    best_model.state_order_ = _state_order(best_model.means_)
    logger.info("HMM fitted: %d states, log-likelihood=%.2f", n_states, best_score)
    return best_model


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
    model.state_order_ = _state_order(model.means_)
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
        Integer array of regime labels, shape ``(n_observations,)``, ordered so
        that 0 is the regime with the lowest mean of the first feature.
    """
    data_2d = np.asarray(data, dtype=float)
    if data_2d.ndim == 1:
        data_2d = data_2d.reshape(-1, 1)

    if model_type not in ("hmm", "gmm"):
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'hmm' or 'gmm'.")

    # HMM labels come from the Viterbi path, which uses the whole sample
    # (including future observations).  Use filtered_regime_probabilities()
    # for a real-time view.
    labels: np.ndarray = model.predict(data_2d)  # type: ignore[union-attr]
    return _relabel(model, labels)


def compute_regime_stats(
    df: pd.DataFrame,
    regimes: np.ndarray,
    equity_col: str = "sp500_return",
    spread_col: str = "baa_spread",
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
    if spread_col in df.columns:
        # Difference on the full series so changes never span two separate regime episodes.
        df["_spread_chg"] = df[spread_col].diff()

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
            spread_chg = subset["_spread_chg"]
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
        order = getattr(model, "state_order_", None)
        if order is not None:
            inv = np.argsort(order)  # inv[k] = raw state with rank k
            mat = mat[np.ix_(inv, inv)]
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


def filtered_regime_probabilities(model: object, data: np.ndarray) -> np.ndarray:
    """Causal (forward-filtered) regime probabilities from a fitted HMM.

    Row *t* is ``P(state_t | x_1 … x_t)`` – it does not use any observation
    after *t*, unlike :func:`label_regimes`.  Columns follow the ordered
    state labels.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_observations, n_states)``.
    """
    from scipy.stats import multivariate_normal

    data_2d = np.asarray(data, dtype=float)
    if data_2d.ndim == 1:
        data_2d = data_2d.reshape(-1, 1)

    n_states = model.n_components  # type: ignore[union-attr]
    covars = np.asarray(model.covars_)  # full covariance matrices for every covariance_type
    log_emit = np.column_stack([
        multivariate_normal(mean=model.means_[k], cov=covars[k], allow_singular=True).logpdf(data_2d)  # type: ignore[union-attr]
        for k in range(n_states)
    ]).reshape(len(data_2d), n_states)
    # Scale each row by its maximum so the likelihoods stay representable.
    emit = np.exp(log_emit - log_emit.max(axis=1, keepdims=True))
    trans = np.asarray(model.transmat_)  # type: ignore[union-attr]

    probs = np.empty_like(emit)
    alpha = np.asarray(model.startprob_) * emit[0]  # type: ignore[union-attr]
    probs[0] = alpha / alpha.sum()
    for t in range(1, len(emit)):
        alpha = (probs[t - 1] @ trans) * emit[t]
        total = alpha.sum()
        probs[t] = alpha / total if total > 0 else probs[t - 1] @ trans

    order = getattr(model, "state_order_", None)
    if order is not None:
        probs = probs[:, np.argsort(order)]
    return probs


def walk_forward_regime_probabilities(
    series: pd.Series,
    n_states: int = 3,
    start: Optional[str] = None,
    refit_freq: str = "YS",
    min_train: int = 756,
    n_init: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Real-time regime probabilities with periodic HMM refits.

    For each period beginning at a ``refit_freq`` boundary, the HMM is fitted
    only on observations before the period and then run as a forward filter
    through the end of the period.  Neither the parameters nor the filter use
    any later data, so the output can be used as a model feature or a trading
    signal.

    Parameters
    ----------
    series:
        Observed series (e.g. a credit spread, already shifted for its
        publication lag).
    n_states:
        Number of regimes.
    start:
        First date for which probabilities are produced.  Defaults to the
        first refit boundary after *min_train* observations.
    refit_freq:
        Pandas offset alias for refit dates (``"YS"`` = every January).
    min_train:
        Minimum number of observations required before the first fit.
    n_init:
        HMM random restarts per fit.

    Returns
    -------
    pd.DataFrame
        Columns ``regime_prob_0 … regime_prob_{n-1}`` (0 = calmest), indexed
        like *series*; NaN before *start* or before enough history exists.
    """
    s = series.dropna()
    cols = [f"regime_prob_{k}" for k in range(n_states)]
    out = pd.DataFrame(np.nan, index=series.index, columns=cols)
    if s.empty:
        return out

    if start is None:
        if len(s) <= min_train:
            return out
        start = str(s.index[min_train].date())
    end = s.index.max()
    boundaries = list(pd.date_range(start, end, freq=refit_freq))
    if not boundaries or boundaries[0] > pd.Timestamp(start):
        boundaries.insert(0, pd.Timestamp(start))
    boundaries.append(end + pd.to_timedelta(1, unit="D"))

    values = s.to_numpy(dtype=float).reshape(-1, 1)
    for period_start, period_end in zip(boundaries[:-1], boundaries[1:]):
        n_train = int(np.searchsorted(s.index.values, np.datetime64(period_start)))
        n_upto = int(np.searchsorted(s.index.values, np.datetime64(period_end)))
        if n_train < min_train or n_upto <= n_train:
            continue
        model = fit_hmm(values[:n_train], n_states=n_states, n_init=n_init, random_state=random_state)
        probs = filtered_regime_probabilities(model, values[:n_upto])
        out.loc[s.index[n_train:n_upto], cols] = probs[n_train:n_upto]
    return out


def real_time_regime_probabilities(
    df: pd.DataFrame,
    spread_col: str = "baa_spread",
    n_states: int = 3,
    publication_lag: int = FRED_DAILY_PUBLICATION_LAG,
    **kwargs,
) -> pd.DataFrame:
    """Walk-forward regime probabilities for *spread_col* as known at each close.

    The spread is shifted by its publication lag before the HMM sees it.
    Extra keyword arguments go to :func:`walk_forward_regime_probabilities`.
    """
    if spread_col not in df.columns:
        raise ValueError(f"Column '{spread_col}' not found in DataFrame.")
    return walk_forward_regime_probabilities(df[spread_col].shift(publication_lag), n_states=n_states, **kwargs)
