"""
Machine-learning models for the Credit Spread Analysis & Prediction Platform.

Implements a unified training-and-evaluation interface for gradient-boosted
trees, random forests, ridge regression, a sign-constrained composite factor
model, and a ridge + LightGBM ensemble.

Validation is always *purged*: the training window ends ``gap`` rows before
the first test row, where ``gap`` equals the forecast horizon, so no training
label overlaps the test period.  :func:`walk_forward_predict` refits the model
periodically (yearly by default) to mimic live use.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit

from config.settings import MODEL_PARAMS
from src.features.engineering import horizon_from_target_name

logger = logging.getLogger(__name__)

TREE_MODELS = ("xgboost", "lightgbm", "random_forest")
MODEL_TYPES = TREE_MODELS + ("ridge", "composite", "ensemble")

# Sign-constrained composite: +1 means a higher value predicts spread widening.
# Signs come from economic priors, not from fitting.
COMPOSITE_WEIGHTS: dict[str, float] = {
    "eq_ret_20": -1.0,       # equity sell-offs lead credit
    "eq_ret_5": -1.0,
    "vix_chg_20": 1.0,       # rising implied volatility
    "spread_chg_20": 1.0,    # spread momentum
    "y10_chg_20": -1.0,      # Moody's yields lag Treasury moves
    "nfci_chg_20": 1.0,      # tightening financial conditions
}

# The same priors for targets that *rise* when spreads tighten (e.g. HY excess returns).
RETURN_COMPOSITE_WEIGHTS: dict[str, float] = {k: -v for k, v in COMPOSITE_WEIGHTS.items()}

# Column used to volatility-scale regression targets for tree models.
DEFAULT_SCALE_COL = "spread_vol60"


class CompositeFactorModel(BaseEstimator):
    """Equal-weight sum of clipped z-scores with fixed economic signs.

    ``fit`` estimates only the z-score normalisation and a single non-negative
    slope (regression) or a one-feature logistic calibration (classification),
    which keeps the model robust to regime change.
    """

    def __init__(self, weights: Optional[dict[str, float]] = None, task: str = "regression") -> None:
        self.weights = weights
        self.task = task

    def _score(self, X: pd.DataFrame) -> np.ndarray:
        z = (X[self.columns_] - self.mu_) / self.sd_
        z = z.clip(-3, 3).fillna(0.0)
        return (z.values * self.signs_).mean(axis=1)

    def fit(self, X: pd.DataFrame, y: Any) -> "CompositeFactorModel":
        weights = self.weights or COMPOSITE_WEIGHTS
        self.columns_ = [c for c in weights if c in X.columns]
        if not self.columns_:
            raise ValueError("None of the composite factor columns are present in X.")
        self.signs_ = np.array([weights[c] for c in self.columns_])
        self.mu_ = X[self.columns_].mean()
        self.sd_ = X[self.columns_].std().replace(0, np.nan)
        score = self._score(X)
        y_arr = np.asarray(y, dtype=float)
        if self.task == "regression":
            denom = float(np.dot(score, score))
            self.beta_ = max(0.0, float(np.dot(score, y_arr)) / denom) if denom > 0 else 0.0
        else:
            from sklearn.linear_model import LogisticRegression

            self.calibrator_ = LogisticRegression().fit(score.reshape(-1, 1), y_arr.astype(int))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "regression":
            return self.beta_ * self._score(X)
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator_.predict_proba(self._score(X).reshape(-1, 1))

    @property
    def feature_importances_(self) -> pd.Series:
        return pd.Series(np.abs(self.signs_) / len(self.signs_), index=self.columns_)


class ScaledTargetRegressor(BaseEstimator):
    """Fit ``y / scale`` and predict ``prediction * scale``.

    Dividing spread changes by trailing spread volatility stops crisis periods
    from dominating the squared-error loss.
    """

    def __init__(self, estimator: Any, scale_col: str = DEFAULT_SCALE_COL) -> None:
        self.estimator = estimator
        self.scale_col = scale_col

    def _scale(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(X[self.scale_col].to_numpy(dtype=float), self.floor_)

    def fit(self, X: pd.DataFrame, y: Any) -> "ScaledTargetRegressor":
        self.floor_ = float(X[self.scale_col].quantile(0.05))
        self.estimator.fit(X, np.asarray(y, dtype=float) / self._scale(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X) * self._scale(X)

    @property
    def feature_importances_(self) -> Any:
        return getattr(self.estimator, "feature_importances_", None)


class EnsembleRegressor(BaseEstimator):
    """Average of ridge (raw target) and LightGBM (volatility-scaled target)."""

    def __init__(self, scale_col: str = DEFAULT_SCALE_COL) -> None:
        self.scale_col = scale_col

    def fit(self, X: pd.DataFrame, y: Any) -> "EnsembleRegressor":
        self.members_ = [_build_model("ridge", "regression")]
        tree = _build_model("lightgbm", "regression")
        if self.scale_col in X.columns:
            tree = ScaledTargetRegressor(tree, self.scale_col)
        self.members_.append(tree)
        for m in self.members_:
            m.fit(X, y)
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.members_], axis=0)

    @property
    def feature_importances_(self) -> pd.Series:
        parts = [feature_importance(m, self.feature_names_) for m in self.members_]
        parts = [p / p.sum() for p in parts if len(p) and p.sum() > 0]
        return sum(parts) / len(parts) if parts else pd.Series(dtype=float)


def _build_model(model_type: str, task: str, params: Optional[dict] = None) -> Any:
    """Instantiate an unfitted model.

    Parameters
    ----------
    model_type:
        One of :data:`MODEL_TYPES`.
    task:
        ``"regression"`` or ``"classification"``.
    params:
        Hyper-parameter overrides.  Defaults to ``config.settings.MODEL_PARAMS``.
    """
    p = dict(params) if params is not None else dict(MODEL_PARAMS.get(model_type, {}))

    if model_type == "xgboost":
        import xgboost as xgb  # type: ignore

        return xgb.XGBRegressor(**p) if task == "regression" else xgb.XGBClassifier(**p)

    if model_type == "lightgbm":
        import lightgbm as lgb  # type: ignore

        return lgb.LGBMRegressor(**p) if task == "regression" else lgb.LGBMClassifier(**p)

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        return RandomForestRegressor(**p) if task == "regression" else RandomForestClassifier(**p)

    if model_type == "ridge":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if task == "regression":
            head = RidgeCV(alphas=p.get("alphas", np.logspace(-1, 4, 12)))
        else:
            head = LogisticRegression(C=p.get("C", 0.05), max_iter=2000)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), head)

    if model_type == "composite":
        return CompositeFactorModel(weights=p.get("weights"), task=task)

    if model_type == "ensemble":
        if task != "regression":
            raise ValueError("The ensemble model supports regression only.")
        return EnsembleRegressor(scale_col=p.get("scale_col", DEFAULT_SCALE_COL))

    raise ValueError(f"Unknown model_type '{model_type}'. Choose from {MODEL_TYPES}.")


def make_model(
    model_type: str,
    task: str = "regression",
    params: Optional[dict] = None,
    scale_col: Optional[str] = None,
) -> Any:
    """Build a model, wrapping it in :class:`ScaledTargetRegressor` when *scale_col* is set."""
    model = _build_model(model_type, task, params)
    if scale_col and task == "regression" and model_type not in ("composite", "ensemble"):
        model = ScaledTargetRegressor(model, scale_col)
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_signal_sharpe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    annualisation_factor: float = 252.0,
    horizon: int = 1,
) -> float:
    """Annualised Sharpe ratio of a sign-based trading signal.

    Position is ``sign(y_pred)`` and the period return is ``position * y_true``.
    For *horizon* > 1 the targets overlap, so only every *horizon*-th
    observation is used and the ratio is annualised with ``sqrt(252 / horizon)``.
    """
    horizon = max(int(horizon), 1)
    signals = np.sign(np.asarray(y_pred, dtype=float))[::horizon]
    strategy_returns = signals * np.asarray(y_true, dtype=float)[::horizon]

    std = strategy_returns.std()
    if len(strategy_returns) < 2 or std == 0:
        return 0.0
    return float(strategy_returns.mean() / std * np.sqrt(annualisation_factor / horizon))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "regression",
    horizon: int = 1,
    base_rate: Optional[float] = None,
) -> dict[str, float]:
    """Compute out-of-sample evaluation metrics.

    Regression
        ``rmse``, ``mae``, ``oos_r2`` (skill versus a zero-change forecast),
        ``ic`` (Spearman rank correlation), ``directional_accuracy`` and
        ``signal_sharpe`` (non-overlapping, see :func:`compute_signal_sharpe`).
    Classification
        ``accuracy``, ``roc_auc``, ``brier``, ``brier_skill`` (versus
        *base_rate*, default: the sample positive rate) and ``base_rate``.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    metrics: dict[str, float] = {}

    if task == "regression":
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        denom = float(np.sum(y_true ** 2))
        metrics["oos_r2"] = float(1 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else float("nan")
        if np.std(y_pred) > 0 and np.std(y_true) > 0:
            metrics["ic"] = float(spearmanr(y_pred, y_true)[0])
        else:
            metrics["ic"] = float("nan")
        metrics["directional_accuracy"] = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
        metrics["signal_sharpe"] = compute_signal_sharpe(y_true, y_pred, horizon=horizon)
    else:
        y_int = y_true.astype(int)
        metrics["accuracy"] = float(accuracy_score(y_int, (y_pred >= 0.5).astype(int)))
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_int, y_pred))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        p = np.clip(y_pred, 0, 1)
        rate = float(np.mean(y_int)) if base_rate is None else float(base_rate)
        brier = float(np.mean((p - y_int) ** 2))
        brier_ref = float(np.mean((rate - y_int) ** 2))
        metrics["brier"] = brier
        metrics["brier_skill"] = float(1 - brier / brier_ref) if brier_ref > 0 else float("nan")
        metrics["base_rate"] = float(np.mean(y_int))

    return metrics


def feature_importance(model: Any, feature_names: list[str]) -> pd.Series:
    """Return a non-negative importance Series for any supported model."""
    fi = getattr(model, "feature_importances_", None)
    if isinstance(fi, pd.Series):
        return fi.reindex(feature_names).fillna(0.0).sort_values(ascending=False)
    if fi is None and hasattr(model, "steps"):
        coef = getattr(model.steps[-1][1], "coef_", None)
        if coef is not None:
            fi = np.abs(np.ravel(coef))
    if fi is None:
        return pd.Series(dtype=float)
    return pd.Series(np.asarray(fi, dtype=float), index=feature_names).sort_values(ascending=False)


def _resolve_gap(y: Any, gap: Optional[int]) -> int:
    if gap is not None:
        return int(gap)
    parsed = horizon_from_target_name(getattr(y, "name", None))
    if parsed is None:
        logger.warning("Could not infer forecast horizon from target name; using gap=0 (labels may overlap).")
        return 0
    return parsed


def _predict(model: Any, X: pd.DataFrame, task: str) -> np.ndarray:
    return model.predict(X) if task == "regression" else model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    task: str = "regression",
    n_splits: int = 5,
    params: Optional[dict] = None,
    gap: Optional[int] = None,
    scale_col: Optional[str] = None,
) -> dict[str, Any]:
    """Train a model with purged time-series cross-validation and evaluate it.

    Parameters
    ----------
    X:
        Feature DataFrame (sorted by time).
    y:
        Target Series aligned with *X*.
    model_type:
        One of :data:`MODEL_TYPES`.
    task:
        ``"regression"`` or ``"classification"``.
    n_splits:
        Number of expanding-window folds.
    params:
        Optional hyper-parameter overrides.
    gap:
        Rows purged between each training window and its test fold.  Defaults
        to the horizon parsed from ``y.name`` (``target_{h}d_*``).
    scale_col:
        Volatility column used to scale regression targets (tree models only).

    Returns
    -------
    dict
        Keys: ``model`` (fitted on all data), ``cv_metrics`` (per fold),
        ``mean_metrics``, ``oof_predictions`` (NaN outside test folds) and
        ``feature_importance``.
    """
    X = pd.DataFrame(X).astype(float)
    y = pd.Series(np.asarray(y, dtype=float), index=X.index, name=getattr(y, "name", None))
    gap = _resolve_gap(y, gap)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    cv_metrics: list[dict[str, float]] = []
    oof_predictions = np.full(len(y), np.nan)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model = make_model(model_type, task, params, scale_col)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = _predict(model, X.iloc[val_idx], task)
        oof_predictions[val_idx] = preds
        base_rate = float(y.iloc[train_idx].mean()) if task == "classification" else None
        fold_metrics = compute_metrics(y.iloc[val_idx].values, preds, task, horizon=max(gap, 1), base_rate=base_rate)
        cv_metrics.append(fold_metrics)
        logger.info("Fold %d: %s", fold + 1, {k: round(v, 4) for k, v in fold_metrics.items()})

    mean_metrics: dict[str, float] = {}
    for k in cv_metrics[0]:
        values = np.array([m[k] for m in cv_metrics], dtype=float)
        mean_metrics[k] = float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else float("nan")

    final_model = make_model(model_type, task, params, scale_col)
    final_model.fit(X, y)

    return {
        "model": final_model,
        "cv_metrics": cv_metrics,
        "mean_metrics": mean_metrics,
        "oof_predictions": oof_predictions,
        "feature_importance": feature_importance(final_model, list(X.columns)),
    }


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "ensemble",
    task: str = "regression",
    start: str = "2000-01-01",
    end: Optional[str] = None,
    refit_freq: str = "YS",
    gap: Optional[int] = None,
    min_train: int = 756,
    params: Optional[dict] = None,
    scale_col: Optional[str] = None,
) -> pd.Series:
    """Generate out-of-sample predictions with periodic refitting.

    For each period starting at ``refit_freq`` boundaries between *start* and
    *end*, the model is fitted on every row whose label is fully realised
    before the period begins (``gap`` rows are purged) and then predicts the
    whole period.

    Returns
    -------
    pd.Series
        Predictions indexed like *X*; NaN before *start* or where there was
        too little training data.
    """
    X = pd.DataFrame(X).astype(float)
    y = pd.Series(np.asarray(y, dtype=float), index=X.index, name=getattr(y, "name", None))
    gap = _resolve_gap(y, gap)
    end_ts = pd.Timestamp(end) if end is not None else X.index.max()
    preds = pd.Series(np.nan, index=X.index, name="prediction")

    boundaries = list(pd.date_range(start, end_ts, freq=refit_freq))
    if not boundaries or boundaries[0] > pd.Timestamp(start):
        boundaries.insert(0, pd.Timestamp(start))
    boundaries.append(end_ts + pd.to_timedelta(1, unit="D"))

    for period_start, period_end in zip(boundaries[:-1], boundaries[1:]):
        test_mask = (X.index >= period_start) & (X.index < period_end)
        if not test_mask.any():
            continue
        train_stop = int(np.argmax(test_mask)) - gap
        if train_stop <= 0:
            continue
        X_tr, y_tr = X.iloc[:train_stop], y.iloc[:train_stop]
        ok = y_tr.notna().to_numpy()
        if ok.sum() < min_train:
            continue
        model = make_model(model_type, task, params, scale_col)
        model.fit(X_tr[ok], y_tr[ok])
        preds[test_mask] = _predict(model, X[test_mask], task)

    return preds


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    model_type: str = "xgboost",
) -> np.ndarray:
    """Compute SHAP values of shape ``(n_samples, n_features)`` for a tree model."""
    try:
        import shap  # type: ignore
    except ImportError as exc:
        raise ImportError("shap is required: pip install shap") from exc

    if isinstance(model, ScaledTargetRegressor):
        model = model.estimator
    if model_type not in TREE_MODELS:
        raise ValueError(f"SHAP values are only supported for tree models, not '{model_type}'.")

    X_arr = np.asarray(X, dtype=float)
    shap_values = shap.TreeExplainer(model).shap_values(X_arr)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:  # (n, features, classes)
        shap_values = shap_values[..., -1]
    return shap_values
