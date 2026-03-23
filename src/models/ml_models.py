"""
Machine-learning models for the Credit Spread Analysis & Prediction Platform.

Implements a unified training-and-evaluation interface supporting XGBoost,
LightGBM, and Random Forest, with time-series cross-validation, SHAP
explainability, and Sharpe-ratio-based signal evaluation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# Default hyper-parameter grids
_DEFAULT_PARAMS: dict[str, dict] = {
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    },
    "lightgbm": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_leaf": 10,
        "max_features": 0.5,
        "random_state": 42,
        "n_jobs": -1,
    },
}


def _build_model(model_type: str, task: str, params: Optional[dict] = None) -> Any:
    """Instantiate a model object.

    Parameters
    ----------
    model_type:
        One of ``"xgboost"``, ``"lightgbm"``, or ``"random_forest"``.
    task:
        ``"regression"`` or ``"classification"``.
    params:
        Hyper-parameter dict.  Defaults to ``_DEFAULT_PARAMS[model_type]``.

    Returns
    -------
    Unfitted model object.
    """
    p = params if params is not None else _DEFAULT_PARAMS.get(model_type, {})

    if model_type == "xgboost":
        import xgboost as xgb  # type: ignore

        if task == "regression":
            return xgb.XGBRegressor(**p)
        return xgb.XGBClassifier(**p)

    elif model_type == "lightgbm":
        import lightgbm as lgb  # type: ignore

        if task == "regression":
            return lgb.LGBMRegressor(**p)
        return lgb.LGBMClassifier(**p)

    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # type: ignore

        if task == "regression":
            return RandomForestRegressor(**p)
        return RandomForestClassifier(**p)

    else:
        raise ValueError(f"Unknown model_type '{model_type}'.")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "regression",
) -> dict[str, float]:
    """Compute evaluation metrics.

    Parameters
    ----------
    y_true:
        Ground-truth array.
    y_pred:
        Predicted values or probabilities.
    task:
        ``"regression"`` or ``"classification"``.

    Returns
    -------
    dict[str, float]
        Dictionary with metric names and values.
    """
    from sklearn.metrics import (  # type: ignore
        accuracy_score,
        mean_absolute_error,
        mean_squared_error,
        roc_auc_score,
    )

    metrics: dict[str, float] = {}

    if task == "regression":
        mse = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = float(np.sqrt(mse))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        # Directional accuracy
        direction_true = np.sign(y_true)
        direction_pred = np.sign(y_pred)
        metrics["directional_accuracy"] = float(
            np.mean(direction_true == direction_pred)
        )
        metrics["signal_sharpe"] = float(compute_signal_sharpe(y_true, y_pred))
    else:
        y_pred_bin = (np.asarray(y_pred) >= 0.5).astype(int)
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred_bin))
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
        except Exception:  # noqa: BLE001
            metrics["roc_auc"] = float("nan")

    return metrics


def compute_signal_sharpe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    annualisation_factor: float = 252.0,
) -> float:
    """Compute the annualised Sharpe ratio of a sign-based trading signal.

    A long position is taken when the predicted value > 0, short otherwise.
    The strategy return for each period is ``sign(y_pred) * y_true``.

    Parameters
    ----------
    y_true:
        Actual returns for each period.
    y_pred:
        Model predictions whose sign determines the trade direction.
    annualisation_factor:
        252 for daily data.

    Returns
    -------
    float
        Annualised Sharpe ratio.
    """
    signals = np.sign(np.asarray(y_pred, dtype=float))
    strategy_returns = signals * np.asarray(y_true, dtype=float)

    std = strategy_returns.std()
    if std == 0:
        return 0.0
    return float(strategy_returns.mean() / std * np.sqrt(annualisation_factor))


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    task: str = "regression",
    n_splits: int = 5,
    params: Optional[dict] = None,
) -> dict[str, Any]:
    """Train a model with time-series cross-validation and evaluate it.

    Parameters
    ----------
    X:
        Feature DataFrame (must be sorted by time).
    y:
        Target Series aligned with *X*.
    model_type:
        One of ``"xgboost"``, ``"lightgbm"``, ``"random_forest"``.
    task:
        ``"regression"`` or ``"classification"``.
    n_splits:
        Number of TimeSeriesSplit folds.
    params:
        Optional override for model hyper-parameters.

    Returns
    -------
    dict
        Keys: ``model`` (fitted on full data), ``cv_metrics`` (list of per-fold
        dicts), ``mean_metrics`` (averaged), ``oof_predictions`` (out-of-fold),
        ``feature_importance`` (Series).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    cv_metrics: list[dict[str, float]] = []
    oof_predictions = np.full(len(y_arr), np.nan)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        model = _build_model(model_type, task, params)
        model.fit(X_train, y_train)

        if task == "regression":
            preds = model.predict(X_val)
        else:
            preds = model.predict_proba(X_val)[:, 1]

        oof_predictions[val_idx] = preds
        fold_metrics = compute_metrics(y_val, preds, task)
        cv_metrics.append(fold_metrics)
        logger.info("Fold %d: %s", fold + 1, fold_metrics)

    # Aggregate CV metrics
    mean_metrics: dict[str, float] = {
        k: float(np.mean([m[k] for m in cv_metrics]))
        for k in cv_metrics[0].keys()
    }

    # Final model on full data
    final_model = _build_model(model_type, task, params)
    final_model.fit(X_arr, y_arr)

    # Feature importance
    feature_names = list(X.columns) if hasattr(X, "columns") else list(range(X_arr.shape[1]))
    fi: np.ndarray | None = getattr(final_model, "feature_importances_", None)
    if fi is not None:
        feature_importance = pd.Series(fi, index=feature_names).sort_values(ascending=False)
    else:
        feature_importance = pd.Series(dtype=float)

    return {
        "model": final_model,
        "cv_metrics": cv_metrics,
        "mean_metrics": mean_metrics,
        "oof_predictions": oof_predictions,
        "feature_importance": feature_importance,
    }


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    model_type: str = "xgboost",
) -> np.ndarray:
    """Compute SHAP values for a fitted model.

    Parameters
    ----------
    model:
        Fitted model.
    X:
        Feature DataFrame to explain.
    model_type:
        Used to choose the fastest SHAP explainer.

    Returns
    -------
    np.ndarray
        SHAP values array of shape ``(n_samples, n_features)``.
    """
    try:
        import shap  # type: ignore
    except ImportError as exc:
        raise ImportError("shap is required: pip install shap") from exc

    X_arr = np.asarray(X, dtype=float)

    if model_type in ("xgboost", "lightgbm", "random_forest"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_arr)
    else:
        explainer = shap.Explainer(model, X_arr)
        shap_values = explainer(X_arr).values

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return np.asarray(shap_values)
