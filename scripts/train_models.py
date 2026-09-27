"""
CLI script to train a forecasting model on cached market data.

Usage
-----
    python scripts/train_models.py --data-path data/market_data_1990-01-01_2026-09-15.parquet
    python scripts/train_models.py --data-path ... --target hyg

Options
-------
    --data-path       Path to Parquet data file (required)
    --target          baa: change in the Moody's Baa – 10y spread (bps, default)
                      hyg: HYG vs IEI excess return (bps), next-day execution
    --model-type      composite | ridge | ensemble | xgboost | lightgbm | random_forest
                      (default: the recommended model for the target / horizon)
    --spread-col      Spread column to model / build features from (default: automatic)
    --target-horizon  Forward horizon in trading days (default: config TARGET_HORIZON)
    --task            regression (size of the move) | classification (up / down)
    --output-dir      Directory to save model + metrics (default: models/saved)

Outputs the fitted model, purged-CV and holdout metrics, feature importance,
and the forecast for the most recent date in the data.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (  # noqa: E402
    HOLDOUT_START,
    MODELS_DIR,
    RECOMMENDED_MODEL,
    RECOMMENDED_MODEL_HY,
    TARGET_HORIZON,
)
from src.models.ml_models import DEFAULT_SCALE_COL, MODEL_TYPES, RETURN_COMPOSITE_WEIGHTS, TREE_MODELS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a credit forecasting model on pre-fetched data.")
    parser.add_argument("--data-path", required=True, help="Path to the Parquet data file")
    parser.add_argument("--target", default="baa", choices=["baa", "hyg"], help="What to forecast")
    parser.add_argument("--spread-col", default=None, help="Spread column to model / build features from")
    parser.add_argument("--model-type", default=None, choices=MODEL_TYPES, help="Model to train")
    parser.add_argument(
        "--target-horizon", type=int, default=TARGET_HORIZON, help="Forecast horizon in trading days"
    )
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    parser.add_argument("--output-dir", default=str(MODELS_DIR), help="Directory to save outputs")
    return parser.parse_args()


def main() -> None:
    """Main entry point for the training script."""
    args = parse_args()
    h = args.target_horizon
    if args.model_type:
        model_type = args.model_type
    elif args.target == "hyg":
        model_type = RECOMMENDED_MODEL_HY
    else:
        model_type = RECOMMENDED_MODEL.get(h, "composite")
    if args.task == "classification" and model_type == "ensemble":
        logger.error("The ensemble model supports regression only; pick another --model-type.")
        sys.exit(1)

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import pandas as pd

    from src.features.engineering import build_feature_matrix, build_hy_feature_matrix
    from src.models.ml_models import compute_metrics, train_and_evaluate, walk_forward_predict

    logger.info("Loading data from %s …", data_path)
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows × %d columns.", *df.shape)

    params = None
    if args.target == "hyg":
        def builder(df, **kwargs):
            return build_hy_feature_matrix(df, spread_col=args.spread_col, **kwargs)
        target_col = f"target_{h}d_lag1_xs_return" if args.task == "regression" else f"target_{h}d_lag1_xs_up"
        if model_type == "composite":
            params = {"weights": RETURN_COMPOSITE_WEIGHTS}
    else:
        def builder(df, **kwargs):
            return build_feature_matrix(df, target_col=args.spread_col, **kwargs)

        target_col = f"target_{h}d_change" if args.task == "regression" else f"target_{h}d_up"

    X, y = builder(df, target_horizon=h)
    target = y[target_col]
    scale_col = DEFAULT_SCALE_COL if (args.task == "regression" and model_type in TREE_MODELS) else None
    logger.info("Target: %s | model: %s | X=%s", target_col, model_type, X.shape)

    # 1) Purged expanding-window CV over the full sample (gap inferred from the target name).
    result = train_and_evaluate(
        X, target, model_type=model_type, task=args.task, n_splits=5, params=params, scale_col=scale_col
    )
    cv_metrics = result["mean_metrics"]
    logger.info("Purged CV mean metrics: %s", {k: round(v, 4) for k, v in cv_metrics.items()})

    # 2) Walk-forward (yearly refit) on the holdout period.
    holdout_pred = walk_forward_predict(
        X, target, model_type=model_type, task=args.task, start=HOLDOUT_START,
        params=params, scale_col=scale_col,
    )
    ok = holdout_pred.notna()
    holdout_metrics = compute_metrics(target[ok].values, holdout_pred[ok].values, args.task, horizon=h)
    logger.info("Holdout (%s →) metrics: %s", HOLDOUT_START, {k: round(v, 4) for k, v in holdout_metrics.items()})

    # 3) Live forecast for the latest available date.
    X_live, _ = builder(df, target_horizon=h, dropna=False)
    X_live = X_live[X.columns].dropna()
    latest = X_live.iloc[[-1]]
    model = result["model"]
    if args.task == "regression":
        forecast = float(model.predict(latest)[0])
    else:
        forecast = float(model.predict_proba(latest)[0, 1])
    logger.info("Forecast for %s (+%dd): %.3f", latest.index[0].date(), h, forecast)

    stem = f"{args.target}_{model_type}_{args.task}_h{h}"
    with open(output_dir / f"{stem}.pkl", "wb") as fh:
        pickle.dump({"model": model, "features": list(X.columns), "target": target_col}, fh)

    summary = {
        "target": args.target,
        "model_type": model_type,
        "task": args.task,
        "horizon": h,
        "target_column": target_col,
        "train_rows": int(len(X)),
        "cv_metrics": cv_metrics,
        "holdout_start": HOLDOUT_START,
        "holdout_metrics": holdout_metrics,
        "latest_date": str(latest.index[0].date()),
        "latest_forecast": forecast,
    }

    def _clean(v: object) -> object:
        if isinstance(v, dict):
            return {k: _clean(x) for k, x in v.items()}
        return None if isinstance(v, float) and not np.isfinite(v) else v

    with open(output_dir / f"{stem}_metrics.json", "w") as fh:
        json.dump(_clean(summary), fh, indent=2)

    result["feature_importance"].to_csv(output_dir / f"{stem}_feature_importance.csv", header=["importance"])
    logger.info("Saved model, metrics and feature importance to %s", output_dir)


if __name__ == "__main__":
    main()
