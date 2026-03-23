"""
CLI script to train ML models on cached market data.

Usage
-----
    python scripts/train_models.py --data-path data/market_data_2000-01-01_2024-12-31.parquet

Options
-------
    --data-path       Path to Parquet data file (required)
    --model-type      xgboost | lightgbm | random_forest  (default: xgboost)
    --target-horizon  Forward horizon in days (default: 5)
    --output-dir      Directory to save model + metrics (default: models/saved)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an ML model on pre-fetched credit-spread data."
    )
    parser.add_argument("--data-path", required=True, help="Path to the Parquet data file")
    parser.add_argument(
        "--model-type",
        default="xgboost",
        choices=["xgboost", "lightgbm", "random_forest"],
        help="Model to train",
    )
    parser.add_argument(
        "--target-horizon", type=int, default=5, help="Forward return horizon in days"
    )
    parser.add_argument(
        "--output-dir", default="models/saved", help="Directory to save outputs"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the training script."""
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from %s …", data_path)
    import pandas as pd  # type: ignore

    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows × %d columns.", *df.shape)

    logger.info("Building feature matrix (horizon=%d) …", args.target_horizon)
    from src.features.engineering import build_feature_matrix  # type: ignore

    X, y = build_feature_matrix(df, target_horizon=args.target_horizon)

    target_col = next((c for c in y.columns if "return" in c), y.columns[0])
    logger.info("Target column: %s  Shape: X=%s y=%s", target_col, X.shape, y[target_col].shape)

    logger.info("Training %s model with TimeSeriesSplit CV …", args.model_type)
    from src.models.ml_models import train_and_evaluate  # type: ignore

    result = train_and_evaluate(
        X, y[target_col], model_type=args.model_type, task="regression", n_splits=5
    )

    mean_m = result["mean_metrics"]
    logger.info("CV Mean Metrics: %s", {k: round(v, 4) for k, v in mean_m.items()})

    # Save model
    model_path = output_dir / f"{args.model_type}_horizon{args.target_horizon}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(result["model"], fh)
    logger.info("Model saved to %s", model_path)

    # Save metrics
    metrics_path = output_dir / f"{args.model_type}_horizon{args.target_horizon}_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(mean_m, fh, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Save feature importance
    fi_path = output_dir / f"{args.model_type}_horizon{args.target_horizon}_feature_importance.csv"
    result["feature_importance"].to_csv(fi_path, header=["importance"])
    logger.info("Feature importance saved to %s", fi_path)

    logger.info("✅ Training complete.")


if __name__ == "__main__":
    main()
