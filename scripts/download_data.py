"""
CLI script to download and cache market data.

Usage
-----
    python scripts/download_data.py --api-key $FRED_API_KEY

Options
-------
    --start-date    Start date in YYYY-MM-DD format (default: 2000-01-01)
    --end-date      End date in YYYY-MM-DD format (default: today)
    --api-key       FRED API key (overrides FRED_API_KEY env var)
    --output-dir    Directory to store the Parquet cache (default: data/)
    --force         Force re-download even if cache exists
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow project-root imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    from config.settings import DEFAULT_END_DATE, DEFAULT_START_DATE, FRED_API_KEY  # type: ignore

    parser = argparse.ArgumentParser(
        description="Download and cache market data for the Credit Spread Analyzer."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="End date YYYY-MM-DD")
    parser.add_argument("--api-key", default=FRED_API_KEY, help="FRED API key")
    parser.add_argument("--output-dir", default="data", help="Output directory for Parquet cache")
    parser.add_argument(
        "--force", action="store_true", help="Force re-download ignoring cache"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the download script."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading data: %s → %s", args.start_date, args.end_date)
    logger.info("Output directory: %s", output_dir.resolve())

    from src.data.fetcher import fetch_all_data  # type: ignore

    df = fetch_all_data(
        start_date=args.start_date,
        end_date=args.end_date,
        api_key=args.api_key,
        cache_dir=output_dir,
        force_refresh=args.force,
    )

    if df.empty:
        logger.error("No data fetched – check API key and network connectivity.")
        sys.exit(1)

    logger.info(
        "✅ Data saved successfully. Shape: %d rows × %d columns.", *df.shape
    )
    logger.info("Columns: %s", list(df.columns))
    logger.info("Date range: %s → %s", df.index.min(), df.index.max())


if __name__ == "__main__":
    main()
