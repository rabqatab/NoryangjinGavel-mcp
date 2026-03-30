#!/usr/bin/env python3
"""
Daily Crawl Script

Crawls yesterday's fish auction data. Designed for daily cron jobs.

Usage:
    uv run python scripts/crawler/daily_crawl.py
    uv run python scripts/crawler/daily_crawl.py --date 2024.01.15  # Specific date

Cron setup (run daily at 12:00 PM KST):
    0 12 * * * cd /path/to/project && uv run python scripts/crawler/daily_crawl.py

Estimated time: ~2 minutes per day
"""
import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from crawler import (
    NoryangjinCrawler,
    ParquetWriter,
    JSONWriter,
    CSVWriter,
    format_date,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> Path:
    """Setup file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"daily_crawl_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    return log_file


async def run_daily_crawl(
    target_date: str,
    output_dir: str,
    output_format: str,
) -> bool:
    """Run the daily crawl for a specific date."""
    logger.info("=" * 60)
    logger.info("NORYANGJIN FISH MARKET - DAILY CRAWL")
    logger.info("=" * 60)
    logger.info(f"Target Date: {target_date}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Format: {output_format}")
    logger.info("=" * 60)

    # Setup writer based on format
    if output_format == "parquet":
        writer = ParquetWriter(output_dir=output_dir)
    elif output_format == "json":
        writer = JSONWriter(output_dir=output_dir)
    elif output_format == "csv":
        csv_path = Path(output_dir) / "fish_prices.csv"
        writer = CSVWriter(output_path=str(csv_path))
    else:
        writer = None

    # Create crawler (no checkpoint needed for daily)
    crawler = NoryangjinCrawler(
        delay_between_requests=1.5,
        writer=writer,
    )

    start_time = datetime.now()

    try:
        async with crawler:
            result = await crawler.crawl_date(target_date)

        end_time = datetime.now()
        elapsed = end_time - start_time

        if result.success:
            if result.records:
                logger.info(f"SUCCESS: {len(result.records)} records crawled")
                logger.info(f"Pages: {result.total_pages}")
                logger.info(f"Elapsed: {result.elapsed_ms:.0f}ms")

                # Log sample data
                if result.records:
                    species_set = set(r.species for r in result.records)
                    logger.info(f"Unique species: {len(species_set)}")

                    # State distribution
                    state_counts = {}
                    for r in result.records:
                        if r.state:
                            state_counts[r.state] = state_counts.get(r.state, 0) + 1
                    logger.info(f"State distribution: {state_counts}")
            else:
                logger.info(f"No records for {target_date} (non-trading day)")

            logger.info("=" * 60)
            return True
        else:
            logger.error(f"FAILED: {result.error}")
            logger.info("=" * 60)
            return False

    except Exception as e:
        logger.error(f"Crawl failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Daily crawl of Noryangjin Fish Market data"
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Target date in YYYY.MM.DD format (default: yesterday)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/parquet/prices",
        help="Output directory for crawled data (default: data/parquet/prices)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "json", "csv", "none"],
        default="parquet",
        help="Output format (default: parquet)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Log directory (default: logs)",
    )

    args = parser.parse_args()

    # Determine target date
    if args.date:
        target_date = args.date
    else:
        yesterday = datetime.now() - timedelta(days=1)
        target_date = format_date(yesterday)

    # Setup logging
    log_file = setup_logging(Path(args.log_dir))
    logger.info(f"Log file: {log_file}")

    # Run crawl
    success = asyncio.run(
        run_daily_crawl(
            target_date=target_date,
            output_dir=args.output_dir,
            output_format=args.format,
        )
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
