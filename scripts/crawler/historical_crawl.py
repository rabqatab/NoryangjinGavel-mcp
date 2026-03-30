#!/usr/bin/env python3
"""
Historical Crawl Script

One-time initial crawl of all historical data from 2004-01-02 to present.
Supports resume via checkpoint file.

Usage:
    uv run python scripts/crawler/historical_crawl.py
    uv run python scripts/crawler/historical_crawl.py --output-dir data/raw
    uv run python scripts/crawler/historical_crawl.py --resume  # Resume from checkpoint

Estimated time: 4-8 hours for full historical data (~21 years)
"""
import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from crawler import (
    NoryangjinCrawler,
    CrawlDayResult,
    CrawlStats,
    ParquetWriter,
    JSONWriter,
    CSVWriter,
    CompositeWriter,
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
    log_file = log_dir / f"historical_crawl_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    return log_file


def progress_callback(result: CrawlDayResult, stats: CrawlStats) -> None:
    """Log progress for each day crawled."""
    status = "OK" if result.success else "FAIL"
    records = len(result.records) if result.records else 0

    # Calculate progress percentage
    # Rough estimate: ~7700 days from 2004 to present
    estimated_total = 7700
    percent = (stats.total_days / estimated_total) * 100

    if stats.total_days % 10 == 0 or not result.success:
        logger.info(
            f"[{status}] {result.date}: {records} records | "
            f"Progress: {stats.total_days} days (~{percent:.1f}%) | "
            f"Total: {stats.total_records:,} records"
        )


async def run_historical_crawl(
    output_dir: str,
    checkpoint_path: str,
    output_format: str,
    delay: float,
) -> bool:
    """Run the historical crawl."""
    logger.info("=" * 60)
    logger.info("NORYANGJIN FISH MARKET - HISTORICAL CRAWL")
    logger.info("=" * 60)
    logger.info(f"Target: https://www.susansijang.co.kr")
    logger.info(f"Period: 2004-01-02 ~ Today")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Format: {output_format}")
    logger.info(f"Delay: {delay}s between requests")
    logger.info("=" * 60)

    # Setup writer based on format
    if output_format == "parquet":
        writer = ParquetWriter(output_dir=output_dir)
    elif output_format == "json":
        writer = JSONWriter(output_dir=output_dir)
    elif output_format == "csv":
        csv_path = Path(output_dir) / "fish_prices.csv"
        writer = CSVWriter(output_path=str(csv_path))
    elif output_format == "both":
        # Both Parquet and JSON
        parquet_writer = ParquetWriter(output_dir=output_dir)
        json_writer = JSONWriter(output_dir=output_dir + "_json")
        writer = CompositeWriter([parquet_writer, json_writer])
    else:
        writer = None

    # Create crawler with checkpoint
    crawler = NoryangjinCrawler(
        delay_between_requests=delay,
        delay_between_days=0.5,
        checkpoint_path=checkpoint_path,
        writer=writer,
    )

    start_time = datetime.now()
    logger.info(f"Started at: {start_time.isoformat()}")

    try:
        async with crawler:
            results, stats = await crawler.crawl_historical(
                progress_callback=progress_callback
            )

        end_time = datetime.now()
        elapsed = end_time - start_time

        logger.info("=" * 60)
        logger.info("CRAWL COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total Days: {stats.total_days}")
        logger.info(f"Successful Days: {stats.successful_days}")
        logger.info(f"Empty Days (holidays): {stats.empty_days}")
        logger.info(f"Failed Days: {stats.failed_days}")
        logger.info(f"Total Records: {stats.total_records:,}")
        logger.info(f"Total Pages: {stats.total_pages:,}")
        logger.info(f"Elapsed Time: {elapsed}")
        logger.info("=" * 60)

        return stats.failed_days == 0

    except KeyboardInterrupt:
        logger.warning("Crawl interrupted by user. Progress saved to checkpoint.")
        return False
    except Exception as e:
        logger.error(f"Crawl failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Historical crawl of Noryangjin Fish Market data"
    )
    parser.add_argument(
        "--output-dir",
        default="data/parquet/prices",
        help="Output directory for crawled data (default: data/parquet/prices)",
    )
    parser.add_argument(
        "--checkpoint",
        default="data/checkpoint.json",
        help="Checkpoint file path (default: data/checkpoint.json)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "json", "csv", "both", "none"],
        default="parquet",
        help="Output format (default: parquet)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Delay between requests in seconds (default: 1.5)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Log directory (default: logs)",
    )

    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging(Path(args.log_dir))
    logger.info(f"Log file: {log_file}")

    # Run crawl
    success = asyncio.run(
        run_historical_crawl(
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            output_format=args.format,
            delay=args.delay,
        )
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
