#!/usr/bin/env python3
"""
Noryangjin Fish Market Crawler - Unified CLI

Supports multiple crawling modes:
- historical: Full crawl from 2004 to present (one-time, with resume)
- daily: Crawl yesterday's data (for cron jobs)
- range: Crawl a specific date range
- date: Crawl a single specific date

Usage:
    # Historical crawl (initial)
    uv run python scripts/crawler/run_crawler.py historical

    # Daily crawl (for cron)
    uv run python scripts/crawler/run_crawler.py daily

    # Date range crawl
    uv run python scripts/crawler/run_crawler.py range --start 2024.01.01 --end 2024.01.31

    # Single date crawl
    uv run python scripts/crawler/run_crawler.py date --date 2024.01.15

Common options:
    --output-dir    Output directory (default: data/parquet/prices)
    --format        Output format: parquet, json, csv, both, none (default: parquet)
    --delay         Delay between requests (default: 1.5s)
    --log-dir       Log directory (default: logs)
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
    CrawlDayResult,
    CrawlStats,
    ParquetWriter,
    JSONWriter,
    CSVWriter,
    CompositeWriter,
    format_date,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, prefix: str) -> Path:
    """Setup file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{prefix}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    return log_file


def create_writer(output_dir: str, output_format: str):
    """Create writer based on format."""
    if output_format == "parquet":
        return ParquetWriter(output_dir=output_dir)
    elif output_format == "json":
        return JSONWriter(output_dir=output_dir)
    elif output_format == "csv":
        csv_path = Path(output_dir) / "fish_prices.csv"
        return CSVWriter(output_path=str(csv_path))
    elif output_format == "both":
        # Both Parquet and JSON
        parquet_writer = ParquetWriter(output_dir=output_dir)
        json_writer = JSONWriter(output_dir=output_dir + "_json")
        return CompositeWriter([parquet_writer, json_writer])
    return None


def make_progress_callback(estimated_total: int = 100):
    """Create a progress callback."""
    def callback(result: CrawlDayResult, stats: CrawlStats) -> None:
        status = "OK" if result.success else "FAIL"
        records = len(result.records) if result.records else 0
        percent = (stats.total_days / estimated_total) * 100 if estimated_total > 0 else 0

        if stats.total_days % 10 == 0 or not result.success:
            logger.info(
                f"[{status}] {result.date}: {records} records | "
                f"Progress: {stats.total_days}/{estimated_total} (~{percent:.1f}%) | "
                f"Total: {stats.total_records:,} records"
            )
    return callback


def log_stats(stats: CrawlStats, elapsed: timedelta) -> None:
    """Log crawl statistics."""
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


# ============================================================================
# Commands
# ============================================================================

async def cmd_historical(args) -> bool:
    """Run historical crawl."""
    logger.info("=" * 60)
    logger.info("NORYANGJIN FISH MARKET - HISTORICAL CRAWL")
    logger.info("=" * 60)
    logger.info(f"Period: 2004-01-02 ~ Today")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info("=" * 60)

    writer = create_writer(args.output_dir, args.format)
    crawler = NoryangjinCrawler(
        delay_between_requests=args.delay,
        delay_between_days=0.5,
        checkpoint_path=args.checkpoint,
        writer=writer,
    )

    start_time = datetime.now()

    try:
        async with crawler:
            results, stats = await crawler.crawl_historical(
                progress_callback=make_progress_callback(7700)
            )

        log_stats(stats, datetime.now() - start_time)
        return stats.failed_days == 0

    except KeyboardInterrupt:
        logger.warning("Crawl interrupted. Progress saved to checkpoint.")
        return False


async def cmd_daily(args) -> bool:
    """Run daily crawl."""
    if args.date:
        target_date = args.date
    else:
        yesterday = datetime.now() - timedelta(days=1)
        target_date = format_date(yesterday)

    logger.info("=" * 60)
    logger.info("NORYANGJIN FISH MARKET - DAILY CRAWL")
    logger.info("=" * 60)
    logger.info(f"Target Date: {target_date}")
    logger.info("=" * 60)

    writer = create_writer(args.output_dir, args.format)
    crawler = NoryangjinCrawler(
        delay_between_requests=args.delay,
        writer=writer,
    )

    async with crawler:
        result = await crawler.crawl_date(target_date)

    if result.success:
        if result.records:
            logger.info(f"SUCCESS: {len(result.records)} records from {result.total_pages} pages")

            # Stats
            species_set = set(r.species for r in result.records)
            state_counts = {}
            for r in result.records:
                if r.state:
                    state_counts[r.state] = state_counts.get(r.state, 0) + 1

            logger.info(f"Unique species: {len(species_set)}")
            logger.info(f"State distribution: {state_counts}")
        else:
            logger.info(f"No records (non-trading day)")
        return True
    else:
        logger.error(f"FAILED: {result.error}")
        return False


async def cmd_range(args) -> bool:
    """Run date range crawl."""
    logger.info("=" * 60)
    logger.info("NORYANGJIN FISH MARKET - RANGE CRAWL")
    logger.info("=" * 60)
    logger.info(f"Period: {args.start} ~ {args.end}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)

    writer = create_writer(args.output_dir, args.format)
    crawler = NoryangjinCrawler(
        delay_between_requests=args.delay,
        delay_between_days=0.5,
        writer=writer,
    )

    # Calculate estimated days
    start_dt = datetime.strptime(args.start, "%Y.%m.%d")
    end_dt = datetime.strptime(args.end, "%Y.%m.%d")
    estimated_days = (end_dt - start_dt).days + 1

    start_time = datetime.now()

    async with crawler:
        results, stats = await crawler.crawl_date_range(
            start_date=args.start,
            end_date=args.end,
            progress_callback=make_progress_callback(estimated_days),
        )

    log_stats(stats, datetime.now() - start_time)
    return stats.failed_days == 0


async def cmd_date(args) -> bool:
    """Run single date crawl."""
    logger.info("=" * 60)
    logger.info("NORYANGJIN FISH MARKET - SINGLE DATE CRAWL")
    logger.info("=" * 60)
    logger.info(f"Target Date: {args.date}")
    logger.info("=" * 60)

    writer = create_writer(args.output_dir, args.format)
    crawler = NoryangjinCrawler(
        delay_between_requests=args.delay,
        writer=writer,
    )

    async with crawler:
        result = await crawler.crawl_date(args.date)

    if result.success:
        logger.info(f"SUCCESS: {len(result.records)} records from {result.total_pages} pages")
        logger.info(f"Elapsed: {result.elapsed_ms:.0f}ms")

        if result.records:
            # Show sample
            logger.info("Sample records:")
            for r in result.records[:5]:
                logger.info(f"  {r.species_raw}: {r.price_avg:,} KRW ({r.origin})")
        return True
    else:
        logger.error(f"FAILED: {result.error}")
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Noryangjin Fish Market Crawler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s historical                    # Full historical crawl (2004~today)
  %(prog)s daily                         # Crawl yesterday's data
  %(prog)s range --start 2024.01.01 --end 2024.01.31
  %(prog)s date --date 2024.01.15
        """,
    )

    # Common arguments
    parser.add_argument(
        "--output-dir",
        default="data/parquet/prices",
        help="Output directory (default: data/parquet/prices)",
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

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Crawl mode")

    # Historical
    hist_parser = subparsers.add_parser("historical", help="Full historical crawl")
    hist_parser.add_argument(
        "--checkpoint",
        default="data/checkpoint.json",
        help="Checkpoint file (default: data/checkpoint.json)",
    )

    # Daily
    daily_parser = subparsers.add_parser("daily", help="Daily crawl (yesterday)")
    daily_parser.add_argument(
        "--date",
        default=None,
        help="Specific date (default: yesterday)",
    )

    # Range
    range_parser = subparsers.add_parser("range", help="Date range crawl")
    range_parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY.MM.DD)",
    )
    range_parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYY.MM.DD)",
    )

    # Single date
    date_parser = subparsers.add_parser("date", help="Single date crawl")
    date_parser.add_argument(
        "--date",
        required=True,
        help="Target date (YYYY.MM.DD)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging
    log_file = setup_logging(Path(args.log_dir), f"crawl_{args.command}")
    logger.info(f"Log file: {log_file}")

    # Run command
    if args.command == "historical":
        success = asyncio.run(cmd_historical(args))
    elif args.command == "daily":
        success = asyncio.run(cmd_daily(args))
    elif args.command == "range":
        success = asyncio.run(cmd_range(args))
    elif args.command == "date":
        success = asyncio.run(cmd_date(args))
    else:
        parser.print_help()
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
