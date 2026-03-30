"""
Noryangjin Fish Market Crawler - Modular Architecture

Crawls fish auction prices (경락시세) from:
https://www.susansijang.co.kr/nsis/miw/ko/info/miw3110

Components:
- NoryangjinCrawler: Main orchestrator
- Fetcher: HTTP requests with rate limiting
- HTMLParser: HTML parsing and data extraction
- Normalizer: Data cleaning and validation
- Scheduler: Date iteration and checkpoint management
- Writers: Data persistence (Parquet, JSON, CSV, Memory)

Usage:
    from crawler import NoryangjinCrawler

    async with NoryangjinCrawler() as crawler:
        # Crawl a single date
        result = await crawler.crawl_date("2024.01.02")
        print(f"Found {len(result.records)} records")

        # Crawl a date range
        results, stats = await crawler.crawl_date_range(
            "2024.01.01", "2024.01.31"
        )
        print(f"Total: {stats.total_records} records")

Advanced Usage (with checkpoint and Parquet writer):
    from crawler import NoryangjinCrawler, ParquetWriter

    writer = ParquetWriter(output_dir="data/parquet/prices")
    async with NoryangjinCrawler(
        checkpoint_path="data/checkpoint.json",
        writer=writer
    ) as crawler:
        results, stats = await crawler.crawl_historical()
"""

# Models
from .models import (
    CheckpointState,
    CrawlDayResult,
    CrawlStats,
    PriceRecord,
)

# Components
from .fetcher import Fetcher, RateLimiter
from .parser import HTMLParser, ParseResult
from .normalizer import Normalizer
from .scheduler import (
    CheckpointManager,
    Scheduler,
    format_date,
    parse_date,
)
from .writer import (
    BaseWriter,
    CompositeWriter,
    CSVWriter,
    JSONWriter,
    MemoryWriter,
    ParquetWriter,
)

# Main Orchestrator
from .noryangjin_crawler import (
    NoryangjinCrawler,
    records_to_dicts,
)

__all__ = [
    # Main Orchestrator
    "NoryangjinCrawler",
    # Models
    "PriceRecord",
    "CrawlDayResult",
    "CrawlStats",
    "CheckpointState",
    "ParseResult",
    # Components
    "Fetcher",
    "RateLimiter",
    "HTMLParser",
    "Normalizer",
    "Scheduler",
    "CheckpointManager",
    # Writers
    "BaseWriter",
    "ParquetWriter",
    "JSONWriter",
    "CSVWriter",
    "MemoryWriter",
    "CompositeWriter",
    # Utilities
    "format_date",
    "parse_date",
    "records_to_dicts",
]

__version__ = "0.2.0"
