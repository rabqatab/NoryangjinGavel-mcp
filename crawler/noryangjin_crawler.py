"""
Noryangjin Fish Market Crawler - Orchestrator

Main crawler that orchestrates all components:
- Scheduler: Date iteration and checkpoint management
- Fetcher: HTTP requests with rate limiting
- Parser: HTML parsing and data extraction
- Normalizer: Data cleaning and validation
- Writer: Data persistence

Target: https://www.susansijang.co.kr/nsis/miw/ko/info/miw3110
Method: POST
Data: Fish auction prices (경락시세) from Jan 2004 ~ Present
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from .fetcher import Fetcher
from .models import CrawlDayResult, CrawlStats, PriceRecord
from .normalizer import Normalizer
from .parser import HTMLParser
from .scheduler import CheckpointManager, Scheduler, format_date
from .writer import BaseWriter, MemoryWriter

logger = logging.getLogger(__name__)


class NoryangjinCrawler:
    """
    Orchestrator for Noryangjin Fish Market crawling.

    Coordinates all components to crawl fish auction prices.
    """

    def __init__(
        self,
        delay_between_requests: float = 1.5,
        delay_between_days: float = 0.5,
        max_retries: int = 3,
        timeout: int = 30,
        page_size: int = 10,
        checkpoint_path: Optional[str] = None,
        writer: Optional[BaseWriter] = None,
    ):
        """
        Initialize the crawler orchestrator.

        Args:
            delay_between_requests: Delay between HTTP requests (seconds)
            delay_between_days: Delay between processing days (seconds)
            max_retries: Maximum retry attempts for failed requests
            timeout: HTTP request timeout (seconds)
            page_size: Items per page
            checkpoint_path: Path for checkpoint file (enables resume)
            writer: Data writer instance (optional)
        """
        # Initialize components
        self.fetcher = Fetcher(
            delay_between_requests=delay_between_requests,
            max_retries=max_retries,
            timeout=timeout,
            page_size=page_size,
        )
        self.parser = HTMLParser()
        self.normalizer = Normalizer()

        # Checkpoint management (optional)
        self.checkpoint = (
            CheckpointManager(checkpoint_path)
            if checkpoint_path
            else None
        )
        self.scheduler = Scheduler(
            checkpoint_manager=self.checkpoint,
            delay_between_days=delay_between_days,
        )

        # Writer (optional)
        self.writer = writer

        # Configuration
        self.delay_between_days = delay_between_days
        self.page_size = page_size

    async def close(self) -> None:
        """Close all resources."""
        await self.fetcher.close()
        if self.writer:
            self.writer.close()

    async def crawl_date(self, date: str) -> CrawlDayResult:
        """
        Crawl all records for a specific date.

        Args:
            date: Date in YYYY.MM.DD format (e.g., "2004.01.02")

        Returns:
            CrawlDayResult with all records for the day
        """
        start_time = time.time()
        all_records: List[PriceRecord] = []

        # Fetch first page
        html, success = await self.fetcher.fetch_page(date, page=1)

        if not success:
            return CrawlDayResult(
                date=date,
                success=False,
                error="Failed to fetch first page",
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        # Parse first page
        parse_result = self.parser.parse(html, date)
        all_records.extend(parse_result.records)
        total_pages = parse_result.total_pages

        # If no records on first page, it's likely a holiday/Sunday
        if parse_result.is_empty or (not parse_result.records and total_pages <= 1):
            return CrawlDayResult(
                date=date,
                success=True,
                records=[],
                total_pages=1,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        # Fetch remaining pages
        for page in range(2, total_pages + 1):
            html, success = await self.fetcher.fetch_page(date, page)
            if success:
                records = self.parser.parse_table(html, date)
                all_records.extend(records)
            else:
                logger.warning(f"Failed to fetch page {page} for {date}")

        # Write records if writer is configured
        if self.writer and all_records:
            self.writer.write_records(all_records, date)

        # Update checkpoint if configured
        if self.checkpoint:
            self.checkpoint.mark_date_completed(date, len(all_records))

        return CrawlDayResult(
            date=date,
            success=True,
            records=all_records,
            total_pages=total_pages,
            elapsed_ms=(time.time() - start_time) * 1000,
        )

    async def crawl_date_range(
        self,
        start_date: str,
        end_date: str,
        progress_callback: Optional[Callable[[CrawlDayResult, CrawlStats], None]] = None,
    ) -> Tuple[List[CrawlDayResult], CrawlStats]:
        """
        Crawl a range of dates.

        Args:
            start_date: Start date in YYYY.MM.DD format
            end_date: End date in YYYY.MM.DD format
            progress_callback: Optional callback(day_result, stats) for progress updates

        Returns:
            Tuple of (results, stats)
        """
        results: List[CrawlDayResult] = []
        stats = CrawlStats()
        crawl_start = time.time()

        # Generate dates
        for date_str in self.scheduler.generate_dates(start_date, end_date):
            stats.total_days += 1

            result = await self.crawl_date(date_str)
            results.append(result)

            if result.success:
                if result.records:
                    stats.successful_days += 1
                    stats.total_records += len(result.records)
                    stats.total_pages += result.total_pages
                else:
                    stats.empty_days += 1
            else:
                stats.failed_days += 1
                if self.checkpoint:
                    self.checkpoint.mark_date_failed(date_str)

            if progress_callback:
                progress_callback(result, stats)

            # Delay between days
            await asyncio.sleep(self.delay_between_days)

        stats.elapsed_seconds = time.time() - crawl_start

        # Mark crawl as completed
        if self.checkpoint:
            self.checkpoint.mark_completed()

        return results, stats

    async def crawl_historical(
        self,
        progress_callback: Optional[Callable[[CrawlDayResult, CrawlStats], None]] = None,
    ) -> Tuple[List[CrawlDayResult], CrawlStats]:
        """
        Crawl all historical data from 2004 to present.

        Supports resume via checkpoint.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (results, stats)
        """
        start_date = "2004.01.02"  # First trading day
        end_date = format_date(datetime.now() - timedelta(days=1))

        return await self.crawl_date_range(
            start_date=start_date,
            end_date=end_date,
            progress_callback=progress_callback,
        )

    async def crawl_yesterday(self) -> CrawlDayResult:
        """
        Crawl yesterday's data (for daily cron jobs).

        Returns:
            CrawlDayResult for yesterday
        """
        yesterday = datetime.now() - timedelta(days=1)
        date_str = format_date(yesterday)
        return await self.crawl_date(date_str)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# ============================================================================
# Utility Functions (for backwards compatibility)
# ============================================================================

def records_to_dicts(records: List[PriceRecord]) -> List[Dict[str, Any]]:
    """Convert PriceRecord list to list of dicts for JSON/DB storage."""
    return [r.to_dict() for r in records]
