"""
Scheduler Component for Noryangjin Crawler.

Handles:
- Date iteration (all dates from 2004-01-01 to present)
- Checkpoint management for resume capability
- Progress tracking
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator, Optional

from .models import CheckpointState

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages crawl checkpoints for resume capability.

    Stores state in a JSON file to allow resuming interrupted crawls.
    """

    def __init__(self, checkpoint_path: str = "data/checkpoint.json"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to checkpoint JSON file
        """
        self.path = Path(checkpoint_path)
        self.state = self._load()

    def _load(self) -> CheckpointState:
        """Load checkpoint state from file."""
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return CheckpointState(
                    last_updated=data.get("last_updated", ""),
                    status=data.get("status", "in_progress"),
                    last_completed_date=data.get("last_completed_date"),
                    total_days_crawled=data.get("statistics", {}).get("total_days_crawled", 0),
                    total_records=data.get("statistics", {}).get("total_records", 0),
                    failed_dates=data.get("statistics", {}).get("failed_dates", []),
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load checkpoint: {e}")

        return CheckpointState(
            last_updated=datetime.now().isoformat(),
            status="new",
        )

    def save(self) -> None:
        """Save checkpoint state to file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "last_updated": datetime.now().isoformat(),
            "status": self.state.status,
            "last_completed_date": self.state.last_completed_date,
            "statistics": {
                "total_days_crawled": self.state.total_days_crawled,
                "total_records": self.state.total_records,
                "failed_dates": self.state.failed_dates,
            },
        }

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_resume_date(self) -> Optional[datetime]:
        """
        Get the date to resume crawling from.

        Returns:
            Date to resume from, or None if starting fresh
        """
        if self.state.last_completed_date:
            try:
                last_date = datetime.strptime(
                    self.state.last_completed_date, "%Y.%m.%d"
                )
                return last_date + timedelta(days=1)
            except ValueError:
                pass
        return None

    def mark_date_completed(self, date: str, record_count: int) -> None:
        """
        Mark a date as completed.

        Args:
            date: Date in YYYY.MM.DD format
            record_count: Number of records crawled
        """
        self.state.last_completed_date = date
        self.state.total_days_crawled += 1
        self.state.total_records += record_count
        self.state.status = "in_progress"
        self.save()

    def mark_date_failed(self, date: str) -> None:
        """
        Mark a date as failed.

        Args:
            date: Date in YYYY.MM.DD format
        """
        if date not in self.state.failed_dates:
            self.state.failed_dates.append(date)
        self.save()

    def mark_completed(self) -> None:
        """Mark the entire crawl as completed."""
        self.state.status = "completed"
        self.save()


class Scheduler:
    """
    Date scheduler for crawling.

    Generates date sequences and manages iteration order.
    """

    # First available data date
    START_DATE = datetime(2004, 1, 2)

    def __init__(
        self,
        checkpoint_manager: Optional[CheckpointManager] = None,
        delay_between_days: float = 0.5,
    ):
        """
        Initialize scheduler.

        Args:
            checkpoint_manager: Optional checkpoint manager for resume
            delay_between_days: Delay between processing days
        """
        self.checkpoint = checkpoint_manager
        self.delay_between_days = delay_between_days

    def generate_dates(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Generate date strings for crawling.

        Args:
            start_date: Start date (YYYY.MM.DD), defaults to 2004.01.02 or checkpoint
            end_date: End date (YYYY.MM.DD), defaults to today

        Yields:
            Date strings in YYYY.MM.DD format
        """
        # Determine start date
        if start_date:
            start = datetime.strptime(start_date, "%Y.%m.%d")
        elif self.checkpoint:
            resume_date = self.checkpoint.get_resume_date()
            start = resume_date if resume_date else self.START_DATE
        else:
            start = self.START_DATE

        # Determine end date
        if end_date:
            end = datetime.strptime(end_date, "%Y.%m.%d")
        else:
            end = datetime.now()

        # Generate dates
        current = start
        while current <= end:
            yield current.strftime("%Y.%m.%d")
            current += timedelta(days=1)

    def count_days(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """
        Count total days in the date range.

        Args:
            start_date: Start date (YYYY.MM.DD)
            end_date: End date (YYYY.MM.DD)

        Returns:
            Number of days in range
        """
        if start_date:
            start = datetime.strptime(start_date, "%Y.%m.%d")
        elif self.checkpoint:
            resume_date = self.checkpoint.get_resume_date()
            start = resume_date if resume_date else self.START_DATE
        else:
            start = self.START_DATE

        if end_date:
            end = datetime.strptime(end_date, "%Y.%m.%d")
        else:
            end = datetime.now()

        return (end - start).days + 1


def format_date(dt: datetime) -> str:
    """
    Format datetime as YYYY.MM.DD for the API.

    Args:
        dt: Datetime object

    Returns:
        Formatted date string
    """
    return dt.strftime("%Y.%m.%d")


def parse_date(date_str: str) -> datetime:
    """
    Parse YYYY.MM.DD date string.

    Args:
        date_str: Date string in YYYY.MM.DD format

    Returns:
        Datetime object
    """
    return datetime.strptime(date_str, "%Y.%m.%d")
