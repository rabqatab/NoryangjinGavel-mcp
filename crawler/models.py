"""
Data Models for Noryangjin Fish Market Crawler.

Contains all dataclasses used across crawler components.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PriceRecord:
    """Single fish price record from miw3110 endpoint."""
    species_raw: str      # e.g., "(활)방어" - raw species with state prefix
    species: str          # e.g., "방어" - species name only
    state: Optional[str]  # e.g., "활" (live), "선" (fresh), "냉" (frozen)
    origin: str           # e.g., "일본", "부산"
    spec: str             # e.g., "2미", "대", "M1"
    packaging: str        # e.g., "kg", "S/P", "box"
    quantity: float       # Weight or count
    price_high: int       # Highest bid price (KRW)
    price_low: int        # Lowest bid price (KRW)
    price_avg: int        # Average price (KRW)
    trade_date: str       # Date in YYYY.MM.DD format

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/DB storage."""
        return {
            "species_raw": self.species_raw,
            "species": self.species,
            "state": self.state,
            "origin": self.origin,
            "spec": self.spec,
            "packaging": self.packaging,
            "quantity": self.quantity,
            "price_high": self.price_high,
            "price_low": self.price_low,
            "price_avg": self.price_avg,
            "trade_date": self.trade_date,
        }


@dataclass
class CrawlDayResult:
    """Result of crawling a single day."""
    date: str
    success: bool
    records: List[PriceRecord] = field(default_factory=list)
    total_pages: int = 0
    elapsed_ms: float = 0
    error: Optional[str] = None


@dataclass
class CrawlStats:
    """Statistics for a crawl session."""
    total_days: int = 0
    successful_days: int = 0
    failed_days: int = 0
    empty_days: int = 0  # Holidays/Sundays
    total_records: int = 0
    total_pages: int = 0
    elapsed_seconds: float = 0


@dataclass
class CheckpointState:
    """Checkpoint state for resume capability."""
    last_updated: str
    status: str  # "in_progress", "completed", "failed"
    last_completed_date: Optional[str] = None
    total_days_crawled: int = 0
    total_records: int = 0
    failed_dates: List[str] = field(default_factory=list)
