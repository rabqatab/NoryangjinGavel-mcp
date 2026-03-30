"""
HTML Parser Component for Noryangjin Crawler.

Handles:
- Table parsing (8 columns)
- Pagination detection
- Empty result detection
"""
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .models import PriceRecord

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of parsing an HTML page."""
    records: List[PriceRecord]
    total_pages: int
    is_empty: bool


class HTMLParser:
    """
    Parser for Noryangjin Fish Market HTML pages.

    Extracts price data from the miw3110 endpoint HTML tables.
    """

    # Regex patterns
    NO_DATA_MESSAGE = "조회된 경락시세가 없습니다"
    TABLE_PATTERN = re.compile(r'<tbody[^>]*>(.*?)</tbody>', re.DOTALL | re.IGNORECASE)
    ROW_PATTERN = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
    CELL_PATTERN = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL | re.IGNORECASE)
    LAST_PAGE_PATTERN = re.compile(r"class=[\"']arr\s+last[\"'][^>]*onclick=[\"']fnList\((\d+)\)")
    ALL_PAGES_PATTERN = re.compile(r'fnList\((\d+)\)')
    STATE_PATTERN = re.compile(r'^\(([^)]+)\)(.+)$')

    def __init__(self):
        """Initialize parser."""
        pass

    @staticmethod
    def _clean_cell(cell: str) -> str:
        """
        Clean HTML cell content.

        Args:
            cell: Raw HTML cell content

        Returns:
            Cleaned text content
        """
        # Remove HTML tags
        cleaned = re.sub(r'<[^>]+>', '', cell)
        # Unescape HTML entities
        cleaned = cleaned.replace('&nbsp;', ' ')
        return cleaned.strip()

    @staticmethod
    def _parse_number(text: str) -> float:
        """
        Parse Korean number format (with commas).

        Args:
            text: Number string possibly with commas

        Returns:
            Parsed float value
        """
        cleaned = re.sub(r'[^\d.]', '', text)
        return float(cleaned) if cleaned else 0

    def extract_state(self, species_raw: str) -> Tuple[Optional[str], str]:
        """
        Extract state prefix from species name.

        Examples:
            "(냉)고등어" -> ("냉", "고등어")
            "(활)방어" -> ("활", "방어")
            "고등어" -> (None, "고등어")

        Args:
            species_raw: Raw species name possibly with state prefix

        Returns:
            Tuple of (state, species_name)
        """
        match = self.STATE_PATTERN.match(species_raw)
        if match:
            return match.group(1), match.group(2)
        return None, species_raw

    def get_total_pages(self, html: str) -> int:
        """
        Extract total page count from pagination HTML.

        Args:
            html: Full HTML page content

        Returns:
            Total number of pages (minimum 1)
        """
        # Try to find the "last" button first
        match = self.LAST_PAGE_PATTERN.search(html)
        if match:
            return int(match.group(1))

        # Fallback: find all fnList(N) and get max
        all_pages = self.ALL_PAGES_PATTERN.findall(html)
        if all_pages:
            return max(int(p) for p in all_pages)

        return 1

    def is_empty_result(self, html: str) -> bool:
        """
        Check if the page contains no data message.

        Args:
            html: HTML page content

        Returns:
            True if no data found
        """
        return self.NO_DATA_MESSAGE in html

    def parse_table(self, html: str, date: str) -> List[PriceRecord]:
        """
        Parse price data from miw3110 HTML page.

        Args:
            html: HTML page content
            date: Trade date in YYYY.MM.DD format

        Returns:
            List of PriceRecord objects
        """
        records = []

        # Check for "no data" message
        if self.is_empty_result(html):
            return records

        # Find table body
        table_match = self.TABLE_PATTERN.search(html)
        if not table_match:
            return records

        tbody = table_match.group(1)
        rows = self.ROW_PATTERN.findall(tbody)

        for row in rows:
            # Skip rows with "no data" class
            if 'no-data' in row or self.NO_DATA_MESSAGE in row:
                continue

            # Extract cell values
            cells = self.CELL_PATTERN.findall(row)

            if len(cells) >= 8:
                species_raw = self._clean_cell(cells[0])
                if not species_raw:
                    continue

                state, species = self.extract_state(species_raw)

                try:
                    record = PriceRecord(
                        species_raw=species_raw,
                        species=species,
                        state=state,
                        origin=self._clean_cell(cells[1]),
                        spec=self._clean_cell(cells[2]),
                        packaging=self._clean_cell(cells[3]),
                        quantity=self._parse_number(self._clean_cell(cells[4])),
                        price_high=int(self._parse_number(self._clean_cell(cells[5]))),
                        price_low=int(self._parse_number(self._clean_cell(cells[6]))),
                        price_avg=int(self._parse_number(self._clean_cell(cells[7]))),
                        trade_date=date,
                    )
                    records.append(record)
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse row: {e}")
                    continue

        return records

    def parse(self, html: str, date: str) -> ParseResult:
        """
        Parse HTML page and extract all data.

        Args:
            html: HTML page content
            date: Trade date in YYYY.MM.DD format

        Returns:
            ParseResult with records, total pages, and empty flag
        """
        is_empty = self.is_empty_result(html)
        total_pages = self.get_total_pages(html) if not is_empty else 1
        records = self.parse_table(html, date) if not is_empty else []

        return ParseResult(
            records=records,
            total_pages=total_pages,
            is_empty=is_empty,
        )
