"""
Fetcher Component for Noryangjin Crawler.

Handles HTTP requests with:
- Session management
- Rate limiting
- Retry logic with exponential backoff
"""
import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter to control request frequency."""

    def __init__(self, delay_seconds: float = 1.5):
        """
        Initialize rate limiter.

        Args:
            delay_seconds: Minimum delay between requests
        """
        self.delay = delay_seconds
        self._last_request_time: float = 0

    async def wait(self) -> None:
        """Wait if necessary to respect rate limit."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        self._last_request_time = time.time()


class Fetcher:
    """
    HTTP fetcher with rate limiting and retry logic.

    Handles all HTTP communication with the target server.
    """

    BASE_URL = "https://www.susansijang.co.kr"
    ENDPOINT = "/nsis/miw/ko/info/miw3110"

    def __init__(
        self,
        delay_between_requests: float = 1.5,
        max_retries: int = 3,
        timeout: int = 30,
        page_size: int = 10,
    ):
        """
        Initialize fetcher.

        Args:
            delay_between_requests: Delay between HTTP requests in seconds
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            page_size: Number of items per page
        """
        self.rate_limiter = RateLimiter(delay_between_requests)
        self.max_retries = max_retries
        self.timeout = timeout
        self.page_size = page_size
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _build_params(self, date: str, page: int = 1) -> Dict[str, Any]:
        """
        Build POST parameters for miw3110 endpoint.

        Args:
            date: Date in YYYY.MM.DD format
            page: Page number (1-indexed)

        Returns:
            Dictionary of POST parameters
        """
        return {
            "pageIndex": page,
            "pageUnit": self.page_size,
            "pageSize": self.page_size,
            "kdfshNm": "",  # Empty = all species
            "searchDe": date,  # Format: YYYY.MM.DD
        }

    async def fetch_page(self, date: str, page: int = 1) -> Tuple[str, bool]:
        """
        Fetch a single page of data with retry logic.

        Args:
            date: Date in YYYY.MM.DD format
            page: Page number (1-indexed)

        Returns:
            Tuple of (HTML content, success flag)
        """
        await self.rate_limiter.wait()

        session = await self._get_session()
        url = f"{self.BASE_URL}{self.ENDPOINT}"
        params = self._build_params(date, page)

        for attempt in range(self.max_retries):
            try:
                async with session.post(url, data=params) as response:
                    if response.status == 200:
                        html = await response.text()
                        return html, True
                    elif response.status == 429:
                        # Rate limited - back off exponentially
                        wait_time = 60 * (attempt + 1)
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(f"HTTP {response.status} for {date} page {page}")

            except aiohttp.ClientError as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break

        return "", False

    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
