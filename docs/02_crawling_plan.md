# Crawling Plan: Daily Fish Price Data (miw3110)

## Executive Summary

| Metric                   | Value                               |
| ------------------------ | ----------------------------------- |
| **Endpoint**             | `/nsis/miw/ko/info/miw3110`         |
| **Data Period**          | January 1, 2004 ~ Present           |
| **Total Duration**       | ~21 years                           |
| **Crawling Unit**        | Per day (all species in one query)  |
| **Estimated Days**       | ~7,700 (all days, empty ones skipped) |
| **Estimated Records**    | 1-2 million                         |
| **Estimated Crawl Time** | 4-8 hours (initial)                 |
| **Daily Update Time**    | ~2 minutes                          |

---

## Crawling Strategy Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DAILY CRAWLING STRATEGY                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  PHASE 1: Initial Historical Crawl                         │     │
│  │  ═══════════════════════════════════════                   │     │
│  │  • Period: Jan 1, 2004 → Today                             │     │
│  │  • Strategy: Iterate ALL dates, handle empty gracefully   │     │
│  │  • Total Days: ~7,700 (try all, skip empty results)        │     │
│  │  • Requests per Day: 1-40 (depending on pages)             │     │
│  │  • Estimated Time: 4-8 hours                               │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  PHASE 2: Daily Incremental Updates                        │     │
│  │  ═══════════════════════════════════════                   │     │
│  │  • Frequency: Daily (via cron)                             │     │
│  │  • Scope: Previous day's data only                         │     │
│  │  • Requests: 1-50 per day                                  │     │
│  │  • Estimated Time: ~2 minutes                              │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Initial Historical Crawl

### Crawl Configuration

```python
CRAWL_CONFIG = {
    "base_url": "https://www.susansijang.co.kr",
    "endpoint": "/nsis/miw/ko/info/miw3110",
    "start_date": "2004-01-01",
    "end_date": None,                   # Current date
    "delay_between_requests": 0.5,      # seconds
    "delay_between_days": 0.5,          # seconds
    "max_retries": 3,
    "timeout": 30,                      # seconds
    "page_size": 10                     # Items per page (server default)
}
```

### Crawling Algorithm

```
For each date from 2004-01-01 to today:
    1. Check if date already crawled (checkpoint)
    2. If not crawled:
        a. Fetch page 1 for the date
        b. If empty result (no data / holiday / Sunday):
           - Mark date as crawled (with 0 records)
           - Continue to next date
        c. Parse total pages from pagination
        d. Fetch remaining pages (if any)
        e. Parse and normalize all records
        f. Save to database
        g. Update checkpoint
        h. Sleep for delay period
```

> **Design Decision**: We intentionally try ALL dates rather than skipping Sundays/holidays.
> This is simpler and more robust than maintaining a complex lunar calendar for Korean holidays.
> The overhead (~1,400 extra empty requests over 21 years) is negligible (~45 minutes).

### Request Parameters

```python
def build_request_params(date: str, page: int = 1) -> dict:
    """Build POST parameters for miw3110 endpoint."""
    return {
        "pageIndex": page,
        "pageUnit": 10,
        "pageSize": 10,
        "kdfshNm": "",           # Empty = all species
        "searchDe": date         # Format: YYYY.MM.DD
    }
```

### Pagination Handling

```python
async def crawl_date(session, date: str) -> list:
    """Crawl all records for a specific date."""
    all_records = []

    # Fetch first page
    params = build_request_params(date, page=1)
    response = await fetch_page(session, params)

    if not response.success:
        return []

    parsed = parse_page(response.content)
    all_records.extend(parsed.records)

    # Get total pages from pagination
    total_pages = parsed.total_pages

    # Fetch remaining pages
    for page in range(2, total_pages + 1):
        await rate_limiter.wait()
        params = build_request_params(date, page=page)
        response = await fetch_page(session, params)

        if response.success:
            parsed = parse_page(response.content)
            all_records.extend(parsed.records)

    return all_records
```

### Pagination Detection

```python
def get_total_pages(html: str) -> int:
    """Extract total page count from pagination HTML."""
    soup = BeautifulSoup(html, 'lxml')
    pagination = soup.select_one('.pagination')

    if not pagination:
        return 1

    # Find the last page link: fnList(N)
    last_link = pagination.select_one('.arr.last')
    if last_link:
        onclick = last_link.get('onclick', '')
        match = re.search(r'fnList\((\d+)\)', onclick)
        if match:
            return int(match.group(1))

    # Fallback: count page links
    page_links = pagination.select('a[onclick*="fnList"]')
    return len(page_links) if page_links else 1
```

---

## Phase 2: Daily Incremental Updates

### Cron Schedule

```bash
# Run daily at 12:00 PM KST (after morning market closes)
0 12 * * * /path/to/venv/bin/python /path/to/scripts/daily_update.py
```

### Daily Update Logic

```python
async def daily_update():
    """Fetch yesterday's data (always try, handle empty gracefully)."""
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime("%Y.%m.%d")

    logger.info(f"Fetching data for {date_str}")

    async with Fetcher() as fetcher:
        records = await crawl_date(fetcher.session, date_str)

        if records:
            await save_records(records, yesterday)
            await compute_daily_summary(yesterday)
            logger.info(f"Saved {len(records)} records")
        else:
            # Empty result is normal (Sunday/holiday/no trading)
            logger.info(f"No records for {date_str} (non-trading day)")
```

---

## Crawler Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CRAWLER COMPONENTS                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐                                                   │
│  │  Scheduler   │  Controls date iteration                          │
│  │              │  • Generates date sequence (ALL dates)            │
│  │              │  • Handles empty results gracefully               │
│  │              │  • Manages resume from checkpoint                 │
│  └──────┬───────┘                                                   │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐                                                   │
│  │   Fetcher    │  Handles HTTP requests                            │
│  │              │  • Session management                             │
│  │              │  • Rate limiting                                  │
│  │              │  • Retry logic                                    │
│  │              │  • Pagination handling                            │
│  └──────┬───────┘                                                   │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐                                                   │
│  │  HTML Parser │  Extracts data from responses                     │
│  │              │  • Table parsing (8 columns)                      │
│  │              │  • Pagination detection                           │
│  │              │  • Empty result detection                         │
│  └──────┬───────┘                                                   │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐                                                   │
│  │  Normalizer  │  Cleans and transforms data                       │
│  │              │  • Extract state from species name                │
│  │              │  • Parse numeric values                           │
│  │              │  • Validate data integrity                        │
│  └──────┬───────┘                                                   │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐                                                   │
│  │  DB Writer   │  Persists data to database                        │
│  │              │  • Batch inserts per day                          │
│  │              │  • Deduplication                                  │
│  │              │  • Transaction management                         │
│  └──────────────┘                                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Rate Limiting Strategy

### Polite Crawling Rules

| Rule                        | Value                     | Rationale                  |
| --------------------------- | ------------------------- | -------------------------- |
| Delay between page requests | 1.5 seconds               | Avoid server overload      |
| Delay between dates         | 0.5 seconds               | Brief pause after each day |
| Max requests per minute     | 30                        | Stay under radar           |
| Concurrent connections      | 1                         | Sequential crawling        |
| Retry attempts              | 3                         | Handle transient failures  |
| Retry backoff               | Exponential (2^n seconds) | Gradual recovery           |

### Implementation

```python
class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.interval = 60.0 / requests_per_minute
        self.last_request = 0

    async def wait(self):
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.interval:
            await asyncio.sleep(self.interval - elapsed)
        self.last_request = time.time()
```

---

## Non-Trading Day Handling

### Strategy: Try All, Handle Empty

Rather than maintaining a complex holiday calendar (which is error-prone for Korean lunar holidays like Seollal and Chuseok), we simply:

1. **Try to crawl every single date**
2. **Handle empty results gracefully**
3. **Mark the date as "crawled" regardless of whether data exists**

```python
async def crawl_date(session, date: str) -> list:
    """Crawl a date. Returns empty list for non-trading days."""
    records = await fetch_all_pages(session, date)

    if not records:
        # This is normal - could be Sunday, holiday, or just no trading
        logger.debug(f"No data for {date} (non-trading day)")

    return records  # Empty list is valid result
```

### Benefits of This Approach

| Benefit | Description |
|---------|-------------|
| **Simplicity** | No need to compute lunar calendar or maintain holiday lists |
| **Robustness** | Handles unexpected closures (weather, special events, etc.) |
| **Maintainability** | Zero configuration for holidays |
| **Overhead** | Only ~45 extra minutes over 21 years (~1,400 empty requests) |

---

## Checkpoint Management

### Checkpoint File Format

```json
{
  "last_updated": "2025-01-02T10:30:00",
  "status": "in_progress",
  "last_completed_date": "2024-06-15",
  "statistics": {
    "total_days_crawled": 5420,
    "total_records": 892340,
    "failed_dates": ["2020-01-25", "2019-09-14"]
  }
}
```

### Checkpoint Logic

```python
class CheckpointManager:
    def __init__(self, checkpoint_path: str = "data/checkpoint.json"):
        self.path = Path(checkpoint_path)
        self.state = self._load()

    def get_start_date(self) -> datetime:
        """Get the date to resume crawling from."""
        if self.state.get("last_completed_date"):
            last_date = datetime.strptime(
                self.state["last_completed_date"], "%Y-%m-%d"
            )
            return last_date + timedelta(days=1)
        return datetime(2004, 1, 2)

    def mark_date_completed(self, date: datetime, record_count: int):
        self.state["last_completed_date"] = date.strftime("%Y-%m-%d")
        self.state["statistics"]["total_days_crawled"] += 1
        self.state["statistics"]["total_records"] += record_count
        self.save()
```

---

## Error Handling

### Error Types and Responses

| Error Type              | Response               | Action          |
| ----------------------- | ---------------------- | --------------- |
| HTTP 429 (Rate Limited) | Back off exponentially | Wait 60 seconds |
| HTTP 5xx (Server Error) | Retry with backoff     | Max 3 retries   |
| Connection Timeout      | Retry immediately      | Max 3 retries   |
| Parse Error             | Log and skip date      | Mark as failed  |
| Empty Response          | Normal (holiday)       | Mark as no data |

### Failed Date Handling

```python
async def crawl_with_retry(date: str) -> tuple[list, bool]:
    """Crawl a date with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            records = await crawl_date(session, date)

            # Empty result is OK (holiday/no trading)
            if not records:
                return [], True

            return records, True

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {date}: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)

    return [], False  # Failed after retries
```

---

## Data Extraction

### HTML Parsing (miw3110 - 8 columns)

```python
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List
import re

@dataclass
class PriceRecord:
    species_raw: str    # e.g., "(활)방어"
    origin: str         # e.g., "일본"
    spec: str           # e.g., "2미"
    packaging: str      # e.g., "kg"
    quantity: float     # e.g., 203.0
    price_high: int     # e.g., 30000
    price_low: int      # e.g., 5000
    price_avg: int      # e.g., 13900

def parse_number(text: str) -> float:
    """Parse Korean number format (with commas)."""
    cleaned = re.sub(r'[^\d.]', '', text)
    return float(cleaned) if cleaned else 0

def parse_price_table(html: str) -> list[PriceRecord]:
    """Parse price data from miw3110 HTML page."""
    soup = BeautifulSoup(html, 'lxml')
    records = []

    table = soup.select_one('.list-table table')
    if not table:
        return records

    rows = table.select('tbody tr')
    for row in rows:
        # Skip "no data" rows
        if row.select_one('.no-data') or '조회된 경락시세가 없습니다' in row.get_text():
            continue

        cells = row.select('td')
        if len(cells) >= 8:
            record = PriceRecord(
                species_raw=cells[0].get_text(strip=True),
                origin=cells[1].get_text(strip=True),
                spec=cells[2].get_text(strip=True),
                packaging=cells[3].get_text(strip=True),
                quantity=parse_number(cells[4].get_text(strip=True)),
                price_high=int(parse_number(cells[5].get_text(strip=True))),
                price_low=int(parse_number(cells[6].get_text(strip=True))),
                price_avg=int(parse_number(cells[7].get_text(strip=True)))
            )
            records.append(record)

    return records
```

### Data Normalization

```python
def extract_state(species_raw: str) -> tuple[str | None, str]:
    """Extract state prefix from species name.

    Examples:
        "(냉)고등어" -> ("냉", "고등어")
        "(활)방어" -> ("활", "방어")
        "고등어" -> (None, "고등어")
    """
    pattern = r'^\(([^)]+)\)(.+)$'
    match = re.match(pattern, species_raw)

    if match:
        return match.group(1), match.group(2)
    return None, species_raw

def normalize_record(record: PriceRecord, trade_date: datetime) -> dict:
    """Normalize a raw price record for database insertion."""
    state, species = extract_state(record.species_raw)

    return {
        'trade_date': int(trade_date.timestamp()),
        'species': species,
        'state': state,
        'origin': record.origin,
        'spec': record.spec if record.spec else None,
        'packaging': record.packaging if record.packaging else None,
        'quantity': record.quantity,
        'price_high': record.price_high,
        'price_low': record.price_low,
        'price_avg': record.price_avg
    }
```

---

## Estimated Timeline

### Initial Crawl Breakdown

| Period    | Total Days | Trading Days | Avg Pages/Day | Requests    | Time (est.) |
| --------- | ---------- | ------------ | ------------- | ----------- | ----------- |
| 2004-2006 | ~1,100     | ~750         | 2             | ~1,850      | ~50 min     |
| 2007-2010 | ~1,460     | ~1,050       | 5             | ~6,300      | ~3 hr       |
| 2011-2015 | ~1,825     | ~1,300       | 10            | ~14,800     | ~6 hr       |
| 2016-2020 | ~1,825     | ~1,300       | 15            | ~21,300     | ~9 hr       |
| 2021-2025 | ~1,460     | ~1,050       | 25            | ~27,500     | ~12 hr      |
| **Total** | **~7,670** | **~5,450**   | -             | **~71,750** | **~4-8 hr** |

> **Note**: We try ALL days (including ~2,220 Sundays/holidays that return empty).
> Empty days take only ~2 seconds each, adding ~45 minutes total overhead.
> With 1.5s delay between requests, actual time is 4-8 hours depending on server response.

### Optimization: Concurrent Date Processing

```python
# Optional: Process multiple dates concurrently (with caution)
async def crawl_batch(dates: list[str], concurrency: int = 3):
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_crawl(date):
        async with semaphore:
            return await crawl_date(session, date)

    tasks = [bounded_crawl(d) for d in dates]
    return await asyncio.gather(*tasks)
```

---

## Monitoring and Logging

### Progress Logging

```python
def log_progress(days_done: int, total_days: int, records_today: int):
    percent = (days_done / total_days) * 100
    logger.info(
        f"Progress: {days_done}/{total_days} days ({percent:.1f}%) "
        f"- Today: {records_today} records"
    )
```

### Daily Summary

```python
async def log_daily_summary():
    with get_db_connection() as conn:
        stats = conn.execute("""
            SELECT
                COUNT(DISTINCT date(trade_date, 'unixepoch')) as days,
                COUNT(*) as records
            FROM fish_prices
        """).fetchone()

    logger.info(f"Database: {stats['days']} days, {stats['records']} records")
```
