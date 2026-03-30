# Noryangjin Fish Market Crawler

Modular crawler for fish auction prices (경락시세) from Noryangjin Fish Market.

| Item | Value |
|------|-------|
| **Target URL** | https://www.susansijang.co.kr/nsis/miw/ko/info/miw3110 |
| **Method** | POST |
| **Data Period** | January 2004 ~ Present |
| **Version** | 0.2.0 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     NoryangjinCrawler                           │
│                       (Orchestrator)                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┬─────────────────┐
        │                   │                   │                 │
        ▼                   ▼                   ▼                 ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Scheduler   │   │   Fetcher    │   │  HTMLParser  │   │  Normalizer  │
│              │   │              │   │              │   │              │
│ • Date       │   │ • HTTP POST  │   │ • Table      │   │ • Validation │
│   generation │   │ • Rate limit │   │   parsing    │   │ • State      │
│ • Checkpoint │   │ • Retry      │   │ • Pagination │   │   extraction │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │    Writer    │
                   │              │
                   │ • Parquet    │
                   │ • JSON       │
                   │ • CSV        │
                   │ • Memory     │
                   └──────────────┘
```

---

## Directory Structure

```
crawler/
├── __init__.py            # Module exports (v0.2.0)
├── models.py              # Data classes
├── fetcher.py             # HTTP client with rate limiting
├── parser.py              # HTML parser
├── normalizer.py          # Data cleaning & validation
├── scheduler.py           # Date iteration & checkpoint
├── writer.py              # Data persistence (JSON/CSV)
├── noryangjin_crawler.py  # Main orchestrator
└── README.md              # This file
```

---

## Modules

### models.py

Data classes used across all components.

| Class | Description |
|-------|-------------|
| `PriceRecord` | Single fish price record (11 fields) |
| `CrawlDayResult` | Result of crawling a single day |
| `CrawlStats` | Statistics for a crawl session |
| `CheckpointState` | Checkpoint state for resume |

```python
@dataclass
class PriceRecord:
    species_raw: str      # "(활)방어"
    species: str          # "방어"
    state: Optional[str]  # "활", "선", "냉", "가공"
    origin: str           # "일본", "부산"
    spec: str             # "2미", "대"
    packaging: str        # "kg", "box"
    quantity: float
    price_high: int       # KRW
    price_low: int        # KRW
    price_avg: int        # KRW
    trade_date: str       # "YYYY.MM.DD"
```

---

### fetcher.py

HTTP client with rate limiting and retry logic.

| Class | Description |
|-------|-------------|
| `RateLimiter` | Controls request frequency |
| `Fetcher` | Async HTTP client |

```python
class Fetcher:
    BASE_URL = "https://www.susansijang.co.kr"
    ENDPOINT = "/nsis/miw/ko/info/miw3110"

    async def fetch_page(date: str, page: int = 1) -> Tuple[str, bool]
```

**Configuration:**
- `delay_between_requests`: 1.5s (default)
- `max_retries`: 3
- `timeout`: 30s

---

### parser.py

HTML parser for extracting price data.

| Class | Description |
|-------|-------------|
| `ParseResult` | Parsed page result |
| `HTMLParser` | Regex-based HTML parser |

```python
class HTMLParser:
    def parse(html: str, date: str) -> ParseResult
    def parse_table(html: str, date: str) -> List[PriceRecord]
    def get_total_pages(html: str) -> int
    def extract_state(species_raw: str) -> Tuple[Optional[str], str]
```

**Features:**
- 8-column table extraction
- Pagination detection via `fnList(N)` pattern
- State prefix extraction: `(활)`, `(선)`, `(냉)`, `(가공)`

---

### normalizer.py

Data cleaning and validation.

| Class | Description |
|-------|-------------|
| `Normalizer` | Validates and transforms records |

```python
class Normalizer:
    STATE_CODES = {
        "선": "fresh",
        "활": "live",
        "냉": "frozen",
        "가공": "processed"
    }

    def validate_record(record: PriceRecord) -> bool
    def normalize_record(record: PriceRecord) -> Dict[str, Any]
    def compute_daily_summary(records: List[PriceRecord]) -> Dict
```

---

### scheduler.py

Date iteration and checkpoint management.

| Class | Description |
|-------|-------------|
| `CheckpointManager` | Resume capability via JSON file |
| `Scheduler` | Date sequence generator |

```python
class Scheduler:
    START_DATE = datetime(2004, 1, 2)

    def generate_dates(start_date: str, end_date: str) -> Generator[str]
    def count_days(start_date: str, end_date: str) -> int

# Utilities
def format_date(dt: datetime) -> str   # -> "YYYY.MM.DD"
def parse_date(date_str: str) -> datetime
```

**Checkpoint Format:**
```json
{
  "last_updated": "2025-01-02T10:30:00",
  "status": "in_progress",
  "last_completed_date": "2024.06.15",
  "statistics": {
    "total_days_crawled": 5420,
    "total_records": 892340,
    "failed_dates": []
  }
}
```

---

### writer.py

Data persistence with multiple output formats.

| Class | Description |
|-------|-------------|
| `BaseWriter` | Abstract base class |
| `ParquetWriter` | Partitioned Parquet files (recommended) |
| `JSONWriter` | One JSON file per day |
| `CSVWriter` | Append to single CSV |
| `MemoryWriter` | In-memory (for testing) |
| `CompositeWriter` | Multiple writers |

```python
# Parquet output (recommended): data/parquet/prices/year=YYYY/month=MM/data.parquet
writer = ParquetWriter(output_dir="data/parquet/prices")

# JSON output: data/raw/YYYY/MM/YYYY-MM-DD.json
writer = JSONWriter(output_dir="data/raw")

# CSV output: data/fish_prices.csv
writer = CSVWriter(output_path="data/fish_prices.csv")

# Both Parquet and JSON (for backup)
writer = CompositeWriter([ParquetWriter(), JSONWriter()])
```

**ParquetWriter Features:**
- Hive-style partitioning by year/month
- Snappy compression (50-70% smaller than CSV)
- Automatic daily append within partition
- Query-ready for DuckDB

---

### noryangjin_crawler.py

Main orchestrator that coordinates all components.

```python
class NoryangjinCrawler:
    def __init__(
        delay_between_requests: float = 1.5,
        delay_between_days: float = 0.5,
        max_retries: int = 3,
        timeout: int = 30,
        page_size: int = 10,
        checkpoint_path: Optional[str] = None,
        writer: Optional[BaseWriter] = None,
    )

    async def crawl_date(date: str) -> CrawlDayResult
    async def crawl_date_range(start: str, end: str, callback) -> Tuple[List, CrawlStats]
    async def crawl_historical(callback) -> Tuple[List, CrawlStats]
    async def crawl_yesterday() -> CrawlDayResult
```

---

## Usage

### Basic Usage

```python
from crawler import NoryangjinCrawler

async with NoryangjinCrawler() as crawler:
    # Single date
    result = await crawler.crawl_date("2024.01.02")
    print(f"Found {len(result.records)} records")

    # Date range
    results, stats = await crawler.crawl_date_range(
        "2024.01.01", "2024.01.31"
    )
    print(f"Total: {stats.total_records} records")
```

### With Checkpoint & Writer

```python
from crawler import NoryangjinCrawler, ParquetWriter

# Recommended: Parquet for production (50-70% smaller, faster queries)
writer = ParquetWriter(output_dir="data/parquet/prices")

async with NoryangjinCrawler(
    checkpoint_path="data/checkpoint.json",
    writer=writer,
) as crawler:
    results, stats = await crawler.crawl_historical()
```

### With JSON Writer (for debugging/backup)

```python
from crawler import NoryangjinCrawler, JSONWriter, CompositeWriter, ParquetWriter

# Both Parquet (for queries) and JSON (for backup/debugging)
writer = CompositeWriter([
    ParquetWriter(output_dir="data/parquet/prices"),
    JSONWriter(output_dir="data/raw"),
])

async with NoryangjinCrawler(
    checkpoint_path="data/checkpoint.json",
    writer=writer,
) as crawler:
    results, stats = await crawler.crawl_historical()
```

### Using Individual Components

```python
from crawler import Fetcher, HTMLParser, Normalizer

async with Fetcher() as fetcher:
    html, success = await fetcher.fetch_page("2024.01.02", page=1)

parser = HTMLParser()
result = parser.parse(html, "2024.01.02")

normalizer = Normalizer()
validated = [r for r in result.records if normalizer.validate_record(r)]
```

---

## Scripts

Located in `scripts/crawler/`:

| Script | Description | Usage |
|--------|-------------|-------|
| `historical_crawl.py` | Full crawl (2004~present) | `uv run python scripts/crawler/historical_crawl.py` |
| `daily_crawl.py` | Yesterday's data (cron) | `uv run python scripts/crawler/daily_crawl.py` |
| `run_crawler.py` | Unified CLI | `uv run python scripts/crawler/run_crawler.py <command>` |

### Unified CLI Commands

```bash
# Historical (with resume)
uv run python scripts/crawler/run_crawler.py historical

# Daily
uv run python scripts/crawler/run_crawler.py daily

# Date range
uv run python scripts/crawler/run_crawler.py range --start 2024.01.01 --end 2024.01.31

# Single date
uv run python scripts/crawler/run_crawler.py date --date 2024.01.15
```

### Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `data/parquet/prices` | Output directory |
| `--format` | `parquet` | parquet, json, csv, both, none |
| `--delay` | `1.5` | Seconds between requests |
| `--log-dir` | `logs` | Log directory |

### Cron Setup

```bash
# Daily at 12:00 PM KST
0 12 * * * cd /path/to/project && uv run python scripts/crawler/daily_crawl.py
```

---

## Exports

```python
from crawler import (
    # Orchestrator
    NoryangjinCrawler,

    # Models
    PriceRecord, CrawlDayResult, CrawlStats, CheckpointState, ParseResult,

    # Components
    Fetcher, RateLimiter, HTMLParser, Normalizer, Scheduler, CheckpointManager,

    # Writers
    BaseWriter, ParquetWriter, JSONWriter, CSVWriter, MemoryWriter, CompositeWriter,

    # Utilities
    format_date, parse_date, records_to_dicts,
)
```

---

## Performance

| Operation | Records | Pages | Time |
|-----------|---------|-------|------|
| Single date | ~300 | ~30 | ~40s |
| 3-day range | ~800 | ~85 | ~90s |
| Full historical | ~1.5M | ~70K | 4-8 hours |
