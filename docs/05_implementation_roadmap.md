# Implementation Roadmap

## Project Overview

| Attribute | Value |
|-----------|-------|
| **Project Name** | Noryangjin Fish Price MCP |
| **Goal** | MCP server providing fish price data |
| **Data Source** | Noryangjin Fish Market (노량진수산시장) |
| **Target Platform** | AWS t4-micro |
| **Storage** | Parquet + DuckDB |

---

## Implementation Phases

```
┌─────────────────────────────────────────────────────────────────────┐
│                     IMPLEMENTATION TIMELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PHASE 1: Project Setup ✅ DONE                                     │
│  ══════════════════════                                             │
│  ✅ Project structure, dependencies, database schema                │
│                                                                      │
│  PHASE 2: Crawler Development ✅ DONE                               │
│  ════════════════════════════                                       │
│  ✅ HTTP client, HTML parser, data normalizer, checkpoint system    │
│  ✅ Species inventory, preprocessing pipeline (5 fixes)             │
│                                                                      │
│  PHASE 3: Historical Data Collection ✅ DONE                        │
│  ════════════════════════════════════                               │
│  ✅ 2.59M rows, 504 species, 2004–2026                             │
│  ✅ Data validation, EDA, aggregation analysis                      │
│                                                                      │
│  PHASE 4: Prediction System ✅ DONE                                 │
│  ═══════════════════════════                                        │
│  ✅ 12 CPU iterations (v1–v10), 7 DL models (GPU)                  │
│  ✅ 20 prediction configs across 15 species                         │
│  ✅ All 7 sashimi species below 17% MAPE                           │
│  ✅ Quantile bands (p10/p50/p90) + conformal intervals             │
│  ✅ v10 preprocessing (18-29% MAPE reduction)                       │
│  ✅ Config registry: docs/15_prediction_config_registry.md          │
│                                                                      │
│  PHASE 5: MCP Server ⬜ NOT STARTED                                 │
│  ════════════════════                                               │
│  □ Server skeleton                                                  │
│  □ Tool implementations (price query + prediction)                  │
│  □ Testing                                                          │
│                                                                      │
│  PHASE 6: Security (Anti-Scraping) ⬜ NOT STARTED                   │
│  ═════════════════════════════════                                  │
│  □ Rate limiting, audit logging, API key auth                       │
│                                                                      │
│  PHASE 7: Deployment ⬜ NOT STARTED                                 │
│  ════════════════════                                               │
│  □ Daily pipeline (crawl → preprocess → predict → serve)            │
│  □ Docker deployment, monitoring                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Project Setup

### 1.1 Create Project Structure

```
mcp_NorayngjinGavel/
├── docs/                           # Documentation (already created)
│   ├── 01_website_analysis.md
│   ├── 02_crawling_plan.md
│   ├── 03_database_design.md
│   ├── 04_mcp_server_design.md
│   ├── 05_implementation_roadmap.md
│   ├── 06_prediction_system.md
│   └── 07_security.md
│
├── src/
│   ├── __init__.py
│   │
│   ├── crawler/                    # Web crawler module
│   │   ├── __init__.py
│   │   ├── fetcher.py              # HTTP request handling
│   │   ├── parser.py               # HTML parsing
│   │   ├── normalizer.py           # Data normalization
│   │   ├── scheduler.py            # Crawl scheduling
│   │   └── checkpoint.py           # Progress tracking
│   │
│   ├── database/                   # Database module
│   │   ├── __init__.py
│   │   ├── connection.py           # DB connection management
│   │   ├── models.py               # Data models
│   │   ├── repository.py           # Query methods
│   │   └── migrations/             # Schema migrations
│   │       └── 001_initial.sql
│   │
│   └── mcp_server/                 # MCP server module
│       ├── __init__.py
│       ├── server.py               # Main server entry
│       ├── tools.py                # Tool implementations
│       ├── resources.py            # Resource handlers
│       ├── formatters.py           # Response formatting
│       └── security/               # Security module (anti-scraping)
│           ├── __init__.py
│           ├── limits.py           # Result set limits
│           ├── rate_limiter.py     # Session rate limiting
│           ├── auth.py             # API key authentication
│           └── audit.py            # Audit logging & abuse detection
│
├── scripts/
│   ├── init_db.py                  # Initialize database
│   ├── run_crawler.py              # Run historical crawl
│   ├── daily_update.py             # Daily update script
│   └── backup_db.sh                # Backup script
│
├── tests/
│   ├── __init__.py
│   ├── test_crawler/
│   ├── test_database/
│   └── test_mcp_server/
│
├── data/
│   └── .gitkeep                    # Database files (gitignored)
│
├── config/
│   ├── settings.py                 # Configuration
│   └── logging.yaml                # Logging config
│
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Development dependencies
├── pyproject.toml                  # Project metadata
├── .gitignore
└── README.md
```

### 1.2 Dependencies

**requirements.txt:**
```txt
# Web Crawling
aiohttp>=3.9.0
beautifulsoup4>=4.12.0
lxml>=5.0.0

# Data Storage (Parquet + DuckDB)
pyarrow>=14.0.0
duckdb>=0.9.0

# MCP Server
mcp>=1.0.0

# Utilities
python-dateutil>=2.8.0
pydantic>=2.0.0
pandas>=2.0.0

# Logging
structlog>=23.0.0
```

**requirements-dev.txt:**
```txt
-r requirements.txt

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0

# Linting
ruff>=0.1.0
mypy>=1.5.0

# Development
ipython>=8.0.0
```

### 1.3 Initialize Database

**scripts/init_db.py:**
```python
#!/usr/bin/env python3
"""Initialize DuckDB database and Parquet directory structure."""

import duckdb
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DUCKDB_PATH = DATA_DIR / "fish_market.duckdb"
PARQUET_DIR = DATA_DIR / "parquet" / "prices"

def init_database():
    """Create DuckDB database and Parquet directories."""
    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    # Connect to DuckDB
    conn = duckdb.connect(str(DUCKDB_PATH))

    # Create lookup tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fish_species (
            id INTEGER PRIMARY KEY,
            name VARCHAR NOT NULL UNIQUE,
            name_en VARCHAR,
            category VARCHAR,
            created_at TIMESTAMP DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS fish_states (
            id INTEGER PRIMARY KEY,
            code VARCHAR NOT NULL UNIQUE,
            name VARCHAR NOT NULL,
            name_en VARCHAR
        );

        CREATE TABLE IF NOT EXISTS origins (
            id INTEGER PRIMARY KEY,
            name VARCHAR NOT NULL UNIQUE,
            region VARCHAR,
            is_domestic BOOLEAN DEFAULT TRUE
        );

        CREATE TABLE IF NOT EXISTS packaging_types (
            id INTEGER PRIMARY KEY,
            code VARCHAR NOT NULL UNIQUE,
            description VARCHAR
        );

        CREATE TABLE IF NOT EXISTS crawl_metadata (
            trade_date DATE PRIMARY KEY,
            record_count INTEGER NOT NULL,
            crawled_at TIMESTAMP DEFAULT now()
        );
    """)

    # Seed lookup data
    seed_lookup_tables(conn)

    # Create view for Parquet data
    conn.execute(f"""
        CREATE OR REPLACE VIEW v_prices AS
        SELECT * FROM read_parquet('{PARQUET_DIR}/**/*.parquet', hive_partitioning=true)
    """)

    conn.close()
    print(f"Database initialized at: {DUCKDB_PATH}")
    print(f"Parquet directory: {PARQUET_DIR}")

def seed_lookup_tables(conn):
    """Insert initial lookup data."""
    # Fish states
    conn.execute("""
        INSERT OR IGNORE INTO fish_states (id, code, name, name_en) VALUES
            (1, '선', '선어', 'Fresh'),
            (2, '활', '활어', 'Live'),
            (3, '냉', '냉동', 'Frozen'),
            (4, '가공', '가공', 'Processed')
    """)

    # Common packaging types
    conn.execute("""
        INSERT OR IGNORE INTO packaging_types (id, code, description) VALUES
            (1, 'kg', 'Per kilogram'),
            (2, 'S/P', 'Styrofoam package'),
            (3, 'box', 'Box/Crate'),
            (4, 'CT/(BT)', 'Carton/Basket')
    """)

if __name__ == "__main__":
    init_database()
```

### 1.4 Checklist

- [ ] Create directory structure
- [ ] Initialize git repository (if not already)
- [ ] Create requirements.txt
- [ ] Create pyproject.toml
- [ ] Create .gitignore
- [ ] Run init_db.py
- [ ] Verify DuckDB created with correct schema
- [ ] Verify Parquet directory structure created

---

## Phase 2: Crawler Development

### 2.1 HTTP Fetcher

**src/crawler/fetcher.py:**
```python
"""HTTP client for fetching web pages."""

import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class FetchResult:
    success: bool
    status_code: int
    content: Optional[str]
    error: Optional[str] = None

class RateLimiter:
    """Rate limiter for polite crawling."""

    def __init__(self, requests_per_minute: int = 30):
        self.interval = 60.0 / requests_per_minute
        self.last_request = 0

    async def wait(self):
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_request
        if elapsed < self.interval:
            await asyncio.sleep(self.interval - elapsed)
        self.last_request = asyncio.get_event_loop().time()

class Fetcher:
    """Async HTTP fetcher with rate limiting and retries."""

    BASE_URL = "https://www.susansijang.co.kr"
    ENDPOINT = "/nsis/miw/ko/info/miw3110"

    def __init__(
        self,
        requests_per_minute: int = 30,
        max_retries: int = 3,
        timeout: int = 30
    ):
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_prices(
        self,
        date: str,
        page: int = 1
    ) -> FetchResult:
        """Fetch price data for a single date (all species)."""

        await self.rate_limiter.wait()

        params = {
            "pageIndex": page,
            "pageUnit": 10,
            "pageSize": 10,
            "kdfshNm": "",  # Empty = all species
            "searchDe": date  # Format: YYYY.MM.DD
        }

        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    f"{self.BASE_URL}{self.ENDPOINT}",
                    data=params
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        return FetchResult(
                            success=True,
                            status_code=200,
                            content=content
                        )
                    else:
                        logger.warning(f"HTTP {response.status} for {species}")

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        return FetchResult(
            success=False,
            status_code=0,
            content=None,
            error="Max retries exceeded"
        )
```

### 2.2 HTML Parser

**src/crawler/parser.py:**
```python
"""HTML parser for extracting price data."""

from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List, Optional
import re

@dataclass
class PriceRecord:
    species_raw: str
    origin: str
    spec: str
    packaging: str
    quantity: float
    price_high: int
    price_low: int
    price_avg: int

@dataclass
class ParseResult:
    records: List[PriceRecord]
    summary: dict
    has_more_pages: bool

def parse_number(text: str) -> float:
    """Parse Korean number format (with commas)."""
    cleaned = re.sub(r'[^\d.]', '', text)
    return float(cleaned) if cleaned else 0

def parse_price_page(html: str) -> ParseResult:
    """Parse price data from HTML page."""
    soup = BeautifulSoup(html, 'lxml')
    records = []

    # Parse table rows
    table = soup.select_one('.list-table table')
    if table:
        rows = table.select('tbody tr')
        for row in rows:
            if row.select_one('.no-data'):
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

    # Parse summary
    summary = {}
    summ_list = soup.select('.data_summ li')
    for item in summ_list:
        text = item.get_text()
        if '물량' in text:
            summary['total_weight'] = parse_number(text.split(':')[1])
        elif '판매액' in text:
            summary['total_sales'] = parse_number(text.split(':')[1])

    # Check pagination - look for last page link
    pagination = soup.select_one('.pagination')
    has_more = False
    if pagination:
        last_link = pagination.select_one('.arr.last')
        has_more = last_link is not None and 'onclick' in str(last_link)

    return ParseResult(
        records=records,
        summary=summary,
        has_more_pages=has_more
    )
```

### 2.3 Data Normalizer

**src/crawler/normalizer.py:**
```python
"""Data normalization for price records."""

import re
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class NormalizedRecord:
    trade_date: int          # Unix timestamp
    species: str
    state: Optional[str]
    origin: str
    spec: str
    packaging: str
    quantity: float
    price_high: int
    price_low: int
    price_avg: int

def extract_state(species_raw: str) -> Tuple[Optional[str], str]:
    """Extract state prefix from species name.

    Examples:
        "(냉)고등어" -> ("냉", "고등어")
        "고등어" -> (None, "고등어")
    """
    pattern = r'^\(([^)]+)\)(.+)$'
    match = re.match(pattern, species_raw)

    if match:
        return match.group(1), match.group(2)
    return None, species_raw

def normalize_record(record, query_date: str) -> NormalizedRecord:
    """Normalize a raw price record."""

    # Parse date
    date_obj = datetime.strptime(query_date, "%Y.%m.%d")
    timestamp = int(date_obj.timestamp())

    # Extract state from species name
    state, species = extract_state(record.species_raw)

    return NormalizedRecord(
        trade_date=timestamp,
        species=species,
        state=state,
        origin=record.origin,
        spec=record.spec,
        packaging=record.packaging,
        quantity=record.quantity,
        price_high=record.price_high,
        price_low=record.price_low,
        price_avg=record.price_avg
    )
```

### 2.4 Checkpoint Manager

**src/crawler/checkpoint.py:**
```python
"""Checkpoint management for crawl progress."""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Set

@dataclass
class CrawlState:
    last_updated: str
    status: str  # "idle", "in_progress", "completed"
    completed_dates: Set[str]  # Set of "YYYY-MM-DD" dates
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_records: int

class CheckpointManager:
    """Manages crawl progress checkpoints."""

    def __init__(self, checkpoint_path: str = "data/checkpoint.json"):
        self.path = Path(checkpoint_path)
        self.state = self._load()

    def _load(self) -> CrawlState:
        if self.path.exists():
            with open(self.path, 'r') as f:
                data = json.load(f)
                data['completed_dates'] = set(data.get('completed_dates', []))
                return CrawlState(**data)
        return CrawlState(
            last_updated=datetime.now().isoformat(),
            status="idle",
            completed_dates=set(),
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            total_records=0
        )

    def save(self):
        self.state.last_updated = datetime.now().isoformat()
        data = asdict(self.state)
        data['completed_dates'] = list(self.state.completed_dates)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def mark_completed(self, date: str):
        """Mark a date as completed. date format: YYYY-MM-DD"""
        self.state.completed_dates.add(date)
        self.state.successful_requests += 1
        self.save()

    def mark_failed(self, date: str):
        self.state.failed_requests += 1
        self.save()

    def add_records(self, count: int):
        self.state.total_records += count
        self.save()

    def should_skip(self, date: str) -> bool:
        """Check if date already crawled. date format: YYYY-MM-DD"""
        return date in self.state.completed_dates

    def set_status(self, status: str):
        self.state.status = status
        self.save()
```

### 2.5 Checklist

- [ ] Implement Fetcher class with rate limiting
- [ ] Implement HTML parser
- [ ] Implement data normalizer
- [ ] Implement checkpoint manager
- [ ] Write unit tests for parser
- [ ] Write unit tests for normalizer
- [ ] Test fetcher against live site (single request)

---

## Phase 3: Historical Data Collection

### 3.1 Run Initial Crawl

**scripts/run_crawler.py:**
```python
#!/usr/bin/env python3
"""Run the historical data crawl."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crawler.fetcher import Fetcher
from crawler.parser import parse_price_page
from crawler.normalizer import normalize_record
from crawler.checkpoint import CheckpointManager
from database.repository import PriceRepository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from datetime import timedelta

# Crawl configuration
START_DATE = "2004-01-02"  # Data available from Jan 2000, but user wants from Jan 2, 2004

# Design Decision: Try ALL dates (including Sundays/holidays)
# rather than maintaining a complex lunar calendar. Empty results
# are handled gracefully - the overhead is only ~45 minutes for 21 years.

async def run_crawl():
    """Run daily crawl from START_DATE to today. Tries ALL dates."""
    checkpoint = CheckpointManager()
    checkpoint.set_status("in_progress")

    repo = PriceRepository()

    # Generate date range
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.now()

    async with Fetcher() as fetcher:
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")

            if checkpoint.should_skip(date_str):
                logger.debug(f"Skipping {date_str}")
                current += timedelta(days=1)
                continue

            logger.info(f"Crawling {date_str}")

            try:
                records = await crawl_date(fetcher, current)

                if records:
                    await repo.insert_prices(records)
                    checkpoint.add_records(len(records))
                else:
                    # Empty result is normal (Sunday/holiday/no trading)
                    logger.debug(f"No data for {date_str} (non-trading day)")

                # Always mark as completed (even if empty)
                checkpoint.mark_completed(date_str)

            except Exception as e:
                logger.error(f"Error on {date_str}: {e}")
                checkpoint.mark_failed(date_str)

            current += timedelta(days=1)

    checkpoint.set_status("completed")
    logger.info("Crawl completed!")

async def crawl_date(fetcher, date: datetime):
    """Crawl all pages for a single date (all species)."""
    date_str = date.strftime("%Y.%m.%d")  # Website format: YYYY.MM.DD

    all_records = []
    page = 1

    while True:
        result = await fetcher.fetch_prices(date_str, page)

        if not result.success:
            break

        parsed = parse_price_page(result.content)

        for record in parsed.records:
            normalized = normalize_record(record, date_str)
            all_records.append(normalized)

        if not parsed.has_more_pages:
            break

        page += 1

        # Add small delay between pages
        await asyncio.sleep(0.5)

    return all_records

if __name__ == "__main__":
    asyncio.run(run_crawl())
```

### 3.2 Daily Summary View

With Parquet + DuckDB, daily summaries are computed on-the-fly via a view (no separate materialization needed):

```sql
-- This view is created in init_db.py
CREATE OR REPLACE VIEW v_daily_summary AS
SELECT
    trade_date,
    species,
    SUM(quantity) AS total_quantity,
    CAST(AVG(price_avg) AS INTEGER) AS avg_price,
    MIN(price_low) AS min_price,
    MAX(price_high) AS max_price,
    COUNT(*) AS record_count
FROM v_prices
GROUP BY trade_date, species;
```

DuckDB's columnar query engine is fast enough that on-the-fly aggregation performs well for this data size.

### 3.3 Checklist

- [ ] Run initial crawl (expect 4-8 hours for ~7,600 days)
- [ ] Monitor progress via checkpoint.json
- [ ] Validate record counts per year
- [ ] Check for data quality issues (holidays should have no data)
- [ ] Verify Parquet partitions created correctly
- [ ] Test DuckDB queries on Parquet data
- [ ] Create data backup

---

## Phase 4: MCP Server

### 4.1 Implement Server

See `04_mcp_server_design.md` for full implementation details.

### 4.2 Checklist

- [ ] Create server.py with MCP server setup
- [ ] Implement get_current_price tool
- [ ] Implement get_historical_price tool
- [ ] Implement get_price_trend tool
- [ ] Implement list_fish_species tool
- [ ] Implement compare_prices tool
- [ ] Add error handling
- [ ] Write integration tests
- [ ] Test with Claude Desktop locally

---

## Phase 5: Prediction System

> **Full documentation**: See [`06_prediction_system.md`](./06_prediction_system.md) for complete architecture, models, database schema, and implementation details.

### 5.1 Overview

| Component | Description |
|-----------|-------------|
| **Models** | Exponential Smoothing (1-7d), ARIMA (7-30d), Prophet (seasonal) |
| **Features** | Forecasting, seasonality, anomaly detection, volatility, insights |
| **Libraries** | statsmodels, prophet, sklearn, scipy |
| **Optimization** | Batch processing, garbage collection, top-30 species limit |

### 5.2 Additional Dependencies

Add to `requirements.txt`:

```txt
# Statistical/ML Libraries
statsmodels>=0.14.0
prophet>=1.1.0
scikit-learn>=1.3.0
scipy>=1.11.0
pandas>=2.0.0
numpy>=1.24.0
```

### 5.3 Project Structure (Prediction Module)

```
src/
└── prediction/
    ├── __init__.py
    ├── pipeline.py          # Main prediction pipeline
    ├── models/
    │   ├── exponential.py   # Exponential smoothing (short-term)
    │   ├── arima.py         # ARIMA (medium-term)
    │   ├── prophet_model.py # Prophet (long-term/seasonal)
    │   └── anomaly.py       # Anomaly detection
    ├── features/
    │   ├── trend.py         # Trend analysis
    │   ├── volatility.py    # Volatility calculation
    │   └── seasonality.py   # Seasonal decomposition
    └── insights/
        └── generator.py     # Market insight generation
```

### 5.4 New MCP Tools

| Tool | Purpose |
|------|---------|
| `predict_price` | Price forecasts (1d, 7d, 14d, 30d) with confidence intervals |
| `get_seasonality` | Monthly seasonal patterns and best buying periods |
| `detect_anomalies` | Unusual price movements (Z-score + Isolation Forest) |
| `get_volatility` | Price stability metrics (7d, 14d, 30d windows) |
| `get_market_insight` | AI-generated buy/sell recommendations |

### 5.5 Checklist

- [ ] Install ML dependencies (statsmodels, prophet, sklearn, scipy)
- [ ] Create prediction module structure
- [ ] Implement Exponential Smoothing predictor
- [ ] Implement ARIMA predictor
- [ ] Implement Prophet predictor (with memory optimization)
- [ ] Implement anomaly detection (Z-score + Isolation Forest)
- [ ] Implement volatility calculator
- [ ] Implement trend analyzer
- [ ] Implement insight generator
- [ ] Create prediction MCP tools (5 new tools)
- [ ] Integrate with daily update script
- [ ] Test on historical data
- [ ] Validate prediction accuracy (backtest)
- [ ] Memory profiling on t4-micro equivalent

---

## Phase 6: Security (Anti-Scraping)

> **Full documentation**: See [`07_security.md`](./07_security.md) for complete security architecture, rate limiting, authentication, and audit logging details.

### 6.1 Overview

Protect the MCP server from data scraping when exposed publicly:

| Protection Layer | Description | Priority |
|------------------|-------------|----------|
| **Result Limits** | Max 100 records/request, 90-day date range | P0 (Required) |
| **Rate Limiting** | 30/min, 300/hr, 1,000/day per session | P0 (Required) |
| **Query Design** | Return aggregates, require filters | P0 (Required) |
| **Audit Logging** | Log all requests, detect abuse patterns | P1 (Recommended) |
| **Authentication** | API keys for elevated access | P2 (Optional) |

### 6.2 Security Module Structure

```
src/mcp_server/security/
├── __init__.py
├── limits.py           # Result set limits configuration
├── rate_limiter.py     # Session-based rate limiting
├── audit.py            # Audit logging & abuse detection
└── auth.py             # Optional API key authentication
```

### 6.3 Minimum Viable Security (MVP)

For initial deployment, implement result limits only:

```python
# src/mcp_server/security/limits.py

MAX_RECORDS_PER_REQUEST = 100
MAX_DATE_RANGE_DAYS = 90
MAX_SPECIES_LIST = 50
MIN_SEARCH_LENGTH = 2
```

Apply limits in every tool:
```python
async def get_historical_price(species, start_date, end_date, **kwargs):
    # Enforce 90-day max range
    if (end - start).days > MAX_DATE_RANGE_DAYS:
        start = end - timedelta(days=MAX_DATE_RANGE_DAYS)

    # Always use LIMIT
    query = f"... LIMIT {MAX_RECORDS_PER_REQUEST}"
```

### 6.4 Checklist

- [ ] Define result limits constants
- [ ] Apply limits to all MCP tools
- [ ] Add rate limiter middleware
- [ ] Create audit_log table in DuckDB
- [ ] Implement audit logging
- [ ] Add scraping pattern detection
- [ ] (Optional) Implement API key authentication
- [ ] Write security tests
- [ ] Document security configuration

---

## Phase 7: Deployment

### 7.1 AWS t4-micro Setup

```bash
# Launch EC2 instance
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t4g.micro \
    --key-name your-key \
    --security-groups ssh-access

# Connect and setup
ssh -i your-key.pem ec2-user@<instance-ip>

# Install dependencies
sudo yum update -y
sudo yum install python3.11 python3.11-pip git -y

# Clone repository
git clone https://github.com/your-repo/mcp_NorayngjinGavel.git
cd mcp_NorayngjinGavel

# Setup virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Initialize database
python scripts/init_db.py

# Transfer pre-crawled data (optional)
scp -i your-key.pem data/fish_market.duckdb ec2-user@<ip>:~/mcp_NorayngjinGavel/data/
scp -r -i your-key.pem data/parquet ec2-user@<ip>:~/mcp_NorayngjinGavel/data/
```

### 7.2 Daily Update Cron

```bash
# Edit crontab
crontab -e

# Add daily update job (12:00 PM KST = 03:00 UTC)
0 3 * * * /home/ec2-user/mcp_NorayngjinGavel/venv/bin/python /home/ec2-user/mcp_NorayngjinGavel/scripts/daily_update.py >> /var/log/fish_crawler.log 2>&1

# Add daily backup job (2 AM KST = 17:00 UTC previous day)
0 17 * * * /home/ec2-user/mcp_NorayngjinGavel/scripts/backup_db.sh
```

### 7.3 Systemd Service (Optional)

```ini
# /etc/systemd/system/fish-mcp.service
[Unit]
Description=Fish Price MCP Server
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/mcp_NorayngjinGavel
ExecStart=/home/ec2-user/mcp_NorayngjinGavel/venv/bin/python -m src.mcp_server.server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 7.4 Checklist

- [ ] Launch t4-micro EC2 instance
- [ ] Install Python and dependencies
- [ ] Clone repository
- [ ] Transfer database or run initial crawl
- [ ] Configure cron jobs
- [ ] Test MCP server manually
- [ ] Setup monitoring/alerting
- [ ] Document connection details

---

## Quick Start Commands

```bash
# 1. Clone and setup
git clone <repo-url>
cd mcp_NorayngjinGavel
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Initialize database (creates DuckDB + Parquet directories)
python scripts/init_db.py

# 3. Run initial crawl (takes 4-8 hours for full history)
python scripts/run_crawler.py

# 4. Verify data (summaries computed on-the-fly via DuckDB views)
python -c "import duckdb; print(duckdb.connect('data/fish_market.duckdb').execute('SELECT COUNT(*) FROM v_prices').fetchone())"

# 5. Test MCP server
python -m src.mcp_server.server

# 6. Daily update (for cron)
python scripts/daily_update.py
```

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Historical data coverage | Jan 2004 - Present (21+ years) |
| Record count | 1 - 2 million records |
| Storage size (Parquet + DuckDB) | < 65 MB |
| Daily update time | < 5 minutes |
| MCP response time | < 500ms |
| Server memory usage | < 200 MB |
| Uptime | 99%+ |
