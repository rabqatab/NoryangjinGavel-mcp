# Database Design: Fish Price MCP (Parquet + DuckDB)

## Target Environment: AWS t4-micro

### Resource Constraints

| Resource | Limit | Consideration |
|----------|-------|---------------|
| vCPUs | 2 | Limited compute |
| RAM | 512 MB | Critical constraint |
| Storage | EBS (pay per GB) | Cost consideration |
| Network | Low-moderate | API latency |

### Storage Selection: **Parquet + DuckDB**

**Rationale:**

| Factor | SQLite | Parquet + DuckDB | Winner |
|--------|--------|------------------|--------|
| Storage size | ~100-120 MB | ~40-60 MB | Parquet |
| Compression | None | 50-70% | Parquet |
| Analytical queries | Good | Excellent | Parquet |
| Time-series data | Good | Excellent | Parquet |
| Memory efficiency | Good | Excellent (columnar) | Parquet |
| Write pattern | Any | Append-only | SQLite |
| Updates/Deletes | Easy | Harder | SQLite |
| t4-micro suitability | Excellent | Excellent | Tie |

**Why Parquet + DuckDB for this use case:**

1. **Append-only data**: Price records are never updated, only inserted
2. **Analytical queries**: MCP tools perform aggregations, trends, comparisons
3. **Columnar efficiency**: Querying specific columns (e.g., `price_avg`) is faster
4. **Compression**: Repeated strings (species names) and integers (prices) compress extremely well
5. **Time-series friendly**: Natural partitioning by date

---

## Storage Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA STORAGE ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  PARQUET FILES (Historical Price Data)                       │    │
│  │  ═══════════════════════════════════                        │    │
│  │                                                              │    │
│  │  data/parquet/prices/                                        │    │
│  │  ├── year=2004/                                              │    │
│  │  │   ├── month=01/data.parquet                              │    │
│  │  │   ├── month=02/data.parquet                              │    │
│  │  │   └── ...                                                 │    │
│  │  ├── year=2024/                                              │    │
│  │  │   └── ...                                                 │    │
│  │  └── year=2025/                                              │    │
│  │      └── month=01/data.parquet                              │    │
│  │                                                              │    │
│  │  • Partitioned by year/month (Hive-style)                   │    │
│  │  • ~1.5M records → ~40-60 MB compressed                     │    │
│  │  • Denormalized (species name, not ID)                      │    │
│  │  • Append-only, immutable                                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  DUCKDB DATABASE (Metadata + Predictions)                    │    │
│  │  ════════════════════════════════════                        │    │
│  │                                                              │    │
│  │  data/fish_market.duckdb (~2-5 MB)                          │    │
│  │                                                              │    │
│  │  Tables:                                                     │    │
│  │  ├── fish_species      (lookup, ~500 rows)                  │    │
│  │  ├── fish_states       (lookup, 4 rows)                     │    │
│  │  ├── origins           (lookup, ~200 rows)                  │    │
│  │  ├── packaging_types   (lookup, ~20 rows)                   │    │
│  │  ├── crawl_metadata    (tracking)                           │    │
│  │  └── prediction_*      (5 tables, daily updates)            │    │
│  │                                                              │    │
│  │  Views:                                                      │    │
│  │  └── v_prices          (queries Parquet files)              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Parquet Schema (Price Data)

### Design Decision: Denormalized

Store species/origin names directly in Parquet instead of foreign key IDs:

| Approach | Pros | Cons |
|----------|------|------|
| **Normalized (IDs)** | Smaller storage, referential integrity | Requires joins, complex queries |
| **Denormalized (names)** | Simple queries, no joins, Parquet compresses strings well | Slightly larger (mitigated by compression) |

**Choice: Denormalized** - Parquet's dictionary encoding compresses repeated strings to ~2-4 bytes each, making storage impact negligible while greatly simplifying queries.

### Parquet File Schema

```python
# Schema definition using PyArrow
import pyarrow as pa

PRICE_SCHEMA = pa.schema([
    ('trade_date', pa.date32()),              # YYYY-MM-DD
    ('species', pa.string()),                 # "고등어", "방어" (dictionary encoded)
    ('state', pa.string()),                   # "선", "활", "냉", "가공", null
    ('origin', pa.string()),                  # "부산", "일본" (dictionary encoded)
    ('spec', pa.string()),                    # "대", "중", "2미", null
    ('packaging', pa.string()),               # "kg", "box", "S/P"
    ('quantity', pa.float32()),               # 203.0, 35.3
    ('price_high', pa.int32()),               # Highest bid (KRW)
    ('price_low', pa.int32()),                # Lowest bid (KRW)
    ('price_avg', pa.int32()),                # Average price (KRW)
])
```

### Partitioning Strategy

```
data/parquet/prices/
├── year=2004/
│   ├── month=01/
│   │   └── data.parquet    # ~9,000 records (~300/day × 30 days)
│   ├── month=02/
│   │   └── data.parquet
│   └── ...
├── year=2024/
│   └── ...
└── _metadata                # Optional: unified metadata file
```

**Why year/month partitioning:**

| Partitioning | Partitions (21 years) | Records per Partition | Query Efficiency |
|--------------|----------------------|----------------------|------------------|
| By day | ~7,600 | ~300 | Too granular |
| **By month** | **~252** | **~9,000** | **Optimal** |
| By year | ~21 | ~70,000 | Too coarse |

### Example Parquet Data

| trade_date | species | state | origin | spec | packaging | quantity | price_high | price_low | price_avg |
|------------|---------|-------|--------|------|-----------|----------|------------|-----------|-----------|
| 2024-01-15 | 고등어 | 선 | 부산 | 대 | kg | 203.0 | 8500 | 7200 | 7850 |
| 2024-01-15 | 방어 | 활 | 일본 | 2미 | kg | 45.0 | 125000 | 98000 | 112000 |
| 2024-01-15 | 광어 | 활 | 제주 | 중 | kg | 120.5 | 42000 | 35000 | 38500 |

---

## DuckDB Schema (Metadata + Predictions)

### Lookup Tables

```sql
-- Fish Species Lookup
CREATE TABLE fish_species (
    id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE,           -- Korean name: "고등어", "방어"
    name_en VARCHAR,                        -- English name: "Mackerel"
    category VARCHAR,                       -- "fish", "shellfish", "crustacean"
    created_at TIMESTAMP DEFAULT now()
);

-- Fish State Lookup
CREATE TABLE fish_states (
    id INTEGER PRIMARY KEY,
    code VARCHAR NOT NULL UNIQUE,           -- "선", "활", "냉", "가공"
    name VARCHAR NOT NULL,                  -- "선어", "활어", "냉동", "가공"
    name_en VARCHAR                         -- "Fresh", "Live", "Frozen", "Processed"
);

-- Origin Locations Lookup
CREATE TABLE origins (
    id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE,           -- "부산(기장)", "제주도", "일본"
    region VARCHAR,                         -- "부산", "제주", "수입"
    is_domestic BOOLEAN DEFAULT TRUE
);

-- Packaging Types Lookup
CREATE TABLE packaging_types (
    id INTEGER PRIMARY KEY,
    code VARCHAR NOT NULL UNIQUE,           -- "kg", "S/P", "box"
    description VARCHAR                     -- "Per kilogram", "Styrofoam package"
);

-- Crawl Metadata
CREATE TABLE crawl_metadata (
    trade_date DATE PRIMARY KEY,
    record_count INTEGER NOT NULL,
    crawled_at TIMESTAMP DEFAULT now()
);
```

### View for Parquet Data

```sql
-- Create view that queries all Parquet files
CREATE OR REPLACE VIEW v_prices AS
SELECT
    trade_date,
    species,
    state,
    origin,
    spec,
    packaging,
    quantity,
    price_high,
    price_low,
    price_avg,
    -- Extract partition columns for filtering
    YEAR(trade_date) AS year,
    MONTH(trade_date) AS month
FROM read_parquet('data/parquet/prices/**/*.parquet', hive_partitioning=true);
```

### Daily Summary View (Computed on-the-fly)

```sql
-- Materialized-style summary (created daily after crawl)
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

---

## Prediction Tables (DuckDB)

> **Pipeline & model details**: See [`06_prediction_system.md`](./06_prediction_system.md) for implementation.

```sql
-- Price Forecasts (1d, 7d, 14d, 30d horizons)
CREATE TABLE prediction_forecasts (
    id INTEGER PRIMARY KEY,
    species VARCHAR NOT NULL,               -- Direct species name (denormalized)
    base_date DATE NOT NULL,
    target_date DATE NOT NULL,
    horizon_days INTEGER NOT NULL,          -- 1, 7, 14, or 30
    predicted_price INTEGER NOT NULL,
    ci_lower INTEGER NOT NULL,              -- 80% CI lower bound
    ci_upper INTEGER NOT NULL,              -- 80% CI upper bound
    model_type VARCHAR NOT NULL,            -- 'exp_smoothing', 'arima', 'prophet'
    created_at TIMESTAMP DEFAULT now(),
    UNIQUE(species, base_date, horizon_days, model_type)
);

-- Seasonality Patterns (monthly)
CREATE TABLE prediction_seasonality (
    id INTEGER PRIMARY KEY,
    species VARCHAR NOT NULL,
    month INTEGER NOT NULL,                 -- 1-12
    seasonal_index DOUBLE NOT NULL,
    avg_price INTEGER NOT NULL,
    best_week INTEGER,
    price_trend VARCHAR,                    -- 'rising', 'falling', 'stable'
    updated_at TIMESTAMP DEFAULT now(),
    UNIQUE(species, month)
);

-- Detected Anomalies
CREATE TABLE prediction_anomalies (
    id INTEGER PRIMARY KEY,
    species VARCHAR NOT NULL,
    detected_date DATE NOT NULL,
    actual_price INTEGER NOT NULL,
    expected_price INTEGER NOT NULL,
    z_score DOUBLE NOT NULL,
    severity VARCHAR NOT NULL,              -- 'low', 'medium', 'high'
    anomaly_type VARCHAR,
    possible_cause VARCHAR,
    created_at TIMESTAMP DEFAULT now()
);

-- Market Insights
CREATE TABLE prediction_insights (
    id INTEGER PRIMARY KEY,
    species VARCHAR NOT NULL,
    insight_date DATE NOT NULL,
    trend_direction VARCHAR NOT NULL,       -- 'up', 'down', 'stable'
    trend_strength DOUBLE NOT NULL,
    volatility_index DOUBLE NOT NULL,
    volatility_label VARCHAR NOT NULL,
    recommendation VARCHAR NOT NULL,        -- 'buy', 'sell', 'hold', 'wait'
    confidence DOUBLE NOT NULL,
    summary_text VARCHAR NOT NULL,
    factors VARCHAR,                        -- JSON array
    created_at TIMESTAMP DEFAULT now(),
    UNIQUE(species, insight_date)
);

-- Model Accuracy Tracking
CREATE TABLE prediction_model_accuracy (
    id INTEGER PRIMARY KEY,
    species VARCHAR NOT NULL,
    model_type VARCHAR NOT NULL,
    horizon_days INTEGER NOT NULL,
    mae DOUBLE NOT NULL,
    mape DOUBLE NOT NULL,
    rmse DOUBLE NOT NULL,
    sample_size INTEGER NOT NULL,
    evaluation_date DATE NOT NULL,
    UNIQUE(species, model_type, horizon_days)
);
```

### Prediction Indexes

```sql
-- DuckDB automatically creates indexes for UNIQUE constraints
-- Additional indexes for common query patterns:
CREATE INDEX idx_forecast_species ON prediction_forecasts(species);
CREATE INDEX idx_anomaly_date ON prediction_anomalies(detected_date);
CREATE INDEX idx_insight_species ON prediction_insights(species);
```

---

## Storage Estimation

### Parquet Storage (Price Data)

```
┌─────────────────────────────────────────────────────────────────┐
│  PARQUET STORAGE ESTIMATION                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw data size (uncompressed):                                  │
│  • ~1.5 million records                                         │
│  • ~80 bytes per record (with strings)                          │
│  • Total: ~120 MB                                               │
│                                                                  │
│  Parquet compression factors:                                   │
│  ├─ Dictionary encoding (species, origin): 90% reduction       │
│  ├─ Run-length encoding (repeated dates): 80% reduction        │
│  ├─ Delta encoding (prices): 60% reduction                     │
│  └─ Snappy compression: additional 30% reduction               │
│                                                                  │
│  Expected Parquet size: ~40-60 MB (50-65% compression)          │
│                                                                  │
│  Per-partition size:                                            │
│  • ~9,000 records/month                                         │
│  • ~150-250 KB per partition                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### DuckDB Storage (Metadata + Predictions)

| Table | Rows | Est. Size |
|-------|------|-----------|
| `fish_species` | ~500 | ~50 KB |
| `fish_states` | 4 | <1 KB |
| `origins` | ~200 | ~20 KB |
| `packaging_types` | ~20 | ~2 KB |
| `crawl_metadata` | ~7,600 | ~150 KB |
| `prediction_forecasts` | ~400 | ~50 KB |
| `prediction_seasonality` | ~400 | ~20 KB |
| `prediction_anomalies` | ~500 | ~30 KB |
| `prediction_insights` | ~1,000 | ~200 KB |
| `prediction_model_accuracy` | ~400 | ~20 KB |
| **Total DuckDB** | | **~2-3 MB** |

### Total Storage

| Component | Size |
|-----------|------|
| Parquet files | ~40-60 MB |
| DuckDB database | ~2-3 MB |
| **Total** | **~45-65 MB** |

**Comparison with SQLite approach: ~50% smaller**

---

## Query Patterns & Examples

### 1. Get Current Day Prices

```sql
SELECT
    species,
    state,
    origin,
    spec,
    packaging,
    quantity,
    price_high,
    price_low,
    price_avg
FROM v_prices
WHERE trade_date = CURRENT_DATE - INTERVAL 1 DAY
ORDER BY species, origin;
```

### 2. Get Price History for Species (Last 30 Days)

```sql
SELECT
    trade_date,
    state,
    origin,
    spec,
    quantity,
    price_avg
FROM v_prices
WHERE species = '고등어'
  AND trade_date >= CURRENT_DATE - INTERVAL 30 DAY
ORDER BY trade_date DESC;
```

### 3. Price Trend with Moving Average

```sql
SELECT
    trade_date,
    price_avg,
    AVG(price_avg) OVER (
        ORDER BY trade_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS ma_7d
FROM v_prices
WHERE species = '방어'
  AND trade_date >= CURRENT_DATE - INTERVAL 90 DAY
ORDER BY trade_date;
```

### 4. Compare Prices Across Origins

```sql
SELECT
    origin,
    AVG(price_avg) AS avg_price,
    SUM(quantity) AS total_quantity,
    COUNT(*) AS record_count
FROM v_prices
WHERE species = '광어'
  AND trade_date >= CURRENT_DATE - INTERVAL 30 DAY
GROUP BY origin
ORDER BY avg_price DESC;
```

### 5. Monthly Seasonal Analysis

```sql
SELECT
    MONTH(trade_date) AS month,
    AVG(price_avg) AS avg_price,
    STDDEV(price_avg) AS price_stddev,
    COUNT(*) AS records
FROM v_prices
WHERE species = '고등어'
  AND trade_date >= CURRENT_DATE - INTERVAL 3 YEAR
GROUP BY MONTH(trade_date)
ORDER BY month;
```

### 6. Year-over-Year Comparison

```sql
WITH yearly AS (
    SELECT
        YEAR(trade_date) AS year,
        AVG(price_avg) AS avg_price
    FROM v_prices
    WHERE species = '참치'
    GROUP BY YEAR(trade_date)
)
SELECT
    year,
    avg_price,
    LAG(avg_price) OVER (ORDER BY year) AS prev_year,
    ROUND((avg_price - LAG(avg_price) OVER (ORDER BY year)) /
          LAG(avg_price) OVER (ORDER BY year) * 100, 2) AS yoy_change_pct
FROM yearly
ORDER BY year DESC
LIMIT 10;
```

### 7. Top Species by Volume (Last Month)

```sql
SELECT
    species,
    SUM(quantity) AS total_quantity,
    AVG(price_avg) AS avg_price,
    COUNT(*) AS records
FROM v_prices
WHERE trade_date >= CURRENT_DATE - INTERVAL 30 DAY
GROUP BY species
ORDER BY total_quantity DESC
LIMIT 20;
```

---

## Data Writing (Parquet)

### Writing Daily Data

```python
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import date

def write_daily_prices(records: list[dict], trade_date: date):
    """Write daily price records to partitioned Parquet."""

    # Convert to PyArrow Table
    table = pa.Table.from_pylist(records, schema=PRICE_SCHEMA)

    # Determine partition path
    partition_path = Path(f"data/parquet/prices/year={trade_date.year}/month={trade_date.month:02d}")
    partition_path.mkdir(parents=True, exist_ok=True)

    # Write or append to partition
    output_file = partition_path / "data.parquet"

    if output_file.exists():
        # Read existing and concatenate
        existing = pq.read_table(output_file)
        combined = pa.concat_tables([existing, table])
        pq.write_table(combined, output_file, compression='snappy')
    else:
        pq.write_table(table, output_file, compression='snappy')
```

### Monthly Compaction (Optional)

```python
def compact_month(year: int, month: int):
    """Compact all daily writes into optimized monthly file."""
    import duckdb

    partition_path = f"data/parquet/prices/year={year}/month={month:02d}"

    # Read all data for the month and rewrite optimized
    conn = duckdb.connect()
    conn.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{partition_path}/*.parquet')
            ORDER BY trade_date, species
        ) TO '{partition_path}/data_compacted.parquet'
        (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000)
    """)

    # Replace old files with compacted version
    # (cleanup logic here)
```

---

## Database Connection

### Python Connection Helper

```python
import duckdb
from pathlib import Path
from contextlib import contextmanager

DATA_DIR = Path("data")
PARQUET_DIR = DATA_DIR / "parquet" / "prices"
DUCKDB_PATH = DATA_DIR / "fish_market.duckdb"

@contextmanager
def get_db_connection(read_only: bool = True):
    """Context manager for DuckDB connections."""
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=read_only)

    # Register Parquet view
    conn.execute(f"""
        CREATE OR REPLACE VIEW v_prices AS
        SELECT * FROM read_parquet('{PARQUET_DIR}/**/*.parquet', hive_partitioning=true)
    """)

    try:
        yield conn
    finally:
        conn.close()

# Usage
with get_db_connection() as conn:
    result = conn.execute("""
        SELECT species, AVG(price_avg) as avg_price
        FROM v_prices
        WHERE trade_date >= CURRENT_DATE - INTERVAL 7 DAY
        GROUP BY species
        ORDER BY avg_price DESC
        LIMIT 10
    """).fetchdf()
```

### Async Support

```python
# DuckDB operations are fast enough that async wrapper is optional
# For true async, use run_in_executor:

import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

async def async_query(sql: str):
    """Run DuckDB query asynchronously."""
    loop = asyncio.get_event_loop()

    def _query():
        with get_db_connection() as conn:
            return conn.execute(sql).fetchdf()

    return await loop.run_in_executor(executor, _query)
```

---

## Database Initialization

### init_db.py

```python
#!/usr/bin/env python3
"""Initialize DuckDB database with schema."""

import duckdb
from pathlib import Path

DATA_DIR = Path("data")
PARQUET_DIR = DATA_DIR / "parquet" / "prices"
DUCKDB_PATH = DATA_DIR / "fish_market.duckdb"

def init_database():
    """Create database and tables."""

    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    # Connect and create schema
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

    # Create prediction tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_forecasts (
            id INTEGER PRIMARY KEY,
            species VARCHAR NOT NULL,
            base_date DATE NOT NULL,
            target_date DATE NOT NULL,
            horizon_days INTEGER NOT NULL,
            predicted_price INTEGER NOT NULL,
            ci_lower INTEGER NOT NULL,
            ci_upper INTEGER NOT NULL,
            model_type VARCHAR NOT NULL,
            created_at TIMESTAMP DEFAULT now(),
            UNIQUE(species, base_date, horizon_days, model_type)
        );

        CREATE TABLE IF NOT EXISTS prediction_seasonality (
            id INTEGER PRIMARY KEY,
            species VARCHAR NOT NULL,
            month INTEGER NOT NULL,
            seasonal_index DOUBLE NOT NULL,
            avg_price INTEGER NOT NULL,
            best_week INTEGER,
            price_trend VARCHAR,
            updated_at TIMESTAMP DEFAULT now(),
            UNIQUE(species, month)
        );

        CREATE TABLE IF NOT EXISTS prediction_anomalies (
            id INTEGER PRIMARY KEY,
            species VARCHAR NOT NULL,
            detected_date DATE NOT NULL,
            actual_price INTEGER NOT NULL,
            expected_price INTEGER NOT NULL,
            z_score DOUBLE NOT NULL,
            severity VARCHAR NOT NULL,
            anomaly_type VARCHAR,
            possible_cause VARCHAR,
            created_at TIMESTAMP DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS prediction_insights (
            id INTEGER PRIMARY KEY,
            species VARCHAR NOT NULL,
            insight_date DATE NOT NULL,
            trend_direction VARCHAR NOT NULL,
            trend_strength DOUBLE NOT NULL,
            volatility_index DOUBLE NOT NULL,
            volatility_label VARCHAR NOT NULL,
            recommendation VARCHAR NOT NULL,
            confidence DOUBLE NOT NULL,
            summary_text VARCHAR NOT NULL,
            factors VARCHAR,
            created_at TIMESTAMP DEFAULT now(),
            UNIQUE(species, insight_date)
        );

        CREATE TABLE IF NOT EXISTS prediction_model_accuracy (
            id INTEGER PRIMARY KEY,
            species VARCHAR NOT NULL,
            model_type VARCHAR NOT NULL,
            horizon_days INTEGER NOT NULL,
            mae DOUBLE NOT NULL,
            mape DOUBLE NOT NULL,
            rmse DOUBLE NOT NULL,
            sample_size INTEGER NOT NULL,
            evaluation_date DATE NOT NULL,
            UNIQUE(species, model_type, horizon_days)
        );
    """)

    # Seed lookup data
    seed_lookup_tables(conn)

    # Create Parquet view
    conn.execute(f"""
        CREATE OR REPLACE VIEW v_prices AS
        SELECT * FROM read_parquet('{PARQUET_DIR}/**/*.parquet', hive_partitioning=true)
    """)

    conn.close()
    print(f"Database initialized at: {DUCKDB_PATH}")

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
            (4, 'CT/(BT)', 'Carton/Basket'),
            (5, 'C/S', 'Case/Box'),
            (6, 'PAN(펜)', 'Pan')
    """)

if __name__ == "__main__":
    init_database()
```

---

## Backup Strategy

### Backup Script

```bash
#!/bin/bash
# backup_data.sh

DATA_DIR="/path/to/data"
BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d)

# Backup DuckDB (small, contains metadata + predictions)
cp "$DATA_DIR/fish_market.duckdb" "$BACKUP_DIR/fish_market_$DATE.duckdb"
gzip "$BACKUP_DIR/fish_market_$DATE.duckdb"

# Backup Parquet (optional - can be regenerated from raw JSON)
# Only backup recent partitions
tar -czf "$BACKUP_DIR/parquet_recent_$DATE.tar.gz" \
    "$DATA_DIR/parquet/prices/year=$(date +%Y)"

# Keep only last 7 days
find "$BACKUP_DIR" -name "*.duckdb.gz" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
```

---

## Performance Optimization

### DuckDB Settings

```python
def get_optimized_connection():
    """Get DuckDB connection with optimized settings for t4-micro."""
    conn = duckdb.connect(str(DUCKDB_PATH))

    # Memory limit for t4-micro (use ~256MB of 512MB)
    conn.execute("SET memory_limit='256MB'")

    # Thread count (t4-micro has 2 vCPUs)
    conn.execute("SET threads=2")

    # Enable progress bar for long queries (optional)
    conn.execute("SET enable_progress_bar=true")

    return conn
```

### Query Optimization Tips

1. **Use partition pruning**: Always filter by `trade_date` when possible
   ```sql
   -- Good: DuckDB only reads relevant partitions
   WHERE trade_date >= '2024-01-01'

   -- Less efficient: Reads all partitions
   WHERE YEAR(trade_date) = 2024
   ```

2. **Project only needed columns**: Parquet is columnar
   ```sql
   -- Good: Only reads price_avg column
   SELECT AVG(price_avg) FROM v_prices WHERE ...

   -- Less efficient: Reads all columns
   SELECT * FROM v_prices WHERE ...
   ```

3. **Use aggregate pushdown**: DuckDB pushes aggregations to Parquet reader
   ```sql
   -- Efficient: Aggregation happens during scan
   SELECT species, AVG(price_avg) FROM v_prices GROUP BY species
   ```

---

## Migration from SQLite (If Applicable)

```python
def migrate_sqlite_to_parquet(sqlite_path: str):
    """Migrate existing SQLite data to Parquet + DuckDB."""
    import sqlite3
    import pandas as pd

    # Read from SQLite
    sqlite_conn = sqlite3.connect(sqlite_path)
    df = pd.read_sql("""
        SELECT
            date(trade_date, 'unixepoch') as trade_date,
            fs.name as species,
            fst.code as state,
            o.name as origin,
            fp.spec,
            pt.code as packaging,
            fp.quantity,
            fp.price_high,
            fp.price_low,
            fp.price_avg
        FROM fish_prices fp
        JOIN fish_species fs ON fp.species_id = fs.id
        LEFT JOIN fish_states fst ON fp.state_id = fst.id
        JOIN origins o ON fp.origin_id = o.id
        LEFT JOIN packaging_types pt ON fp.packaging_id = pt.id
    """, sqlite_conn)

    # Convert and write to Parquet (partitioned)
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    for (year, month), group in df.groupby([df['trade_date'].dt.year, df['trade_date'].dt.month]):
        partition_path = PARQUET_DIR / f"year={year}" / f"month={month:02d}"
        partition_path.mkdir(parents=True, exist_ok=True)
        group.to_parquet(partition_path / "data.parquet", compression='snappy')

    print(f"Migrated {len(df)} records to Parquet")
```

---

## Monitoring Queries

```sql
-- Database size
SELECT
    'DuckDB' as component,
    pg_size_pretty(database_size) as size
FROM pragma_database_size();

-- Parquet statistics (requires scanning metadata)
SELECT
    COUNT(*) as total_files,
    SUM(file_size) as total_bytes
FROM glob('data/parquet/prices/**/*.parquet');

-- Record counts by year
SELECT
    YEAR(trade_date) as year,
    COUNT(*) as records
FROM v_prices
GROUP BY YEAR(trade_date)
ORDER BY year;

-- Recent crawl activity
SELECT
    trade_date,
    record_count,
    crawled_at
FROM crawl_metadata
ORDER BY trade_date DESC
LIMIT 10;
```
