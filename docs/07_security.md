# MCP Server Security: Anti-Scraping Protection

## Overview

This document describes security measures to prevent unauthorized bulk data extraction (scraping) when the MCP server is exposed publicly. The goal is to make the server useful for AI assistants while impractical for systematic database crawling.

### Threat Model

| Attack Vector | Description | Risk Level |
|---------------|-------------|------------|
| Date Iteration | Query each trading day (2004-2025 = ~7,500 days) | High |
| Species Enumeration | Get species list, then query full history for each | High |
| Wide Date Ranges | Single query for 20+ years of data | High |
| Automated Scraping | Scripts systematically calling all tools | High |
| Expensive Queries | Resource-exhaustion attacks on t4-micro | Medium |

---

## Protection Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP SERVER SECURITY LAYERS                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 1: RESULT LIMITS (Must Have)                             │
│    └─ Max 100 records per request                               │
│    └─ Max 90-day date range                                     │
│    └─ No wildcard/bulk queries                                  │
│                                                                  │
│  Layer 2: RATE LIMITING (Must Have)                             │
│    └─ 30 requests/minute                                        │
│    └─ 300 requests/hour                                         │
│    └─ 1,000 requests/day                                        │
│                                                                  │
│  Layer 3: QUERY DESIGN (Must Have)                              │
│    └─ Return aggregates over raw data                           │
│    └─ Semantic tools, not data dumps                            │
│    └─ No pagination with total count                            │
│                                                                  │
│  Layer 4: AUTHENTICATION (Optional)                             │
│    └─ API keys for elevated limits                              │
│    └─ Tiered access levels                                      │
│                                                                  │
│  Layer 5: AUDIT LOGGING (Recommended)                           │
│    └─ Log all requests                                          │
│    └─ Detect scraping patterns                                  │
│    └─ Auto-block suspicious sessions                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Result Limits

Hard caps on data returned per request. **This is the most critical defense.**

### Configuration

```python
# src/mcp_server/security/limits.py

from dataclasses import dataclass

@dataclass
class ResultLimits:
    """Hard limits on query results."""

    # Maximum records per single request
    MAX_RECORDS_PER_REQUEST: int = 100

    # Maximum date range for historical queries (days)
    MAX_DATE_RANGE_DAYS: int = 90

    # Maximum species in list response
    MAX_SPECIES_LIST: int = 50

    # Minimum search term length
    MIN_SEARCH_LENGTH: int = 2

    # Maximum comparison groups
    MAX_COMPARISON_GROUPS: int = 10


# Singleton instance
LIMITS = ResultLimits()
```

### Implementation in Tools

```python
# src/mcp_server/tools.py

from datetime import datetime, timedelta
from .security.limits import LIMITS

async def get_historical_price(
    species: str,
    start_date: str,
    end_date: str,
    **kwargs
) -> dict:
    """Get historical prices with enforced limits."""

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Enforce date range limit
    date_range = (end - start).days
    if date_range > LIMITS.MAX_DATE_RANGE_DAYS:
        # Truncate to most recent N days instead of error
        start = end - timedelta(days=LIMITS.MAX_DATE_RANGE_DAYS)
        truncated = True
    else:
        truncated = False

    # Query with LIMIT
    query = """
        SELECT trade_date, species, price_avg, price_high, price_low
        FROM v_prices
        WHERE species = $1 AND trade_date BETWEEN $2 AND $3
        ORDER BY trade_date DESC
        LIMIT $4
    """

    with get_db_connection() as conn:
        result = conn.execute(query, [
            species, start, end, LIMITS.MAX_RECORDS_PER_REQUEST
        ]).fetchdf()

    return {
        "species": species,
        "period": {"start": str(start), "end": str(end)},
        "truncated": truncated,
        "limit_applied": LIMITS.MAX_RECORDS_PER_REQUEST,
        "records_returned": len(result),
        "data": result.to_dict('records')
    }


async def list_fish_species(
    category: str = "all",
    search: str = None,
    **kwargs
) -> dict:
    """List species with enforced limits."""

    # Enforce minimum search length
    if search and len(search) < LIMITS.MIN_SEARCH_LENGTH:
        return {
            "error": "SEARCH_TOO_SHORT",
            "message": f"Search term must be at least {LIMITS.MIN_SEARCH_LENGTH} characters"
        }

    query = "SELECT name, name_en, category FROM fish_species"
    params = []

    if category != "all":
        query += " WHERE category = $1"
        params.append(category)

    if search:
        if params:
            query += f" AND name LIKE '%' || ${len(params) + 1} || '%'"
        else:
            query += " WHERE name LIKE '%' || $1 || '%'"
        params.append(search)

    # Always limit results
    query += f" ORDER BY name LIMIT {LIMITS.MAX_SPECIES_LIST}"

    with get_db_connection() as conn:
        result = conn.execute(query, params).fetchdf()

    return {
        "total_returned": len(result),
        "limit_applied": LIMITS.MAX_SPECIES_LIST,
        "species": result.to_dict('records')
    }
```

### Impact Analysis

With these limits in place, extracting the full database becomes impractical:

| Scenario | Without Limits | With Limits |
|----------|----------------|-------------|
| All prices for 1 species (20 years) | 1 query, ~7,500 records | 84 queries (90-day chunks) |
| All species history | ~76 queries | ~6,384 queries |
| Full database extraction | ~100 queries | Thousands of requests + rate limits |

---

## Layer 2: Rate Limiting

Session-based rate limiting to prevent rapid automated queries.

### Implementation

```python
# src/mcp_server/security/rate_limiter.py

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class RateLimitConfig:
    """Rate limit thresholds."""
    REQUESTS_PER_MINUTE: int = 30
    REQUESTS_PER_HOUR: int = 300
    REQUESTS_PER_DAY: int = 1000
    BLOCK_DURATION_SECONDS: int = 3600  # 1 hour block on violation


class RateLimiter:
    """In-memory rate limiter for MCP sessions."""

    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.blocked_until: Dict[str, float] = {}

    def _cleanup_old_requests(self, session_id: str, now: float):
        """Remove requests older than 24 hours."""
        cutoff = now - 86400
        self.requests[session_id] = [
            t for t in self.requests[session_id] if t > cutoff
        ]

    def check_rate_limit(self, session_id: str) -> tuple[bool, str]:
        """
        Check if request is allowed.

        Returns:
            (allowed: bool, reason: str)
        """
        now = time.time()

        # Check if session is blocked
        if session_id in self.blocked_until:
            if now < self.blocked_until[session_id]:
                remaining = int(self.blocked_until[session_id] - now)
                return False, f"RATE_BLOCKED: Try again in {remaining} seconds"
            else:
                del self.blocked_until[session_id]

        self._cleanup_old_requests(session_id, now)
        requests = self.requests[session_id]

        # Count requests in each window
        minute_count = sum(1 for t in requests if now - t < 60)
        hour_count = sum(1 for t in requests if now - t < 3600)
        day_count = len(requests)

        # Check limits
        if minute_count >= self.config.REQUESTS_PER_MINUTE:
            self.blocked_until[session_id] = now + 60
            return False, f"RATE_LIMIT_MINUTE: Max {self.config.REQUESTS_PER_MINUTE}/min exceeded"

        if hour_count >= self.config.REQUESTS_PER_HOUR:
            self.blocked_until[session_id] = now + self.config.BLOCK_DURATION_SECONDS
            return False, f"RATE_LIMIT_HOUR: Max {self.config.REQUESTS_PER_HOUR}/hour exceeded"

        if day_count >= self.config.REQUESTS_PER_DAY:
            # Block until next day
            self.blocked_until[session_id] = now + 86400
            return False, f"RATE_LIMIT_DAY: Max {self.config.REQUESTS_PER_DAY}/day exceeded"

        # Record request
        self.requests[session_id].append(now)
        return True, "OK"

    def get_remaining(self, session_id: str) -> dict:
        """Get remaining quota for a session."""
        now = time.time()
        self._cleanup_old_requests(session_id, now)
        requests = self.requests[session_id]

        minute_count = sum(1 for t in requests if now - t < 60)
        hour_count = sum(1 for t in requests if now - t < 3600)
        day_count = len(requests)

        return {
            "minute": self.config.REQUESTS_PER_MINUTE - minute_count,
            "hour": self.config.REQUESTS_PER_HOUR - hour_count,
            "day": self.config.REQUESTS_PER_DAY - day_count
        }


# Global instance
rate_limiter = RateLimiter()
```

### Integration with MCP Server

```python
# src/mcp_server/server.py

from .security.rate_limiter import rate_limiter

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    # Get session ID from context (MCP provides this)
    session_id = get_current_session_id()  # Implementation varies by MCP SDK

    # Check rate limit
    allowed, reason = rate_limiter.check_rate_limit(session_id)
    if not allowed:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": True,
                "code": "RATE_LIMITED",
                "message": reason,
                "remaining": rate_limiter.get_remaining(session_id)
            })
        )]

    # Proceed with tool execution
    # ... rest of implementation
```

---

## Layer 3: Query Design

Design tools to return useful information without enabling bulk extraction.

### Principles

1. **Aggregation over raw data**: Return statistics, not row dumps
2. **Semantic queries**: Answer questions, don't export data
3. **No total counts**: Don't reveal database size
4. **Mandatory filters**: Require species name, no wildcards

### Example: Safe Tool Design

```python
# UNSAFE: Returns raw records (enables scraping)
async def get_all_prices(species: str, start: str, end: str) -> dict:
    return {
        "data": [...thousands of raw price records...]
    }

# SAFE: Returns analysis with limited samples
async def get_price_analysis(species: str, year: int) -> dict:
    """Analyze prices for a species in a given year."""

    with get_db_connection() as conn:
        # Aggregated stats only
        stats = conn.execute("""
            SELECT
                AVG(price_avg) as avg_price,
                MIN(price_low) as min_price,
                MAX(price_high) as max_price,
                STDDEV(price_avg) as price_volatility,
                COUNT(DISTINCT trade_date) as trading_days
            FROM v_prices
            WHERE species = $1
              AND EXTRACT(YEAR FROM trade_date) = $2
        """, [species, year]).fetchone()

        # Only 3 sample points (not full data)
        samples = conn.execute("""
            SELECT trade_date, price_avg
            FROM v_prices
            WHERE species = $1
              AND EXTRACT(YEAR FROM trade_date) = $2
              AND EXTRACT(DAY FROM trade_date) = 15  -- Mid-month only
            ORDER BY trade_date
            LIMIT 3
        """, [species, year]).fetchdf()

    # Determine trend from samples
    prices = samples['price_avg'].tolist()
    if len(prices) >= 2:
        trend = "increasing" if prices[-1] > prices[0] else "decreasing"
    else:
        trend = "insufficient_data"

    return {
        "species": species,
        "year": year,
        "statistics": {
            "avg_price": int(stats[0]),
            "min_price": int(stats[1]),
            "max_price": int(stats[2]),
            "volatility": round(stats[3], 2) if stats[3] else None,
            "trading_days": int(stats[4])
        },
        "trend": trend,
        "sample_dates": samples['trade_date'].tolist()[:3],
        "sample_prices": [int(p) for p in prices[:3]]
    }
```

### Tool Comparison

| Tool Type | Data Exposure | Use Case |
|-----------|---------------|----------|
| `get_prices(date)` | Single day (~100 records) | OK - bounded |
| `get_price_history(90 days)` | ~90 records with aggregation | OK - limited |
| `get_price_analysis()` | Stats + 3 samples | IDEAL - minimal exposure |
| `export_all()` | Everything | NEVER implement |

---

## Layer 4: Authentication (Optional)

Tiered access for different user types.

### Access Levels

```python
# src/mcp_server/security/auth.py

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import hashlib
import secrets

class AuthLevel(Enum):
    ANONYMOUS = "anonymous"
    REGISTERED = "registered"
    PREMIUM = "premium"
    ADMIN = "admin"


@dataclass
class AccessConfig:
    """Configuration per access level."""
    max_results: int
    max_date_range_days: int
    requests_per_hour: int
    can_access_predictions: bool
    can_access_raw_data: bool


AUTH_CONFIGS = {
    AuthLevel.ANONYMOUS: AccessConfig(
        max_results=50,
        max_date_range_days=30,
        requests_per_hour=100,
        can_access_predictions=False,
        can_access_raw_data=False
    ),
    AuthLevel.REGISTERED: AccessConfig(
        max_results=200,
        max_date_range_days=180,
        requests_per_hour=500,
        can_access_predictions=True,
        can_access_raw_data=False
    ),
    AuthLevel.PREMIUM: AccessConfig(
        max_results=1000,
        max_date_range_days=365,
        requests_per_hour=2000,
        can_access_predictions=True,
        can_access_raw_data=True
    ),
    AuthLevel.ADMIN: AccessConfig(
        max_results=10000,
        max_date_range_days=36500,  # 100 years
        requests_per_hour=10000,
        can_access_predictions=True,
        can_access_raw_data=True
    )
}


class APIKeyManager:
    """Simple API key authentication."""

    def __init__(self, duckdb_path: str):
        self.duckdb_path = duckdb_path
        self._init_table()

    def _init_table(self):
        """Create API keys table if not exists."""
        import duckdb
        with duckdb.connect(self.duckdb_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash VARCHAR PRIMARY KEY,
                    auth_level VARCHAR NOT NULL,
                    owner_name VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)

    def generate_key(self, auth_level: AuthLevel, owner_name: str) -> str:
        """Generate new API key."""
        key = f"fpm_{secrets.token_urlsafe(32)}"  # fpm = fish price mcp
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        import duckdb
        with duckdb.connect(self.duckdb_path) as conn:
            conn.execute("""
                INSERT INTO api_keys (key_hash, auth_level, owner_name)
                VALUES ($1, $2, $3)
            """, [key_hash, auth_level.value, owner_name])

        return key  # Return unhashed key to user (only time it's visible)

    def validate_key(self, api_key: Optional[str]) -> tuple[AuthLevel, AccessConfig]:
        """Validate API key and return access config."""

        if not api_key:
            return AuthLevel.ANONYMOUS, AUTH_CONFIGS[AuthLevel.ANONYMOUS]

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        import duckdb
        with duckdb.connect(self.duckdb_path) as conn:
            result = conn.execute("""
                SELECT auth_level FROM api_keys
                WHERE key_hash = $1
                  AND is_active = TRUE
                  AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """, [key_hash]).fetchone()

        if result:
            level = AuthLevel(result[0])
            return level, AUTH_CONFIGS[level]

        # Invalid key - treat as anonymous but log suspicious activity
        return AuthLevel.ANONYMOUS, AUTH_CONFIGS[AuthLevel.ANONYMOUS]
```

### Tool Schema with API Key

```json
{
    "name": "get_historical_price",
    "inputSchema": {
        "properties": {
            "species": {"type": "string"},
            "start_date": {"type": "string", "format": "date"},
            "end_date": {"type": "string", "format": "date"},
            "api_key": {
                "type": "string",
                "description": "Optional API key for elevated access limits"
            }
        },
        "required": ["species", "start_date", "end_date"]
    }
}
```

---

## Layer 5: Audit Logging

Track all requests for analysis and abuse detection.

### Database Schema

```sql
-- In DuckDB: fish_market.duckdb

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR,
    tool_name VARCHAR NOT NULL,
    params JSON,
    result_count INTEGER,
    execution_ms INTEGER,
    auth_level VARCHAR DEFAULT 'anonymous',
    ip_hint VARCHAR,  -- Partial IP or hash for pattern detection
    flagged BOOLEAN DEFAULT FALSE,
    flag_reason VARCHAR
);

CREATE INDEX idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX idx_audit_session ON audit_log(session_id);
CREATE INDEX idx_audit_flagged ON audit_log(flagged);
```

### Logging Implementation

```python
# src/mcp_server/security/audit.py

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass
class AuditEntry:
    session_id: str
    tool_name: str
    params: dict
    result_count: int
    execution_ms: int
    auth_level: str = "anonymous"
    ip_hint: Optional[str] = None


class AuditLogger:
    """Audit logging with abuse detection."""

    def __init__(self, duckdb_path: str):
        self.duckdb_path = duckdb_path

    def log(self, entry: AuditEntry) -> int:
        """Log a request and return the log ID."""
        import duckdb

        with duckdb.connect(self.duckdb_path) as conn:
            result = conn.execute("""
                INSERT INTO audit_log
                    (session_id, tool_name, params, result_count,
                     execution_ms, auth_level, ip_hint)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
            """, [
                entry.session_id,
                entry.tool_name,
                json.dumps(entry.params),
                entry.result_count,
                entry.execution_ms,
                entry.auth_level,
                entry.ip_hint
            ]).fetchone()

        return result[0]

    def detect_scraping_patterns(self, session_id: str) -> list[str]:
        """Detect suspicious patterns for a session."""
        import duckdb

        flags = []

        with duckdb.connect(self.duckdb_path) as conn:
            # Pattern 1: Sequential date queries
            sequential = conn.execute("""
                WITH date_params AS (
                    SELECT
                        json_extract_string(params, '$.date') as query_date,
                        timestamp
                    FROM audit_log
                    WHERE session_id = $1
                      AND tool_name IN ('get_current_price', 'get_historical_price')
                      AND timestamp > CURRENT_TIMESTAMP - INTERVAL 1 HOUR
                    ORDER BY timestamp
                )
                SELECT COUNT(*)
                FROM date_params d1
                JOIN date_params d2 ON d2.query_date = d1.query_date::DATE + INTERVAL 1 DAY
            """, [session_id]).fetchone()[0]

            if sequential > 5:
                flags.append(f"SEQUENTIAL_DATES: {sequential} consecutive date queries")

            # Pattern 2: High volume in short time
            recent_count = conn.execute("""
                SELECT COUNT(*) FROM audit_log
                WHERE session_id = $1
                  AND timestamp > CURRENT_TIMESTAMP - INTERVAL 10 MINUTE
            """, [session_id]).fetchone()[0]

            if recent_count > 50:
                flags.append(f"HIGH_VOLUME: {recent_count} requests in 10 minutes")

            # Pattern 3: Enumerating all species
            unique_species = conn.execute("""
                SELECT COUNT(DISTINCT json_extract_string(params, '$.species'))
                FROM audit_log
                WHERE session_id = $1
                  AND timestamp > CURRENT_TIMESTAMP - INTERVAL 1 HOUR
            """, [session_id]).fetchone()[0]

            if unique_species > 20:
                flags.append(f"SPECIES_ENUMERATION: {unique_species} different species queried")

        return flags

    def flag_session(self, session_id: str, reason: str):
        """Flag all requests from a session as suspicious."""
        import duckdb

        with duckdb.connect(self.duckdb_path) as conn:
            conn.execute("""
                UPDATE audit_log
                SET flagged = TRUE, flag_reason = $2
                WHERE session_id = $1 AND flagged = FALSE
            """, [session_id, reason])
```

### Automated Abuse Detection

```python
# src/mcp_server/security/abuse_detector.py

from .audit import AuditLogger

class AbuseDetector:
    """Automated abuse detection and blocking."""

    def __init__(self, audit_logger: AuditLogger, rate_limiter):
        self.audit = audit_logger
        self.rate_limiter = rate_limiter

    def check_and_block(self, session_id: str) -> Optional[str]:
        """
        Check for abuse patterns and block if detected.

        Returns:
            Block reason if blocked, None otherwise
        """
        flags = self.audit.detect_scraping_patterns(session_id)

        if not flags:
            return None

        # Block the session
        block_reason = "; ".join(flags)
        self.rate_limiter.blocked_until[session_id] = time.time() + 86400  # 24h block
        self.audit.flag_session(session_id, block_reason)

        return block_reason
```

---

## Security Configuration

### Environment Variables

```bash
# .env.example

# Security settings
MCP_RATE_LIMIT_MINUTE=30
MCP_RATE_LIMIT_HOUR=300
MCP_RATE_LIMIT_DAY=1000

MCP_MAX_RESULTS=100
MCP_MAX_DATE_RANGE_DAYS=90

MCP_ENABLE_AUTH=false
MCP_ENABLE_AUDIT=true

# For production, set these:
# MCP_ENABLE_AUTH=true
# MCP_ADMIN_KEY=your_admin_key_here
```

### Server Configuration

```json
{
    "name": "fish-price-mcp",
    "version": "1.0.0",
    "security": {
        "rate_limiting": {
            "enabled": true,
            "requests_per_minute": 30,
            "requests_per_hour": 300,
            "requests_per_day": 1000
        },
        "result_limits": {
            "max_records": 100,
            "max_date_range_days": 90,
            "max_species_list": 50
        },
        "authentication": {
            "enabled": false,
            "require_key_for_predictions": true
        },
        "audit": {
            "enabled": true,
            "detect_abuse": true,
            "auto_block": true
        }
    }
}
```

---

## Implementation Priority

For AWS t4-micro (512MB RAM) with budget constraints:

| Priority | Layer | Memory Cost | Implementation Effort |
|----------|-------|-------------|----------------------|
| **P0** | Result Limits | None | Low - just constants |
| **P0** | Query Design | None | Medium - tool refactoring |
| **P1** | Rate Limiting | ~10MB | Low - in-memory dict |
| **P2** | Audit Logging | ~2MB + disk | Medium - DuckDB table |
| **P3** | Authentication | ~5MB | High - key management |

### Minimum Viable Security (MVP)

For initial deployment, implement only P0:

```python
# Minimal security - just add to each tool

MAX_RESULTS = 100
MAX_DATE_RANGE_DAYS = 90

async def get_historical_price(species, start_date, end_date, **kwargs):
    # Enforce limits
    start = parse_date(start_date)
    end = parse_date(end_date)

    if (end - start).days > MAX_DATE_RANGE_DAYS:
        start = end - timedelta(days=MAX_DATE_RANGE_DAYS)

    # Always use LIMIT
    query = f"... LIMIT {MAX_RESULTS}"

    # Return aggregate stats, not just raw data
    ...
```

---

## Extraction Time Analysis

With all protections enabled:

| Protection | Effect |
|------------|--------|
| 100 records/request | Need 5,000+ requests for full DB |
| 30 requests/min | ~3 hours minimum for enumeration |
| 90-day range limit | 84 requests per species history |
| 1,000/day limit | Would take 5+ days to extract all |

**Result**: Bulk extraction requires 5+ days of continuous requests with highly visible activity patterns, making it easily detectable and blockable.

---

## Testing Security

```python
# tests/test_security.py

import pytest
from mcp_server.security.rate_limiter import RateLimiter
from mcp_server.security.limits import LIMITS

def test_rate_limiter_blocks_after_limit():
    limiter = RateLimiter()
    session = "test_session"

    # Should allow up to limit
    for _ in range(30):
        allowed, _ = limiter.check_rate_limit(session)
        assert allowed

    # 31st request should be blocked
    allowed, reason = limiter.check_rate_limit(session)
    assert not allowed
    assert "RATE_LIMIT_MINUTE" in reason

def test_result_limits_enforced():
    assert LIMITS.MAX_RECORDS_PER_REQUEST == 100
    assert LIMITS.MAX_DATE_RANGE_DAYS == 90

@pytest.mark.asyncio
async def test_historical_price_truncates_range():
    result = await get_historical_price(
        species="고등어",
        start_date="2020-01-01",
        end_date="2024-12-31"  # ~5 years
    )

    # Should be truncated to 90 days
    assert result["truncated"] == True
    assert len(result["data"]) <= 100
```

---

## References

- [MCP Security Best Practices](https://modelcontextprotocol.io/docs/security)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [Rate Limiting Patterns](https://cloud.google.com/architecture/rate-limiting-strategies)
