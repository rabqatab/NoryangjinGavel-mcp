# MCP Server Design: Fish Price Information

## Overview

This document describes the Model Context Protocol (MCP) server design for exposing fish price data from Noryangjin Fish Market.

### Purpose

Provide AI assistants with:
- Current fish prices
- Historical price data
- Price trends and analysis
- Fish species information

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MCP SERVER ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│    ┌─────────────┐                                                  │
│    │   Claude    │                                                  │
│    │  or other   │                                                  │
│    │  AI Client  │                                                  │
│    └──────┬──────┘                                                  │
│           │ MCP Protocol (stdio/SSE)                                │
│           ▼                                                          │
│    ┌─────────────────────────────────────────────────────────┐      │
│    │                    MCP SERVER                            │      │
│    ├─────────────────────────────────────────────────────────┤      │
│    │                                                          │      │
│    │  ┌────────────────────────────────────────────────────┐ │      │
│    │  │              SECURITY LAYER                         │ │      │
│    │  │  • Rate Limiting (30/min, 300/hr, 1000/day)        │ │      │
│    │  │  • Result Limits (max 100 records, 90-day range)   │ │      │
│    │  │  • Audit Logging                                    │ │      │
│    │  │  • Optional: API Key Authentication                 │ │      │
│    │  └────────────────────────────────────────────────────┘ │      │
│    │                            │                              │      │
│    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │      │
│    │  │    Tools     │  │  Resources   │  │   Prompts    │  │      │
│    │  │   Handler    │  │   Handler    │  │   Handler    │  │      │
│    │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │      │
│    │         │                  │                  │          │      │
│    │         └──────────────────┼──────────────────┘          │      │
│    │                            │                              │      │
│    │                            ▼                              │      │
│    │              ┌─────────────────────────┐                 │      │
│    │              │    Query Service        │                 │      │
│    │              │    ─────────────        │                 │      │
│    │              │  • Price queries        │                 │      │
│    │              │  • Aggregations         │                 │      │
│    │              │  • Trend analysis       │                 │      │
│    │              └───────────┬─────────────┘                 │      │
│    │                          │                                │      │
│    │                          ▼                                │      │
│    │              ┌─────────────────────────┐                 │      │
│    │              │   DuckDB + Parquet      │                 │      │
│    │              │   ─────────────────     │                 │      │
│    │              │  • fish_market.duckdb   │                 │      │
│    │              │  • parquet/prices/**    │                 │      │
│    │              └─────────────────────────┘                 │      │
│    │                                                          │      │
│    └─────────────────────────────────────────────────────────┘      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

> **Security Documentation**: See [`07_security.md`](./07_security.md) for comprehensive anti-scraping protection details.

---

## MCP Tools

### Tool 1: `get_current_price`

Get today's fish auction prices.

**Schema:**
```json
{
    "name": "get_current_price",
    "description": "Get current (today's) fish auction prices from Noryangjin Fish Market. Returns prices for all or specific fish species.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "species": {
                "type": "string",
                "description": "Fish species name in Korean (e.g., '고등어', '갈치'). If omitted, returns all species."
            },
            "origin": {
                "type": "string",
                "description": "Origin location filter (e.g., '부산', '제주도'). Optional."
            },
            "state": {
                "type": "string",
                "enum": ["선", "활", "냉", "가공"],
                "description": "Fish state filter: 선(Fresh), 활(Live), 냉(Frozen), 가공(Processed). Optional."
            },
            "limit": {
                "type": "integer",
                "default": 50,
                "description": "Maximum number of results to return."
            }
        },
        "required": []
    }
}
```

**Example Response:**
```json
{
    "date": "2025-01-02",
    "total_records": 45,
    "prices": [
        {
            "species": "고등어",
            "species_en": "Mackerel",
            "state": "냉동",
            "origin": "부산(기장)",
            "spec": "22미",
            "packaging": "CT/(BT)",
            "quantity": 719,
            "price_high": 35000,
            "price_low": 34000,
            "price_avg": 34100
        }
    ]
}
```

---

### Tool 2: `get_historical_price`

Get historical price data for a specific fish species.

**Schema:**
```json
{
    "name": "get_historical_price",
    "description": "Get historical fish auction prices for a specific species within a date range.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "species": {
                "type": "string",
                "description": "Fish species name in Korean (required)."
            },
            "start_date": {
                "type": "string",
                "format": "date",
                "description": "Start date in YYYY-MM-DD format."
            },
            "end_date": {
                "type": "string",
                "format": "date",
                "description": "End date in YYYY-MM-DD format."
            },
            "origin": {
                "type": "string",
                "description": "Filter by origin location. Optional."
            },
            "aggregation": {
                "type": "string",
                "enum": ["daily", "weekly", "monthly"],
                "default": "daily",
                "description": "Aggregation level for the results."
            }
        },
        "required": ["species", "start_date", "end_date"]
    }
}
```

**Example Response:**
```json
{
    "species": "고등어",
    "species_en": "Mackerel",
    "period": {
        "start": "2024-12-01",
        "end": "2024-12-31"
    },
    "aggregation": "daily",
    "data": [
        {
            "date": "2024-12-01",
            "avg_price": 32500,
            "min_price": 28000,
            "max_price": 38000,
            "total_quantity": 15420,
            "record_count": 45
        }
    ]
}
```

---

### Tool 3: `get_price_trend`

Analyze price trends for a fish species.

**Schema:**
```json
{
    "name": "get_price_trend",
    "description": "Analyze price trends for a fish species over a specified period. Returns trend direction, percentage change, and statistics.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "species": {
                "type": "string",
                "description": "Fish species name in Korean."
            },
            "period": {
                "type": "string",
                "enum": ["7d", "30d", "90d", "1y", "5y", "all"],
                "default": "30d",
                "description": "Analysis period: 7d (week), 30d (month), 90d (quarter), 1y (year), 5y (5 years), all (all available)."
            },
            "compare_to_previous": {
                "type": "boolean",
                "default": true,
                "description": "Include comparison to previous period."
            }
        },
        "required": ["species"]
    }
}
```

**Example Response:**
```json
{
    "species": "고등어",
    "species_en": "Mackerel",
    "period": "30d",
    "analysis": {
        "current_avg_price": 34500,
        "period_start_price": 32000,
        "period_end_price": 35000,
        "min_price": 28000,
        "max_price": 42000,
        "price_change": 3000,
        "price_change_percent": 9.38,
        "trend": "increasing",
        "volatility": "moderate"
    },
    "comparison_to_previous": {
        "previous_avg_price": 31000,
        "change_percent": 11.29,
        "trend": "higher"
    },
    "statistics": {
        "total_quantity": 485000,
        "total_transactions": 1250,
        "trading_days": 22
    }
}
```

---

### Tool 4: `list_fish_species`

List all available fish species with metadata.

**Schema:**
```json
{
    "name": "list_fish_species",
    "description": "List all available fish species in the database with Korean/English names and categories.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["fish", "shellfish", "crustacean", "mollusk", "other", "all"],
                "default": "all",
                "description": "Filter by category."
            },
            "include_stats": {
                "type": "boolean",
                "default": false,
                "description": "Include recent trading statistics for each species."
            }
        },
        "required": []
    }
}
```

**Example Response:**
```json
{
    "total_species": 76,
    "categories": {
        "fish": 18,
        "shellfish": 5,
        "crustacean": 2,
        "mollusk": 2,
        "other": 2
    },
    "species": [
        {
            "name": "고등어",
            "name_en": "Mackerel",
            "category": "fish",
            "stats": {
                "last_trade_date": "2025-01-02",
                "avg_price_30d": 34500,
                "total_quantity_30d": 485000
            }
        }
    ]
}
```

---

### Tool 5: `compare_prices`

Compare prices across different dimensions.

**Schema:**
```json
{
    "name": "compare_prices",
    "description": "Compare fish prices across different origins, time periods, or fish states.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "species": {
                "type": "string",
                "description": "Fish species name in Korean."
            },
            "compare_by": {
                "type": "string",
                "enum": ["origin", "state", "year", "month", "season"],
                "description": "Dimension to compare by."
            },
            "date_range": {
                "type": "object",
                "properties": {
                    "start": { "type": "string", "format": "date" },
                    "end": { "type": "string", "format": "date" }
                },
                "description": "Date range for comparison. Optional - defaults to last 30 days."
            }
        },
        "required": ["species", "compare_by"]
    }
}
```

**Example Response (compare_by: origin):**
```json
{
    "species": "고등어",
    "compare_by": "origin",
    "period": "2024-12-01 to 2024-12-31",
    "comparison": [
        {
            "origin": "부산(기장)",
            "avg_price": 35000,
            "total_quantity": 125000,
            "record_count": 450,
            "rank": 1
        },
        {
            "origin": "제주도",
            "avg_price": 32000,
            "total_quantity": 85000,
            "record_count": 280,
            "rank": 2
        }
    ]
}
```

---

### Tool 6: `search_by_price_range`

Find fish within a specific price range.

**Schema:**
```json
{
    "name": "search_by_price_range",
    "description": "Find fish species and products within a specified price range.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "min_price": {
                "type": "integer",
                "description": "Minimum price per kg in KRW."
            },
            "max_price": {
                "type": "integer",
                "description": "Maximum price per kg in KRW."
            },
            "category": {
                "type": "string",
                "enum": ["fish", "shellfish", "crustacean", "mollusk", "other", "all"],
                "default": "all"
            },
            "date": {
                "type": "string",
                "format": "date",
                "description": "Date to search. Defaults to today."
            }
        },
        "required": ["min_price", "max_price"]
    }
}
```

---

## MCP Resources

### Resource 1: `fish://species/{name}`

Direct access to species information.

```json
{
    "uri": "fish://species/고등어",
    "name": "Mackerel Species Info",
    "mimeType": "application/json"
}
```

### Resource 2: `fish://price/{species}/{date}`

Direct access to price data for a specific date.

```json
{
    "uri": "fish://price/고등어/2025-01-02",
    "name": "Mackerel Price on 2025-01-02",
    "mimeType": "application/json"
}
```

### Resource 3: `fish://summary/daily`

Daily market summary.

```json
{
    "uri": "fish://summary/daily",
    "name": "Daily Market Summary",
    "mimeType": "application/json"
}
```

---

## MCP Prompts

### Prompt 1: `analyze_market`

```json
{
    "name": "analyze_market",
    "description": "Analyze the current fish market conditions",
    "arguments": [
        {
            "name": "focus",
            "description": "Area to focus on",
            "required": false
        }
    ]
}
```

### Prompt 2: `price_recommendation`

```json
{
    "name": "price_recommendation",
    "description": "Get buying recommendations based on price trends",
    "arguments": [
        {
            "name": "budget",
            "description": "Budget in KRW",
            "required": true
        },
        {
            "name": "preferences",
            "description": "Fish type preferences",
            "required": false
        }
    ]
}
```

---

## Implementation

### Server Entry Point

```python
# src/mcp_server/server.py

import asyncio
import json
import time
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, Resource, TextContent

from .tools import (
    get_current_price,
    get_historical_price,
    get_price_trend,
    list_fish_species,
    compare_prices,
    search_by_price_range
)
from .database import get_db_connection
from .security.rate_limiter import rate_limiter
from .security.audit import AuditLogger, AuditEntry

# Create server instance
server = Server("fish-price-mcp")
audit_logger = AuditLogger("data/fish_market.duckdb")

# Tool definitions
TOOLS = [
    Tool(
        name="get_current_price",
        description="Get current fish auction prices from Noryangjin Fish Market",
        inputSchema={
            "type": "object",
            "properties": {
                "species": {"type": "string"},
                "origin": {"type": "string"},
                "state": {"type": "string", "enum": ["선", "활", "냉", "가공"]},
                "limit": {"type": "integer", "default": 50, "maximum": 100}
            }
        }
    ),
    # ... other tools
]

@server.list_tools()
async def list_tools():
    return TOOLS

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    # Get session ID (implementation varies by MCP SDK)
    session_id = getattr(server, '_session_id', 'anonymous')

    # SECURITY: Rate limiting check
    allowed, reason = rate_limiter.check_rate_limit(session_id)
    if not allowed:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": True,
                "code": "RATE_LIMITED",
                "message": reason,
                "remaining": rate_limiter.get_remaining(session_id)
            }, ensure_ascii=False)
        )]

    start_time = time.time()

    match name:
        case "get_current_price":
            result = await get_current_price(**arguments)
        case "get_historical_price":
            result = await get_historical_price(**arguments)
        case "get_price_trend":
            result = await get_price_trend(**arguments)
        case "list_fish_species":
            result = await list_fish_species(**arguments)
        case "compare_prices":
            result = await compare_prices(**arguments)
        case "search_by_price_range":
            result = await search_by_price_range(**arguments)
        case _:
            raise ValueError(f"Unknown tool: {name}")

    # SECURITY: Audit logging
    execution_ms = int((time.time() - start_time) * 1000)
    result_count = len(result.get("data", result.get("prices", [])))

    audit_logger.log(AuditEntry(
        session_id=session_id,
        tool_name=name,
        params=arguments,
        result_count=result_count,
        execution_ms=execution_ms
    ))

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

> **Security Details**: See [`07_security.md`](./07_security.md) for complete rate limiting, authentication, and audit logging implementation.

### Tool Implementation Example

```python
# src/mcp_server/tools.py

from datetime import datetime, timedelta
from typing import Optional
import json

from .database import get_db_connection

async def get_current_price(
    species: Optional[str] = None,
    origin: Optional[str] = None,
    state: Optional[str] = None,
    limit: int = 50
) -> dict:
    """Get current day's fish prices from Parquet via DuckDB."""

    today = datetime.now().date()

    # DuckDB query on Parquet view (denormalized - no joins needed)
    query = """
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
        WHERE trade_date = $1
    """
    params = [today]

    if species:
        query += " AND species = $2"
        params.append(species)

    if origin:
        query += f" AND origin LIKE '%' || ${len(params) + 1} || '%'"
        params.append(origin)

    if state:
        query += f" AND state = ${len(params) + 1}"
        params.append(state)

    query += f" ORDER BY species, origin LIMIT {limit}"

    with get_db_connection() as conn:
        result = conn.execute(query, params).fetchdf()

    # Join with species lookup for English names
    species_en_map = {}
    if not result.empty:
        species_names = result['species'].unique().tolist()
        en_result = conn.execute(
            "SELECT name, name_en FROM fish_species WHERE name = ANY($1)",
            [species_names]
        ).fetchdf()
        species_en_map = dict(zip(en_result['name'], en_result['name_en']))

    prices = []
    for _, row in result.iterrows():
        prices.append({
            "species": row['species'],
            "species_en": species_en_map.get(row['species']),
            "state": row['state'],
            "origin": row['origin'],
            "spec": row['spec'],
            "packaging": row['packaging'],
            "quantity": row['quantity'],
            "price_high": row['price_high'],
            "price_low": row['price_low'],
            "price_avg": row['price_avg']
        })

    return {
        "date": today.isoformat(),
        "total_records": len(prices),
        "prices": prices
    }


async def get_price_trend(
    species: str,
    period: str = "30d",
    compare_to_previous: bool = True
) -> dict:
    """Analyze price trends for a species using DuckDB."""

    # Calculate date ranges
    period_days = {
        "7d": 7,
        "30d": 30,
        "90d": 90,
        "1y": 365,
        "5y": 1825,
        "all": 10000
    }
    days = period_days.get(period, 30)

    with get_db_connection() as conn:
        # Get species English name
        species_row = conn.execute(
            "SELECT name_en FROM fish_species WHERE name = $1",
            [species]
        ).fetchone()
        species_en = species_row[0] if species_row else None

        # Get aggregated daily data from Parquet
        result = conn.execute("""
            SELECT
                trade_date,
                AVG(price_avg) AS avg_price,
                MIN(price_low) AS min_price,
                MAX(price_high) AS max_price,
                SUM(quantity) AS total_quantity
            FROM v_prices
            WHERE species = $1
              AND trade_date >= CURRENT_DATE - INTERVAL $2 DAY
            GROUP BY trade_date
            ORDER BY trade_date
        """, [species, days]).fetchdf()

    if result.empty:
        return {"error": f"No data found for '{species}'"}

    prices = result['avg_price'].tolist()
    quantities = result['total_quantity'].tolist()

    first_price = prices[0] if prices else 0
    last_price = prices[-1] if prices else 0
    change = last_price - first_price
    change_pct = (change / first_price * 100) if first_price else 0

    # Determine trend
    if change_pct > 5:
        trend = "increasing"
    elif change_pct < -5:
        trend = "decreasing"
    else:
        trend = "stable"

    return {
        "species": species,
        "species_en": species_en,
        "period": period,
        "analysis": {
            "current_avg_price": int(sum(prices) / len(prices)) if prices else 0,
            "period_start_price": int(first_price),
            "period_end_price": int(last_price),
            "min_price": int(result['min_price'].min()),
            "max_price": int(result['max_price'].max()),
            "price_change": int(change),
            "price_change_percent": round(change_pct, 2),
            "trend": trend
        },
        "statistics": {
            "total_quantity": int(sum(quantities)),
            "trading_days": len(result)
        }
    }
```

---

## Configuration

### Server Configuration File

```json
{
    "name": "fish-price-mcp",
    "version": "1.0.0",
    "description": "MCP server for Noryangjin Fish Market price data",
    "storage": {
        "duckdb_path": "data/fish_market.duckdb",
        "parquet_dir": "data/parquet/prices",
        "memory_limit_mb": 256
    },
    "server": {
        "transport": "stdio",
        "log_level": "INFO"
    }
}
```

### Claude Desktop Integration

```json
{
    "mcpServers": {
        "fish-price": {
            "command": "python",
            "args": ["-m", "mcp_server.server"],
            "cwd": "/path/to/mcp_NorayngjinGavel",
            "env": {
                "DUCKDB_PATH": "data/fish_market.duckdb",
                "PARQUET_DIR": "data/parquet/prices"
            }
        }
    }
}
```

---

## Error Handling

### Error Response Format

```json
{
    "error": true,
    "code": "SPECIES_NOT_FOUND",
    "message": "Species '잘못된어종' not found in database",
    "suggestions": ["고등어", "갈치", "삼치"]
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `SPECIES_NOT_FOUND` | Requested species doesn't exist |
| `DATE_RANGE_INVALID` | Invalid date range provided |
| `NO_DATA` | No data for the query parameters |
| `DATABASE_ERROR` | Database connection/query error |
| `INVALID_PARAMETER` | Invalid parameter value |
| `RATE_LIMITED` | Too many requests (see [`07_security.md`](./07_security.md)) |
| `DATE_RANGE_TRUNCATED` | Date range exceeded limit, truncated to 90 days |
| `SEARCH_TOO_SHORT` | Search term must be at least 2 characters |

---

## Prediction Tools (ML/Statistical)

> **Full documentation**: See [`06_prediction_system.md`](./06_prediction_system.md) for complete implementation details.

### Overview

| Tool | Description | Models Used |
|------|-------------|-------------|
| `predict_price` | Price forecasts with confidence intervals | Exp. Smoothing, ARIMA, Prophet |
| `get_seasonality` | Monthly patterns, best/worst months | Prophet decomposition |
| `detect_anomalies` | Unusual price movements | Z-score, IQR, Isolation Forest |
| `get_market_insight` | Trend, volatility, recommendations | Linear Regression, ensemble |
| `get_volatility` | Price stability analysis | Rolling statistics |

### Tool Schemas (Summary)

#### Tool 7: `predict_price`
```json
{
    "name": "predict_price",
    "inputSchema": {
        "properties": {
            "species": {"type": "string", "description": "Fish species in Korean"},
            "horizon": {"type": "string", "enum": ["1d", "7d", "14d", "30d", "all"]}
        },
        "required": ["species"]
    }
}
```

#### Tool 8: `get_seasonality`
```json
{
    "name": "get_seasonality",
    "inputSchema": {
        "properties": {
            "species": {"type": "string"},
            "include_weekly": {"type": "boolean", "default": false}
        },
        "required": ["species"]
    }
}
```

#### Tool 9: `detect_anomalies`
```json
{
    "name": "detect_anomalies",
    "inputSchema": {
        "properties": {
            "species": {"type": "string"},
            "period": {"type": "string", "enum": ["7d", "30d", "90d", "1y"]},
            "min_severity": {"type": "string", "enum": ["low", "medium", "high"]}
        }
    }
}
```

#### Tool 10: `get_market_insight`
```json
{
    "name": "get_market_insight",
    "inputSchema": {
        "properties": {
            "species": {"type": "string"},
            "insight_type": {"type": "string", "enum": ["summary", "detailed", "recommendation"]}
        }
    }
}
```

#### Tool 11: `get_volatility`
```json
{
    "name": "get_volatility",
    "inputSchema": {
        "properties": {
            "species": {"type": "string"},
            "period": {"type": "string", "enum": ["30d", "90d", "1y", "all"]}
        }
    }
}
```

---

## Testing

### Test Cases

```python
# tests/test_tools.py

import pytest
from mcp_server.tools import get_current_price, get_price_trend

@pytest.mark.asyncio
async def test_get_current_price_all():
    result = await get_current_price()
    assert "prices" in result
    assert "date" in result

@pytest.mark.asyncio
async def test_get_current_price_species():
    result = await get_current_price(species="고등어")
    assert all(p["species"] == "고등어" for p in result["prices"])

@pytest.mark.asyncio
async def test_get_price_trend():
    result = await get_price_trend(species="고등어", period="30d")
    assert "analysis" in result
    assert "trend" in result["analysis"]

@pytest.mark.asyncio
async def test_invalid_species():
    result = await get_price_trend(species="잘못된어종")
    assert "error" in result
```
