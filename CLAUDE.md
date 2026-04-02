# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Noryangjin Fish Market Crawler with MCP Server — a modular async crawler for fish auction prices (경락시세) from Noryangjin Fish Market (노량진수산시장). Crawls from `https://www.susansijang.co.kr/nsis/miw/ko/info/miw3110` via POST requests. Dataset spans 2004–present with 2.5M+ records stored as Hive-partitioned Parquet files.

## Commands

**Always use `uv run` to execute Python scripts.** This project uses uv as its package manager (not pip). Never run Python scripts directly without the `uv run` prefix.

```bash
# Run crawler
uv run python scripts/crawler/run_crawler.py historical          # Full historical crawl (checkpoint-resumable)
uv run python scripts/crawler/run_crawler.py daily                # Yesterday's data (for cron)
uv run python scripts/crawler/run_crawler.py range --start 2024.01.01 --end 2024.01.31
uv run python scripts/crawler/run_crawler.py date --date 2024.01.15

# Run all tests (uses real network requests, no mocking)
uv run pytest tests/test_noryangjin.py -v

# Run a single test
uv run pytest tests/test_noryangjin.py::test_crawl_recent_date -v

# Install dependencies
uv sync
uv sync --extra dev   # Include dev dependencies (pytest, pytest-asyncio)
```

## Architecture

```
NoryangjinCrawler (orchestrator — noryangjin_crawler.py)
    ├── Scheduler (scheduler.py)     — Date iteration + CheckpointManager (JSON state)
    ├── Fetcher (fetcher.py)         — Async HTTP POST via aiohttp + RateLimiter (token-bucket, 1.5s)
    ├── HTMLParser (parser.py)       — Regex-based table extraction + pagination via fnList(N)
    ├── Normalizer (normalizer.py)   — Species aliases, spec normalization, state codes, empty-string coercion, price fixes, species inventory, validation
    └── Writer (writer.py)           — BaseWriter ABC → ParquetWriter, JSONWriter, CSVWriter, MemoryWriter, CompositeWriter
```

**Data flow:** Scheduler → Fetcher → HTMLParser → Normalizer → Writer → Checkpoint

**Data models** (`models.py`): `PriceRecord`, `CrawlDayResult`, `CrawlStats`, `CheckpointState` — all dataclasses.

### Storage Layout

Parquet files use Hive partitioning: `data/parquet/prices/year=YYYY/month=MM/YYYY-MM-DD.parquet`
Checkpoint state lives at `data/checkpoint.json` for resumable crawls.

### Entry Pattern

All crawler usage follows the async context manager pattern:
```python
async with NoryangjinCrawler(checkpoint_path="data/checkpoint.json", writer=writer) as crawler:
    results, stats = await crawler.crawl_date_range(start, end)
```

## Data Preprocessing Workflow

When adding or modifying data preprocessing rules, follow this order strictly:

1. **Document first** — Write the issue and fix to `docs/08_data_preprocessing.md` (or add to it). Include: what's wrong, affected row counts, the chosen canonical form, and why.
2. **Handle live data** — Implement the fix in `crawler/normalizer.py` so future crawled data is normalized automatically. If a new species appears, update the inventory (`crawler/species_inventory.json`) via `scripts/update_species_inventory.py`.
3. **Fix existing data** — Apply to stored Parquet files via `scripts/normalize_data.py` (dry-run first, then `--apply`). Verify with a query afterward.

Key files:
- `docs/08_data_preprocessing.md` — All preprocessing rules with rationale
- `crawler/normalizer.py` — `Normalizer` class with all rules; `normalize_price_record()` is the single entry point
- `crawler/species_inventory.json` — Canonical species registry (504 names); new species are logged as warnings during crawling
- `scripts/normalize_data.py` — Batch script to apply all fixes to existing Parquet files (supports `--apply` and dry-run)
- `scripts/update_species_inventory.py` — Rebuilds species inventory from current Parquet data

## Key Conventions

- **Python 3.12+** required (see `.python-version`)
- All HTTP operations are async (`aiohttp`). New network code must use `async/await`.
- Writers implement `BaseWriter` ABC. Use `CompositeWriter` to chain multiple output formats.
- Tests are integration tests hitting the real website — they take ~40s+ per date crawled and require network access.
- Date format throughout the codebase is `"YYYY.MM.DD"` (dot-separated).
- The `data/` and `logs/` directories are gitignored.

## Project Status and Roadmap

**Phase 1 (Crawler):** Complete. Normalization pipeline, species inventory, daily crawl support.

**Phase 2 (Prediction):** Complete. v1-v11 CPU + DL v1/v2 GPU. 15/20 configs below 20% MAPE. Key docs:
- `docs/12_poc_prediction_report.md` — Full v1→v11 + DL v1/v2 comparison results
- `docs/13_deep_learning_literature_review.md` — 35+ papers surveyed
- `docs/14_advanced_preprocessing.md` — 5 preprocessing fixes (biggest improvement)
- `docs/15_prediction_config_registry.md` — 20 configs, v2 results, expansion tracking
- `docs/16_model_construction_candidates.md` — 188 species, 1,195 viable configs, 125 training

**Phase 2b (DL v2 + Expansion):** In progress. Optuna HPO, per-config loss, weather features (76 total).
- `scripts/train_dl_v2.py` — Enhanced training with Optuna + loss routing + CQR
- `data/new_configs_to_train.json` — 125 new configs (35 A-grade + 88 B-grade)
- `data/weather/coastal_weather_daily.csv` — 36K rows, 5 ports, 2006-2026

**Phase 2c (Dashboard):** Complete. Streamlit dashboard with 5 pages.
- `dashboard/app.py` — Entry point, all 504 species in price trends
- Run: `uv run streamlit run dashboard/app.py --server.port 8501`

**Phase 2d (Daily Pipeline):** Ready for cron.
- Crawler: `uv run python scripts/crawler/run_crawler.py daily`
- KHOA: `uv run python scripts/fetch_khoa_daily.py`
- Weather: `uv run python scripts/fetch_coastal_weather.py`

**Phase 3 (MCP Server):** Not started. Design at `docs/04_mcp_server_design.md`.
- **Security Layer** (`docs/07_security.md`): Rate limiting, audit logging, optional API key auth

## Prediction Model Configs

Each prediction config is identified by a full tuple: `(species, state, packaging, spec, origin_class)`.
Same species with different configs = different market = different model.

### Tested Configs (20 original + 125 expansion, DL v2 on GB10)

**Sashimi (중 grade, 7 configs):**
넙치_활_kg_중, 우럭_활_kg_중, 방어_선_kg_중_dom, 참돔_활_kg_중_dom, 농어_활_kg_중_dom, 도다리_활_kg_중, 감성돔_활_kg_중_dom

**Standard (8 configs):**
감숭어_활_kg_중, 참숭어_활_kg_중, 쭈꾸미_선_box_중_dom, 민어_선_SP_중, 깐굴_선_box_소, 바위굴_활_box_대, 수꽃게_활_kg_중, 암꽃게_활_kg_중

**Expansion (125 configs training):** 61 species across all market segments. See `docs/16_model_construction_candidates.md`.

**Premium 활어 (5 configs):**
수꽃게_활_kg_대, 암꽃게_활_kg_대, 넙치_활_kg_2미, 참돔_활_kg_2미_dom, 농어_활_kg_1미_dom
