# Data Preprocessing Rules

Normalization rules applied to raw crawl data from `https://www.susansijang.co.kr/nsis/miw/ko/info/miw3110`.

All rules are implemented in `crawler/normalizer.py` and applied both during live crawling (`noryangjin_crawler.py`) and via the batch script (`scripts/normalize_data.py`).

---

## Fix 1: Species Name Aliases

**Problem:** Variant spellings and typos in the `species` field.

**Affected rows:** ~1,079 (across 2.59M total)

| Variant (raw) | Canonical | Reason | Row count |
|---|---|---|---|
| `고둥 갯고동` | `고등 갯고동` | Typo: `둥`→`등`, standard fisheries term | 946 |
| `망둑어` | `망둥어` | 표준국어대사전 standard spelling | 39 |
| `쭈구미` | `쭈꾸미` | 국립국어원 standard spelling | 78 |
| `학공치` | `학꽁치` | Standard spelling, dominant by 3,400x | 2 |
| `깐우렁` | `깐우렁이` | Truncation of full form `우렁이` | 8 |
| `갑오징어기타` | `갑오징어 기타` | Missing space; all other `기타` entries use space | 6 |

**Implementation:** `Normalizer.SPECIES_ALIASES` dict, applied via `normalize_species()`.

**Selection criteria:** Map to (1) the majority form, (2) the most recent form, (3) the most correct per standard Korean dictionaries.

---

## Fix 2: Missing State Codes

**Problem:** `STATE_CODES` only covers 4 values (`선`, `활`, `냉`, `가공`) but the source data contains 2 more legitimate states.

**Affected rows:** 14,975

| State | English | Meaning | Row count |
|---|---|---|---|
| `냉건` | `frozen_dried` | Frozen then dried (냉동건조) — e.g. 코다리명태, 오징어채 | 14,758 |
| `건` | `dried` | Dried (건조) — e.g. 재래김, 돌김, 청태김 | 217 |

**Implementation:** Added to `Normalizer.STATE_CODES`.

**Note:** These states have been present since 2006 and continue in current data. They are not errors.

---

## Fix 3: Zero-Padded Spec Values

**Problem:** Spec field uses zero-padded format (`01미`–`09미`) for data from 2006–2010, while post-2010 data uses unpadded (`1미`–`9미`). These represent the same fish count specification.

**Affected rows:** 69,160

| Padded | Unpadded | Padded count | Unpadded count |
|---|---|---|---|
| `01미` | `1미` | 2,908 | 34,179 |
| `02미` | `2미` | 5,287 | 36,865 |
| `03미` | `3미` | 6,930 | 42,562 |
| `04미` | `4미` | 9,442 | 43,387 |
| `05미` | `5미` | 8,635 | 40,465 |
| `06미` | `6미` | 12,395 | 47,797 |
| `07미` | `7미` | 5,111 | 23,131 |
| `08미` | `8미` | 13,184 | 48,278 |
| `09미` | `9미` | 5,268 | 16,185 |

**Date range:** 2006.03.15 – 2010.04.30 (source website changed formatting around 2010)

**Implementation:** `Normalizer.normalize_spec()` — strips leading zeros from numeric-suffixed specs (e.g. `08미`→`8미`), but does NOT strip from range specs like `09/10` (these are ambiguous display formats, not duplicates of existing values).

---

## Fix 4: Empty Strings Instead of Null

**Problem:** Some fields contain empty strings `""` where they should be `None`/null. Caused by blank HTML cells that `_clean_cell()` returns as `""`.

**Affected rows:** 69 total

| Field | Empty string count |
|---|---|
| `state` | 8 |
| `origin` | 2 |
| `spec` | 30 |
| `packaging` | 29 |

**Implementation:** `Normalizer.normalize_price_record()` coerces empty strings to `None` on `state`, `origin`, `spec`, and `packaging` fields.

---

## Fix 5: Anomalous price_avg=0 Rows

**Problem:** 2 rows (2025.09.22) have `price_high=10, price_low=10, price_avg=0`. The average of 10 and 10 should be 10, not 0. This is a source data error.

**Affected rows:** 2

| Date | Species | price_high | price_low | price_avg (raw) | price_avg (fixed) |
|---|---|---|---|---|---|
| 2025.09.22 | 넙치 | 10 | 10 | 0 | 10 |
| 2025.09.22 | 잡어 | 10 | 10 | 0 | 10 |

**Implementation:** `Normalizer.normalize_price_record()` — if `price_avg == 0` and both `price_high > 0` and `price_low > 0`, recalculate as `(price_high + price_low) // 2`.

---

## Summary

| Fix | Rows affected | Severity |
|---|---|---|
| Species aliases | 1,079 | Medium — breaks groupby/aggregation on species |
| Missing state codes | 14,975 | Low — only affects English translation, data already correct |
| Zero-padded specs | 69,160 | High — breaks groupby/aggregation on spec |
| Empty strings | 69 | Low — cosmetic, may cause issues with null-aware queries |
| price_avg=0 | 2 | Low — edge case in source data |

Total rows affected: ~85,285 out of 2,589,655 (3.3%)

---

## Species Inventory

A canonical inventory of all 504 known species names is maintained at `crawler/species_inventory.json`. This file maps each species name to its total record count across all parquet data.

**Purpose:**
- Detect new species appearing in live-crawled data (logged as warnings)
- Serve as a reference for downstream consumers (MCP server, prediction system)
- Track vocabulary growth over time

**Live detection:** During crawling, `Normalizer.check_new_species()` compares each record's species against the inventory and logs `WARNING: New species detected: '<name>'` on first encounter. New species are collected in `normalizer.get_new_species()` for the session.

**Updating the inventory:**
```bash
uv run python scripts/update_species_inventory.py
```

Run this after historical crawls, after adding new aliases, or periodically to incorporate legitimately new species from daily data. The script shows a diff of added/removed species.

---

## Row Aggregation Strategy

> Results from EDA notebook: `notebooks/eda_aggregation.py`
> Spec: `docs/superpowers/specs/2026-03-25-aggregation-eda-design.md`

This section will be populated after running the EDA notebook. The notebook applies a decision tree testing state, packaging, spec, and origin dimensions to determine the optimal GROUP BY for daily price aggregation.

### How to Run

```bash
# Interactive (recommended — see charts and tables)
uv run marimo edit notebooks/eda_aggregation.py

# Headless (stdout summary only)
uv run marimo run notebooks/eda_aggregation.py
```

### Aggregation Rules

See `docs/09_aggregation_eda_report.md` for the full EDA report. Key conclusions:

- **No universal blended aggregation works** — each species needs per-species GROUP BY configuration
- **Spec-class is the most impactful dimension** (segments price 1.5-6x for 7/10 top species)
- **Use unweighted mean** — quantity units are incomparable across packaging types
- **5/10 top species need state partitioning** (아귀, 낙지, 오징어, 넙치, 고등어)
- **9/10 top species need dominant-packaging filtering** (all except 전복)

### DuckDB View

Dominant GROUP BY pattern (4/10 top species):

```sql
CREATE OR REPLACE VIEW v_daily_prices AS
SELECT trade_date, species, packaging, spec_class,
       SUM(quantity) AS total_quantity,
       MAX(price_high) AS price_high,
       MIN(price_low) AS price_low,
       CAST(AVG(price_avg) AS INTEGER) AS price_avg,
       COUNT(*) AS n_lots
FROM read_parquet('data/parquet/prices/**/*.parquet', hive_partitioning=true)
WHERE state IS NOT NULL AND packaging IS NOT NULL
GROUP BY trade_date, species, packaging, spec_class
ORDER BY trade_date, species, packaging, spec_class;
```

Note: Multi-state species need additional state filtering. See the per-species decision table in the full report.
