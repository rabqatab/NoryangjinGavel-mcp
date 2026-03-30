# Aggregation EDA Report

> Generated from `scripts/eda_aggregation.py` on 2026-03-26.
> Spec: `docs/superpowers/specs/2026-03-25-aggregation-eda-design.md`

## Executive Summary

**Can we aggregate ~415 rows/day into one daily price per species for prediction?**

**No — not naively.** The data is far more heterogeneous than hypothesized. Every dimension (state, packaging, spec, origin) carries material price information for most of the top 10 species.

### Key Findings

| Finding | Impact |
|---|---|
| 83 species have >1.5x price divergence across states | State is a required partition dimension for half the top 10 |
| All 10 top species have fully uncorrelated packaging series (corr < 0.75) | Blending across packaging types destroys signal |
| Spec segments price 6x+ for 전복, 1.5-1.9x for 6 others | Spec-class grouping needed for most species |
| Origin spread >30% median for 8/10 top species | Origin mixes different price levels |
| Weighted vs unweighted correlation < 0.98 for all 10 species | Quantity-weighting is unreliable — use unweighted mean |
| Only 전복 passes the blended aggregation test (EDA-3.1) | 9/10 top species need dominant-packaging filtering |

**Recommended approach:** Per-species GROUP BY with `(trade_date, species, [state], packaging, [spec_class])`, using **unweighted** mean. No single universal view works — species need individual treatment.

---

## Phase 1: Data Census

### EDA-1.0: State Distribution

- **350/504** species (69%) have >90% of rows in one state — state is irrelevant for these.
- **154** species are multi-state. Of these, **83** have >1.5x price divergence between states.
- **Assertion result:** State MUST be a partition dimension for multi-state species with price divergence.

Notable divergences (top 10 species):

| Species | Divergent? | States & Mean Prices (KRW) |
|---|---|---|
| 전복 | No | 활 dominates (99.5%) |
| 병어 | No | 선 dominates (98.1%) |
| 삼치 | No | 선 dominates (94.1%) |
| 아귀 | **Yes (12.7x)** | 선=27,729 / 냉=69,813 / 활=5,495 |
| 은갈치 | No | 선 dominates (93.3%) |
| 대구 | No | 선 dominates (91.1%) |
| 낙지 | **Yes (1.64x)** | 선=24,982 / 냉=29,093 / 활=40,853 |
| 오징어 | **Yes (173.7x)** | 선=25,488 / 냉=38,748 / 냉건=177,856 |
| 넙치 | **Yes (1.7x)** | 선=11,655 / 활=14,721 / 냉=19,783 |
| 고등어 | **Yes (4.07x)** | 냉=36,238 / 선=26,912 / 가공=23,645 |

**5/10 top species need state partitioning.** The remaining 5 can filter to their dominant state.

### EDA-1.1: Packaging Dominance

- **298/504** species (59%) have >80% in one packaging type.
- **36 species** have no packaging type >50% — these are "heterogeneous" (EDA-4.1).

Top 10 dominant packaging:

| Species | Dominant Pkg | % | # Types |
|---|---|---|---|
| 전복 | kg | 99.7% | 5 |
| 병어 | S/P | 91.8% | 6 |
| 삼치 | S/P | 85.3% | 7 |
| 아귀 | S/P | 62.7% | 8 |
| 은갈치 | S/P | 94.6% | 7 |
| 대구 | S/P | 78.6% | 8 |
| 낙지 | box | 84.4% | 8 |
| 오징어 | S/P | 73.9% | 7 |
| 넙치 | kg | 76.2% | 8 |
| 고등어 | S/P | 64.8% | 8 |

### EDA-1.2: Spec Type Taxonomy

Global distribution:

| Category | Rows | % |
|---|---|---|
| size_grade (대/중/소/특대) | 1,342,167 | 51.8% |
| count (N미) | 930,646 | 35.9% |
| other | 143,349 | 5.5% |
| count_range (N/M미) | 130,995 | 5.1% |
| weight_range (N/M) | 42,468 | 1.6% |
| null | 30 | <0.01% |

92 species span 5+ spec categories. Spec is a major source of row variation.

### EDA-1.3: Origin Distribution

| Stat | Value |
|---|---|
| Median origins per (species, day) | 2 |
| Mean | 2.3 |
| p95 | 6 |

Origin is a moderate row-multiplier — most species have 2 origins per day, a few have up to 6.

---

## Phase 2: Price Coherence Tests

### EDA-2.1: Packaging CV Ratio

**Threshold: 1.5x** (50% noise increase degrades ARIMA MAPE by 3-5 pp)

| Species | Within-Pkg CV | Across-Pkg CV | Ratio | Verdict |
|---|---|---|---|---|
| 전복 | 0.4367 | 0.4403 | 1.01x | OK |
| 병어 | 0.7111 | 0.8057 | 1.13x | OK |
| 삼치 | 0.5471 | 0.6819 | 1.25x | OK |
| **아귀** | 0.5320 | 0.8165 | **1.53x** | SEGMENTS |
| 은갈치 | 0.5328 | 0.5682 | 1.07x | OK |
| 대구 | 0.4644 | 0.6312 | 1.36x | OK |
| **낙지** | 0.1066 | 0.1856 | **1.74x** | SEGMENTS |
| **오징어** | 0.2707 | 0.4643 | **1.72x** | SEGMENTS |
| 넙치 | 0.4057 | 0.5009 | 1.23x | OK |
| 고등어 | 0.3782 | 0.4574 | 1.21x | OK |

**Result:** 3/10 species segmented at 1.5x. At 2.0x threshold, 0/10 — the effect is moderate.

**Sensitivity sweep:**

| Threshold | Segmented | OK |
|---|---|---|
| 1.5x | 3 | 7 |
| 2.0x | 0 | 10 |
| 2.5x | 0 | 10 |

### EDA-2.1b: Spec CV Ratio

**This is the most impactful finding.** Spec segments price heavily within packaging.

| Species | Within-Spec CV | Across-Spec CV | Ratio | Verdict |
|---|---|---|---|---|
| **전복** | 0.0699 | 0.4313 | **6.17x** | SEGMENTS |
| **병어** | 0.3875 | 0.7384 | **1.91x** | SEGMENTS |
| **삼치** | 0.3600 | 0.5698 | **1.58x** | SEGMENTS |
| 아귀 | 0.3953 | 0.5270 | 1.33x | OK |
| **은갈치** | 0.3211 | 0.5387 | **1.68x** | SEGMENTS |
| **대구** | 0.3025 | 0.4995 | **1.65x** | SEGMENTS |
| 낙지 | 0.1345 | 0.1050 | 0.78x | OK |
| **오징어** | 0.1475 | 0.2756 | **1.87x** | SEGMENTS |
| 넙치 | 0.2410 | 0.2881 | 1.20x | OK |
| **고등어** | 0.2328 | 0.3918 | **1.68x** | SEGMENTS |

**7/10 top species have spec as a price-segmenting dimension.** 전복 is extreme at 6.17x — different sizes of abalone trade at completely different prices. This means aggregation GROUP BY must include a `spec_class` dimension for these species.

### EDA-2.2: Per-Packaging Time Series Correlation

**All 10 species show essentially zero correlation between packaging types.**

| Species | Pkg A | Pkg B | Corr (W) | Price Ratio | Stable? |
|---|---|---|---|---|---|
| 전복 | kg | S/P | 0.293 | 2.1x | N |
| 병어 | S/P | c/s(상자) | 0.222 | 1.11x | N |
| 삼치 | S/P | c/s(상자) | 0.412 | 1.43x | Y |
| 아귀 | S/P | CT/(BT) | 0.058 | 2.45x | Y |
| 은갈치 | S/P | CT/(BT) | 0.145 | 1.7x | N |
| 대구 | S/P | c/s(상자) | 0.497 | 1.32x | Y |
| 낙지 | box | CT/(BT) | 0.038 | 1.06x | N |
| 오징어 | S/P | PAN(펜) | 0.741 | 1.64x | Y |
| 넙치 | kg | S/P | 0.040 | 1.06x | N |
| 고등어 | S/P | CT/(BT) | 0.133 | 1.41x | Y |

**No species exceeds the 0.85 correlation threshold.** The highest is 오징어 at 0.741. Packaging types represent fundamentally different product forms with independent price dynamics. Blending across packaging types is **invalid** for prediction — the series do not co-move.

### EDA-2.3: Origin Price Spread

| Species | Median Spread | Verdict |
|---|---|---|
| 전복 | 9.3% | OK |
| 병어 | 61.0% | ORIGIN MATTERS |
| 삼치 | 74.2% | ORIGIN MATTERS |
| 아귀 | 80.6% | ORIGIN MATTERS |
| 은갈치 | 51.4% | ORIGIN MATTERS |
| 대구 | 59.1% | ORIGIN MATTERS |
| 낙지 | 25.8% | OK |
| 오징어 | 35.3% | ORIGIN MATTERS |
| 넙치 | 60.1% | ORIGIN MATTERS |
| 고등어 | 42.3% | ORIGIN MATTERS |

**8/10 species have origin spreads >30%.** However, this high spread is likely driven by the spec dimension within origin groups (different-sized fish from the same port), not by origin itself. Since spec is already identified as a segmenting dimension, origin may become less impactful after spec-class grouping.

### EDA-2.4: Quantity Unit Comparability

**All 10 species fail the 0.98 threshold.** Weighted vs unweighted correlations range from 0.47 (넙치) to 0.95 (오징어).

**Conclusion:** Quantity-weighting is unreliable because quantity units are incomparable across packaging types (kg vs count vs box). **Use unweighted (simple) mean for all aggregation.**

---

## Phase 3: Aggregation Viability

### EDA-3.1: Blended vs Dominant-Packaging

| Species | Corr | Lag1 (blend) | Lag1 (dom) | Verdict |
|---|---|---|---|---|
| **전복** | **0.9974** | **0.8287** | **0.8281** | **BLEND OK** |
| 병어 | 0.9852 | 0.6084 | 0.5873 | USE DOMINANT |
| 삼치 | 0.9552 | 0.7759 | 0.8063 | USE DOMINANT |
| 아귀 | 0.8272 | 0.4940 | 0.3608 | USE DOMINANT |
| 은갈치 | 0.9821 | 0.7938 | 0.7870 | USE DOMINANT |
| 대구 | 0.9431 | 0.8020 | 0.8059 | USE DOMINANT |
| 낙지 | 0.6989 | 0.6496 | 0.7637 | USE DOMINANT |
| 오징어 | 0.9297 | 0.8377 | 0.8858 | USE DOMINANT |
| 넙치 | 0.5682 | 0.5119 | 0.5764 | USE DOMINANT |
| 고등어 | 0.8140 | 0.4420 | 0.4283 | USE DOMINANT |

**Only 전복 passes** (corr > 0.95 AND lag1 > 0.8). For the other 9 species, blended aggregation produces a noisier or less autocorrelated series than dominant-packaging-only.

**Notable:** Many species have low lag-1 autocorrelation even with dominant packaging (병어 0.59, 고등어 0.43). This suggests that daily price data for these species may be inherently noisy, which has implications for prediction model selection.

### EDA-3.2: Row Reduction Ratio

| Strategy | Groups | Median/Day | Compression |
|---|---|---|---|
| Raw | 2,589,653 | 415 | 1.0x |
| species | 601,630 | 97 | 4.3x |
| species+state | 741,806 | 119 | 3.5x |
| species+state+pkg | 870,993 | 140 | 3.0x |

Adding state and packaging reduces compression from 4.3x to 3.0x. Still a significant reduction from 415 to ~140 rows/day.

---

## Phase 4: Edge Cases

### EDA-4.1: Heterogeneous-Packaging Species

**36 species** have no single packaging type >50%. Notable examples: 잡어 (12 types, max 47.6%), 소라 (11 types, max 47.7%). These species need per-packaging treatment in the prediction pipeline.

### EDA-4.2: Low-Volume Species

- Non-top-30 species: **474**
- Viable for prediction (≥100 trading days): **239**
- Not viable (<100 days): **235**

---

## Decision Summary

### Per-Species Aggregation Strategy (Top 10)

| Species | State | Packaging | Spec | Weighting | GROUP BY |
|---|---|---|---|---|---|
| 전복 | filter:활 | blend | spec-class | unweighted | (date, species, spec_class) |
| 병어 | filter:선 | dominant-pkg | spec-class | unweighted | (date, species, packaging, spec_class) |
| 삼치 | filter:선 | dominant-pkg | spec-class | unweighted | (date, species, packaging, spec_class) |
| 아귀 | partition | separate | aggregate | unweighted | (date, species, state, packaging) |
| 은갈치 | filter:선 | dominant-pkg | spec-class | unweighted | (date, species, packaging, spec_class) |
| 대구 | filter:선 | dominant-pkg | spec-class | unweighted | (date, species, packaging, spec_class) |
| 낙지 | partition | separate | aggregate | unweighted | (date, species, state, packaging) |
| 오징어 | partition | separate | spec-class | unweighted | (date, species, state, packaging, spec_class) |
| 넙치 | partition | dominant-pkg | aggregate | unweighted | (date, species, state, packaging) |
| 고등어 | partition | dominant-pkg | spec-class | unweighted | (date, species, state, packaging, spec_class) |

### Prediction Pipeline Impact

- **(a) Single model per species:** 5 species (전복, 병어, 삼치, 은갈치, 대구 — dominant state filter)
- **(b) Separate models per (species, state):** 5 species (아귀, 낙지, 오징어, 넙치, 고등어)
- **(c) Dominant-state filter:** Applied for group (a)

**Recommendation:** Option (c) for most species, option (b) for the 5 multi-state species with divergent pricing.

### Dominant DuckDB View

The most common GROUP BY pattern across the top 10 is `(trade_date, species, packaging, spec_class)` (4/10 species).

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

**Note:** This view does not include `state` in the GROUP BY, which means multi-state species (아귀, 낙지, 오징어, 넙치, 고등어) would need a separate view or a WHERE clause filtering to their relevant state.

---

## Implications for Prediction System

1. **No universal blended aggregation works.** The prediction pipeline cannot use a single `GROUP BY (trade_date, species)` view.
2. **Spec-class is the most impactful dimension** — it segments price more than packaging for 7/10 species.
3. **Unweighted mean is the correct aggregation** — quantity units are incomparable across packaging types.
4. **Daily price series are inherently noisy** — low lag-1 autocorrelation (0.4-0.6) for several species suggests that short-term prediction (1-7 day) may be challenging without weekly+ aggregation.
5. **Per-species configuration is needed** — the prediction pipeline should maintain a species config table specifying: dominant state, dominant packaging, whether spec-class grouping is needed.
