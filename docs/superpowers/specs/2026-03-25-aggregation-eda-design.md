# EDA Design: Row Aggregation Viability for Price Prediction

## Context

The raw Parquet dataset contains ~2.59M rows across 6,739 trading days (~415 rows/day median). Each row represents a unique auction lot: a combination of (species, state, origin, spec, packaging) with quantity and price data. No duplicates exist within a day.

The prediction system (`docs/06_prediction_system.md`) needs one daily price signal per species for time-series models (Exp. Smoothing, ARIMA, Prophet). The MCP server (`docs/04_mcp_server_design.md`) needs both detailed and aggregated views.

**Goal:** Determine whether rows can be aggregated into a single daily price per species via DuckDB queries at runtime, without maintaining redundant materialized tables.

**Key question:** Does combining rows across different states, packaging types, spec grades, and origins produce a statistically meaningful price signal, or does it introduce noise that would degrade prediction accuracy?

### Downstream Consumer Alignment

The prediction pipeline (`06_prediction_system.md`) calls `get_price_history(species_id, days)` and returns a flat `numpy` array — no state or packaging dimension. The existing `v_daily_summary` view uses `GROUP BY (trade_date, species)` with no state dimension. This EDA must determine whether that is sufficient, or whether state must be added as a partition dimension — which would require the prediction pipeline to either:
- **(a)** Train separate models per (species, state), or
- **(b)** Pick the dominant state per species and filter to it

The EDA's deliverables will include a concrete recommendation on this point.

---

## Approach: Hybrid — Unit Census + Signal Test

### Phase 1: Data Census

#### EDA-1.0: State Distribution Per Species

For each species, compute the percentage of rows in each state (선/활/냉/가공/냉건/건/NULL).

- **Hypothesis:** Most species are dominated by a single state (>90% of rows), so state partitioning adds little value for prediction for those species.
- **Assertion:** If a species has >90% of rows in one state, that state is canonical and the others can be excluded from prediction for that species.
- **Output:** Table of (species, state, row_pct, row_count). Count of single-state-dominated species vs multi-state species.
- **Follow-up:** For multi-state species (no state >90%), test whether price distributions across states diverge (mean price ratio between states). If they diverge >50%, state must be a partition dimension for those species.

#### EDA-1.1: Packaging Dominance Per Species

For each species, compute the percentage of rows using each packaging type. Rows with NULL packaging are excluded (<0.01% of data).

- **Hypothesis:** Most species are dominated by 1-2 packaging types (>80% of rows).
- **Assertion:** If a species has >80% of rows in one packaging type, that type is "canonical" for that species.
- **Output:** Table of (species, packaging, row_pct, rank) for all species. Count of species meeting the 80% threshold.

#### EDA-1.2: Spec Type Taxonomy

Classify all `spec` values into categories:
- **Size grade:** `대`, `중`, `소`, `특대`
- **Count-based:** `N미` (e.g., `20미`, `8미`)
- **Weight range:** `300/400`, `200/300`
- **Count range:** `10/11미`, `7/8미`
- **Other:** `바라`, `진통`, `3단`, etc.

Per species, how many spec categories are present?

- **Hypothesis:** Spec variety is high, but specs within the same packaging type share comparable price ranges.
- **Output:** Spec taxonomy table. Per-species spec category count.

#### EDA-1.3: Origin Distribution Per Species

Per species per day, how many distinct origins appear?

- **Hypothesis:** Origin is the primary row-multiplier — the same fish in the same spec/packaging arrives from multiple ports.
- **Output:** Distribution of origin count per (species, day).

---

### Phase 2: Price Coherence Tests

All Phase 2 tests operate on the top 10 species by row count. Rows with NULL values in the tested dimension are excluded.

#### EDA-2.1: Intra-Day Price Spread by Packaging

For the top 10 species, on each trading day:
1. Compute `price_avg` mean and standard deviation **within** each packaging type
2. Compute `price_avg` mean and standard deviation **across** all packaging types
3. Calculate coefficient of variation (CV = std/mean) for within-packaging and across-packaging

- **Hypothesis:** Within-packaging CV is much smaller than across-packaging CV. If true, packaging types represent genuinely different price bands.
- **Assertion:**
  - `across_CV > 1.5 * within_CV` → Packaging meaningfully segments price → blended aggregation is suspect
  - `across_CV <= 1.5 * within_CV` → Packaging does not meaningfully segment price → blended aggregation is valid
- **Threshold rationale:** 1.5x is chosen because ARIMA forecast MAPE at 7-day horizon is typically 6-10% for this data class; a 50% noise increase would degrade forecast accuracy by ~3-5 percentage points, which is material. The notebook will include a sensitivity sweep at 1.5x, 2x, and 2.5x to visualize the tradeoff.
- **Output:** Per-species table of (median within_CV, median across_CV, ratio).

#### EDA-2.1b: Intra-Day Price Spread by Spec

For the top 10 species, within each (species, state, packaging) group on each day:
1. Compute `price_avg` mean and standard deviation across different spec values
2. Calculate CV within-spec vs across-spec

- **Hypothesis:** Spec segments price within a packaging type (e.g., `대` mackerel costs more than `소` mackerel).
- **Assertion:**
  - If across-spec CV > 1.5× within-spec CV → Spec is a price-segmenting dimension → aggregation across specs adds noise
  - If across-spec CV ≤ 1.5× → Spec variation is moderate → can aggregate across specs
- **Note:** If spec segments price significantly, the aggregation GROUP BY needs to include a spec-class dimension (size grade or count bracket), which affects the row-reduction ratio. The decision tree addresses this branch.
- **Output:** Per-species table of spec price impact.

#### EDA-2.2: Per-Packaging Time Series Correlation

For top 10 species that have 2+ packaging types each with >100 trading days of data:
1. Compute daily quantity-weighted average price per packaging type
2. Also compute simple (unweighted) mean as a comparison — see note on quantity units below
3. Align the two series by date (inner join on trading days)
4. Compute Pearson correlation
5. **Temporal stability check:** Compute correlation for the first half and second half of the date range separately. If the difference exceeds 0.15, flag the species as having an unstable packaging relationship.

- **Hypothesis:** If correlation > 0.85, the packaging-level series move together and blending is safe.
- **Assertion:**
  - `corr > 0.85` → Co-movement confirmed → blended aggregation safe
  - `corr <= 0.85` → Series diverge → must separate by packaging or use dominant-packaging strategy
- **Note on high-correlation offset:** If correlation is high (>0.85) but the mean price ratio between packaging types exceeds 1.5x, this means the series co-move at different price levels. A blended average would be dominated by the higher-priced packaging. In this case, dominant-packaging filtering is preferred over blending.
- **Output:** Per-species correlation matrix between packaging types, with temporal stability flags.

#### EDA-2.3: Origin Price Spread

For cases where (species, state, packaging, spec) are identical and only origin varies:
1. Compute the spread: `(max_price - min_price) / mean_price` per group per day
2. Aggregate across all days

- **Hypothesis:** Origin explains modest price variation (10-20%), not fundamental price-level differences.
- **Assertion:**
  - Median spread < 30% → Origin can safely be aggregated away
  - Median spread >= 30% → Origin carries material price information
- **Output:** Distribution of origin-driven spread across species.

#### EDA-2.4: Quantity Unit Comparability

The `quantity` field means different things for different packaging types: kg for weight-based, count for 미-based specs, units for box/S/P. Quantity-weighting across incompatible units could produce meaningless averages.

For the top 10 species:
1. Group rows by (species, packaging)
2. Compare the distribution of `quantity` values across packaging types
3. Compute both quantity-weighted and unweighted (simple mean) daily price aggregations
4. Measure the divergence between them

- **Hypothesis:** For species dominated by a single packaging type, weighted and unweighted produce similar results. For multi-packaging species, they may diverge.
- **Assertion:** If weighted vs unweighted daily price correlation > 0.98, quantity-weighting is safe. If < 0.98, use unweighted mean or filter to dominant packaging.
- **Output:** Per-species (weighted_vs_unweighted_corr, dominant_pkg_pct) table.

---

### Phase 3: Aggregation Viability

#### EDA-3.1: Blended vs Dominant-Packaging Comparison

For the top 10 species, produce two daily time series:
- **(a) Blended:** aggregated average of `price_avg` across all rows (weighting method chosen based on EDA-2.4 results)
- **(b) Dominant-only:** aggregated average using only the most common packaging type

Measure:
- Pearson correlation between (a) and (b)
- Lag-1 autocorrelation of each series (smoothness proxy)

- **Hypothesis:** The two series are highly correlated (>0.95).
- **Assertion:**
  - `corr > 0.95` AND both `lag1_autocorr > 0.8` → Blended aggregation is sufficient for prediction
  - `corr <= 0.95` → Investigate: likely driven by packaging composition shifts over time. Use dominant-packaging strategy.
- **Output:** Per-species (corr, lag1_blended, lag1_dominant) table.

#### EDA-3.2: Row Reduction Ratio

Compute row counts per day under each aggregation strategy:

| Strategy | GROUP BY |
|---|---|
| Raw | (no aggregation) |
| Per species | `(trade_date, species)` |
| Per species+state | `(trade_date, species, state)` |
| Per species+state+packaging | `(trade_date, species, state, packaging)` |
| Per species+state+packaging+origin | `(trade_date, species, state, packaging, origin)` |

- **Output:** Table of (strategy, median_rows_per_day, total_rows, compression_ratio).

---

### Phase 4: Edge Cases & Outliers

#### EDA-4.1: Heterogeneous-Packaging Species

Identify species where no single packaging type exceeds 50% of rows.

- These are candidates that may need per-packaging treatment or special handling in the prediction pipeline.
- **Output:** List of species with their packaging distribution.

#### EDA-4.2: Low-Volume Species Threshold

For species outside the top 30 by total row count:
- How many distinct trading days does each appear?
- What is the longest consecutive trading day streak?

- **Assertion:** Species with <100 total trading days are excluded from prediction (insufficient data for stable parameter estimation of weekly-seasonal models). Note: the prediction pipeline applies its own runtime minimum of 30 data points (`len(prices) < 30`); the 100-day threshold here is stricter because this EDA evaluates whether aggregation analysis itself is reliable. Species passing 100 total days but having <30 in the model's 365-day lookback window would only occur for species that stopped trading, which the pipeline already handles gracefully.
- **Output:** Species viability table with (species, total_days, max_streak, viable_flag).

---

## Decision Tree

```
EDA-1.0: State test
├── Species has >90% in one state
│   → Use dominant state only (filter)
│   → Proceed to packaging tests below
│
└── Species is multi-state (no state >90%)
    → Check mean price ratio between states
    ├── Ratio > 1.5× → State segments price
    │   → Prediction: train separate models per (species, state)
    │   → Or: pick dominant state only
    └── Ratio ≤ 1.5× → State doesn't segment price
        → Can group across states

EDA-2.1: Packaging CV ratio test
├── across_CV ≤ 1.5× within_CV
│   → Packaging doesn't segment price
│   → Confirm with EDA-3.1 (corr > 0.95)
│       ├── Confirmed → Blended aggregation valid
│       │   → DuckDB: GROUP BY (trade_date, species[, state if needed])
│       └── Not confirmed (corr ≤ 0.95)
│           → Investigate compositional shift
│           → Fallback: dominant-packaging strategy
│
└── across_CV > 1.5× within_CV
    → Packaging segments price
    → Check EDA-2.2: correlation test
        ├── corr > 0.85 AND mean price ratio < 1.5×
        │   → Series co-move at similar levels
        │   → Blended still OK
        │
        ├── corr > 0.85 AND mean price ratio ≥ 1.5×
        │   → Series co-move but at different price levels
        │   → Use dominant-packaging filtering (not blending)
        │
        └── corr ≤ 0.85
            → Independent signals per packaging
            → Use dominant-packaging strategy
            → DuckDB: GROUP BY (trade_date, species, state, packaging)

EDA-2.1b: Spec CV test (within dominant packaging)
├── across-spec CV ≤ 1.5× → Spec can be aggregated away
└── across-spec CV > 1.5× → Consider spec-class grouping
    → Group specs into brackets (대/중/소, or count ranges)
    → Adds one more GROUP BY dimension
```

## Deliverables

1. **Marimo notebook** (`notebooks/eda_aggregation.py`) — all EDA steps with visualizations
2. **Decision summary** — which aggregation strategy is validated, per species; explicit recommendation on whether state is a required partition dimension
3. **DuckDB view definition** — the validated aggregation query for the prediction pipeline, reconciled with the existing `v_daily_summary` view design
4. **Prediction pipeline impact** — concrete recommendation: (a) one model per species, (b) one model per (species, state), or (c) dominant-state filtering; with rationale from EDA-1.0
5. **Update to `docs/08_data_preprocessing.md`** — aggregation rules added as a new section

## Implementation Notes

- Use `pyarrow.dataset` for reading Parquet (already available, no new deps)
- Use DuckDB for aggregation queries if installed, else pure pyarrow/Python
- Top 10 species for Phase 2-3 selected by total row count
- Rows with NULL values in state/packaging/spec are excluded from Phase 2 tests (they represent <0.01% of data, per `docs/08_data_preprocessing.md`). The final aggregation view will include a note on NULL handling.
- All assertions use thresholds that can be adjusted; the notebook will include sensitivity sweeps for the CV ratio (1.5x, 2x, 2.5x) and correlation thresholds (0.8, 0.85, 0.9)
- Temporal stability: EDA-2.2 includes split-half correlation checks. If packaging relationships are unstable over the 20-year dataset, the prediction pipeline (which uses a 365-day lookback) is already insulated from this — but it will be documented.
