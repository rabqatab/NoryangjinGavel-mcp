# Aggregation EDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Marimo notebook that executes all 12 EDA steps from the aggregation viability spec, produces visualizations, validates hypotheses, and outputs a concrete aggregation strategy for the prediction pipeline.

**Architecture:** Single Marimo notebook (`notebooks/eda_aggregation.py`) with one cell per EDA step, using `pyarrow.dataset` for Parquet reads and Python/numpy for statistics. Each cell produces a result table and/or chart. A final summary cell applies the decision tree and outputs the recommended DuckDB view definition.

**Tech Stack:** marimo, pyarrow, numpy

**Prerequisite:** Add `"numpy>=1.26.0"` to `dependencies` in `pyproject.toml` before starting (not currently listed).

**Spec:** `docs/superpowers/specs/2026-03-25-aggregation-eda-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `notebooks/eda_aggregation.py` | Marimo notebook — all EDA cells, visualizations, decision logic |
| `notebooks/eda_helpers.py` | Shared helper functions (data loading, CV calculation, spec classification, correlation) |
| `docs/08_data_preprocessing.md` | Update: add aggregation rules section with EDA findings |

No test files — this is an analytical notebook, not library code. Validation is done via assertions within the notebook cells themselves.

---

### Task 1: Project Setup and Data Loading Helpers

**Files:**
- Create: `notebooks/eda_helpers.py`
- Create: `notebooks/eda_aggregation.py` (initial skeleton)

- [ ] **Step 0: Add numpy dependency**

```bash
uv add "numpy>=1.26.0"
```

Expected: `pyproject.toml` updated, `uv.lock` regenerated.

- [ ] **Step 1: Create notebooks directory and helper module**

```python
# notebooks/eda_helpers.py
"""Shared helpers for aggregation EDA notebook."""
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

DATA_ROOT = Path(__file__).parent.parent / "data" / "parquet" / "prices"

# Column set we need for EDA (avoid loading year/month partition cols)
EDA_COLUMNS = [
    "trade_date", "species", "state", "origin", "spec",
    "packaging", "quantity", "price_high", "price_low", "price_avg",
]


def load_all_data() -> dict[str, list]:
    """Load all parquet data as a dict of column lists."""
    dataset = ds.dataset(str(DATA_ROOT), format="parquet", partitioning="hive")
    table = dataset.to_table(columns=EDA_COLUMNS)
    return {col: table.column(col).to_pylist() for col in EDA_COLUMNS}


def load_top_species(data: dict, n: int = 10) -> list[str]:
    """Return top N species by row count."""
    from collections import Counter
    counts = Counter(data["species"])
    return [name for name, _ in counts.most_common(n)]


def cv(values: list[float]) -> float:
    """Coefficient of variation. Returns 0 if mean is 0."""
    arr = np.array(values, dtype=float)
    mean = np.mean(arr)
    if mean == 0:
        return 0.0
    return float(np.std(arr) / mean)


def weighted_avg(prices: list[int], quantities: list[float]) -> float:
    """Quantity-weighted average price."""
    p = np.array(prices, dtype=float)
    q = np.array(quantities, dtype=float)
    total_q = q.sum()
    if total_q == 0:
        return float(np.mean(p))
    return float(np.dot(p, q) / total_q)


def pearson_corr(a: list[float], b: list[float]) -> float:
    """Pearson correlation coefficient."""
    arr_a = np.array(a)
    arr_b = np.array(b)
    if len(arr_a) < 3:
        return 0.0
    corr_matrix = np.corrcoef(arr_a, arr_b)
    return float(corr_matrix[0, 1])


def lag1_autocorr(values: list[float]) -> float:
    """Lag-1 autocorrelation of a time series."""
    arr = np.array(values, dtype=float)
    if len(arr) < 3:
        return 0.0
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


# Spec classification regex patterns
_SPEC_SIZE_GRADE = re.compile(r"^(특대|대|중|소)$")
_SPEC_COUNT = re.compile(r"^\d+미$")
_SPEC_WEIGHT_RANGE = re.compile(r"^\d+/\d+$")
_SPEC_COUNT_RANGE = re.compile(r"^\d+/\d+미$")


def classify_spec(spec: Optional[str]) -> str:
    """Classify a spec value into a category."""
    if spec is None:
        return "null"
    if _SPEC_SIZE_GRADE.match(spec):
        return "size_grade"
    if _SPEC_COUNT.match(spec):
        return "count"
    if _SPEC_COUNT_RANGE.match(spec):
        return "count_range"
    if _SPEC_WEIGHT_RANGE.match(spec):
        return "weight_range"
    return "other"
```

- [ ] **Step 2: Create Marimo notebook skeleton**

```python
# notebooks/eda_aggregation.py
import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def setup():
    import marimo as mo
    import numpy as np
    from collections import Counter, defaultdict

    from eda_helpers import (
        load_all_data, load_top_species, cv, weighted_avg,
        pearson_corr, lag1_autocorr, classify_spec,
    )

    mo.md("# Aggregation EDA: Row Integration Viability")
    return mo, np, Counter, defaultdict, load_all_data, load_top_species, cv, weighted_avg, pearson_corr, lag1_autocorr, classify_spec


@app.cell
def load_data(load_all_data, load_top_species, mo):
    mo.md("## Data Loading")
    data = load_all_data()
    n_rows = len(data["trade_date"])
    top10 = load_top_species(data, 10)
    mo.md(f"Loaded **{n_rows:,}** rows. Top 10 species: {', '.join(top10)}")
    return data, n_rows, top10


if __name__ == "__main__":
    app.run()
```

- [ ] **Step 3: Verify notebook loads**

Run: `cd /home/alphabridge/PythonProjects/NoryangjinGavel-mcp && uv run marimo edit notebooks/eda_aggregation.py --headless --no-token`
Expected: Server starts, no import errors. Ctrl+C to stop.

- [ ] **Step 4: Commit**

```bash
git add notebooks/eda_helpers.py notebooks/eda_aggregation.py
git commit -m "feat: add EDA notebook skeleton and helpers for aggregation analysis"
```

---

### Task 2: Phase 1 — EDA-1.0 State Distribution

**Files:**
- Modify: `notebooks/eda_aggregation.py`

- [ ] **Step 1: Add EDA-1.0 cell**

Append this cell to the notebook:

```python
@app.cell
def eda_1_0_state(data, Counter, defaultdict, np, mo):
    """EDA-1.0: State distribution per species."""
    species_state_counts = defaultdict(Counter)
    for i in range(len(data["trade_date"])):
        sp = data["species"][i]
        st = data["state"][i] or "(null)"
        species_state_counts[sp][st] += 1

    # For each species, find dominant state and its percentage
    state_summary = []
    multi_state_species = []
    for sp, counts in sorted(species_state_counts.items(), key=lambda x: -sum(x[1].values())):
        total = sum(counts.values())
        dominant_state, dominant_count = counts.most_common(1)[0]
        dominant_pct = dominant_count / total * 100
        state_summary.append({
            "species": sp,
            "dominant_state": dominant_state,
            "dominant_pct": round(dominant_pct, 1),
            "total_rows": total,
            "n_states": len(counts),
        })
        if dominant_pct < 90:
            multi_state_species.append(sp)

    single_state = sum(1 for s in state_summary if s["dominant_pct"] >= 90)
    # For multi-state species, check price divergence
    multi_state_prices = {}
    if multi_state_species:
        for sp in multi_state_species:
            state_prices = defaultdict(list)
            for i in range(len(data["trade_date"])):
                if data["species"][i] == sp and data["state"][i]:
                    state_prices[data["state"][i]].append(data["price_avg"][i])

            means = {st: np.mean(prices) for st, prices in state_prices.items() if len(prices) > 10}
            if len(means) >= 2:
                vals = list(means.values())
                ratio = max(vals) / min(vals) if min(vals) > 0 else 0
                multi_state_prices[sp] = {"means": means, "ratio": round(ratio, 2)}

    divergent = {sp: v for sp, v in multi_state_prices.items() if v["ratio"] > 1.5}
    parts = [
        mo.md("## Phase 1: Data Census\n### EDA-1.0: State Distribution Per Species"),
        mo.md(
            f"**Result:** {single_state}/{len(state_summary)} species ({single_state/len(state_summary)*100:.0f}%) "
            f"have >90% rows in one state.\n\n"
            f"**Multi-state species ({len(multi_state_species)}):** {', '.join(multi_state_species[:20])}"
            f"{'...' if len(multi_state_species) > 20 else ''}"
        ),
        mo.md(
            f"**Multi-state price divergence (ratio > 1.5×):** {len(divergent)} species\n\n"
            + "\n".join(f"- {sp}: ratio={v['ratio']}× — {v['means']}" for sp, v in sorted(divergent.items()))
        ) if divergent else mo.md("No multi-state species with price divergence > 1.5×."),
    ]
    mo.vstack(parts)

    return state_summary, multi_state_species, multi_state_prices
```

- [ ] **Step 2: Run notebook and verify cell executes**

Run: `uv run marimo run notebooks/eda_aggregation.py 2>&1 | head -5`
Expected: No errors. Output shows state distribution stats.

- [ ] **Step 3: Commit**

```bash
git add notebooks/eda_aggregation.py
git commit -m "feat(eda): add EDA-1.0 state distribution per species"
```

---

### Task 3: Phase 1 — EDA-1.1 Packaging Dominance

**Files:**
- Modify: `notebooks/eda_aggregation.py`

- [ ] **Step 1: Add EDA-1.1 cell**

```python
@app.cell
def eda_1_1_packaging(data, Counter, defaultdict, mo):
    """EDA-1.1: Packaging dominance per species."""
    mo.md("### EDA-1.1: Packaging Dominance Per Species")

    species_pkg_counts = defaultdict(Counter)
    for i in range(len(data["trade_date"])):
        pkg = data["packaging"][i]
        if pkg is None:
            continue
        species_pkg_counts[data["species"][i]][pkg] += 1

    pkg_summary = []
    for sp, counts in sorted(species_pkg_counts.items(), key=lambda x: -sum(x[1].values())):
        total = sum(counts.values())
        dominant_pkg, dominant_count = counts.most_common(1)[0]
        dominant_pct = dominant_count / total * 100
        pkg_summary.append({
            "species": sp,
            "dominant_pkg": dominant_pkg,
            "dominant_pct": round(dominant_pct, 1),
            "total_rows": total,
            "n_pkg_types": len(counts),
        })

    above_80 = sum(1 for s in pkg_summary if s["dominant_pct"] >= 80)
    mo.md(
        f"**Result:** {above_80}/{len(pkg_summary)} species ({above_80/len(pkg_summary)*100:.0f}%) "
        f"have >80% rows in one packaging type.\n\n"
        f"**Top 20 species packaging:**\n\n"
        + "\n".join(
            f"| {s['species']} | {s['dominant_pkg']} | {s['dominant_pct']}% | {s['n_pkg_types']} types |"
            for s in pkg_summary[:20]
        )
    )

    return pkg_summary,
```

- [ ] **Step 2: Run and verify**

Run: `uv run marimo run notebooks/eda_aggregation.py 2>&1 | head -5`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add notebooks/eda_aggregation.py
git commit -m "feat(eda): add EDA-1.1 packaging dominance per species"
```

---

### Task 4: Phase 1 — EDA-1.2 Spec Taxonomy + EDA-1.3 Origin Distribution

**Files:**
- Modify: `notebooks/eda_aggregation.py`

- [ ] **Step 1: Add EDA-1.2 cell**

```python
@app.cell
def eda_1_2_spec(data, Counter, defaultdict, classify_spec, mo):
    """EDA-1.2: Spec type taxonomy."""
    mo.md("### EDA-1.2: Spec Type Taxonomy")

    # Global spec classification
    all_specs = [data["spec"][i] for i in range(len(data["trade_date"]))]
    spec_classes = Counter(classify_spec(s) for s in all_specs)
    mo.md(
        "**Spec categories (global):**\n\n"
        + "\n".join(f"- {k}: {v:,} rows" for k, v in spec_classes.most_common())
    )

    # Per species: how many spec categories?
    species_spec_cats = defaultdict(set)
    for i in range(len(data["trade_date"])):
        sp = data["species"][i]
        cat = classify_spec(data["spec"][i])
        species_spec_cats[sp].add(cat)

    cat_counts = Counter(len(cats) for cats in species_spec_cats.values())
    mo.md(
        f"\n**Spec categories per species:**\n\n"
        + "\n".join(f"- {n} categories: {c} species" for n, c in sorted(cat_counts.items()))
    )

    return spec_classes, species_spec_cats
```

- [ ] **Step 2: Add EDA-1.3 cell**

```python
@app.cell
def eda_1_3_origin(data, Counter, defaultdict, np, mo):
    """EDA-1.3: Origin distribution per species per day."""
    mo.md("### EDA-1.3: Origin Distribution Per Species")

    # Count distinct origins per (species, day)
    day_species_origins = defaultdict(set)
    for i in range(len(data["trade_date"])):
        key = (data["trade_date"][i], data["species"][i])
        origin = data["origin"][i]
        if origin:
            day_species_origins[key].add(origin)

    origin_counts = [len(v) for v in day_species_origins.values()]
    arr = np.array(origin_counts)
    mo.md(
        f"**Origins per (species, day):**\n"
        f"- Min: {arr.min()}, Max: {arr.max()}, Median: {np.median(arr):.0f}, Mean: {arr.mean():.1f}\n"
        f"- p25={np.percentile(arr, 25):.0f}, p75={np.percentile(arr, 75):.0f}, p95={np.percentile(arr, 95):.0f}\n\n"
        f"**Hypothesis check:** Origin is the primary row-multiplier → "
        f"median origins per species-day = {np.median(arr):.0f}"
    )

    return origin_counts,
```

- [ ] **Step 3: Run and verify**

Run: `uv run marimo run notebooks/eda_aggregation.py 2>&1 | head -5`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/eda_aggregation.py
git commit -m "feat(eda): add EDA-1.2 spec taxonomy and EDA-1.3 origin distribution"
```

---

### Task 5: Phase 2 — EDA-2.1 Packaging CV + EDA-2.1b Spec CV

**Files:**
- Modify: `notebooks/eda_aggregation.py`

- [ ] **Step 1: Add EDA-2.1 cell**

```python
@app.cell
def eda_2_1_packaging_cv(data, top10, defaultdict, np, cv, mo):
    """EDA-2.1: Intra-day price spread by packaging."""
    mo.md("## Phase 2: Price Coherence Tests\n### EDA-2.1: Intra-Day Price Spread by Packaging")

    # Build per-(species, date, packaging) price lists
    day_pkg_prices = defaultdict(lambda: defaultdict(list))
    day_all_prices = defaultdict(list)

    for i in range(len(data["trade_date"])):
        sp = data["species"][i]
        if sp not in top10:
            continue
        pkg = data["packaging"][i]
        if pkg is None:
            continue
        key = (sp, data["trade_date"][i])
        day_pkg_prices[key][pkg].append(data["price_avg"][i])
        day_all_prices[key].append(data["price_avg"][i])

    # Compute CV ratios per species
    cv_results = {}
    for sp in top10:
        within_cvs = []
        across_cvs = []
        for key in day_all_prices:
            if key[0] != sp:
                continue
            # Across-packaging CV
            all_prices = day_all_prices[key]
            if len(all_prices) >= 3:
                across_cvs.append(cv(all_prices))

            # Within-packaging CV (average across packaging types)
            pkg_cvs = []
            for pkg, prices in day_pkg_prices[key].items():
                if len(prices) >= 2:
                    pkg_cvs.append(cv(prices))
            if pkg_cvs:
                within_cvs.append(np.mean(pkg_cvs))

        if within_cvs and across_cvs:
            med_within = np.median(within_cvs)
            med_across = np.median(across_cvs)
            ratio = med_across / med_within if med_within > 0 else 0
            cv_results[sp] = {
                "within_cv": round(med_within, 4),
                "across_cv": round(med_across, 4),
                "ratio": round(ratio, 2),
            }

    mo.md(
        "| Species | Within-Pkg CV | Across-Pkg CV | Ratio | Verdict |\n"
        "|-|-|-|-|-|\n"
        + "\n".join(
            f"| {sp} | {v['within_cv']} | {v['across_cv']} | {v['ratio']}× | "
            f"{'SEGMENTS' if v['ratio'] > 1.5 else 'OK'} |"
            for sp, v in cv_results.items()
        )
        + f"\n\n**Threshold: 1.5×.** "
        f"Species where packaging segments price: "
        f"{sum(1 for v in cv_results.values() if v['ratio'] > 1.5)}/{len(cv_results)}"
    )

    return cv_results,
```

- [ ] **Step 2: Add sensitivity sweep cell**

```python
@app.cell
def eda_2_1_sensitivity(cv_results, mo):
    """Sensitivity sweep for CV ratio thresholds."""
    mo.md("#### Sensitivity Sweep: CV Ratio Thresholds")

    thresholds = [1.5, 2.0, 2.5]
    lines = ["| Threshold | Species Segmented | Species OK |", "|-|-|-|"]
    for t in thresholds:
        seg = sum(1 for v in cv_results.values() if v["ratio"] > t)
        ok = len(cv_results) - seg
        lines.append(f"| {t}× | {seg} | {ok} |")

    mo.md("\n".join(lines))
    return
```

- [ ] **Step 3: Add EDA-2.1b cell**

```python
@app.cell
def eda_2_1b_spec_cv(data, top10, defaultdict, np, cv, mo):
    """EDA-2.1b: Intra-day price spread by spec (within packaging)."""
    mo.md("### EDA-2.1b: Intra-Day Price Spread by Spec")

    # Build per-(species, date, state, packaging, spec) price lists
    group_prices = defaultdict(lambda: defaultdict(list))
    pkg_all_prices = defaultdict(list)

    for i in range(len(data["trade_date"])):
        sp = data["species"][i]
        if sp not in top10:
            continue
        pkg = data["packaging"][i]
        spec_val = data["spec"][i]
        if pkg is None or spec_val is None:
            continue
        outer_key = (sp, data["trade_date"][i], data["state"][i], pkg)
        group_prices[outer_key][spec_val].append(data["price_avg"][i])
        pkg_all_prices[outer_key].append(data["price_avg"][i])

    spec_cv_results = {}
    for sp in top10:
        within_cvs = []
        across_cvs = []
        for key in pkg_all_prices:
            if key[0] != sp:
                continue
            all_prices = pkg_all_prices[key]
            if len(all_prices) >= 3:
                across_cvs.append(cv(all_prices))
            spec_cvs = []
            for spec_val, prices in group_prices[key].items():
                if len(prices) >= 2:
                    spec_cvs.append(cv(prices))
            if spec_cvs:
                within_cvs.append(np.mean(spec_cvs))

        if within_cvs and across_cvs:
            med_within = np.median(within_cvs)
            med_across = np.median(across_cvs)
            ratio = med_across / med_within if med_within > 0 else 0
            spec_cv_results[sp] = {
                "within_cv": round(med_within, 4),
                "across_cv": round(med_across, 4),
                "ratio": round(ratio, 2),
            }

    mo.md(
        "| Species | Within-Spec CV | Across-Spec CV | Ratio | Verdict |\n"
        "|-|-|-|-|-|\n"
        + "\n".join(
            f"| {sp} | {v['within_cv']} | {v['across_cv']} | {v['ratio']}× | "
            f"{'SEGMENTS' if v['ratio'] > 1.5 else 'OK'} |"
            for sp, v in spec_cv_results.items()
        )
    )

    return spec_cv_results,
```

- [ ] **Step 3: Run and verify**

Run: `uv run marimo run notebooks/eda_aggregation.py 2>&1 | head -5`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/eda_aggregation.py
git commit -m "feat(eda): add EDA-2.1 packaging CV and EDA-2.1b spec CV tests"
```

---

### Task 6: Phase 2 — EDA-2.2 Per-Packaging Correlation + EDA-2.3 Origin Spread

**Files:**
- Modify: `notebooks/eda_aggregation.py`

- [ ] **Step 1: Add EDA-2.2 cell**

```python
@app.cell
def eda_2_2_pkg_correlation(data, top10, defaultdict, np, weighted_avg, pearson_corr, mo):
    """EDA-2.2: Per-packaging time series correlation."""
    mo.md("### EDA-2.2: Per-Packaging Time Series Correlation")

    # Build daily weighted avg price per (species, packaging)
    day_pkg_data = defaultdict(lambda: defaultdict(lambda: {"prices": [], "quantities": []}))
    for i in range(len(data["trade_date"])):
        sp = data["species"][i]
        if sp not in top10:
            continue
        pkg = data["packaging"][i]
        if pkg is None:
            continue
        d = data["trade_date"][i]
        day_pkg_data[(sp, pkg)][d]["prices"].append(data["price_avg"][i])
        day_pkg_data[(sp, pkg)][d]["quantities"].append(data["quantity"][i])

    # Compute daily series per (species, packaging)
    daily_series = {}
    for (sp, pkg), day_map in day_pkg_data.items():
        if len(day_map) < 100:
            continue
        series = {}
        for d, vals in sorted(day_map.items()):
            series[d] = {
                "weighted": weighted_avg(vals["prices"], vals["quantities"]),
                "simple": np.mean(vals["prices"]),
            }
        daily_series[(sp, pkg)] = series

    # For each species, correlate top-2 packaging types
    pkg_corr_results = {}
    for sp in top10:
        sp_pkgs = [(k, v) for k, v in daily_series.items() if k[0] == sp]
        sp_pkgs.sort(key=lambda x: -len(x[1]))
        if len(sp_pkgs) < 2:
            continue
        (_, pkg_a), series_a = sp_pkgs[0]
        (_, pkg_b), series_b = sp_pkgs[1]

        common_dates = sorted(set(series_a.keys()) & set(series_b.keys()))
        if len(common_dates) < 30:
            continue

        a_vals = [series_a[d]["weighted"] for d in common_dates]
        b_vals = [series_b[d]["weighted"] for d in common_dates]
        corr = pearson_corr(a_vals, b_vals)

        # Also compute simple (unweighted) correlation for comparison
        a_simple = [series_a[d]["simple"] for d in common_dates]
        b_simple = [series_b[d]["simple"] for d in common_dates]
        corr_simple = pearson_corr(a_simple, b_simple)

        # Price ratio
        mean_a = np.mean(a_vals)
        mean_b = np.mean(b_vals)
        price_ratio = max(mean_a, mean_b) / min(mean_a, mean_b) if min(mean_a, mean_b) > 0 else 0

        # Temporal stability: split-half
        mid = len(common_dates) // 2
        corr_first = pearson_corr(a_vals[:mid], b_vals[:mid])
        corr_second = pearson_corr(a_vals[mid:], b_vals[mid:])
        stable = abs(corr_first - corr_second) <= 0.15

        pkg_corr_results[sp] = {
            "pkg_a": pkg_a, "pkg_b": pkg_b,
            "corr_weighted": round(corr, 3),
            "corr_simple": round(corr_simple, 3),
            "price_ratio": round(price_ratio, 2),
            "corr_first_half": round(corr_first, 3),
            "corr_second_half": round(corr_second, 3),
            "stable": stable,
            "common_days": len(common_dates),
        }

    mo.md(
        "| Species | Pkg A | Pkg B | Corr (W) | Corr (S) | Price Ratio | 1st-half | 2nd-half | Stable | Verdict |\n"
        "|-|-|-|-|-|-|-|-|-|-|\n"
        + "\n".join(
            f"| {sp} | {v['pkg_a']} | {v['pkg_b']} | {v['corr_weighted']} | {v['corr_simple']} | {v['price_ratio']}× | "
            f"{v['corr_first_half']} | {v['corr_second_half']} | {'Y' if v['stable'] else 'N'} | "
            f"{'BLEND OK' if v['corr_weighted'] > 0.85 and v['price_ratio'] < 1.5 else 'DOM-PKG' if v['corr_weighted'] > 0.85 else 'SEPARATE'} |"
            for sp, v in pkg_corr_results.items()
        )
    )

    return pkg_corr_results,
```

- [ ] **Step 2: Add EDA-2.3 cell**

```python
@app.cell
def eda_2_3_origin_spread(data, top10, defaultdict, np, mo):
    """EDA-2.3: Origin price spread."""
    mo.md("### EDA-2.3: Origin Price Spread")

    # Group by (species, date, state, packaging, spec) — only origin varies
    group_prices = defaultdict(list)
    for i in range(len(data["trade_date"])):
        sp = data["species"][i]
        if sp not in top10:
            continue
        key = (sp, data["trade_date"][i], data["state"][i], data["packaging"][i], data["spec"][i])
        group_prices[key].append(data["price_avg"][i])

    # Compute spread for groups with 2+ origins
    spreads_by_species = defaultdict(list)
    for key, prices in group_prices.items():
        if len(prices) < 2:
            continue
        mean_p = np.mean(prices)
        if mean_p == 0:
            continue
        spread = (max(prices) - min(prices)) / mean_p
        spreads_by_species[key[0]].append(spread)

    origin_spread_results = {}
    for sp in top10:
        if sp in spreads_by_species and spreads_by_species[sp]:
            arr = np.array(spreads_by_species[sp])
            origin_spread_results[sp] = {
                "median_spread": round(np.median(arr) * 100, 1),
                "mean_spread": round(np.mean(arr) * 100, 1),
                "p95_spread": round(np.percentile(arr, 95) * 100, 1),
                "n_groups": len(arr),
            }

    mo.md(
        "| Species | Median Spread | Mean Spread | p95 Spread | Groups | Verdict |\n"
        "|-|-|-|-|-|-|\n"
        + "\n".join(
            f"| {sp} | {v['median_spread']}% | {v['mean_spread']}% | {v['p95_spread']}% | {v['n_groups']} | "
            f"{'AGGREGATE OK' if v['median_spread'] < 30 else 'ORIGIN MATTERS'} |"
            for sp, v in origin_spread_results.items()
        )
    )

    return origin_spread_results,
```

- [ ] **Step 3: Run and verify**

Run: `uv run marimo run notebooks/eda_aggregation.py 2>&1 | head -5`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/eda_aggregation.py
git commit -m "feat(eda): add EDA-2.2 packaging correlation and EDA-2.3 origin spread"
```

---

### Task 7: Phase 2 — EDA-2.4 Quantity Unit Comparability

**Files:**
- Modify: `notebooks/eda_aggregation.py`

- [ ] **Step 1: Add EDA-2.4 cell**

```python
@app.cell
def eda_2_4_quantity(data, top10, pkg_summary, defaultdict, np, weighted_avg, pearson_corr, mo):
    """EDA-2.4: Quantity unit comparability."""
    mo.md("### EDA-2.4: Quantity Unit Comparability")

    # Lookup dominant packaging pct per species
    dom_pkg_pct = {s["species"]: s["dominant_pct"] for s in pkg_summary if s["species"] in top10}

    # Build per (species, date) weighted and unweighted daily averages
    day_data = defaultdict(lambda: {"prices": [], "quantities": []})
    for i in range(len(data["trade_date"])):
        sp = data["species"][i]
        if sp not in top10:
            continue
        key = (sp, data["trade_date"][i])
        day_data[key]["prices"].append(data["price_avg"][i])
        day_data[key]["quantities"].append(data["quantity"][i])

    qty_results = {}
    for sp in top10:
        dates = sorted(set(k[1] for k in day_data if k[0] == sp))
        weighted_series = []
        unweighted_series = []
        for d in dates:
            key = (sp, d)
            if key not in day_data:
                continue
            w = weighted_avg(day_data[key]["prices"], day_data[key]["quantities"])
            u = np.mean(day_data[key]["prices"])
            weighted_series.append(w)
            unweighted_series.append(u)

        if len(weighted_series) >= 30:
            corr = pearson_corr(weighted_series, unweighted_series)
            qty_results[sp] = {
                "weighted_vs_unweighted_corr": round(corr, 4),
                "dominant_pkg_pct": dom_pkg_pct.get(sp, 0),
                "n_days": len(weighted_series),
            }

    mo.md(
        "| Species | W vs UW Corr | Dom Pkg % | Days | Safe? |\n"
        "|-|-|-|-|-|\n"
        + "\n".join(
            f"| {sp} | {v['weighted_vs_unweighted_corr']} | {v['dominant_pkg_pct']}% | {v['n_days']} | "
            f"{'YES' if v['weighted_vs_unweighted_corr'] > 0.98 else 'NO — use unweighted'} |"
            for sp, v in qty_results.items()
        )
    )

    return qty_results,
```

- [ ] **Step 2: Run and verify**

Run: `uv run marimo run notebooks/eda_aggregation.py 2>&1 | head -5`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add notebooks/eda_aggregation.py
git commit -m "feat(eda): add EDA-2.4 quantity unit comparability test"
```

---

### Task 8: Phase 3 — EDA-3.1 Blended vs Dominant + EDA-3.2 Row Reduction

**Files:**
- Modify: `notebooks/eda_aggregation.py`

- [ ] **Step 1: Add EDA-3.1 cell**

```python
@app.cell
def eda_3_1_blended_vs_dominant(data, top10, pkg_summary, defaultdict, np, weighted_avg, pearson_corr, lag1_autocorr, qty_results, mo):
    """EDA-3.1: Blended vs dominant-packaging comparison."""
    mo.md("## Phase 3: Aggregation Viability\n### EDA-3.1: Blended vs Dominant-Packaging")

    # Find dominant packaging per species
    dominant_pkg = {}
    for s in pkg_summary:
        if s["species"] in top10:
            dominant_pkg[s["species"]] = s["dominant_pkg"]

    # Build daily series: blended and dominant-only
    day_all = defaultdict(lambda: {"prices": [], "quantities": []})
    day_dom = defaultdict(lambda: {"prices": [], "quantities": []})

    for i in range(len(data["trade_date"])):
        sp = data["species"][i]
        if sp not in top10:
            continue
        key = (sp, data["trade_date"][i])
        day_all[key]["prices"].append(data["price_avg"][i])
        day_all[key]["quantities"].append(data["quantity"][i])
        if data["packaging"][i] == dominant_pkg.get(sp):
            day_dom[key]["prices"].append(data["price_avg"][i])
            day_dom[key]["quantities"].append(data["quantity"][i])

    blend_results = {}
    for sp in top10:
        dates = sorted(set(k[1] for k in day_all if k[0] == sp))

        # Choose weighting method based on EDA-2.4
        use_weighted = qty_results.get(sp, {}).get("weighted_vs_unweighted_corr", 1.0) > 0.98

        blended = []
        dominant = []
        common_dates = []
        for d in dates:
            ka = (sp, d)
            kd = (sp, d)
            if ka in day_all and kd in day_dom and day_dom[kd]["prices"]:
                if use_weighted:
                    b = weighted_avg(day_all[ka]["prices"], day_all[ka]["quantities"])
                    dom = weighted_avg(day_dom[kd]["prices"], day_dom[kd]["quantities"])
                else:
                    b = np.mean(day_all[ka]["prices"])
                    dom = np.mean(day_dom[kd]["prices"])
                blended.append(b)
                dominant.append(dom)
                common_dates.append(d)

        if len(blended) >= 30:
            corr = pearson_corr(blended, dominant)
            lag1_b = lag1_autocorr(blended)
            lag1_d = lag1_autocorr(dominant)
            blend_results[sp] = {
                "corr": round(corr, 4),
                "lag1_blended": round(lag1_b, 4),
                "lag1_dominant": round(lag1_d, 4),
                "n_days": len(blended),
                "weighting": "weighted" if use_weighted else "unweighted",
            }

    mo.md(
        "| Species | Corr | Lag1 (blend) | Lag1 (dom) | Weighting | Verdict |\n"
        "|-|-|-|-|-|-|\n"
        + "\n".join(
            f"| {sp} | {v['corr']} | {v['lag1_blended']} | {v['lag1_dominant']} | {v['weighting']} | "
            f"{'BLEND OK' if v['corr'] > 0.95 and v['lag1_blended'] > 0.8 else 'USE DOMINANT'} |"
            for sp, v in blend_results.items()
        )
    )

    return blend_results,
```

- [ ] **Step 2: Add EDA-3.2 cell**

```python
@app.cell
def eda_3_2_row_reduction(data, defaultdict, np, mo):
    """EDA-3.2: Row reduction ratio."""
    mo.md("### EDA-3.2: Row Reduction Ratio")

    strategies = {
        "Raw": lambda i: (data["trade_date"][i], data["species"][i], data["state"][i], data["origin"][i], data["spec"][i], data["packaging"][i]),
        "species": lambda i: (data["trade_date"][i], data["species"][i]),
        "species+state": lambda i: (data["trade_date"][i], data["species"][i], data["state"][i]),
        "species+state+pkg": lambda i: (data["trade_date"][i], data["species"][i], data["state"][i], data["packaging"][i]),
        "species+state+pkg+origin": lambda i: (data["trade_date"][i], data["species"][i], data["state"][i], data["packaging"][i], data["origin"][i]),
    }

    n = len(data["trade_date"])
    results = {}
    for name, key_fn in strategies.items():
        groups = set()
        day_counts = defaultdict(int)
        for i in range(n):
            k = key_fn(i)
            groups.add(k)
            day_counts[data["trade_date"][i]] = 0  # just counting days
        # Recount per day
        day_groups = defaultdict(set)
        for i in range(n):
            day_groups[data["trade_date"][i]].add(key_fn(i))
        per_day = [len(v) for v in day_groups.values()]
        results[name] = {
            "total_groups": len(groups),
            "median_per_day": round(np.median(per_day)),
            "mean_per_day": round(np.mean(per_day), 1),
        }

    raw_total = results["Raw"]["total_groups"]
    mo.md(
        "| Strategy | Total Groups | Median/Day | Compression |\n"
        "|-|-|-|-|\n"
        + "\n".join(
            f"| {name} | {v['total_groups']:,} | {v['median_per_day']} | "
            f"{raw_total / v['total_groups']:.1f}× |"
            for name, v in results.items()
        )
    )

    return results,
```

- [ ] **Step 3: Run and verify**

Run: `uv run marimo run notebooks/eda_aggregation.py 2>&1 | head -5`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/eda_aggregation.py
git commit -m "feat(eda): add EDA-3.1 blended vs dominant and EDA-3.2 row reduction"
```

---

### Task 9: Phase 4 — Edge Cases + Decision Summary

**Files:**
- Modify: `notebooks/eda_aggregation.py`

- [ ] **Step 1: Add EDA-4.1 + EDA-4.2 cells**

```python
@app.cell
def eda_4_1_heterogeneous(pkg_summary, mo):
    """EDA-4.1: Heterogeneous-packaging species."""
    mo.md("## Phase 4: Edge Cases\n### EDA-4.1: Heterogeneous-Packaging Species")

    hetero = [s for s in pkg_summary if s["dominant_pct"] < 50]
    if hetero:
        mo.md(
            f"**{len(hetero)} species** with no packaging type >50%:\n\n"
            + "\n".join(
                f"- {s['species']}: {s['dominant_pkg']} at {s['dominant_pct']}% ({s['n_pkg_types']} types)"
                for s in hetero
            )
        )
    else:
        mo.md("All species have a dominant packaging type (>50%). No heterogeneous cases.")

    return hetero,


@app.cell
def eda_4_2_low_volume(data, defaultdict, np, Counter, mo):
    """EDA-4.2: Low-volume species threshold."""
    mo.md("### EDA-4.2: Low-Volume Species Threshold")

    species_counts = Counter(data["species"])
    top30 = set(name for name, _ in species_counts.most_common(30))

    # For non-top-30, count trading days and max streak
    species_days = defaultdict(set)
    for i in range(len(data["trade_date"])):
        sp = data["species"][i]
        if sp not in top30:
            species_days[sp].add(data["trade_date"][i])

    from datetime import datetime as _dt

    low_vol = []
    for sp, days in sorted(species_days.items(), key=lambda x: -len(x[1])):
        sorted_days = sorted(days)
        parsed = [_dt.strptime(d, "%Y.%m.%d") for d in sorted_days]
        max_streak = streak = 1
        for j in range(1, len(parsed)):
            gap = (parsed[j] - parsed[j-1]).days
            if gap <= 7:  # Allow weekend/holiday gaps
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1

        low_vol.append({
            "species": sp,
            "total_days": len(days),
            "max_streak": max_streak,
            "viable": len(days) >= 100,
        })

    viable = sum(1 for s in low_vol if s["viable"])
    mo.md(
        f"**Non-top-30 species:** {len(low_vol)}\n"
        f"- Viable (≥100 trading days): {viable}\n"
        f"- Not viable (<100 days): {len(low_vol) - viable}\n\n"
        f"**Bottom 10 (fewest days):**\n\n"
        + "\n".join(
            f"- {s['species']}: {s['total_days']} days"
            for s in sorted(low_vol, key=lambda x: x["total_days"])[:10]
        )
    )

    return low_vol,
```

- [ ] **Step 2: Add decision summary cell**

```python
@app.cell
def decision_summary(
    state_summary, multi_state_prices,
    cv_results, spec_cv_results, pkg_corr_results,
    origin_spread_results, qty_results, blend_results,
    mo,
):
    """Final decision summary applying the decision tree."""
    mo.md("## Decision Summary")

    lines = ["### Per-Species Aggregation Strategy\n"]
    lines.append("| Species | State | Packaging | Spec | Weighting | Recommended GROUP BY |")
    lines.append("|-|-|-|-|-|-|")

    for sp, cv_r in cv_results.items():
        # State decision
        state_info = next((s for s in state_summary if s["species"] == sp), None)
        if state_info and state_info["dominant_pct"] >= 90:
            state_verdict = f"filter to {state_info['dominant_state']}"
        elif sp in multi_state_prices and multi_state_prices[sp]["ratio"] > 1.5:
            state_verdict = "partition by state"
        else:
            state_verdict = "aggregate across"

        # Packaging decision (with EDA-3.1 confirmation per decision tree)
        if cv_r["ratio"] <= 1.5:
            br = blend_results.get(sp, {})
            if br.get("corr", 0) > 0.95 and br.get("lag1_blended", 0) > 0.8:
                pkg_verdict = "blend (confirmed)"
            else:
                pkg_verdict = "dominant-pkg (blend unconfirmed)"
        elif sp in pkg_corr_results:
            pc = pkg_corr_results[sp]
            if pc["corr_weighted"] > 0.85 and pc["price_ratio"] < 1.5:
                pkg_verdict = "blend (co-move)"
            elif pc["corr_weighted"] > 0.85:
                pkg_verdict = "dominant-pkg"
            else:
                pkg_verdict = "separate"
        else:
            pkg_verdict = "blend (single-pkg)"

        # Spec decision
        spec_r = spec_cv_results.get(sp, {})
        spec_verdict = "aggregate" if spec_r.get("ratio", 0) <= 1.5 else "spec-class"

        # Weighting
        wt = "weighted" if qty_results.get(sp, {}).get("weighted_vs_unweighted_corr", 1.0) > 0.98 else "unweighted"

        # Recommended GROUP BY
        group_by_parts = ["trade_date", "species"]
        if "partition" in state_verdict:
            group_by_parts.append("state")
        if pkg_verdict in ("separate", "dominant-pkg"):
            group_by_parts.append("packaging")
        if spec_verdict == "spec-class":
            group_by_parts.append("spec_class")

        lines.append(
            f"| {sp} | {state_verdict} | {pkg_verdict} | {spec_verdict} | {wt} | "
            f"`({', '.join(group_by_parts)})` |"
        )

    mo.md("\n".join(lines))

    # Deliverable 4: Pipeline impact recommendation
    state_partition = sum(1 for sp, cv_r in cv_results.items()
                         if any(s["species"] == sp and s["dominant_pct"] < 90 for s in state_summary)
                         and sp in multi_state_prices and multi_state_prices[sp]["ratio"] > 1.5)
    state_filter = sum(1 for sp, cv_r in cv_results.items()
                       if any(s["species"] == sp and s["dominant_pct"] >= 90 for s in state_summary))

    pipeline_lines = [
        "\n### Prediction Pipeline Impact\n",
        f"- **(a) Single model per species:** {len(cv_results) - state_partition} species",
        f"- **(b) Separate models per (species, state):** {state_partition} species",
        f"- **(c) Dominant-state filter applied:** {state_filter} species\n",
    ]
    if state_partition == 0:
        pipeline_lines.append("**Recommendation:** Option (a) — one model per species. "
                              "State partitioning is not needed; dominant-state filtering handles the rest.")
    else:
        pipeline_lines.append(f"**Recommendation:** Option (c) for most species, option (b) for "
                              f"the {state_partition} multi-state species with divergent pricing.")

    # Deliverable 3: DuckDB view SQL
    # Re-derive per-species packaging verdict to determine GROUP BY
    from collections import Counter as _Counter
    group_by_patterns = _Counter()
    for sp, cv_r in cv_results.items():
        parts = ["trade_date", "species"]
        # State dimension
        si = next((s for s in state_summary if s["species"] == sp), None)
        if si and si["dominant_pct"] < 90 and sp in multi_state_prices and multi_state_prices[sp]["ratio"] > 1.5:
            parts.append("state")
        # Packaging dimension (mirror the verdict logic above)
        if cv_r["ratio"] > 1.5:
            pc = pkg_corr_results.get(sp, {})
            if pc.get("corr_weighted", 0) <= 0.85:
                parts.append("packaging")
            elif pc.get("price_ratio", 0) >= 1.5:
                parts.append("packaging")
        else:
            br = blend_results.get(sp, {})
            if not (br.get("corr", 0) > 0.95 and br.get("lag1_blended", 0) > 0.8):
                parts.append("packaging")
        group_by_patterns[tuple(parts)] += 1

    dominant_pattern = group_by_patterns.most_common(1)[0][0] if group_by_patterns else ("trade_date", "species")
    group_cols = ", ".join(dominant_pattern)

    sql_lines = [
        "\n### Recommended DuckDB Aggregation View\n",
        "```sql",
        "CREATE OR REPLACE VIEW v_daily_prices AS",
        "SELECT",
        f"    {group_cols},",
        "    SUM(quantity) AS total_quantity,",
        "    MAX(price_high) AS price_high,",
        "    MIN(price_low) AS price_low,",
        "    CAST(AVG(price_avg) AS INTEGER) AS price_avg,",
        "    COUNT(*) AS n_lots",
        "FROM read_parquet('data/parquet/prices/**/*.parquet', hive_partitioning=true)",
        "WHERE state IS NOT NULL AND packaging IS NOT NULL",
        f"GROUP BY {group_cols}",
        f"ORDER BY {group_cols};",
        "```",
        f"\nDominant GROUP BY pattern: `({group_cols})` — used by {group_by_patterns.most_common(1)[0][1] if group_by_patterns else 0}/{len(cv_results)} species.",
    ]

    mo.md("\n".join(lines) + "\n" + "\n".join(pipeline_lines) + "\n" + "\n".join(sql_lines))

    return
```

- [ ] **Step 3: Run full notebook end-to-end**

Run: `uv run marimo run notebooks/eda_aggregation.py 2>&1 | head -5`
Expected: No errors, all cells execute.

- [ ] **Step 4: Commit**

```bash
git add notebooks/eda_aggregation.py
git commit -m "feat(eda): add edge cases, decision summary, complete aggregation EDA"
```

---

### Task 10: Update Preprocessing Documentation

**Files:**
- Modify: `docs/08_data_preprocessing.md`

- [ ] **Step 1: Add aggregation section stub**

Append to `docs/08_data_preprocessing.md`:

```markdown
---

## Row Aggregation Strategy

> Results from EDA notebook: `notebooks/eda_aggregation.py`
> Spec: `docs/superpowers/specs/2026-03-25-aggregation-eda-design.md`

This section will be populated after running the EDA notebook. The notebook applies the decision tree from the spec and produces per-species aggregation recommendations.

### How to Run

```bash
# Interactive (recommended — see charts)
uv run marimo edit notebooks/eda_aggregation.py

# Headless (stdout summary only)
uv run marimo run notebooks/eda_aggregation.py
```

### Aggregation Rules

*To be filled after EDA execution — see the Decision Summary cell in the notebook.*

### DuckDB View

*To be filled after EDA execution — the validated GROUP BY query for the prediction pipeline.*
```

- [ ] **Step 2: Commit**

```bash
git add docs/08_data_preprocessing.md
git commit -m "docs: add aggregation strategy stub to preprocessing doc"
```

---

### Task 11: Final Verification

- [ ] **Step 1: Run full notebook and capture output**

```bash
uv run marimo run notebooks/eda_aggregation.py 2>&1 | tail -50
```

Expected: All cells execute without errors. Decision summary table is printed.

- [ ] **Step 2: Verify no import errors on fresh install**

```bash
uv sync && uv run python -c "from notebooks.eda_helpers import load_all_data, cv, classify_spec; print('OK')"
```

- [ ] **Step 3: Commit all remaining changes**

```bash
git add -A && git status
git commit -m "feat: complete aggregation EDA notebook with all 12 analyses"
```
