"""
Narrow down species to viable prediction candidates.

Filters the 504 species through 4 gates:
  Gate 1: Data volume (minimum trading days)
  Gate 2: Trading consistency (recent activity + regularity)
  Gate 3: Signal quality (autocorrelation after dominant-pkg filtering)
  Gate 4: Market relevance (trading volume)

Outputs a ranked shortlist and saves to data/prediction_candidates.json.

Usage:
    uv run python scripts/eda_prediction_candidates.py
"""
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "parquet" / "prices"
EDA_RESULTS = PROJECT_ROOT / "data" / "eda_results.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "prediction_candidates.json"

EDA_COLUMNS = [
    "trade_date", "species", "state", "origin", "spec",
    "packaging", "quantity", "price_high", "price_low", "price_avg",
]


def load_data():
    print("Loading data...", end=" ", flush=True)
    dataset = ds.dataset(str(DATA_ROOT), format="parquet", partitioning="hive")
    table = dataset.to_table(columns=EDA_COLUMNS)
    data = {col: table.column(col).to_pylist() for col in EDA_COLUMNS}
    n = len(data["trade_date"])
    print(f"{n:,} rows.")
    return data, n


def gate_1_data_volume(data, n):
    """Gate 1: Minimum 200 trading days total."""
    print("\n=== GATE 1: Data Volume (≥200 trading days) ===")
    species_days = defaultdict(set)
    for i in range(n):
        species_days[data["species"][i]].add(data["trade_date"][i])

    results = {}
    for sp, days in species_days.items():
        results[sp] = len(days)

    passed = {sp: d for sp, d in results.items() if d >= 200}
    failed = len(results) - len(passed)
    print(f"  Total species: {len(results)}")
    print(f"  Passed (≥200 days): {len(passed)}")
    print(f"  Failed: {failed}")
    return passed, results


def gate_2_consistency(data, n, candidates):
    """Gate 2: Recent activity (traded in last 365 days) + regularity (≥60% of trading days in last year)."""
    print("\n=== GATE 2: Trading Consistency ===")

    # Find all trading days in the dataset
    all_dates = sorted(set(data["trade_date"]))
    cutoff_date = all_dates[-1]  # most recent date in dataset
    cutoff_dt = datetime.strptime(cutoff_date, "%Y.%m.%d")

    # Last 365 days of trading days
    recent_dates = set()
    for d in all_dates:
        dt = datetime.strptime(d, "%Y.%m.%d")
        if (cutoff_dt - dt).days <= 365:
            recent_dates.add(d)

    n_recent_trading_days = len(recent_dates)
    print(f"  Most recent date: {cutoff_date}")
    print(f"  Trading days in last 365 days: {n_recent_trading_days}")

    # Per species: how many of those recent trading days do they appear?
    species_recent = defaultdict(set)
    for i in range(n):
        sp = data["species"][i]
        if sp not in candidates:
            continue
        d = data["trade_date"][i]
        if d in recent_dates:
            species_recent[sp].add(d)

    results = {}
    for sp in candidates:
        recent_count = len(species_recent.get(sp, set()))
        regularity = recent_count / n_recent_trading_days if n_recent_trading_days > 0 else 0
        results[sp] = {
            "recent_days": recent_count,
            "regularity": round(regularity, 3),
            "active": recent_count > 0,
        }

    # Pass: active in last year AND regularity ≥ 60%
    passed = {sp: v for sp, v in results.items() if v["active"] and v["regularity"] >= 0.6}
    inactive = sum(1 for v in results.values() if not v["active"])
    irregular = sum(1 for v in results.values() if v["active"] and v["regularity"] < 0.6)

    print(f"  Candidates in: {len(candidates)}")
    print(f"  No recent trading: {inactive}")
    print(f"  Active but irregular (<60%): {irregular}")
    print(f"  Passed: {len(passed)}")

    return passed, results


def gate_3_signal_quality(data, n, candidates):
    """Gate 3: Signal quality — lag-1 autocorrelation ≥ 0.3 using dominant packaging."""
    print("\n=== GATE 3: Signal Quality (lag-1 autocorr ≥ 0.3) ===")

    # First find dominant packaging per species
    sp_pkg = defaultdict(Counter)
    for i in range(n):
        sp = data["species"][i]
        if sp not in candidates:
            continue
        pkg = data["packaging"][i]
        if pkg:
            sp_pkg[sp][pkg] += 1

    dom_pkg = {}
    for sp, counts in sp_pkg.items():
        dom_pkg[sp] = counts.most_common(1)[0][0]

    # Also find dominant state per species
    sp_state = defaultdict(Counter)
    for i in range(n):
        sp = data["species"][i]
        if sp not in candidates:
            continue
        st = data["state"][i]
        if st:
            sp_state[sp][st] += 1

    dom_state = {}
    for sp, counts in sp_state.items():
        dom_state[sp] = counts.most_common(1)[0][0]

    # Build daily price series using dominant state + dominant packaging
    day_prices = defaultdict(list)
    for i in range(n):
        sp = data["species"][i]
        if sp not in candidates:
            continue
        if data["packaging"][i] != dom_pkg.get(sp):
            continue
        if data["state"][i] != dom_state.get(sp):
            continue
        day_prices[(sp, data["trade_date"][i])].append(data["price_avg"][i])

    # Compute daily avg and lag-1 autocorrelation
    results = {}
    for sp in candidates:
        dates = sorted(set(k[1] for k in day_prices if k[0] == sp))
        if len(dates) < 30:
            results[sp] = {"lag1": 0, "n_days": len(dates), "dom_pkg": dom_pkg.get(sp, "?"),
                           "dom_state": dom_state.get(sp, "?")}
            continue

        series = [np.mean(day_prices[(sp, d)]) for d in dates]
        arr = np.array(series)
        lag1 = float(np.corrcoef(arr[:-1], arr[1:])[0, 1]) if len(arr) >= 3 else 0

        # Also compute 7-day rolling mean lag-1 (smoothed signal)
        if len(arr) >= 14:
            rolling = np.convolve(arr, np.ones(7)/7, mode="valid")
            lag1_smooth = float(np.corrcoef(rolling[:-1], rolling[1:])[0, 1])
        else:
            lag1_smooth = lag1

        # Price stats
        mean_price = round(float(np.mean(arr)))
        price_std = round(float(np.std(arr)))
        cv = price_std / mean_price if mean_price > 0 else 0

        results[sp] = {
            "lag1": round(lag1, 4),
            "lag1_7d": round(lag1_smooth, 4),
            "n_days": len(dates),
            "mean_price": mean_price,
            "price_std": price_std,
            "cv": round(cv, 4),
            "dom_pkg": dom_pkg.get(sp, "?"),
            "dom_state": dom_state.get(sp, "?"),
        }

    # Pass: lag-1 ≥ 0.3 (raw or smoothed)
    passed = {sp: v for sp, v in results.items() if max(v["lag1"], v.get("lag1_7d", 0)) >= 0.3}
    print(f"  Candidates in: {len(candidates)}")
    print(f"  Passed (lag1 ≥ 0.3): {len(passed)}")
    print(f"  Failed (noisy): {len(candidates) - len(passed)}")

    return passed, results


def gate_4_market_relevance(data, n, candidates, signal_results):
    """Gate 4: Market relevance — rank by total quantity traded."""
    print("\n=== GATE 4: Market Relevance (rank by volume) ===")

    sp_qty = defaultdict(float)
    sp_rows = Counter()
    for i in range(n):
        sp = data["species"][i]
        if sp not in candidates:
            continue
        sp_qty[sp] += data["quantity"][i]
        sp_rows[sp] += 1

    # Score = composite of volume rank + signal quality
    ranked = []
    for sp in candidates:
        info = signal_results[sp]
        ranked.append({
            "species": sp,
            "total_qty": round(sp_qty[sp], 1),
            "total_rows": sp_rows[sp],
            "lag1": info["lag1"],
            "lag1_7d": info.get("lag1_7d", info["lag1"]),
            "mean_price": info["mean_price"],
            "cv": info["cv"],
            "dom_state": info["dom_state"],
            "dom_pkg": info["dom_pkg"],
            "n_days": info["n_days"],
        })

    ranked.sort(key=lambda x: -x["total_qty"])

    # Assign tiers
    for i, r in enumerate(ranked):
        if i < 15:
            r["tier"] = "A"
        elif i < 30:
            r["tier"] = "B"
        else:
            r["tier"] = "C"

    print(f"  Total candidates: {len(ranked)}")
    print(f"  Tier A (top 15): {sum(1 for r in ranked if r['tier'] == 'A')}")
    print(f"  Tier B (16-30): {sum(1 for r in ranked if r['tier'] == 'B')}")
    print(f"  Tier C (rest): {sum(1 for r in ranked if r['tier'] == 'C')}")

    return ranked


def main():
    data, n = load_data()

    # Gate 1
    g1_passed, g1_all = gate_1_data_volume(data, n)

    # Gate 2
    g2_passed, g2_all = gate_2_consistency(data, n, g1_passed)

    # Gate 3
    g3_passed, g3_all = gate_3_signal_quality(data, n, g2_passed)

    # Gate 4
    ranked = gate_4_market_relevance(data, n, g3_passed, g3_all)

    # Summary
    print("\n" + "=" * 80)
    print("PREDICTION CANDIDATES — FINAL SHORTLIST")
    print("=" * 80)
    print(f"\n{'':>3} {'Species':<15} {'Tier':>4} {'Days':>5} {'Lag1':>6} {'Lag1(7d)':>8} {'MeanPrice':>10} {'CV':>6} {'State':<5} {'Pkg':<10} {'Qty':>12}")
    print("-" * 100)
    for i, r in enumerate(ranked):
        print(f"{i+1:>3} {r['species']:<15} {r['tier']:>4} {r['n_days']:>5} {r['lag1']:>6.3f} {r['lag1_7d']:>8.3f} "
              f"{r['mean_price']:>10,} {r['cv']:>6.3f} {r['dom_state']:<5} {r['dom_pkg']:<10} {r['total_qty']:>12,.1f}")

    # Funnel summary
    print(f"\n--- Funnel ---")
    print(f"  All species:        504")
    print(f"  Gate 1 (≥200 days): {len(g1_passed)}")
    print(f"  Gate 2 (consistent):{len(g2_passed)}")
    print(f"  Gate 3 (signal):    {len(g3_passed)}")
    print(f"  Final candidates:   {len(ranked)}")
    print(f"    Tier A (top 15):  {sum(1 for r in ranked if r['tier'] == 'A')}")
    print(f"    Tier B (16-30):   {sum(1 for r in ranked if r['tier'] == 'B')}")

    # Save
    output = {
        "generated_at": datetime.now().isoformat(),
        "funnel": {
            "all_species": 504,
            "gate_1_volume": len(g1_passed),
            "gate_2_consistency": len(g2_passed),
            "gate_3_signal": len(g3_passed),
            "final": len(ranked),
        },
        "candidates": ranked,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
