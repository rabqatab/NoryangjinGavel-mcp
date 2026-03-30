"""
EDA: Row Aggregation Viability for Price Prediction.

Runs all 13 EDA steps from docs/superpowers/specs/2026-03-25-aggregation-eda-design.md,
prints results to stdout, and saves structured output to data/eda_results.json.

Usage:
    uv run python scripts/eda_aggregation.py
"""
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "parquet" / "prices"
OUTPUT_PATH = PROJECT_ROOT / "data" / "eda_results.json"

EDA_COLUMNS = [
    "trade_date", "species", "state", "origin", "spec",
    "packaging", "quantity", "price_high", "price_low", "price_avg",
]

# --- Helpers ---

def cv(values):
    arr = np.array(values, dtype=float)
    mean = np.mean(arr)
    return float(np.std(arr) / mean) if mean != 0 else 0.0

def weighted_avg(prices, quantities):
    p, q = np.array(prices, dtype=float), np.array(quantities, dtype=float)
    total_q = q.sum()
    return float(np.dot(p, q) / total_q) if total_q > 0 else float(np.mean(p))

def pearson_corr(a, b):
    if len(a) < 3:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def lag1_autocorr(values):
    arr = np.array(values, dtype=float)
    if len(arr) < 3:
        return 0.0
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])

_SPEC_SIZE = re.compile(r"^(특대|대|중|소)$")
_SPEC_COUNT = re.compile(r"^\d+미$")
_SPEC_WRANGE = re.compile(r"^\d+/\d+$")
_SPEC_CRANGE = re.compile(r"^\d+/\d+미$")

def classify_spec(spec):
    if spec is None: return "null"
    if _SPEC_SIZE.match(spec): return "size_grade"
    if _SPEC_COUNT.match(spec): return "count"
    if _SPEC_CRANGE.match(spec): return "count_range"
    if _SPEC_WRANGE.match(spec): return "weight_range"
    return "other"

# --- Main ---

def load_data():
    print("Loading data...", end=" ", flush=True)
    dataset = ds.dataset(str(DATA_ROOT), format="parquet", partitioning="hive")
    table = dataset.to_table(columns=EDA_COLUMNS)
    data = {col: table.column(col).to_pylist() for col in EDA_COLUMNS}
    n = len(data["trade_date"])
    print(f"{n:,} rows loaded.")
    return data, n

def eda_1_0(data, n):
    """State distribution per species."""
    print("\n=== EDA-1.0: State Distribution ===")
    sp_state = defaultdict(Counter)
    for i in range(n):
        sp_state[data["species"][i]][data["state"][i] or "(null)"] += 1

    results = []
    multi_state = []
    for sp, counts in sorted(sp_state.items(), key=lambda x: -sum(x[1].values())):
        total = sum(counts.values())
        dom_st, dom_ct = counts.most_common(1)[0]
        pct = dom_ct / total * 100
        results.append({"species": sp, "dominant_state": dom_st, "dominant_pct": round(pct, 1),
                        "total_rows": total, "n_states": len(counts),
                        "all_states": {k: v for k, v in counts.most_common()}})
        if pct < 90:
            multi_state.append(sp)

    single = sum(1 for r in results if r["dominant_pct"] >= 90)
    print(f"  {single}/{len(results)} species have >90% in one state")
    print(f"  Multi-state species: {len(multi_state)}")

    # Price divergence for multi-state
    multi_prices = {}
    for sp in multi_state:
        st_prices = defaultdict(list)
        for i in range(n):
            if data["species"][i] == sp and data["state"][i]:
                st_prices[data["state"][i]].append(data["price_avg"][i])
        means = {st: round(np.mean(p)) for st, p in st_prices.items() if len(p) > 10}
        if len(means) >= 2:
            vals = list(means.values())
            ratio = max(vals) / min(vals) if min(vals) > 0 else 0
            multi_prices[sp] = {"means": means, "ratio": round(ratio, 2)}

    divergent = {sp: v for sp, v in multi_prices.items() if v["ratio"] > 1.5}
    print(f"  Divergent (ratio > 1.5x): {len(divergent)}")
    for sp, v in sorted(divergent.items()):
        print(f"    {sp}: {v['ratio']}x — {v['means']}")

    return {"summary": results, "multi_state": multi_state, "multi_state_prices": multi_prices, "divergent": list(divergent.keys())}

def eda_1_1(data, n):
    """Packaging dominance per species."""
    print("\n=== EDA-1.1: Packaging Dominance ===")
    sp_pkg = defaultdict(Counter)
    for i in range(n):
        pkg = data["packaging"][i]
        if pkg: sp_pkg[data["species"][i]][pkg] += 1

    results = []
    for sp, counts in sorted(sp_pkg.items(), key=lambda x: -sum(x[1].values())):
        total = sum(counts.values())
        dom_pkg, dom_ct = counts.most_common(1)[0]
        results.append({"species": sp, "dominant_pkg": dom_pkg,
                        "dominant_pct": round(dom_ct / total * 100, 1),
                        "total_rows": total, "n_pkg_types": len(counts)})

    above_80 = sum(1 for r in results if r["dominant_pct"] >= 80)
    print(f"  {above_80}/{len(results)} species have >80% in one packaging type")
    return results

def eda_1_2(data, n):
    """Spec type taxonomy."""
    print("\n=== EDA-1.2: Spec Type Taxonomy ===")
    spec_classes = Counter(classify_spec(data["spec"][i]) for i in range(n))
    print("  Global spec categories:")
    for k, v in spec_classes.most_common():
        print(f"    {k}: {v:,}")

    sp_cats = defaultdict(set)
    for i in range(n):
        sp_cats[data["species"][i]].add(classify_spec(data["spec"][i]))
    cat_dist = Counter(len(v) for v in sp_cats.values())
    print("  Spec categories per species:")
    for k, v in sorted(cat_dist.items()):
        print(f"    {k} categories: {v} species")

    return {"global": dict(spec_classes), "per_species_cat_count": dict(cat_dist)}

def eda_1_3(data, n):
    """Origin distribution per species per day."""
    print("\n=== EDA-1.3: Origin Distribution ===")
    day_sp_origins = defaultdict(set)
    for i in range(n):
        o = data["origin"][i]
        if o: day_sp_origins[(data["trade_date"][i], data["species"][i])].add(o)

    counts = [len(v) for v in day_sp_origins.values()]
    arr = np.array(counts)
    stats = {"min": int(arr.min()), "max": int(arr.max()), "median": float(np.median(arr)),
             "mean": round(float(arr.mean()), 1),
             "p25": float(np.percentile(arr, 25)), "p75": float(np.percentile(arr, 75)),
             "p95": float(np.percentile(arr, 95))}
    print(f"  Origins per (species, day): median={stats['median']:.0f}, mean={stats['mean']}, p95={stats['p95']:.0f}")
    return stats

def eda_2_1(data, n, top10):
    """Intra-day price spread by packaging."""
    print("\n=== EDA-2.1: Packaging CV Ratio ===")
    top_set = set(top10)
    day_pkg = defaultdict(lambda: defaultdict(list))
    day_all = defaultdict(list)
    for i in range(n):
        sp = data["species"][i]
        if sp not in top_set: continue
        pkg = data["packaging"][i]
        if not pkg: continue
        key = (sp, data["trade_date"][i])
        day_pkg[key][pkg].append(data["price_avg"][i])
        day_all[key].append(data["price_avg"][i])

    results = {}
    for sp in top10:
        within_cvs, across_cvs = [], []
        for key in day_all:
            if key[0] != sp: continue
            if len(day_all[key]) >= 3:
                across_cvs.append(cv(day_all[key]))
            pkg_cvs = [cv(p) for p in day_pkg[key].values() if len(p) >= 2]
            if pkg_cvs:
                within_cvs.append(np.mean(pkg_cvs))
        if within_cvs and across_cvs:
            mw, ma = np.median(within_cvs), np.median(across_cvs)
            ratio = ma / mw if mw > 0 else 0
            results[sp] = {"within_cv": round(mw, 4), "across_cv": round(ma, 4), "ratio": round(ratio, 2)}

    print(f"  {'Species':<15} {'Within':>8} {'Across':>8} {'Ratio':>6} {'Verdict'}")
    for sp, v in results.items():
        verdict = "SEGMENTS" if v["ratio"] > 1.5 else "OK"
        print(f"  {sp:<15} {v['within_cv']:>8.4f} {v['across_cv']:>8.4f} {v['ratio']:>5.2f}x {verdict}")

    # Sensitivity sweep
    for t in [1.5, 2.0, 2.5]:
        seg = sum(1 for v in results.values() if v["ratio"] > t)
        print(f"  Threshold {t}x: {seg} segmented, {len(results)-seg} OK")

    return results

def eda_2_1b(data, n, top10):
    """Intra-day price spread by spec within packaging."""
    print("\n=== EDA-2.1b: Spec CV Ratio ===")
    top_set = set(top10)
    group_prices = defaultdict(lambda: defaultdict(list))
    pkg_all = defaultdict(list)
    for i in range(n):
        sp = data["species"][i]
        if sp not in top_set: continue
        pkg, spec = data["packaging"][i], data["spec"][i]
        if not pkg or not spec: continue
        outer = (sp, data["trade_date"][i], data["state"][i], pkg)
        group_prices[outer][spec].append(data["price_avg"][i])
        pkg_all[outer].append(data["price_avg"][i])

    results = {}
    for sp in top10:
        within_cvs, across_cvs = [], []
        for key in pkg_all:
            if key[0] != sp: continue
            if len(pkg_all[key]) >= 3:
                across_cvs.append(cv(pkg_all[key]))
            s_cvs = [cv(p) for p in group_prices[key].values() if len(p) >= 2]
            if s_cvs:
                within_cvs.append(np.mean(s_cvs))
        if within_cvs and across_cvs:
            mw, ma = np.median(within_cvs), np.median(across_cvs)
            ratio = ma / mw if mw > 0 else 0
            results[sp] = {"within_cv": round(mw, 4), "across_cv": round(ma, 4), "ratio": round(ratio, 2)}

    print(f"  {'Species':<15} {'Within':>8} {'Across':>8} {'Ratio':>6} {'Verdict'}")
    for sp, v in results.items():
        verdict = "SEGMENTS" if v["ratio"] > 1.5 else "OK"
        print(f"  {sp:<15} {v['within_cv']:>8.4f} {v['across_cv']:>8.4f} {v['ratio']:>5.2f}x {verdict}")
    return results

def eda_2_2(data, n, top10):
    """Per-packaging time series correlation."""
    print("\n=== EDA-2.2: Packaging Time Series Correlation ===")
    top_set = set(top10)
    day_pkg = defaultdict(lambda: defaultdict(lambda: {"p": [], "q": []}))
    for i in range(n):
        sp = data["species"][i]
        if sp not in top_set: continue
        pkg = data["packaging"][i]
        if not pkg: continue
        day_pkg[(sp, pkg)][data["trade_date"][i]]["p"].append(data["price_avg"][i])
        day_pkg[(sp, pkg)][data["trade_date"][i]]["q"].append(data["quantity"][i])

    series = {}
    for (sp, pkg), dmap in day_pkg.items():
        if len(dmap) < 100: continue
        s = {d: {"w": weighted_avg(v["p"], v["q"]), "s": np.mean(v["p"])} for d, v in sorted(dmap.items())}
        series[(sp, pkg)] = s

    results = {}
    for sp in top10:
        sp_pkgs = sorted([(k, v) for k, v in series.items() if k[0] == sp], key=lambda x: -len(x[1]))
        if len(sp_pkgs) < 2: continue
        (_, pkg_a), sa = sp_pkgs[0]
        (_, pkg_b), sb = sp_pkgs[1]
        common = sorted(set(sa) & set(sb))
        if len(common) < 30: continue

        aw = [sa[d]["w"] for d in common]
        bw = [sb[d]["w"] for d in common]
        corr_w = pearson_corr(aw, bw)
        corr_s = pearson_corr([sa[d]["s"] for d in common], [sb[d]["s"] for d in common])

        mean_a, mean_b = np.mean(aw), np.mean(bw)
        price_ratio = max(mean_a, mean_b) / min(mean_a, mean_b) if min(mean_a, mean_b) > 0 else 0

        mid = len(common) // 2
        corr_1st = pearson_corr(aw[:mid], bw[:mid])
        corr_2nd = pearson_corr(aw[mid:], bw[mid:])
        stable = abs(corr_1st - corr_2nd) <= 0.15

        results[sp] = {"pkg_a": pkg_a, "pkg_b": pkg_b,
                        "corr_weighted": round(corr_w, 3), "corr_simple": round(corr_s, 3),
                        "price_ratio": round(price_ratio, 2),
                        "corr_1st_half": round(corr_1st, 3), "corr_2nd_half": round(corr_2nd, 3),
                        "stable": stable, "common_days": len(common)}

    for sp, v in results.items():
        vd = "BLEND" if v["corr_weighted"] > 0.85 and v["price_ratio"] < 1.5 else "DOM-PKG" if v["corr_weighted"] > 0.85 else "SEPARATE"
        print(f"  {sp:<15} {v['pkg_a']:>8} vs {v['pkg_b']:<8} corr(W)={v['corr_weighted']:.3f} corr(S)={v['corr_simple']:.3f} "
              f"ratio={v['price_ratio']}x stable={'Y' if v['stable'] else 'N'} → {vd}")
    return results

def eda_2_3(data, n, top10):
    """Origin price spread."""
    print("\n=== EDA-2.3: Origin Price Spread ===")
    top_set = set(top10)
    groups = defaultdict(list)
    for i in range(n):
        sp = data["species"][i]
        if sp not in top_set: continue
        groups[(sp, data["trade_date"][i], data["state"][i], data["packaging"][i], data["spec"][i])].append(data["price_avg"][i])

    spreads = defaultdict(list)
    for key, prices in groups.items():
        if len(prices) < 2: continue
        m = np.mean(prices)
        if m == 0: continue
        spreads[key[0]].append((max(prices) - min(prices)) / m)

    results = {}
    for sp in top10:
        if sp in spreads and spreads[sp]:
            arr = np.array(spreads[sp])
            results[sp] = {"median": round(np.median(arr) * 100, 1),
                           "mean": round(np.mean(arr) * 100, 1),
                           "p95": round(np.percentile(arr, 95) * 100, 1),
                           "n_groups": len(arr)}

    for sp, v in results.items():
        vd = "OK" if v["median"] < 30 else "ORIGIN MATTERS"
        print(f"  {sp:<15} median={v['median']:>5.1f}% mean={v['mean']:>5.1f}% p95={v['p95']:>5.1f}% → {vd}")
    return results

def eda_2_4(data, n, top10, pkg_summary):
    """Quantity unit comparability."""
    print("\n=== EDA-2.4: Quantity Unit Comparability ===")
    top_set = set(top10)
    dom_pct = {s["species"]: s["dominant_pct"] for s in pkg_summary if s["species"] in top_set}
    day_data = defaultdict(lambda: {"p": [], "q": []})
    for i in range(n):
        sp = data["species"][i]
        if sp not in top_set: continue
        day_data[(sp, data["trade_date"][i])]["p"].append(data["price_avg"][i])
        day_data[(sp, data["trade_date"][i])]["q"].append(data["quantity"][i])

    results = {}
    for sp in top10:
        dates = sorted(set(k[1] for k in day_data if k[0] == sp))
        ws, us = [], []
        for d in dates:
            k = (sp, d)
            if k not in day_data: continue
            ws.append(weighted_avg(day_data[k]["p"], day_data[k]["q"]))
            us.append(np.mean(day_data[k]["p"]))
        if len(ws) >= 30:
            corr = pearson_corr(ws, us)
            results[sp] = {"corr": round(corr, 4), "dom_pkg_pct": dom_pct.get(sp, 0), "n_days": len(ws)}

    for sp, v in results.items():
        safe = "YES" if v["corr"] > 0.98 else "NO"
        print(f"  {sp:<15} corr={v['corr']:.4f} dom_pkg={v['dom_pkg_pct']}% → {safe}")
    return results

def eda_3_1(data, n, top10, pkg_summary, qty_results):
    """Blended vs dominant-packaging comparison."""
    print("\n=== EDA-3.1: Blended vs Dominant-Packaging ===")
    top_set = set(top10)
    dom_pkg = {s["species"]: s["dominant_pkg"] for s in pkg_summary if s["species"] in top_set}

    day_all = defaultdict(lambda: {"p": [], "q": []})
    day_dom = defaultdict(lambda: {"p": [], "q": []})
    for i in range(n):
        sp = data["species"][i]
        if sp not in top_set: continue
        k = (sp, data["trade_date"][i])
        day_all[k]["p"].append(data["price_avg"][i])
        day_all[k]["q"].append(data["quantity"][i])
        if data["packaging"][i] == dom_pkg.get(sp):
            day_dom[k]["p"].append(data["price_avg"][i])
            day_dom[k]["q"].append(data["quantity"][i])

    results = {}
    for sp in top10:
        use_w = qty_results.get(sp, {}).get("corr", 1.0) > 0.98
        dates = sorted(set(k[1] for k in day_all if k[0] == sp))
        blended, dominant = [], []
        for d in dates:
            k = (sp, d)
            if k in day_all and k in day_dom and day_dom[k]["p"]:
                if use_w:
                    blended.append(weighted_avg(day_all[k]["p"], day_all[k]["q"]))
                    dominant.append(weighted_avg(day_dom[k]["p"], day_dom[k]["q"]))
                else:
                    blended.append(np.mean(day_all[k]["p"]))
                    dominant.append(np.mean(day_dom[k]["p"]))

        if len(blended) >= 30:
            corr = pearson_corr(blended, dominant)
            l1b = lag1_autocorr(blended)
            l1d = lag1_autocorr(dominant)
            results[sp] = {"corr": round(corr, 4), "lag1_blend": round(l1b, 4), "lag1_dom": round(l1d, 4),
                           "n_days": len(blended), "weighting": "weighted" if use_w else "unweighted"}

    for sp, v in results.items():
        vd = "BLEND OK" if v["corr"] > 0.95 and v["lag1_blend"] > 0.8 else "USE DOMINANT"
        print(f"  {sp:<15} corr={v['corr']:.4f} lag1(b)={v['lag1_blend']:.4f} lag1(d)={v['lag1_dom']:.4f} → {vd}")
    return results

def eda_3_2(data, n):
    """Row reduction ratio."""
    print("\n=== EDA-3.2: Row Reduction Ratio ===")
    strategies = {
        "Raw":                  lambda i: (data["trade_date"][i], data["species"][i], data["state"][i], data["origin"][i], data["spec"][i], data["packaging"][i]),
        "species":              lambda i: (data["trade_date"][i], data["species"][i]),
        "species+state":        lambda i: (data["trade_date"][i], data["species"][i], data["state"][i]),
        "species+state+pkg":    lambda i: (data["trade_date"][i], data["species"][i], data["state"][i], data["packaging"][i]),
    }
    results = {}
    for name, fn in strategies.items():
        day_groups = defaultdict(set)
        for i in range(n):
            day_groups[data["trade_date"][i]].add(fn(i))
        per_day = [len(v) for v in day_groups.values()]
        total = sum(len(v) for v in day_groups.values())  # not unique, just sum
        unique = set()
        for v in day_groups.values():
            unique.update(v)
        results[name] = {"total_groups": len(unique), "median_per_day": round(np.median(per_day)),
                         "mean_per_day": round(np.mean(per_day), 1)}

    raw_total = results["Raw"]["total_groups"]
    print(f"  {'Strategy':<25} {'Groups':>10} {'Med/Day':>8} {'Compress':>10}")
    for name, v in results.items():
        print(f"  {name:<25} {v['total_groups']:>10,} {v['median_per_day']:>8} {raw_total/v['total_groups']:>9.1f}x")
    return results

def eda_4_1(pkg_summary):
    """Heterogeneous-packaging species."""
    print("\n=== EDA-4.1: Heterogeneous-Packaging Species ===")
    hetero = [s for s in pkg_summary if s["dominant_pct"] < 50]
    if hetero:
        print(f"  {len(hetero)} species with no packaging >50%:")
        for s in hetero:
            print(f"    {s['species']}: {s['dominant_pkg']} at {s['dominant_pct']}% ({s['n_pkg_types']} types)")
    else:
        print("  All species have a dominant packaging (>50%).")
    return [s["species"] for s in hetero]

def eda_4_2(data, n):
    """Low-volume species threshold."""
    print("\n=== EDA-4.2: Low-Volume Species Threshold ===")
    species_counts = Counter(data["species"])
    top30 = set(name for name, _ in species_counts.most_common(30))

    sp_days = defaultdict(set)
    for i in range(n):
        sp = data["species"][i]
        if sp not in top30:
            sp_days[sp].add(data["trade_date"][i])

    results = []
    for sp, days in sorted(sp_days.items(), key=lambda x: -len(x[1])):
        sd = sorted(days)
        parsed = [datetime.strptime(d, "%Y.%m.%d") for d in sd]
        max_streak = streak = 1
        for j in range(1, len(parsed)):
            if (parsed[j] - parsed[j-1]).days <= 7:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1
        results.append({"species": sp, "total_days": len(days), "max_streak": max_streak,
                        "viable": len(days) >= 100})

    viable = sum(1 for r in results if r["viable"])
    print(f"  Non-top-30 species: {len(results)}")
    print(f"  Viable (>=100 days): {viable}")
    print(f"  Not viable (<100 days): {len(results) - viable}")
    return results

def decision_summary(state_res, pkg_summary, cv_res, spec_cv_res, pkg_corr_res, qty_res, blend_res):
    """Apply decision tree and produce per-species strategy."""
    print("\n" + "=" * 70)
    print("DECISION SUMMARY")
    print("=" * 70)

    decisions = {}
    for sp, cv_r in cv_res.items():
        # State
        si = next((s for s in state_res["summary"] if s["species"] == sp), None)
        if si and si["dominant_pct"] >= 90:
            state_v = f"filter:{si['dominant_state']}"
        elif sp in state_res.get("multi_state_prices", {}) and state_res["multi_state_prices"][sp]["ratio"] > 1.5:
            state_v = "partition"
        else:
            state_v = "aggregate"

        # Packaging (with EDA-3.1 confirmation)
        if cv_r["ratio"] <= 1.5:
            br = blend_res.get(sp, {})
            if br.get("corr", 0) > 0.95 and br.get("lag1_blend", 0) > 0.8:
                pkg_v = "blend"
            else:
                pkg_v = "dominant-pkg"
        elif sp in pkg_corr_res:
            pc = pkg_corr_res[sp]
            if pc["corr_weighted"] > 0.85 and pc["price_ratio"] < 1.5:
                pkg_v = "blend"
            elif pc["corr_weighted"] > 0.85:
                pkg_v = "dominant-pkg"
            else:
                pkg_v = "separate"
        else:
            pkg_v = "blend"

        # Spec
        sr = spec_cv_res.get(sp, {})
        spec_v = "aggregate" if sr.get("ratio", 0) <= 1.5 else "spec-class"

        # Weighting
        wt = "weighted" if qty_res.get(sp, {}).get("corr", 1.0) > 0.98 else "unweighted"

        # GROUP BY
        group_by = ["trade_date", "species"]
        if state_v == "partition":
            group_by.append("state")
        if pkg_v in ("separate", "dominant-pkg"):
            group_by.append("packaging")
        if spec_v == "spec-class":
            group_by.append("spec_class")

        decisions[sp] = {"state": state_v, "packaging": pkg_v, "spec": spec_v,
                         "weighting": wt, "group_by": group_by}

    print(f"\n  {'Species':<15} {'State':<20} {'Packaging':<15} {'Spec':<12} {'Weight':<10} GROUP BY")
    for sp, d in decisions.items():
        gb = ", ".join(d["group_by"])
        print(f"  {sp:<15} {d['state']:<20} {d['packaging']:<15} {d['spec']:<12} {d['weighting']:<10} ({gb})")

    # Pipeline recommendation
    n_partition = sum(1 for d in decisions.values() if d["state"] == "partition")
    n_filter = sum(1 for d in decisions.values() if d["state"].startswith("filter"))
    print(f"\n  Pipeline: {len(decisions)-n_partition} species single model, {n_partition} need state partition, {n_filter} use dominant-state filter")

    # Dominant GROUP BY
    patterns = Counter(tuple(d["group_by"]) for d in decisions.values())
    dom_pattern = patterns.most_common(1)[0]
    group_cols = ", ".join(dom_pattern[0])
    print(f"  Dominant GROUP BY: ({group_cols}) — {dom_pattern[1]}/{len(decisions)} species")

    print(f"\n  DuckDB view:")
    print(f"    CREATE OR REPLACE VIEW v_daily_prices AS")
    print(f"    SELECT {group_cols},")
    print(f"           SUM(quantity) AS total_quantity, MAX(price_high) AS price_high,")
    print(f"           MIN(price_low) AS price_low, CAST(AVG(price_avg) AS INTEGER) AS price_avg,")
    print(f"           COUNT(*) AS n_lots")
    print(f"    FROM read_parquet('data/parquet/prices/**/*.parquet', hive_partitioning=true)")
    print(f"    WHERE state IS NOT NULL AND packaging IS NOT NULL")
    print(f"    GROUP BY {group_cols}")
    print(f"    ORDER BY {group_cols};")

    return decisions


def main():
    data, n = load_data()
    species_counts = Counter(data["species"])
    top10 = [name for name, _ in species_counts.most_common(10)]
    print(f"Top 10 species: {', '.join(top10)}")

    # Phase 1
    r_1_0 = eda_1_0(data, n)
    r_1_1 = eda_1_1(data, n)
    r_1_2 = eda_1_2(data, n)
    r_1_3 = eda_1_3(data, n)

    # Phase 2
    r_2_1 = eda_2_1(data, n, top10)
    r_2_1b = eda_2_1b(data, n, top10)
    r_2_2 = eda_2_2(data, n, top10)
    r_2_3 = eda_2_3(data, n, top10)
    r_2_4 = eda_2_4(data, n, top10, r_1_1)

    # Phase 3
    r_3_1 = eda_3_1(data, n, top10, r_1_1, r_2_4)
    r_3_2 = eda_3_2(data, n)

    # Phase 4
    r_4_1 = eda_4_1(r_1_1)
    r_4_2 = eda_4_2(data, n)

    # Decision
    decisions = decision_summary(r_1_0, r_1_1, r_2_1, r_2_1b, r_2_2, r_2_4, r_3_1)

    # Save results
    output = {
        "generated_at": datetime.now().isoformat(),
        "total_rows": n,
        "top10": top10,
        "eda_1_0_state": {"multi_state": r_1_0["multi_state"], "divergent": r_1_0["divergent"],
                          "multi_state_prices": r_1_0["multi_state_prices"]},
        "eda_1_1_packaging": r_1_1[:30],
        "eda_1_2_spec": r_1_2,
        "eda_1_3_origin": r_1_3,
        "eda_2_1_packaging_cv": r_2_1,
        "eda_2_1b_spec_cv": r_2_1b,
        "eda_2_2_pkg_correlation": r_2_2,
        "eda_2_3_origin_spread": r_2_3,
        "eda_2_4_quantity": r_2_4,
        "eda_3_1_blended_vs_dominant": r_3_1,
        "eda_3_2_row_reduction": r_3_2,
        "eda_4_1_heterogeneous": r_4_1,
        "eda_4_2_low_volume": {"viable": sum(1 for r in r_4_2 if r["viable"]),
                               "not_viable": sum(1 for r in r_4_2 if not r["viable"]),
                               "total": len(r_4_2)},
        "decisions": decisions,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
