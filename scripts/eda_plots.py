"""
Generate EDA plots from eda_results.json.

Saves PNG files to docs/images/eda/ for embedding in reports.

Usage:
    uv run python scripts/eda_plots.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "data" / "eda_results.json"
IMG_DIR = PROJECT_ROOT / "docs" / "images" / "eda"

# Korean font fallback
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_results():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def plot_state_dominance(res):
    """EDA-1.0: State dominance distribution (histogram + divergent species bar)."""
    summary = res["eda_1_0_state"]
    multi_prices = summary["multi_state_prices"]

    # Plot 1: Divergent species price ratios
    divergent = {sp: v for sp, v in multi_prices.items() if v["ratio"] > 1.5}
    # Take top 20 by ratio
    top_div = sorted(divergent.items(), key=lambda x: -x[1]["ratio"])[:20]
    if not top_div:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    species = [sp for sp, _ in top_div]
    ratios = [v["ratio"] for _, v in top_div]
    colors = ["#e74c3c" if r > 5 else "#f39c12" if r > 2 else "#3498db" for r in ratios]

    bars = ax.barh(range(len(species)), ratios, color=colors)
    ax.set_yticks(range(len(species)))
    ax.set_yticklabels(species, fontsize=9)
    ax.set_xlabel("Price Ratio Between States (×)")
    ax.set_title("EDA-1.0: Multi-State Price Divergence (Top 20)")
    ax.axvline(x=1.5, color="gray", linestyle="--", linewidth=1, label="1.5× threshold")
    ax.legend()
    ax.invert_yaxis()

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{ratio}×", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(IMG_DIR / "eda_1_0_state_divergence.png", dpi=150)
    plt.close()
    print("  eda_1_0_state_divergence.png")


def plot_packaging_dominance(res):
    """EDA-1.1: Packaging dominance distribution."""
    pkg = res["eda_1_1_packaging"]
    pcts = [s["dominant_pct"] for s in pkg]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pcts, bins=20, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(x=80, color="#e74c3c", linestyle="--", linewidth=2, label="80% threshold")
    ax.axvline(x=50, color="#f39c12", linestyle="--", linewidth=2, label="50% threshold")
    ax.set_xlabel("Dominant Packaging Type (%)")
    ax.set_ylabel("Number of Species")
    ax.set_title("EDA-1.1: Packaging Dominance Distribution (504 species)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "eda_1_1_packaging_dominance.png", dpi=150)
    plt.close()
    print("  eda_1_1_packaging_dominance.png")


def plot_spec_taxonomy(res):
    """EDA-1.2: Spec category distribution."""
    spec = res["eda_1_2_spec"]["global"]

    fig, ax = plt.subplots(figsize=(8, 5))
    cats = list(spec.keys())
    vals = list(spec.values())
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#95a5a6"]
    ax.bar(cats, vals, color=colors[:len(cats)], edgecolor="white")
    ax.set_ylabel("Row Count")
    ax.set_title("EDA-1.2: Spec Type Distribution (2.59M rows)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "eda_1_2_spec_taxonomy.png", dpi=150)
    plt.close()
    print("  eda_1_2_spec_taxonomy.png")


def plot_packaging_cv(res):
    """EDA-2.1: Packaging CV ratio (grouped bar chart)."""
    cv_res = res["eda_2_1_packaging_cv"]
    species = list(cv_res.keys())
    within = [cv_res[sp]["within_cv"] for sp in species]
    across = [cv_res[sp]["across_cv"] for sp in species]
    ratios = [cv_res[sp]["ratio"] for sp in species]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(species))
    w = 0.35
    ax1.bar(x - w/2, within, w, label="Within-Packaging CV", color="#3498db")
    ax1.bar(x + w/2, across, w, label="Across-Packaging CV", color="#e74c3c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(species, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Coefficient of Variation")
    ax1.set_title("EDA-2.1: Within vs Across Packaging CV")
    ax1.legend()

    colors = ["#e74c3c" if r > 1.5 else "#3498db" for r in ratios]
    ax2.bar(x, ratios, color=colors, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(species, rotation=45, ha="right", fontsize=9)
    ax2.axhline(y=1.5, color="gray", linestyle="--", label="1.5× threshold")
    ax2.set_ylabel("CV Ratio (Across / Within)")
    ax2.set_title("EDA-2.1: Packaging CV Ratio")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(IMG_DIR / "eda_2_1_packaging_cv.png", dpi=150)
    plt.close()
    print("  eda_2_1_packaging_cv.png")


def plot_spec_cv(res):
    """EDA-2.1b: Spec CV ratio."""
    cv_res = res["eda_2_1b_spec_cv"]
    species = list(cv_res.keys())
    ratios = [cv_res[sp]["ratio"] for sp in species]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if r > 1.5 else "#3498db" for r in ratios]
    bars = ax.bar(species, ratios, color=colors, edgecolor="white")
    ax.axhline(y=1.5, color="gray", linestyle="--", linewidth=1.5, label="1.5× threshold")
    ax.set_ylabel("CV Ratio (Across-Spec / Within-Spec)")
    ax.set_title("EDA-2.1b: Spec Price Segmentation (within packaging)")
    ax.legend()

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{ratio}×", ha="center", fontsize=9)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "eda_2_1b_spec_cv.png", dpi=150)
    plt.close()
    print("  eda_2_1b_spec_cv.png")


def plot_pkg_correlation(res):
    """EDA-2.2: Packaging correlation heatmap-style bar chart."""
    corr_res = res["eda_2_2_pkg_correlation"]
    species = list(corr_res.keys())
    corr_w = [corr_res[sp]["corr_weighted"] for sp in species]
    corr_s = [corr_res[sp]["corr_simple"] for sp in species]
    price_ratio = [corr_res[sp]["price_ratio"] for sp in species]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(species))
    w = 0.35
    ax1.bar(x - w/2, corr_w, w, label="Weighted", color="#3498db")
    ax1.bar(x + w/2, corr_s, w, label="Simple", color="#2ecc71")
    ax1.axhline(y=0.85, color="#e74c3c", linestyle="--", linewidth=1.5, label="0.85 threshold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(species, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Pearson Correlation")
    ax1.set_title("EDA-2.2: Per-Packaging Series Correlation")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8)

    colors = ["#e74c3c" if r > 1.5 else "#3498db" for r in price_ratio]
    ax2.bar(x, price_ratio, color=colors, edgecolor="white")
    ax2.axhline(y=1.5, color="gray", linestyle="--", label="1.5× threshold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(species, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Mean Price Ratio Between Packaging Types")
    ax2.set_title("EDA-2.2: Price Level Difference")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(IMG_DIR / "eda_2_2_pkg_correlation.png", dpi=150)
    plt.close()
    print("  eda_2_2_pkg_correlation.png")


def plot_origin_spread(res):
    """EDA-2.3: Origin price spread."""
    origin = res["eda_2_3_origin_spread"]
    species = list(origin.keys())
    medians = [origin[sp]["median"] for sp in species]
    p95s = [origin[sp]["p95"] for sp in species]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(species))
    w = 0.35
    ax.bar(x - w/2, medians, w, label="Median Spread", color="#3498db")
    ax.bar(x + w/2, p95s, w, label="P95 Spread", color="#f39c12", alpha=0.7)
    ax.axhline(y=30, color="#e74c3c", linestyle="--", linewidth=1.5, label="30% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Price Spread (%)")
    ax.set_title("EDA-2.3: Origin Price Spread")
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "eda_2_3_origin_spread.png", dpi=150)
    plt.close()
    print("  eda_2_3_origin_spread.png")


def plot_blended_vs_dominant(res):
    """EDA-3.1: Blended vs dominant comparison."""
    blend = res["eda_3_1_blended_vs_dominant"]
    species = list(blend.keys())
    corr = [blend[sp]["corr"] for sp in species]
    lag1_b = [blend[sp]["lag1_blend"] for sp in species]
    lag1_d = [blend[sp]["lag1_dom"] for sp in species]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#2ecc71" if c > 0.95 else "#e74c3c" for c in corr]
    ax1.bar(species, corr, color=colors, edgecolor="white")
    ax1.axhline(y=0.95, color="gray", linestyle="--", linewidth=1.5, label="0.95 threshold")
    ax1.set_ylabel("Correlation (Blended vs Dominant)")
    ax1.set_title("EDA-3.1: Blended vs Dominant-Packaging Correlation")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=9)

    x = np.arange(len(species))
    w = 0.35
    ax2.bar(x - w/2, lag1_b, w, label="Blended Lag-1", color="#3498db")
    ax2.bar(x + w/2, lag1_d, w, label="Dominant Lag-1", color="#2ecc71")
    ax2.axhline(y=0.8, color="gray", linestyle="--", linewidth=1.5, label="0.8 threshold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(species, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Lag-1 Autocorrelation")
    ax2.set_title("EDA-3.1: Series Smoothness (Lag-1 Autocorrelation)")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(IMG_DIR / "eda_3_1_blended_vs_dominant.png", dpi=150)
    plt.close()
    print("  eda_3_1_blended_vs_dominant.png")


def plot_row_reduction(res):
    """EDA-3.2: Row reduction ratio."""
    rr = res["eda_3_2_row_reduction"]
    strategies = list(rr.keys())
    medians = [rr[s]["median_per_day"] for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#95a5a6", "#3498db", "#2ecc71", "#f39c12"]
    ax.bar(strategies, medians, color=colors, edgecolor="white")
    ax.set_ylabel("Median Rows Per Day")
    ax.set_title("EDA-3.2: Row Reduction by Aggregation Strategy")

    for i, (s, m) in enumerate(zip(strategies, medians)):
        ax.text(i, m + 5, str(m), ha="center", fontsize=10, fontweight="bold")

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "eda_3_2_row_reduction.png", dpi=150)
    plt.close()
    print("  eda_3_2_row_reduction.png")


def plot_decision_summary(res):
    """Decision summary: per-species GROUP BY complexity."""
    decisions = res["decisions"]
    species = list(decisions.keys())
    n_dims = [len(decisions[sp]["group_by"]) for sp in species]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if d <= 3 else "#f39c12" if d <= 4 else "#e74c3c" for d in n_dims]
    bars = ax.bar(species, n_dims, color=colors, edgecolor="white")
    ax.set_ylabel("GROUP BY Dimensions")
    ax.set_title("Decision Summary: Aggregation Complexity Per Species")
    ax.set_ylim(0, max(n_dims) + 1)

    for bar, sp in zip(bars, species):
        gb = ", ".join(decisions[sp]["group_by"])
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"({gb})", ha="center", fontsize=7, rotation=30)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "decision_summary.png", dpi=150)
    plt.close()
    print("  decision_summary.png")


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    res = load_results()
    print("Generating plots...")

    plot_state_dominance(res)
    plot_packaging_dominance(res)
    plot_spec_taxonomy(res)
    plot_packaging_cv(res)
    plot_spec_cv(res)
    plot_pkg_correlation(res)
    plot_origin_spread(res)
    plot_blended_vs_dominant(res)
    plot_row_reduction(res)
    plot_decision_summary(res)

    print(f"\nAll plots saved to {IMG_DIR}/")


if __name__ == "__main__":
    main()
