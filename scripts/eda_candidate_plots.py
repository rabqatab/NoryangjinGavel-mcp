"""
Generate plots for prediction candidate analysis.

Usage:
    uv run python scripts/eda_candidate_plots.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "data" / "prediction_candidates.json"
IMG_DIR = PROJECT_ROOT / "docs" / "images" / "eda"

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def plot_funnel(res):
    """Funnel chart showing species filtering stages."""
    funnel = res["funnel"]
    stages = ["All Species", "Gate 1\n(≥200 days)", "Gate 2\n(Consistent)", "Gate 3\n(Signal)", "Final"]
    counts = [funnel["all_species"], funnel["gate_1_volume"], funnel["gate_2_consistency"],
              funnel["gate_3_signal"], funnel["final"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#95a5a6", "#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax.bar(stages, counts, color=colors, edgecolor="white", width=0.6)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                str(count), ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("Number of Species")
    ax.set_title("Prediction Candidate Funnel: 504 → 74 Species")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "candidate_funnel.png", dpi=150)
    plt.close()
    print("  candidate_funnel.png")


def plot_tier_a_b(res):
    """Tier A and B species: lag-1 autocorrelation vs volume."""
    candidates = res["candidates"]
    ab = [c for c in candidates if c["tier"] in ("A", "B")]

    fig, ax = plt.subplots(figsize=(14, 7))

    for c in ab:
        color = "#e74c3c" if c["tier"] == "A" else "#3498db"
        marker = "o" if c["tier"] == "A" else "s"
        ax.scatter(c["total_qty"], c["lag1_7d"], color=color, s=80, marker=marker,
                   edgecolors="white", linewidth=0.5, zorder=3)
        ax.annotate(c["species"], (c["total_qty"], c["lag1_7d"]),
                    fontsize=8, ha="left", va="bottom", xytext=(5, 3),
                    textcoords="offset points")

    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="lag1(7d) = 0.95")
    ax.set_xlabel("Total Quantity Traded")
    ax.set_ylabel("7-Day Smoothed Lag-1 Autocorrelation")
    ax.set_title("Tier A & B Prediction Candidates: Signal Quality vs Market Volume")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))

    # Legend
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label="Tier A (top 15)"),
              Line2D([0], [0], marker="s", color="w", markerfacecolor="#3498db", markersize=10, label="Tier B (16-30)")]
    ax.legend(handles=legend, loc="lower right")

    plt.tight_layout()
    plt.savefig(IMG_DIR / "candidate_tier_ab.png", dpi=150)
    plt.close()
    print("  candidate_tier_ab.png")


def plot_signal_quality_bars(res):
    """Tier A species: raw lag-1 vs 7d-smoothed lag-1."""
    candidates = res["candidates"]
    tier_a = [c for c in candidates if c["tier"] == "A"]

    fig, ax = plt.subplots(figsize=(12, 6))
    species = [c["species"] for c in tier_a]
    lag1_raw = [c["lag1"] for c in tier_a]
    lag1_7d = [c["lag1_7d"] for c in tier_a]

    x = np.arange(len(species))
    w = 0.35
    ax.bar(x - w/2, lag1_raw, w, label="Raw Daily Lag-1", color="#e74c3c", alpha=0.8)
    ax.bar(x + w/2, lag1_7d, w, label="7-Day Smoothed Lag-1", color="#2ecc71", alpha=0.8)

    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="0.8 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Lag-1 Autocorrelation")
    ax.set_title("Tier A: Daily vs 7-Day Smoothed Signal Quality")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(IMG_DIR / "candidate_signal_quality.png", dpi=150)
    plt.close()
    print("  candidate_signal_quality.png")


def plot_price_volatility(res):
    """CV (price volatility) for Tier A+B species."""
    candidates = res["candidates"]
    ab = [c for c in candidates if c["tier"] in ("A", "B")]
    ab.sort(key=lambda x: x["cv"])

    fig, ax = plt.subplots(figsize=(14, 6))
    species = [c["species"] for c in ab]
    cvs = [c["cv"] for c in ab]
    colors = ["#2ecc71" if c < 0.5 else "#f39c12" if c < 1.0 else "#e74c3c" for c in cvs]

    ax.barh(range(len(species)), cvs, color=colors)
    ax.set_yticks(range(len(species)))
    ax.set_yticklabels(species, fontsize=8)
    ax.set_xlabel("Coefficient of Variation (std/mean)")
    ax.set_title("Price Volatility: Tier A & B Candidates")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="CV = 0.5")
    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5, label="CV = 1.0")
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(IMG_DIR / "candidate_volatility.png", dpi=150)
    plt.close()
    print("  candidate_volatility.png")


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    res = load()
    print("Generating candidate plots...")
    plot_funnel(res)
    plot_tier_a_b(res)
    plot_signal_quality_bars(res)
    plot_price_volatility(res)
    print(f"\nAll plots saved to {IMG_DIR}/")


if __name__ == "__main__":
    main()
