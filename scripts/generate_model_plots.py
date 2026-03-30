"""
Generate comprehensive visualization plots for all prediction model results.

Usage:
    uv run python scripts/generate_model_plots.py

Outputs:
    docs/images/models/best_of_breed_all.png
    docs/images/models/model_heatmap.png
    docs/images/models/loss_function_comparison.png
    docs/images/models/mape_progression.png
    docs/images/models/cpu_vs_gpu.png
    docs/images/models/quantile_bands_top5.png
    docs/images/models/cqr_coverage.png
    docs/images/models/feature_importance_top20.png
    docs/images/models/band_vs_mape.png
    docs/images/models/dl_architecture_boxplot.png
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Use non-interactive backend for server environments
matplotlib.use("Agg")

# Korean font support
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "data" / "poc_results"
OUTPUT_DIR = BASE_DIR / "docs" / "images" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Consistent colour scheme
# ──────────────────────────────────────────────
MODEL_COLORS: dict[str, str] = {
    "LightGBM": "#1f77b4",       # blue
    "TFT": "#d62728",            # red
    "GRU": "#2ca02c",            # green
    "GRU-Q": "#2ca02c",
    "LSTM": "#9467bd",           # purple
    "BiLSTM+Attn": "#8c564b",   # brown
    "CNN-LSTM": "#e377c2",       # pink
    "Transformer": "#ff7f0e",    # orange
    "Transformer-Q": "#ff7f0e",
    "PatchTST": "#17becf",       # cyan
    "GRU+VMD": "#bcbd22",        # yellow-green
    "LSTM+VMD": "#7f7f7f",
    "BiLSTM+Attn+VMD": "#aec7e8",
    "CNN-LSTM+VMD": "#ffbb78",
    "Transformer+VMD": "#ff9896",
    "PatchTST+VMD": "#98df8a",
    "Naive": "#c7c7c7",
    "SMA7": "#d8d8d8",
    "ARIMA": "#b5b5b5",
}

DPI = 150
SASHIMI_CONFIGS = [
    "넙치_활_kg_중",
    "우럭_활_kg_중",
    "방어_선_kg_중_dom",
    "참돔_활_kg_중_dom",
    "농어_활_kg_중_dom",
    "도다리_활_kg_중",
    "감성돔_활_kg_중_dom",
]


# ──────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────

def load_json(filename: str) -> dict[str, Any]:
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"  [WARN] {filename} not found — skipping dependent plots")
        return {}
    with open(path) as f:
        return json.load(f)


def load_v11() -> dict[str, Any]:
    return load_json("poc_v11_results.json")


def load_dl_mse() -> dict[str, Any]:
    return load_json("dl_comparison_results.json")


def load_dl_mae() -> dict[str, Any]:
    return load_json("dl_mae_results.json")


def load_loss_node1() -> dict[str, Any]:
    return load_json("loss_comparison_results.json")


def load_loss_node2() -> dict[str, Any]:
    return load_json("loss_comparison_results_node2.json")


def load_version(filename: str) -> dict[str, Any]:
    return load_json(filename)


# ──────────────────────────────────────────────
# Helper: extract best DL MAPE per config from dl results
# ──────────────────────────────────────────────

def best_dl_mape_per_config(dl_data: dict) -> dict[str, float]:
    """Return {config_key: best_mape} across all models in results_raw / results_pp."""
    out: dict[str, float] = {}
    for section_key in ("results_raw", "results_pp"):
        section = dl_data.get(section_key, {})
        for config, models in section.items():
            for _model, metrics in models.items():
                mape = metrics.get("mape")
                if mape is not None:
                    if config not in out or mape < out[config]:
                        out[config] = mape
    return out


def all_dl_mape_per_model(dl_data: dict) -> dict[str, list[float]]:
    """Return {model_name: [mape, ...]} across all configs (for box plot)."""
    out: dict[str, list[float]] = defaultdict(list)
    for section_key in ("results_raw", "results_pp"):
        section = dl_data.get(section_key, {})
        for _config, models in section.items():
            for model, metrics in models.items():
                mape = metrics.get("mape")
                if mape is not None:
                    out[model].append(mape)
    return dict(out)


# ──────────────────────────────────────────────
# Plot 1 — Best-of-Breed MAPE bar chart
# ──────────────────────────────────────────────

def plot_best_of_breed(v11: dict, dl_mse: dict) -> None:
    print("Plot 1: Best-of-Breed MAPE Bar Chart...")

    if not v11:
        print("  [SKIP] No v11 data")
        return

    # Build best result per config from v11 + best DL
    v11_map: dict[str, float] = {}
    v11_model_map: dict[str, str] = {}
    for r in v11.get("results", []):
        sp = r["species"]
        mape = r.get("mape_best")
        model = r.get("best_variant", "v11")
        if mape is not None and (sp not in v11_map or mape < v11_map[sp]):
            v11_map[sp] = mape
            v11_model_map[sp] = f"LightGBM-{model}"

    dl_best = best_dl_mape_per_config(dl_mse) if dl_mse else {}

    # Merge: use whichever is better
    combined: dict[str, tuple[float, str]] = {}  # {config: (mape, model_type)}
    all_configs = set(v11_map.keys()) | set(dl_best.keys())
    for cfg in all_configs:
        lgbm = v11_map.get(cfg)
        dl = dl_best.get(cfg)
        if lgbm is not None and (dl is None or lgbm <= dl):
            combined[cfg] = (lgbm, "LightGBM")
        elif dl is not None:
            combined[cfg] = (dl, "DL")

    if not combined:
        print("  [SKIP] No combined data")
        return

    # Sort by MAPE ascending (best on top when horizontal)
    sorted_items = sorted(combined.items(), key=lambda x: x[1][0], reverse=True)
    labels = [cfg for cfg, _ in sorted_items]
    mapes = [v[0] for _, v in sorted_items]
    model_types = [v[1] for _, v in sorted_items]

    colors = [MODEL_COLORS.get(mt, "#999999") for mt in model_types]

    fig, ax = plt.subplots(figsize=(14, max(8, len(labels) * 0.4)))
    bars = ax.barh(labels, mapes, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, mapes):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=8,
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=MODEL_COLORS["LightGBM"], label="LightGBM (CPU)"),
        mpatches.Patch(color="#999999", label="Best DL (GPU)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right")

    ax.set_xlabel("MAPE (%)")
    ax.set_title("Best Model Per Config — Final Results", fontsize=14, fontweight="bold")
    ax.axvline(x=20, color="gray", linestyle="--", alpha=0.5, label="20% threshold")
    ax.set_xlim(0, max(mapes) * 1.15)
    plt.tight_layout()
    out = OUTPUT_DIR / "best_of_breed_all.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# Plot 2 — Model × Config MAPE Heatmap
# ──────────────────────────────────────────────

def plot_model_heatmap(v11: dict, dl_mse: dict) -> None:
    print("Plot 2: Model x Config MAPE Heatmap...")

    if not dl_mse:
        print("  [SKIP] No DL MSE data")
        return

    # Collect full-name configs from results_pp (MSE loss)
    section = dl_mse.get("results_pp", dl_mse.get("results_raw", {}))
    all_models_set: set[str] = set()
    for _cfg, models in section.items():
        all_models_set.update(models.keys())

    # Also add LightGBM row
    v11_map: dict[str, float] = {}
    if v11:
        for r in v11.get("results", []):
            sp = r["species"]
            mape = r.get("mape_best")
            if mape is not None and sp not in v11_map:
                v11_map[sp] = mape

    # Use top 10 configs by LightGBM or alphabetical
    configs = list(section.keys())[:20]
    models = sorted(all_models_set)
    models_with_lgbm = ["LightGBM"] + models

    # Build matrix
    matrix = np.full((len(models_with_lgbm), len(configs)), np.nan)
    for j, cfg in enumerate(configs):
        # LightGBM row
        if cfg in v11_map:
            matrix[0, j] = v11_map[cfg]
        # DL rows
        cfg_models = section.get(cfg, {})
        for i, model in enumerate(models, start=1):
            mape = cfg_models.get(model, {}).get("mape")
            if mape is not None:
                matrix[i, j] = mape

    # Shorten config labels for readability
    short_configs = [c.replace("_활_kg_", " ").replace("_선_", " ").replace("_dom", "").replace("_중", " mid") for c in configs]
    short_models = models_with_lgbm

    fig, ax = plt.subplots(figsize=(max(14, len(configs) * 0.8), max(6, len(models_with_lgbm) * 0.55)))
    # Mask NaN for coloring
    masked = np.ma.array(matrix, mask=np.isnan(matrix))
    im = ax.imshow(masked, aspect="auto", cmap="RdYlGn_r", vmin=10, vmax=80)

    ax.set_xticks(range(len(short_configs)))
    ax.set_xticklabels(short_configs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(short_models)))
    ax.set_yticklabels(short_models, fontsize=8)

    # Annotate cells
    for i in range(len(models_with_lgbm)):
        for j in range(len(configs)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if val > 45 else "black")

    plt.colorbar(im, ax=ax, label="MAPE (%)")
    ax.set_title("Model x Config MAPE Heatmap (MSE Loss)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "model_heatmap.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# Plot 3 — Loss Function Comparison
# ──────────────────────────────────────────────

def plot_loss_comparison(node1: dict, node2: dict) -> None:
    print("Plot 3: Loss Function Comparison...")

    if not node1 and not node2:
        print("  [SKIP] No loss comparison data")
        return

    loss_names = ["MSE", "MAE", "MAPE", "sMAPE", "Huber", "LogCosh"]
    models = ["GRU", "Transformer", "CNN-LSTM"]

    # Accumulate MAPE values per (model, loss)
    accum: dict[str, dict[str, list[float]]] = {m: {l: [] for l in loss_names} for m in models}

    for data in (node1, node2):
        if not data:
            continue
        for _cfg, cfg_models in data.get("results", {}).items():
            for model in models:
                if model not in cfg_models:
                    continue
                for loss in loss_names:
                    mape = cfg_models[model].get(loss, {}).get("mape")
                    if mape is not None:
                        accum[model][loss].append(mape)

    # Compute averages
    avg: dict[str, dict[str, float]] = {}
    for model in models:
        avg[model] = {}
        for loss in loss_names:
            vals = accum[model][loss]
            avg[model][loss] = float(np.mean(vals)) if vals else np.nan

    x = np.arange(len(loss_names))
    width = 0.25
    offsets = [-width, 0, width]
    model_colors_list = [MODEL_COLORS.get(m, "#888") for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model, offset, color) in enumerate(zip(models, offsets, model_colors_list)):
        vals = [avg[model][l] for l in loss_names]
        bars = ax.bar(x + offset, vals, width, label=model, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(loss_names)
    ax.set_ylabel("Average MAPE (%) across configs")
    ax.set_title("Loss Function Comparison (GRU / Transformer / CNN-LSTM)", fontsize=13, fontweight="bold")
    ax.legend()
    # Highlight MAE column
    mae_idx = loss_names.index("MAE")
    ax.axvspan(mae_idx - 0.5, mae_idx + 0.5, alpha=0.08, color="green", label="_MAE highlight")
    ax.text(mae_idx, ax.get_ylim()[1] * 0.95, "Winner", ha="center", va="top",
            color="green", fontsize=9, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "loss_function_comparison.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# Plot 4 — v1 → v11 MAPE Progression
# ──────────────────────────────────────────────

def _extract_lgbm_mape(data: dict, species_key: str) -> float | None:
    """Extract best MAPE for the given species from any version JSON."""
    # v11 structure
    for r in data.get("results", []):
        sp = r.get("species", "")
        if sp == species_key or sp.startswith(species_key.split("_")[0]):
            # horizon=7 preference
            if r.get("horizon", 7) == 7:
                for field in ("mape_best", "mape", "mape_v10_baseline"):
                    val = r.get(field)
                    if val is not None:
                        return float(val)
    # backtest_results (v1)
    for r in data.get("backtest_results", []):
        sp = r.get("species", "")
        if sp == species_key.split("_")[0]:
            if r.get("model") not in ("Naive", "SMA-7", "SMA-30", "ExpSmooth") and r.get("horizon", 7) == 7:
                return float(r["mape"])
    return None


def plot_mape_progression() -> None:
    print("Plot 4: MAPE Progression v1 -> v11...")

    version_files = [
        ("v1", "poc_results.json"),
        ("v2", "poc_v2_results.json"),
        ("v3", "poc_v3_results.json"),
        ("v4", "poc_v4_results.json"),
        ("v5", "poc_v5_results.json"),
        ("v6", "poc_v6_results.json"),
        ("v7", "poc_v7_results.json"),
        ("v10", "poc_v10_results.json"),
        ("v11", "poc_v11_results.json"),
    ]

    # Map short species label -> full config key used in v10+
    sashimi = {
        "넙치": "넙치_활_kg_중",
        "우럭": "우럭_활_kg_중",
        "방어": "방어_선_kg_중_dom",
        "참돔": "참돔_활_kg_중_dom",
        "농어": "농어_활_kg_중_dom",
        "도다리": "도다리_활_kg_중",
        "감성돔": "감성돔_활_kg_중_dom",
    }

    # Build extraction lookup per version
    version_labels = [v for v, _ in version_files]
    progression: dict[str, list[float | None]] = {sp: [] for sp in sashimi}

    for _ver, fname in version_files:
        data = load_version(fname)
        if not data:
            for sp in sashimi:
                progression[sp].append(None)
            continue
        for sp, config_key in sashimi.items():
            # Try full config key first, then short name
            mape = _extract_lgbm_mape(data, config_key)
            if mape is None:
                mape = _extract_lgbm_mape(data, sp)
            progression[sp].append(mape)

    # Manual fallback values from known report numbers (v5 = VMD results)
    # These override None when the extraction doesn't find the right row
    v5_known: dict[str, float] = {
        "넙치": 14.92, "우럭": 19.31, "방어": 50.22,
        "참돔": 20.18, "농어": 20.84, "도다리": 24.72, "감성돔": 21.96,
    }
    v5_idx = version_labels.index("v5")
    for sp, val in v5_known.items():
        if progression[sp][v5_idx] is None:
            progression[sp][v5_idx] = val

    fig, ax = plt.subplots(figsize=(12, 7))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    species_labels = list(sashimi.keys())

    color_cycle = plt.cm.tab10(np.linspace(0, 0.9, len(species_labels)))
    for sp, color, marker in zip(species_labels, color_cycle, markers):
        vals = progression[sp]
        valid_x = [i for i, v in enumerate(vals) if v is not None]
        valid_y = [vals[i] for i in valid_x]
        if not valid_y:
            continue
        ax.plot(
            valid_x, valid_y,
            marker=marker, linewidth=2, markersize=6,
            label=sp, color=color,
        )

    ax.set_xticks(range(len(version_labels)))
    ax.set_xticklabels(version_labels)
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("Version")
    ax.set_title("MAPE Evolution: v1 Through v11 (7 Sashimi Species)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = OUTPUT_DIR / "mape_progression.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# Plot 5 — CPU vs GPU Comparison
# ──────────────────────────────────────────────

def plot_cpu_vs_gpu(v11: dict, dl_mse: dict) -> None:
    print("Plot 5: CPU vs GPU comparison...")

    if not v11:
        print("  [SKIP] No v11 data")
        return

    v11_map: dict[str, float] = {}
    for r in v11.get("results", []):
        sp = r["species"]
        mape = r.get("mape_best")
        if mape is not None and sp not in v11_map:
            v11_map[sp] = mape

    dl_best = best_dl_mape_per_config(dl_mse) if dl_mse else {}

    # Find configs that have both
    configs = [cfg for cfg in v11_map if cfg in dl_best]
    if not configs:
        # Fall back: all configs with v11, mark GPU as N/A
        configs = list(v11_map.keys())

    # Sort by v11 MAPE
    configs.sort(key=lambda c: v11_map[c])
    lgbm_vals = [v11_map[c] for c in configs]
    dl_vals = [dl_best.get(c, np.nan) for c in configs]

    x = np.arange(len(configs))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(14, len(configs) * 0.75), 7))
    b1 = ax.bar(x - width / 2, lgbm_vals, width, label="LightGBM v11 (CPU)", color=MODEL_COLORS["LightGBM"], alpha=0.9)
    b2 = ax.bar(x + width / 2, dl_vals, width, label="Best DL (GPU)", color="#ff7f0e", alpha=0.9)

    for bar, val in zip(list(b1) + list(b2), lgbm_vals + dl_vals):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    # Short labels
    short = [c.replace("_활_kg_", "\n").replace("_선_", "\n").replace("_dom", "").replace("_중", "중").replace("_winter", "\nwinter") for c in configs]
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("CPU (LightGBM v11) vs GPU (Best DL) Per Config", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = OUTPUT_DIR / "cpu_vs_gpu.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# Plot 6 — Quantile Band Visualization (top 5 configs)
# ──────────────────────────────────────────────

def plot_quantile_bands(v11: dict) -> None:
    print("Plot 6: Quantile Band Visualization (top 5 configs)...")

    if not v11:
        print("  [SKIP] No v11 data")
        return

    q_bands = v11.get("quantile_bands", {})
    if not q_bands:
        print("  [SKIP] No quantile_bands in v11 data")
        return

    # Build sorted list by mape_best from results
    mape_map: dict[str, float] = {}
    for r in v11.get("results", []):
        sp = r["species"]
        mape = r.get("mape_best")
        if mape is not None and sp not in mape_map:
            mape_map[sp] = mape

    # Top 5 by best MAPE that also have quantile data
    eligible = [(sp, mape_map[sp]) for sp in q_bands if sp in mape_map]
    eligible.sort(key=lambda x: x[1])
    top5 = eligible[:5]

    if not top5:
        print("  [SKIP] No eligible configs")
        return

    fig, axes = plt.subplots(1, len(top5), figsize=(14, 5), sharey=False)
    if len(top5) == 1:
        axes = [axes]

    for ax, (cfg, mape) in zip(axes, top5):
        band = q_bands[cfg]
        q10 = band.get("q10_mean", 0)
        q50 = band.get("q50_mean", 0)
        q90 = band.get("q90_mean", 0)
        bw_pct = band.get("band_width_pct", 0)
        cov_pct = band.get("coverage_pct", 0)

        # Draw band
        ax.barh(
            0, q90 - q10, left=q10, height=0.4,
            color=MODEL_COLORS["LightGBM"], alpha=0.3, label="p10-p90 band"
        )
        ax.scatter([q50], [0], color=MODEL_COLORS["LightGBM"], zorder=5, s=80, label="p50 (median)")
        ax.axvline(q50, color="gray", linestyle="--", alpha=0.4)

        short_name = cfg.replace("_활_kg_", " ").replace("_선_", " ").replace("_dom", "")
        ax.set_title(f"{short_name}\nMAPE={mape:.1f}%", fontsize=8)
        ax.set_xlabel("Price (KRW)", fontsize=7)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7)
        ax.text(
            0.5, -0.22,
            f"Band={bw_pct:.0f}%  Cov={cov_pct:.0f}%",
            transform=ax.transAxes, ha="center", fontsize=7, color="gray"
        )

    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Quantile Bands — Top 5 Configs (p10 / p50 / p90 mean)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "quantile_bands_top5.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# Plot 7 — CQR Coverage Before/After
# ──────────────────────────────────────────────

def plot_cqr_coverage(v11: dict) -> None:
    print("Plot 7: CQR Coverage Before/After...")

    if not v11:
        print("  [SKIP] No v11 data")
        return

    cqr = v11.get("cqr_bands", {})
    q_bands = v11.get("quantile_bands", {})
    if not cqr:
        print("  [SKIP] No cqr_bands in v11 data")
        return

    configs = []
    raw_covs = []
    cqr_covs = []

    for cfg, data in cqr.items():
        raw_cov = q_bands.get(cfg, {}).get("coverage_pct")
        cqr_cov = data.get("cqr_coverage")
        if raw_cov is not None and cqr_cov is not None:
            configs.append(cfg)
            raw_covs.append(raw_cov)
            cqr_covs.append(cqr_cov)

    if not configs:
        print("  [SKIP] No matching quantile/cqr data")
        return

    # Sort by CQR coverage descending
    order = sorted(range(len(configs)), key=lambda i: cqr_covs[i], reverse=True)
    configs = [configs[i] for i in order]
    raw_covs = [raw_covs[i] for i in order]
    cqr_covs = [cqr_covs[i] for i in order]

    x = np.arange(len(configs))
    width = 0.38
    short = [c.replace("_활_kg_", "\n").replace("_선_", "\n").replace("_dom", "").replace("_중", "중").replace("_winter", "\nwinter") for c in configs]

    fig, ax = plt.subplots(figsize=(max(14, len(configs) * 0.75), 6))
    ax.bar(x - width / 2, raw_covs, width, label="Raw quantile coverage", color="#aec7e8", edgecolor="white")
    ax.bar(x + width / 2, cqr_covs, width, label="CQR-adjusted coverage", color=MODEL_COLORS["LightGBM"], edgecolor="white")
    ax.axhline(y=80, color="red", linestyle="--", linewidth=1.5, label="80% target")

    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0, 115)
    ax.set_title("CQR Coverage: Raw vs Adjusted (80% Target)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = OUTPUT_DIR / "cqr_coverage.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# Plot 8 — Feature Importance Top 20
# ──────────────────────────────────────────────

def plot_feature_importance(v11: dict) -> None:
    print("Plot 8: Feature Importance Top 20...")

    if not v11:
        print("  [SKIP] No v11 data")
        return

    accum: dict[str, list[float]] = defaultdict(list)
    for r in v11.get("results", []):
        for feat, val in r.get("importance", {}).items():
            if val > 0:
                accum[feat].append(float(val))

    if not accum:
        print("  [SKIP] No importance data")
        return

    # Average across configs
    avg_imp = {feat: float(np.mean(vals)) for feat, vals in accum.items()}
    sorted_feats = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)[:20]
    feats = [f for f, _ in sorted_feats]
    vals = [v for _, v in sorted_feats]

    # Reversed for barh (top feature at top)
    feats_r = feats[::-1]
    vals_r = vals[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(feats_r, vals_r, color=MODEL_COLORS["LightGBM"], alpha=0.85)
    for bar, val in zip(bars, vals_r):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left", fontsize=8)

    ax.set_xlabel("Average Feature Importance (% gain) across all configs")
    ax.set_title("Top 20 Features — Averaged Across All v11 Configs", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(vals_r) * 1.15)
    plt.tight_layout()
    out = OUTPUT_DIR / "feature_importance_top20.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# Plot 9 — Band Width vs MAPE Scatter
# ──────────────────────────────────────────────

def plot_band_vs_mape(v11: dict) -> None:
    print("Plot 9: Band Width vs MAPE Scatter...")

    if not v11:
        print("  [SKIP] No v11 data")
        return

    q_bands = v11.get("quantile_bands", {})
    mape_map: dict[str, float] = {}
    for r in v11.get("results", []):
        sp = r["species"]
        mape = r.get("mape_best")
        if mape is not None and sp not in mape_map:
            mape_map[sp] = mape

    if not q_bands:
        print("  [SKIP] No quantile_bands data")
        return

    xs, ys, labels = [], [], []
    for cfg, band in q_bands.items():
        mape = mape_map.get(cfg)
        bw = band.get("band_width_pct")
        if mape is not None and bw is not None:
            xs.append(mape)
            ys.append(bw)
            short = cfg.replace("_활_kg_", " ").replace("_선_", " ").replace("_dom", "").replace("_중", "중")
            labels.append(short)

    if not xs:
        print("  [SKIP] No matching data")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(xs, ys, c=xs, cmap="RdYlGn_r", s=80, zorder=3, vmin=min(xs), vmax=max(xs))
    plt.colorbar(scatter, ax=ax, label="MAPE (%)")

    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(
            lbl, (x, y),
            textcoords="offset points", xytext=(5, 3),
            fontsize=7, alpha=0.8,
        )

    ax.set_xlabel("MAPE (%)")
    ax.set_ylabel("Band Width % (p10-p90 / p50)")
    ax.set_title("Price Band Width vs MAPE — Confidence-Accuracy Tradeoff", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUTPUT_DIR / "band_vs_mape.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# Plot 10 — DL Architecture Box Plot
# ──────────────────────────────────────────────

def plot_dl_architecture_boxplot(dl_mse: dict, dl_mae: dict) -> None:
    print("Plot 10: DL Architecture Boxplot...")

    if not dl_mse and not dl_mae:
        print("  [SKIP] No DL data")
        return

    # Combine MSE and MAE results into per-model distributions
    accum: dict[str, list[float]] = defaultdict(list)
    for data in (dl_mse, dl_mae):
        if not data:
            continue
        mape_by_model = all_dl_mape_per_model(data)
        for model, vals in mape_by_model.items():
            accum[model].extend(vals)

    if not accum:
        print("  [SKIP] No model data")
        return

    # Preferred display order
    preferred_order = ["GRU", "LSTM", "BiLSTM+Attn", "CNN-LSTM", "Transformer", "PatchTST",
                       "GRU+VMD", "LSTM+VMD", "BiLSTM+Attn+VMD", "CNN-LSTM+VMD",
                       "Transformer+VMD", "PatchTST+VMD"]
    model_order = [m for m in preferred_order if m in accum]
    # Append any not in preferred list
    model_order += [m for m in accum if m not in model_order]

    data_list = [accum[m] for m in model_order]
    colors = [MODEL_COLORS.get(m, "#888") for m in model_order]

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(
        data_list,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.4),
        whiskerprops=dict(linewidth=1.2),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(model_order) + 1))
    ax.set_xticklabels(model_order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("DL Architecture Comparison — MAPE Distribution Across Configs", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Annotate medians
    for i, vals in enumerate(data_list, start=1):
        if vals:
            med = float(np.median(vals))
            ax.text(i, med + 1, f"{med:.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out = OUTPUT_DIR / "dl_architecture_boxplot.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Generating model visualization plots")
    print(f"Results dir : {RESULTS_DIR}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print("=" * 60)

    # Load data once
    v11 = load_v11()
    dl_mse = load_dl_mse()
    dl_mae = load_dl_mae()
    loss_node1 = load_loss_node1()
    loss_node2 = load_loss_node2()

    # Generate each plot
    plot_best_of_breed(v11, dl_mse)
    plot_model_heatmap(v11, dl_mse)
    plot_loss_comparison(loss_node1, loss_node2)
    plot_mape_progression()
    plot_cpu_vs_gpu(v11, dl_mse)
    plot_quantile_bands(v11)
    plot_cqr_coverage(v11)
    plot_feature_importance(v11)
    plot_band_vs_mape(v11)
    plot_dl_architecture_boxplot(dl_mse, dl_mae)

    # Summary
    print("=" * 60)
    generated = sorted(OUTPUT_DIR.glob("*.png"))
    print(f"Generated {len(generated)} plots:")
    for p in generated:
        size_kb = p.stat().st_size // 1024
        print(f"  {p.name:45s}  {size_kb:5d} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
