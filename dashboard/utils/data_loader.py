"""Cached data loading for the Noryangjin dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import streamlit as st

from .constants import FOREIGN_KW, SPECIES_CONFIGS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "parquet" / "prices"
RESULTS_DIR = PROJECT_ROOT / "data" / "poc_results"
IMAGES_DIR = PROJECT_ROOT / "docs" / "images" / "models"
SCATTER_DIR = IMAGES_DIR / "scatter"


def _is_foreign(origin: str | None) -> bool:
    if not origin:
        return False
    return any(kw in origin for kw in FOREIGN_KW)


@st.cache_data(ttl=3600)
def load_parquet_full() -> pd.DataFrame:
    """Load entire Parquet dataset (~2.5M rows, ~228MB)."""
    dataset = ds.dataset(str(DATA_ROOT), format="parquet", partitioning="hive")
    table = dataset.to_table(columns=[
        "trade_date", "species", "state", "origin", "spec",
        "packaging", "quantity", "price_high", "price_low", "price_avg",
    ])
    df = table.to_pandas()
    df["trade_date_dt"] = pd.to_datetime(df["trade_date"], format="%Y.%m.%d")
    df["is_foreign"] = df["origin"].apply(_is_foreign)
    return df


@st.cache_data(ttl=3600)
def load_daily_prices() -> pd.DataFrame:
    """Daily aggregated prices per config tuple. ~300K rows."""
    df = load_parquet_full()
    rows = []
    for cfg in SPECIES_CONFIGS:
        mask = (
            (df["species"] == cfg["species"])
            & (df["state"] == cfg["state"])
            & (df["spec"] == cfg["spec"])
        )
        if cfg["pkg"] != "S/P":
            mask &= df["packaging"] == cfg["pkg"]
        if cfg["domestic"]:
            mask &= ~df["is_foreign"]

        sub = df[mask]
        if sub.empty:
            continue

        daily = sub.groupby("trade_date_dt").agg(
            price_avg=("price_avg", "mean"),
            price_high=("price_high", "max"),
            price_low=("price_low", "min"),
            quantity=("quantity", "sum"),
            lot_count=("price_avg", "count"),
        ).reset_index()
        daily["config_id"] = cfg["id"]
        rows.append(daily)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_all_species_daily() -> pd.DataFrame:
    """Daily aggregated prices for ALL species (not just 20 configs). Used for price trends page."""
    df = load_parquet_full()
    daily = df.groupby(["trade_date_dt", "species"]).agg(
        price_avg=("price_avg", "mean"),
        price_high=("price_high", "max"),
        price_low=("price_low", "min"),
        quantity=("quantity", "sum"),
        lot_count=("price_avg", "count"),
    ).reset_index()
    return daily


@st.cache_data(ttl=3600)
def get_all_species() -> list[str]:
    """Return sorted list of all species in the dataset."""
    df = load_parquet_full()
    return sorted(df["species"].dropna().unique().tolist())


@st.cache_data
def load_config_registry() -> list[dict]:
    path = PROJECT_ROOT / "data" / "prediction_config_registry.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("configs", [])


@st.cache_data
def load_dl_v2_results() -> dict:
    """Load DL v2 merged results (original 20 + expansion configs)."""
    path = RESULTS_DIR / "dl_v2_merged.json"
    if not path.exists():
        path = RESULTS_DIR / "dl_v2_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_tft_results() -> dict:
    path = RESULTS_DIR / "tft_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# TFT results use bare species names; map to config IDs
_TFT_SPECIES_TO_CONFIG = {
    "넙치": "넙치_활_kg_중", "우럭": "우럭_활_kg_중", "방어": "방어_선_kg_중_dom",
    "참돔": "참돔_활_kg_중_dom", "농어": "농어_활_kg_중_dom", "도다리": "도다리_활_kg_중",
    "감성돔": "감성돔_활_kg_중_dom",
}


@st.cache_data
def load_dl_results() -> dict:
    """Load DL results and merge TFT from its separate file."""
    path = RESULTS_DIR / "dl_mae_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        dl = json.load(f)

    # Merge TFT into results_preprocessing
    tft = load_tft_results()
    tft_results = tft.get("results", {})
    rp = dl.get("results_preprocessing", {})

    for species_name, metrics in tft_results.items():
        config_id = _TFT_SPECIES_TO_CONFIG.get(species_name)
        if not config_id or not isinstance(metrics, dict):
            continue
        if config_id in rp:
            rp[config_id]["TFT"] = metrics

    dl["results_preprocessing"] = rp
    return dl


@st.cache_data
def load_v11_results() -> dict:
    path = RESULTS_DIR / "poc_v11_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_v10_results() -> dict:
    path = RESULTS_DIR / "poc_v10_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_loss_comparison() -> dict:
    path = RESULTS_DIR / "loss_comparison_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def get_scatter_image(config_id: str, plot_type: str = "timeseries") -> Path | None:
    """Return path to an existing scatter/timeseries PNG, or None."""
    path = SCATTER_DIR / f"{config_id}_{plot_type}.png"
    return path if path.exists() else None


def get_summary_image(name: str) -> Path | None:
    """Return path to a summary model plot, or None."""
    path = IMAGES_DIR / name
    return path if path.exists() else None
