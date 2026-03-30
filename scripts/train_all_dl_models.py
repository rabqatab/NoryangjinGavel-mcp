"""
Unified GPU Training: Deep Learning Model Comparison for Fish Price Prediction.

Trains and tests 6 DL architectures (+ TFT results loaded from separate run)
across 7 sashimi species, producing a comprehensive comparison table.

Models:
  1. GRU (2 layers, 64 hidden)
  2. LSTM (2 layers, 64 hidden)
  3. BiLSTM + Additive Attention
  4. CNN-LSTM (Conv1D + LSTM)
  5. TFT (loaded from train_tft.py results)
  6. Simplified Informer (Transformer encoder, direct multi-step decoder)
  7. PatchTST-style (patched input + Transformer encoder)

Preprocessing (v10 — 5 advanced fixes on top of v6 68-feature set):
  Fix 1: Winsorized Mean   — clip daily lot prices to p10/p90 of rolling 30-day window
  Fix 2: Log-Transform Target — predict log(price), exp() for final MAPE
  Fix 3: Outlier Day Removal  — remove days >3σ from rolling 30d mean (training only)
  Fix 4: Origin-Weighted Aggregation — weight lots by origin trading frequency
  Fix 5: Adaptive VMD K   — K=5 for high-volatility, K=3 for low

Runs inside Docker container with PyTorch + CUDA.

Usage (inside Docker):
    python scripts/train_all_dl_models.py

Usage (from host):
    docker run --gpus all -e NVIDIA_DISABLE_REQUIRE=1 --ipc=host ...
"""
import json
import math
import os
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.dataset as ds
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "parquet" / "prices"
OUTPUT_DIR = PROJECT_ROOT / "data" / "poc_results"

# ── Constants ──────────────────────────────────────────────────────

LOOKBACK = 30
HORIZON = 7
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 10
LR = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
MIN_DAYS = 300

FOREIGN_KW = [
    "일본", "중국", "미국", "러시아", "캐나다", "노르웨이", "뉴질랜드", "대만", "칠레",
    "아르헨티나", "영국", "아일랜드", "온두라스", "북한", "(원양)", "인도", "인도네시아",
    "태국", "베트남", "필리핀", "호주", "스페인", "네덜란드", "페루", "모로코", "아프리카",
    "파키스탄", "라스팔마스", "포클랜드", "멕시코",
]

SASHIMI_SPECIES = ["넙치", "우럭", "방어", "참돔", "농어", "도다리", "감성돔",
                    "감숭어", "참숭어", "쭈꾸미", "민어", "깐굴", "바위굴", "수꽃게", "암꽃게"]

SPECIES_CONFIGS = [
    # Original sashimi species (중 grade)
    {"id": "넙치_활_kg_중", "species": "넙치", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False},
    {"id": "우럭_활_kg_중", "species": "우럭", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False},
    {"id": "방어_선_kg_중_dom", "species": "방어", "state": "선", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": True, "regime_split": True},
    {"id": "참돔_활_kg_중_dom", "species": "참돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": False},
    {"id": "농어_활_kg_중_dom", "species": "농어", "state": "활", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": False},
    {"id": "도다리_활_kg_중", "species": "도다리", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": True},
    {"id": "감성돔_활_kg_중_dom", "species": "감성돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": False},
    # Additional species
    {"id": "감숭어_활_kg_중", "species": "감숭어", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False},
    {"id": "참숭어_활_kg_중", "species": "참숭어", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False},
    {"id": "쭈꾸미_선_box_중_dom", "species": "쭈꾸미", "state": "선", "pkg": "box", "spec": "중", "domestic": True, "smoothed": False},
    {"id": "민어_선_SP_중", "species": "민어", "state": "선", "pkg": "S/P", "spec": "중", "domestic": False, "smoothed": False},
    {"id": "깐굴_선_box_소", "species": "깐굴", "state": "선", "pkg": "box", "spec": "소", "domestic": False, "smoothed": False},
    {"id": "바위굴_활_box_대", "species": "바위굴", "state": "활", "pkg": "box", "spec": "대", "domestic": False, "smoothed": False},
    {"id": "수꽃게_활_kg_중", "species": "수꽃게", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False},
    {"id": "암꽃게_활_kg_중", "species": "암꽃게", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False},
    # Premium 활어 grades (大/1미/2미)
    {"id": "수꽃게_활_kg_대", "species": "수꽃게", "state": "활", "pkg": "kg", "spec": "대", "domestic": False, "smoothed": False},
    {"id": "암꽃게_활_kg_대", "species": "암꽃게", "state": "활", "pkg": "kg", "spec": "대", "domestic": False, "smoothed": False},
    {"id": "넙치_활_kg_2미", "species": "넙치", "state": "활", "pkg": "kg", "spec": "2미", "domestic": False, "smoothed": False},
    {"id": "참돔_활_kg_2미_dom", "species": "참돔", "state": "활", "pkg": "kg", "spec": "2미", "domestic": True, "smoothed": False},
    {"id": "농어_활_kg_1미_dom", "species": "농어", "state": "활", "pkg": "kg", "spec": "1미", "domestic": True, "smoothed": False},
]

KOREAN_HOLIDAYS = {
    y: h for y, h in {
        2018: {"seollal": "2018.02.16", "chuseok": "2018.09.24"},
        2019: {"seollal": "2019.02.05", "chuseok": "2019.09.13"},
        2020: {"seollal": "2020.01.25", "chuseok": "2020.10.01"},
        2021: {"seollal": "2021.02.12", "chuseok": "2021.09.21"},
        2022: {"seollal": "2022.02.01", "chuseok": "2022.09.10"},
        2023: {"seollal": "2023.01.22", "chuseok": "2023.09.29"},
        2024: {"seollal": "2024.02.10", "chuseok": "2024.09.17"},
        2025: {"seollal": "2025.01.29", "chuseok": "2025.10.06"},
    }.items()
}

MODEL_NAMES = ["GRU", "LSTM", "BiLSTM+Attn", "CNN-LSTM", "TFT", "Transformer", "PatchTST"]


def is_foreign(origin: Optional[str]) -> bool:
    if not origin:
        return False
    return any(kw in origin for kw in FOREIGN_KW)


def parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y.%m.%d")


def days_to_holiday(dt: datetime) -> dict:
    r = {"seollal": 999, "chuseok": 999}
    for y in [dt.year - 1, dt.year, dt.year + 1]:
        if y not in KOREAN_HOLIDAYS:
            continue
        for name, hd in KOREAN_HOLIDAYS[y].items():
            diff = (parse_date(hd) - dt).days
            if abs(diff) < abs(r[name]):
                r[name] = diff
    return r


# ── Technical Indicators ──────────────────────────────────────────


def ema(prices, span):
    """Exponential moving average."""
    a = np.array(prices, dtype=float)
    out = np.empty_like(a)
    out[0] = a[0]
    alpha = 2 / (span + 1)
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
    return out


def macd_signal(prices):
    """MACD line and signal (12/26/9)."""
    e12 = ema(prices, 12)
    e26 = ema(prices, 26)
    macd_line = e12 - e26
    signal = ema(macd_line, 9)
    return macd_line, signal


def rsi(prices, period=14):
    """Relative Strength Index."""
    a = np.array(prices, dtype=float)
    deltas = np.diff(a)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    out = np.full(len(a), 50.0)
    if len(gains) < period:
        return out
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100 - (100 / (1 + rs))
    return out


# ── Fix 1: Winsorized Mean ────────────────────────────────────────


def winsorized_daily_price(day_prices, recent_30d_prices):
    """Clip extreme lots to p10/p90 of recent 30-day distribution."""
    if len(recent_30d_prices) < 10:
        return float(np.mean(day_prices))
    p10, p90 = np.percentile(recent_30d_prices, [10, 90])
    clipped = [max(p10, min(p90, p)) for p in day_prices]
    return float(np.mean(clipped))


# ── Fix 4: Origin-Weighted Aggregation ───────────────────────────


def origin_weight(origin, origin_freq_30d, max_freq_30d):
    """Higher weight for frequently-trading origins."""
    freq = origin_freq_30d.get(origin, 1)
    return freq / max_freq_30d if max_freq_30d > 0 else 1.0


def weighted_mean(prices, weights):
    """Weighted mean; falls back to simple mean if weights sum to 0."""
    w = np.array(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        return float(np.mean(prices))
    return float(np.dot(prices, w) / total)


# ── Fix 5: Adaptive VMD K ────────────────────────────────────────


def adaptive_vmd_k(prices, window=90):
    """Use K=5 for high-volatility periods, K=3 for low."""
    a = np.array(prices, dtype=float)
    if len(a) < window:
        return 3
    recent_std = np.std(a[-window:])
    overall_std = np.std(a)
    return 5 if recent_std > overall_std else 3


# ── Fix 3: Outlier Day Detection ─────────────────────────────────


def flag_outlier_days(prices, window=30, n_sigma=3):
    """
    Return a boolean array (True = outlier) using rolling mean/std.
    Outliers are flagged but only excluded from training.
    """
    is_outlier = np.zeros(len(prices), dtype=bool)
    for i in range(window, len(prices)):
        window_prices = prices[max(0, i - window):i]
        mu = np.mean(window_prices)
        sigma = np.std(window_prices)
        if sigma > 0 and abs(prices[i] - mu) > n_sigma * sigma:
            is_outlier[i] = True
    return is_outlier


# ── VMD Decomposition ─────────────────────────────────────────────


def decompose_vmd(series, K=3, alpha=2000):
    """Decompose a 1D series into K modes via Variational Mode Decomposition."""
    try:
        from vmdpy import VMD
        u, _, _ = VMD(series, alpha, 0, K, 0, 1, 1e-7)
        return [u[k] for k in range(K)]
    except Exception:
        return [series]


# ── Data Preparation ──────────────────────────────────────────────


def load_parquet_data() -> dict:
    """Load raw parquet data into column-oriented dict."""
    print("Loading parquet data...", end=" ", flush=True)
    dataset = ds.dataset(str(DATA_ROOT), format="parquet", partitioning="hive")
    cols = [
        "trade_date", "species", "state", "origin", "spec",
        "packaging", "price_avg", "price_high", "price_low", "quantity",
    ]
    table = dataset.to_table(columns=cols)
    data = {col: table.column(col).to_pylist() for col in cols}
    n = len(data["trade_date"])
    print(f"{n:,} rows.")
    return data


def build_supply_context(data: dict, n: int) -> dict:
    """Build cross-species supply context arrays indexed by trading date."""
    all_dates = sorted(set(data["trade_date"]))
    date_idx = {d: i for i, d in enumerate(all_dates)}
    nd = len(all_dates)
    sp_qty = {sp: np.zeros(nd) for sp in SASHIMI_SPECIES}
    sp_lots = {sp: np.zeros(nd) for sp in SASHIMI_SPECIES}
    market_lots = np.zeros(nd)
    for i in range(n):
        di = date_idx[data["trade_date"][i]]
        market_lots[di] += 1
        sp = data["species"][i]
        if sp in sp_qty:
            sp_qty[sp][di] += data["quantity"][i]
            sp_lots[sp][di] += 1
    k = 7
    return {
        "dates": all_dates, "date_idx": date_idx,
        "sp_qty": sp_qty, "sp_lots": sp_lots,
        "sp_qty_7d": {s: np.convolve(q, np.ones(k) / k, mode="same") for s, q in sp_qty.items()},
        "sp_lots_7d": {s: np.convolve(l, np.ones(k) / k, mode="same") for s, l in sp_lots.items()},
        "market_lots": market_lots,
        "market_lots_7d": np.convolve(market_lots, np.ones(k) / k, mode="same"),
        "total_sashimi": sum(sp_qty.values()),
        "total_sashimi_7d": np.convolve(sum(sp_qty.values()), np.ones(k) / k, mode="same"),
    }


def build_species_daily_series(data: dict, cfg: dict) -> dict:
    """
    Extract daily price/high/low/lots/origins/qty for one species config.

    v10 Fixes applied:
      Fix 1: Winsorized Mean — clip extreme lots to p10/p90 of rolling 30-day window
      Fix 4: Origin-Weighted Aggregation — weight lots by origin trading frequency

    Returns a dict with gap-filled continuous daily arrays and corresponding date strings.
    Forward-fills non-trading days (for price, high, low; lots/origins/qty become 0 on non-trading days).
    """
    n = len(data["trade_date"])
    # Collect raw per-lot data: (price, qty, origin) tuples per day
    day_lots = defaultdict(list)
    day_highs = defaultdict(list)
    day_lows = defaultdict(list)
    day_origins = defaultdict(set)

    for i in range(n):
        if data["species"][i] != cfg["species"]:
            continue
        if data["state"][i] != cfg["state"]:
            continue
        if data["packaging"][i] != cfg["pkg"]:
            continue
        if data["spec"][i] != cfg["spec"]:
            continue
        if cfg["domestic"] and is_foreign(data["origin"][i]):
            continue
        d = data["trade_date"][i]
        price = data["price_avg"][i]
        qty = data["quantity"][i]
        origin = data["origin"][i] or ""
        day_lots[d].append((price, qty, origin))
        day_highs[d].append(data["price_high"][i])
        day_lows[d].append(data["price_low"][i])
        if origin:
            day_origins[d].add(origin)

    sorted_dates = sorted(day_lots.keys())

    if len(sorted_dates) < MIN_DAYS:
        return {"prices": np.array([]), "dates": [], "trading_dates": set()}

    # Build rolling 30-day lot-price buffer and origin-frequency counter
    trading_records = {}
    rolling_lot_prices = []  # flat list of all lot prices in last 30 days
    origin_days_seen = defaultdict(list)  # origin -> list of day-indices

    for day_i, d in enumerate(sorted_dates):
        lots = day_lots[d]
        day_prices = [lp[0] for lp in lots]
        day_origins_list = [lp[2] for lp in lots]

        # --- Build rolling 30-day origin frequency ---
        origin_freq_30d = defaultdict(int)
        for origin_d, origin_day_idxs in origin_days_seen.items():
            count = sum(1 for di in origin_day_idxs if day_i - 30 <= di < day_i)
            if count > 0:
                origin_freq_30d[origin_d] = count
        max_freq_30d = max(origin_freq_30d.values()) if origin_freq_30d else 1

        # --- Fix 4: per-lot weights by origin frequency ---
        lot_weights = [
            origin_weight(orig, origin_freq_30d, max_freq_30d)
            for orig in day_origins_list
        ]

        # --- Fix 1: winsorize using rolling 30d window, then weighted mean ---
        if len(rolling_lot_prices) >= 10:
            p10, p90 = np.percentile(rolling_lot_prices, [10, 90])
            clipped_prices = [max(p10, min(p90, p)) for p in day_prices]
        else:
            clipped_prices = day_prices

        daily_price = weighted_mean(clipped_prices, lot_weights)

        trading_records[d] = {
            "price": daily_price,
            "high": max(day_highs[d]),
            "low": min(day_lows[d]),
            "n_lots": len(lots),
            "n_origins": len(day_origins[d]),
            "qty": sum(lp[1] for lp in lots),
        }

        # Update rolling buffer (keep ~30 days of lot prices)
        rolling_lot_prices.extend(day_prices)
        if day_i >= 30:
            rolling_lot_prices = [
                p
                for d2 in sorted_dates[max(0, day_i - 29):day_i + 1]
                for p in [lp[0] for lp in day_lots[d2]]
            ]

        # Update origin days-seen index
        for orig in day_origins[d]:
            origin_days_seen[orig].append(day_i)

    # Build continuous daily index with forward-fill
    first_dt = datetime.strptime(sorted_dates[0], "%Y.%m.%d")
    last_dt = datetime.strptime(sorted_dates[-1], "%Y.%m.%d")

    calendar_days = []
    cur = first_dt
    while cur <= last_dt:
        calendar_days.append(cur.strftime("%Y.%m.%d"))
        cur += timedelta(days=1)

    filled_prices = []
    filled_highs = []
    filled_lows = []
    filled_lots = []
    filled_origins = []
    filled_qtys = []
    filled_dates = []
    trading_dates_set = set(sorted_dates)

    last_rec = None
    for d in calendar_days:
        if d in trading_records:
            last_rec = trading_records[d]
            filled_prices.append(last_rec["price"])
            filled_highs.append(last_rec["high"])
            filled_lows.append(last_rec["low"])
            filled_lots.append(last_rec["n_lots"])
            filled_origins.append(last_rec["n_origins"])
            filled_qtys.append(last_rec["qty"])
            filled_dates.append(d)
        elif last_rec is not None:
            # Non-trading day: forward-fill price/high/low; lots/origins/qty = 0
            filled_prices.append(last_rec["price"])
            filled_highs.append(last_rec["high"])
            filled_lows.append(last_rec["low"])
            filled_lots.append(0)
            filled_origins.append(0)
            filled_qtys.append(0)
            filled_dates.append(d)

    return {
        "prices": np.array(filled_prices, dtype=np.float64),
        "highs": np.array(filled_highs, dtype=np.float64),
        "lows": np.array(filled_lows, dtype=np.float64),
        "lots": np.array(filled_lots, dtype=np.float64),
        "origins": np.array(filled_origins, dtype=np.float64),
        "qtys": np.array(filled_qtys, dtype=np.float64),
        "dates": filled_dates,
        "trading_dates": trading_dates_set,
    }


def build_features_68(series: dict, ctx: dict, target_sp: str) -> tuple[np.ndarray, int]:
    """
    Build 68-feature matrix matching LightGBM v6 feature set.
    Operates on the gap-filled continuous daily series.
    Returns (features_array of shape (n, 68), min_offset) where min_offset
    is the number of leading rows that should be skipped (need 90 rows for percentile_90d).

    Features (68 total):
      Calendar (7): dow, month, dom, is_weekend, woy, quarter, is_monday
      Holiday (4): days_seollal, days_chuseok, abs_seollal, abs_chuseok
      Price History (5): price_lag1, price_lag7, price_lag30, price_7d_avg, price_30d_avg
      Momentum (4): pchg_1d, pchg_7d, pchg_30d, pchg_7v30
      Volatility (3): price_std_7d, price_std_30d, price_range_7d
      Own Supply (5): own_qty_7d, own_lots_7d, own_qty_ratio_30d, own_qty_chg_7d, own_lots_chg_7d
      Cross Supply (5): other_sashimi_qty_7d, market_lots_7d, sashimi_concentration,
                         total_sashimi_chg_7d, market_chg_7d
      Seasonal (4): price_vs_month_avg, month_sin, month_cos, is_peak_season
      Weather Proxy (4): gap_days, lots_drop, qty_drop, supply_shock
      Technical Indicators (8): ema_7, ema_30, macd, macd_signal, macd_hist,
                                bollinger_pct, rsi_14, momentum_14d
      Fourier (6): fourier_sin_365, fourier_cos_365, fourier_sin_182, fourier_cos_182,
                   fourier_sin_7, fourier_cos_7
      Advanced Calendar (5): is_friday, is_pre_holiday, consecutive_gap, week_position,
                             days_left_in_week
      Price Distribution (4): skewness_30d, kurtosis_30d, percentile_90d, zscore_30d
      Advanced Supply (4): own_qty_yoy_ratio, origin_diversity_7d, avg_lot_size_7d, hl_spread_7d
    """
    prices = series["prices"]
    highs = series["highs"]
    lows = series["lows"]
    lots = series["lots"]
    origins_arr = series["origins"]
    qtys = series["qtys"]
    dates = series["dates"]
    n = len(prices)

    di_map = ctx["date_idx"]  # maps trading date string -> index in supply arrays

    # Pre-compute technical indicators on full price series
    ema7 = ema(prices, 7)
    ema30 = ema(prices, 30)
    macd_line, macd_sig = macd_signal(prices)
    rsi_14 = rsi(prices, 14)

    # Monthly average prices for seasonal feature
    monthly_avg = defaultdict(list)
    for i in range(n):
        monthly_avg[parse_date(dates[i]).month].append(prices[i])
    monthly_avg = {m: np.mean(v) for m, v in monthly_avg.items()}

    # Build feature matrix row by row
    # We need 90 past rows for percentile_90d, so min_offset = 90
    min_offset = 90
    feat_rows = []
    for i in range(n):
        dt = parse_date(dates[i])
        dow = dt.weekday()
        doy = dt.timetuple().tm_yday

        # Supply context index: map gap-filled date to nearest trading date in ctx
        di = di_map.get(dates[i], 0)

        # Previous day's datetime (for gap calculation)
        dt_prev = parse_date(dates[i - 1]) if i > 0 else dt

        # Holiday distances
        hol = days_to_holiday(dt)

        # Price lags
        p = prices[i]
        p1 = prices[i - 1] if i >= 1 else p
        p7 = prices[i - 7] if i >= 7 else p1
        p30 = prices[i - 30] if i >= 30 else p1
        a7 = np.mean(prices[max(0, i - 7):i]) if i >= 1 else p
        a30 = np.mean(prices[max(0, i - 30):i]) if i >= 1 else p

        # Volatility
        s7 = np.std(prices[max(0, i - 7):i]) if i >= 1 else 0.0
        s30 = np.std(prices[max(0, i - 30):i]) if i >= 1 else 0.0
        r7 = float(max(prices[max(0, i - 7):i]) - min(prices[max(0, i - 7):i])) if i >= 1 else 0.0

        # Own supply (from trading-day supply context)
        oq7 = ctx["sp_qty_7d"][target_sp][di]
        ol7 = ctx["sp_lots_7d"][target_sp][di]
        oq30 = np.mean(ctx["sp_qty"][target_sp][max(0, di - 30):di]) if di >= 1 else oq7
        oqr = oq7 / oq30 if oq30 > 0 else 1.0
        oqc = ((ctx["sp_qty_7d"][target_sp][di] - ctx["sp_qty_7d"][target_sp][max(0, di - 7)])
               / max(ctx["sp_qty_7d"][target_sp][max(0, di - 7)], 1))
        olc = ((ctx["sp_lots_7d"][target_sp][di] - ctx["sp_lots_7d"][target_sp][max(0, di - 7)])
               / max(ctx["sp_lots_7d"][target_sp][max(0, di - 7)], 1))

        # Cross supply
        otq = ctx["total_sashimi_7d"][di] - ctx["sp_qty_7d"][target_sp][di]
        ml7 = ctx["market_lots_7d"][di]
        con = (ctx["sp_qty"][target_sp][di] / ctx["total_sashimi"][di]
               if ctx["total_sashimi"][di] > 0 else 0)
        tsc = ((ctx["total_sashimi_7d"][di] - ctx["total_sashimi_7d"][max(0, di - 7)])
               / max(ctx["total_sashimi_7d"][max(0, di - 7)], 1))
        mc = ((ctx["market_lots_7d"][di] - ctx["market_lots_7d"][max(0, di - 7)])
              / max(ctx["market_lots_7d"][max(0, di - 7)], 1))

        # Seasonal
        pvm = p / monthly_avg.get(dt.month, p) if monthly_avg.get(dt.month, p) > 0 else 1.0

        # Weather proxy
        gap_d = (dt - dt_prev).days
        ld = int(ol7 < ctx["sp_lots_7d"][target_sp][max(0, di - 14)] * 0.5) if di >= 14 else 0
        qd = int(oq7 < oq30 * 0.5) if oq30 > 0 else 0

        # Technical Indicators
        boll_upper = a30 + 2 * s30
        boll_lower = a30 - 2 * s30
        boll_pct = ((p - boll_lower) / (boll_upper - boll_lower)
                    if (boll_upper - boll_lower) > 0 else 0.5)
        mom_14 = ((p - prices[i - 14]) / prices[i - 14] * 100
                  if i >= 14 and prices[i - 14] > 0 else 0)

        # Fourier
        f_sin_365 = np.sin(2 * np.pi * doy / 365)
        f_cos_365 = np.cos(2 * np.pi * doy / 365)
        f_sin_182 = np.sin(2 * np.pi * doy / 182.5)
        f_cos_182 = np.cos(2 * np.pi * doy / 182.5)
        f_sin_7 = np.sin(2 * np.pi * dow / 7)
        f_cos_7 = np.cos(2 * np.pi * dow / 7)

        # Advanced Calendar
        is_friday = int(dow == 4)
        next_gap = (parse_date(dates[i + 1]) - dt).days if i + 1 < n else 1
        is_pre_hol = int(next_gap > 2)
        consec_gap = gap_d
        week_pos = dow / 4 if dow <= 4 else 1.0
        days_left = max(0, 4 - dow)

        # Price Distribution
        window_30 = prices[max(0, i - 30):i] if i >= 1 else np.array([p])
        window_90 = prices[max(0, i - 90):i] if i >= 1 else np.array([p])
        skew_30 = float(scipy_stats.skew(window_30)) if len(window_30) >= 3 else 0.0
        kurt_30 = float(scipy_stats.kurtosis(window_30)) if len(window_30) >= 3 else 0.0
        pct_90 = (float(scipy_stats.percentileofscore(window_90, p)) / 100
                  if len(window_90) >= 3 else 0.5)
        zscore_30 = (p - a30) / s30 if s30 > 0 else 0.0

        # Advanced Supply
        woy_now = dt.isocalendar()[1]
        same_woy_records = ([prices[j] for j in range(max(0, i - 365), max(0, i - 300))
                             if parse_date(dates[j]).isocalendar()[1] == woy_now]
                            if i >= 300 else [])
        yoy_ratio = (oq7 / np.mean(same_woy_records)
                     if same_woy_records and np.mean(same_woy_records) > 0 else 1.0)

        origin_div = np.mean(origins_arr[max(0, i - 7):i]) if i >= 1 else origins_arr[i]
        avg_lot = (np.mean(qtys[max(0, i - 7):i] / np.maximum(lots[max(0, i - 7):i], 1))
                   if i >= 1 else 0.0)
        hl_spread = np.mean(highs[max(0, i - 7):i] - lows[max(0, i - 7):i]) if i >= 1 else 0.0

        features = [
            # Calendar (7)
            dow, dt.month, dt.day, int(dow >= 5), dt.isocalendar()[1], (dt.month - 1) // 3 + 1,
            int(dow == 0),
            # Holiday (4)
            hol["seollal"], hol["chuseok"], abs(hol["seollal"]), abs(hol["chuseok"]),
            # Price History (5)
            p, p1, p7, a7, a30,
            # Momentum (4)
            (p - p1) / p1 * 100 if p1 > 0 else 0,
            (p - p7) / p7 * 100 if p7 > 0 else 0,
            (p - p30) / p30 * 100 if p30 > 0 else 0,
            a7 / a30 - 1 if a30 > 0 else 0,
            # Volatility (3)
            s7, s30, r7,
            # Own Supply (5)
            oq7, ol7, oqr, oqc, olc,
            # Cross Supply (5)
            otq, ml7, con, tsc, mc,
            # Seasonal (4)
            pvm, np.sin(2 * np.pi * dt.month / 12), np.cos(2 * np.pi * dt.month / 12),
            int(dt.month in [11, 12, 1, 2]),
            # Weather Proxy (4)
            gap_d, ld, qd, ld + qd + int(gap_d > 3),
            # Technical Indicators (8)
            ema7[i], ema30[i], macd_line[i], macd_sig[i], macd_line[i] - macd_sig[i],
            boll_pct, rsi_14[i], mom_14,
            # Fourier (6)
            f_sin_365, f_cos_365, f_sin_182, f_cos_182, f_sin_7, f_cos_7,
            # Advanced Calendar (5)
            is_friday, is_pre_hol, consec_gap, week_pos, days_left,
            # Price Distribution (4)
            skew_30, kurt_30, pct_90, zscore_30,
            # Advanced Supply (4)
            yoy_ratio, origin_div, avg_lot, hl_spread,
        ]
        feat_rows.append(features)

    result = np.array(feat_rows, dtype=np.float64)
    # Replace NaN/Inf with 0 — these arise from division by zero in momentum/ratio features
    # on gap-filled days where forward-filled values create zero denominators
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result, min_offset


def normalize_features(train_features: np.ndarray, test_features: np.ndarray):
    """Per-feature z-score normalization. Returns normalized arrays and stats."""
    # Replace any remaining NaN/Inf before computing stats
    train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
    test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero
    train_norm = (train_features - mean) / std
    test_norm = (test_features - mean) / std
    # Final safety: clamp extreme z-scores
    train_norm = np.clip(train_norm, -10, 10)
    test_norm = np.clip(test_norm, -10, 10)
    return train_norm, test_norm, mean, std


# ── Dataset ───────────────────────────────────────────────────────


class SlidingWindowDataset(Dataset):
    """Sliding window dataset: (lookback, features) -> (horizon,) prices."""

    def __init__(self, features: np.ndarray, prices: np.ndarray,
                 lookback: int = LOOKBACK, horizon: int = HORIZON):
        self.X = []
        self.y = []
        for i in range(lookback, len(features) - horizon + 1):
            self.X.append(features[i - lookback:i])
            self.y.append(prices[i:i + horizon])
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])


# ── Models ────────────────────────────────────────────────────────


class GRUModel(nn.Module):
    """2-layer GRU encoder with linear decoder for multi-step output."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, horizon: int = HORIZON, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (batch, lookback, features)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])  # (batch, horizon)


class LSTMModel(nn.Module):
    """2-layer LSTM encoder with linear decoder for multi-step output."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, horizon: int = HORIZON, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM with additive attention and linear decoder."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, horizon: int = HORIZON, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        # Additive attention
        self.attn_w = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1, bias=False)
        self.fc = nn.Linear(hidden_size * 2, horizon)

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Additive attention: score = v^T tanh(W h_t)
        energy = torch.tanh(self.attn_w(lstm_out))  # (batch, seq_len, hidden)
        scores = self.attn_v(energy).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)

        # Weighted sum of encoder outputs
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden*2)
        return self.fc(context)  # (batch, horizon)


class CNNLSTMModel(nn.Module):
    """1D CNN for local pattern extraction followed by LSTM for temporal modeling."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, horizon: int = HORIZON,
                 cnn_filters: int = 32, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_filters),
        )
        self.lstm = nn.LSTM(cnn_filters, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (batch, seq_len, features)
        # Conv1d expects (batch, channels, seq_len)
        c = self.conv(x.permute(0, 2, 1))  # (batch, filters, seq_len)
        c = c.permute(0, 2, 1)  # (batch, seq_len, filters)
        out, _ = self.lstm(c)
        return self.fc(out[:, -1, :])


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # handle odd d_model
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SimpleTransformer(nn.Module):
    """
    Simplified Informer-style model.
    Standard Transformer encoder with direct multi-step linear decoder (non-autoregressive).
    """

    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, horizon: int = HORIZON, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        h = self.pos_enc(h)
        h = self.encoder(h)  # (batch, seq_len, d_model)
        return self.fc(h[:, -1, :])  # use last token


class PatchTransformer(nn.Module):
    """
    PatchTST-style model.
    Patches the input sequence into overlapping patches, then applies a Transformer encoder.
    Channel-independent: operates on the combined feature dimension per patch.
    """

    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, horizon: int = HORIZON,
                 patch_len: int = 7, stride: int = 3, seq_len: int = LOOKBACK,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        # Number of patches
        self.n_patches = (seq_len - patch_len) // stride + 1
        patch_dim = patch_len * input_size
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=self.n_patches + 10)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * self.n_patches, horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)
        patches = []
        for i in range(self.n_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :].reshape(batch_size, -1)  # (batch, patch_len * input_size)
            patches.append(patch)
        patches = torch.stack(patches, dim=1)  # (batch, n_patches, patch_dim)
        h = self.patch_proj(patches)  # (batch, n_patches, d_model)
        h = self.pos_enc(h)
        h = self.encoder(h)  # (batch, n_patches, d_model)
        h = h.reshape(batch_size, -1)  # (batch, n_patches * d_model)
        return self.fc(h)


# ── Quantile Loss ─────────────────────────────────────────────────


class PinballLoss(nn.Module):
    """Quantile regression loss for simultaneous multi-quantile prediction."""

    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred, actual):
        # pred: (batch, n_quantiles), actual: (batch, 1) or (batch,)
        actual = actual.unsqueeze(-1) if actual.dim() == 1 else actual
        losses = []
        for i, q in enumerate(self.quantiles):
            diff = actual - pred[:, i:i + 1]
            loss = torch.where(diff >= 0, q * diff, (q - 1) * diff)
            losses.append(loss.mean())
        return sum(losses) / len(losses)


# ── Quantile Model Variants ──────────────────────────────────────


class GRUQuantile(nn.Module):
    """GRU with 3-output head for quantile prediction (p10, p50, p90)."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, n_quantiles: int = 3, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, n_quantiles)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])  # (batch, n_quantiles)


class TransformerQuantile(nn.Module):
    """Transformer encoder with 3-output head for quantile prediction."""

    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, n_quantiles: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, n_quantiles)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        return self.fc(h[:, -1, :])


class CNNLSTMQuantile(nn.Module):
    """CNN-LSTM with 3-output head for quantile prediction."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, n_quantiles: int = 3,
                 cnn_filters: int = 32, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_filters),
        )
        self.lstm = nn.LSTM(cnn_filters, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, n_quantiles)

    def forward(self, x):
        c = self.conv(x.permute(0, 2, 1))
        c = c.permute(0, 2, 1)
        out, _ = self.lstm(c)
        return self.fc(out[:, -1, :])


QUANTILE_MODELS = {
    "GRU": GRUQuantile,
    "Transformer": TransformerQuantile,
    "CNN-LSTM": CNNLSTMQuantile,
}


# ── Training Utilities ────────────────────────────────────────────


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = EPOCHS, lr: float = LR, device: str = "cuda",
                patience: int = PATIENCE) -> dict:
    """
    Shared training loop with early stopping and LR scheduling.
    Returns dict with training metadata.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n_train = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            n_train += x.size(0)

        # Validate
        model.train(False)
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
                n_val += x.size(0)

        avg_train = train_loss / max(n_train, 1)
        avg_val = val_loss / max(n_val, 1)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_epoch": best_epoch, "best_val_loss": best_val_loss}


def run_test(model: nn.Module, test_loader: DataLoader, device: str,
             price_mean: float, price_std: float) -> dict:
    """
    Run model on test set and compute metrics.
    Returns MAPE (%), RMSE (original scale), and direction accuracy (%).

    Fix 2 (log-transform target): denormalization chain is:
      log_price = pred * price_std + price_mean   (undo z-score)
      raw_price = exp(log_price)                   (undo log transform)
    MAPE is computed on raw (exp'd) prices.
    """
    model.train(False)
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_actuals.append(y.numpy())

    if not all_preds:
        return {"mape": 999.0, "rmse": 999.0, "dir_acc": 0.0, "n_samples": 0}

    preds = np.concatenate(all_preds, axis=0)   # (n, horizon)
    actuals = np.concatenate(all_actuals, axis=0)  # (n, horizon)

    # Denormalize z-score to get log-prices
    preds_log = preds * price_std + price_mean
    actuals_log = actuals * price_std + price_mean

    # Fix 2: exp() to get raw prices
    preds_denorm = np.exp(preds_log)
    actuals_denorm = np.exp(actuals_log)

    # MAPE over all steps (on raw prices)
    valid = actuals_denorm > 0
    if valid.any():
        mape = float(np.mean(
            np.abs(preds_denorm[valid] - actuals_denorm[valid]) / actuals_denorm[valid]
        )) * 100
    else:
        mape = 999.0

    # RMSE (on raw prices)
    rmse = float(np.sqrt(np.mean((preds_denorm - actuals_denorm) ** 2)))

    # Direction accuracy: compare direction of day-1 prediction vs actual
    if preds_denorm.shape[0] > 1:
        pred_first = preds_denorm[:, 0]
        actual_first = actuals_denorm[:, 0]
        pred_dir = pred_first[1:] > pred_first[:-1]
        actual_dir = actual_first[1:] > actual_first[:-1]
        dir_acc = float(np.mean(pred_dir == actual_dir)) * 100
    else:
        dir_acc = 50.0

    return {
        "mape": round(mape, 2),
        "rmse": round(rmse, 0),
        "dir_acc": round(dir_acc, 1),
        "n_samples": len(preds),
    }


# ── Quantile Band Evaluation ──────────────────────────────────────


def evaluate_bands(pred_q10, pred_q50, pred_q90, actuals):
    """Evaluate quantile prediction bands on denormalized (raw) prices."""
    coverage = np.mean((actuals >= pred_q10) & (actuals <= pred_q90)) * 100
    band_width = np.mean(pred_q90 - pred_q10)
    band_pct = np.mean(
        (pred_q90 - pred_q10) / np.where(pred_q50 > 0, pred_q50, 1)
    ) * 100
    mape_p50 = np.mean(
        np.abs(pred_q50 - actuals) / np.where(actuals > 0, actuals, 1)
    ) * 100
    return {
        "mape_p50": round(mape_p50, 1),
        "coverage": round(coverage, 1),
        "band_width_avg": round(float(band_width)),
        "band_pct": round(band_pct, 1),
    }


def compute_conformal_bands(pred_p50_denorm, actuals_denorm, alpha=0.1):
    """
    Compute conformal prediction bands from point prediction residuals.
    Uses the calibration set (all but the last 20% of data) to find the
    (1-alpha) quantile of absolute residuals, then applies that as a
    symmetric band around p50 predictions.
    Returns (conformal_lo, conformal_hi, conformal_coverage, conformal_width).
    """
    n = len(pred_p50_denorm)
    cal_size = int(n * 0.8)
    if cal_size < 10:
        # Not enough data for calibration
        return pred_p50_denorm, pred_p50_denorm, 0.0, 0.0

    cal_residuals = np.abs(pred_p50_denorm[:cal_size] - actuals_denorm[:cal_size])
    q_hat = np.quantile(cal_residuals, 1 - alpha)

    conformal_lo = pred_p50_denorm - q_hat
    conformal_hi = pred_p50_denorm + q_hat

    # Evaluate on the remaining 20%
    test_actuals = actuals_denorm[cal_size:]
    test_lo = conformal_lo[cal_size:]
    test_hi = conformal_hi[cal_size:]
    if len(test_actuals) > 0:
        coverage = float(np.mean((test_actuals >= test_lo) & (test_actuals <= test_hi))) * 100
        width = float(np.mean(test_hi - test_lo))
    else:
        coverage = 0.0
        width = 0.0

    return conformal_lo, conformal_hi, coverage, width


# ── Quantile Dataset ─────────────────────────────────────────────


class QuantileDataset(Dataset):
    """Sliding window dataset for quantile prediction: (lookback, features) -> single next-day price."""

    def __init__(self, features: np.ndarray, prices: np.ndarray,
                 lookback: int = LOOKBACK):
        self.X = []
        self.y = []
        for i in range(lookback, len(features)):
            self.X.append(features[i - lookback:i])
            self.y.append(prices[i])  # single next-day target
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])


# ── Quantile Training ────────────────────────────────────────────


def train_quantile_model(model: nn.Module, train_loader: DataLoader,
                         val_loader: DataLoader, epochs: int = EPOCHS,
                         lr: float = LR, device: str = "cuda",
                         patience: int = PATIENCE) -> dict:
    """
    Training loop using PinballLoss for quantile regression.
    Returns dict with training metadata.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )
    criterion = PinballLoss(quantiles=[0.1, 0.5, 0.9])

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n_train = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)  # (batch, 3)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            n_train += x.size(0)

        # Validate
        model.train(False)
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
                n_val += x.size(0)

        avg_train = train_loss / max(n_train, 1)
        avg_val = val_loss / max(n_val, 1)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_epoch": best_epoch, "best_val_loss": best_val_loss}


def run_quantile_test(model: nn.Module, test_loader: DataLoader, device: str,
                      price_mean: float, price_std: float) -> dict:
    """
    Run quantile model on test set.
    Returns dict with denormalized q10, q50, q90 arrays and actuals.
    Denormalization: exp(pred_q * std + mean) (same log-transform reversal as point models).
    """
    model.train(False)
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x)  # (batch, 3)
            all_preds.append(pred.cpu().numpy())
            all_actuals.append(y.numpy())

    if not all_preds:
        return {"q10": np.array([]), "q50": np.array([]), "q90": np.array([]),
                "actuals": np.array([])}

    preds = np.concatenate(all_preds, axis=0)    # (n, 3)
    actuals = np.concatenate(all_actuals, axis=0)  # (n, 1)
    actuals = actuals.squeeze(-1)  # (n,)

    # Denormalize: undo z-score then exp
    q10_log = preds[:, 0] * price_std + price_mean
    q50_log = preds[:, 1] * price_std + price_mean
    q90_log = preds[:, 2] * price_std + price_mean
    actuals_log = actuals * price_std + price_mean

    q10_raw = np.exp(q10_log)
    q50_raw = np.exp(q50_log)
    q90_raw = np.exp(q90_log)
    actuals_raw = np.exp(actuals_log)

    return {
        "q10": q10_raw,
        "q50": q50_raw,
        "q90": q90_raw,
        "actuals": actuals_raw,
    }


# ── Model Factory ─────────────────────────────────────────────────


def create_model(name: str, input_size: int) -> nn.Module:
    """Instantiate a model by name."""
    if name == "GRU":
        return GRUModel(input_size)
    elif name == "LSTM":
        return LSTMModel(input_size)
    elif name == "BiLSTM+Attn":
        return BiLSTMAttention(input_size)
    elif name == "CNN-LSTM":
        return CNNLSTMModel(input_size)
    elif name == "Transformer":
        return SimpleTransformer(input_size)
    elif name == "PatchTST":
        return PatchTransformer(input_size, seq_len=LOOKBACK)
    else:
        raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── TFT Results Loader ────────────────────────────────────────────


def load_tft_results() -> dict:
    """Load TFT results from a previous train_tft.py run, if available."""
    tft_path = OUTPUT_DIR / "tft_results.json"
    if not tft_path.exists():
        print("  [TFT] No results file found at", tft_path)
        print("  [TFT] Run 'python scripts/train_tft.py' separately first.")
        return {}
    with open(tft_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    for sp_name, sp_data in data.get("results", {}).items():
        if sp_name == "all_species":
            continue
        results[sp_name] = {
            "mape": sp_data.get("mape", 999.0),
            "rmse": 0.0,  # TFT results may not have RMSE
            "dir_acc": 0.0,
            "n_samples": sp_data.get("n_samples", 0),
        }
    return results


# ── Main Pipeline ─────────────────────────────────────────────────


def train_and_evaluate_on_split(
    train_norm, test_norm, train_targets_norm, test_targets_norm,
    price_mean, price_std, n_features, trainable_models, device,
    label_prefix="", use_vmd=False, raw_train_prices=None,
):
    """
    Train all DL models on a single train/test split and return metrics.
    If use_vmd=True, decompose targets using adaptive K (Fix 5), train per-mode models,
    and recombine predictions by summation.

    Fix 2: targets are log-transformed. Denormalization:
      log_price = pred * price_std + price_mean
      raw_price = exp(log_price)

    Fix 5: adaptive VMD K based on recent volatility (needs raw_train_prices).

    Returns (results_dict, timing_dict) where keys are model names.
    """
    results = {}
    timings = {}

    # Build datasets for the raw (non-VMD) path
    train_ds = SlidingWindowDataset(train_norm, train_targets_norm)
    test_ds = SlidingWindowDataset(test_norm, test_targets_norm)

    if len(train_ds) < 50 or len(test_ds) < 10:
        print(f"    SKIP: insufficient samples (train={len(train_ds)}, test={len(test_ds)})")
        return results, timings

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Validation split from training data
    val_split = int(len(train_ds) * 0.9)
    train_subset = torch.utils.data.Subset(train_ds, range(val_split))
    val_subset = torch.utils.data.Subset(train_ds, range(val_split, len(train_ds)))
    train_sub_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_sub_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"    Samples: train={len(train_ds)}, test={len(test_ds)}")

    if not use_vmd:
        # Standard training (no VMD)
        for model_name in trainable_models:
            full_name = f"{label_prefix}{model_name}" if label_prefix else model_name
            print(f"\n    [{full_name}]", end=" ", flush=True)
            t0 = time.time()
            try:
                model = create_model(model_name, n_features)
                n_params = count_parameters(model)
                print(f"({n_params:,} params)", end=" ", flush=True)

                train_info = train_model(
                    model, train_sub_loader, val_sub_loader,
                    epochs=EPOCHS, lr=LR, device=device, patience=PATIENCE,
                )
                print(f"ep={train_info['best_epoch']}", end=" ", flush=True)

                metrics = run_test(model, test_loader, device, price_mean, price_std)
                elapsed = time.time() - t0

                results[full_name] = metrics
                timings[full_name] = round(elapsed, 1)

                print(f"MAPE={metrics['mape']:.1f}% RMSE={metrics['rmse']:.0f} "
                      f"Dir={metrics['dir_acc']:.1f}% [{elapsed:.1f}s]")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAILED: {e} [{elapsed:.1f}s]")
                results[full_name] = {
                    "mape": 999.0, "rmse": 999.0, "dir_acc": 0.0,
                    "n_samples": 0, "error": str(e),
                }
                timings[full_name] = round(elapsed, 1)

            if device == "cuda":
                torch.cuda.empty_cache()
    else:
        # VMD decomposition training
        # Fix 5: adaptive K based on recent volatility of raw (not log) prices
        if raw_train_prices is not None and len(raw_train_prices) > 0:
            VMD_K = adaptive_vmd_k(raw_train_prices)
        else:
            VMD_K = 3
        print(f"    Adaptive VMD K={VMD_K}")

        # Decompose the training targets (1D series of all training target values)
        train_target_flat = train_targets_norm.copy()
        modes = decompose_vmd(train_target_flat, K=VMD_K)

        for model_name in trainable_models:
            full_name = f"{label_prefix}{model_name}+VMD"
            print(f"\n    [{full_name}]", end=" ", flush=True)
            t0 = time.time()

            try:
                mode_models = []
                for k, mode_series in enumerate(modes):
                    # Create dataset for this VMD mode
                    mode_ds = SlidingWindowDataset(train_norm, mode_series)
                    if len(mode_ds) < 50:
                        print(f"mode{k}:skip", end=" ", flush=True)
                        continue
                    mode_val_split = int(len(mode_ds) * 0.9)
                    mode_train_sub = torch.utils.data.Subset(mode_ds, range(mode_val_split))
                    mode_val_sub = torch.utils.data.Subset(mode_ds, range(mode_val_split, len(mode_ds)))
                    mode_train_loader = DataLoader(mode_train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
                    mode_val_loader = DataLoader(mode_val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

                    mode_model = create_model(model_name, n_features)
                    train_model(
                        mode_model, mode_train_loader, mode_val_loader,
                        epochs=EPOCHS, lr=LR, device=device, patience=PATIENCE,
                    )
                    mode_models.append(mode_model)
                    print(f"m{k}", end=" ", flush=True)

                if not mode_models:
                    raise ValueError("All VMD modes skipped")

                # Test: predict each mode and sum
                model_test = mode_models[0]
                model_test.train(False)
                all_preds = []
                all_actuals = []

                with torch.no_grad():
                    for x, y in test_loader:
                        x_dev = x.to(device)
                        combined_pred = torch.zeros(x.size(0), HORIZON).to(device)
                        for mm in mode_models:
                            mm.train(False)
                            combined_pred += mm(x_dev)
                        all_preds.append(combined_pred.cpu().numpy())
                        all_actuals.append(y.numpy())

                preds = np.concatenate(all_preds, axis=0)
                actuals = np.concatenate(all_actuals, axis=0)

                # Denormalize z-score to get log-prices
                preds_log = preds * price_std + price_mean
                actuals_log = actuals * price_std + price_mean

                # Fix 2: exp() to get raw prices
                preds_denorm = np.exp(preds_log)
                actuals_denorm = np.exp(actuals_log)

                # MAPE (on raw prices)
                valid = actuals_denorm > 0
                if valid.any():
                    mape = float(np.mean(
                        np.abs(preds_denorm[valid] - actuals_denorm[valid]) / actuals_denorm[valid]
                    )) * 100
                else:
                    mape = 999.0

                # RMSE (on raw prices)
                rmse = float(np.sqrt(np.mean((preds_denorm - actuals_denorm) ** 2)))

                # Direction accuracy
                if preds_denorm.shape[0] > 1:
                    pred_first = preds_denorm[:, 0]
                    actual_first = actuals_denorm[:, 0]
                    pred_dir = pred_first[1:] > pred_first[:-1]
                    actual_dir = actual_first[1:] > actual_first[:-1]
                    dir_acc = float(np.mean(pred_dir == actual_dir)) * 100
                else:
                    dir_acc = 50.0

                metrics = {
                    "mape": round(mape, 2),
                    "rmse": round(rmse, 0),
                    "dir_acc": round(dir_acc, 1),
                    "n_samples": len(preds),
                }
                elapsed = time.time() - t0
                results[full_name] = metrics
                timings[full_name] = round(elapsed, 1)

                print(f"MAPE={metrics['mape']:.1f}% RMSE={metrics['rmse']:.0f} "
                      f"Dir={metrics['dir_acc']:.1f}% [{elapsed:.1f}s]")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAILED: {e} [{elapsed:.1f}s]")
                results[full_name] = {
                    "mape": 999.0, "rmse": 999.0, "dir_acc": 0.0,
                    "n_samples": 0, "error": str(e),
                }
                timings[full_name] = round(elapsed, 1)

            if device == "cuda":
                torch.cuda.empty_cache()

    return results, timings


def print_comparison_table(title, results_dict, species_list, model_names):
    """Print a MAPE comparison table for a set of models across species."""
    print("\n")
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)

    header = f"  {'Model':<20}"
    for sp in species_list:
        header += f" {sp:>8}"
    header += f" {'AVG':>8}"
    print(header)
    print("  " + "-" * (20 + 9 * (len(species_list) + 1)))

    model_avgs = {}
    for model_name in model_names:
        row = f"  {model_name:<20}"
        mapes = []
        for sp in species_list:
            if model_name in results_dict.get(sp, {}):
                mape = results_dict[sp][model_name]["mape"]
                row += f" {mape:>7.1f}%"
                if mape < 900:
                    mapes.append(mape)
            else:
                row += f" {'N/A':>8}"
        avg_mape = np.mean(mapes) if mapes else 999.0
        model_avgs[model_name] = avg_mape
        row += f" {avg_mape:>7.1f}%"
        print(row)

    # Best per species
    print()
    print("  " + "-" * (20 + 9 * (len(species_list) + 1)))
    best_row = f"  {'BEST':<20}"
    for sp in species_list:
        sp_results = results_dict.get(sp, {})
        valid_models = {m: sp_results[m] for m in model_names if m in sp_results}
        if valid_models:
            best_model = min(valid_models, key=lambda m: valid_models[m].get("mape", 999))
            best_row += f" {valid_models[best_model]['mape']:>7.1f}%"
        else:
            best_row += f" {'N/A':>8}"
    best_row += f" {'':>8}"
    print(best_row)

    return model_avgs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("  Unified DL Model Comparison -- Fish Price Prediction")
    print("  (with v10 preprocessing: winsorized, log-target, outlier removal,")
    print("   origin-weight, adaptive VMD, smoothed target, regime split)")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        mem_gb = props.total_memory / 1e9
        print(f"Memory: {mem_gb:.1f} GB")
    print(f"Lookback: {LOOKBACK}, Horizon: {HORIZON}, Batch: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}, Patience: {PATIENCE}, LR: {LR}")
    print()

    # Load data once
    data = load_parquet_data()
    n_rows = len(data["trade_date"])

    # Build supply context (shared across all species)
    print("Building supply context...", end=" ", flush=True)
    ctx = build_supply_context(data, n_rows)
    print(f"{len(ctx['dates'])} trading days.")

    # Models to train (TFT handled separately)
    trainable_models = ["GRU", "LSTM", "BiLSTM+Attn", "CNN-LSTM", "Transformer", "PatchTST"]

    # Results containers
    # Table 1: raw DL (no preprocessing)
    raw_results = defaultdict(dict)      # {species: {model_name: metrics}}
    raw_timing = defaultdict(dict)
    # Table 2: +preprocessing (smoothed target + regime split + VMD)
    pp_results = defaultdict(dict)
    pp_timing = defaultdict(dict)

    # Cache per-species preprocessed data for quantile training reuse
    species_cache = {}  # {species: {train_norm, test_norm, train_log_norm, test_log_norm, ...}}

    # Load TFT results if available
    print("\n--- Loading TFT results ---")
    tft_results = load_tft_results()
    for sp_name, res in tft_results.items():
        raw_results[sp_name]["TFT"] = res
        print(f"  {sp_name}: MAPE={res['mape']:.1f}%")
    if not tft_results:
        print("  (no TFT results available)")

    # Support config slicing for multi-node parallel training
    # Set CONFIG_SLICE="0:10" to process only configs 0-9
    config_slice = os.environ.get("CONFIG_SLICE", None)
    if config_slice:
        start, end = map(int, config_slice.split(":"))
        configs_to_run = SPECIES_CONFIGS[start:end]
        print(f"\n--- CONFIG_SLICE={config_slice}: processing {len(configs_to_run)}/{len(SPECIES_CONFIGS)} configs ---")
    else:
        configs_to_run = SPECIES_CONFIGS

    # Process each species config
    for cfg in configs_to_run:
        sp = cfg.get("id", cfg["species"])
        species_name = cfg["species"]
        use_smoothed = cfg.get("smoothed", False)
        use_regime = cfg.get("regime_split", False)

        print(f"\n{'=' * 70}")
        flags = ["v10"]
        if use_smoothed:
            flags.append("smoothed")
        if use_regime:
            flags.append("regime_split")
        flag_str = f" [{', '.join(flags)}]"
        print(f"  Species: {sp} (state={cfg['state']}, spec={cfg['spec']}){flag_str}")
        print(f"{'=' * 70}")

        # Build daily series (gap-filled) — Fix 1 + Fix 4 applied inside
        series = build_species_daily_series(data, cfg)
        prices = series["prices"]
        dates = series["dates"]
        if len(prices) < MIN_DAYS:
            print(f"  SKIP: insufficient data ({len(prices)} days < {MIN_DAYS})")
            continue

        # Build 68 features (computed on winsorized/origin-weighted prices)
        features, min_offset = build_features_68(series, ctx, species_name)
        # Trim leading rows that lack enough history for percentile_90d etc.
        features = features[min_offset:]
        prices = prices[min_offset:]
        dates = dates[min_offset:]
        n_features = features.shape[1]
        print(f"  Data: {len(prices)} days (after {min_offset}-day warmup), {n_features} features")

        # Fix 3: flag outlier days (on raw winsorized prices)
        outlier_mask = flag_outlier_days(prices)
        n_outliers = int(outlier_mask.sum())
        print(f"  Outlier days flagged: {n_outliers} (excluded from training only)")

        # Fix 2: log-transform target
        log_prices = np.log(np.maximum(prices, 1.0))

        # Smoothed target: 7-day moving average applied AFTER log-transform
        if use_smoothed and len(log_prices) > 7:
            target_log_prices = np.convolve(log_prices, np.ones(7) / 7, mode="same")
            print(f"  Smoothed target: 7-day MA on log-prices ({len(target_log_prices)} pts)")
        else:
            target_log_prices = log_prices

        # ────────────────────────────────────────────────────────────
        # TABLE 1: Raw DL models (no VMD, no smoothed target, no regime)
        #   Still uses log-target + outlier removal + winsorized prices
        # ────────────────────────────────────────────────────────────
        print(f"\n  --- Table 1: DL with v10 preprocessing ({sp}) ---")

        split_idx = int(len(features) * 0.8)

        # Fix 3: remove outlier days from training split only
        train_outlier_mask = outlier_mask[:split_idx]
        train_clean_mask = ~train_outlier_mask
        n_train_outliers = int(train_outlier_mask.sum())
        if n_train_outliers > 0:
            print(f"  Removing {n_train_outliers} outlier days from training")

        train_feat = features[:split_idx][train_clean_mask]
        test_feat = features[split_idx:]
        train_log_prices = log_prices[:split_idx][train_clean_mask]
        test_log_prices = log_prices[split_idx:]
        # Keep raw prices for adaptive VMD K computation (Fix 5)
        train_prices_raw_for_vmd = prices[:split_idx]

        # Normalize features
        train_norm, test_norm, _, _ = normalize_features(train_feat, test_feat)

        # Fix 2: z-score normalize log-prices (not raw prices)
        log_price_mean = float(np.mean(train_log_prices))
        log_price_std = float(np.std(train_log_prices))
        if log_price_std < 1e-8:
            log_price_std = 1.0

        train_log_norm = (train_log_prices - log_price_mean) / log_price_std
        test_log_norm = (test_log_prices - log_price_mean) / log_price_std

        res, tim = train_and_evaluate_on_split(
            train_norm, test_norm, train_log_norm, test_log_norm,
            log_price_mean, log_price_std, n_features, trainable_models, device,
            use_vmd=False, raw_train_prices=train_prices_raw_for_vmd,
        )
        for k, v in res.items():
            raw_results[sp][k] = v
        for k, v in tim.items():
            raw_timing[sp][k] = v

        # Cache preprocessed data for quantile training reuse
        species_cache[sp] = {
            "train_norm": train_norm,
            "test_norm": test_norm,
            "train_log_norm": train_log_norm,
            "test_log_norm": test_log_norm,
            "log_price_mean": log_price_mean,
            "log_price_std": log_price_std,
            "n_features": n_features,
        }

        # ────────────────────────────────────────────────────────────
        # TABLE 2: +Preprocessing (smoothed log-target + VMD + regime split)
        # ────────────────────────────────────────────────────────────
        if use_regime:
            # Regime split for 방어: train/eval separately for winter vs off-season
            print(f"\n  --- Table 2: +Preprocessing with regime split ({sp}) ---")

            regimes = [
                ({11, 12, 1, 2}, "winter"),
                ({3, 4, 5, 6, 7, 8, 9, 10}, "other"),
            ]
            for regime_months, regime_name in regimes:
                print(f"\n  >> Regime: {regime_name} (months={sorted(regime_months)})")

                # Filter to regime months
                month_mask = np.array([parse_date(d).month in regime_months for d in dates])
                regime_features = features[month_mask]
                regime_target_log = target_log_prices[month_mask]
                regime_prices = prices[month_mask]
                regime_outlier_mask = outlier_mask[month_mask]

                if len(regime_features) < 100:
                    print(f"    SKIP: regime '{regime_name}' has {len(regime_features)} < 100 samples")
                    continue

                # Train/test split: 80/20
                r_split = int(len(regime_features) * 0.8)

                # Fix 3: remove outlier days from training
                r_train_outliers = regime_outlier_mask[:r_split]
                r_train_clean = ~r_train_outliers
                n_r_outliers = int(r_train_outliers.sum())
                if n_r_outliers > 0:
                    print(f"    Removing {n_r_outliers} outlier days from regime training")

                r_train_feat = regime_features[:r_split][r_train_clean]
                r_test_feat = regime_features[r_split:]
                r_train_targets = regime_target_log[:r_split][r_train_clean]
                r_test_targets = regime_target_log[r_split:]
                r_raw_train_prices = regime_prices[:r_split]

                # Normalize features
                r_train_norm, r_test_norm, _, _ = normalize_features(r_train_feat, r_test_feat)

                # Fix 2: z-score normalize log-prices for regime
                r_log_mean = float(np.mean(r_train_targets))
                r_log_std = float(np.std(r_train_targets))
                if r_log_std < 1e-8:
                    r_log_std = 1.0

                r_train_norm_targets = (r_train_targets - r_log_mean) / r_log_std
                r_test_norm_targets = (r_test_targets - r_log_mean) / r_log_std

                # Non-VMD with smoothed log-target
                res_pp, tim_pp = train_and_evaluate_on_split(
                    r_train_norm, r_test_norm, r_train_norm_targets, r_test_norm_targets,
                    r_log_mean, r_log_std, n_features, trainable_models, device,
                    label_prefix="", use_vmd=False,
                    raw_train_prices=r_raw_train_prices,
                )
                # Store regime results with regime label
                for k, v in res_pp.items():
                    pp_results[f"{sp}({regime_name})"][k] = v
                for k, v in tim_pp.items():
                    pp_timing[f"{sp}({regime_name})"][k] = v

                # VMD variant (Fix 5: adaptive K)
                print(f"\n    >> VMD decomposition for {regime_name} regime")
                res_vmd, tim_vmd = train_and_evaluate_on_split(
                    r_train_norm, r_test_norm, r_train_norm_targets, r_test_norm_targets,
                    r_log_mean, r_log_std, n_features, trainable_models, device,
                    use_vmd=True, raw_train_prices=r_raw_train_prices,
                )
                for k, v in res_vmd.items():
                    pp_results[f"{sp}({regime_name})"][k] = v
                for k, v in tim_vmd.items():
                    pp_timing[f"{sp}({regime_name})"][k] = v

            # Use winter result for the main pp_results[sp] entry
            if f"{sp}(winter)" in pp_results:
                pp_results[sp] = dict(pp_results[f"{sp}(winter)"])
                pp_timing[sp] = dict(pp_timing.get(f"{sp}(winter)", {}))

        else:
            # No regime split: use smoothed log-target + VMD on full data
            print(f"\n  --- Table 2: +Preprocessing ({sp}) ---")

            # Fix 3: remove outlier days from training
            pp_train_outliers = outlier_mask[:split_idx]
            pp_train_clean = ~pp_train_outliers
            n_pp_outliers = int(pp_train_outliers.sum())
            if n_pp_outliers > 0:
                print(f"  Removing {n_pp_outliers} outlier days from training")

            train_targets_pp = target_log_prices[:split_idx][pp_train_clean]
            test_targets_pp = target_log_prices[split_idx:]
            pp_train_feat = features[:split_idx][pp_train_clean]

            # Re-normalize features for the cleaned training set
            pp_train_norm, pp_test_norm, _, _ = normalize_features(pp_train_feat, test_feat)

            # Fix 2: z-score normalize log-prices (smoothed or raw log)
            pp_log_mean = float(np.mean(train_targets_pp))
            pp_log_std = float(np.std(train_targets_pp))
            if pp_log_std < 1e-8:
                pp_log_std = 1.0

            train_targets_pp_norm = (train_targets_pp - pp_log_mean) / pp_log_std
            test_targets_pp_norm = (test_targets_pp - pp_log_mean) / pp_log_std

            # Non-VMD with smoothed log-target
            res_pp, tim_pp = train_and_evaluate_on_split(
                pp_train_norm, pp_test_norm, train_targets_pp_norm, test_targets_pp_norm,
                pp_log_mean, pp_log_std, n_features, trainable_models, device,
                use_vmd=False, raw_train_prices=train_prices_raw_for_vmd,
            )
            for k, v in res_pp.items():
                pp_results[sp][k] = v
            for k, v in tim_pp.items():
                pp_timing[sp][k] = v

            # VMD variant (Fix 5: adaptive K)
            print(f"\n    >> VMD decomposition ({sp})")
            res_vmd, tim_vmd = train_and_evaluate_on_split(
                pp_train_norm, pp_test_norm, train_targets_pp_norm, test_targets_pp_norm,
                pp_log_mean, pp_log_std, n_features, trainable_models, device,
                use_vmd=True, raw_train_prices=train_prices_raw_for_vmd,
            )
            for k, v in res_vmd.items():
                pp_results[sp][k] = v
            for k, v in tim_vmd.items():
                pp_timing[sp][k] = v

    # ── Results Summary ───────────────────────────────────────────

    species_list = [cfg.get("id", cfg["species"]) for cfg in SPECIES_CONFIGS if cfg.get("id", cfg["species"]) in raw_results]

    # Table 1: Raw DL models
    raw_model_names = trainable_models + (["TFT"] if tft_results else [])
    raw_avgs = print_comparison_table(
        "Table 1: DL Models — v10 preprocessing (winsorized, log-target, outlier removal)",
        raw_results, species_list, raw_model_names,
    )

    # Table 2: +Preprocessing models
    pp_species_list = sorted(pp_results.keys())
    pp_model_names_set = set()
    for sp_data in pp_results.values():
        pp_model_names_set.update(sp_data.keys())
    # Order: non-VMD first, then VMD variants
    pp_model_names_ordered = []
    for m in trainable_models:
        if m in pp_model_names_set:
            pp_model_names_ordered.append(m)
    for m in trainable_models:
        vmd_name = f"{m}+VMD"
        if vmd_name in pp_model_names_set:
            pp_model_names_ordered.append(vmd_name)
    # Add any remaining
    for m in sorted(pp_model_names_set):
        if m not in pp_model_names_ordered:
            pp_model_names_ordered.append(m)

    pp_avgs = print_comparison_table(
        "Table 2: DL Models — +v10 PP (smoothed log-target + regime split + adaptive VMD)",
        pp_results, pp_species_list, pp_model_names_ordered,
    )

    # Table 3: Best DL+v10 preprocessing vs v10 LightGBM
    print("\n")
    print("=" * 90)
    print("  Table 3: Best DL+v10 PP vs v10 LightGBM")
    print("=" * 90)
    print(f"  {'Species':<20} {'Best DL+PP':>15} {'Model':>20}")
    print("  " + "-" * 55)
    for sp in species_list:
        sp_pp = pp_results.get(sp, {})
        if sp_pp:
            best_model = min(sp_pp, key=lambda m: sp_pp[m].get("mape", 999))
            best_mape = sp_pp[best_model]["mape"]
            print(f"  {sp:<20} {best_mape:>14.1f}% {best_model:>20}")
        else:
            print(f"  {sp:<20} {'N/A':>15} {'N/A':>20}")
    print()
    print("  (Compare against v10 LightGBM results from poc_v10 run)")

    # Overall ranking across both tables
    print("\n")
    print("=" * 50)
    print("  OVERALL MODEL RANKING — Raw (by avg MAPE)")
    print("=" * 50)
    ranked_raw = sorted(raw_avgs.items(), key=lambda x: x[1])
    for rank, (model_name, avg) in enumerate(ranked_raw, 1):
        marker = " <-- BEST" if rank == 1 else ""
        if avg < 900:
            print(f"  {rank}. {model_name:<20} {avg:>7.2f}%{marker}")
        else:
            print(f"  {rank}. {model_name:<20}     N/A{marker}")

    print("\n")
    print("=" * 50)
    print("  OVERALL MODEL RANKING — +Preprocessing (by avg MAPE)")
    print("=" * 50)
    ranked_pp = sorted(pp_avgs.items(), key=lambda x: x[1])
    for rank, (model_name, avg) in enumerate(ranked_pp, 1):
        marker = " <-- BEST" if rank == 1 else ""
        if avg < 900:
            print(f"  {rank}. {model_name:<20} {avg:>7.2f}%{marker}")
        else:
            print(f"  {rank}. {model_name:<20}     N/A{marker}")

    # Direction accuracy table (raw)
    print("\n")
    print("=" * 90)
    print("  Direction Accuracy (%) — Raw -- Higher is Better")
    print("=" * 90)
    header = f"  {'Model':<20}"
    for sp in species_list:
        header += f" {sp:>8}"
    print(header)
    print("  " + "-" * (20 + 9 * len(species_list)))
    for model_name in raw_model_names:
        row = f"  {model_name:<20}"
        for sp in species_list:
            if model_name in raw_results.get(sp, {}):
                da = raw_results[sp][model_name].get("dir_acc", 0.0)
                row += f" {da:>7.1f}%"
            else:
                row += f" {'N/A':>8}"
        print(row)

    # Training time (raw)
    print("\n")
    print("=" * 90)
    print("  TRAINING TIME — Raw (seconds)")
    print("=" * 90)
    header = f"  {'Model':<20}"
    for sp in species_list:
        header += f" {sp:>8}"
    print(header)
    print("  " + "-" * (20 + 9 * len(species_list)))
    for model_name in trainable_models:
        row = f"  {model_name:<20}"
        for sp in species_list:
            t = raw_timing.get(sp, {}).get(model_name, 0)
            row += f" {t:>7.1f}s"
        print(row)

    # ── Quantile Band Predictions ───────────────────────────────────
    print("\n" + "=" * 70)
    print("QUANTILE BAND PREDICTIONS (DL Models)")
    print("=" * 70)
    print("Training p10/p50/p90 quantile models for top 3 DL architectures...")
    print("(GRU, Transformer, CNN-LSTM with PinballLoss)")
    print()

    quantile_model_names = ["GRU", "Transformer", "CNN-LSTM"]
    quantile_results = {}  # {species: {model: {mape_p50, coverage, band_width_avg, band_pct, conformal_*}}}

    for cfg in configs_to_run:
        sp = cfg.get("id", cfg["species"])
        species_name = cfg["species"]
        if sp not in species_cache:
            print(f"\n  [{sp}] SKIP: no cached data (config was skipped earlier)")
            continue

        cache = species_cache[sp]
        train_norm_q = cache["train_norm"]
        test_norm_q = cache["test_norm"]
        train_log_norm_q = cache["train_log_norm"]
        test_log_norm_q = cache["test_log_norm"]
        log_price_mean_q = cache["log_price_mean"]
        log_price_std_q = cache["log_price_std"]
        n_features_q = cache["n_features"]

        print(f"\n  === {sp} ===")

        # Build quantile datasets (next-day prediction, not horizon=7)
        train_q_ds = QuantileDataset(train_norm_q, train_log_norm_q, lookback=LOOKBACK)
        test_q_ds = QuantileDataset(test_norm_q, test_log_norm_q, lookback=LOOKBACK)

        if len(train_q_ds) < 50 or len(test_q_ds) < 10:
            print(f"    SKIP: insufficient samples (train={len(train_q_ds)}, test={len(test_q_ds)})")
            continue

        train_q_loader = DataLoader(train_q_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_q_loader = DataLoader(test_q_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Validation split
        val_split_q = int(len(train_q_ds) * 0.9)
        train_sub_q = torch.utils.data.Subset(train_q_ds, range(val_split_q))
        val_sub_q = torch.utils.data.Subset(train_q_ds, range(val_split_q, len(train_q_ds)))
        train_sub_q_loader = DataLoader(train_sub_q, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_sub_q_loader = DataLoader(val_sub_q, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        quantile_results[sp] = {}

        for qm_name in quantile_model_names:
            print(f"    [{qm_name}]", end=" ", flush=True)
            t0 = time.time()
            try:
                qm_cls = QUANTILE_MODELS[qm_name]
                qm = qm_cls(n_features_q)
                n_params = count_parameters(qm)
                print(f"({n_params:,} params)", end=" ", flush=True)

                train_info = train_quantile_model(
                    qm, train_sub_q_loader, val_sub_q_loader,
                    epochs=EPOCHS, lr=LR, device=device, patience=PATIENCE,
                )
                print(f"ep={train_info['best_epoch']}", end=" ", flush=True)

                # Get quantile predictions
                qt_result = run_quantile_test(
                    qm, test_q_loader, device,
                    log_price_mean_q, log_price_std_q,
                )

                q10 = qt_result["q10"]
                q50 = qt_result["q50"]
                q90 = qt_result["q90"]
                actuals_q = qt_result["actuals"]

                if len(q50) > 0:
                    # Evaluate quantile bands
                    band_metrics = evaluate_bands(q10, q50, q90, actuals_q)

                    # Compute conformal bands from p50 residuals
                    _, _, conf_cov, conf_width = compute_conformal_bands(
                        q50, actuals_q, alpha=0.1,
                    )

                    band_metrics["conformal_coverage"] = round(conf_cov, 1)
                    band_metrics["conformal_width"] = round(float(conf_width))

                    # Store example forecast (last test sample)
                    band_metrics["example_forecast"] = {
                        "p10": round(float(q10[-1])),
                        "p50": round(float(q50[-1])),
                        "p90": round(float(q90[-1])),
                        "actual": round(float(actuals_q[-1])),
                    }

                    elapsed = time.time() - t0
                    quantile_results[sp][qm_name] = band_metrics

                    print(f"p50 MAPE={band_metrics['mape_p50']:.1f}% "
                          f"Coverage={band_metrics['coverage']:.1f}% "
                          f"Band={band_metrics['band_pct']:.1f}% "
                          f"[{elapsed:.1f}s]")
                else:
                    elapsed = time.time() - t0
                    print(f"EMPTY predictions [{elapsed:.1f}s]")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAILED: {e} [{elapsed:.1f}s]")
                quantile_results[sp][qm_name] = {
                    "mape_p50": 999.0, "coverage": 0.0,
                    "band_width_avg": 0, "band_pct": 0.0,
                    "error": str(e),
                }

            if device == "cuda":
                torch.cuda.empty_cache()

    # Print quantile band results table
    print("\n")
    print("=" * 90)
    print("  QUANTILE BAND RESULTS")
    print("=" * 90)
    print(f"  {'Model':<20} {'Species':<10} {'p50 MAPE':>10} {'Coverage':>10} {'Band Width':>15}")
    print("  " + "-" * 65)
    for sp in species_list:
        if sp not in quantile_results:
            continue
        for qm_name in quantile_model_names:
            if qm_name not in quantile_results[sp]:
                continue
            qr = quantile_results[sp][qm_name]
            mape_str = f"{qr['mape_p50']:.1f}%"
            cov_str = f"{qr['coverage']:.1f}%"
            bw = qr.get("band_width_avg", 0)
            bp = qr.get("band_pct", 0)
            bw_str = f"{bw:,} ({bp:.0f}%)"
            print(f"  {qm_name:<20} {sp:<10} {mape_str:>10} {cov_str:>10} {bw_str:>15}")

    # Print conformal band comparison
    print("\n")
    print("=" * 90)
    print("  CONFORMAL BAND COMPARISON (90% target coverage)")
    print("=" * 90)
    print(f"  {'Model':<20} {'Species':<10} {'Quantile Cov':>14} {'Conformal Cov':>15} {'Conf Width':>12}")
    print("  " + "-" * 71)
    for sp in species_list:
        if sp not in quantile_results:
            continue
        for qm_name in quantile_model_names:
            if qm_name not in quantile_results[sp]:
                continue
            qr = quantile_results[sp][qm_name]
            q_cov = f"{qr['coverage']:.1f}%"
            c_cov = f"{qr.get('conformal_coverage', 0):.1f}%"
            c_width = f"{qr.get('conformal_width', 0):,}"
            print(f"  {qm_name:<20} {sp:<10} {q_cov:>14} {c_cov:>15} {c_width:>12}")

    # Print consumer-friendly example forecasts
    print("\n")
    print("=" * 70)
    print("  EXAMPLE FORECASTS (last test sample per species)")
    print("=" * 70)
    for sp in species_list:
        if sp not in quantile_results:
            continue
        # Use best quantile model (lowest p50 MAPE)
        sp_qr = quantile_results[sp]
        valid_qms = {m: r for m, r in sp_qr.items() if "example_forecast" in r}
        if not valid_qms:
            continue
        best_qm = min(valid_qms, key=lambda m: valid_qms[m].get("mape_p50", 999))
        qr = valid_qms[best_qm]
        ex = qr["example_forecast"]
        print(f"\n  {sp} next-day forecast ({best_qm} quantile):")
        print(f"    Low estimate (p10):  {ex['p10']:>10,} KRW/kg")
        print(f"    Expected (p50):      {ex['p50']:>10,} KRW/kg")
        print(f"    High estimate (p90): {ex['p90']:>10,} KRW/kg")
        print(f"    Confidence: {qr['coverage']:.0f}% of actual prices fall within this range")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "lr": LR,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "n_features": 68,
            "feature_set": "v10 (v6 68-feature set + 5 advanced preprocessing fixes)",
            "preprocessing": {
                "fix1_winsorized_mean": "clip daily lot prices to p10/p90 of rolling 30-day window",
                "fix2_log_target": "predict log(price), exp() for final MAPE",
                "fix3_outlier_removal": "remove days >3sigma from rolling 30d mean (training only)",
                "fix4_origin_weighted": "weight lots by origin trading frequency (rolling 30d)",
                "fix5_adaptive_vmd_k": "K=5 for high-volatility, K=3 for low",
                "smoothed_target": "7-day MA on log-prices for species with smoothed=True",
                "regime_split": "winter (11,12,1,2) vs off-season for species with regime_split=True",
            },
            "models_raw": trainable_models + ["TFT"],
            "models_pp": pp_model_names_ordered,
            "species": [c["species"] for c in SPECIES_CONFIGS],
        },
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "N/A",
        "results_raw": {
            sp: {model: raw_results[sp][model] for model in raw_results[sp]}
            for sp in raw_results
        },
        "results_preprocessing": {
            sp: {model: pp_results[sp][model] for model in pp_results[sp]}
            for sp in pp_results
        },
        "timing_raw": dict(raw_timing),
        "timing_preprocessing": dict(pp_timing),
        "ranking_raw": [
            {"rank": i + 1, "model": name, "avg_mape": round(avg, 2)}
            for i, (name, avg) in enumerate(ranked_raw)
            if avg < 900
        ],
        "ranking_preprocessing": [
            {"rank": i + 1, "model": name, "avg_mape": round(avg, 2)}
            for i, (name, avg) in enumerate(ranked_pp)
            if avg < 900
        ],
        "quantile_results": {
            sp: {
                model: qr_data
                for model, qr_data in sp_qr_data.items()
            }
            for sp, sp_qr_data in quantile_results.items()
        },
    }
    out_path = OUTPUT_DIR / "dl_comparison_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
