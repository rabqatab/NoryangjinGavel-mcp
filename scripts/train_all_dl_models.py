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

Runs inside Docker container with PyTorch + CUDA.

Usage (inside Docker):
    python scripts/train_all_dl_models.py

Usage (from host):
    docker run --gpus all -e NVIDIA_DISABLE_REQUIRE=1 --ipc=host ...
"""
import json
import math
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

SASHIMI_SPECIES = ["넙치", "우럭", "방어", "참돔", "농어", "도다리", "감성돔"]

SPECIES_CONFIGS = [
    {"species": "넙치", "state": "활", "pkg": "kg", "spec": "중", "domestic": False},
    {"species": "우럭", "state": "활", "pkg": "kg", "spec": "중", "domestic": False},
    {"species": "방어", "state": "선", "pkg": "kg", "spec": "중", "domestic": True},
    {"species": "참돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True},
    {"species": "농어", "state": "활", "pkg": "kg", "spec": "중", "domestic": True},
    {"species": "도다리", "state": "활", "pkg": "kg", "spec": "중", "domestic": False},
    {"species": "감성돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True},
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
    Returns a dict with gap-filled continuous daily arrays and corresponding date strings.
    Forward-fills non-trading days (for price, high, low; lots/origins/qty become 0 on non-trading days).
    """
    n = len(data["trade_date"])
    day_data = defaultdict(lambda: {"prices": [], "highs": [], "lows": [], "origins": set(), "qty": 0})
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
        day_data[d]["prices"].append(data["price_avg"][i])
        day_data[d]["highs"].append(data["price_high"][i])
        day_data[d]["lows"].append(data["price_low"][i])
        if data["origin"][i]:
            day_data[d]["origins"].add(data["origin"][i])
        day_data[d]["qty"] += data["quantity"][i]

    # Compute per-day aggregates on trading days
    trading_records = {}
    for d in sorted(day_data.keys()):
        dd = day_data[d]
        trading_records[d] = {
            "price": float(np.mean(dd["prices"])),
            "high": max(dd["highs"]),
            "low": min(dd["lows"]),
            "n_lots": len(dd["prices"]),
            "n_origins": len(dd["origins"]),
            "qty": dd["qty"],
        }
    sorted_dates = sorted(trading_records.keys())

    if len(sorted_dates) < MIN_DAYS:
        return {"prices": np.array([]), "dates": [], "trading_dates": set()}

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
    price_mean/price_std are used to denormalize the price column (feature index 0).
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

    # Denormalize: targets were normalized with price stats (feature index 0)
    preds_denorm = preds * price_std + price_mean
    actuals_denorm = actuals * price_std + price_mean

    # MAPE over all steps
    valid = actuals_denorm > 0
    if valid.any():
        mape = float(np.mean(
            np.abs(preds_denorm[valid] - actuals_denorm[valid]) / actuals_denorm[valid]
        )) * 100
    else:
        mape = 999.0

    # RMSE
    rmse = float(np.sqrt(np.mean((preds_denorm - actuals_denorm) ** 2)))

    # Direction accuracy: compare direction of day-1 prediction vs actual
    if preds_denorm.shape[0] > 1:
        # Direction: does next-day price go up or down relative to current window?
        # Use first horizon step
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("  Unified DL Model Comparison -- Fish Price Prediction")
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

    # Results: {species: {model_name: {mape, rmse, dir_acc, ...}}}
    all_results = defaultdict(dict)
    timing = defaultdict(dict)

    # Load TFT results if available
    print("\n--- Loading TFT results ---")
    tft_results = load_tft_results()
    for sp_name, res in tft_results.items():
        all_results[sp_name]["TFT"] = res
        print(f"  {sp_name}: MAPE={res['mape']:.1f}%")
    if not tft_results:
        print("  (no TFT results available)")

    # Process each species
    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        print(f"\n{'=' * 70}")
        print(f"  Species: {sp} (state={cfg['state']}, spec={cfg['spec']})")
        print(f"{'=' * 70}")

        # Build daily series (gap-filled)
        series = build_species_daily_series(data, cfg)
        prices = series["prices"]
        dates = series["dates"]
        if len(prices) < MIN_DAYS:
            print(f"  SKIP: insufficient data ({len(prices)} days < {MIN_DAYS})")
            continue

        # Build 68 v6 features
        features, min_offset = build_features_68(series, ctx, sp)
        # Trim leading rows that lack enough history for percentile_90d etc.
        features = features[min_offset:]
        prices = prices[min_offset:]
        dates = dates[min_offset:]
        n_features = features.shape[1]
        print(f"  Data: {len(prices)} days (after {min_offset}-day warmup), {n_features} features")

        # Train/test split: 80/20
        split_idx = int(len(features) * 0.8)
        train_feat = features[:split_idx]
        test_feat = features[split_idx:]
        train_prices = prices[:split_idx]
        test_prices = prices[split_idx:]

        # Normalize features
        train_norm, test_norm, feat_mean, feat_std = normalize_features(train_feat, test_feat)

        # Price stats for denormalization (feature index 11 = price_lag1 in v6,
        # but we use the raw price column from the series for targets).
        # Compute price normalization stats from training prices directly.
        price_mean = float(np.mean(train_prices))
        price_std = float(np.std(train_prices))
        if price_std < 1e-8:
            price_std = 1.0

        # Normalize targets too (using price stats)
        train_prices_norm = (train_prices - price_mean) / price_std
        test_prices_norm = (test_prices - price_mean) / price_std

        # Create datasets
        train_ds = SlidingWindowDataset(train_norm, train_prices_norm)
        test_ds = SlidingWindowDataset(test_norm, test_prices_norm)

        if len(train_ds) < 50 or len(test_ds) < 10:
            print(f"  SKIP: insufficient samples (train={len(train_ds)}, test={len(test_ds)})")
            continue

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        print(f"  Samples: train={len(train_ds)}, test={len(test_ds)}")

        # Also create a validation split from training data for early stopping
        val_split = int(len(train_ds) * 0.9)
        train_subset = torch.utils.data.Subset(train_ds, range(val_split))
        val_subset = torch.utils.data.Subset(train_ds, range(val_split, len(train_ds)))
        train_sub_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_sub_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        for model_name in trainable_models:
            print(f"\n  [{model_name}]", end=" ", flush=True)
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

                all_results[sp][model_name] = metrics
                timing[sp][model_name] = round(elapsed, 1)

                print(f"MAPE={metrics['mape']:.1f}% RMSE={metrics['rmse']:.0f} "
                      f"Dir={metrics['dir_acc']:.1f}% [{elapsed:.1f}s]")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAILED: {e} [{elapsed:.1f}s]")
                all_results[sp][model_name] = {
                    "mape": 999.0, "rmse": 999.0, "dir_acc": 0.0,
                    "n_samples": 0, "error": str(e),
                }
                timing[sp][model_name] = round(elapsed, 1)

            # Free GPU memory between models
            if device == "cuda":
                torch.cuda.empty_cache()

    # ── Results Summary ───────────────────────────────────────────

    print("\n")
    print("=" * 90)
    print("  RESULTS: MAPE (%) -- Lower is Better")
    print("=" * 90)

    # Header
    species_list = [cfg["species"] for cfg in SPECIES_CONFIGS if cfg["species"] in all_results]
    header = f"  {'Model':<15}"
    for sp in species_list:
        header += f" {sp:>8}"
    header += f" {'AVG':>8}"
    print(header)
    print("  " + "-" * (15 + 9 * (len(species_list) + 1)))

    # Per-model rows
    model_avgs = {}
    for model_name in MODEL_NAMES:
        row = f"  {model_name:<15}"
        mapes = []
        for sp in species_list:
            if model_name in all_results.get(sp, {}):
                mape = all_results[sp][model_name]["mape"]
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
    print("  " + "-" * (15 + 9 * (len(species_list) + 1)))
    best_row = f"  {'BEST':<15}"
    for sp in species_list:
        sp_results = all_results.get(sp, {})
        if sp_results:
            best_model = min(sp_results, key=lambda m: sp_results[m].get("mape", 999))
            best_mape = sp_results[best_model]["mape"]
            best_row += f" {best_mape:>7.1f}%"
        else:
            best_row += f" {'N/A':>8}"
    best_row += f" {'':>8}"
    print(best_row)

    best_model_row = f"  {'(model)':<15}"
    for sp in species_list:
        sp_results = all_results.get(sp, {})
        if sp_results:
            best_model = min(sp_results, key=lambda m: sp_results[m].get("mape", 999))
            # Abbreviate long names
            abbrev = best_model[:8]
            best_model_row += f" {abbrev:>8}"
        else:
            best_model_row += f" {'N/A':>8}"
    best_model_row += f" {'':>8}"
    print(best_model_row)

    # Overall ranking
    print("\n")
    print("=" * 50)
    print("  OVERALL MODEL RANKING (by avg MAPE)")
    print("=" * 50)
    ranked = sorted(model_avgs.items(), key=lambda x: x[1])
    for rank, (model_name, avg) in enumerate(ranked, 1):
        marker = " <-- BEST" if rank == 1 else ""
        if avg < 900:
            print(f"  {rank}. {model_name:<15} {avg:>7.2f}%{marker}")
        else:
            print(f"  {rank}. {model_name:<15}     N/A{marker}")

    # Direction accuracy table
    print("\n")
    print("=" * 90)
    print("  RESULTS: Direction Accuracy (%) -- Higher is Better")
    print("=" * 90)
    header = f"  {'Model':<15}"
    for sp in species_list:
        header += f" {sp:>8}"
    print(header)
    print("  " + "-" * (15 + 9 * len(species_list)))
    for model_name in MODEL_NAMES:
        row = f"  {model_name:<15}"
        for sp in species_list:
            if model_name in all_results.get(sp, {}):
                da = all_results[sp][model_name].get("dir_acc", 0.0)
                row += f" {da:>7.1f}%"
            else:
                row += f" {'N/A':>8}"
        print(row)

    # Timing
    print("\n")
    print("=" * 90)
    print("  TRAINING TIME (seconds)")
    print("=" * 90)
    header = f"  {'Model':<15}"
    for sp in species_list:
        header += f" {sp:>8}"
    print(header)
    print("  " + "-" * (15 + 9 * len(species_list)))
    for model_name in trainable_models:
        row = f"  {model_name:<15}"
        for sp in species_list:
            t = timing.get(sp, {}).get(model_name, 0)
            row += f" {t:>7.1f}s"
        print(row)

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
            "feature_set": "v6 (same as LightGBM)",
            "models": MODEL_NAMES,
            "species": [c["species"] for c in SPECIES_CONFIGS],
        },
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "N/A",
        "results": {
            sp: {
                model: all_results[sp][model]
                for model in all_results[sp]
            }
            for sp in all_results
        },
        "timing": dict(timing),
        "ranking": [
            {"rank": i + 1, "model": name, "avg_mape": round(avg, 2)}
            for i, (name, avg) in enumerate(ranked)
            if avg < 900
        ],
    }
    out_path = OUTPUT_DIR / "dl_comparison_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
