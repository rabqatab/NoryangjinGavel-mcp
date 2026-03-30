"""
Loss Function Comparison: 6 losses x 3 DL models x 20 species configs.

Tests whether changing the training loss function improves MAPE on the test set.

Loss functions compared:
  1. MSE      -- standard mean squared error (baseline)
  2. MAE      -- mean absolute error (L1)
  3. MAPE     -- direct MAPE loss (aligns training with evaluation metric)
  4. sMAPE    -- symmetric MAPE (handles zero actuals)
  5. Huber    -- MSE for small errors, MAE for large (robust to outliers)
  6. LogCosh  -- smoother than Huber, twice differentiable

Models: GRU, Transformer, CNN-LSTM
Preprocessing: v10 (winsorized, log-transform target, outlier removal, origin-weighted)
Features: 68 (same as train_all_dl_models.py)
Lookback: 30 days, Horizon: 7 days

Usage (inside Docker):
    python scripts/train_loss_comparison.py

Dual-node splitting:
    CONFIG_SLICE=0:10  python scripts/train_loss_comparison.py
    CONFIG_SLICE=10:20 python scripts/train_loss_comparison.py

Output: data/poc_results/loss_comparison_results.json
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
EPOCHS = 30
PATIENCE = 7
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
    {"id": "방어_선_kg_중_dom", "species": "방어", "state": "선", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": True},
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
    # Premium 활어 grades
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

TARGET_MODELS = ["GRU", "Transformer", "CNN-LSTM"]
LOSS_NAMES = ["MSE", "MAE", "MAPE", "sMAPE", "Huber", "LogCosh"]


# ── Loss Functions ─────────────────────────────────────────────────


class MAPELoss(nn.Module):
    """Direct MAPE loss -- aligns training with evaluation metric."""

    def forward(self, pred, actual):
        return torch.mean(
            torch.abs(pred - actual) / torch.clamp(torch.abs(actual), min=1e-8)
        )


class SmoothedMAPELoss(nn.Module):
    """Symmetric MAPE -- no division-by-zero, handles zero actuals."""

    def forward(self, pred, actual):
        return torch.mean(
            2 * torch.abs(pred - actual)
            / (torch.abs(actual) + torch.abs(pred) + 1e-8)
        )


class HuberLoss(nn.Module):
    """Huber loss -- MSE for small errors, MAE for large. Robust to outliers."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, actual):
        diff = torch.abs(pred - actual)
        return torch.mean(
            torch.where(
                diff <= self.delta,
                0.5 * diff ** 2,
                self.delta * (diff - 0.5 * self.delta),
            )
        )


class LogCoshLoss(nn.Module):
    """Log-cosh loss -- smoother than Huber, twice differentiable."""

    def forward(self, pred, actual):
        diff = pred - actual
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))


def get_loss_fn(loss_name: str) -> nn.Module:
    """Instantiate a loss function by name."""
    if loss_name == "MSE":
        return nn.MSELoss()
    elif loss_name == "MAE":
        return nn.L1Loss()
    elif loss_name == "MAPE":
        return MAPELoss()
    elif loss_name == "sMAPE":
        return SmoothedMAPELoss()
    elif loss_name == "Huber":
        return HuberLoss(delta=1.0)
    elif loss_name == "LogCosh":
        return LogCoshLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


# ── Helper utilities (copied from train_all_dl_models.py) ──────────


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


# ── v10 Preprocessing Helpers ─────────────────────────────────────


def winsorized_daily_price(day_prices, recent_30d_prices):
    """Clip extreme lots to p10/p90 of recent 30-day distribution."""
    if len(recent_30d_prices) < 10:
        return float(np.mean(day_prices))
    p10, p90 = np.percentile(recent_30d_prices, [10, 90])
    clipped = [max(p10, min(p90, p)) for p in day_prices]
    return float(np.mean(clipped))


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


# ── Data Loading ──────────────────────────────────────────────────


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
      Fix 1: Winsorized Mean -- clip extreme lots to p10/p90 of rolling 30-day window
      Fix 4: Origin-Weighted Aggregation -- weight lots by origin trading frequency

    Returns a dict with gap-filled continuous daily arrays and corresponding date strings.
    Forward-fills non-trading days (price/high/low; lots/origins/qty become 0).
    """
    n = len(data["trade_date"])
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

    trading_records = {}
    rolling_lot_prices = []
    origin_days_seen = defaultdict(list)

    for day_i, d in enumerate(sorted_dates):
        lots = day_lots[d]
        day_prices = [lp[0] for lp in lots]
        day_origins_list = [lp[2] for lp in lots]

        # Build rolling 30-day origin frequency
        origin_freq_30d = defaultdict(int)
        for origin_d, origin_day_idxs in origin_days_seen.items():
            count = sum(1 for di in origin_day_idxs if day_i - 30 <= di < day_i)
            if count > 0:
                origin_freq_30d[origin_d] = count
        max_freq_30d = max(origin_freq_30d.values()) if origin_freq_30d else 1

        # Fix 4: per-lot weights by origin frequency
        lot_weights = [
            origin_weight(orig, origin_freq_30d, max_freq_30d)
            for orig in day_origins_list
        ]

        # Fix 1: winsorize using rolling 30d window, then weighted mean
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

        rolling_lot_prices.extend(day_prices)
        if day_i >= 30:
            rolling_lot_prices = [
                p
                for d2 in sorted_dates[max(0, day_i - 29):day_i + 1]
                for p in [lp[0] for lp in day_lots[d2]]
            ]

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

    filled_prices, filled_highs, filled_lows = [], [], []
    filled_lots, filled_origins, filled_qtys, filled_dates = [], [], [], []
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
    Returns (features_array of shape (n, 68), min_offset).
    min_offset = 90 (rows needing full 90-day history).
    """
    prices = series["prices"]
    highs = series["highs"]
    lows = series["lows"]
    lots = series["lots"]
    origins_arr = series["origins"]
    qtys = series["qtys"]
    dates = series["dates"]
    n = len(prices)

    di_map = ctx["date_idx"]

    ema7 = ema(prices, 7)
    ema30 = ema(prices, 30)
    macd_line, macd_sig = macd_signal(prices)
    rsi_14 = rsi(prices, 14)

    monthly_avg = defaultdict(list)
    for i in range(n):
        monthly_avg[parse_date(dates[i]).month].append(prices[i])
    monthly_avg = {m: np.mean(v) for m, v in monthly_avg.items()}

    min_offset = 90
    feat_rows = []
    for i in range(n):
        dt = parse_date(dates[i])
        dow = dt.weekday()
        doy = dt.timetuple().tm_yday

        di = di_map.get(dates[i], 0)
        dt_prev = parse_date(dates[i - 1]) if i > 0 else dt
        hol = days_to_holiday(dt)

        p = prices[i]
        p1 = prices[i - 1] if i >= 1 else p
        p7 = prices[i - 7] if i >= 7 else p1
        p30 = prices[i - 30] if i >= 30 else p1
        a7 = np.mean(prices[max(0, i - 7):i]) if i >= 1 else p
        a30 = np.mean(prices[max(0, i - 30):i]) if i >= 1 else p

        s7 = np.std(prices[max(0, i - 7):i]) if i >= 1 else 0.0
        s30 = np.std(prices[max(0, i - 30):i]) if i >= 1 else 0.0
        r7 = float(max(prices[max(0, i - 7):i]) - min(prices[max(0, i - 7):i])) if i >= 1 else 0.0

        oq7 = ctx["sp_qty_7d"][target_sp][di]
        ol7 = ctx["sp_lots_7d"][target_sp][di]
        oq30 = np.mean(ctx["sp_qty"][target_sp][max(0, di - 30):di]) if di >= 1 else oq7
        oqr = oq7 / oq30 if oq30 > 0 else 1.0
        oqc = ((ctx["sp_qty_7d"][target_sp][di] - ctx["sp_qty_7d"][target_sp][max(0, di - 7)])
               / max(ctx["sp_qty_7d"][target_sp][max(0, di - 7)], 1))
        olc = ((ctx["sp_lots_7d"][target_sp][di] - ctx["sp_lots_7d"][target_sp][max(0, di - 7)])
               / max(ctx["sp_lots_7d"][target_sp][max(0, di - 7)], 1))

        otq = ctx["total_sashimi_7d"][di] - ctx["sp_qty_7d"][target_sp][di]
        ml7 = ctx["market_lots_7d"][di]
        con = (ctx["sp_qty"][target_sp][di] / ctx["total_sashimi"][di]
               if ctx["total_sashimi"][di] > 0 else 0)
        tsc = ((ctx["total_sashimi_7d"][di] - ctx["total_sashimi_7d"][max(0, di - 7)])
               / max(ctx["total_sashimi_7d"][max(0, di - 7)], 1))
        mc = ((ctx["market_lots_7d"][di] - ctx["market_lots_7d"][max(0, di - 7)])
              / max(ctx["market_lots_7d"][max(0, di - 7)], 1))

        pvm = p / monthly_avg.get(dt.month, p) if monthly_avg.get(dt.month, p) > 0 else 1.0

        gap_d = (dt - dt_prev).days
        ld = int(ol7 < ctx["sp_lots_7d"][target_sp][max(0, di - 14)] * 0.5) if di >= 14 else 0
        qd = int(oq7 < oq30 * 0.5) if oq30 > 0 else 0

        boll_upper = a30 + 2 * s30
        boll_lower = a30 - 2 * s30
        boll_pct = ((p - boll_lower) / (boll_upper - boll_lower)
                    if (boll_upper - boll_lower) > 0 else 0.5)
        mom_14 = ((p - prices[i - 14]) / prices[i - 14] * 100
                  if i >= 14 and prices[i - 14] > 0 else 0)

        f_sin_365 = np.sin(2 * np.pi * doy / 365)
        f_cos_365 = np.cos(2 * np.pi * doy / 365)
        f_sin_182 = np.sin(2 * np.pi * doy / 182.5)
        f_cos_182 = np.cos(2 * np.pi * doy / 182.5)
        f_sin_7 = np.sin(2 * np.pi * dow / 7)
        f_cos_7 = np.cos(2 * np.pi * dow / 7)

        is_friday = int(dow == 4)
        next_gap = (parse_date(dates[i + 1]) - dt).days if i + 1 < n else 1
        is_pre_hol = int(next_gap > 2)
        consec_gap = gap_d
        week_pos = dow / 4 if dow <= 4 else 1.0
        days_left = max(0, 4 - dow)

        window_30 = prices[max(0, i - 30):i] if i >= 1 else np.array([p])
        window_90 = prices[max(0, i - 90):i] if i >= 1 else np.array([p])
        skew_30 = float(scipy_stats.skew(window_30)) if len(window_30) >= 3 else 0.0
        kurt_30 = float(scipy_stats.kurtosis(window_30)) if len(window_30) >= 3 else 0.0
        pct_90 = (float(scipy_stats.percentileofscore(window_90, p)) / 100
                  if len(window_90) >= 3 else 0.5)
        zscore_30 = (p - a30) / s30 if s30 > 0 else 0.0

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
            dow, dt.month, dt.day, int(dow >= 5), dt.isocalendar()[1],
            (dt.month - 1) // 3 + 1, int(dow == 0),
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
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result, min_offset


def normalize_features(train_features: np.ndarray, test_features: np.ndarray):
    """Per-feature z-score normalization. Returns normalized arrays and stats."""
    train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
    test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std < 1e-8] = 1.0
    train_norm = np.clip((train_features - mean) / std, -10, 10)
    test_norm = np.clip((test_features - mean) / std, -10, 10)
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


# ── Model Architectures ───────────────────────────────────────────


class GRUModel(nn.Module):
    """2-layer GRU encoder with linear decoder."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, horizon: int = HORIZON, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SimpleTransformer(nn.Module):
    """Transformer encoder, 2 layers, 4 heads, d_model=64."""

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
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        return self.fc(h[:, -1, :])


class CNNLSTMModel(nn.Module):
    """Conv1d(32 filters, kernel=3) + 2-layer LSTM(64)."""

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
        c = self.conv(x.permute(0, 2, 1))
        c = c.permute(0, 2, 1)
        out, _ = self.lstm(c)
        return self.fc(out[:, -1, :])


def create_model(name: str, input_size: int) -> nn.Module:
    """Instantiate a model by name."""
    if name == "GRU":
        return GRUModel(input_size)
    elif name == "Transformer":
        return SimpleTransformer(input_size)
    elif name == "CNN-LSTM":
        return CNNLSTMModel(input_size)
    else:
        raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Training Loop ─────────────────────────────────────────────────


def train_with_loss_fn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    epochs: int = EPOCHS,
    lr: float = LR,
    device: str = "cuda",
    patience: int = PATIENCE,
) -> dict:
    """
    Train a model with a specific loss function (early stopping patience=7).
    Returns dict with best_epoch and best_val_loss.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * x.size(0)
            n_train += x.size(0)

        # Validation
        model.train(False)
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss_sum += loss.item() * x.size(0)
                n_val += x.size(0)

        avg_val = val_loss_sum / max(n_val, 1)
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


def compute_test_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    price_mean: float,
    price_std: float,
) -> dict:
    """
    Run model on test set; compute MAPE and RMSE on raw (exp-scale) prices.
    Denormalization: raw_price = exp(pred * price_std + price_mean)
    Fix 2 reversal: undo z-score then undo log-transform.
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
        return {"mape": 999.0, "rmse": 999.0, "n_samples": 0}

    preds = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)

    # Denormalize: undo z-score then exp (Fix 2 reversal)
    preds_denorm = np.exp(preds * price_std + price_mean)
    actuals_denorm = np.exp(actuals * price_std + price_mean)

    valid = actuals_denorm > 0
    if valid.any():
        mape = float(np.mean(
            np.abs(preds_denorm[valid] - actuals_denorm[valid]) / actuals_denorm[valid]
        )) * 100
    else:
        mape = 999.0

    rmse = float(np.sqrt(np.mean((preds_denorm - actuals_denorm) ** 2)))

    return {
        "mape": round(mape, 2),
        "rmse": round(rmse, 0),
        "n_samples": len(preds),
    }


# ── Main Pipeline ─────────────────────────────────────────────────


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  Loss Function Comparison")
    print("  6 losses x 3 models x 20 species configs")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device:  {device}")
    if device == "cuda":
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Memory:  {props.total_memory / 1e9:.1f} GB")
    print(f"Lookback={LOOKBACK}  Horizon={HORIZON}  Batch={BATCH_SIZE}")
    print(f"Epochs={EPOCHS}  Patience={PATIENCE}  LR={LR}")
    print(f"Loss functions: {LOSS_NAMES}")
    print(f"Models:         {TARGET_MODELS}")
    print()

    # Load data once
    data = load_parquet_data()
    n_rows = len(data["trade_date"])

    print("Building supply context...", end=" ", flush=True)
    ctx = build_supply_context(data, n_rows)
    print(f"{len(ctx['dates'])} trading days.")

    # CONFIG_SLICE support for dual-node splitting
    config_slice = os.environ.get("CONFIG_SLICE", None)
    if config_slice:
        start_idx, end_idx = map(int, config_slice.split(":"))
        configs_to_run = SPECIES_CONFIGS[start_idx:end_idx]
        print(f"\nCONFIG_SLICE={config_slice}: "
              f"{len(configs_to_run)}/{len(SPECIES_CONFIGS)} configs")
    else:
        configs_to_run = SPECIES_CONFIGS

    total_runs = len(configs_to_run) * len(TARGET_MODELS) * len(LOSS_NAMES)
    print(f"Total training runs: {total_runs} "
          f"({len(configs_to_run)} configs x {len(TARGET_MODELS)} models x {len(LOSS_NAMES)} losses)")
    print()

    # Results: {species_id: {model_name: {loss_name: {mape, rmse, n_samples, elapsed}}}}
    results = {}
    run_count = 0
    t_start_total = time.time()

    for cfg in configs_to_run:
        sp = cfg.get("id", cfg["species"])
        species_name = cfg["species"]
        use_smoothed = cfg.get("smoothed", False)

        print(f"\n{'=' * 70}")
        print(f"  Species: {sp}  (state={cfg['state']}, spec={cfg['spec']})")
        print(f"{'=' * 70}")

        # Build daily series (Fix 1 + Fix 4 inside)
        series = build_species_daily_series(data, cfg)
        prices = series["prices"]
        dates = series["dates"]

        if len(prices) < MIN_DAYS:
            print(f"  SKIP: insufficient data ({len(prices)} days < {MIN_DAYS})")
            continue

        # Build 68 features
        features, min_offset = build_features_68(series, ctx, species_name)
        features = features[min_offset:]
        prices = prices[min_offset:]
        dates = dates[min_offset:]
        n_features = features.shape[1]
        print(f"  Data: {len(prices)} days (after {min_offset}-day warmup), {n_features} features")

        # Fix 3: flag outlier days
        outlier_mask = flag_outlier_days(prices)
        n_outliers = int(outlier_mask.sum())
        print(f"  Outliers flagged: {n_outliers}")

        # Fix 2: log-transform target
        log_prices = np.log(np.maximum(prices, 1.0))

        # Smoothed target: 7-day MA on log-prices (for species with smoothed=True)
        if use_smoothed and len(log_prices) > 7:
            target_log_prices = np.convolve(log_prices, np.ones(7) / 7, mode="same")
        else:
            target_log_prices = log_prices

        # 80/20 train/test split
        split_idx = int(len(features) * 0.8)

        # Fix 3: remove outlier days from training only
        train_outlier_mask = outlier_mask[:split_idx]
        train_clean_mask = ~train_outlier_mask
        n_train_outliers = int(train_outlier_mask.sum())
        if n_train_outliers > 0:
            print(f"  Removing {n_train_outliers} outlier days from training")

        train_feat = features[:split_idx][train_clean_mask]
        test_feat = features[split_idx:]
        train_log = target_log_prices[:split_idx][train_clean_mask]
        test_log = log_prices[split_idx:]  # always raw log for test MAPE

        # Normalize features
        train_norm, test_norm, _, _ = normalize_features(train_feat, test_feat)

        # Z-score normalize log-prices (Fix 2)
        log_price_mean = float(np.mean(train_log))
        log_price_std = float(np.std(train_log))
        if log_price_std < 1e-8:
            log_price_std = 1.0

        train_log_norm = (train_log - log_price_mean) / log_price_std
        test_log_norm = (test_log - log_price_mean) / log_price_std

        # Build datasets
        train_ds = SlidingWindowDataset(train_norm, train_log_norm)
        test_ds = SlidingWindowDataset(test_norm, test_log_norm)

        if len(train_ds) < 50 or len(test_ds) < 10:
            print(f"  SKIP: insufficient samples (train={len(train_ds)}, test={len(test_ds)})")
            continue

        # Validation split: last 10% of training set
        val_split = int(len(train_ds) * 0.9)
        train_subset = torch.utils.data.Subset(train_ds, range(val_split))
        val_subset = torch.utils.data.Subset(train_ds, range(val_split, len(train_ds)))
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        print(f"  Samples: train={len(train_subset)}, val={len(val_subset)}, test={len(test_ds)}")

        sp_results = {}

        for model_name in TARGET_MODELS:
            sp_results[model_name] = {}
            for loss_name in LOSS_NAMES:
                run_count += 1
                label = f"[{run_count}/{total_runs}] {sp} / {model_name} / {loss_name}"
                print(f"\n  {label}", end=" ", flush=True)

                t0 = time.time()
                try:
                    model = create_model(model_name, n_features)
                    n_params = count_parameters(model)
                    print(f"({n_params:,}p)", end=" ", flush=True)

                    loss_fn = get_loss_fn(loss_name)

                    train_info = train_with_loss_fn(
                        model, train_loader, val_loader, loss_fn,
                        epochs=EPOCHS, lr=LR, device=device, patience=PATIENCE,
                    )

                    metrics = compute_test_metrics(
                        model, test_loader, device, log_price_mean, log_price_std,
                    )
                    elapsed = time.time() - t0

                    metrics["elapsed_s"] = round(elapsed, 1)
                    metrics["best_epoch"] = train_info["best_epoch"]
                    sp_results[model_name][loss_name] = metrics

                    print(f"ep={train_info['best_epoch']} "
                          f"MAPE={metrics['mape']:.1f}% "
                          f"RMSE={metrics['rmse']:.0f} "
                          f"[{elapsed:.1f}s]")

                except Exception as exc:
                    elapsed = time.time() - t0
                    print(f"FAILED: {exc} [{elapsed:.1f}s]")
                    sp_results[model_name][loss_name] = {
                        "mape": 999.0, "rmse": 999.0, "n_samples": 0,
                        "elapsed_s": round(elapsed, 1), "best_epoch": 0,
                        "error": str(exc),
                    }

                if device == "cuda":
                    torch.cuda.empty_cache()

        results[sp] = sp_results

    total_elapsed = time.time() - t_start_total

    # ── Comparison Tables ─────────────────────────────────────────

    species_done = list(results.keys())

    print("\n\n" + "=" * 90)
    print("  MAPE COMPARISON TABLE -- by Loss Function (avg across species)")
    print("=" * 90)

    # Per-model table: loss x species
    for model_name in TARGET_MODELS:
        print(f"\n  Model: {model_name}")
        header = f"    {'Loss':<10}"
        for sp in species_done:
            short = sp[:7]
            header += f" {short:>8}"
        header += f"  {'AVG':>7}"
        print(header)
        print("    " + "-" * (10 + 9 * len(species_done) + 10))

        loss_avgs = {}
        for loss_name in LOSS_NAMES:
            row = f"    {loss_name:<10}"
            mapes = []
            for sp in species_done:
                val = results[sp].get(model_name, {}).get(loss_name, {})
                mape = val.get("mape", 999.0)
                if mape < 900:
                    row += f" {mape:>7.1f}%"
                    mapes.append(mape)
                else:
                    row += f"     N/A "
            avg = np.mean(mapes) if mapes else 999.0
            loss_avgs[loss_name] = avg
            row += f"  {avg:>6.1f}%"
            print(row)

        best_loss = min(loss_avgs, key=lambda k: loss_avgs[k])
        print(f"    --> Best loss for {model_name}: {best_loss} "
              f"(avg MAPE={loss_avgs[best_loss]:.2f}%)")

    # Win count: which loss function wins most often
    print("\n\n" + "=" * 70)
    print("  LOSS FUNCTION WIN COUNT -- how often each loss is best (by MAPE)")
    print("=" * 70)

    win_counts = defaultdict(int)
    total_contests = 0

    for sp in species_done:
        for model_name in TARGET_MODELS:
            model_loss_results = results[sp].get(model_name, {})
            valid = {
                ln: model_loss_results[ln]["mape"]
                for ln in LOSS_NAMES
                if ln in model_loss_results and model_loss_results[ln]["mape"] < 900
            }
            if valid:
                best = min(valid, key=valid.get)
                win_counts[best] += 1
                total_contests += 1

    for loss_name in sorted(win_counts, key=lambda k: -win_counts[k]):
        pct = win_counts[loss_name] / total_contests * 100 if total_contests > 0 else 0
        print(f"  {loss_name:<10}  wins={win_counts[loss_name]:3d}  ({pct:.1f}%)")

    # Overall average MAPE per loss
    print("\n\n" + "=" * 70)
    print("  AVERAGE MAPE PER LOSS FUNCTION (all models x all species)")
    print("=" * 70)

    loss_overall_mapes = defaultdict(list)
    for sp in species_done:
        for model_name in TARGET_MODELS:
            for loss_name in LOSS_NAMES:
                val = results[sp].get(model_name, {}).get(loss_name, {})
                mape = val.get("mape", 999.0)
                if mape < 900:
                    loss_overall_mapes[loss_name].append(mape)

    ranked = sorted(
        [(ln, np.mean(v)) for ln, v in loss_overall_mapes.items() if v],
        key=lambda x: x[1],
    )
    for rank, (loss_name, avg_mape) in enumerate(ranked, 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"  {rank}. {loss_name:<10}  avg MAPE={avg_mape:.2f}%{marker}")

    print(f"\nTotal elapsed: {total_elapsed / 60:.1f} min")

    # ── Save Results ──────────────────────────────────────────────

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "loss_comparison_results.json"

    summary = {
        "meta": {
            "script": "train_loss_comparison.py",
            "timestamp": datetime.now().isoformat(),
            "device": device,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "loss_names": LOSS_NAMES,
            "models": TARGET_MODELS,
            "config_slice": config_slice,
            "total_elapsed_min": round(total_elapsed / 60, 1),
        },
        "results": results,
        "win_counts": dict(win_counts),
        "avg_mape_per_loss": {
            ln: round(float(np.mean(v)), 2)
            for ln, v in loss_overall_mapes.items()
            if v
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {out_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
