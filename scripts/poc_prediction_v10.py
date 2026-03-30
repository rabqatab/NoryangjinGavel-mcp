"""
PoC v10: 5 Advanced Preprocessing Fixes on top of v6 (68 features).

New over v6:
  Fix 1: Winsorized Mean   — clip daily lot prices to p10/p90 of rolling 30-day window
  Fix 2: Log-Transform Target — predict log(price), exp() for final MAPE
  Fix 3: Outlier Day Removal  — remove days >3σ from rolling 30d mean (training only)
  Fix 4: Origin-Weighted Aggregation — weight lots by origin trading frequency
  Fix 5: Adaptive VMD K   — K=5 for high-volatility, K=3 for low

Usage:
    uv run python scripts/poc_prediction_v10.py
"""
import json
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats as scipy_stats
from vmdpy import VMD

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "parquet" / "prices"
OUTPUT_DIR = PROJECT_ROOT / "data" / "poc_results"

FOREIGN_KW = ['일본','중국','미국','러시아','캐나다','노르웨이','뉴질랜드','대만','칠레',
              '아르헨티나','영국','아일랜드','온두라스','북한','(원양)','인도','인도네시아',
              '태국','베트남','필리핀','호주','스페인','네덜란드','페루','모로코','아프리카',
              '파키스탄','라스팔마스','포클랜드','멕시코']

SASHIMI_SPECIES = ["넙치", "우럭", "방어", "참돔", "농어", "도다리", "감성돔"]

SPECIES_CONFIGS = [
    {"species": "넙치", "state": "활", "pkg": "kg", "spec": "중", "domestic": False,
     "smoothed": False, "label": "넙치", "method": "vmd"},
    {"species": "우럭", "state": "활", "pkg": "kg", "spec": "중", "domestic": False,
     "smoothed": False, "label": "우럭", "method": "ensemble"},
    {"species": "방어", "state": "선", "pkg": "kg", "spec": "중", "domestic": True,
     "smoothed": True, "label": "방어", "method": "vmd", "regime_split": True},
    {"species": "참돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True,
     "smoothed": False, "label": "참돔", "method": "vmd"},
    {"species": "농어", "state": "활", "pkg": "kg", "spec": "중", "domestic": True,
     "smoothed": False, "label": "농어", "method": "vmd"},
    {"species": "도다리", "state": "활", "pkg": "kg", "spec": "중", "domestic": False,
     "smoothed": True, "label": "도다리", "method": "vmd"},
    {"species": "감성돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True,
     "smoothed": False, "label": "감성돔", "method": "vmd"},
]


def is_foreign(o):
    if not o: return False
    return any(kw in o for kw in FOREIGN_KW)


def parse_date(d):
    return datetime.strptime(d, "%Y.%m.%d")


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


def days_to_holiday(dt):
    r = {"seollal": 999, "chuseok": 999}
    for y in [dt.year - 1, dt.year, dt.year + 1]:
        if y not in KOREAN_HOLIDAYS: continue
        for name, hd in KOREAN_HOLIDAYS[y].items():
            diff = (parse_date(hd) - dt).days
            if abs(diff) < abs(r[name]): r[name] = diff
    return r


# ── Technical Indicators ────────────────────────────────────────────

def ema(prices, span):
    """Exponential moving average."""
    a = np.array(prices, dtype=float)
    out = np.empty_like(a)
    out[0] = a[0]
    alpha = 2 / (span + 1)
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i-1]
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


# ── Fix 1: Winsorized Mean ───────────────────────────────────────────

def winsorized_daily_price(day_prices, recent_30d_prices):
    """Clip extreme lots to p10/p90 of recent 30-day distribution."""
    if len(recent_30d_prices) < 10:
        return float(np.mean(day_prices))
    p10, p90 = np.percentile(recent_30d_prices, [10, 90])
    clipped = [max(p10, min(p90, p)) for p in day_prices]
    return float(np.mean(clipped))


# ── Fix 4: Origin-Weighted Aggregation ──────────────────────────────

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


# ── Fix 5: Adaptive VMD K ───────────────────────────────────────────

def adaptive_vmd_k(prices, window=90):
    """Use K=5 for high-volatility periods, K=3 for low."""
    a = np.array(prices, dtype=float)
    if len(a) < window:
        return 3
    recent_std = np.std(a[-window:])
    overall_std = np.std(a)
    return 5 if recent_std > overall_std else 3


# ── Data Loading ────────────────────────────────────────────────────

def load_all():
    import pyarrow.dataset as ds
    print("Loading data...", end=" ", flush=True)
    dataset = ds.dataset(str(DATA_ROOT), format="parquet", partitioning="hive")
    cols = ["trade_date", "species", "state", "origin", "spec", "packaging",
            "price_avg", "price_high", "price_low", "quantity"]
    table = dataset.to_table(columns=cols)
    data = {col: table.column(col).to_pylist() for col in cols}
    print(f"{len(data['trade_date']):,} rows.")
    return data


def build_supply_context(data, n):
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
        "sp_qty_7d": {s: np.convolve(q, np.ones(k)/k, mode="same") for s, q in sp_qty.items()},
        "sp_lots_7d": {s: np.convolve(l, np.ones(k)/k, mode="same") for s, l in sp_lots.items()},
        "market_lots": market_lots,
        "market_lots_7d": np.convolve(market_lots, np.ones(k)/k, mode="same"),
        "total_sashimi": sum(sp_qty.values()),
        "total_sashimi_7d": np.convolve(sum(sp_qty.values()), np.ones(k)/k, mode="same"),
    }


def extract_records_v10(data, n, cfg):
    """
    Extended extract: collects per-lot (price, qty, origin) tuples per day,
    then applies Fix 1 (winsorized mean) + Fix 4 (origin-weighted) to build
    the daily price series.
    """
    # Collect raw lots per day
    day_lots = defaultdict(list)   # date -> list of (price, qty, origin)
    day_highs = defaultdict(list)
    day_lows = defaultdict(list)
    day_origins = defaultdict(set)

    for i in range(n):
        if data["species"][i] != cfg["species"]: continue
        if data["state"][i] != cfg["state"]: continue
        if data["packaging"][i] != cfg["pkg"]: continue
        if data["spec"][i] != cfg["spec"]: continue
        if cfg["domestic"] and is_foreign(data["origin"][i]): continue
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

    # Build rolling 30-day lot-price buffer and origin-frequency counter
    records = []
    rolling_lot_prices = []  # flat list of all lot prices in last 30 days
    # origin_days_seen[origin] = list of date indices where that origin appeared
    origin_days_seen = defaultdict(list)  # origin -> list of day-indices

    for day_i, d in enumerate(sorted_dates):
        lots = day_lots[d]
        day_prices = [lp[0] for lp in lots]
        day_origins_list = [lp[2] for lp in lots]

        # --- Build rolling 30-day origin frequency ---
        # origins that appear in any day in [day_i-30, day_i)
        origin_freq_30d = defaultdict(int)
        for origin_d, origin_day_idxs in origin_days_seen.items():
            # count how many days in last 30 this origin appeared
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

        records.append({
            "date": d,
            "price": daily_price,
            "high": max(day_highs[d]),
            "low": min(day_lows[d]),
            "n_lots": len(lots),
            "n_origins": len(day_origins[d]),
            "qty": sum(lp[1] for lp in lots),
        })

        # Update rolling buffer (keep ~30 days of lot prices)
        rolling_lot_prices.extend(day_prices)
        # Prune: keep only last 30 days' lots (approximate by day count)
        if day_i >= 30:
            # Remove oldest day's prices from buffer by rebuilding
            # More efficient: just keep last 30 days' raw lots
            cutoff_date = sorted_dates[max(0, day_i - 29)]
            rolling_lot_prices = [
                p
                for di2, d2 in enumerate(sorted_dates[max(0, day_i - 29):day_i + 1])
                for p in [lp[0] for lp in day_lots[d2]]
            ]

        # Update origin days-seen index
        for orig in day_origins[d]:
            origin_days_seen[orig].append(day_i)

    return records


# ── Fix 3: Outlier Day Detection ─────────────────────────────────────

def flag_outlier_days(records, window=30, n_sigma=3):
    """
    Return a boolean array (True = outlier) using rolling mean/std.
    Outliers are flagged but only excluded from training.
    """
    prices = np.array([r["price"] for r in records])
    is_outlier = np.zeros(len(prices), dtype=bool)
    for i in range(window, len(prices)):
        window_prices = prices[max(0, i - window):i]
        mu = np.mean(window_prices)
        sigma = np.std(window_prices)
        if sigma > 0 and abs(prices[i] - mu) > n_sigma * sigma:
            is_outlier[i] = True
    return is_outlier


# ── Feature Engineering (v10: same 68 features, built on winsorized prices) ──

def build_features_v10(records, ctx, target_sp, offset=7, use_smoothed=False,
                       outlier_mask=None):
    """
    Same 68 features as v6. Features are computed on winsorized daily prices.
    Target is log(price) (Fix 2). Outlier days excluded from training (Fix 3).
    outlier_mask: boolean array len==len(records), True=outlier (skip in train).
    """
    prices = np.array([r["price"] for r in records])
    highs = np.array([r["high"] for r in records])
    lows = np.array([r["low"] for r in records])
    lots = np.array([r["n_lots"] for r in records])
    origins = np.array([r["n_origins"] for r in records])
    qtys = np.array([r["qty"] for r in records])
    dates = [r["date"] for r in records]
    di_map = ctx["date_idx"]

    if outlier_mask is None:
        outlier_mask = np.zeros(len(prices), dtype=bool)

    # Fix 2: log-transform target
    log_prices = np.log(np.maximum(prices, 1.0))

    targets = np.convolve(log_prices, np.ones(7)/7, mode="same") if use_smoothed and len(log_prices) > 7 else log_prices

    # Pre-compute technical indicators on WINSORIZED prices (not log)
    ema7 = ema(prices, 7)
    ema30 = ema(prices, 30)
    macd_line, macd_sig = macd_signal(prices)
    rsi_14 = rsi(prices, 14)

    monthly_avg = defaultdict(list)
    for r in records:
        monthly_avg[parse_date(r["date"]).month].append(r["price"])
    monthly_avg = {m: np.mean(v) for m, v in monthly_avg.items()}

    fnames = [
        # Calendar (7)
        "dow", "month", "dom", "is_weekend", "woy", "quarter", "is_monday",
        # Holiday (4)
        "days_seollal", "days_chuseok", "abs_seollal", "abs_chuseok",
        # Price History (5)
        "price_lag1", "price_lag7", "price_lag30", "price_7d", "price_30d",
        # Momentum (4)
        "pchg_1d", "pchg_7d", "pchg_30d", "pchg_7v30",
        # Volatility (3)
        "std_7d", "std_30d", "range_7d",
        # Own Supply (5)
        "own_q7", "own_l7", "own_q_ratio", "own_q_chg", "own_l_chg",
        # Cross Supply (5)
        "other_q7", "mkt_l7", "concentration", "sashimi_chg", "mkt_chg",
        # Seasonal (4)
        "price_vs_month", "month_sin", "month_cos", "is_peak",
        # Weather Proxy (4)
        "gap", "lots_drop", "qty_drop", "shock",
        # Technical Indicators (8)
        "ema_7", "ema_30", "macd", "macd_signal", "macd_hist",
        "bollinger_pct", "rsi_14", "momentum_14d",
        # Fourier (6)
        "fourier_sin_365", "fourier_cos_365", "fourier_sin_182", "fourier_cos_182",
        "fourier_sin_7", "fourier_cos_7",
        # Advanced Calendar (5)
        "is_friday", "is_pre_holiday", "consecutive_gap", "week_position", "days_left_in_week",
        # Price Distribution (4)
        "skewness_30d", "kurtosis_30d", "percentile_90d", "zscore_30d",
        # Advanced Supply (4)
        "own_q_yoy_ratio", "origin_diversity_7d", "avg_lot_size_7d", "hl_spread_7d",
    ]

    X, y, od, is_outlier_sample = [], [], [], []
    for i in range(90, len(records) - offset):
        dt = parse_date(dates[i])
        di = di_map.get(dates[i], 0)
        dt_prev = parse_date(dates[i-1]) if i > 0 else dt
        hol = days_to_holiday(dt)
        dow = dt.weekday()
        doy = dt.timetuple().tm_yday

        p = prices[i]
        p1 = prices[i-1] if i >= 1 else p
        p7 = prices[i-7] if i >= 7 else p1
        p30 = prices[i-30] if i >= 30 else p1
        a7 = np.mean(prices[max(0,i-7):i])
        a30 = np.mean(prices[max(0,i-30):i])
        s7 = np.std(prices[max(0,i-7):i])
        s30 = np.std(prices[max(0,i-30):i])
        r7 = float(max(prices[max(0,i-7):i]) - min(prices[max(0,i-7):i]))

        # Supply (same as v6)
        oq7 = ctx["sp_qty_7d"][target_sp][di]
        ol7 = ctx["sp_lots_7d"][target_sp][di]
        oq30 = np.mean(ctx["sp_qty"][target_sp][max(0,di-30):di]) if di >= 1 else oq7
        oqr = oq7 / oq30 if oq30 > 0 else 1
        oqc = (ctx["sp_qty_7d"][target_sp][di] - ctx["sp_qty_7d"][target_sp][max(0,di-7)]) / max(ctx["sp_qty_7d"][target_sp][max(0,di-7)], 1)
        olc = (ctx["sp_lots_7d"][target_sp][di] - ctx["sp_lots_7d"][target_sp][max(0,di-7)]) / max(ctx["sp_lots_7d"][target_sp][max(0,di-7)], 1)
        otq = ctx["total_sashimi_7d"][di] - ctx["sp_qty_7d"][target_sp][di]
        ml7 = ctx["market_lots_7d"][di]
        con = ctx["sp_qty"][target_sp][di] / ctx["total_sashimi"][di] if ctx["total_sashimi"][di] > 0 else 0
        tsc = (ctx["total_sashimi_7d"][di] - ctx["total_sashimi_7d"][max(0,di-7)]) / max(ctx["total_sashimi_7d"][max(0,di-7)], 1)
        mc = (ctx["market_lots_7d"][di] - ctx["market_lots_7d"][max(0,di-7)]) / max(ctx["market_lots_7d"][max(0,di-7)], 1)
        pvm = p / monthly_avg.get(dt.month, p) if monthly_avg.get(dt.month, p) > 0 else 1
        gap_d = (dt - dt_prev).days
        ld = int(ol7 < ctx["sp_lots_7d"][target_sp][max(0,di-14)] * 0.5) if di >= 14 else 0
        qd = int(oq7 < oq30 * 0.5) if oq30 > 0 else 0

        # --- Technical Indicators ---
        boll_upper = a30 + 2 * s30
        boll_lower = a30 - 2 * s30
        boll_pct = (p - boll_lower) / (boll_upper - boll_lower) if (boll_upper - boll_lower) > 0 else 0.5
        mom_14 = (p - prices[i-14]) / prices[i-14] * 100 if i >= 14 and prices[i-14] > 0 else 0

        # --- Fourier ---
        f_sin_365 = np.sin(2 * np.pi * doy / 365)
        f_cos_365 = np.cos(2 * np.pi * doy / 365)
        f_sin_182 = np.sin(2 * np.pi * doy / 182.5)
        f_cos_182 = np.cos(2 * np.pi * doy / 182.5)
        f_sin_7 = np.sin(2 * np.pi * dow / 7)
        f_cos_7 = np.cos(2 * np.pi * dow / 7)

        # --- Advanced Calendar ---
        is_friday = int(dow == 4)
        next_gap = (parse_date(dates[i+1]) - dt).days if i + 1 < len(dates) else 1
        is_pre_hol = int(next_gap > 2)
        consec_gap = gap_d
        week_pos = dow / 4 if dow <= 4 else 1.0
        days_left = max(0, 4 - dow)

        # --- Price Distribution ---
        window_30 = prices[max(0,i-30):i]
        window_90 = prices[max(0,i-90):i]
        skew_30 = float(scipy_stats.skew(window_30)) if len(window_30) >= 3 else 0
        kurt_30 = float(scipy_stats.kurtosis(window_30)) if len(window_30) >= 3 else 0
        pct_90 = float(scipy_stats.percentileofscore(window_90, p)) / 100 if len(window_90) >= 3 else 0.5
        zscore_30 = (p - a30) / s30 if s30 > 0 else 0

        # --- Advanced Supply ---
        woy_now = dt.isocalendar()[1]
        same_woy_records = [prices[j] for j in range(max(0, i-365), max(0, i-300))
                           if parse_date(dates[j]).isocalendar()[1] == woy_now] if i >= 300 else []
        yoy_ratio = oq7 / np.mean(same_woy_records) if same_woy_records and np.mean(same_woy_records) > 0 else 1

        origin_div = np.mean(origins[max(0,i-7):i]) if i >= 1 else origins[i]
        avg_lot = np.mean(qtys[max(0,i-7):i] / np.maximum(lots[max(0,i-7):i], 1)) if i >= 1 else 0
        hl_spread = np.mean(highs[max(0,i-7):i] - lows[max(0,i-7):i]) if i >= 1 else 0

        features = [
            # Calendar (7)
            dow, dt.month, dt.day, int(dow >= 5), dt.isocalendar()[1], (dt.month-1)//3+1, int(dow==0),
            # Holiday (4)
            hol["seollal"], hol["chuseok"], abs(hol["seollal"]), abs(hol["chuseok"]),
            # Price History (5) — using winsorized prices
            p, p1, p7, a7, a30,
            # Momentum (4)
            (p-p1)/p1*100 if p1>0 else 0, (p-p7)/p7*100 if p7>0 else 0,
            (p-p30)/p30*100 if p30>0 else 0, a7/a30-1 if a30>0 else 0,
            # Volatility (3)
            s7, s30, r7,
            # Own Supply (5)
            oq7, ol7, oqr, oqc, olc,
            # Cross Supply (5)
            otq, ml7, con, tsc, mc,
            # Seasonal (4)
            pvm, np.sin(2*np.pi*dt.month/12), np.cos(2*np.pi*dt.month/12), int(dt.month in [11,12,1,2]),
            # Weather Proxy (4)
            gap_d, ld, qd, ld+qd+int(gap_d>3),
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

        X.append(features)
        y.append(targets[i + offset])
        od.append(dates[i])
        is_outlier_sample.append(bool(outlier_mask[i + offset]))

    return np.array(X), np.array(y), fnames, od, is_outlier_sample


# ── Model Training / Backtesting ────────────────────────────────────

def train_lgbm(X_tr, y_tr, X_te):
    params = {
        "objective": "regression", "metric": "mae",
        "learning_rate": 0.03, "num_leaves": 31,
        "min_child_samples": 20, "feature_fraction": 0.7,
        "bagging_fraction": 0.7, "bagging_freq": 5,
        "reg_alpha": 0.1, "reg_lambda": 0.1,
        "verbose": -1, "n_jobs": 1,
    }
    model = lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=1200)
    return model.predict(X_te), model


def decompose_vmd(prices, K=3, alpha=2000):
    try:
        u, _, _ = VMD(prices, alpha, 0, K, 0, 1, 1e-7)
        return [u[k] for k in range(K)]
    except Exception:
        trend = np.convolve(prices, np.ones(30)/30, mode="same")
        return [trend, prices - trend]


def backtest_v10(X, y, fnames, prices_raw, species, horizon, method="vmd",
                 n_splits=5, outlier_flags=None):
    """
    5-fold time-series backtest.
    Fix 2: y is log(price); predictions are exp()'d before MAPE.
    Fix 3: outlier days excluded from training folds.
    Fix 5: adaptive VMD K.
    """
    n = len(X)
    min_train = int(n * 0.5)
    step = (n - min_train) // n_splits
    if step < 10 or min_train < 100:
        return None, None

    if outlier_flags is None:
        outlier_flags = [False] * n

    all_preds_log, all_actuals_log, all_prev = [], [], []
    last_model = None

    for s in range(n_splits):
        te = min_train + s * step
        te_end = min(te + step, n)
        if te_end <= te: continue

        # Fix 3: filter outliers from training indices
        train_mask = np.array([not outlier_flags[i] for i in range(te)])
        X_tr = X[:te][train_mask]
        y_tr = y[:te][train_mask]
        X_te = X[te:te_end]
        y_te = y[te:te_end]

        if len(X_tr) < 50:
            # Too few clean training samples; fall back to no filtering
            X_tr, y_tr = X[:te], y[:te]

        if method == "vmd":
            # Fix 5: adaptive K based on recent volatility (on raw prices, not log)
            K = adaptive_vmd_k(prices_raw[:te])
            try:
                modes = decompose_vmd(y_tr, K=K)
            except Exception:
                modes = [y_tr]
            combined = np.zeros(te_end - te)
            n_tr = len(X_tr)
            for mode in modes:
                m_arr = np.array(mode)
                # Ensure mode length matches training rows (safety trim/pad)
                if len(m_arr) != n_tr:
                    if len(m_arr) > n_tr:
                        m_arr = m_arr[:n_tr]
                    else:
                        m_arr = np.pad(m_arr, (0, n_tr - len(m_arr)), mode="edge")
                pred, last_model = train_lgbm(X_tr, m_arr, X_te)
                combined += pred
            all_preds_log.extend(combined)
        elif method == "ensemble":
            lgbm_pred, last_model = train_lgbm(X_tr, y_tr, X_te)
            arima_preds = []
            # ARIMA on log prices
            log_raw = np.log(np.maximum(prices_raw, 1.0))
            for t in range(te, te_end):
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    m = ARIMA(log_raw[max(0,t-365):t], order=(2,1,2)).fit()
                    arima_preds.append(m.forecast(steps=horizon)[-1])
                except Exception:
                    arima_preds.append(log_raw[t-1] if t > 0 else log_raw[0])
            combined = 0.6 * lgbm_pred + 0.4 * np.array(arima_preds)
            all_preds_log.extend(combined)
        else:
            pred, last_model = train_lgbm(X_tr, y_tr, X_te)
            all_preds_log.extend(pred)

        all_actuals_log.extend(y_te)
        all_prev.extend(X[te:te_end, 11])  # price_lag1 (winsorized)

    if not all_preds_log:
        return None, None

    P_log = np.array(all_preds_log)
    A_log = np.array(all_actuals_log)

    # Fix 2: exp() predictions and actuals for MAPE
    P = np.exp(P_log)
    A = np.exp(A_log)
    Pr = np.array(all_prev)

    mape = float(np.mean(np.abs(P - A) / np.where(A > 0, A, 1))) * 100
    rmse = float(np.sqrt(mean_squared_error(A, P)))
    mae = float(mean_absolute_error(A, P))
    dir_acc = float(np.mean((A > Pr) == (P > Pr))) * 100

    imp = {}
    if last_model:
        raw_imp = dict(zip(fnames, last_model.feature_importance(importance_type="gain")))
        total = sum(raw_imp.values())
        if total > 0:
            imp = {k: round(v/total*100, 2) for k, v in sorted(raw_imp.items(), key=lambda x: -x[1])}

    result = {"species": species, "model": f"v10-{method}", "horizon": horizon,
              "mape": round(mape, 2), "rmse": round(rmse), "mae": round(mae),
              "dir_acc": round(dir_acc, 1), "n_tests": len(P), "importance": imp}
    return result, imp


# ── Main ────────────────────────────────────────────────────────────

def main():
    data = load_all()
    n = len(data["trade_date"])
    ctx = build_supply_context(data, n)
    print(f"Supply context: {len(ctx['dates'])} days\n")

    all_results = []
    all_importance = {}
    outlier_counts = {}

    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        method = cfg.get("method", "vmd")
        print(f"{'='*70}")
        print(f"  {cfg['label']} — v10 ({method}), 68 features + 5 preprocessing fixes")
        print(f"{'='*70}")

        # Fix 1 + Fix 4: winsorized + origin-weighted daily aggregation
        records = extract_records_v10(data, n, cfg)
        if len(records) < 200:
            print(f"  SKIP — {len(records)} days\n"); continue

        prices_raw = np.array([r["price"] for r in records])
        print(f"  {len(records)} days | mean={np.mean(prices_raw):,.0f}")

        # Fix 3: flag outlier days
        outlier_mask = flag_outlier_days(records)
        n_outliers = int(outlier_mask.sum())
        outlier_counts[sp] = n_outliers
        print(f"  Outlier days flagged (will be excluded from training): {n_outliers}")

        if cfg.get("regime_split"):
            for months, tag, label in [({11,12,1,2}, "winter", "IN-SEASON"), ({3,4,5,6,7,8,9,10}, "other", "OFF-SEASON")]:
                recs = [r for r in records if parse_date(r["date"]).month in months]
                om = np.array([outlier_mask[i] for i, r in enumerate(records)
                               if parse_date(r["date"]).month in months])
                if len(recs) < 100: continue
                rp = np.array([r["price"] for r in recs])
                X, y, fnames, dates, ol_flags = build_features_v10(
                    recs, ctx, sp, 7, cfg.get("smoothed", False), outlier_mask=om)
                if len(X) < 100: continue
                r, imp = backtest_v10(X, y, fnames, rp, f"{sp}_{tag}", 7, method,
                                      outlier_flags=ol_flags)
                if r:
                    all_results.append(r)
                    all_importance[f"{sp}_{tag}"] = imp
                    print(f"\n  {label} 7d: MAPE={r['mape']:.1f}%  dir={r['dir_acc']:.1f}%")
                    for feat, v in list(imp.items())[:7]:
                        print(f"    {feat:<28} {v:>6.2f}%")
        else:
            for horizon in [7]:
                X, y, fnames, dates, ol_flags = build_features_v10(
                    records, ctx, sp, horizon, cfg.get("smoothed", False),
                    outlier_mask=outlier_mask)
                if len(X) < 200: continue
                r, imp = backtest_v10(X, y, fnames, prices_raw, sp, horizon, method,
                                      outlier_flags=ol_flags)
                if r:
                    all_results.append(r)
                    all_importance[sp] = imp
                    print(f"\n  {horizon}d: MAPE={r['mape']:.1f}%  dir={r['dir_acc']:.1f}%")
                    for feat, v in list(imp.items())[:7]:
                        print(f"    {feat:<28} {v:>6.2f}%")
        print()

    # Outlier count summary
    print("\n" + "=" * 70)
    print("FIX 3: OUTLIER DAYS REMOVED PER SPECIES")
    print("=" * 70)
    for sp, cnt in outlier_counts.items():
        print(f"  {sp:<15} {cnt:>4} days excluded from training")

    # v6 vs v10 comparison
    print("\n" + "=" * 70)
    print("v6 vs v10 COMPARISON (7-day horizon)")
    print("=" * 70)
    v6p = OUTPUT_DIR / "poc_v6_results.json"
    v6_data = {}
    if v6p.exists():
        with open(v6p) as f:
            for item in json.load(f).get("results", []):
                v6_data[item["species"]] = item.get("mape")

    print(f"\n  {'Species':<25} {'v6 MAPE':>8} {'v10 MAPE':>9} {'Δ':>8} {'v10 Dir%':>9}")
    print(f"  {'-'*63}")
    summary = {}
    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        if cfg.get("regime_split"):
            v10r = next((r for r in all_results if r["species"] == f"{sp}_winter"), None)
            label = f"{sp} (winter)"
            v6_key = f"{sp}_winter"
        else:
            v10r = next((r for r in all_results if r["species"] == sp), None)
            label = sp
            v6_key = sp
        if not v10r: continue
        v6m = v6_data.get(v6_key)
        if v6m and v6m > 0:
            delta = f"{(v6m - v10r['mape']) / v6m * 100:+.0f}%"
        else:
            delta = "—"
        v6_str = f"{v6m:.1f}%" if v6m else "n/a"
        print(f"  {label:<25} {v6_str:>8} {v10r['mape']:>8.1f}% {delta:>8} {v10r['dir_acc']:>8.1f}%")
        summary[sp] = {
            "v6": v6m,
            "v10": v10r["mape"],
            "dir_acc": v10r["dir_acc"],
            "outlier_days_removed": outlier_counts.get(sp, 0),
            "top_features": dict(list(all_importance.get(
                sp, all_importance.get(f"{sp}_winter", {})).items())[:15]),
        }

    # Preprocessing fixes summary
    print("\n" + "=" * 70)
    print("PREPROCESSING FIXES APPLIED")
    print("=" * 70)
    print("  Fix 1: Winsorized Mean — daily lots clipped to p10/p90 of rolling 30d")
    print("  Fix 2: Log-Transform Target — predict log(price), exp() for MAPE")
    print("  Fix 3: Outlier Day Removal — days >3σ excluded from training")
    print("  Fix 4: Origin-Weighted Aggregation — lots weighted by origin frequency")
    print("  Fix 5: Adaptive VMD K — K=5 (high vol) or K=3 (low vol)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_at": datetime.now().isoformat(),
        "version": "v10",
        "total_features": 68,
        "preprocessing_fixes": [
            "winsorized_mean",
            "log_transform_target",
            "outlier_day_removal",
            "origin_weighted_aggregation",
            "adaptive_vmd_k",
        ],
        "outlier_days_removed": outlier_counts,
        "results": all_results,
        "feature_importance": {k: dict(list(v.items())[:20]) for k, v in all_importance.items()},
        "summary": summary,
    }
    out_path = OUTPUT_DIR / "poc_v10_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
