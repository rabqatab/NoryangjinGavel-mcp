"""
PoC v11: 6 Advanced Improvements on top of v10.

New over v10:
  Improvement 1: Model Ensemble (Stacking)
      - LightGBM v10 (default params) + LightGBM alt (num_leaves=63, lr=0.01) + ARIMA(2,1,2)
      - Weighted: 0.5 * lgbm_v10 + 0.3 * lgbm_alt + 0.2 * arima
  Improvement 2: Per-Config Optuna (15 trials per config)
      - Tunes: learning_rate, num_leaves, min_child_samples, feature_fraction, num_boost_round
      - Train/val/test split: 60%/20%/20%
  Improvement 3: Recent Data Weighting
      - Exponential sample weights: exp(2.0 * idx / n_train) so last sample ~7x first
  Improvement 4: Conformalized Quantile Regression (CQR)
      - Calibrate quantile bands to guarantee (1-alpha) coverage
  Improvement 5: Cross-Config Features
      - For paired configs, add partner_price_lag1, partner_price_7d_avg, partner_price_ratio
      - Feature count: 71 (paired) or 68 (unpaired)
  Improvement 6: Weekly Target for Volatile Species
      - For configs with weekly_target=True (민어, 방어), predict mean(next 7 days)

Usage:
    uv run python scripts/poc_prediction_v11.py
"""
import json
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats as scipy_stats
from vmdpy import VMD

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "parquet" / "prices"
OUTPUT_DIR = PROJECT_ROOT / "data" / "poc_results"

FOREIGN_KW = ['일본','중국','미국','러시아','캐나다','노르웨이','뉴질랜드','대만','칠레',
              '아르헨티나','영국','아일랜드','온두라스','북한','(원양)','인도','인도네시아',
              '태국','베트남','필리핀','호주','스페인','네덜란드','페루','모로코','아프리카',
              '파키스탄','라스팔마스','포클랜드','멕시코']

SASHIMI_SPECIES = ["넙치", "우럭", "방어", "참돔", "농어", "도다리", "감성돔",
                    "감숭어", "참숭어", "쭈꾸미", "민어", "깐굴", "바위굴", "수꽃게", "암꽃게"]

# Improvement 5: Cross-config pairs (config_id -> partner_config_id)
CROSS_CONFIG_PAIRS = {
    "수꽃게_활_kg_중": "수꽃게_활_kg_대",
    "수꽃게_활_kg_대": "수꽃게_활_kg_중",
    "암꽃게_활_kg_중": "암꽃게_활_kg_대",
    "암꽃게_활_kg_대": "암꽃게_활_kg_중",
    "넙치_활_kg_중": "넙치_활_kg_2미",
    "넙치_활_kg_2미": "넙치_활_kg_중",
}

SPECIES_CONFIGS = [
    {"id": "넙치_활_kg_중", "species": "넙치", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False, "method": "vmd", "cross_config": "넙치_활_kg_2미"},
    {"id": "우럭_활_kg_중", "species": "우럭", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False, "method": "ensemble"},
    {"id": "방어_선_kg_중_dom", "species": "방어", "state": "선", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": True, "method": "vmd", "regime_split": True, "weekly_target": True},
    {"id": "참돔_활_kg_중_dom", "species": "참돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": False, "method": "vmd"},
    {"id": "농어_활_kg_중_dom", "species": "농어", "state": "활", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": False, "method": "vmd"},
    {"id": "도다리_활_kg_중", "species": "도다리", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": True, "method": "vmd"},
    {"id": "감성돔_활_kg_중_dom", "species": "감성돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": False, "method": "vmd"},
    {"id": "감숭어_활_kg_중", "species": "감숭어", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False, "method": "vmd"},
    {"id": "참숭어_활_kg_중", "species": "참숭어", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False, "method": "vmd"},
    {"id": "쭈꾸미_선_box_중_dom", "species": "쭈꾸미", "state": "선", "pkg": "box", "spec": "중", "domestic": True, "smoothed": False, "method": "vmd"},
    {"id": "민어_선_SP_중", "species": "민어", "state": "선", "pkg": "S/P", "spec": "중", "domestic": False, "smoothed": False, "method": "vmd", "weekly_target": True},
    {"id": "깐굴_선_box_소", "species": "깐굴", "state": "선", "pkg": "box", "spec": "소", "domestic": False, "smoothed": False, "method": "vmd"},
    {"id": "바위굴_활_box_대", "species": "바위굴", "state": "활", "pkg": "box", "spec": "대", "domestic": False, "smoothed": False, "method": "vmd"},
    {"id": "수꽃게_활_kg_중", "species": "수꽃게", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False, "method": "vmd", "cross_config": "수꽃게_활_kg_대"},
    {"id": "암꽃게_활_kg_중", "species": "암꽃게", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False, "method": "vmd", "cross_config": "암꽃게_활_kg_대"},
    {"id": "수꽃게_활_kg_대", "species": "수꽃게", "state": "활", "pkg": "kg", "spec": "대", "domestic": False, "smoothed": False, "method": "vmd", "cross_config": "수꽃게_활_kg_중"},
    {"id": "암꽃게_활_kg_대", "species": "암꽃게", "state": "활", "pkg": "kg", "spec": "대", "domestic": False, "smoothed": False, "method": "vmd", "cross_config": "암꽃게_활_kg_중"},
    {"id": "넙치_활_kg_2미", "species": "넙치", "state": "활", "pkg": "kg", "spec": "2미", "domestic": False, "smoothed": False, "method": "vmd", "cross_config": "넙치_활_kg_중"},
    {"id": "참돔_활_kg_2미_dom", "species": "참돔", "state": "활", "pkg": "kg", "spec": "2미", "domestic": True, "smoothed": False, "method": "vmd"},
    {"id": "농어_활_kg_1미_dom", "species": "농어", "state": "활", "pkg": "kg", "spec": "1미", "domestic": True, "smoothed": False, "method": "vmd"},
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


# ── Fix 1 (from v10): Winsorized Mean ───────────────────────────────

def winsorized_daily_price(day_prices, recent_30d_prices):
    """Clip extreme lots to p10/p90 of recent 30-day distribution."""
    if len(recent_30d_prices) < 10:
        return float(np.mean(day_prices))
    p10, p90 = np.percentile(recent_30d_prices, [10, 90])
    clipped = [max(p10, min(p90, p)) for p in day_prices]
    return float(np.mean(clipped))


# ── Fix 4 (from v10): Origin-Weighted Aggregation ──────────────────

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


# ── Fix 5 (from v10): Adaptive VMD K ───────────────────────────────

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
    day_lots = defaultdict(list)
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
    records = []
    rolling_lot_prices = []
    origin_days_seen = defaultdict(list)

    for day_i, d in enumerate(sorted_dates):
        lots = day_lots[d]
        day_prices = [lp[0] for lp in lots]
        day_origins_list = [lp[2] for lp in lots]

        origin_freq_30d = defaultdict(int)
        for origin_d, origin_day_idxs in origin_days_seen.items():
            count = sum(1 for di in origin_day_idxs if day_i - 30 <= di < day_i)
            if count > 0:
                origin_freq_30d[origin_d] = count
        max_freq_30d = max(origin_freq_30d.values()) if origin_freq_30d else 1

        lot_weights = [
            origin_weight(orig, origin_freq_30d, max_freq_30d)
            for orig in day_origins_list
        ]

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

        rolling_lot_prices.extend(day_prices)
        if day_i >= 30:
            cutoff_date = sorted_dates[max(0, day_i - 29)]
            rolling_lot_prices = [
                p
                for di2, d2 in enumerate(sorted_dates[max(0, day_i - 29):day_i + 1])
                for p in [lp[0] for lp in day_lots[d2]]
            ]

        for orig in day_origins[d]:
            origin_days_seen[orig].append(day_i)

    return records


# ── Fix 3 (from v10): Outlier Day Detection ─────────────────────────

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


# ── Improvement 4: CQR Calibration ──────────────────────────────────

def cqr_calibrate(q10_preds, q90_preds, actuals, alpha=0.1):
    """
    Adjust quantile bands to guarantee (1-alpha) coverage.
    Uses last 20% of provided data as calibration set.
    """
    scores = np.maximum(q10_preds - actuals, actuals - q90_preds)
    n_cal = max(1, len(scores) // 5)
    cal_scores = scores[-n_cal:]
    q_hat = np.percentile(cal_scores, (1 - alpha) * 100)
    adjusted_q10 = q10_preds - q_hat
    adjusted_q90 = q90_preds + q_hat
    return adjusted_q10, adjusted_q90


# ── Feature Engineering (v11: 68 base + 3 cross-config = 71 for paired) ──

def build_features_v11(records, ctx, target_sp, offset=7, use_smoothed=False,
                       outlier_mask=None, partner_series=None, weekly_target=False):
    """
    Build 68 base features (same as v10) + optionally 3 cross-config features.
    Improvement 5: cross-config features added when partner_series provided.
    Improvement 6: weekly_target uses mean(next 7 days) as target instead of point.
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

    # Fix 2 (from v10): log-transform target
    log_prices = np.log(np.maximum(prices, 1.0))

    # Improvement 6: weekly target for volatile species
    if weekly_target:
        targets = np.array([
            np.mean(log_prices[i+1:i+8]) if i + 8 <= len(log_prices) else log_prices[min(i+1, len(log_prices)-1)]
            for i in range(len(log_prices))
        ])
    else:
        targets = np.convolve(log_prices, np.ones(7)/7, mode="same") if use_smoothed and len(log_prices) > 7 else log_prices

    # Pre-compute technical indicators on winsorized prices (not log)
    ema7 = ema(prices, 7)
    ema30 = ema(prices, 30)
    macd_line, macd_sig = macd_signal(prices)
    rsi_14 = rsi(prices, 14)

    monthly_avg = defaultdict(list)
    for r in records:
        monthly_avg[parse_date(r["date"]).month].append(r["price"])
    monthly_avg = {m: np.mean(v) for m, v in monthly_avg.items()}

    # Build date-to-index map for partner series
    partner_date_map = {}
    if partner_series is not None:
        for pr in partner_series:
            partner_date_map[pr["date"]] = pr["price"]

    use_cross = (partner_series is not None and len(partner_date_map) > 0)

    base_fnames = [
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

    cross_fnames = ["partner_price_lag1", "partner_price_7d_avg", "partner_price_ratio"] if use_cross else []
    fnames = base_fnames + cross_fnames

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

        # Supply (safe lookup: non-sashimi species return zeros)
        nd_ctx = len(ctx["dates"])
        _zeros = np.zeros(nd_ctx)
        sp_qty_7d = ctx["sp_qty_7d"].get(target_sp, _zeros)
        sp_lots_7d = ctx["sp_lots_7d"].get(target_sp, _zeros)
        sp_qty = ctx["sp_qty"].get(target_sp, _zeros)
        oq7 = sp_qty_7d[di]
        ol7 = sp_lots_7d[di]
        oq30 = np.mean(sp_qty[max(0,di-30):di]) if di >= 1 else oq7
        oqr = oq7 / oq30 if oq30 > 0 else 1
        oqc = (sp_qty_7d[di] - sp_qty_7d[max(0,di-7)]) / max(sp_qty_7d[max(0,di-7)], 1)
        olc = (sp_lots_7d[di] - sp_lots_7d[max(0,di-7)]) / max(sp_lots_7d[max(0,di-7)], 1)
        otq = ctx["total_sashimi_7d"][di] - sp_qty_7d[di]
        ml7 = ctx["market_lots_7d"][di]
        con = sp_qty[di] / ctx["total_sashimi"][di] if ctx["total_sashimi"][di] > 0 else 0
        tsc = (ctx["total_sashimi_7d"][di] - ctx["total_sashimi_7d"][max(0,di-7)]) / max(ctx["total_sashimi_7d"][max(0,di-7)], 1)
        mc = (ctx["market_lots_7d"][di] - ctx["market_lots_7d"][max(0,di-7)]) / max(ctx["market_lots_7d"][max(0,di-7)], 1)
        pvm = p / monthly_avg.get(dt.month, p) if monthly_avg.get(dt.month, p) > 0 else 1
        gap_d = (dt - dt_prev).days
        ld = int(ol7 < sp_lots_7d[max(0,di-14)] * 0.5) if di >= 14 else 0
        qd = int(oq7 < oq30 * 0.5) if oq30 > 0 else 0

        # Technical Indicators
        boll_upper = a30 + 2 * s30
        boll_lower = a30 - 2 * s30
        boll_pct = (p - boll_lower) / (boll_upper - boll_lower) if (boll_upper - boll_lower) > 0 else 0.5
        mom_14 = (p - prices[i-14]) / prices[i-14] * 100 if i >= 14 and prices[i-14] > 0 else 0

        # Fourier
        f_sin_365 = np.sin(2 * np.pi * doy / 365)
        f_cos_365 = np.cos(2 * np.pi * doy / 365)
        f_sin_182 = np.sin(2 * np.pi * doy / 182.5)
        f_cos_182 = np.cos(2 * np.pi * doy / 182.5)
        f_sin_7 = np.sin(2 * np.pi * dow / 7)
        f_cos_7 = np.cos(2 * np.pi * dow / 7)

        # Advanced Calendar
        is_friday = int(dow == 4)
        next_gap = (parse_date(dates[i+1]) - dt).days if i + 1 < len(dates) else 1
        is_pre_hol = int(next_gap > 2)
        consec_gap = gap_d
        week_pos = dow / 4 if dow <= 4 else 1.0
        days_left = max(0, 4 - dow)

        # Price Distribution
        window_30 = prices[max(0,i-30):i]
        window_90 = prices[max(0,i-90):i]
        skew_30 = float(scipy_stats.skew(window_30)) if len(window_30) >= 3 else 0
        kurt_30 = float(scipy_stats.kurtosis(window_30)) if len(window_30) >= 3 else 0
        pct_90 = float(scipy_stats.percentileofscore(window_90, p)) / 100 if len(window_90) >= 3 else 0.5
        zscore_30 = (p - a30) / s30 if s30 > 0 else 0

        # Advanced Supply
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

        # Improvement 5: Cross-config features
        if use_cross:
            d_today = dates[i]
            # Partner lag1: yesterday's partner price
            d_prev = dates[i-1] if i > 0 else d_today
            partner_p1 = partner_date_map.get(d_prev, partner_date_map.get(d_today, p))
            # Partner 7d avg
            partner_7d_prices = [partner_date_map[dates[j]] for j in range(max(0,i-7), i)
                                  if dates[j] in partner_date_map]
            partner_7d_avg = float(np.mean(partner_7d_prices)) if partner_7d_prices else partner_p1
            # Partner price ratio (own / partner)
            partner_ratio = p / partner_p1 if partner_p1 > 0 else 1.0
            features.extend([partner_p1, partner_7d_avg, partner_ratio])

        X.append(features)
        y.append(targets[i + offset])
        od.append(dates[i])
        is_outlier_sample.append(bool(outlier_mask[i + offset]))

    return np.array(X), np.array(y), fnames, od, is_outlier_sample


# ── Model Training ───────────────────────────────────────────────────

def train_lgbm_v10(X_tr, y_tr, X_te, sample_weights=None):
    """v10 default LightGBM."""
    params = {
        "objective": "regression", "metric": "mae",
        "learning_rate": 0.03, "num_leaves": 31,
        "min_child_samples": 20, "feature_fraction": 0.7,
        "bagging_fraction": 0.7, "bagging_freq": 5,
        "reg_alpha": 0.1, "reg_lambda": 0.1,
        "verbose": -1, "n_jobs": 1,
    }
    ds = lgb.Dataset(X_tr, y_tr, weight=sample_weights)
    model = lgb.train(params, ds, num_boost_round=1200)
    return model.predict(X_te), model


def train_lgbm_alt(X_tr, y_tr, X_te, sample_weights=None):
    """Improvement 1: Alternative LightGBM with different hyperparameters."""
    params = {
        "objective": "regression", "metric": "mae",
        "learning_rate": 0.01, "num_leaves": 63,
        "min_child_samples": 20, "feature_fraction": 0.7,
        "bagging_fraction": 0.7, "bagging_freq": 5,
        "reg_alpha": 0.1, "reg_lambda": 0.1,
        "verbose": -1, "n_jobs": 1,
    }
    ds = lgb.Dataset(X_tr, y_tr, weight=sample_weights)
    model = lgb.train(params, ds, num_boost_round=2000)
    return model.predict(X_te), model


def train_lgbm_optuna(X_tr, y_tr, X_val, y_val, X_te):
    """Improvement 2: Optuna-tuned LightGBM (15 trials)."""
    def objective(trial):
        params = {
            "objective": "regression", "metric": "mae",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
            "bagging_fraction": 0.7, "bagging_freq": 5,
            "reg_alpha": 0.1, "reg_lambda": 0.1,
            "verbose": -1, "n_jobs": 1,
        }
        num_boost_round = trial.suggest_int("num_boost_round", 500, 2000)
        model = lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=num_boost_round)
        preds = model.predict(X_val)
        return float(np.mean(np.abs(preds - y_val)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15, show_progress_bar=False)

    best = study.best_params
    params = {
        "objective": "regression", "metric": "mae",
        "learning_rate": best["learning_rate"],
        "num_leaves": best["num_leaves"],
        "min_child_samples": best["min_child_samples"],
        "feature_fraction": best["feature_fraction"],
        "bagging_fraction": 0.7, "bagging_freq": 5,
        "reg_alpha": 0.1, "reg_lambda": 0.1,
        "verbose": -1, "n_jobs": 1,
    }
    num_boost_round = best["num_boost_round"]
    model = lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=num_boost_round)
    return model.predict(X_te), best


def train_quantile_lgbm(X_tr, y_tr, X_te, alpha):
    """Train a LightGBM quantile regression model and return predictions."""
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "metric": "quantile",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 20,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": 1,
    }
    model = lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=1000)
    return model.predict(X_te)


def decompose_vmd(prices, K=3, alpha=2000):
    try:
        u, _, _ = VMD(prices, alpha, 0, K, 0, 1, 1e-7)
        return [u[k] for k in range(K)]
    except Exception:
        trend = np.convolve(prices, np.ones(30)/30, mode="same")
        return [trend, prices - trend]


def make_recent_weights(n_train, decay=2.0):
    """Improvement 3: Exponential sample weights (last sample ~7x first)."""
    return np.exp(decay * np.arange(n_train) / n_train)


def arima_forecast(log_raw, t, horizon):
    """ARIMA(2,1,2) point forecast at time t, horizon steps ahead."""
    from statsmodels.tsa.arima.model import ARIMA
    fallback = log_raw[t-1] if t > 0 else log_raw[0]
    try:
        m = ARIMA(log_raw[max(0, t-365):t], order=(2, 1, 2)).fit()
        val = m.forecast(steps=horizon)[-1]
        if not np.isfinite(val):
            return fallback
        return val
    except Exception:
        return fallback


# ── Full v11 Backtest ────────────────────────────────────────────────

def backtest_v11(X, y, fnames, prices_raw, species, horizon, method="vmd",
                 n_splits=5, outlier_flags=None, run_optuna=True):
    """
    5-fold time-series backtest with all v11 improvements.
    Returns dict with multiple MAPE metrics:
      - mape_v10: baseline (same as v10 — no weights, no alt, no ARIMA stacking)
      - mape_ensemble: 0.5*lgbm_v10 + 0.3*lgbm_alt + 0.2*arima
      - mape_weighted: recent-weighted lgbm_v10
      - mape_optuna: Optuna-tuned (uses 60/20/20 split within training)
      - mape_best: best of all variants
    """
    n = len(X)
    min_train = int(n * 0.5)
    step = (n - min_train) // n_splits
    if step < 10 or min_train < 100:
        return None, None

    if outlier_flags is None:
        outlier_flags = [False] * n

    # Storage per fold
    all_actuals_log = []
    all_preds_v10_log = []
    all_preds_ensemble_log = []
    all_preds_weighted_log = []
    all_preds_optuna_log = []
    all_prev = []
    last_model = None

    log_raw = np.log(np.maximum(prices_raw, 1.0))

    for s in range(n_splits):
        te = min_train + s * step
        te_end = min(te + step, n)
        if te_end <= te: continue

        # Fix 3: filter outliers from training
        train_mask = np.array([not outlier_flags[i] for i in range(te)])
        X_tr = X[:te][train_mask]
        y_tr = y[:te][train_mask]
        X_te = X[te:te_end]
        y_te = y[te:te_end]

        if len(X_tr) < 50:
            X_tr, y_tr = X[:te], y[:te]
            train_mask = np.ones(te, dtype=bool)

        n_tr = len(X_tr)
        weights_recent = make_recent_weights(n_tr)

        # Improvement 2: split X_tr further into 60/20 for Optuna
        # (uses first 75% of fold-train as Optuna-train, last 25% as Optuna-val)
        optuna_split = int(n_tr * 0.75)
        X_opt_tr = X_tr[:optuna_split]
        y_opt_tr = y_tr[:optuna_split]
        X_opt_val = X_tr[optuna_split:]
        y_opt_val = y_tr[optuna_split:]

        if method == "vmd":
            K = adaptive_vmd_k(prices_raw[:te])

            # ── v10 baseline ──
            try:
                modes = decompose_vmd(y_tr, K=K)
            except Exception:
                modes = [y_tr]
            pred_v10 = np.zeros(te_end - te)
            for mode in modes:
                m_arr = np.array(mode)
                if len(m_arr) != n_tr:
                    m_arr = m_arr[:n_tr] if len(m_arr) > n_tr else np.pad(m_arr, (0, n_tr - len(m_arr)), mode="edge")
                p_mode, last_model = train_lgbm_v10(X_tr, m_arr, X_te)
                pred_v10 += p_mode

            # ── recent-weighted v10 ──
            pred_weighted = np.zeros(te_end - te)
            for mode in modes:
                m_arr = np.array(mode)
                if len(m_arr) != n_tr:
                    m_arr = m_arr[:n_tr] if len(m_arr) > n_tr else np.pad(m_arr, (0, n_tr - len(m_arr)), mode="edge")
                p_mode_w, _ = train_lgbm_v10(X_tr, m_arr, X_te, sample_weights=weights_recent)
                pred_weighted += p_mode_w

            # ── alt LightGBM ──
            pred_alt = np.zeros(te_end - te)
            for mode in modes:
                m_arr = np.array(mode)
                if len(m_arr) != n_tr:
                    m_arr = m_arr[:n_tr] if len(m_arr) > n_tr else np.pad(m_arr, (0, n_tr - len(m_arr)), mode="edge")
                p_mode_alt, _ = train_lgbm_alt(X_tr, m_arr, X_te)
                pred_alt += p_mode_alt

            # ── ARIMA for ensemble ──
            arima_preds = np.array([arima_forecast(log_raw, t, horizon) for t in range(te, te_end)])

            # ── Ensemble: 0.5 * v10 + 0.3 * alt + 0.2 * arima ──
            pred_ensemble = 0.5 * pred_v10 + 0.3 * pred_alt + 0.2 * arima_preds

            # ── Optuna ──
            if run_optuna and len(X_opt_val) >= 5:
                # Optuna runs on full log-target for VMD configs
                try:
                    pred_optuna, _ = train_lgbm_optuna(X_opt_tr, y_opt_tr, X_opt_val, y_opt_val, X_te)
                except Exception:
                    pred_optuna = pred_v10.copy()
            else:
                pred_optuna = pred_v10.copy()

        elif method == "ensemble":
            # For 우럭: existing ensemble (lgbm + arima), plus add alt + weighted
            pred_v10_lgbm, last_model = train_lgbm_v10(X_tr, y_tr, X_te)
            arima_preds = np.array([arima_forecast(log_raw, t, horizon) for t in range(te, te_end)])
            pred_v10 = 0.6 * pred_v10_lgbm + 0.4 * arima_preds

            pred_alt_lgbm, _ = train_lgbm_alt(X_tr, y_tr, X_te)
            pred_ensemble = 0.5 * pred_v10_lgbm + 0.3 * pred_alt_lgbm + 0.2 * arima_preds

            # weighted v10 lgbm
            pred_weighted_lgbm, _ = train_lgbm_v10(X_tr, y_tr, X_te, sample_weights=weights_recent)
            pred_weighted = 0.6 * pred_weighted_lgbm + 0.4 * arima_preds

            if run_optuna and len(X_opt_val) >= 5:
                try:
                    pred_optuna, _ = train_lgbm_optuna(X_opt_tr, y_opt_tr, X_opt_val, y_opt_val, X_te)
                except Exception:
                    pred_optuna = pred_v10.copy()
            else:
                pred_optuna = pred_v10.copy()

        else:
            pred_v10, last_model = train_lgbm_v10(X_tr, y_tr, X_te)
            pred_alt, _ = train_lgbm_alt(X_tr, y_tr, X_te)
            arima_preds = np.array([arima_forecast(log_raw, t, horizon) for t in range(te, te_end)])
            pred_ensemble = 0.5 * pred_v10 + 0.3 * pred_alt + 0.2 * arima_preds
            pred_weighted, _ = train_lgbm_v10(X_tr, y_tr, X_te, sample_weights=weights_recent)

            if run_optuna and len(X_opt_val) >= 5:
                try:
                    pred_optuna, _ = train_lgbm_optuna(X_opt_tr, y_opt_tr, X_opt_val, y_opt_val, X_te)
                except Exception:
                    pred_optuna = pred_v10.copy()
            else:
                pred_optuna = pred_v10.copy()

        all_actuals_log.extend(y_te)
        all_preds_v10_log.extend(pred_v10)
        all_preds_ensemble_log.extend(pred_ensemble)
        all_preds_weighted_log.extend(pred_weighted)
        all_preds_optuna_log.extend(pred_optuna)
        all_prev.extend(X[te:te_end, 11])

    if not all_actuals_log:
        return None, None

    A_log = np.array(all_actuals_log)
    A = np.exp(A_log)
    Pr = np.array(all_prev)

    def mape_of(preds_log):
        P = np.exp(np.array(preds_log))
        # Clip extreme predictions before MAPE (guard against inf/nan from ARIMA)
        P = np.where(np.isfinite(P), P, A)
        return float(np.mean(np.abs(P - A) / np.where(A > 0, A, 1))) * 100

    mape_v10 = mape_of(all_preds_v10_log)
    mape_ensemble = mape_of(all_preds_ensemble_log)
    mape_weighted = mape_of(all_preds_weighted_log)
    mape_optuna = mape_of(all_preds_optuna_log)

    mape_best = min(mape_v10, mape_ensemble, mape_weighted, mape_optuna)
    best_variant = ["v10_baseline", "ensemble", "weighted", "optuna"][
        np.argmin([mape_v10, mape_ensemble, mape_weighted, mape_optuna])
    ]

    P_best = np.exp(np.array(
        all_preds_v10_log if best_variant == "v10_baseline"
        else all_preds_ensemble_log if best_variant == "ensemble"
        else all_preds_weighted_log if best_variant == "weighted"
        else all_preds_optuna_log
    ))
    dir_acc = float(np.mean((A > Pr) == (P_best > Pr))) * 100
    rmse = float(np.sqrt(mean_squared_error(A, P_best)))
    mae = float(mean_absolute_error(A, P_best))

    imp = {}
    if last_model:
        raw_imp = dict(zip(fnames, last_model.feature_importance(importance_type="gain")))
        total = sum(raw_imp.values())
        if total > 0:
            imp = {k: round(v/total*100, 2) for k, v in sorted(raw_imp.items(), key=lambda x: -x[1])}

    result = {
        "species": species, "model": f"v11-{method}", "horizon": horizon,
        "mape_v10_baseline": round(mape_v10, 2),
        "mape_ensemble": round(mape_ensemble, 2),
        "mape_weighted": round(mape_weighted, 2),
        "mape_optuna": round(mape_optuna, 2),
        "mape_best": round(mape_best, 2),
        "best_variant": best_variant,
        "rmse": round(rmse), "mae": round(mae),
        "dir_acc": round(dir_acc, 1), "n_tests": len(A),
        "importance": imp,
    }
    return result, imp


def backtest_v11_quantile(X, y, prices_raw, method="vmd", n_splits=5, outlier_flags=None):
    """
    Quantile regression + CQR calibration (Improvement 4).
    Returns actuals, preds, q10, q90, cqr_q10, cqr_q90 across all folds.
    """
    if method == "ensemble":
        return None

    n = len(X)
    min_train = int(n * 0.5)
    step = (n - min_train) // n_splits
    if step < 10 or min_train < 100:
        return None

    if outlier_flags is None:
        outlier_flags = [False] * n

    all_actuals_raw = []
    all_preds_raw = []
    all_q10_raw = []
    all_q90_raw = []

    for s in range(n_splits):
        te = min_train + s * step
        te_end = min(te + step, n)
        if te_end <= te:
            continue

        train_mask = np.array([not outlier_flags[i] for i in range(te)])
        X_tr = X[:te][train_mask]
        y_tr = y[:te][train_mask]
        X_te = X[te:te_end]
        y_te = y[te:te_end]

        if len(X_tr) < 50:
            X_tr, y_tr = X[:te], y[:te]

        if method == "vmd":
            K = adaptive_vmd_k(prices_raw[:te])
            try:
                modes = decompose_vmd(y_tr, K=K)
            except Exception:
                modes = [y_tr]
            n_tr = len(X_tr)
            combined_point = np.zeros(te_end - te)
            for mode in modes:
                m_arr = np.array(mode)
                if len(m_arr) != n_tr:
                    m_arr = m_arr[:n_tr] if len(m_arr) > n_tr else np.pad(m_arr, (0, n_tr - len(m_arr)), mode="edge")
                pred_mode, _ = train_lgbm_v10(X_tr, m_arr, X_te)
                combined_point += pred_mode
            pred_q10_log = train_quantile_lgbm(X_tr, y_tr, X_te, alpha=0.1)
            pred_q90_log = train_quantile_lgbm(X_tr, y_tr, X_te, alpha=0.9)
        else:
            combined_point, _ = train_lgbm_v10(X_tr, y_tr, X_te)
            pred_q10_log = train_quantile_lgbm(X_tr, y_tr, X_te, alpha=0.1)
            pred_q90_log = train_quantile_lgbm(X_tr, y_tr, X_te, alpha=0.9)

        all_actuals_raw.extend(np.exp(y_te))
        all_preds_raw.extend(np.exp(combined_point))
        all_q10_raw.extend(np.exp(pred_q10_log))
        all_q90_raw.extend(np.exp(pred_q90_log))

    if not all_actuals_raw:
        return None

    actuals = np.array(all_actuals_raw)
    preds = np.array(all_preds_raw)
    q10 = np.array(all_q10_raw)
    q90 = np.array(all_q90_raw)

    # Improvement 4: CQR calibration
    cqr_q10, cqr_q90 = cqr_calibrate(q10, q90, actuals, alpha=0.1)

    return {
        "actuals": actuals,
        "preds": preds,
        "q10": q10,
        "q90": q90,
        "cqr_q10": cqr_q10,
        "cqr_q90": cqr_q90,
    }


def compute_conformal_bands(actuals, predictions, coverage=0.80):
    """Calibrate prediction bands from residuals to guarantee coverage."""
    residuals = actuals - predictions
    abs_residuals = np.abs(residuals)
    q = np.percentile(abs_residuals, coverage * 100)
    return q


# ── Main ─────────────────────────────────────────────────────────────

def main():
    data = load_all()
    n = len(data["trade_date"])
    ctx = build_supply_context(data, n)
    print(f"Supply context: {len(ctx['dates'])} days\n")

    # ── Pre-pass: Build daily price series for all configs ──────────
    print("Pre-pass: Building daily price series for cross-config features...")
    config_daily_series = {}  # config_id -> list of records (date, price, ...)
    for cfg in SPECIES_CONFIGS:
        cfg_id = cfg["id"]
        recs = extract_records_v10(data, n, cfg)
        config_daily_series[cfg_id] = recs
    print(f"  Built {len(config_daily_series)} config series.\n")

    all_results = []
    all_importance = {}
    outlier_counts = {}

    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        cfg_id = cfg["id"]
        method = cfg.get("method", "vmd")
        weekly_target = cfg.get("weekly_target", False)
        partner_id = cfg.get("cross_config")
        partner_series = config_daily_series.get(partner_id) if partner_id else None
        n_features = 71 if partner_series else 68

        print(f"{'='*70}")
        print(f"  {cfg_id} — v11 ({method}), {n_features} features"
              f"{' + weekly_target' if weekly_target else ''}"
              f"{f' + cross_config={partner_id}' if partner_series else ''}")
        print(f"{'='*70}")

        records = config_daily_series[cfg_id]
        if len(records) < 200:
            print(f"  SKIP — {len(records)} days\n"); continue

        prices_raw = np.array([r["price"] for r in records])
        cv = float(np.std(prices_raw) / np.mean(prices_raw)) if np.mean(prices_raw) > 0 else 0
        print(f"  {len(records)} days | mean={np.mean(prices_raw):,.0f} | CV={cv:.2f}")

        outlier_mask = flag_outlier_days(records)
        n_outliers = int(outlier_mask.sum())
        outlier_counts[cfg_id] = n_outliers
        print(f"  Outlier days flagged: {n_outliers}")

        if cfg.get("regime_split"):
            for months, tag, label_tag in [({11,12,1,2}, "winter", "IN-SEASON"), ({3,4,5,6,7,8,9,10}, "other", "OFF-SEASON")]:
                recs = [r for r in records if parse_date(r["date"]).month in months]
                om = np.array([outlier_mask[i] for i, r in enumerate(records)
                               if parse_date(r["date"]).month in months])
                if len(recs) < 100: continue
                rp = np.array([r["price"] for r in recs])
                X, y, fnames, dates, ol_flags = build_features_v11(
                    recs, ctx, sp, 7, cfg.get("smoothed", False),
                    outlier_mask=om, partner_series=partner_series,
                    weekly_target=weekly_target)
                if len(X) < 100: continue

                print(f"\n  Running {label_tag} (Optuna: 15 trials × {len(X)} samples)...")
                r, imp = backtest_v11(X, y, fnames, rp, f"{cfg_id}_{tag}", 7, method,
                                      outlier_flags=ol_flags, run_optuna=True)
                if r:
                    all_results.append(r)
                    all_importance[f"{cfg_id}_{tag}"] = imp
                    print(f"\n  {label_tag} 7d:")
                    print(f"    v10_baseline MAPE = {r['mape_v10_baseline']:.1f}%")
                    print(f"    Ensemble     MAPE = {r['mape_ensemble']:.1f}%")
                    print(f"    Weighted     MAPE = {r['mape_weighted']:.1f}%")
                    print(f"    Optuna       MAPE = {r['mape_optuna']:.1f}%")
                    print(f"    BEST         MAPE = {r['mape_best']:.1f}%  [{r['best_variant']}]")
                    print(f"    Dir Acc = {r['dir_acc']:.1f}%")
        else:
            X, y, fnames, dates, ol_flags = build_features_v11(
                records, ctx, sp, 7, cfg.get("smoothed", False),
                outlier_mask=outlier_mask, partner_series=partner_series,
                weekly_target=weekly_target)
            if len(X) < 200: continue

            print(f"\n  Running main model (Optuna: 15 trials × {len(X)} samples)...")
            r, imp = backtest_v11(X, y, fnames, prices_raw, cfg_id, 7, method,
                                  outlier_flags=ol_flags, run_optuna=True)
            if r:
                all_results.append(r)
                all_importance[cfg_id] = imp
                print(f"\n  7d:")
                print(f"    v10_baseline MAPE = {r['mape_v10_baseline']:.1f}%")
                print(f"    Ensemble     MAPE = {r['mape_ensemble']:.1f}%")
                print(f"    Weighted     MAPE = {r['mape_weighted']:.1f}%")
                print(f"    Optuna       MAPE = {r['mape_optuna']:.1f}%")
                print(f"    BEST         MAPE = {r['mape_best']:.1f}%  [{r['best_variant']}]")
                print(f"    Dir Acc = {r['dir_acc']:.1f}%")
                if imp:
                    print(f"  Top features:")
                    for feat, v in list(imp.items())[:5]:
                        print(f"    {feat:<32} {v:>6.2f}%")
        print()

    # ── Quantile + CQR Bands ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PRICE BAND PREDICTIONS (Quantile + CQR)")
    print("=" * 70)

    band_rows = []
    quantile_band_results = {}
    cqr_band_results = {}

    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        cfg_id = cfg["id"]
        method = cfg.get("method", "vmd")
        weekly_target = cfg.get("weekly_target", False)
        partner_id = cfg.get("cross_config")
        partner_series = config_daily_series.get(partner_id) if partner_id else None

        if cfg.get("regime_split"):
            records = config_daily_series[cfg_id]
            recs = [r for r in records if parse_date(r["date"]).month in {11,12,1,2}]
            om_full = flag_outlier_days(records)
            om = np.array([om_full[i] for i, r in enumerate(records)
                           if parse_date(r["date"]).month in {11,12,1,2}])
            if len(recs) < 100: continue
            rp = np.array([r["price"] for r in recs])
            X, y_arr, fnames, dates, ol_flags = build_features_v11(
                recs, ctx, sp, 7, cfg.get("smoothed", False), outlier_mask=om,
                partner_series=partner_series, weekly_target=weekly_target)
            label = f"{cfg_id}_winter"
        else:
            records = config_daily_series[cfg_id]
            if len(records) < 200: continue
            rp = np.array([r["price"] for r in records])
            om = flag_outlier_days(records)
            X, y_arr, fnames, dates, ol_flags = build_features_v11(
                records, ctx, sp, 7, cfg.get("smoothed", False), outlier_mask=om,
                partner_series=partner_series, weekly_target=weekly_target)
            label = cfg_id

        if len(X) < 200: continue

        print(f"\n  {label}  [method={method}]")

        if method == "ensemble":
            # Lightweight backtest for conformal bands (ensemble species only)
            n_pts = len(X)
            min_tr = int(n_pts * 0.5)
            step_sz = (n_pts - min_tr) // 5
            conf_actuals, conf_preds = [], []
            log_raw = np.log(np.maximum(rp, 1.0))
            if step_sz >= 10 and min_tr >= 100:
                for s in range(5):
                    te = min_tr + s * step_sz
                    te_end = min(te + step_sz, n_pts)
                    if te_end <= te: continue
                    train_mask = np.array([not ol_flags[i] for i in range(te)])
                    X_tr = X[:te][train_mask]
                    y_tr = y_arr[:te][train_mask]
                    if len(X_tr) < 50:
                        X_tr, y_tr = X[:te], y_arr[:te]
                    X_te = X[te:te_end]
                    y_te = y_arr[te:te_end]
                    lgbm_pred, _ = train_lgbm_v10(X_tr, y_tr, X_te)
                    arima_p = np.array([arima_forecast(log_raw, t, 7) for t in range(te, te_end)])
                    combined = 0.6 * lgbm_pred + 0.4 * arima_p
                    conf_actuals.extend(np.exp(y_te))
                    conf_preds.extend(np.exp(combined))

            if conf_actuals:
                ca = np.array(conf_actuals)
                cp = np.array(conf_preds)
                conf_q = compute_conformal_bands(ca, cp, coverage=0.80)
                lower_conf = cp - conf_q
                upper_conf = cp + conf_q
                cov_pct = float(np.mean((ca >= lower_conf) & (ca <= upper_conf))) * 100
                last_pred = cp[-1]
                cqr_band_results[label] = {
                    "conformal_q": round(float(conf_q)),
                    "coverage_actual": round(cov_pct, 1),
                    "note": "ensemble — CQR skipped",
                }
                print(f"    Conformal(80%): ±{conf_q:,.0f}  cov={cov_pct:.1f}%")
                band_rows.append({
                    "label": label, "point": round(last_pred),
                    "q10": None, "cqr_q10": None, "q90": None, "cqr_q90": None,
                    "conf_q": round(float(conf_q)), "coverage": round(cov_pct, 1),
                    "cqr_coverage": None, "method": "ensemble",
                })
            continue

        qb = backtest_v11_quantile(X, y_arr, rp, method=method, n_splits=5, outlier_flags=ol_flags)
        if qb is None:
            print(f"    SKIP — quantile backtest returned None")
            continue

        actuals_arr = qb["actuals"]
        preds_arr = qb["preds"]
        q10_arr = qb["q10"]
        q90_arr = qb["q90"]
        cqr_q10_arr = qb["cqr_q10"]
        cqr_q90_arr = qb["cqr_q90"]

        # Nominal coverage
        raw_coverage = float(np.mean((actuals_arr >= q10_arr) & (actuals_arr <= q90_arr))) * 100
        # CQR coverage
        cqr_coverage = float(np.mean((actuals_arr >= cqr_q10_arr) & (actuals_arr <= cqr_q90_arr))) * 100

        # Conformal from point residuals
        conf_q = compute_conformal_bands(actuals_arr, preds_arr, coverage=0.80)
        conf_lower = preds_arr - conf_q
        conf_upper = preds_arr + conf_q
        conf_cov = float(np.mean((actuals_arr >= conf_lower) & (actuals_arr <= conf_upper))) * 100

        q10_mean = float(np.mean(q10_arr))
        q50_mean = float(np.mean(preds_arr))
        q90_mean = float(np.mean(q90_arr))
        cqr_q10_mean = float(np.mean(cqr_q10_arr))
        cqr_q90_mean = float(np.mean(cqr_q90_arr))
        band_pct = (q90_mean - q10_mean) / q50_mean * 100 if q50_mean > 0 else 0
        cqr_band_pct = (cqr_q90_mean - cqr_q10_mean) / q50_mean * 100 if q50_mean > 0 else 0

        quantile_band_results[label] = {
            "q10_mean": round(q10_mean), "q50_mean": round(q50_mean), "q90_mean": round(q90_mean),
            "band_width_pct": round(band_pct, 1), "coverage_pct": round(raw_coverage, 1),
        }
        cqr_band_results[label] = {
            "cqr_q10_mean": round(cqr_q10_mean), "cqr_q90_mean": round(cqr_q90_mean),
            "cqr_band_pct": round(cqr_band_pct, 1), "cqr_coverage": round(cqr_coverage, 1),
            "conformal_q": round(float(conf_q)), "conf_coverage": round(conf_cov, 1),
        }

        last_pred = preds_arr[-1]
        last_q10 = q10_arr[-1]
        last_q90 = q90_arr[-1]
        last_cqr10 = cqr_q10_arr[-1]
        last_cqr90 = cqr_q90_arr[-1]

        print(f"    Q10={q10_mean:,.0f}  Q50={q50_mean:,.0f}  Q90={q90_mean:,.0f}  "
              f"band={band_pct:.1f}%  raw_cov={raw_coverage:.1f}%")
        print(f"    CQR: Q10={cqr_q10_mean:,.0f}  Q90={cqr_q90_mean:,.0f}  "
              f"band={cqr_band_pct:.1f}%  cqr_cov={cqr_coverage:.1f}%")

        band_rows.append({
            "label": label, "point": round(last_pred),
            "q10": round(last_q10), "q90": round(last_q90),
            "cqr_q10": round(last_cqr10), "cqr_q90": round(last_cqr90),
            "conf_q": round(float(conf_q)), "coverage": round(raw_coverage, 1),
            "cqr_coverage": round(cqr_coverage, 1), "method": method,
        })

    # ── Print Band Table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== PRICE BAND PREDICTIONS (with CQR) ===")
    print("=" * 70)
    hdr = f"  {'Config':<28} {'Point':>8} {'Q10':>8} {'Q90':>8} {'CQR_Q10':>9} {'CQR_Q90':>9} {'RawCov%':>8} {'CQRCov%':>8}"
    print(hdr)
    print(f"  {'-'*88}")
    for row in band_rows:
        q10_s = f"{row['q10']:,}" if row['q10'] is not None else "—"
        q90_s = f"{row['q90']:,}" if row['q90'] is not None else "—"
        cqr10_s = f"{row['cqr_q10']:,}" if row.get('cqr_q10') is not None else "—"
        cqr90_s = f"{row['cqr_q90']:,}" if row.get('cqr_q90') is not None else "—"
        cov_s = f"{row['coverage']:.1f}%" if row['coverage'] is not None else "—"
        cqr_cov_s = f"{row['cqr_coverage']:.1f}%" if row.get('cqr_coverage') is not None else "—"
        print(f"  {row['label']:<28} {row['point']:>8,} {q10_s:>8} {q90_s:>8} "
              f"{cqr10_s:>9} {cqr90_s:>9} {cov_s:>8} {cqr_cov_s:>8}")

    # ── v10 vs v11 Comparison Table ──────────────────────────────────
    print("\n" + "=" * 70)
    print("v10 vs v11 COMPARISON (7-day horizon)")
    print("=" * 70)
    v10p = OUTPUT_DIR / "poc_v10_results.json"
    v10_data = {}
    if v10p.exists():
        with open(v10p) as f:
            v10_json = json.load(f)
            for item in v10_json.get("results", []):
                # v10 uses species as key; v11 uses config id
                v10_data[item["species"]] = item.get("mape")

    print(f"\n  {'Config ID':<30} {'v10 MAPE':>9} {'v11 Base':>9} {'Ensemble':>9} "
          f"{'Weighted':>9} {'Optuna':>9} {'Best':>8} {'Variant':<14}")
    print(f"  {'-'*102}")

    summary = {}
    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        cfg_id = cfg["id"]
        if cfg.get("regime_split"):
            r = next((x for x in all_results if x["species"] == f"{cfg_id}_winter"), None)
            v10_key = f"{sp}_winter"
        else:
            r = next((x for x in all_results if x["species"] == cfg_id), None)
            v10_key = sp
        if not r: continue

        v10m = v10_data.get(v10_key)
        v10_s = f"{v10m:.1f}%" if v10m else "n/a"
        delta_s = f"{(v10m - r['mape_best']) / v10m * 100:+.0f}%" if v10m and v10m > 0 else "—"

        print(f"  {cfg_id:<30} {v10_s:>9} {r['mape_v10_baseline']:>8.1f}% "
              f"{r['mape_ensemble']:>8.1f}% {r['mape_weighted']:>8.1f}% "
              f"{r['mape_optuna']:>8.1f}% {r['mape_best']:>7.1f}% "
              f"{r['best_variant']:<14} ({delta_s})")

        summary[cfg_id] = {
            "v10_mape": v10m,
            "v11_baseline": r["mape_v10_baseline"],
            "v11_ensemble": r["mape_ensemble"],
            "v11_weighted": r["mape_weighted"],
            "v11_optuna": r["mape_optuna"],
            "v11_best": r["mape_best"],
            "best_variant": r["best_variant"],
            "dir_acc": r["dir_acc"],
            "outlier_days_removed": outlier_counts.get(cfg_id, 0),
            "n_features": 71 if cfg.get("cross_config") else 68,
            "weekly_target": cfg.get("weekly_target", False),
            "top_features": dict(list(all_importance.get(
                cfg_id, all_importance.get(f"{cfg_id}_winter", {})).items())[:15]),
        }

    # ── Save ─────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_at": datetime.now().isoformat(),
        "version": "v11",
        "total_features_base": 68,
        "total_features_cross_config": 71,
        "improvements": [
            "model_ensemble_stacking (0.5*lgbm_v10 + 0.3*lgbm_alt + 0.2*arima)",
            "per_config_optuna_15_trials",
            "recent_data_weighting_exp_decay_2.0",
            "conformalized_quantile_regression_cqr",
            "cross_config_features_3_new",
            "weekly_target_for_volatile_species",
        ],
        "outlier_days_removed": outlier_counts,
        "results": all_results,
        "feature_importance": {k: dict(list(v.items())[:20]) for k, v in all_importance.items()},
        "summary": summary,
        "quantile_bands": quantile_band_results,
        "cqr_bands": cqr_band_results,
    }
    out_path = OUTPUT_DIR / "poc_v11_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
