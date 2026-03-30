"""
General-purpose multi-species price prediction test script.

Tests all prediction approaches across configurable species/state/pkg/spec combinations.

Models tested:
  1. Naive (last value)
  2. SMA-7
  3. ARIMA(2,1,2)
  4. v10 LightGBM (VMD + 68 features + 5 preprocessing fixes)

All models use 7-day horizon. v10 includes quantile bands (p10/p50/p90).
Each config is uniquely identified by a tuple ID: species_state_pkg_spec[_dom].

Usage:
    uv run python scripts/poc_test_mullet.py
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
from statsmodels.tsa.arima.model import ARIMA
from vmdpy import VMD

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "parquet" / "prices"
OUTPUT_DIR = PROJECT_ROOT / "data" / "poc_results"

FOREIGN_KW = [
    '일본', '중국', '미국', '러시아', '캐나다', '노르웨이', '뉴질랜드', '대만', '칠레',
    '아르헨티나', '영국', '아일랜드', '온두라스', '북한', '(원양)', '인도', '인도네시아',
    '태국', '베트남', '필리핀', '호주', '스페인', '네덜란드', '페루', '모로코', '아프리카',
    '파키스탄', '라스팔마스', '포클랜드', '멕시코',
]

SASHIMI_SPECIES = ["넙치", "우럭", "방어", "참돔", "농어", "도다리", "감성돔",
                    "감숭어", "참숭어", "쭈꾸미", "민어", "깐굴", "바위굴", "수꽃게", "암꽃게"]

SPECIES_CONFIGS = [
    {"id": "감숭어_활_kg_중", "species": "감숭어", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False, "label": "감숭어 (mullet)", "method": "vmd"},
    {"id": "참숭어_활_kg_중", "species": "참숭어", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False, "label": "참숭어 (grey mullet)", "method": "vmd"},
    {"id": "쭈꾸미_선_box_중_dom", "species": "쭈꾸미", "state": "선", "pkg": "box", "spec": "중", "domestic": True, "smoothed": False, "label": "쭈꾸미 domestic (webfoot octopus)", "method": "vmd"},
    {"id": "민어_선_SP_중", "species": "민어", "state": "선", "pkg": "S/P", "spec": "중", "domestic": False, "smoothed": False, "label": "민어 (croaker)", "method": "vmd"},
    {"id": "깐굴_선_box_소", "species": "깐굴", "state": "선", "pkg": "box", "spec": "소", "domestic": False, "smoothed": False, "label": "깐굴 (shucked oyster)", "method": "vmd"},
    {"id": "바위굴_활_box_대", "species": "바위굴", "state": "활", "pkg": "box", "spec": "대", "domestic": False, "smoothed": False, "label": "바위굴 (rock oyster)", "method": "vmd"},
    {"id": "수꽃게_활_kg_중", "species": "수꽃게", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False, "label": "수꽃게 (male blue crab)", "method": "vmd"},
    {"id": "암꽃게_활_kg_중", "species": "암꽃게", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False, "label": "암꽃게 (female blue crab)", "method": "vmd"},
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


# ── Helpers ──────────────────────────────────────────────────────────

def is_foreign(o):
    if not o:
        return False
    return any(kw in o for kw in FOREIGN_KW)


def parse_date(d):
    return datetime.strptime(d, "%Y.%m.%d")


def days_to_holiday(dt):
    r = {"seollal": 999, "chuseok": 999}
    for y in [dt.year - 1, dt.year, dt.year + 1]:
        if y not in KOREAN_HOLIDAYS:
            continue
        for name, hd in KOREAN_HOLIDAYS[y].items():
            diff = (parse_date(hd) - dt).days
            if abs(diff) < abs(r[name]):
                r[name] = diff
    return r


# ── Technical Indicators ─────────────────────────────────────────────

def ema(prices, span):
    a = np.array(prices, dtype=float)
    out = np.empty_like(a)
    out[0] = a[0]
    alpha = 2 / (span + 1)
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
    return out


def macd_signal(prices):
    e12 = ema(prices, 12)
    e26 = ema(prices, 26)
    macd_line = e12 - e26
    signal = ema(macd_line, 9)
    return macd_line, signal


def rsi(prices, period=14):
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


# ── Fix 1: Winsorized Mean ────────────────────────────────────────────

def origin_weight(origin, origin_freq_30d, max_freq_30d):
    freq = origin_freq_30d.get(origin, 1)
    return freq / max_freq_30d if max_freq_30d > 0 else 1.0


def weighted_mean(prices, weights):
    w = np.array(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        return float(np.mean(prices))
    return float(np.dot(prices, w) / total)


# ── Fix 5: Adaptive VMD K ─────────────────────────────────────────────

def adaptive_vmd_k(prices, window=90):
    a = np.array(prices, dtype=float)
    if len(a) < window:
        return 3
    recent_std = np.std(a[-window:])
    overall_std = np.std(a)
    return 5 if recent_std > overall_std else 3


# ── Data Loading ──────────────────────────────────────────────────────

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
        "sp_qty_7d": {s: np.convolve(q, np.ones(k) / k, mode="same") for s, q in sp_qty.items()},
        "sp_lots_7d": {s: np.convolve(l, np.ones(k) / k, mode="same") for s, l in sp_lots.items()},
        "market_lots": market_lots,
        "market_lots_7d": np.convolve(market_lots, np.ones(k) / k, mode="same"),
        "total_sashimi": sum(sp_qty.values()),
        "total_sashimi_7d": np.convolve(sum(sp_qty.values()), np.ones(k) / k, mode="same"),
    }


def extract_records_v10(data, n, cfg):
    """Apply Fix 1 (winsorized mean) + Fix 4 (origin-weighted) to build daily price series."""
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
            rolling_lot_prices = [
                p
                for di2, d2 in enumerate(sorted_dates[max(0, day_i - 29):day_i + 1])
                for p in [lp[0] for lp in day_lots[d2]]
            ]

        for orig in day_origins[d]:
            origin_days_seen[orig].append(day_i)

    return records


# ── Fix 3: Outlier Day Detection ──────────────────────────────────────

def flag_outlier_days(records, window=30, n_sigma=3):
    prices = np.array([r["price"] for r in records])
    is_outlier = np.zeros(len(prices), dtype=bool)
    for i in range(window, len(prices)):
        window_prices = prices[max(0, i - window):i]
        mu = np.mean(window_prices)
        sigma = np.std(window_prices)
        if sigma > 0 and abs(prices[i] - mu) > n_sigma * sigma:
            is_outlier[i] = True
    return is_outlier


# ── Feature Engineering (68 features, v10) ───────────────────────────

def build_features_v10(records, ctx, target_sp, offset=7, use_smoothed=False,
                       outlier_mask=None):
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

    log_prices = np.log(np.maximum(prices, 1.0))
    targets = (np.convolve(log_prices, np.ones(7) / 7, mode="same")
               if use_smoothed and len(log_prices) > 7 else log_prices)

    ema7 = ema(prices, 7)
    ema30 = ema(prices, 30)
    macd_line, macd_sig = macd_signal(prices)
    rsi_14 = rsi(prices, 14)

    monthly_avg = defaultdict(list)
    for r in records:
        monthly_avg[parse_date(r["date"]).month].append(r["price"])
    monthly_avg = {m: np.mean(v) for m, v in monthly_avg.items()}

    # Supply context — use first sashimi species as proxy if target not in ctx
    sp_ctx = target_sp if target_sp in ctx["sp_qty"] else SASHIMI_SPECIES[0]

    fnames = [
        "dow", "month", "dom", "is_weekend", "woy", "quarter", "is_monday",
        "days_seollal", "days_chuseok", "abs_seollal", "abs_chuseok",
        "price_lag1", "price_lag7", "price_lag30", "price_7d", "price_30d",
        "pchg_1d", "pchg_7d", "pchg_30d", "pchg_7v30",
        "std_7d", "std_30d", "range_7d",
        "own_q7", "own_l7", "own_q_ratio", "own_q_chg", "own_l_chg",
        "other_q7", "mkt_l7", "concentration", "sashimi_chg", "mkt_chg",
        "price_vs_month", "month_sin", "month_cos", "is_peak",
        "gap", "lots_drop", "qty_drop", "shock",
        "ema_7", "ema_30", "macd", "macd_signal", "macd_hist",
        "bollinger_pct", "rsi_14", "momentum_14d",
        "fourier_sin_365", "fourier_cos_365", "fourier_sin_182", "fourier_cos_182",
        "fourier_sin_7", "fourier_cos_7",
        "is_friday", "is_pre_holiday", "consecutive_gap", "week_position", "days_left_in_week",
        "skewness_30d", "kurtosis_30d", "percentile_90d", "zscore_30d",
        "own_q_yoy_ratio", "origin_diversity_7d", "avg_lot_size_7d", "hl_spread_7d",
    ]

    X, y, od, is_outlier_sample = [], [], [], []

    for i in range(90, len(records) - offset):
        dt = parse_date(dates[i])
        di = di_map.get(dates[i], 0)
        dt_prev = parse_date(dates[i - 1]) if i > 0 else dt
        hol = days_to_holiday(dt)
        dow = dt.weekday()
        doy = dt.timetuple().tm_yday

        p = prices[i]
        p1 = prices[i - 1] if i >= 1 else p
        p7 = prices[i - 7] if i >= 7 else p1
        p30 = prices[i - 30] if i >= 30 else p1
        a7 = np.mean(prices[max(0, i - 7):i])
        a30 = np.mean(prices[max(0, i - 30):i])
        s7 = np.std(prices[max(0, i - 7):i])
        s30 = np.std(prices[max(0, i - 30):i])
        r7 = float(max(prices[max(0, i - 7):i]) - min(prices[max(0, i - 7):i]))

        oq7 = ctx["sp_qty_7d"][sp_ctx][di]
        ol7 = ctx["sp_lots_7d"][sp_ctx][di]
        oq30 = np.mean(ctx["sp_qty"][sp_ctx][max(0, di - 30):di]) if di >= 1 else oq7
        oqr = oq7 / oq30 if oq30 > 0 else 1
        oqc = ((ctx["sp_qty_7d"][sp_ctx][di] - ctx["sp_qty_7d"][sp_ctx][max(0, di - 7)])
               / max(ctx["sp_qty_7d"][sp_ctx][max(0, di - 7)], 1))
        olc = ((ctx["sp_lots_7d"][sp_ctx][di] - ctx["sp_lots_7d"][sp_ctx][max(0, di - 7)])
               / max(ctx["sp_lots_7d"][sp_ctx][max(0, di - 7)], 1))
        otq = ctx["total_sashimi_7d"][di] - ctx["sp_qty_7d"][sp_ctx][di]
        ml7 = ctx["market_lots_7d"][di]
        con = (ctx["sp_qty"][sp_ctx][di] / ctx["total_sashimi"][di]
               if ctx["total_sashimi"][di] > 0 else 0)
        tsc = ((ctx["total_sashimi_7d"][di] - ctx["total_sashimi_7d"][max(0, di - 7)])
               / max(ctx["total_sashimi_7d"][max(0, di - 7)], 1))
        mc = ((ctx["market_lots_7d"][di] - ctx["market_lots_7d"][max(0, di - 7)])
              / max(ctx["market_lots_7d"][max(0, di - 7)], 1))
        pvm = p / monthly_avg.get(dt.month, p) if monthly_avg.get(dt.month, p) > 0 else 1
        gap_d = (dt - dt_prev).days
        ld = int(ol7 < ctx["sp_lots_7d"][sp_ctx][max(0, di - 14)] * 0.5) if di >= 14 else 0
        qd = int(oq7 < oq30 * 0.5) if oq30 > 0 else 0

        boll_upper = a30 + 2 * s30
        boll_lower = a30 - 2 * s30
        boll_pct = ((p - boll_lower) / (boll_upper - boll_lower)
                    if (boll_upper - boll_lower) > 0 else 0.5)
        mom_14 = (p - prices[i - 14]) / prices[i - 14] * 100 if i >= 14 and prices[i - 14] > 0 else 0

        f_sin_365 = np.sin(2 * np.pi * doy / 365)
        f_cos_365 = np.cos(2 * np.pi * doy / 365)
        f_sin_182 = np.sin(2 * np.pi * doy / 182.5)
        f_cos_182 = np.cos(2 * np.pi * doy / 182.5)
        f_sin_7 = np.sin(2 * np.pi * dow / 7)
        f_cos_7 = np.cos(2 * np.pi * dow / 7)

        is_friday = int(dow == 4)
        next_gap = (parse_date(dates[i + 1]) - dt).days if i + 1 < len(dates) else 1
        is_pre_hol = int(next_gap > 2)
        consec_gap = gap_d
        week_pos = dow / 4 if dow <= 4 else 1.0
        days_left = max(0, 4 - dow)

        window_30 = prices[max(0, i - 30):i]
        window_90 = prices[max(0, i - 90):i]
        skew_30 = float(scipy_stats.skew(window_30)) if len(window_30) >= 3 else 0
        kurt_30 = float(scipy_stats.kurtosis(window_30)) if len(window_30) >= 3 else 0
        pct_90 = float(scipy_stats.percentileofscore(window_90, p)) / 100 if len(window_90) >= 3 else 0.5
        zscore_30 = (p - a30) / s30 if s30 > 0 else 0

        woy_now = dt.isocalendar()[1]
        same_woy_records = [
            prices[j] for j in range(max(0, i - 365), max(0, i - 300))
            if parse_date(dates[j]).isocalendar()[1] == woy_now
        ] if i >= 300 else []
        yoy_ratio = oq7 / np.mean(same_woy_records) if same_woy_records and np.mean(same_woy_records) > 0 else 1

        origin_div = np.mean(origins[max(0, i - 7):i]) if i >= 1 else origins[i]
        avg_lot = np.mean(qtys[max(0, i - 7):i] / np.maximum(lots[max(0, i - 7):i], 1)) if i >= 1 else 0
        hl_spread = np.mean(highs[max(0, i - 7):i] - lows[max(0, i - 7):i]) if i >= 1 else 0

        features = [
            dow, dt.month, dt.day, int(dow >= 5), dt.isocalendar()[1], (dt.month - 1) // 3 + 1, int(dow == 0),
            hol["seollal"], hol["chuseok"], abs(hol["seollal"]), abs(hol["chuseok"]),
            p, p1, p7, a7, a30,
            (p - p1) / p1 * 100 if p1 > 0 else 0,
            (p - p7) / p7 * 100 if p7 > 0 else 0,
            (p - p30) / p30 * 100 if p30 > 0 else 0,
            a7 / a30 - 1 if a30 > 0 else 0,
            s7, s30, r7,
            oq7, ol7, oqr, oqc, olc,
            otq, ml7, con, tsc, mc,
            pvm, np.sin(2 * np.pi * dt.month / 12), np.cos(2 * np.pi * dt.month / 12),
            int(dt.month in [11, 12, 1, 2]),
            gap_d, ld, qd, ld + qd + int(gap_d > 3),
            ema7[i], ema30[i], macd_line[i], macd_sig[i], macd_line[i] - macd_sig[i],
            boll_pct, rsi_14[i], mom_14,
            f_sin_365, f_cos_365, f_sin_182, f_cos_182, f_sin_7, f_cos_7,
            is_friday, is_pre_hol, consec_gap, week_pos, days_left,
            skew_30, kurt_30, pct_90, zscore_30,
            yoy_ratio, origin_div, avg_lot, hl_spread,
        ]

        X.append(features)
        y.append(targets[i + offset])
        od.append(dates[i])
        is_outlier_sample.append(bool(outlier_mask[i + offset]))

    return np.array(X), np.array(y), fnames, od, is_outlier_sample


# ── LightGBM training ─────────────────────────────────────────────────

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


def train_quantile_lgbm(X_tr, y_tr, X_te, alpha):
    params = {
        "objective": "quantile", "alpha": alpha,
        "metric": "quantile",
        "learning_rate": 0.03, "num_leaves": 31,
        "min_child_samples": 20, "feature_fraction": 0.7,
        "bagging_fraction": 0.7, "bagging_freq": 5,
        "reg_alpha": 0.1, "reg_lambda": 0.1,
        "verbose": -1, "n_jobs": 1,
    }
    model = lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=1000)
    return model.predict(X_te)


def decompose_vmd(prices, K=3, alpha=2000):
    try:
        u, _, _ = VMD(prices, alpha, 0, K, 0, 1, 1e-7)
        return [u[k] for k in range(K)]
    except Exception:
        trend = np.convolve(prices, np.ones(30) / 30, mode="same")
        return [trend, prices - trend]


# ── Baseline Models ───────────────────────────────────────────────────

def backtest_naive(records, horizon=7):
    """Last-value naive: predict price[i] for price[i+horizon]."""
    prices = [r["price"] for r in records]
    errs = []
    for i in range(180, len(prices) - horizon, 7):
        actual = prices[i + horizon]
        pred = prices[i]
        if actual > 0:
            errs.append(abs(pred - actual) / actual)
    mape = float(np.mean(errs)) * 100 if errs else 999.0
    return {"model": "naive", "mape": round(mape, 2), "n_tests": len(errs)}


def backtest_sma7(records, horizon=7):
    """SMA-7: predict 7-day moving average for price[i+horizon]."""
    prices = [r["price"] for r in records]
    errs = []
    for i in range(180, len(prices) - horizon, 7):
        if i < 7:
            continue
        actual = prices[i + horizon]
        pred = float(np.mean(prices[i - 7:i]))
        if actual > 0:
            errs.append(abs(pred - actual) / actual)
    mape = float(np.mean(errs)) * 100 if errs else 999.0
    return {"model": "sma7", "mape": round(mape, 2), "n_tests": len(errs)}


def backtest_arima(records, horizon=7):
    """ARIMA(2,1,2) on log prices; fit on up to 365 recent days."""
    prices = np.array([r["price"] for r in records])
    log_prices = np.log(np.maximum(prices, 1.0))
    errs = []
    # Sample every 14 days to keep runtime reasonable
    indices = list(range(180, len(prices) - horizon, 14))
    for i in indices:
        actual = prices[i + horizon]
        if actual <= 0:
            continue
        window = log_prices[max(0, i - 365):i]
        if len(window) < 20:
            continue
        try:
            m = ARIMA(window, order=(2, 1, 2)).fit()
            log_pred = m.forecast(steps=horizon)[-1]
            pred = np.exp(log_pred)
            errs.append(abs(pred - actual) / actual)
        except Exception:
            pass
    mape = float(np.mean(errs)) * 100 if errs else 999.0
    return {"model": "arima212", "mape": round(mape, 2), "n_tests": len(errs)}


# ── v10 LightGBM Backtest ─────────────────────────────────────────────

def backtest_v10(X, y, fnames, prices_raw, species, horizon=7, method="vmd",
                 n_splits=5, outlier_flags=None):
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
            combined = np.zeros(te_end - te)
            n_tr = len(X_tr)
            for mode in modes:
                m_arr = np.array(mode)
                if len(m_arr) != n_tr:
                    if len(m_arr) > n_tr:
                        m_arr = m_arr[:n_tr]
                    else:
                        m_arr = np.pad(m_arr, (0, n_tr - len(m_arr)), mode="edge")
                pred, last_model = train_lgbm(X_tr, m_arr, X_te)
                combined += pred
            all_preds_log.extend(combined)
        else:
            pred, last_model = train_lgbm(X_tr, y_tr, X_te)
            all_preds_log.extend(pred)

        all_actuals_log.extend(y_te)
        all_prev.extend(X[te:te_end, 11])  # price_lag1 feature index

    if not all_preds_log:
        return None, None

    P_log = np.array(all_preds_log)
    A_log = np.array(all_actuals_log)
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
            imp = {k: round(v / total * 100, 2)
                   for k, v in sorted(raw_imp.items(), key=lambda x: -x[1])}

    result = {
        "species": species, "model": f"v10-{method}", "horizon": horizon,
        "mape": round(mape, 2), "rmse": round(rmse), "mae": round(mae),
        "dir_acc": round(dir_acc, 1), "n_tests": len(P), "importance": imp,
    }
    return result, imp


# ── Quantile Backtest ─────────────────────────────────────────────────

def compute_conformal_bands(actuals, predictions, coverage=0.80):
    residuals = actuals - predictions
    abs_residuals = np.abs(residuals)
    return np.percentile(abs_residuals, coverage * 100)


def backtest_quantile(X, y, prices_raw, method="vmd", n_splits=5, outlier_flags=None):
    n = len(X)
    min_train = int(n * 0.5)
    step = (n - min_train) // n_splits
    if step < 10 or min_train < 100:
        return None

    if outlier_flags is None:
        outlier_flags = [False] * n

    all_actuals_raw, all_preds_raw, all_q10_raw, all_q90_raw = [], [], [], []

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
            combined_point = np.zeros(te_end - te)
            n_tr = len(X_tr)
            for mode in modes:
                m_arr = np.array(mode)
                if len(m_arr) != n_tr:
                    if len(m_arr) > n_tr:
                        m_arr = m_arr[:n_tr]
                    else:
                        m_arr = np.pad(m_arr, (0, n_tr - len(m_arr)), mode="edge")
                pred_mode, _ = train_lgbm(X_tr, m_arr, X_te)
                combined_point += pred_mode
            pred_q10_log = train_quantile_lgbm(X_tr, y_tr, X_te, alpha=0.1)
            pred_q90_log = train_quantile_lgbm(X_tr, y_tr, X_te, alpha=0.9)
        else:
            combined_point, _ = train_lgbm(X_tr, y_tr, X_te)
            pred_q10_log = train_quantile_lgbm(X_tr, y_tr, X_te, alpha=0.1)
            pred_q90_log = train_quantile_lgbm(X_tr, y_tr, X_te, alpha=0.9)

        all_actuals_raw.extend(np.exp(y_te))
        all_preds_raw.extend(np.exp(combined_point))
        all_q10_raw.extend(np.exp(pred_q10_log))
        all_q90_raw.extend(np.exp(pred_q90_log))

    if not all_actuals_raw:
        return None

    return {
        "actuals": np.array(all_actuals_raw),
        "preds":   np.array(all_preds_raw),
        "q10":     np.array(all_q10_raw),
        "q90":     np.array(all_q90_raw),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    data = load_all()
    n = len(data["trade_date"])
    ctx = build_supply_context(data, n)
    print(f"Supply context: {len(ctx['dates'])} trading days\n")

    all_results = []
    band_rows = []
    quantile_band_results = {}
    conformal_band_results = {}

    for cfg in SPECIES_CONFIGS:
        cfg_id = cfg["id"]
        sp = cfg["species"]
        label = cfg["label"]
        method = cfg["method"]

        print(f"{'=' * 70}")
        print(f"  {label} — v10 ({method}), 68 features + 5 preprocessing fixes")
        print(f"{'=' * 70}")

        # Fix 1 + Fix 4: extract with winsorized + origin-weighted aggregation
        records = extract_records_v10(data, n, cfg)
        if len(records) < 200:
            print(f"  SKIP — only {len(records)} trading days\n")
            continue

        prices_raw = np.array([r["price"] for r in records])
        print(f"  {len(records)} trading days  |  mean={np.mean(prices_raw):,.0f}  "
              f"min={np.min(prices_raw):,.0f}  max={np.max(prices_raw):,.0f}")

        # Fix 3: flag outlier days
        outlier_mask = flag_outlier_days(records)
        n_outliers = int(outlier_mask.sum())
        print(f"  Outlier days flagged (excluded from training): {n_outliers}")

        # ── Baseline models ─────────────────────────────────────────
        print(f"\n  Running baselines...")
        naive_r = backtest_naive(records, horizon=7)
        sma7_r = backtest_sma7(records, horizon=7)
        print(f"    Naive:   MAPE={naive_r['mape']:.1f}%  (n={naive_r['n_tests']})")
        print(f"    SMA-7:   MAPE={sma7_r['mape']:.1f}%  (n={sma7_r['n_tests']})")

        print(f"    ARIMA(2,1,2): running...", end=" ", flush=True)
        arima_r = backtest_arima(records, horizon=7)
        print(f"MAPE={arima_r['mape']:.1f}%  (n={arima_r['n_tests']})")

        # ── v10 LightGBM ────────────────────────────────────────────
        print(f"\n  Building v10 features (68)...")
        X, y, fnames, dates_out, ol_flags = build_features_v10(
            records, ctx, sp, offset=7,
            use_smoothed=cfg.get("smoothed", False),
            outlier_mask=outlier_mask,
        )
        print(f"  Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

        if len(X) < 200:
            print(f"  SKIP v10 — too few samples ({len(X)})\n")
            continue

        print(f"  Running v10 LightGBM (5-fold time-series CV, {method})...")
        v10_r, imp = backtest_v10(X, y, fnames, prices_raw, sp, horizon=7,
                                  method=method, n_splits=5, outlier_flags=ol_flags)

        if v10_r is None:
            print(f"  v10 returned None — insufficient data\n")
            continue

        print(f"    v10-{method}: MAPE={v10_r['mape']:.1f}%  "
              f"RMSE={v10_r['rmse']:,.0f}  dir={v10_r['dir_acc']:.1f}%")

        # ── Comparison table ────────────────────────────────────────
        print(f"\n  {'Model':<22} {'MAPE':>7}  {'vs Naive':>9}")
        print(f"  {'-' * 42}")
        for model_r, mname in [
            (naive_r, "Naive (last value)"),
            (sma7_r,  "SMA-7"),
            (arima_r, "ARIMA(2,1,2)"),
            (v10_r,   f"v10 LightGBM ({method})"),
        ]:
            mape = model_r["mape"]
            if mname == "Naive (last value)":
                delta = "—"
            else:
                delta = f"{(naive_r['mape'] - mape) / naive_r['mape'] * 100:+.0f}%"
            print(f"  {mname:<22} {mape:>6.1f}%  {delta:>9}")

        # ── Feature Importance Top 10 ───────────────────────────────
        if imp:
            print(f"\n  Top 10 Features (v10):")
            for feat, val in list(imp.items())[:10]:
                print(f"    {feat:<28} {val:>6.2f}%")

        # ── Quantile bands ──────────────────────────────────────────
        print(f"\n  Running quantile regression (p10/p50/p90)...")
        qb = backtest_quantile(X, y, prices_raw, method=method, n_splits=5, outlier_flags=ol_flags)

        if qb is not None:
            actuals_arr = qb["actuals"]
            preds_arr   = qb["preds"]
            q10_arr     = qb["q10"]
            q90_arr     = qb["q90"]

            conf_q = compute_conformal_bands(actuals_arr, preds_arr, coverage=0.80)
            lower_conf = preds_arr - conf_q
            upper_conf = preds_arr + conf_q
            coverage_actual = float(np.mean(
                (actuals_arr >= lower_conf) & (actuals_arr <= upper_conf)
            )) * 100

            q10_mean = float(np.mean(q10_arr))
            q50_mean = float(np.mean(preds_arr))
            q90_mean = float(np.mean(q90_arr))
            band_width_mean = q90_mean - q10_mean
            band_pct = band_width_mean / q50_mean * 100 if q50_mean > 0 else 0
            conf_pct = conf_q / q50_mean * 100 if q50_mean > 0 else 0

            print(f"    Q10={q10_mean:,.0f}  Q50={q50_mean:,.0f}  Q90={q90_mean:,.0f}")
            print(f"    Band width: {band_width_mean:,.0f} ({band_pct:.1f}%)")
            print(f"    Conformal(80%): ±{conf_q:,.0f}  actual_coverage={coverage_actual:.1f}%")

            last_pred = preds_arr[-1]
            last_q10 = q10_arr[-1]
            last_q90 = q90_arr[-1]
            last_band = last_q90 - last_q10
            last_band_pct = last_band / last_pred * 100 if last_pred > 0 else 0

            quantile_band_results[cfg_id] = {
                "q10_mean": round(q10_mean),
                "q50_mean": round(q50_mean),
                "q90_mean": round(q90_mean),
                "band_width_pct": round(band_pct, 1),
            }
            conformal_band_results[cfg_id] = {
                "conformal_q": round(float(conf_q)),
                "coverage_actual": round(coverage_actual, 1),
                "band_width_pct": round(conf_pct * 2, 1),
            }
            band_rows.append({
                "id": cfg_id,
                "label": label,
                "species": sp,
                "point": round(last_pred),
                "q10": round(last_q10),
                "q90": round(last_q90),
                "band_width": round(last_band),
                "band_pct": round(last_band_pct, 1),
                "conf_q": round(float(conf_q)),
                "coverage": round(coverage_actual, 1),
                "method": method,
            })
        else:
            print(f"    Quantile backtest returned None (insufficient data)")

        all_results.append({
            "id": cfg_id,
            "species": sp,
            "label": label,
            "n_days": len(records),
            "mean_price": round(float(np.mean(prices_raw))),
            "outlier_days": n_outliers,
            "naive": naive_r,
            "sma7": sma7_r,
            "arima": arima_r,
            "v10": v10_r,
        })
        print()

    # ── Final Summary Table ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL COMPARISON SUMMARY — all configs (7-day horizon)")
    print("=" * 80)
    print(f"\n  {'Config ID':<34} {'Naive':>7} {'SMA-7':>7} {'ARIMA':>7} {'v10 LGBM':>9} {'Improv':>8}")
    print(f"  {'-' * 74}")
    for r in all_results:
        naive_m = r["naive"]["mape"]
        sma_m   = r["sma7"]["mape"]
        arima_m = r["arima"]["mape"]
        v10_m   = r["v10"]["mape"] if r["v10"] else 999.0
        improv  = f"{(naive_m - v10_m) / naive_m * 100:+.0f}%" if naive_m > 0 else "—"
        print(f"  {r['id']:<34} {naive_m:>6.1f}% {sma_m:>6.1f}% {arima_m:>6.1f}% "
              f"{v10_m:>8.1f}% {improv:>8}")

    # ── Band Summary Table ────────────────────────────────────────────
    if band_rows:
        print("\n" + "=" * 80)
        print("PRICE BAND PREDICTIONS (quantile + conformal)")
        print("=" * 80)
        hdr = (f"  {'Config ID':<34} {'Point':>8} {'Q10':>8} {'Q90':>8} "
               f"{'BandW%':>7} {'Conf±':>8} {'Cov%':>6}")
        print(hdr)
        print(f"  {'-' * 76}")
        for row in band_rows:
            print(f"  {row['id']:<34} {row['point']:>8,} {row['q10']:>8,} {row['q90']:>8,} "
                  f"{row['band_pct']:>6.0f}% {row['conf_q']:>8,} {row['coverage']:>5.1f}%")

    # ── MCP-style Output ──────────────────────────────────────────────
    if band_rows:
        print("\n" + "=" * 80)
        print("MCP SERVER OUTPUT (consumer-friendly)")
        print("=" * 80)
        for row in band_rows:
            point = row["point"]
            conf_q = row["conf_q"]
            likely_lo = max(0, point - conf_q)
            likely_hi = point + conf_q
            print(f"\n[{row['id']}] {row['label']} 7-day forecast:")
            print(f"  Expected:      {point:,} KRW/kg")
            print(f"  Likely range:  {likely_lo:,} ~ {likely_hi:,} KRW/kg (80% confidence)")
            print(f"  Budget range:  {row['q10']:,} ~ {row['q90']:,} KRW/kg (p10 ~ p90)")

    # ── Save results ──────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_at": datetime.now().isoformat(),
        "version": "multi_species_v1",
        "configs_tested": [cfg["id"] for cfg in SPECIES_CONFIGS],
        "horizon_days": 7,
        "preprocessing_fixes": [
            "winsorized_mean",
            "log_transform_target",
            "outlier_day_removal",
            "origin_weighted_aggregation",
            "adaptive_vmd_k",
        ],
        "total_features": 68,
        "results": {r["id"]: r for r in all_results},
        "quantile_bands": quantile_band_results,
        "conformal_bands": conformal_band_results,
    }
    out_path = OUTPUT_DIR / "poc_mullet_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
