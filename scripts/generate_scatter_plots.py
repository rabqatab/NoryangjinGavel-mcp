"""
Generate actual vs predicted scatter plots for all 20 species configs using the v10 LightGBM pipeline.

Runs a single 80/20 train/test split (not CV) per config, collects actual and predicted
prices from the test set, then generates:
  - One scatter plot per config  → docs/images/models/scatter/{config_id}_scatter.png
  - One combined 4x5 grid       → docs/images/models/scatter/all_configs_scatter_grid.png

Usage:
    uv run python scripts/generate_scatter_plots.py
"""
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from vmdpy import VMD

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "parquet" / "prices"
SCATTER_DIR = PROJECT_ROOT / "docs" / "images" / "models" / "scatter"

# ── Constants ────────────────────────────────────────────────────────

FOREIGN_KW = [
    '일본', '중국', '미국', '러시아', '캐나다', '노르웨이', '뉴질랜드', '대만', '칠레',
    '아르헨티나', '영국', '아일랜드', '온두라스', '북한', '(원양)', '인도', '인도네시아',
    '태국', '베트남', '필리핀', '호주', '스페인', '네덜란드', '페루', '모로코', '아프리카',
    '파키스탄', '라스팔마스', '포클랜드', '멕시코',
]

SASHIMI_SPECIES = [
    "넙치", "우럭", "방어", "참돔", "농어", "도다리", "감성돔",
    "감숭어", "참숭어", "쭈꾸미", "민어", "깐굴", "바위굴", "수꽃게", "암꽃게",
]

# All 20 configs: 7 from v10 + 13 from mullet test
SPECIES_CONFIGS = [
    # v10 original 7
    {"id": "넙치_활_kg_중", "species": "넙치", "state": "활", "pkg": "kg", "spec": "중",
     "domestic": False, "smoothed": False, "method": "vmd"},
    {"id": "우럭_활_kg_중", "species": "우럭", "state": "활", "pkg": "kg", "spec": "중",
     "domestic": False, "smoothed": False, "method": "ensemble"},
    {"id": "방어_선_kg_중_dom", "species": "방어", "state": "선", "pkg": "kg", "spec": "중",
     "domestic": True, "smoothed": True, "method": "vmd", "regime_split": True},
    {"id": "참돔_활_kg_중_dom", "species": "참돔", "state": "활", "pkg": "kg", "spec": "중",
     "domestic": True, "smoothed": False, "method": "vmd"},
    {"id": "농어_활_kg_중_dom", "species": "농어", "state": "활", "pkg": "kg", "spec": "중",
     "domestic": True, "smoothed": False, "method": "vmd"},
    {"id": "도다리_활_kg_중", "species": "도다리", "state": "활", "pkg": "kg", "spec": "중",
     "domestic": False, "smoothed": True, "method": "vmd"},
    {"id": "감성돔_활_kg_중_dom", "species": "감성돔", "state": "활", "pkg": "kg", "spec": "중",
     "domestic": True, "smoothed": False, "method": "vmd"},
    # mullet / extended 13
    {"id": "감숭어_활_kg_중", "species": "감숭어", "state": "활", "pkg": "kg", "spec": "중",
     "domestic": False, "smoothed": False, "method": "vmd"},
    {"id": "참숭어_활_kg_중", "species": "참숭어", "state": "활", "pkg": "kg", "spec": "중",
     "domestic": False, "smoothed": False, "method": "vmd"},
    {"id": "쭈꾸미_선_box_중_dom", "species": "쭈꾸미", "state": "선", "pkg": "box", "spec": "중",
     "domestic": True, "smoothed": False, "method": "vmd"},
    {"id": "민어_선_SP_중", "species": "민어", "state": "선", "pkg": "S/P", "spec": "중",
     "domestic": False, "smoothed": False, "method": "vmd"},
    {"id": "깐굴_선_box_소", "species": "깐굴", "state": "선", "pkg": "box", "spec": "소",
     "domestic": False, "smoothed": False, "method": "vmd"},
    {"id": "바위굴_활_box_대", "species": "바위굴", "state": "활", "pkg": "box", "spec": "대",
     "domestic": False, "smoothed": False, "method": "vmd"},
    {"id": "수꽃게_활_kg_중", "species": "수꽃게", "state": "활", "pkg": "kg", "spec": "중",
     "domestic": False, "smoothed": False, "method": "vmd"},
    {"id": "암꽃게_활_kg_중", "species": "암꽃게", "state": "활", "pkg": "kg", "spec": "중",
     "domestic": False, "smoothed": False, "method": "vmd"},
    {"id": "수꽃게_활_kg_대", "species": "수꽃게", "state": "활", "pkg": "kg", "spec": "대",
     "domestic": False, "smoothed": False, "method": "vmd"},
    {"id": "암꽃게_활_kg_대", "species": "암꽃게", "state": "활", "pkg": "kg", "spec": "대",
     "domestic": False, "smoothed": False, "method": "vmd"},
    {"id": "넙치_활_kg_2미", "species": "넙치", "state": "활", "pkg": "kg", "spec": "2미",
     "domestic": False, "smoothed": False, "method": "vmd"},
    {"id": "참돔_활_kg_2미_dom", "species": "참돔", "state": "활", "pkg": "kg", "spec": "2미",
     "domestic": True, "smoothed": False, "method": "vmd"},
    {"id": "농어_활_kg_1미_dom", "species": "농어", "state": "활", "pkg": "kg", "spec": "1미",
     "domestic": True, "smoothed": False, "method": "vmd"},
    {"id": "방어_활_kg_1미_dom", "species": "방어", "state": "활", "pkg": "kg", "spec": "1미",
     "domestic": True, "smoothed": True, "method": "vmd"},
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


# ── Helpers (copied from poc_prediction_v10.py) ───────────────────────

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


def origin_weight(origin, origin_freq_30d, max_freq_30d):
    freq = origin_freq_30d.get(origin, 1)
    return freq / max_freq_30d if max_freq_30d > 0 else 1.0


def weighted_mean(prices, weights):
    w = np.array(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        return float(np.mean(prices))
    return float(np.dot(prices, w) / total)


def adaptive_vmd_k(prices, window=90):
    a = np.array(prices, dtype=float)
    if len(a) < window:
        return 3
    recent_std = np.std(a[-window:])
    overall_std = np.std(a)
    return 5 if recent_std > overall_std else 3


# ── Data Loading ─────────────────────────────────────────────────────

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

        oq7 = ctx["sp_qty_7d"][target_sp][di] if target_sp in ctx["sp_qty_7d"] else 0
        ol7 = ctx["sp_lots_7d"][target_sp][di] if target_sp in ctx["sp_lots_7d"] else 0
        oq30 = (np.mean(ctx["sp_qty"][target_sp][max(0, di - 30):di])
                if di >= 1 and target_sp in ctx["sp_qty"] else oq7)
        oqr = oq7 / oq30 if oq30 > 0 else 1
        oqc = ((ctx["sp_qty_7d"][target_sp][di] - ctx["sp_qty_7d"][target_sp][max(0, di - 7)])
               / max(ctx["sp_qty_7d"][target_sp][max(0, di - 7)], 1)
               if target_sp in ctx["sp_qty_7d"] else 0)
        olc = ((ctx["sp_lots_7d"][target_sp][di] - ctx["sp_lots_7d"][target_sp][max(0, di - 7)])
               / max(ctx["sp_lots_7d"][target_sp][max(0, di - 7)], 1)
               if target_sp in ctx["sp_lots_7d"] else 0)
        otq = (ctx["total_sashimi_7d"][di] -
               (ctx["sp_qty_7d"][target_sp][di] if target_sp in ctx["sp_qty_7d"] else 0))
        ml7 = ctx["market_lots_7d"][di]
        con = ((ctx["sp_qty"][target_sp][di] / ctx["total_sashimi"][di])
               if target_sp in ctx["sp_qty"] and ctx["total_sashimi"][di] > 0 else 0)
        tsc = ((ctx["total_sashimi_7d"][di] - ctx["total_sashimi_7d"][max(0, di - 7)])
               / max(ctx["total_sashimi_7d"][max(0, di - 7)], 1))
        mc = ((ctx["market_lots_7d"][di] - ctx["market_lots_7d"][max(0, di - 7)])
              / max(ctx["market_lots_7d"][max(0, di - 7)], 1))
        pvm = p / monthly_avg.get(dt.month, p) if monthly_avg.get(dt.month, p) > 0 else 1
        gap_d = (dt - dt_prev).days
        ld = int(ol7 < ctx["sp_lots_7d"][target_sp][max(0, di - 14)] * 0.5) if (
            target_sp in ctx["sp_lots_7d"] and di >= 14) else 0
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
        next_gap = (parse_date(dates[i + 1]) - dt).days if i + 1 < len(dates) else 1
        is_pre_hol = int(next_gap > 2)
        consec_gap = gap_d
        week_pos = dow / 4 if dow <= 4 else 1.0
        days_left = max(0, 4 - dow)

        window_30 = prices[max(0, i - 30):i]
        window_90 = prices[max(0, i - 90):i]
        skew_30 = float(scipy_stats.skew(window_30)) if len(window_30) >= 3 else 0
        kurt_30 = float(scipy_stats.kurtosis(window_30)) if len(window_30) >= 3 else 0
        pct_90 = (float(scipy_stats.percentileofscore(window_90, p)) / 100
                  if len(window_90) >= 3 else 0.5)
        zscore_30 = (p - a30) / s30 if s30 > 0 else 0

        woy_now = dt.isocalendar()[1]
        same_woy_records = [
            prices[j] for j in range(max(0, i - 365), max(0, i - 300))
            if parse_date(dates[j]).isocalendar()[1] == woy_now
        ] if i >= 300 else []
        yoy_ratio = (oq7 / np.mean(same_woy_records)
                     if same_woy_records and np.mean(same_woy_records) > 0 else 1)

        origin_div = np.mean(origins[max(0, i - 7):i]) if i >= 1 else origins[i]
        avg_lot = (np.mean(qtys[max(0, i - 7):i] / np.maximum(lots[max(0, i - 7):i], 1))
                   if i >= 1 else 0)
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

    return np.array(X), np.array(y), od, is_outlier_sample


# ── Model helpers ─────────────────────────────────────────────────────

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
    return model.predict(X_te)


def train_quantile_lgbm(X_tr, y_tr, X_te, alpha):
    params = {
        "objective": "quantile", "alpha": alpha, "metric": "quantile",
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


def run_single_split(X, y, prices_raw, method, outlier_flags):
    """
    Single 80/20 train/test split.
    Returns dict with actual, predicted, q10, q90 arrays (all in raw price space).
    """
    n = len(X)
    split = int(n * 0.8)
    if split < 100 or (n - split) < 20:
        return None

    train_mask = np.array([not outlier_flags[i] for i in range(split)])
    X_tr = X[:split][train_mask]
    y_tr = y[:split][train_mask]
    if len(X_tr) < 50:
        X_tr, y_tr = X[:split], y[:split]
    X_te = X[split:]
    y_te = y[split:]

    if method == "vmd":
        K = adaptive_vmd_k(prices_raw[:split])
        try:
            modes = decompose_vmd(y_tr, K=K)
        except Exception:
            modes = [y_tr]
        combined = np.zeros(len(X_te))
        n_tr = len(X_tr)
        for mode in modes:
            m_arr = np.array(mode)
            if len(m_arr) != n_tr:
                m_arr = m_arr[:n_tr] if len(m_arr) > n_tr else np.pad(m_arr, (0, n_tr - len(m_arr)), mode="edge")
            pred = train_lgbm(X_tr, m_arr, X_te)
            combined += pred
        pred_q10_log = train_quantile_lgbm(X_tr, y_tr, X_te, alpha=0.1)
        pred_q90_log = train_quantile_lgbm(X_tr, y_tr, X_te, alpha=0.9)

    elif method == "ensemble":
        lgbm_pred = train_lgbm(X_tr, y_tr, X_te)
        arima_preds = []
        log_raw = np.log(np.maximum(prices_raw, 1.0))
        for t in range(split, n):
            try:
                from statsmodels.tsa.arima.model import ARIMA as _ARIMA
                m = _ARIMA(log_raw[max(0, t - 365):t], order=(2, 1, 2)).fit()
                arima_preds.append(m.forecast(steps=7)[-1])
            except Exception:
                arima_preds.append(log_raw[t - 1] if t > 0 else log_raw[0])
        combined = 0.6 * lgbm_pred + 0.4 * np.array(arima_preds)
        pred_q10_log = None
        pred_q90_log = None
    else:
        combined = train_lgbm(X_tr, y_tr, X_te)
        pred_q10_log = train_quantile_lgbm(X_tr, y_tr, X_te, alpha=0.1)
        pred_q90_log = train_quantile_lgbm(X_tr, y_tr, X_te, alpha=0.9)

    actual = np.exp(y_te)
    predicted = np.exp(combined)
    q10 = np.exp(pred_q10_log) if pred_q10_log is not None else None
    q90 = np.exp(pred_q90_log) if pred_q90_log is not None else None

    return {"actual": actual, "predicted": predicted, "q10": q10, "q90": q90}


# ── Plotting ──────────────────────────────────────────────────────────

def generate_scatter(actual, predicted, q10, q90, config_id, ax=None,
                     save_path=None):
    """
    Generate a scatter plot of actual vs predicted prices.

    Parameters
    ----------
    actual, predicted : np.ndarray  — raw price arrays (KRW)
    q10, q90         : np.ndarray or None — quantile bands
    config_id        : str
    ax               : matplotlib Axes (for grid subplot); if None, creates a new fig
    save_path        : Path; if given, saves the standalone figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    # Metrics
    mape = float(np.mean(np.abs(predicted - actual) / np.maximum(actual, 1))) * 100
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
    n_pts = len(actual)

    # Axis limits with 5% padding
    all_vals = list(actual) + list(predicted)
    if q10 is not None:
        all_vals += list(q10) + list(q90)
    mn = min(all_vals)
    mx = max(all_vals)
    pad = (mx - mn) * 0.05
    lim_lo = mn - pad
    lim_hi = mx + pad

    # ±20% error bands around diagonal
    diag = np.array([lim_lo, lim_hi])
    ax.fill_between(diag, diag * 0.8, diag * 1.2, alpha=0.10, color="gray",
                    label="±20% band")

    # Perfect prediction line
    ax.plot(diag, diag, "r--", alpha=0.6, linewidth=1.5, label="Perfect")

    # Quantile shading (scatter centred on actual, band from q10 to q90)
    if q10 is not None and q90 is not None:
        sort_idx = np.argsort(actual)
        ax.fill_between(actual[sort_idx], q10[sort_idx], q90[sort_idx],
                        alpha=0.15, color="steelblue", label="p10–p90")

    # Color dots by error magnitude
    errors = np.abs(predicted - actual) / np.maximum(actual, 1) * 100
    sc = ax.scatter(actual, predicted, c=errors, cmap="RdYlGn_r",
                    s=18 if standalone else 6, alpha=0.65, vmin=0, vmax=50,
                    zorder=3)
    if standalone:
        plt.colorbar(sc, ax=ax, label="Error %", fraction=0.046, pad=0.04)

    # Regression line annotation
    slope, intercept, rval, pval, _ = scipy_stats.linregress(actual, predicted)
    reg_x = np.array([lim_lo, lim_hi])
    reg_y = slope * reg_x + intercept
    ax.plot(reg_x, reg_y, "b-", alpha=0.4, linewidth=1.0)
    sign = "+" if intercept >= 0 else "-"
    ax.annotate(
        f"y = {slope:.2f}x {sign} {abs(intercept):,.0f}",
        xy=(0.05, 0.92), xycoords="axes fraction",
        fontsize=7 if standalone else 5,
        color="blue", alpha=0.8,
    )

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual Price (KRW)", fontsize=9 if standalone else 6)
    ax.set_ylabel("Predicted Price (KRW)", fontsize=9 if standalone else 6)

    if standalone:
        ax.set_title(
            f"{config_id}\nMAPE={mape:.1f}%, RMSE={rmse:,.0f}, R²={r2:.3f}, n={n_pts}",
            fontsize=10,
        )
        ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        ax.set_title(
            f"{config_id}\nMAPE={mape:.1f}% R²={r2:.2f}",
            fontsize=6,
        )

    return {"mape": mape, "rmse": rmse, "r2": r2, "n": n_pts}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    SCATTER_DIR.mkdir(parents=True, exist_ok=True)

    data = load_all()
    n = len(data["trade_date"])
    ctx = build_supply_context(data, n)
    print(f"Supply context: {len(ctx['dates'])} days\n")

    # Prepare combined grid figure
    n_configs = len(SPECIES_CONFIGS)
    grid_cols = 4
    grid_rows = (n_configs + grid_cols - 1) // grid_cols  # ceil
    fig_grid, axes_grid = plt.subplots(
        grid_rows, grid_cols, figsize=(24, grid_rows * 6), dpi=150
    )
    axes_flat = axes_grid.flatten()
    # Hide any unused subplots immediately
    for ax in axes_flat[n_configs:]:
        ax.set_visible(False)

    generated = []
    failed = []

    for cfg_idx, cfg in enumerate(SPECIES_CONFIGS):
        config_id = cfg["id"]
        sp = cfg["species"]
        method = cfg.get("method", "vmd")
        is_regime = cfg.get("regime_split", False)

        print(f"[{cfg_idx + 1:02d}/{n_configs}] {config_id}  method={method}", end="  ")

        records = extract_records_v10(data, n, cfg)

        if is_regime:
            # Use winter months (Nov–Feb) only, matching v10 main loop
            all_records = records
            records = [r for r in all_records if parse_date(r["date"]).month in {11, 12, 1, 2}]
            om_full = flag_outlier_days(all_records)
            outlier_mask = np.array(
                [om_full[i] for i, r in enumerate(all_records)
                 if parse_date(r["date"]).month in {11, 12, 1, 2}]
            )
        else:
            outlier_mask = flag_outlier_days(records)

        if len(records) < 200:
            print(f"SKIP (only {len(records)} days)")
            failed.append(config_id)
            axes_flat[cfg_idx].set_visible(False)
            continue

        prices_raw = np.array([r["price"] for r in records])

        X, y, od, ol_flags = build_features_v10(
            records, ctx, sp, 7, cfg.get("smoothed", False),
            outlier_mask=outlier_mask,
        )

        if len(X) < 100:
            print(f"SKIP (only {len(X)} feature rows)")
            failed.append(config_id)
            axes_flat[cfg_idx].set_visible(False)
            continue

        result = run_single_split(X, y, prices_raw, method, ol_flags)

        if result is None:
            print("SKIP (split too small)")
            failed.append(config_id)
            axes_flat[cfg_idx].set_visible(False)
            continue

        actual = result["actual"]
        predicted = result["predicted"]
        q10 = result["q10"]
        q90 = result["q90"]

        save_path = SCATTER_DIR / f"{config_id}_scatter.png"

        # Standalone plot
        metrics = generate_scatter(actual, predicted, q10, q90, config_id,
                                   ax=None, save_path=save_path)
        # Mini subplot in grid
        generate_scatter(actual, predicted, q10, q90, config_id,
                         ax=axes_flat[cfg_idx], save_path=None)

        print(f"MAPE={metrics['mape']:.1f}%  R²={metrics['r2']:.3f}  n={metrics['n']}")
        generated.append(config_id)

    # Finalize grid
    fig_grid.suptitle("Actual vs Predicted — All 20 Configs", fontsize=16, y=1.002)
    fig_grid.tight_layout()
    grid_path = SCATTER_DIR / "all_configs_scatter_grid.png"
    fig_grid.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig_grid)

    print(f"\n{'='*60}")
    print(f"Generated {len(generated)} individual scatter plots")
    if failed:
        print(f"Skipped ({len(failed)}): {', '.join(failed)}")
    print(f"Combined grid saved to: {grid_path}")
    print(f"Individual plots in:    {SCATTER_DIR}")
    print(f"Total plots: {len(generated)} individual + 1 grid = {len(generated) + 1}")


if __name__ == "__main__":
    main()
