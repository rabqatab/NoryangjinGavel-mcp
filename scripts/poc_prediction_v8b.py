"""
PoC v8b: Ocean Features on 2020+ Data Only — Fair A/B Comparison.

Key difference from v8:
  - Only uses data from 2020.01.01 onwards (where ocean data is available).
  - Drops any rows where ocean features are None (no NaN-tolerance needed).
  - Runs two comparisons:
      (A) v8b (2020+, with ocean) vs v6 (full history) — overall impact
      (B) v8b (2020+, with ocean) vs v8b-baseline (2020+, without ocean)
          — isolates the true ocean feature contribution

Usage:
    uv run python scripts/poc_prediction_v8b.py
"""
import csv
import json
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats as scipy_stats
from vmdpy import VMD

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "parquet" / "prices"
OCEAN_CSV = PROJECT_ROOT / "data" / "ocean" / "ocean_daily.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "poc_results"

CUTOFF_DATE = "2020.01.01"

FOREIGN_KW = ['일본','중국','미국','러시아','캐나다','노르웨이','뉴질랜드','대만','칠레',
              '아르헨티나','영국','아일랜드','온두라스','북한','(원양)','인도','인도네시아',
              '태국','베트남','필리핀','호주','스페인','네덜란드','페루','모로코','아프리카',
              '파키스탄','라스팔마스','포클랜드','멕시코']

SASHIMI_SPECIES = ["넙치", "우럭", "방어", "참돔", "농어", "도다리", "감성돔"]

SPECIES_CONFIGS = [
    {"species": "넙치", "state": "활", "pkg": "kg", "spec": "중", "domestic": False,
     "smoothed": False, "label": "넙치 (flatfish)", "method": "vmd", "ocean_station": "busan"},
    {"species": "우럭", "state": "활", "pkg": "kg", "spec": "중", "domestic": False,
     "smoothed": False, "label": "우럭 (rockfish)", "method": "ensemble", "ocean_station": "busan"},
    {"species": "방어", "state": "선", "pkg": "kg", "spec": "중", "domestic": True,
     "smoothed": True, "label": "방어 (yellowtail)", "method": "vmd", "regime_split": True, "ocean_station": "jeju"},
    {"species": "참돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True,
     "smoothed": False, "label": "참돔 (seabream)", "method": "vmd", "ocean_station": "jeju"},
    {"species": "농어", "state": "활", "pkg": "kg", "spec": "중", "domestic": True,
     "smoothed": False, "label": "농어 (sea bass)", "method": "vmd", "ocean_station": "incheon"},
    {"species": "도다리", "state": "활", "pkg": "kg", "spec": "중", "domestic": False,
     "smoothed": True, "label": "도다리 (flounder)", "method": "vmd", "ocean_station": "yeosu"},
    {"species": "감성돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True,
     "smoothed": False, "label": "감성돔 (black porgy)", "method": "vmd", "ocean_station": "busan"},
]

OCEAN_FEATURE_NAMES = [
    "wave_height_max_lag1",
    "swell_height_max_lag1",
    "wind_speed_max_lag1",
    "wind_gust_max_lag1",
    "temp_mean_lag1",
    "pressure_msl_lag1",
    "precipitation_lag1",
    "sunshine_hours_lag1",
    "wave_height_7d_avg",
    "wind_speed_7d_avg",
    "temp_7d_avg",
    "is_storm",
]


def is_foreign(o):
    if not o: return False
    return any(kw in o for kw in FOREIGN_KW)


def parse_date(d): return datetime.strptime(d, "%Y.%m.%d")


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


# ── Ocean Data Loading ───────────────────────────────────────────────

def load_ocean_data():
    """Load ocean CSV into nested dict: {location: {date_str: row_dict}}."""
    ocean = defaultdict(dict)
    with open(OCEAN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loc = row["location"]
            date = row["date"]
            parsed = {}
            for col in ["wave_height_max", "swell_height_max", "temp_mean", "temp_max",
                        "temp_min", "wind_speed_max", "wind_gust_max", "precipitation",
                        "pressure_msl", "sunshine_hours", "wave_period_max",
                        "swell_period_max", "wave_direction"]:
                v = row.get(col, "")
                parsed[col] = float(v) if v not in ("", None) else None
            ocean[loc][date] = parsed
    print(f"Ocean data loaded: {sum(len(v) for v in ocean.values())} rows, "
          f"locations: {sorted(ocean.keys())}")
    return dict(ocean)


def build_ocean_lookup(ocean_data, station):
    """Return dict: date_str -> {col: val} for a single station."""
    return ocean_data.get(station, {})


def get_ocean_features(station_data, date_str, dates_sorted):
    """
    Return 12 ocean feature values for a given date.
    Uses lag-1 (yesterday's conditions) and 7-day rolling average up to yesterday.
    Returns a list that may contain None values — caller must filter.
    """
    dt = parse_date(date_str)
    lag1_date = (dt - timedelta(days=1)).strftime("%Y.%m.%d")
    lag1 = station_data.get(lag1_date, {})

    # 7-day rolling average: d-7 .. d-1
    window7_wave, window7_wind, window7_temp = [], [], []
    for k in range(1, 8):
        past = (dt - timedelta(days=k)).strftime("%Y.%m.%d")
        row = station_data.get(past, {})
        if row.get("wave_height_max") is not None:
            window7_wave.append(row["wave_height_max"])
        if row.get("wind_speed_max") is not None:
            window7_wind.append(row["wind_speed_max"])
        if row.get("temp_mean") is not None:
            window7_temp.append(row["temp_mean"])

    wave_7d = float(np.mean(window7_wave)) if window7_wave else None
    wind_7d = float(np.mean(window7_wind)) if window7_wind else None
    temp_7d = float(np.mean(window7_temp)) if window7_temp else None

    gust = lag1.get("wind_gust_max")
    is_storm = int(gust > 50) if gust is not None else None

    return [
        lag1.get("wave_height_max"),       # wave_height_max_lag1
        lag1.get("swell_height_max"),      # swell_height_max_lag1
        lag1.get("wind_speed_max"),        # wind_speed_max_lag1
        lag1.get("wind_gust_max"),         # wind_gust_max_lag1
        lag1.get("temp_mean"),             # temp_mean_lag1
        lag1.get("pressure_msl"),          # pressure_msl_lag1
        lag1.get("precipitation"),         # precipitation_lag1
        lag1.get("sunshine_hours"),        # sunshine_hours_lag1
        wave_7d,                           # wave_height_7d_avg
        wind_7d,                           # wind_speed_7d_avg
        temp_7d,                           # temp_7d_avg
        is_storm,                          # is_storm
    ]


# ── Data Loading ────────────────────────────────────────────────────

def load_all():
    import pyarrow.dataset as ds
    print("Loading price data...", end=" ", flush=True)
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


def extract_records(data, n, cfg):
    day_data = defaultdict(lambda: {"prices": [], "highs": [], "lows": [], "origins": set(), "qty": 0})
    for i in range(n):
        if data["species"][i] != cfg["species"]: continue
        if data["state"][i] != cfg["state"]: continue
        if data["packaging"][i] != cfg["pkg"]: continue
        if data["spec"][i] != cfg["spec"]: continue
        if cfg["domestic"] and is_foreign(data["origin"][i]): continue
        d = data["trade_date"][i]
        day_data[d]["prices"].append(data["price_avg"][i])
        day_data[d]["highs"].append(data["price_high"][i])
        day_data[d]["lows"].append(data["price_low"][i])
        if data["origin"][i]: day_data[d]["origins"].add(data["origin"][i])
        day_data[d]["qty"] += data["quantity"][i]
    return [{
        "date": d, "price": float(np.mean(dd["prices"])),
        "high": max(dd["highs"]), "low": min(dd["lows"]),
        "n_lots": len(dd["prices"]), "n_origins": len(dd["origins"]), "qty": dd["qty"],
    } for d, dd in sorted(day_data.items())]


# ── Feature Engineering ──────────────────────────────────────────────

def _build_base_features(records, ctx, target_sp, offset=7, use_smoothed=False):
    """
    Shared inner builder for the 68 v6-style features (no ocean).
    Returns (X, y, fnames, dates) where X is a plain float array.
    """
    prices = np.array([r["price"] for r in records])
    highs = np.array([r["high"] for r in records])
    lows = np.array([r["low"] for r in records])
    lots = np.array([r["n_lots"] for r in records])
    origins = np.array([r["n_origins"] for r in records])
    qtys = np.array([r["qty"] for r in records])
    dates = [r["date"] for r in records]
    di_map = ctx["date_idx"]

    targets = np.convolve(prices, np.ones(7)/7, mode="same") if use_smoothed and len(prices) > 7 else prices

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

    X, y, od = [], [], []
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

        boll_upper = a30 + 2 * s30
        boll_lower = a30 - 2 * s30
        boll_pct = (p - boll_lower) / (boll_upper - boll_lower) if (boll_upper - boll_lower) > 0 else 0.5
        mom_14 = (p - prices[i-14]) / prices[i-14] * 100 if i >= 14 and prices[i-14] > 0 else 0

        doy = dt.timetuple().tm_yday
        f_sin_365 = np.sin(2 * np.pi * doy / 365)
        f_cos_365 = np.cos(2 * np.pi * doy / 365)
        f_sin_182 = np.sin(2 * np.pi * doy / 182.5)
        f_cos_182 = np.cos(2 * np.pi * doy / 182.5)
        f_sin_7 = np.sin(2 * np.pi * dow / 7)
        f_cos_7 = np.cos(2 * np.pi * dow / 7)

        is_friday = int(dow == 4)
        next_gap = (parse_date(dates[i+1]) - dt).days if i + 1 < len(dates) else 1
        is_pre_hol = int(next_gap > 2)
        consec_gap = gap_d
        week_pos = dow / 4 if dow <= 4 else 1.0
        days_left = max(0, 4 - dow)

        window_30 = prices[max(0,i-30):i]
        window_90 = prices[max(0,i-90):i]
        skew_30 = float(scipy_stats.skew(window_30)) if len(window_30) >= 3 else 0
        kurt_30 = float(scipy_stats.kurtosis(window_30)) if len(window_30) >= 3 else 0
        pct_90 = float(scipy_stats.percentileofscore(window_90, p)) / 100 if len(window_90) >= 3 else 0.5
        zscore_30 = (p - a30) / s30 if s30 > 0 else 0

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
            # Price (5)
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

    return np.array(X, dtype=float), np.array(y), fnames, od


def build_features_baseline(records, ctx, target_sp, offset=7, use_smoothed=False):
    """
    68 v6-style features only (no ocean). Used for v8b-baseline comparison.
    """
    return _build_base_features(records, ctx, target_sp, offset, use_smoothed)


def build_features_v8b(records, ctx, target_sp, station_data, offset=7, use_smoothed=False):
    """
    80 features = 68 v6-style + 12 ocean. Drops rows where any ocean feature is None.
    Returns (X, y, fnames, dates).
    """
    X_base, y_base, fnames_base, dates_base = _build_base_features(
        records, ctx, target_sp, offset, use_smoothed
    )

    fnames = fnames_base + OCEAN_FEATURE_NAMES

    dates_sorted = sorted(station_data.keys()) if station_data else []

    # Rebuild the dates list corresponding to X_base rows
    # _build_base_features starts at index 90 in records
    prices = np.array([r["price"] for r in records])
    rec_dates = [r["date"] for r in records]

    X_v8b, y_v8b, od_v8b = [], [], []
    for row_idx, (x_row, y_val, date_str) in enumerate(zip(X_base, y_base, dates_base)):
        ocean_feats = get_ocean_features(station_data, date_str, dates_sorted)
        # Drop row if any ocean feature is None
        if any(v is None for v in ocean_feats):
            continue
        X_v8b.append(list(x_row) + ocean_feats)
        y_v8b.append(y_val)
        od_v8b.append(date_str)

    if not X_v8b:
        return np.array([]), np.array([]), fnames, []

    return np.array(X_v8b, dtype=float), np.array(y_v8b), fnames, od_v8b


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


def backtest(X, y, fnames, prices_raw, species, horizon, method="vmd", n_splits=5):
    n = len(X)
    min_train = int(n * 0.5)
    step = (n - min_train) // n_splits
    if step < 10 or min_train < 100: return None, None

    all_preds, all_actuals, all_prev = [], [], []
    last_model = None

    X_num = X.astype(float)
    # price_lag1 is at column index 11 (same position in both 68 and 80 feature sets)
    PRICE_LAG1_IDX = 11

    for s in range(n_splits):
        te = min_train + s * step
        te_end = min(te + step, n)
        if te_end <= te: continue

        if method == "vmd":
            try:
                modes = decompose_vmd(y[:te], K=3)
            except:
                modes = [y[:te]]
            combined = np.zeros(te_end - te)
            for mode in modes:
                m = mode[:te] if len(mode) >= te else np.pad(mode, (0, te - len(mode)), mode="edge")
                pred, last_model = train_lgbm(X_num[:te], m, X_num[te:te_end])
                combined += pred
            all_preds.extend(combined)
        elif method == "ensemble":
            lgbm_pred, last_model = train_lgbm(X_num[:te], y[:te], X_num[te:te_end])
            arima_preds = []
            for t in range(te, te_end):
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    m = ARIMA(prices_raw[max(0,t-365):t], order=(2,1,2)).fit()
                    arima_preds.append(m.forecast(steps=horizon)[-1])
                except:
                    arima_preds.append(prices_raw[t-1])
            combined = 0.6 * lgbm_pred + 0.4 * np.array(arima_preds)
            all_preds.extend(combined)
        else:
            pred, last_model = train_lgbm(X_num[:te], y[:te], X_num[te:te_end])
            all_preds.extend(pred)

        all_actuals.extend(y[te:te_end])
        all_prev.extend(X_num[te:te_end, PRICE_LAG1_IDX])

    if not all_preds: return None, None
    P, A, Pr = np.array(all_preds), np.array(all_actuals), np.array(all_prev)

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

    result = {"species": species, "model": f"v8b-{method}", "horizon": horizon,
              "mape": round(mape, 2), "rmse": round(rmse), "mae": round(mae),
              "dir_acc": round(dir_acc, 1), "n_tests": len(P), "importance": imp}
    return result, imp


# ── Main ────────────────────────────────────────────────────────────

def main():
    data = load_all()
    n = len(data["trade_date"])
    ctx = build_supply_context(data, n)
    print(f"Supply context: {len(ctx['dates'])} trading days\n")

    ocean_data = load_ocean_data()
    print()

    all_results_v8b = []
    all_results_baseline = []
    all_importance_v8b = {}
    all_importance_baseline = {}

    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        method = cfg.get("method", "vmd")
        station = cfg["ocean_station"]
        station_data = build_ocean_lookup(ocean_data, station)

        print(f"{'='*70}")
        print(f"  {cfg['label']} — {method}, station={station}")
        print(f"{'='*70}")

        # Extract all records first (full history for supply context computation)
        records_all = extract_records(data, n, cfg)

        # ── Filter to 2020+ only ──────────────────────────────────────
        cutoff_dt = parse_date(CUTOFF_DATE)
        records = [r for r in records_all if parse_date(r["date"]) >= cutoff_dt]

        if len(records) < 200:
            print(f"  SKIP — only {len(records)} days from 2020+ (need >= 200)\n")
            continue

        prices_raw = np.array([r["price"] for r in records])
        print(f"  {len(records)} days from 2020+ (out of {len(records_all)} total) | mean={np.mean(prices_raw):,.0f}")

        if cfg.get("regime_split"):
            for months, tag, label in [({11,12,1,2}, "winter", "IN-SEASON"), ({3,4,5,6,7,8,9,10}, "other", "OFF-SEASON")]:
                recs = [r for r in records if parse_date(r["date"]).month in months]
                if len(recs) < 100:
                    print(f"  SKIP {label} — only {len(recs)} days")
                    continue

                rp = np.array([r["price"] for r in recs])
                sp_key = f"{sp}_{tag}"

                # v8b (with ocean, drop None rows)
                X_v8b, y_v8b, fnames_v8b, dates_v8b = build_features_v8b(
                    recs, ctx, sp, station_data, 7, cfg.get("smoothed", False))
                if len(X_v8b) < 100:
                    print(f"  SKIP {label} v8b — only {len(X_v8b)} rows after dropping None ocean")
                    continue
                r_v8b, imp_v8b = backtest(X_v8b, y_v8b, fnames_v8b, rp, sp_key, 7, method)
                if r_v8b:
                    all_results_v8b.append(r_v8b)
                    all_importance_v8b[sp_key] = imp_v8b
                    print(f"\n  {label} v8b  7d: MAPE={r_v8b['mape']:.1f}%  dir={r_v8b['dir_acc']:.1f}%  n={len(X_v8b)}")
                    for feat, v in list(imp_v8b.items())[:5]:
                        print(f"    {feat:<30} {v:>6.2f}%")

                # baseline (no ocean, same 2020+ data)
                X_bl, y_bl, fnames_bl, dates_bl = build_features_baseline(
                    recs, ctx, sp, 7, cfg.get("smoothed", False))
                if len(X_bl) < 100:
                    continue
                r_bl, imp_bl = backtest(X_bl, y_bl, fnames_bl, rp, sp_key, 7, method)
                if r_bl:
                    all_results_baseline.append(r_bl)
                    all_importance_baseline[sp_key] = imp_bl
                    print(f"  {label} base 7d: MAPE={r_bl['mape']:.1f}%  dir={r_bl['dir_acc']:.1f}%  n={len(X_bl)}")
        else:
            rp = prices_raw
            sp_key = sp

            # v8b (with ocean, drop None rows)
            X_v8b, y_v8b, fnames_v8b, dates_v8b = build_features_v8b(
                records, ctx, sp, station_data, 7, cfg.get("smoothed", False))
            if len(X_v8b) < 200:
                print(f"  SKIP v8b — only {len(X_v8b)} rows after dropping None ocean\n")
            else:
                r_v8b, imp_v8b = backtest(X_v8b, y_v8b, fnames_v8b, rp, sp_key, 7, method)
                if r_v8b:
                    all_results_v8b.append(r_v8b)
                    all_importance_v8b[sp_key] = imp_v8b
                    print(f"\n  v8b  7d: MAPE={r_v8b['mape']:.1f}%  dir={r_v8b['dir_acc']:.1f}%  n={len(X_v8b)}")
                    for feat, v in list(imp_v8b.items())[:5]:
                        print(f"    {feat:<30} {v:>6.2f}%")

            # baseline (no ocean, same 2020+ data)
            X_bl, y_bl, fnames_bl, dates_bl = build_features_baseline(
                records, ctx, sp, 7, cfg.get("smoothed", False))
            if len(X_bl) >= 200:
                r_bl, imp_bl = backtest(X_bl, y_bl, fnames_bl, rp, sp_key, 7, method)
                if r_bl:
                    all_results_baseline.append(r_bl)
                    all_importance_baseline[sp_key] = imp_bl
                    print(f"  base 7d: MAPE={r_bl['mape']:.1f}%  dir={r_bl['dir_acc']:.1f}%  n={len(X_bl)}")
        print()

    # ── Comparison A: v8b (2020+, ocean) vs v6 (full history) ────────
    print("\n" + "=" * 80)
    print("COMPARISON A: v8b (2020+ with ocean) vs v6 (full history)")
    print("=" * 80)

    v6_path = OUTPUT_DIR / "poc_v6_results.json"
    v6_data = {}
    if v6_path.exists():
        with open(v6_path) as f:
            for sp_key, info in json.load(f).get("summary", {}).items():
                v6_data[sp_key] = info.get("v6")

    ocean_feat_set = set(OCEAN_FEATURE_NAMES)
    ocean_contrib_v8b = {}
    for sp_key, imp in all_importance_v8b.items():
        ocean_contrib_v8b[sp_key] = sum(imp.get(f, 0) for f in ocean_feat_set)

    print(f"\n  {'Species':<25} {'v6 MAPE':>8} {'v8b MAPE':>9} {'Δ%':>8} {'v8b Dir%':>9} {'Ocean%':>8}")
    print(f"  {'-'*72}")
    summary_a = {}
    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        if cfg.get("regime_split"):
            v8b_r = next((r for r in all_results_v8b if r["species"] == f"{sp}_winter"), None)
            label = f"{sp} (winter)"
            sp_key = f"{sp}_winter"
        else:
            v8b_r = next((r for r in all_results_v8b if r["species"] == sp), None)
            label = sp
            sp_key = sp
        if not v8b_r: continue

        v6_mape = v6_data.get(sp, None)
        if v6_mape is not None:
            delta = f"{(v6_mape - v8b_r['mape']) / v6_mape * 100:+.1f}%"
            v6_str = f"{v6_mape:>7.1f}%"
        else:
            delta = "    —"
            v6_str = "    N/A"

        ocean_pct = ocean_contrib_v8b.get(sp_key, 0)
        print(f"  {label:<25} {v6_str} {v8b_r['mape']:>8.1f}% {delta:>8} {v8b_r['dir_acc']:>8.1f}% {ocean_pct:>7.1f}%")
        summary_a[sp] = {
            "v6_mape": v6_mape,
            "v8b_mape": v8b_r["mape"],
            "dir_acc": v8b_r["dir_acc"],
            "ocean_contribution_pct": round(ocean_contrib_v8b.get(sp_key, 0), 2),
        }

    # ── Comparison B: v8b vs baseline (both on 2020+ data) ───────────
    print("\n" + "=" * 80)
    print("COMPARISON B: v8b (2020+ with ocean) vs baseline (2020+ without ocean)")
    print("Isolates the true ocean feature contribution on the same dataset")
    print("=" * 80)

    baseline_by_sp = {}
    for r in all_results_baseline:
        baseline_by_sp[r["species"]] = r

    print(f"\n  {'Species':<25} {'Base MAPE':>10} {'v8b MAPE':>9} {'Δ%':>8} {'Ocean%':>8} {'Verdict':>10}")
    print(f"  {'-'*75}")
    summary_b = {}
    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        if cfg.get("regime_split"):
            v8b_r = next((r for r in all_results_v8b if r["species"] == f"{sp}_winter"), None)
            bl_r = baseline_by_sp.get(f"{sp}_winter")
            label = f"{sp} (winter)"
            sp_key = f"{sp}_winter"
        else:
            v8b_r = next((r for r in all_results_v8b if r["species"] == sp), None)
            bl_r = baseline_by_sp.get(sp)
            label = sp
            sp_key = sp
        if not v8b_r or not bl_r: continue

        delta_pct = (bl_r["mape"] - v8b_r["mape"]) / bl_r["mape"] * 100
        delta_str = f"{delta_pct:+.1f}%"
        ocean_pct = ocean_contrib_v8b.get(sp_key, 0)
        verdict = "BETTER" if delta_pct > 0 else ("WORSE" if delta_pct < 0 else "SAME")
        print(f"  {label:<25} {bl_r['mape']:>9.1f}% {v8b_r['mape']:>8.1f}% {delta_str:>8} {ocean_pct:>7.1f}% {verdict:>10}")
        summary_b[sp] = {
            "baseline_mape": bl_r["mape"],
            "v8b_mape": v8b_r["mape"],
            "delta_pct": round(delta_pct, 2),
            "ocean_contribution_pct": round(ocean_pct, 2),
            "verdict": verdict,
            "n_v8b": v8b_r["n_tests"],
            "n_baseline": bl_r["n_tests"],
        }

    # ── Ocean Feature Importance ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("OCEAN FEATURE IMPORTANCE (v8b, 2020+ only)")
    print("=" * 80)
    print(f"\n  {'Species':<20} {'OceanTotal%':>12} {'Top Ocean Feature':<32} {'%':>6}")
    print(f"  {'-'*74}")
    for sp_key, imp in all_importance_v8b.items():
        ocean_total = ocean_contrib_v8b.get(sp_key, 0)
        top_ocean = max(((f, imp.get(f, 0)) for f in ocean_feat_set), key=lambda x: x[1])
        print(f"  {sp_key:<20} {ocean_total:>11.2f}% {top_ocean[0]:<32} {top_ocean[1]:>5.2f}%")

    avg_ocean = np.mean(list(ocean_contrib_v8b.values())) if ocean_contrib_v8b else 0
    print(f"\n  Average ocean contribution: {avg_ocean:.2f}%")

    # ── Save ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_at": datetime.now().isoformat(),
        "description": "v8b: ocean features on 2020+ data only; fair A/B test vs baseline",
        "cutoff_date": CUTOFF_DATE,
        "total_features_v8b": 80,
        "total_features_baseline": 68,
        "ocean_features": OCEAN_FEATURE_NAMES,
        "results_v8b": all_results_v8b,
        "results_baseline": all_results_baseline,
        "feature_importance_v8b": {k: dict(list(v.items())[:25]) for k, v in all_importance_v8b.items()},
        "ocean_contribution_v8b": ocean_contrib_v8b,
        "comparison_a_v8b_vs_v6_fullhistory": summary_a,
        "comparison_b_v8b_vs_baseline_2020plus": summary_b,
    }
    outpath = OUTPUT_DIR / "poc_v8b_results.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
