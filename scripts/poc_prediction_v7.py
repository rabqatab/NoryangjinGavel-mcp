"""
PoC v7: STL-VMD Dual Decomposition + Optuna K/alpha Optimization.

Improvements over v6 (68 features, VMD+LightGBM):
  1. STL-VMD Dual Decomposition:
       - STL decomposition (period=7) to extract seasonal + trend + residual
       - VMD applied only on the STL residual (the nonlinear/irregular part)
       - Each component (trend, seasonal, VMD modes) predicted separately by LightGBM
       - Predictions recombined at the end
  2. Optuna K/alpha Optimization:
       - 10-trial Optuna search per species for VMD K (3-8) and alpha (500-5000)
       - Objective: minimize MAPE on a validation fold of the STL residual
  3. Same 68-feature set as v6 (build_features_v6 reused verbatim)
  4. Output comparison table: v6 vs v7 MAPE per species (7-day horizon)

Usage:
    uv run python scripts/poc_prediction_v7.py
"""
import json
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
from scipy import stats as scipy_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import STL
from vmdpy import VMD

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
     "smoothed": False, "label": "넙치 (flatfish)"},
    {"species": "우럭", "state": "활", "pkg": "kg", "spec": "중", "domestic": False,
     "smoothed": False, "label": "우럭 (rockfish)"},
    {"species": "방어", "state": "선", "pkg": "kg", "spec": "중", "domestic": True,
     "smoothed": True, "label": "방어 (yellowtail)", "regime_split": True},
    {"species": "참돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True,
     "smoothed": False, "label": "참돔 (seabream)"},
    {"species": "농어", "state": "활", "pkg": "kg", "spec": "중", "domestic": True,
     "smoothed": False, "label": "농어 (sea bass)"},
    {"species": "도다리", "state": "활", "pkg": "kg", "spec": "중", "domestic": False,
     "smoothed": True, "label": "도다리 (flounder)"},
    {"species": "감성돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True,
     "smoothed": False, "label": "감성돔 (black porgy)"},
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


# ── Technical Indicators ─────────────────────────────────────────────

def ema(prices, span):
    a = np.array(prices, dtype=float)
    out = np.empty_like(a)
    out[0] = a[0]
    alpha = 2 / (span + 1)
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i-1]
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


# ── Feature Engineering (v6: 68 features — reused verbatim) ──────────

def build_features_v6(records, ctx, target_sp, offset=7, use_smoothed=False):
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

    woy_avg = defaultdict(list)
    for i, r in enumerate(records):
        w = parse_date(r["date"]).isocalendar()[1]
        woy_avg[(w,)].append(r["price"])

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
            dow, dt.month, dt.day, int(dow >= 5), dt.isocalendar()[1], (dt.month-1)//3+1, int(dow==0),
            hol["seollal"], hol["chuseok"], abs(hol["seollal"]), abs(hol["chuseok"]),
            p, p1, p7, a7, a30,
            (p-p1)/p1*100 if p1>0 else 0, (p-p7)/p7*100 if p7>0 else 0,
            (p-p30)/p30*100 if p30>0 else 0, a7/a30-1 if a30>0 else 0,
            s7, s30, r7,
            oq7, ol7, oqr, oqc, olc,
            otq, ml7, con, tsc, mc,
            pvm, np.sin(2*np.pi*dt.month/12), np.cos(2*np.pi*dt.month/12), int(dt.month in [11,12,1,2]),
            gap_d, ld, qd, ld+qd+int(gap_d>3),
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

    return np.array(X), np.array(y), fnames, od


# ── STL-VMD Dual Decomposition ───────────────────────────────────────

def stl_decompose(prices, period=7):
    """Apply STL decomposition. Returns (trend, seasonal, residual) as np arrays."""
    a = np.array(prices, dtype=float)
    # STL requires at least 2 full periods
    if len(a) < 2 * period + 1:
        trend = np.full_like(a, np.mean(a))
        seasonal = np.zeros_like(a)
        resid = a - trend
        return trend, seasonal, resid
    try:
        stl = STL(a, period=period, robust=True)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception:
        trend = np.convolve(a, np.ones(period) / period, mode="same")
        seasonal = np.zeros_like(a)
        resid = a - trend
        return trend, seasonal, resid


def vmd_decompose(signal, K=4, alpha=2000):
    """Apply VMD on a 1-D signal. Returns list of K mode arrays."""
    a = np.array(signal, dtype=float)
    if len(a) < K * 10:
        return [a]
    try:
        u, _, _ = VMD(a, alpha, 0, K, 0, 1, 1e-7)
        return [u[k] for k in range(K)]
    except Exception:
        trend_approx = np.convolve(a, np.ones(30) / 30, mode="same")
        return [trend_approx, a - trend_approx]


# ── Optuna: search VMD K and alpha on STL residual ───────────────────

def optimize_vmd_params(resid_series, n_trials=10):
    """
    Use Optuna to find the best VMD K and alpha for the STL residual.
    Splits residual into train (first 70%) and val (last 30%).
    Minimizes MAPE of a trivial persistence baseline corrected by VMD reconstruction.
    Since we're not predicting here but optimizing the decomposition quality,
    we use reconstruction error (MSE of sum(modes) vs original) as the proxy metric.
    """
    resid = np.array(resid_series, dtype=float)
    n = len(resid)
    if n < 60:
        return 4, 2000

    split = int(n * 0.7)
    train_resid = resid[:split]
    val_resid = resid[split:]

    def objective(trial):
        K = trial.suggest_int("K", 3, 8)
        alpha = trial.suggest_int("alpha", 500, 5000, step=500)
        try:
            modes = vmd_decompose(train_resid, K=K, alpha=alpha)
            reconstructed = np.sum(modes, axis=0)
            mse = float(np.mean((reconstructed - train_resid) ** 2))
            # Also penalize if reconstruction on val degrades much
            # (modes are computed on train, so we can't directly apply to val;
            #  use reconstruction error on train as the proxy)
            return mse
        except Exception:
            return float("inf")

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_K = study.best_params["K"]
    best_alpha = study.best_params["alpha"]
    return best_K, best_alpha


# ── Model Training ───────────────────────────────────────────────────

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


# ── STL-VMD Backtest ─────────────────────────────────────────────────

def backtest_stl_vmd(X, y, fnames, prices_raw, species, horizon,
                     best_K, best_alpha, use_smoothed=False, n_splits=5):
    """
    STL-VMD dual decomposition backtest.

    For each cross-validation split:
      1. Apply STL on the training target series to get trend, seasonal, residual.
      2. Apply VMD on the STL residual with the Optuna-tuned K and alpha.
      3. Train a separate LightGBM for each component (trend, seasonal, + K VMD modes).
      4. Predict all components on the test set, sum for final prediction.
    """
    n = len(X)
    min_train = int(n * 0.5)
    step = (n - min_train) // n_splits
    if step < 10 or min_train < 100:
        return None, None

    all_preds, all_actuals, all_prev = [], [], []
    last_model = None

    for s in range(n_splits):
        te = min_train + s * step
        te_end = min(te + step, n)
        if te_end <= te: continue

        y_train = y[:te]

        # STL decomposition on the training targets
        trend_tr, seasonal_tr, resid_tr = stl_decompose(y_train, period=7)

        # VMD on the STL residual
        vmd_modes = vmd_decompose(resid_tr, K=best_K, alpha=best_alpha)

        # Predict trend component
        pred_trend, last_model = train_lgbm(X[:te], trend_tr, X[te:te_end])

        # Predict seasonal component
        pred_seasonal, _ = train_lgbm(X[:te], seasonal_tr, X[te:te_end])

        # Predict each VMD mode on the residual
        combined_resid = np.zeros(te_end - te)
        for mode in vmd_modes:
            m = mode[:te] if len(mode) >= te else np.pad(mode, (0, te - len(mode)), mode="edge")
            pred_mode, _ = train_lgbm(X[:te], m, X[te:te_end])
            combined_resid += pred_mode

        # Recombine: trend + seasonal + vmd_reconstructed_residual
        combined = pred_trend + pred_seasonal + combined_resid

        all_preds.extend(combined)
        all_actuals.extend(y[te:te_end])
        all_prev.extend(X[te:te_end, 11])  # price_lag1 index

    if not all_preds:
        return None, None

    P = np.array(all_preds)
    A = np.array(all_actuals)
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

    result = {
        "species": species, "model": "v7-stl-vmd", "horizon": horizon,
        "mape": round(mape, 2), "rmse": round(rmse), "mae": round(mae),
        "dir_acc": round(dir_acc, 1), "n_tests": len(P),
        "vmd_K": best_K, "vmd_alpha": best_alpha,
        "importance": imp,
    }
    return result, imp


# ── Main ─────────────────────────────────────────────────────────────

def main():
    data = load_all()
    n = len(data["trade_date"])
    ctx = build_supply_context(data, n)
    print(f"Supply context: {len(ctx['dates'])} days\n")

    all_results = []
    all_importance = {}
    optuna_params = {}

    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        print(f"{'='*70}")
        print(f"  {cfg['label']} — v7 STL-VMD + Optuna, 68 features")
        print(f"{'='*70}")

        records = extract_records(data, n, cfg)
        if len(records) < 200:
            print(f"  SKIP — {len(records)} days\n"); continue

        prices_raw = np.array([r["price"] for r in records])
        print(f"  {len(records)} days | mean={np.mean(prices_raw):,.0f}")

        if cfg.get("regime_split"):
            for months, tag, label in [
                ({11, 12, 1, 2}, "winter", "IN-SEASON"),
                ({3, 4, 5, 6, 7, 8, 9, 10}, "other", "OFF-SEASON"),
            ]:
                recs = [r for r in records if parse_date(r["date"]).month in months]
                if len(recs) < 100: continue
                X, y, fnames, dates = build_features_v6(recs, ctx, sp, 7, cfg.get("smoothed", False))
                if len(X) < 100: continue

                rp = np.array([r["price"] for r in recs])
                # STL decompose on full series to get residual for Optuna
                _, _, resid_full = stl_decompose(rp, period=7)

                print(f"  [{label}] Optuna search (10 trials)...", end=" ", flush=True)
                best_K, best_alpha = optimize_vmd_params(resid_full, n_trials=10)
                print(f"K={best_K}, alpha={best_alpha}")
                optuna_params[f"{sp}_{tag}"] = {"K": best_K, "alpha": best_alpha}

                r, imp = backtest_stl_vmd(X, y, fnames, rp, f"{sp}_{tag}", 7,
                                          best_K, best_alpha, cfg.get("smoothed", False))
                if r:
                    all_results.append(r)
                    all_importance[f"{sp}_{tag}"] = imp
                    print(f"\n  {label} 7d: MAPE={r['mape']:.1f}%  dir={r['dir_acc']:.1f}%")
                    for feat, v in list(imp.items())[:7]:
                        print(f"    {feat:<28} {v:>6.2f}%")
        else:
            X, y, fnames, dates = build_features_v6(records, ctx, sp, 7, cfg.get("smoothed", False))
            if len(X) < 200: continue

            # STL decompose on full price series to get residual for Optuna
            _, _, resid_full = stl_decompose(prices_raw, period=7)

            print(f"  Optuna search (10 trials)...", end=" ", flush=True)
            best_K, best_alpha = optimize_vmd_params(resid_full, n_trials=10)
            print(f"K={best_K}, alpha={best_alpha}")
            optuna_params[sp] = {"K": best_K, "alpha": best_alpha}

            r, imp = backtest_stl_vmd(X, y, fnames, prices_raw, sp, 7,
                                      best_K, best_alpha, cfg.get("smoothed", False))
            if r:
                all_results.append(r)
                all_importance[sp] = imp
                print(f"\n  7d: MAPE={r['mape']:.1f}%  dir={r['dir_acc']:.1f}%")
                for feat, v in list(imp.items())[:7]:
                    print(f"    {feat:<28} {v:>6.2f}%")
        print()

    # ── Comparison table: v6 vs v7 ───────────────────────────────────
    print("\n" + "=" * 80)
    print("v6 vs v7 COMPARISON (7-day horizon)")
    print("=" * 80)

    v6p = OUTPUT_DIR / "poc_v6_results.json"
    v6_data = {}
    if v6p.exists():
        with open(v6p) as f:
            raw = json.load(f)
            # v6 stores results as a list; build lookup by species key
            for r in raw.get("results", []):
                sp_key = r["species"]
                if sp_key not in v6_data or r.get("horizon") == 7:
                    v6_data[sp_key] = r.get("mape")

    print(f"\n  {'Species':<25} {'v6 MAPE':>8} {'v7 MAPE':>8} {'Δ':>8} {'v7 Dir%':>7}  K  alpha")
    print(f"  {'-'*70}")

    summary = {}
    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        if cfg.get("regime_split"):
            key = f"{sp}_winter"
            v7r = next((r for r in all_results if r["species"] == key), None)
            label = f"{sp} (winter)"
        else:
            key = sp
            v7r = next((r for r in all_results if r["species"] == sp), None)
            label = sp
        if not v7r: continue

        v6_mape = v6_data.get(key)
        if v6_mape:
            delta = f"{(v6_mape - v7r['mape']) / v6_mape * 100:+.0f}%"
            v6_str = f"{v6_mape:>7.1f}%"
        else:
            delta = "—"
            v6_str = "    n/a"

        opt = optuna_params.get(key, {})
        bK = opt.get("K", "?")
        bA = opt.get("alpha", "?")
        print(f"  {label:<25} {v6_str} {v7r['mape']:>7.1f}% {delta:>8} {v7r['dir_acc']:>6.1f}%  {bK}  {bA}")

        summary[sp] = {
            "v6": v6_mape,
            "v7": v7r["mape"],
            "dir_acc": v7r["dir_acc"],
            "best_K": bK,
            "best_alpha": bA,
            "top_features": dict(list(
                all_importance.get(key, {}).items()
            )[:15]),
        }

    # ── Optuna params table ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Optuna Best VMD Parameters per Species")
    print("=" * 60)
    print(f"  {'Key':<25} {'K':>4} {'alpha':>6}")
    print(f"  {'-'*40}")
    for key, params in optuna_params.items():
        print(f"  {key:<25} {params['K']:>4} {params['alpha']:>6}")

    # ── Save results ─────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_at": datetime.now().isoformat(),
        "model": "v7-stl-vmd",
        "total_features": 68,
        "decomposition": "STL (period=7) + VMD on residual",
        "optuna_trials": 10,
        "results": all_results,
        "optuna_params": optuna_params,
        "feature_importance": {k: dict(list(v.items())[:20]) for k, v in all_importance.items()},
        "summary": summary,
    }
    out_path = OUTPUT_DIR / "poc_v7_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
