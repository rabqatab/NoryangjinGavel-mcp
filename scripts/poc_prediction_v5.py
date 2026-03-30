"""
PoC v5: VMD Signal Decomposition + LightGBM Ensemble.

Key improvements over v4:
  1. VMD (Variational Mode Decomposition) decomposes price into trend + modes
  2. Separate LightGBM model per VMD mode → recombine predictions
  3. ARIMA+LightGBM ensemble for 우럭 (where ARIMA beat LightGBM)
  4. Regime-aware features for seasonal species

CPU-only. GPU models (LSTM) require Docker — see docker/ directory.

Usage:
    uv run python scripts/poc_prediction_v5.py
"""
import json
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
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
     "smoothed": False, "label": "넙치 (flatfish)", "method": "vmd"},
    {"species": "우럭", "state": "활", "pkg": "kg", "spec": "중", "domestic": False,
     "smoothed": False, "label": "우럭 (rockfish)", "method": "ensemble"},
    {"species": "방어", "state": "선", "pkg": "kg", "spec": "중", "domestic": True,
     "smoothed": True, "label": "방어 (yellowtail)", "method": "vmd", "regime_split": True},
    {"species": "참돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True,
     "smoothed": False, "label": "참돔 (seabream)", "method": "vmd"},
    {"species": "농어", "state": "활", "pkg": "kg", "spec": "중", "domestic": True,
     "smoothed": False, "label": "농어 (sea bass)", "method": "vmd"},
    {"species": "도다리", "state": "활", "pkg": "kg", "spec": "중", "domestic": False,
     "smoothed": True, "label": "도다리 (flounder)", "method": "vmd"},
    {"species": "감성돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True,
     "smoothed": False, "label": "감성돔 (black porgy)", "method": "vmd"},
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


# ── VMD Decomposition ───────────────────────────────────────────────

def decompose_vmd(prices: np.ndarray, K: int = 3, alpha: int = 2000) -> list[np.ndarray]:
    """Decompose price series into K modes using VMD."""
    try:
        u, _, _ = VMD(prices, alpha, 0, K, 0, 1, 1e-7)
        return [u[k] for k in range(K)]
    except Exception:
        # Fallback: simple trend + residual
        from numpy.polynomial import polynomial as P
        trend = np.convolve(prices, np.ones(30)/30, mode="same")
        residual = prices - trend
        return [trend, residual]


# ── Data Loading ────────────────────────────────────────────────────

def load_all():
    import pyarrow.dataset as ds
    print("Loading data...", end=" ", flush=True)
    dataset = ds.dataset(str(DATA_ROOT), format="parquet", partitioning="hive")
    cols = ["trade_date", "species", "state", "origin", "spec", "packaging", "price_avg", "quantity"]
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
        d = data["trade_date"][i]
        di = date_idx[d]
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
    day_data = defaultdict(lambda: {"prices": [], "origins": set(), "qty": 0})
    for i in range(n):
        if data["species"][i] != cfg["species"]: continue
        if data["state"][i] != cfg["state"]: continue
        if data["packaging"][i] != cfg["pkg"]: continue
        if data["spec"][i] != cfg["spec"]: continue
        if cfg["domestic"] and is_foreign(data["origin"][i]): continue
        d = data["trade_date"][i]
        day_data[d]["prices"].append(data["price_avg"][i])
        if data["origin"][i]: day_data[d]["origins"].add(data["origin"][i])
        day_data[d]["qty"] += data["quantity"][i]
    return [{
        "date": d, "price": float(np.mean(dd["prices"])),
        "n_lots": len(dd["prices"]), "n_origins": len(dd["origins"]), "qty": dd["qty"],
    } for d, dd in sorted(day_data.items())]


# ── Feature Building (v4 features reused) ───────────────────────────

def build_features(records, ctx, target_sp, offset=7, use_smoothed=False):
    prices = [r["price"] for r in records]
    dates = [r["date"] for r in records]
    di_map = ctx["date_idx"]

    targets = np.convolve(prices, np.ones(7)/7, mode="same").tolist() if use_smoothed and len(prices) > 7 else prices

    monthly_avg = defaultdict(list)
    for r in records:
        monthly_avg[parse_date(r["date"]).month].append(r["price"])
    monthly_avg = {m: np.mean(v) for m, v in monthly_avg.items()}

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
    ]

    X, y, od = [], [], []
    for i in range(30, len(records) - offset):
        dt = parse_date(dates[i])
        di = di_map.get(dates[i], 0)
        hol = days_to_holiday(dt)
        dow = dt.weekday()
        p = prices[i]
        p1 = prices[i-1] if i >= 1 else p
        p7 = prices[i-7] if i >= 7 else p1
        p30 = prices[i-30] if i >= 30 else p1
        a7 = np.mean(prices[max(0,i-7):i])
        a30 = np.mean(prices[max(0,i-30):i])
        s7 = np.std(prices[max(0,i-7):i])
        s30 = np.std(prices[max(0,i-30):i])
        r7 = max(prices[max(0,i-7):i]) - min(prices[max(0,i-7):i])

        oq7 = ctx["sp_qty_7d"][target_sp][di]
        ol7 = ctx["sp_lots_7d"][target_sp][di]
        oq30 = np.mean(ctx["sp_qty"][target_sp][max(0,di-30):di]) if di >= 1 else oq7
        oqr = oq7 / oq30 if oq30 > 0 else 1
        oqc = (ctx["sp_qty_7d"][target_sp][di] - ctx["sp_qty_7d"][target_sp][max(0,di-7)]) / max(ctx["sp_qty_7d"][target_sp][max(0,di-7)], 1)
        olc = (ctx["sp_lots_7d"][target_sp][di] - ctx["sp_lots_7d"][target_sp][max(0,di-7)]) / max(ctx["sp_lots_7d"][target_sp][max(0,di-7)], 1)
        otq = ctx["total_sashimi_7d"][di] - ctx["sp_qty_7d"][target_sp][di]
        ml7 = ctx["market_lots_7d"][di]
        ts = ctx["total_sashimi"][di]
        con = ctx["sp_qty"][target_sp][di] / ts if ts > 0 else 0
        tsc = (ctx["total_sashimi_7d"][di] - ctx["total_sashimi_7d"][max(0,di-7)]) / max(ctx["total_sashimi_7d"][max(0,di-7)], 1)
        mc = (ctx["market_lots_7d"][di] - ctx["market_lots_7d"][max(0,di-7)]) / max(ctx["market_lots_7d"][max(0,di-7)], 1)
        pvm = p / monthly_avg.get(dt.month, p) if monthly_avg.get(dt.month, p) > 0 else 1
        gap = (dt - parse_date(dates[i-1])).days if i > 0 else 1
        ld = int(ol7 < ctx["sp_lots_7d"][target_sp][max(0,di-14)] * 0.5) if di >= 14 else 0
        qd = int(oq7 < oq30 * 0.5) if oq30 > 0 else 0

        X.append([
            dow, dt.month, dt.day, int(dow >= 5), dt.isocalendar()[1], (dt.month-1)//3+1, int(dow==0),
            hol["seollal"], hol["chuseok"], abs(hol["seollal"]), abs(hol["chuseok"]),
            p, p1, p7, a7, a30,
            (p-p1)/p1*100 if p1>0 else 0, (p-p7)/p7*100 if p7>0 else 0,
            (p-p30)/p30*100 if p30>0 else 0, a7/a30-1 if a30>0 else 0,
            s7, s30, r7,
            oq7, ol7, oqr, oqc, olc,
            otq, ml7, con, tsc, mc,
            pvm, np.sin(2*np.pi*dt.month/12), np.cos(2*np.pi*dt.month/12), int(dt.month in [11,12,1,2]),
            gap, ld, qd, ld+qd+int(gap>3),
        ])
        y.append(targets[i + offset])
        od.append(dates[i])

    return np.array(X), np.array(y), fnames, od


# ── VMD + LightGBM Pipeline ────────────────────────────────────────

def train_lgbm(X_tr, y_tr, X_te):
    params = {
        "objective": "regression", "metric": "mae",
        "learning_rate": 0.03, "num_leaves": 31,
        "min_child_samples": 20, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5,
        "reg_alpha": 0.1, "reg_lambda": 0.1,
        "verbose": -1, "n_jobs": 1,
    }
    model = lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=1000)
    return model.predict(X_te), model


def backtest_vmd_lgbm(X, y, fnames, prices_raw, species, horizon, n_splits=5, K=3):
    """VMD decomposition → per-mode LightGBM → recombine."""
    n = len(X)
    min_train = int(n * 0.5)
    step = (n - min_train) // n_splits
    if step < 10 or min_train < 100: return None

    all_preds, all_actuals, all_prev = [], [], []

    for s in range(n_splits):
        te = min_train + s * step
        te_end = min(te + step, n)
        if te_end <= te: continue

        # Decompose training target via VMD
        y_train = y[:te]
        try:
            modes = decompose_vmd(y_train, K=K)
        except Exception:
            modes = [y_train]

        # Train one LightGBM per mode
        combined_pred = np.zeros(te_end - te)
        for mode in modes:
            # Align mode length to training
            m = mode[:te] if len(mode) >= te else np.pad(mode, (0, te - len(mode)), mode="edge")
            pred, _ = train_lgbm(X[:te], m, X[te:te_end])
            combined_pred += pred

        all_preds.extend(combined_pred)
        all_actuals.extend(y[te:te_end])
        all_prev.extend(X[te:te_end, 11])

    if not all_preds: return None
    P, A, Pr = np.array(all_preds), np.array(all_actuals), np.array(all_prev)

    mape = float(np.mean(np.abs(P - A) / np.where(A > 0, A, 1))) * 100
    rmse = float(np.sqrt(mean_squared_error(A, P)))
    mae = float(mean_absolute_error(A, P))
    dir_acc = float(np.mean((A > Pr) == (P > Pr))) * 100

    return {"species": species, "model": "VMD+LightGBM", "horizon": horizon,
            "mape": round(mape, 2), "rmse": round(rmse), "mae": round(mae),
            "dir_acc": round(dir_acc, 1), "n_tests": len(P)}


def backtest_ensemble(X, y, fnames, prices_raw, species, horizon, n_splits=5):
    """ARIMA + LightGBM ensemble (weighted average)."""
    n = len(X)
    min_train = int(n * 0.5)
    step = (n - min_train) // n_splits
    if step < 10 or min_train < 100: return None

    all_preds, all_actuals, all_prev = [], [], []

    for s in range(n_splits):
        te = min_train + s * step
        te_end = min(te + step, n)
        if te_end <= te: continue

        # LightGBM predictions
        lgbm_pred, _ = train_lgbm(X[:te], y[:te], X[te:te_end])

        # ARIMA predictions (on raw price series)
        arima_preds = []
        for t in range(te, te_end):
            try:
                model = ARIMAModel(prices_raw[max(0, t-365):t], order=(2, 1, 2)).fit()
                fc = model.forecast(steps=horizon)
                arima_preds.append(fc[-1])
            except Exception:
                arima_preds.append(prices_raw[t-1])
        arima_pred = np.array(arima_preds)

        # Weighted ensemble: 60% LightGBM + 40% ARIMA (ARIMA is better for 우럭)
        combined = 0.6 * lgbm_pred + 0.4 * arima_pred

        all_preds.extend(combined)
        all_actuals.extend(y[te:te_end])
        all_prev.extend(X[te:te_end, 11])

    if not all_preds: return None
    P, A, Pr = np.array(all_preds), np.array(all_actuals), np.array(all_prev)

    mape = float(np.mean(np.abs(P - A) / np.where(A > 0, A, 1))) * 100
    rmse = float(np.sqrt(mean_squared_error(A, P)))
    dir_acc = float(np.mean((A > Pr) == (P > Pr))) * 100

    return {"species": species, "model": "ARIMA+LightGBM", "horizon": horizon,
            "mape": round(mape, 2), "rmse": round(rmse),
            "dir_acc": round(dir_acc, 1), "n_tests": len(P)}


def backtest_plain_lgbm(X, y, fnames, species, horizon, n_splits=5):
    """Plain LightGBM (v4 baseline for comparison)."""
    n = len(X)
    min_train = int(n * 0.5)
    step = (n - min_train) // n_splits
    if step < 10 or min_train < 100: return None

    all_preds, all_actuals, all_prev = [], [], []
    last_model = None

    for s in range(n_splits):
        te = min_train + s * step
        te_end = min(te + step, n)
        if te_end <= te: continue
        pred, last_model = train_lgbm(X[:te], y[:te], X[te:te_end])
        all_preds.extend(pred)
        all_actuals.extend(y[te:te_end])
        all_prev.extend(X[te:te_end, 11])

    if not all_preds: return None
    P, A, Pr = np.array(all_preds), np.array(all_actuals), np.array(all_prev)

    mape = float(np.mean(np.abs(P - A) / np.where(A > 0, A, 1))) * 100
    rmse = float(np.sqrt(mean_squared_error(A, P)))
    dir_acc = float(np.mean((A > Pr) == (P > Pr))) * 100

    imp = dict(zip(fnames, last_model.feature_importance(importance_type="gain")))
    total = sum(imp.values())
    imp = {k: round(v/total*100, 1) for k, v in sorted(imp.items(), key=lambda x: -x[1])} if total else {}

    return {"species": species, "model": "LightGBM-v4", "horizon": horizon,
            "mape": round(mape, 2), "rmse": round(rmse),
            "dir_acc": round(dir_acc, 1), "n_tests": len(P), "importance": imp}


# ── Main ────────────────────────────────────────────────────────────

def main():
    data = load_all()
    n = len(data["trade_date"])
    ctx = build_supply_context(data, n)
    print(f"Supply context: {len(ctx['dates'])} days\n")

    all_results = []

    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        method = cfg.get("method", "vmd")
        print(f"{'='*70}")
        print(f"  {cfg['label']} — method: {method}")
        print(f"{'='*70}")

        records = extract_records(data, n, cfg)
        if len(records) < 200:
            print(f"  SKIP — {len(records)} days\n"); continue

        prices_raw = np.array([r["price"] for r in records])
        print(f"  {len(records)} days | mean={np.mean(prices_raw):,.0f}")

        # Handle regime split
        if cfg.get("regime_split"):
            regimes = [
                ({11,12,1,2}, "winter", "IN-SEASON"),
                ({3,4,5,6,7,8,9,10}, "other", "OFF-SEASON"),
            ]
        else:
            regimes = [(None, "", "")]

        for months, regime_tag, regime_label in regimes:
            if months:
                r_records = [r for r in records if parse_date(r["date"]).month in months]
                label = f"{sp}_{regime_tag}"
                if len(r_records) < 100:
                    print(f"  {regime_label}: skip ({len(r_records)} records)"); continue
            else:
                r_records = records
                label = sp

            for horizon in [7]:
                X, y, fnames, dates = build_features(r_records, ctx, sp, horizon, cfg.get("smoothed", False))
                if len(X) < 200 and not months:
                    print(f"  {horizon}d: too few samples"); continue
                if len(X) < 100: continue

                r_prices = np.array([r["price"] for r in r_records])

                # v4 baseline
                v4 = backtest_plain_lgbm(X, y, fnames, label, horizon)

                # Method-specific
                if method == "vmd":
                    v5 = backtest_vmd_lgbm(X, y, fnames, r_prices, label, horizon, K=3)
                elif method == "ensemble":
                    v5 = backtest_ensemble(X, y, fnames, r_prices, label, horizon)
                else:
                    v5 = v4

                prefix = f"  {regime_label + ' ' if regime_label else ''}{horizon}d:"
                if v4:
                    all_results.append(v4)
                    print(f"{prefix} v4-LightGBM  MAPE={v4['mape']:.1f}%  dir={v4['dir_acc']:.1f}%")
                if v5 and v5 != v4:
                    all_results.append(v5)
                    improv = (v4['mape'] - v5['mape']) / v4['mape'] * 100 if v4 else 0
                    print(f"{prefix} v5-{v5['model']:<16} MAPE={v5['mape']:.1f}%  dir={v5['dir_acc']:.1f}%  ({improv:+.0f}% vs v4)")

        print()

    # Summary
    print("\n" + "=" * 80)
    print("v4 vs v5 COMPARISON (7-day horizon)")
    print("=" * 80)

    # Load previous
    prev_v4 = {}
    v4p = OUTPUT_DIR / "poc_v4_results.json"
    if v4p.exists():
        with open(v4p) as f:
            for sp, info in json.load(f).get("summary", {}).items():
                prev_v4[sp] = info.get("v4_mape")

    print(f"\n  {'Species':<25} {'v4 prev':>8} {'v4 now':>8} {'v5':>8} {'Δ':>8} {'v5 Dir%':>7}")
    print(f"  {'-'*66}")

    summary = {}
    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        if cfg.get("regime_split"):
            v4r = next((r for r in all_results if r["species"] == f"{sp}_winter" and "v4" in r["model"]), None)
            v5r = next((r for r in all_results if r["species"] == f"{sp}_winter" and "v4" not in r["model"]), None)
            label = f"{sp} (winter)"
        else:
            v4r = next((r for r in all_results if r["species"] == sp and "v4" in r["model"] and r["horizon"] == 7), None)
            v5r = next((r for r in all_results if r["species"] == sp and "v4" not in r["model"] and r["horizon"] == 7), None)
            label = sp

        if not v4r: continue
        v4_prev = prev_v4.get(sp, v4r["mape"])
        v5_mape = v5r["mape"] if v5r else v4r["mape"]
        v5_dir = v5r["dir_acc"] if v5r else v4r["dir_acc"]
        delta = f"{(v4r['mape'] - v5_mape) / v4r['mape'] * 100:+.0f}%" if v5r else "—"

        print(f"  {label:<25} {v4_prev:>7.1f}% {v4r['mape']:>7.1f}% {v5_mape:>7.1f}% {delta:>8} {v5_dir:>6.1f}%")
        summary[sp] = {"v4": v4r["mape"], "v5": v5_mape, "v5_model": v5r["model"] if v5r else "—",
                       "dir_acc": v5_dir}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_at": datetime.now().isoformat(),
        "models": {
            "VMD+LightGBM": "VMD decomposition into 3 modes, separate LightGBM per mode, recombine",
            "ARIMA+LightGBM": "60% LightGBM + 40% ARIMA ensemble (for species where ARIMA excels)",
        },
        "results": all_results,
        "summary": summary,
    }
    with open(OUTPUT_DIR / "poc_v5_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'poc_v5_results.json'}")


if __name__ == "__main__":
    main()
