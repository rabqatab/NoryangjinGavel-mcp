"""
PoC v3: Feature-Engineered Prediction with Simulated Ocean Weather Features.

Since KHOA/KMA ocean APIs require registration and manual CSV downloads,
this version simulates the effect of ocean weather features using:
  1. Supply-shock detection from our own data (proxy for bad weather)
  2. Calendar seasonality (captures seasonal weather patterns indirectly)
  3. Rolling supply volatility (captures weather disruption patterns)
  4. Lagged quantity-gap detection (no trading = likely storm/holiday)

Also adds 7-day smoothed target for species where daily is too noisy.

When KHOA API keys are obtained, the TODO sections show where to plug in
real wave_height, water_temp, wind_speed features.

Usage:
    uv run python scripts/poc_prediction_v3.py
"""
import json
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "parquet" / "prices"
OUTPUT_DIR = PROJECT_ROOT / "data" / "poc_results"

# ── Species Config ──────────────────────────────────────────────────

FOREIGN_KW = ['일본','중국','미국','러시아','캐나다','노르웨이','뉴질랜드','대만','칠레',
              '아르헨티나','영국','아일랜드','온두라스','북한','(원양)','인도','인도네시아',
              '태국','베트남','필리핀','호주','스페인','네덜란드','페루','모로코','아프리카',
              '파키스탄','라스팔마스','포클랜드','멕시코']

@dataclass
class SpeciesConfig:
    species: str
    state: str
    packaging: str
    spec: str
    domestic_only: bool = False
    label: str = ""
    use_smoothed: bool = False  # Use 7-day smoothed target for noisy species

SPECIES_CONFIGS = [
    SpeciesConfig("넙치", "활", "kg", "중", label="넙치 (flatfish)"),
    SpeciesConfig("우럭", "활", "kg", "중", label="우럭 (rockfish)"),
    SpeciesConfig("방어", "선", "kg", "중", domestic_only=True, label="방어 (yellowtail)", use_smoothed=True),
    SpeciesConfig("참돔", "활", "kg", "중", domestic_only=True, label="참돔 (seabream)"),
    SpeciesConfig("농어", "활", "kg", "중", domestic_only=True, label="농어 (sea bass)"),
    SpeciesConfig("도다리", "활", "kg", "중", label="도다리 (flounder)", use_smoothed=True),
    SpeciesConfig("감성돔", "활", "kg", "중", domestic_only=True, label="감성돔 (black porgy)"),
]

def is_foreign(origin):
    if not origin: return False
    for kw in FOREIGN_KW:
        if kw in origin: return True
    return False

# Korean holidays
KOREAN_HOLIDAYS = {
    2018: {"seollal": "2018.02.16", "chuseok": "2018.09.24"},
    2019: {"seollal": "2019.02.05", "chuseok": "2019.09.13"},
    2020: {"seollal": "2020.01.25", "chuseok": "2020.10.01"},
    2021: {"seollal": "2021.02.12", "chuseok": "2021.09.21"},
    2022: {"seollal": "2022.02.01", "chuseok": "2022.09.10"},
    2023: {"seollal": "2023.01.22", "chuseok": "2023.09.29"},
    2024: {"seollal": "2024.02.10", "chuseok": "2024.09.17"},
    2025: {"seollal": "2025.01.29", "chuseok": "2025.10.06"},
}

def parse_date(d): return datetime.strptime(d, "%Y.%m.%d")

def days_to_holiday(dt):
    result = {"days_to_seollal": 999, "days_to_chuseok": 999}
    for y in [dt.year - 1, dt.year, dt.year + 1]:
        if y not in KOREAN_HOLIDAYS: continue
        for name, hd in KOREAN_HOLIDAYS[y].items():
            diff = (parse_date(hd) - dt).days
            key = f"days_to_{name}"
            if abs(diff) < abs(result[key]): result[key] = diff
    return result


# ── Data Extraction ─────────────────────────────────────────────────

def extract_records(data, n, cfg):
    day_data = defaultdict(lambda: {"prices": [], "origins": set(), "qty": 0})
    for i in range(n):
        if data["species"][i] != cfg.species: continue
        if data["state"][i] != cfg.state: continue
        if data["packaging"][i] != cfg.packaging: continue
        if data["spec"][i] != cfg.spec: continue
        if cfg.domestic_only and is_foreign(data["origin"][i]): continue
        d = data["trade_date"][i]
        day_data[d]["prices"].append(data["price_avg"][i])
        if data["origin"][i]: day_data[d]["origins"].add(data["origin"][i])
        day_data[d]["qty"] += data["quantity"][i]

    records = []
    for d in sorted(day_data):
        dd = day_data[d]
        records.append({
            "date": d, "price": float(np.mean(dd["prices"])),
            "n_lots": len(dd["prices"]), "n_origins": len(dd["origins"]),
            "qty": dd["qty"],
        })
    return records


def extract_market_wide_supply(data, n):
    """Market-wide daily lot count — proxy for overall fishing conditions."""
    day_lots = defaultdict(int)
    for i in range(n):
        day_lots[data["trade_date"][i]] += 1
    return day_lots


# ── Feature Engineering (v3: enhanced) ──────────────────────────────

def build_features_v3(records, market_lots, target_offset=7, use_smoothed=False):
    """
    Enhanced features:
    - Calendar (7): dow, month, dom, weekend, woy, quarter, is_monday
    - Holiday (4): days_to_seollal, days_to_chuseok, abs versions
    - Price history (5): lag1, lag7, lag30, 7d_avg, 30d_avg
    - Price momentum (4): change_1d, change_7d, change_30d, change_7d_vs_30d
    - Volatility (3): std_7d, std_30d, range_7d (max-min)
    - Supply proxy (7): n_lots_lag1, n_origins_lag1, qty_lag1, lots_7d_avg,
                         lots_change_7d, market_lots_lag1, market_lots_7d_avg
    - Weather proxy (4): trading_gap_days, lots_drop_flag, qty_drop_7d, supply_shock
    """
    feature_names = [
        # Calendar (7)
        "dow", "month", "dom", "is_weekend", "woy", "quarter", "is_monday",
        # Holiday (4)
        "days_to_seollal", "days_to_chuseok", "abs_days_seollal", "abs_days_chuseok",
        # Price history (5)
        "price_lag1", "price_lag7", "price_lag30", "price_7d_avg", "price_30d_avg",
        # Price momentum (4)
        "pchg_1d", "pchg_7d", "pchg_30d", "pchg_7d_vs_30d",
        # Volatility (3)
        "price_std_7d", "price_std_30d", "price_range_7d",
        # Supply proxy (7)
        "lots_lag1", "origins_lag1", "qty_lag1", "lots_7d_avg",
        "lots_chg_7d", "mkt_lots_lag1", "mkt_lots_7d_avg",
        # Weather proxy (4)
        "gap_days", "lots_drop_flag", "qty_drop_7d", "supply_shock",
    ]

    prices = [r["price"] for r in records]
    lots = [r["n_lots"] for r in records]
    origins = [r["n_origins"] for r in records]
    qtys = [r["qty"] for r in records]
    dates = [r["date"] for r in records]

    # Smoothed target
    if use_smoothed and len(prices) > 7:
        smoothed = np.convolve(prices, np.ones(7)/7, mode="same")
        targets = smoothed.tolist()
    else:
        targets = prices

    X, y, out_dates = [], [], []

    for i in range(30, len(records) - target_offset):
        dt = parse_date(dates[i])
        dt_prev = parse_date(dates[i-1]) if i > 0 else dt
        holidays = days_to_holiday(dt)
        dow = dt.weekday()

        # Price features
        p = prices[i]
        p1 = prices[i-1] if i >= 1 else p
        p7 = prices[i-7] if i >= 7 else p1
        p30 = prices[i-30] if i >= 30 else p1
        avg7 = np.mean(prices[max(0,i-7):i])
        avg30 = np.mean(prices[max(0,i-30):i])
        std7 = np.std(prices[max(0,i-7):i])
        std30 = np.std(prices[max(0,i-30):i])
        range7 = max(prices[max(0,i-7):i]) - min(prices[max(0,i-7):i])

        chg1 = (p - p1) / p1 * 100 if p1 > 0 else 0
        chg7 = (p - p7) / p7 * 100 if p7 > 0 else 0
        chg30 = (p - p30) / p30 * 100 if p30 > 0 else 0
        chg_7v30 = avg7 / avg30 - 1 if avg30 > 0 else 0

        # Supply
        lots_avg7 = np.mean(lots[max(0,i-7):i])
        lots_chg7 = (lots[i] - lots_avg7) / lots_avg7 if lots_avg7 > 0 else 0

        # Market-wide
        mkt_lag1 = market_lots.get(dates[i-1], 0) if i >= 1 else 0
        mkt_7d = np.mean([market_lots.get(dates[j], 0) for j in range(max(0,i-7), i)])

        # Weather proxy: trading gap (days since last record)
        gap = (dt - dt_prev).days if i > 0 else 1

        # Supply shock: lots dropped >50% from 7d avg
        lots_drop = 1 if lots[i] < lots_avg7 * 0.5 and lots_avg7 > 0 else 0

        # Quantity drop
        qty_avg7 = np.mean(qtys[max(0,i-7):i]) if i >= 1 else qtys[i]
        qty_drop = 1 if qtys[i] < qty_avg7 * 0.5 and qty_avg7 > 0 else 0

        # Composite supply shock
        supply_shock = lots_drop + qty_drop + (1 if gap > 3 else 0)

        # TODO: Replace weather proxies with real KHOA data:
        # wave_height_lag1 = khoa_data.get(dates[i-1], {}).get("wave_height", None)
        # water_temp_lag1  = khoa_data.get(dates[i-1], {}).get("water_temp", None)
        # wind_speed_lag1  = khoa_data.get(dates[i-1], {}).get("wind_speed", None)

        features = [
            dow, dt.month, dt.day, int(dow >= 5), dt.isocalendar()[1],
            (dt.month - 1) // 3 + 1, int(dow == 0),
            holidays["days_to_seollal"], holidays["days_to_chuseok"],
            abs(holidays["days_to_seollal"]), abs(holidays["days_to_chuseok"]),
            p, p1, p7, avg7, avg30,
            chg1, chg7, chg30, chg_7v30,
            std7, std30, range7,
            lots[i-1] if i >= 1 else 0, origins[i-1] if i >= 1 else 0,
            qtys[i-1] if i >= 1 else 0, lots_avg7,
            lots_chg7, mkt_lag1, mkt_7d,
            gap, lots_drop, qty_drop, supply_shock,
        ]

        target = targets[i + target_offset]
        X.append(features)
        y.append(target)
        out_dates.append(dates[i])

    return np.array(X), np.array(y), feature_names, out_dates


# ── Backtesting ─────────────────────────────────────────────────────

def backtest_lgbm(X, y, feat_names, species, horizon, n_splits=5):
    n = len(X)
    min_train = int(n * 0.5)
    step = (n - min_train) // n_splits
    if step < 10 or min_train < 100:
        return None

    all_preds, all_actuals, all_prev = [], [], []

    for split in range(n_splits):
        te = min_train + split * step
        te_end = min(te + step, n)
        X_tr, y_tr = X[:te], y[:te]
        X_te, y_te = X[te:te_end], y[te:te_end]
        if len(X_te) == 0: continue

        params = {
            "objective": "regression", "metric": "mae",
            "learning_rate": 0.03, "num_leaves": 31,
            "min_child_samples": 20, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5,
            "reg_alpha": 0.1, "reg_lambda": 0.1,
            "verbose": -1, "n_jobs": 1,
        }
        model = lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=800)
        preds = model.predict(X_te)
        all_preds.extend(preds)
        all_actuals.extend(y_te)
        # price_lag1 is index 11
        all_prev.extend(X_te[:, 11])

    if not all_preds: return None
    preds, actuals, prev = np.array(all_preds), np.array(all_actuals), np.array(all_prev)

    mape = float(np.mean(np.abs(preds - actuals) / np.where(actuals > 0, actuals, 1))) * 100
    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mae = float(mean_absolute_error(actuals, preds))
    dir_acc = float(np.mean((actuals > prev) == (preds > prev))) * 100

    imp = dict(zip(feat_names, model.feature_importance(importance_type="gain")))
    total = sum(imp.values())
    if total > 0:
        imp = {k: round(v/total*100, 1) for k, v in sorted(imp.items(), key=lambda x: -x[1])}

    return {
        "species": species, "model": "LightGBM-v3", "horizon": horizon,
        "mape": round(mape, 2), "rmse": round(rmse), "mae": round(mae),
        "dir_acc": round(dir_acc, 1), "n_tests": len(preds),
        "importance": imp,
    }


def backtest_naive(records, horizon, species):
    prices = [r["price"] for r in records]
    errs = []
    for i in range(180, len(prices) - horizon, 7):
        a, p = prices[i + horizon], prices[i]
        if a > 0: errs.append(abs(p - a) / a)
    return {
        "species": species, "model": "Naive", "horizon": horizon,
        "mape": round(float(np.mean(errs)) * 100, 2) if errs else 999,
        "n_tests": len(errs),
    }


# ── Main ────────────────────────────────────────────────────────────

def main():
    import pyarrow.dataset as ds

    print("Loading data...", end=" ", flush=True)
    dataset = ds.dataset(str(DATA_ROOT), format="parquet", partitioning="hive")
    cols = ["trade_date", "species", "state", "origin", "spec", "packaging", "price_avg", "quantity"]
    table = dataset.to_table(columns=cols)
    data = {col: table.column(col).to_pylist() for col in cols}
    n = len(data["trade_date"])
    print(f"{n:,} rows.")

    market_lots = extract_market_wide_supply(data, n)
    print(f"Market-wide supply: {len(market_lots)} trading days\n")

    all_results = []

    for cfg in SPECIES_CONFIGS:
        print(f"{'='*70}")
        print(f"  {cfg.label} {'(7d-smoothed target)' if cfg.use_smoothed else ''}")
        print(f"{'='*70}")

        records = extract_records(data, n, cfg)
        if len(records) < 200:
            print(f"  SKIP — {len(records)} days\n")
            continue

        prices = np.array([r["price"] for r in records])
        print(f"  {len(records)} days | mean={np.mean(prices):,.0f} | lag1={np.corrcoef(prices[:-1], prices[1:])[0,1]:.3f}")

        for horizon in [7, 14]:
            X, y, fnames, dates = build_features_v3(records, market_lots, horizon, cfg.use_smoothed)
            if len(X) < 200:
                print(f"  {horizon}d: too few samples ({len(X)})")
                continue

            naive = backtest_naive(records, horizon, cfg.species)
            lgbm = backtest_lgbm(X, y, fnames, cfg.species, horizon)

            print(f"\n  {horizon}-day: {'Model':<18} {'MAPE':>7} {'RMSE':>8} {'Dir%':>6}")
            print(f"          {'Naive':<18} {naive['mape']:>6.1f}%")
            if lgbm:
                all_results.append(lgbm)
                improv = (naive['mape'] - lgbm['mape']) / naive['mape'] * 100
                print(f"          {'LightGBM-v3':<18} {lgbm['mape']:>6.1f}% {lgbm['rmse']:>8,} {lgbm['dir_acc']:>5.1f}%  ({improv:+.0f}% vs naive)")
                top5 = list(lgbm["importance"].items())[:5]
                for feat, imp in top5:
                    print(f"            {feat:<25} {imp:>5.1f}%")

        print()

    # Load v1 and v2 for comparison
    v1_path = OUTPUT_DIR / "poc_results.json"
    v2_path = OUTPUT_DIR / "poc_v2_results.json"
    v1_best, v2_best = {}, {}
    if v1_path.exists():
        with open(v1_path) as f:
            for sp, info in json.load(f).get("summary", {}).items():
                v1_best[sp] = info["mape_7d"]
    if v2_path.exists():
        with open(v2_path) as f:
            for sp, info in json.load(f).get("summary", {}).items():
                v2_best[sp] = info["v2_mape"]

    print("\n" + "=" * 80)
    print("COMPARISON: v1 (AR) vs v2 (LightGBM) vs v3 (Enhanced+WeatherProxy) — 7d")
    print("=" * 80)
    print(f"  {'Species':<12} {'v1 AR':>8} {'v2 LGBM':>8} {'v3 Enh':>8} {'v2→v3':>8} {'v3 Dir%':>7}")
    print(f"  {'-'*58}")

    summary = {}
    for cfg in SPECIES_CONFIGS:
        sp = cfg.species
        v3 = next((r for r in all_results if r["species"] == sp and r["horizon"] == 7), None)
        if not v3: continue
        v1 = v1_best.get(sp)
        v2 = v2_best.get(sp)
        v1s = f"{v1:.1f}%" if v1 else "N/A"
        v2s = f"{v2:.1f}%" if v2 else "N/A"
        improv = f"{(v2 - v3['mape']) / v2 * 100:+.0f}%" if v2 else "N/A"
        print(f"  {sp:<12} {v1s:>8} {v2s:>8} {v3['mape']:>7.1f}% {improv:>8} {v3['dir_acc']:>6.1f}%")
        summary[sp] = {"v1": v1, "v2": v2, "v3": v3["mape"], "dir_acc": v3["dir_acc"],
                       "top_features": dict(list(v3["importance"].items())[:10])}

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_at": datetime.now().isoformat(),
        "model": "LightGBM-v3",
        "features": "calendar(7) + holiday(4) + price(5) + momentum(4) + volatility(3) + supply(7) + weather_proxy(4) = 34 features",
        "notes": [
            "Weather proxy features simulate the effect of ocean conditions using supply disruption patterns",
            "TODO: Replace with real KHOA wave_height, water_temp, wind_speed when API key is obtained",
            "Species with use_smoothed=True use 7-day moving average as target",
        ],
        "results": all_results,
        "summary": summary,
    }
    with open(OUTPUT_DIR / "poc_v3_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'poc_v3_results.json'}")


if __name__ == "__main__":
    main()
