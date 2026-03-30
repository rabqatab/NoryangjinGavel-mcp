"""
PoC v4: Cross-Species Supply Features + Regime Splitting.

New over v3:
  1. Cross-species supply features (own vs market vs other sashimi)
  2. Supply-demand ratio features
  3. Regime split for 방어 (in-season Nov-Feb vs off-season)
  4. Price relative to seasonal norm

Usage:
    uv run python scripts/poc_prediction_v4.py
"""
import json
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    {"species": "넙치", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False, "label": "넙치 (flatfish)"},
    {"species": "우럭", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False, "label": "우럭 (rockfish)"},
    {"species": "방어", "state": "선", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": True, "regime_split": True, "label": "방어 (yellowtail)"},
    {"species": "참돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": False, "label": "참돔 (seabream)"},
    {"species": "농어", "state": "활", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": False, "label": "농어 (sea bass)"},
    {"species": "도다리", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": True, "label": "도다리 (flounder)"},
    {"species": "감성돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": False, "label": "감성돔 (black porgy)"},
]

def is_foreign(o):
    if not o: return False
    return any(kw in o for kw in FOREIGN_KW)

def parse_date(d): return datetime.strptime(d, "%Y.%m.%d")

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

def days_to_holiday(dt):
    r = {"seollal": 999, "chuseok": 999}
    for y in [dt.year - 1, dt.year, dt.year + 1]:
        if y not in KOREAN_HOLIDAYS: continue
        for name, hd in KOREAN_HOLIDAYS[y].items():
            diff = (parse_date(hd) - dt).days
            if abs(diff) < abs(r[name]): r[name] = diff
    return r


# ── Data Extraction ─────────────────────────────────────────────────

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
    """Build daily supply context for all sashimi species + market total."""
    all_dates = sorted(set(data["trade_date"]))
    date_idx = {d: i for i, d in enumerate(all_dates)}
    nd = len(all_dates)

    # Per-species daily quantity
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

    # 7-day rolling averages
    k = 7
    sp_qty_7d = {sp: np.convolve(q, np.ones(k)/k, mode="same") for sp, q in sp_qty.items()}
    sp_lots_7d = {sp: np.convolve(l, np.ones(k)/k, mode="same") for sp, l in sp_lots.items()}
    market_lots_7d = np.convolve(market_lots, np.ones(k)/k, mode="same")

    # Total sashimi supply
    total_sashimi = sum(sp_qty.values())
    total_sashimi_7d = np.convolve(total_sashimi, np.ones(k)/k, mode="same")

    return {
        "dates": all_dates, "date_idx": date_idx,
        "sp_qty": sp_qty, "sp_lots": sp_lots,
        "sp_qty_7d": sp_qty_7d, "sp_lots_7d": sp_lots_7d,
        "market_lots": market_lots, "market_lots_7d": market_lots_7d,
        "total_sashimi": total_sashimi, "total_sashimi_7d": total_sashimi_7d,
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


# ── Feature Engineering (v4) ────────────────────────────────────────

def build_features_v4(records, ctx, target_species, target_offset=7, use_smoothed=False):
    """
    v4 features (41 total):
    - Calendar (7): dow, month, dom, weekend, woy, quarter, is_monday
    - Holiday (4): days_to_seollal/chuseok + abs versions
    - Price history (5): lag1, lag7, lag30, 7d_avg, 30d_avg
    - Price momentum (4): chg_1d, chg_7d, chg_30d, 7d_vs_30d
    - Volatility (3): std_7d, std_30d, range_7d
    - Own supply (5): qty_7d, lots_7d, qty_ratio_30d, qty_chg_7d, lots_chg_7d
    - Cross supply (5): other_sashimi_qty_7d, market_lots_7d, sashimi_concentration,
                         total_sashimi_chg_7d, market_chg_7d
    - Seasonal context (4): price_vs_monthly_avg, month_sin, month_cos, is_peak_season
    - Weather proxy (4): gap_days, lots_drop, qty_drop, supply_shock
    """
    feature_names = [
        "dow", "month", "dom", "is_weekend", "woy", "quarter", "is_monday",
        "days_to_seollal", "days_to_chuseok", "abs_seollal", "abs_chuseok",
        "price_lag1", "price_lag7", "price_lag30", "price_7d_avg", "price_30d_avg",
        "pchg_1d", "pchg_7d", "pchg_30d", "pchg_7v30",
        "price_std_7d", "price_std_30d", "price_range_7d",
        "own_qty_7d", "own_lots_7d", "own_qty_ratio_30d", "own_qty_chg_7d", "own_lots_chg_7d",
        "other_sashimi_qty_7d", "market_lots_7d", "sashimi_concentration",
        "total_sashimi_chg_7d", "market_chg_7d",
        "price_vs_month_avg", "month_sin", "month_cos", "is_peak_season",
        "gap_days", "lots_drop", "qty_drop", "supply_shock",
    ]

    prices = [r["price"] for r in records]
    dates = [r["date"] for r in records]
    date_idx = ctx["date_idx"]

    if use_smoothed and len(prices) > 7:
        targets = np.convolve(prices, np.ones(7)/7, mode="same").tolist()
    else:
        targets = prices

    # Monthly average price (for seasonal context)
    monthly_avg = defaultdict(list)
    for r in records:
        m = parse_date(r["date"]).month
        monthly_avg[m].append(r["price"])
    monthly_avg = {m: np.mean(v) for m, v in monthly_avg.items()}

    X, y, out_dates = [], [], []

    for i in range(30, len(records) - target_offset):
        dt = parse_date(dates[i])
        di = date_idx.get(dates[i], 0)
        di_prev = date_idx.get(dates[i-1], di) if i > 0 else di
        hol = days_to_holiday(dt)
        dow = dt.weekday()

        p = prices[i]
        p1 = prices[i-1] if i >= 1 else p
        p7 = prices[i-7] if i >= 7 else p1
        p30 = prices[i-30] if i >= 30 else p1
        avg7 = np.mean(prices[max(0,i-7):i])
        avg30 = np.mean(prices[max(0,i-30):i])
        std7 = np.std(prices[max(0,i-7):i])
        std30 = np.std(prices[max(0,i-30):i])
        rng7 = max(prices[max(0,i-7):i]) - min(prices[max(0,i-7):i])

        chg1 = (p - p1) / p1 * 100 if p1 > 0 else 0
        chg7 = (p - p7) / p7 * 100 if p7 > 0 else 0
        chg30 = (p - p30) / p30 * 100 if p30 > 0 else 0
        chg_7v30 = avg7 / avg30 - 1 if avg30 > 0 else 0

        # Own supply (from context, using date index)
        own_q7 = ctx["sp_qty_7d"][target_species][di]
        own_l7 = ctx["sp_lots_7d"][target_species][di]
        own_q30 = np.mean(ctx["sp_qty"][target_species][max(0,di-30):di]) if di >= 1 else own_q7
        own_q_ratio = own_q7 / own_q30 if own_q30 > 0 else 1
        own_q_chg = (ctx["sp_qty_7d"][target_species][di] - ctx["sp_qty_7d"][target_species][max(0,di-7)]) / max(ctx["sp_qty_7d"][target_species][max(0,di-7)], 1)
        own_l_chg = (ctx["sp_lots_7d"][target_species][di] - ctx["sp_lots_7d"][target_species][max(0,di-7)]) / max(ctx["sp_lots_7d"][target_species][max(0,di-7)], 1)

        # Cross supply
        other_q7 = ctx["total_sashimi_7d"][di] - ctx["sp_qty_7d"][target_species][di]
        mkt_l7 = ctx["market_lots_7d"][di]
        total_s = ctx["total_sashimi"][di]
        own_s = ctx["sp_qty"][target_species][di]
        concentration = own_s / total_s if total_s > 0 else 0

        ts_chg = (ctx["total_sashimi_7d"][di] - ctx["total_sashimi_7d"][max(0,di-7)]) / max(ctx["total_sashimi_7d"][max(0,di-7)], 1)
        mkt_chg = (ctx["market_lots_7d"][di] - ctx["market_lots_7d"][max(0,di-7)]) / max(ctx["market_lots_7d"][max(0,di-7)], 1)

        # Seasonal context
        m_avg = monthly_avg.get(dt.month, p)
        price_vs_month = p / m_avg if m_avg > 0 else 1
        month_sin = np.sin(2 * np.pi * dt.month / 12)
        month_cos = np.cos(2 * np.pi * dt.month / 12)
        is_peak = int(dt.month in [11, 12, 1, 2])  # winter peak for sashimi

        # Weather proxy
        gap = (dt - parse_date(dates[i-1])).days if i > 0 else 1
        lots_drop = int(own_l7 < ctx["sp_lots_7d"][target_species][max(0,di-14)] * 0.5) if di >= 14 else 0
        qty_drop = int(own_q7 < own_q30 * 0.5) if own_q30 > 0 else 0
        shock = lots_drop + qty_drop + int(gap > 3)

        features = [
            dow, dt.month, dt.day, int(dow >= 5), dt.isocalendar()[1],
            (dt.month - 1) // 3 + 1, int(dow == 0),
            hol["seollal"], hol["chuseok"], abs(hol["seollal"]), abs(hol["chuseok"]),
            p, p1, p7, avg7, avg30,
            chg1, chg7, chg30, chg_7v30,
            std7, std30, rng7,
            own_q7, own_l7, own_q_ratio, own_q_chg, own_l_chg,
            other_q7, mkt_l7, concentration, ts_chg, mkt_chg,
            price_vs_month, month_sin, month_cos, is_peak,
            gap, lots_drop, qty_drop, shock,
        ]

        X.append(features)
        y.append(targets[i + target_offset])
        out_dates.append(dates[i])

    return np.array(X), np.array(y), feature_names, out_dates


# ── Backtesting ─────────────────────────────────────────────────────

def backtest_lgbm(X, y, fnames, species, horizon, n_splits=5):
    n = len(X)
    min_train = int(n * 0.5)
    step = (n - min_train) // n_splits
    if step < 10 or min_train < 100: return None

    preds_all, actuals_all, prev_all = [], [], []
    last_model = None

    for s in range(n_splits):
        te = min_train + s * step
        te_end = min(te + step, n)
        Xtr, ytr = X[:te], y[:te]
        Xte, yte = X[te:te_end], y[te:te_end]
        if len(Xte) == 0: continue

        params = {
            "objective": "regression", "metric": "mae",
            "learning_rate": 0.03, "num_leaves": 31,
            "min_child_samples": 20, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5,
            "reg_alpha": 0.1, "reg_lambda": 0.1,
            "verbose": -1, "n_jobs": 1,
        }
        last_model = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=1000)
        preds_all.extend(last_model.predict(Xte))
        actuals_all.extend(yte)
        prev_all.extend(Xte[:, 11])  # price_lag1

    if not preds_all: return None
    P, A, Pr = np.array(preds_all), np.array(actuals_all), np.array(prev_all)

    mape = float(np.mean(np.abs(P - A) / np.where(A > 0, A, 1))) * 100
    rmse = float(np.sqrt(mean_squared_error(A, P)))
    mae = float(mean_absolute_error(A, P))
    dir_acc = float(np.mean((A > Pr) == (P > Pr))) * 100

    imp = dict(zip(fnames, last_model.feature_importance(importance_type="gain")))
    total = sum(imp.values())
    imp = {k: round(v/total*100, 1) for k, v in sorted(imp.items(), key=lambda x: -x[1])} if total else {}

    return {"species": species, "horizon": horizon, "mape": round(mape, 2),
            "rmse": round(rmse), "mae": round(mae), "dir_acc": round(dir_acc, 1),
            "n_tests": len(P), "importance": imp}


def backtest_naive(records, horizon, species):
    prices = [r["price"] for r in records]
    errs = [abs(prices[i] - prices[i+horizon]) / prices[i+horizon]
            for i in range(180, len(prices) - horizon, 7) if prices[i+horizon] > 0]
    return {"species": species, "mape": round(float(np.mean(errs))*100, 2) if errs else 999}


# ── Main ────────────────────────────────────────────────────────────

def main():
    data = load_all()
    n = len(data["trade_date"])

    print("Building supply context...", end=" ", flush=True)
    ctx = build_supply_context(data, n)
    print(f"{len(ctx['dates'])} trading days.\n")

    all_results = []

    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        print(f"{'='*70}")
        print(f"  {cfg['label']} {'(smoothed)' if cfg.get('smoothed') else ''}")
        print(f"{'='*70}")

        records = extract_records(data, n, cfg)
        if len(records) < 200:
            print(f"  SKIP — {len(records)} days\n"); continue

        prices = np.array([r["price"] for r in records])
        print(f"  {len(records)} days | mean={np.mean(prices):,.0f} | lag1={np.corrcoef(prices[:-1], prices[1:])[0,1]:.3f}")

        # Regime split for 방어: separate in-season (Nov-Feb) model
        if cfg.get("regime_split"):
            for regime, months, regime_label in [
                ("winter", {11,12,1,2}, "IN-SEASON (Nov-Feb)"),
                ("other", {3,4,5,6,7,8,9,10}, "OFF-SEASON (Mar-Oct)"),
            ]:
                regime_records = [r for r in records if parse_date(r["date"]).month in months]
                if len(regime_records) < 100:
                    print(f"  {regime_label}: only {len(regime_records)} records, skip"); continue

                for horizon in [7]:
                    X, y, fnames, dates = build_features_v4(regime_records, ctx, sp, horizon, cfg.get("smoothed", False))
                    if len(X) < 100: continue
                    naive = backtest_naive(regime_records, horizon, sp)
                    r = backtest_lgbm(X, y, fnames, f"{sp}_{regime}", horizon)
                    if r:
                        r["regime"] = regime_label
                        all_results.append(r)
                        print(f"\n  {regime_label} {horizon}d: MAPE={r['mape']:.1f}% (naive={naive['mape']:.1f}%) dir={r['dir_acc']:.1f}%")
                        for feat, imp in list(r["importance"].items())[:5]:
                            print(f"    {feat:<28} {imp:>5.1f}%")
        else:
            for horizon in [7, 14]:
                X, y, fnames, dates = build_features_v4(records, ctx, sp, horizon, cfg.get("smoothed", False))
                if len(X) < 200: continue

                naive = backtest_naive(records, horizon, sp)
                r = backtest_lgbm(X, y, fnames, sp, horizon)
                if r:
                    all_results.append(r)
                    improv = (naive['mape'] - r['mape']) / naive['mape'] * 100
                    print(f"\n  {horizon}d: MAPE={r['mape']:.1f}% (naive={naive['mape']:.1f}%, {improv:+.0f}%) dir={r['dir_acc']:.1f}%")
                    for feat, imp in list(r["importance"].items())[:5]:
                        print(f"    {feat:<28} {imp:>5.1f}%")

        print()

    # Comparison table
    print("\n" + "=" * 80)
    print("v1 → v2 → v3 → v4 COMPARISON (7-day horizon)")
    print("=" * 80)

    # Load previous
    prev = {}
    for ver, fname in [("v1", "poc_results.json"), ("v2", "poc_v2_results.json"), ("v3", "poc_v3_results.json")]:
        p = OUTPUT_DIR / fname
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            for sp, info in d.get("summary", {}).items():
                if ver == "v1": prev.setdefault(sp, {})[ver] = info.get("mape_7d")
                elif ver == "v2": prev.setdefault(sp, {})[ver] = info.get("v2_mape")
                elif ver == "v3": prev.setdefault(sp, {})[ver] = info.get("v3")

    print(f"\n  {'Species':<12} {'v1 AR':>8} {'v2':>8} {'v3':>8} {'v4':>8} {'v3→v4':>8} {'Dir%':>7}")
    print(f"  {'-'*62}")

    summary = {}
    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        # Find v4 result (7d, non-regime or winter regime for 방어)
        if cfg.get("regime_split"):
            v4 = next((r for r in all_results if r["species"] == f"{sp}_winter" and r["horizon"] == 7), None)
        else:
            v4 = next((r for r in all_results if r["species"] == sp and r["horizon"] == 7), None)
        if not v4: continue

        p = prev.get(sp, {})
        v1s = f"{p.get('v1', 0):.1f}%" if p.get('v1') else "N/A"
        v2s = f"{p.get('v2', 0):.1f}%" if p.get('v2') else "N/A"
        v3s = f"{p.get('v3', 0):.1f}%" if p.get('v3') else "N/A"
        v3_val = p.get('v3')
        improv = f"{(v3_val - v4['mape']) / v3_val * 100:+.0f}%" if v3_val else "N/A"

        label = f"{sp}" + (f" ({v4.get('regime', '')})" if v4.get('regime') else "")
        print(f"  {label:<20} {v1s:>8} {v2s:>8} {v3s:>8} {v4['mape']:>7.1f}% {improv:>8} {v4['dir_acc']:>6.1f}%")
        summary[sp] = {"v4_mape": v4["mape"], "dir_acc": v4["dir_acc"],
                       "top_features": dict(list(v4["importance"].items())[:10])}

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_at": datetime.now().isoformat(),
        "model": "LightGBM-v4",
        "features": "41 features: calendar(7) + holiday(4) + price(9) + momentum(4) + volatility(3) + own_supply(5) + cross_supply(5) + seasonal(4) + weather_proxy(4)",
        "new_in_v4": [
            "Cross-species supply: other_sashimi_qty_7d, sashimi_concentration, total_sashimi_chg_7d",
            "Own supply enhanced: qty_ratio_vs_30d, qty_chg_7d, lots_chg_7d",
            "Seasonal context: price_vs_monthly_avg, month_sin/cos, is_peak_season",
            "Regime split for 방어 (winter vs off-season)",
        ],
        "results": all_results,
        "summary": summary,
    }
    with open(OUTPUT_DIR / "poc_v4_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'poc_v4_results.json'}")


if __name__ == "__main__":
    main()
