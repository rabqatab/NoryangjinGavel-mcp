"""
PoC v2: Feature-Engineered Price Prediction.

Replaces pure autoregression with calendar + supply-proxy + price-history features.
Uses LightGBM for regression. Backtests against v1 AR models.

Usage:
    uv run python scripts/poc_prediction_v2.py
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

# ── Species Config (same as v1) ─────────────────────────────────────

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

SPECIES_CONFIGS = [
    SpeciesConfig("넙치", "활", "kg", "중", label="넙치 (flatfish)"),
    SpeciesConfig("우럭", "활", "kg", "중", label="우럭 (rockfish)"),
    SpeciesConfig("방어", "선", "kg", "중", domestic_only=True, label="방어 (yellowtail)"),
    SpeciesConfig("참돔", "활", "kg", "중", domestic_only=True, label="참돔 (seabream)"),
    SpeciesConfig("농어", "활", "kg", "중", domestic_only=True, label="농어 (sea bass)"),
    SpeciesConfig("도다리", "활", "kg", "중", label="도다리 (flounder)"),
    SpeciesConfig("감성돔", "활", "kg", "중", domestic_only=True, label="감성돔 (black porgy)"),
]

def is_foreign(origin):
    if not origin: return False
    for kw in FOREIGN_KW:
        if kw in origin: return True
    return False

# ── Korean Holidays (설날/추석 approximate dates) ───────────────────

# Major holidays affecting fish market demand (lunar calendar based)
# These are approximate — a proper implementation would use korean_lunar_calendar
KOREAN_HOLIDAYS = {
    # 설날 (Lunar New Year) ± 1 day
    2020: {"seollal": "2020.01.25", "chuseok": "2020.10.01"},
    2021: {"seollal": "2021.02.12", "chuseok": "2021.09.21"},
    2022: {"seollal": "2022.02.01", "chuseok": "2022.09.10"},
    2023: {"seollal": "2023.01.22", "chuseok": "2023.09.29"},
    2024: {"seollal": "2024.02.10", "chuseok": "2024.09.17"},
    2025: {"seollal": "2025.01.29", "chuseok": "2025.10.06"},
}


def parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y.%m.%d")


def days_to_nearest_holiday(dt: datetime) -> dict:
    """Compute days to nearest 설날 and 추석."""
    year = dt.year
    result = {"days_to_seollal": 999, "days_to_chuseok": 999}

    for y in [year - 1, year, year + 1]:
        if y not in KOREAN_HOLIDAYS:
            continue
        for hol_name, hol_date in KOREAN_HOLIDAYS[y].items():
            hol_dt = parse_date(hol_date)
            diff = (hol_dt - dt).days
            key = f"days_to_{hol_name}"
            if abs(diff) < abs(result[key]):
                result[key] = diff

    return result


# ── Feature Engineering ─────────────────────────────────────────────

@dataclass
class DailyRecord:
    date: str
    price: float
    n_lots: int = 0           # number of auction lots that day (supply proxy)
    n_origins: int = 0        # number of distinct origins (supply diversity)
    total_quantity: float = 0  # total quantity traded


def extract_daily_records(data: dict, n: int, cfg: SpeciesConfig) -> list[DailyRecord]:
    """Extract filtered daily records with supply-proxy features."""
    day_data = defaultdict(lambda: {"prices": [], "origins": set(), "qty": 0})

    for i in range(n):
        if data["species"][i] != cfg.species:
            continue
        if data["state"][i] != cfg.state:
            continue
        if data["packaging"][i] != cfg.packaging:
            continue
        if data["spec"][i] != cfg.spec:
            continue
        if cfg.domestic_only and is_foreign(data["origin"][i]):
            continue

        d = data["trade_date"][i]
        day_data[d]["prices"].append(data["price_avg"][i])
        if data["origin"][i]:
            day_data[d]["origins"].add(data["origin"][i])
        day_data[d]["qty"] += data["quantity"][i]

    records = []
    for d in sorted(day_data.keys()):
        dd = day_data[d]
        records.append(DailyRecord(
            date=d,
            price=float(np.mean(dd["prices"])),
            n_lots=len(dd["prices"]),
            n_origins=len(dd["origins"]),
            total_quantity=dd["qty"],
        ))
    return records


def build_features(records: list[DailyRecord], target_offset: int = 7) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Build feature matrix and target vector.

    Features:
    - Calendar: day_of_week, month, is_weekend
    - Holiday proximity: days_to_seollal, days_to_chuseok
    - Price history: price_lag1, price_lag7, price_lag30, price_7d_avg, price_30d_avg
    - Price momentum: price_change_1d, price_change_7d, price_change_30d
    - Volatility: price_std_7d, price_std_30d
    - Supply proxy: n_lots_lag1, n_origins_lag1, qty_lag1, n_lots_7d_avg
    """
    feature_names = [
        # Calendar (5)
        "day_of_week", "month", "day_of_month", "is_weekend", "week_of_year",
        # Holiday (2)
        "days_to_seollal", "days_to_chuseok",
        # Price history (5)
        "price_lag1", "price_lag7", "price_lag30", "price_7d_avg", "price_30d_avg",
        # Price momentum (3)
        "price_change_1d_pct", "price_change_7d_pct", "price_change_30d_pct",
        # Volatility (2)
        "price_std_7d", "price_std_30d",
        # Supply proxy (4)
        "n_lots_lag1", "n_origins_lag1", "qty_lag1", "n_lots_7d_avg",
    ]

    # Build lookup by index for lag features
    prices = [r.price for r in records]
    n_lots = [r.n_lots for r in records]
    n_origins = [r.n_origins for r in records]
    qtys = [r.total_quantity for r in records]
    dates_str = [r.date for r in records]

    X, y, out_dates = [], [], []

    for i in range(30, len(records) - target_offset):
        dt = parse_date(records[i].date)

        # Calendar
        dow = dt.weekday()
        holidays = days_to_nearest_holiday(dt)

        # Price lags
        p = prices[i]
        p1 = prices[i - 1]
        p7 = prices[i - 7] if i >= 7 else p1
        p30 = prices[i - 30] if i >= 30 else p1
        avg7 = np.mean(prices[max(0, i-7):i])
        avg30 = np.mean(prices[max(0, i-30):i])
        std7 = np.std(prices[max(0, i-7):i]) if i >= 7 else 0
        std30 = np.std(prices[max(0, i-30):i]) if i >= 30 else 0

        # Momentum
        chg1 = (p - p1) / p1 * 100 if p1 > 0 else 0
        chg7 = (p - p7) / p7 * 100 if p7 > 0 else 0
        chg30 = (p - p30) / p30 * 100 if p30 > 0 else 0

        # Supply
        lots_avg7 = np.mean(n_lots[max(0, i-7):i])

        features = [
            dow, dt.month, dt.day, int(dow >= 5), dt.isocalendar()[1],
            holidays["days_to_seollal"], holidays["days_to_chuseok"],
            p, p1 if i >= 1 else p, p7, avg7, avg30,
            chg1, chg7, chg30,
            std7, std30,
            n_lots[i-1] if i >= 1 else 0,
            n_origins[i-1] if i >= 1 else 0,
            qtys[i-1] if i >= 1 else 0,
            lots_avg7,
        ]

        # Target: price at t + target_offset
        target = prices[i + target_offset]

        X.append(features)
        y.append(target)
        out_dates.append(dates_str[i])

    return np.array(X), np.array(y), feature_names, out_dates


# ── Backtesting ─────────────────────────────────────────────────────

@dataclass
class ModelResult:
    species: str
    model: str
    horizon: int
    mape: float
    rmse: float
    mae: float
    directional_acc: float
    n_tests: int
    feature_importance: dict = field(default_factory=dict)


def backtest_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    species: str,
    horizon: int,
    train_ratio: float = 0.7,
    n_splits: int = 5,
) -> Optional[ModelResult]:
    """Time-series cross-validation for LightGBM."""

    n = len(X)
    min_train = int(n * 0.5)
    step = (n - min_train) // n_splits

    if step < 10 or min_train < 100:
        return None

    all_preds, all_actuals, all_prev = [], [], []

    for split in range(n_splits):
        train_end = min_train + split * step
        test_end = min(train_end + step, n)

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]

        if len(X_test) == 0:
            continue

        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": 1,
        }

        train_data = lgb.Dataset(X_train, y_train)
        model = lgb.train(
            params, train_data,
            num_boost_round=500,
        )

        preds = model.predict(X_test)
        all_preds.extend(preds)
        all_actuals.extend(y_test)
        # Previous price for directional accuracy (price_lag1 is feature index 7)
        all_prev.extend(X_test[:, 7])

    if not all_preds:
        return None

    preds = np.array(all_preds)
    actuals = np.array(all_actuals)
    prev = np.array(all_prev)

    # Metrics
    mape = float(np.mean(np.abs(preds - actuals) / np.where(actuals > 0, actuals, 1))) * 100
    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mae = float(mean_absolute_error(actuals, preds))

    # Directional accuracy
    actual_dir = actuals > prev
    pred_dir = preds > prev
    dir_acc = float(np.mean(actual_dir == pred_dir)) * 100

    # Feature importance from last model
    importance = dict(zip(feature_names, model.feature_importance(importance_type="gain")))
    # Normalize
    total = sum(importance.values())
    if total > 0:
        importance = {k: round(v / total * 100, 1) for k, v in sorted(importance.items(), key=lambda x: -x[1])}

    return ModelResult(
        species=species, model="LightGBM", horizon=horizon,
        mape=round(mape, 2), rmse=round(rmse), mae=round(mae),
        directional_acc=round(dir_acc, 1), n_tests=len(preds),
        feature_importance=importance,
    )


def backtest_naive(records: list[DailyRecord], horizon: int, species: str) -> ModelResult:
    """Baseline: naive (last value) forecast."""
    errors_pct, errors_abs, errors_sq, dir_correct = [], [], [], []
    prices = [r.price for r in records]

    for i in range(180, len(prices) - horizon, 7):
        actual = prices[i + horizon]
        pred = prices[i]
        if actual > 0:
            errors_pct.append(abs(pred - actual) / actual)
            errors_abs.append(abs(pred - actual))
            errors_sq.append((pred - actual) ** 2)
        if i > 0:
            dir_correct.append((actual > prices[i]) == (pred > prices[i]))

    return ModelResult(
        species=species, model="Naive", horizon=horizon,
        mape=round(float(np.mean(errors_pct)) * 100, 2),
        rmse=round(float(np.sqrt(np.mean(errors_sq)))),
        mae=round(float(np.mean(errors_abs))),
        directional_acc=round(float(np.mean(dir_correct)) * 100, 1) if dir_correct else 0,
        n_tests=len(dir_correct),
    )


# ── Main ────────────────────────────────────────────────────────────

def main():
    import pyarrow.dataset as ds

    print("Loading data...", end=" ", flush=True)
    dataset = ds.dataset(str(DATA_ROOT), format="parquet", partitioning="hive")
    cols = ["trade_date", "species", "state", "origin", "spec", "packaging", "price_avg", "quantity"]
    table = dataset.to_table(columns=cols)
    data = {col: table.column(col).to_pylist() for col in cols}
    n = len(data["trade_date"])
    print(f"{n:,} rows.\n")

    all_results = []

    for cfg in SPECIES_CONFIGS:
        print(f"{'='*70}")
        print(f"  {cfg.label}")
        print(f"  Filter: state={cfg.state}, pkg={cfg.packaging}, spec={cfg.spec}"
              f"{', domestic' if cfg.domestic_only else ''}")
        print(f"{'='*70}")

        records = extract_daily_records(data, n, cfg)
        if len(records) < 200:
            print(f"  SKIP — only {len(records)} trading days\n")
            continue

        prices = np.array([r.price for r in records])
        print(f"  Series: {len(records)} days ({records[0].date} ~ {records[-1].date})")
        print(f"  Price: mean={np.mean(prices):,.0f}, std={np.std(prices):,.0f}")
        print(f"  Lag-1: {np.corrcoef(prices[:-1], prices[1:])[0,1]:.4f}")

        for horizon in [7, 14]:
            print(f"\n  --- {horizon}-day horizon ---")

            # Build features
            X, y, feat_names, dates = build_features(records, target_offset=horizon)
            if len(X) < 200:
                print(f"  Too few samples ({len(X)}) after feature engineering")
                continue

            print(f"  Samples: {len(X)}")

            # Naive baseline
            naive = backtest_naive(records, horizon, cfg.species)
            all_results.append(naive)

            # LightGBM
            lgbm = backtest_lgbm(X, y, feat_names, cfg.species, horizon)
            if lgbm:
                all_results.append(lgbm)

            # Compare
            print(f"  {'Model':<15} {'MAPE':>7} {'RMSE':>8} {'MAE':>8} {'Dir%':>6} {'Tests':>6}")
            print(f"  {naive.model:<15} {naive.mape:>6.1f}% {naive.rmse:>8,} {naive.mae:>8,} "
                  f"{naive.directional_acc:>5.1f}% {naive.n_tests:>5}")
            if lgbm:
                print(f"  {lgbm.model:<15} {lgbm.mape:>6.1f}% {lgbm.rmse:>8,} {lgbm.mae:>8,} "
                      f"{lgbm.directional_acc:>5.1f}% {lgbm.n_tests:>5}")
                improvement = (naive.mape - lgbm.mape) / naive.mape * 100
                print(f"  → MAPE improvement: {improvement:+.1f}%")

                # Top features
                print(f"\n  Top 5 features:")
                for feat, imp in list(lgbm.feature_importance.items())[:5]:
                    print(f"    {feat:<25} {imp:>5.1f}%")

        print()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: v1 (AR) vs v2 (Feature-Engineered) — 7-day horizon")
    print("=" * 80)

    # Load v1 results for comparison
    v1_path = OUTPUT_DIR / "poc_results.json"
    v1_best = {}
    if v1_path.exists():
        with open(v1_path) as f:
            v1_data = json.load(f)
        for sp, info in v1_data.get("summary", {}).items():
            v1_best[sp] = info["mape_7d"]

    print(f"\n  {'Species':<12} {'v1 AR':>8} {'v2 LGBM':>8} {'Naive':>8} {'Improv':>8} {'v2 Dir%':>7}")
    print(f"  {'-'*55}")

    summary = {}
    for cfg in SPECIES_CONFIGS:
        sp = cfg.species
        lgbm_7d = next((r for r in all_results if r.species == sp and r.model == "LightGBM" and r.horizon == 7), None)
        naive_7d = next((r for r in all_results if r.species == sp and r.model == "Naive" and r.horizon == 7), None)
        v1_mape = v1_best.get(sp, None)

        if lgbm_7d and naive_7d:
            v1_str = f"{v1_mape:.1f}%" if v1_mape else "N/A"
            improv = f"{(v1_mape - lgbm_7d.mape) / v1_mape * 100:+.0f}%" if v1_mape else "N/A"
            print(f"  {sp:<12} {v1_str:>8} {lgbm_7d.mape:>7.1f}% {naive_7d.mape:>7.1f}% {improv:>8} {lgbm_7d.directional_acc:>6.1f}%")
            summary[sp] = {
                "v1_mape": v1_mape,
                "v2_mape": lgbm_7d.mape,
                "naive_mape": naive_7d.mape,
                "directional_acc": lgbm_7d.directional_acc,
                "top_features": dict(list(lgbm_7d.feature_importance.items())[:10]),
            }

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now().isoformat(),
        "model": "LightGBM",
        "features": "calendar + holiday + price_history + momentum + volatility + supply_proxy",
        "results": [{
            "species": r.species, "model": r.model, "horizon": r.horizon,
            "mape": r.mape, "rmse": r.rmse, "mae": r.mae,
            "directional_acc": r.directional_acc, "n_tests": r.n_tests,
            "feature_importance": r.feature_importance,
        } for r in all_results],
        "summary": summary,
    }
    out_path = OUTPUT_DIR / "poc_v2_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
