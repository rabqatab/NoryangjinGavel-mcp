"""
PoC: Price Prediction for Sashimi Species.

Extracts filtered daily price series for 7 sashimi species,
fits Exponential Smoothing + ARIMA, backtests, and reports accuracy.

Usage:
    uv run python scripts/poc_prediction.py
"""
import json
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.dataset as ds

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


SPECIES_CONFIGS = [
    SpeciesConfig("넙치", "활", "kg", "중", label="넙치 (flatfish)"),
    SpeciesConfig("우럭", "활", "kg", "중", label="우럭 (rockfish)"),
    SpeciesConfig("방어", "선", "kg", "중", domestic_only=True, label="방어 (yellowtail)"),
    SpeciesConfig("참돔", "활", "kg", "중", domestic_only=True, label="참돔 (seabream)"),
    SpeciesConfig("농어", "활", "kg", "중", domestic_only=True, label="농어 (sea bass)"),
    SpeciesConfig("도다리", "활", "kg", "중", label="도다리 (flounder)"),
    SpeciesConfig("감성돔", "활", "kg", "중", domestic_only=True, label="감성돔 (black porgy)"),
]


def is_foreign(origin: str) -> bool:
    if not origin:
        return False
    for kw in FOREIGN_KW:
        if kw in origin:
            return True
    return False


# ── Data Extraction ─────────────────────────────────────────────────

def extract_daily_series(data: dict, n: int, cfg: SpeciesConfig) -> dict[str, float]:
    """Extract filtered daily avg price series for a species config."""
    day_prices = defaultdict(list)
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
        day_prices[data["trade_date"][i]].append(data["price_avg"][i])

    return {d: float(np.mean(p)) for d, p in sorted(day_prices.items())}


# ── Models ──────────────────────────────────────────────────────────

def forecast_naive(train: np.ndarray, horizon: int) -> np.ndarray:
    """Naive: last value repeated."""
    return np.full(horizon, train[-1])


def forecast_sma(train: np.ndarray, horizon: int, window: int = 7) -> np.ndarray:
    """Simple Moving Average."""
    return np.full(horizon, np.mean(train[-window:]))


def forecast_ema(train: np.ndarray, horizon: int) -> Optional[np.ndarray]:
    """Exponential Smoothing (Holt-Winters additive)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        model = ExponentialSmoothing(
            train, trend="add", seasonal=None,
        ).fit(optimized=True)
        return model.forecast(horizon)
    except Exception:
        return None


def forecast_arima(train: np.ndarray, horizon: int) -> Optional[np.ndarray]:
    """ARIMA(2,1,2)."""
    from statsmodels.tsa.arima.model import ARIMA
    try:
        model = ARIMA(train, order=(2, 1, 2)).fit()
        return model.forecast(steps=horizon)
    except Exception:
        return None


# ── Backtesting ─────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    species: str
    model: str
    horizon: int
    mape: float
    rmse: float
    mae: float
    directional_acc: float
    n_tests: int


def backtest(
    series: np.ndarray,
    forecast_fn,
    model_name: str,
    species: str,
    horizon: int = 7,
    min_train: int = 180,
    step: int = 7,
) -> Optional[BacktestResult]:
    """Rolling-window backtest."""
    errors_pct = []
    errors_abs = []
    errors_sq = []
    dir_correct = []

    for t in range(min_train, len(series) - horizon, step):
        train = series[:t]
        actual = series[t:t + horizon]

        pred = forecast_fn(train, horizon)
        if pred is None:
            continue

        # MAPE components
        for j in range(min(len(actual), len(pred))):
            if actual[j] > 0:
                errors_pct.append(abs(pred[j] - actual[j]) / actual[j])
                errors_abs.append(abs(pred[j] - actual[j]))
                errors_sq.append((pred[j] - actual[j]) ** 2)

        # Directional accuracy (did we predict up/down correctly?)
        if len(actual) >= horizon and len(pred) >= horizon:
            actual_dir = actual[-1] > train[-1]
            pred_dir = pred[-1] > train[-1]
            dir_correct.append(actual_dir == pred_dir)

    if not errors_pct:
        return None

    return BacktestResult(
        species=species,
        model=model_name,
        horizon=horizon,
        mape=round(float(np.mean(errors_pct)) * 100, 2),
        rmse=round(float(np.sqrt(np.mean(errors_sq)))),
        mae=round(float(np.mean(errors_abs))),
        directional_acc=round(float(np.mean(dir_correct)) * 100, 1) if dir_correct else 0,
        n_tests=len(dir_correct),
    )


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("Loading data...", end=" ", flush=True)
    dataset = ds.dataset(str(DATA_ROOT), format="parquet", partitioning="hive")
    cols = ["trade_date", "species", "state", "origin", "spec", "packaging", "price_avg"]
    table = dataset.to_table(columns=cols)
    data = {col: table.column(col).to_pylist() for col in cols}
    n = len(data["trade_date"])
    print(f"{n:,} rows.\n")

    all_results = []
    series_data = {}

    for cfg in SPECIES_CONFIGS:
        print(f"{'='*60}")
        print(f"  {cfg.label}")
        print(f"  Filter: state={cfg.state}, pkg={cfg.packaging}, spec={cfg.spec}"
              f"{', domestic' if cfg.domestic_only else ''}")
        print(f"{'='*60}")

        daily = extract_daily_series(data, n, cfg)
        if len(daily) < 200:
            print(f"  SKIP — only {len(daily)} trading days (need 200+)\n")
            continue

        dates = list(daily.keys())
        prices = np.array(list(daily.values()))
        series_data[cfg.species] = {"dates": dates, "prices": prices.tolist()}

        print(f"  Series: {len(dates)} trading days ({dates[0]} ~ {dates[-1]})")
        print(f"  Price: mean={np.mean(prices):,.0f}, std={np.std(prices):,.0f}, "
              f"min={np.min(prices):,.0f}, max={np.max(prices):,.0f}")
        print(f"  Lag-1 autocorr: {np.corrcoef(prices[:-1], prices[1:])[0,1]:.4f}")

        # Backtest each model at 7-day and 14-day horizons
        models = [
            ("Naive", lambda train, h: forecast_naive(train, h)),
            ("SMA-7", lambda train, h: forecast_sma(train, h, 7)),
            ("SMA-30", lambda train, h: forecast_sma(train, h, 30)),
            ("ExpSmooth", lambda train, h: forecast_ema(train, h)),
            ("ARIMA(2,1,2)", lambda train, h: forecast_arima(train, h)),
        ]

        for horizon in [7, 14]:
            print(f"\n  --- {horizon}-day horizon ---")
            print(f"  {'Model':<15} {'MAPE':>7} {'RMSE':>8} {'MAE':>8} {'Dir%':>6} {'Tests':>6}")

            for name, fn in models:
                result = backtest(prices, fn, name, cfg.species, horizon=horizon)
                if result:
                    all_results.append(result)
                    print(f"  {name:<15} {result.mape:>6.1f}% {result.rmse:>8,} {result.mae:>8,} "
                          f"{result.directional_acc:>5.1f}% {result.n_tests:>5}")
                else:
                    print(f"  {name:<15} {'FAILED':>7}")

        # Generate latest forecast
        print(f"\n  --- Latest forecast (from {dates[-1]}) ---")
        for name, fn in models:
            pred = fn(prices, 7)
            if pred is not None:
                print(f"  {name:<15} 7d-ahead: {pred[-1]:>10,.0f} KRW "
                      f"(vs current {prices[-1]:,.0f}, change {(pred[-1]/prices[-1]-1)*100:+.1f}%)")

        print()

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Best Model Per Species (7-day horizon)")
    print("=" * 80)
    print(f"  {'Species':<15} {'Best Model':<15} {'MAPE':>7} {'RMSE':>8} {'Dir%':>6}")
    print("-" * 55)

    summary = {}
    for cfg in SPECIES_CONFIGS:
        sp_results = [r for r in all_results if r.species == cfg.species and r.horizon == 7]
        if not sp_results:
            continue
        best = min(sp_results, key=lambda r: r.mape)
        print(f"  {cfg.species:<15} {best.model:<15} {best.mape:>6.1f}% {best.rmse:>8,} {best.directional_acc:>5.1f}%")
        summary[cfg.species] = {
            "best_model": best.model, "mape_7d": best.mape, "rmse_7d": best.rmse,
            "directional_acc": best.directional_acc,
            "all_models": {r.model: {"mape": r.mape, "rmse": r.rmse, "dir_acc": r.directional_acc}
                          for r in sp_results},
        }

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now().isoformat(),
        "configs": [{
            "species": c.species, "state": c.state, "packaging": c.packaging,
            "spec": c.spec, "domestic_only": c.domestic_only, "label": c.label,
        } for c in SPECIES_CONFIGS],
        "backtest_results": [{
            "species": r.species, "model": r.model, "horizon": r.horizon,
            "mape": r.mape, "rmse": r.rmse, "mae": r.mae,
            "directional_acc": r.directional_acc, "n_tests": r.n_tests,
        } for r in all_results],
        "summary": summary,
        "series": series_data,
    }
    out_path = OUTPUT_DIR / "poc_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
