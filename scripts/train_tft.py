"""
GPU Training: Temporal Fusion Transformer (TFT) for Fish Price Prediction.

Runs inside Docker container with PyTorch + CUDA.
Uses pytorch-forecasting's TFT implementation.

Architecture:
  - Static covariates: species_id, packaging_type
  - Known future inputs: dow, month, woy, is_holiday, days_to_seollal, days_to_chuseok
  - Observed past inputs: price, quantity, n_lots, n_origins, ema_7, ema_30, rsi_14,
                           own_qty_7d, other_sashimi_7d, market_lots_7d
  - Encoder length: 30 days lookback
  - Prediction horizon: 7 days
  - Quantile outputs: 10%, 50%, 90%

Usage (inside Docker):
    python scripts/train_tft.py

Usage (from host):
    ./docker/run_gpu_training.sh tft
"""
import json
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MAPE
import pyarrow.dataset as ds

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "parquet" / "prices"
OUTPUT_DIR = PROJECT_ROOT / "data" / "poc_results"

FOREIGN_KW = ['일본','중국','미국','러시아','캐나다','노르웨이','뉴질랜드','대만','칠레',
              '아르헨티나','영국','아일랜드','온두라스','북한','(원양)','인도','인도네시아',
              '태국','베트남','필리핀','호주','스페인','네덜란드','페루','모로코','아프리카',
              '파키스탄','라스팔마스','포클랜드','멕시코']

SPECIES_CONFIGS = [
    {"species": "넙치", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "id": 0},
    {"species": "우럭", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "id": 1},
    {"species": "방어", "state": "선", "pkg": "kg", "spec": "중", "domestic": True, "id": 2},
    {"species": "참돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True, "id": 3},
    {"species": "농어", "state": "활", "pkg": "kg", "spec": "중", "domestic": True, "id": 4},
    {"species": "도다리", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "id": 5},
    {"species": "감성돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True, "id": 6},
]

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


def is_foreign(o):
    if not o: return False
    return any(kw in o for kw in FOREIGN_KW)


def parse_date(d):
    return datetime.strptime(d, "%Y.%m.%d")


def days_to_holiday(dt):
    r = {"seollal": 999, "chuseok": 999}
    for y in [dt.year - 1, dt.year, dt.year + 1]:
        if y not in KOREAN_HOLIDAYS: continue
        for name, hd in KOREAN_HOLIDAYS[y].items():
            diff = (parse_date(hd) - dt).days
            if abs(diff) < abs(r[name]): r[name] = diff
    return r


def ema(prices, span):
    a = np.array(prices, dtype=float)
    out = np.empty_like(a)
    out[0] = a[0]
    alpha = 2 / (span + 1)
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i-1]
    return out


def rsi(prices, period=14):
    a = np.array(prices, dtype=float)
    deltas = np.diff(a)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    out = np.full(len(a), 50.0)
    if len(gains) < period: return out
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0: out[i + 1] = 100.0
        else: out[i + 1] = 100 - (100 / (1 + avg_gain / avg_loss))
    return out


# ── Data Preparation ────────────────────────────────────────────────

def build_tft_dataframe():
    """Build a pandas-like dict-of-lists for TFT TimeSeriesDataSet."""
    import pandas as pd

    print("Loading parquet data...", end=" ", flush=True)
    dataset = ds.dataset(str(DATA_ROOT), format="parquet", partitioning="hive")
    cols = ["trade_date", "species", "state", "origin", "spec", "packaging",
            "price_avg", "price_high", "price_low", "quantity"]
    table = dataset.to_table(columns=cols)
    data = {col: table.column(col).to_pylist() for col in cols}
    n = len(data["trade_date"])
    print(f"{n:,} rows.")

    # Build market-wide supply context
    all_dates = sorted(set(data["trade_date"]))
    date_idx = {d: i for i, d in enumerate(all_dates)}
    nd = len(all_dates)

    sashimi_species = [c["species"] for c in SPECIES_CONFIGS]
    sp_qty = {sp: np.zeros(nd) for sp in sashimi_species}
    market_lots = np.zeros(nd)
    for i in range(n):
        di = date_idx[data["trade_date"][i]]
        market_lots[di] += 1
        if data["species"][i] in sp_qty:
            sp_qty[data["species"][i]][di] += data["quantity"][i]

    total_sashimi = sum(sp_qty.values())
    k = 7
    sp_qty_7d = {s: np.convolve(q, np.ones(k)/k, mode="same") for s, q in sp_qty.items()}
    market_lots_7d = np.convolve(market_lots, np.ones(k)/k, mode="same")
    total_sashimi_7d = np.convolve(total_sashimi, np.ones(k)/k, mode="same")

    # Build per-species daily records
    rows = []
    time_idx_counter = {}

    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        day_data = defaultdict(lambda: {"prices": [], "highs": [], "lows": [], "origins": set(), "qty": 0})

        for i in range(n):
            if data["species"][i] != sp: continue
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

        sorted_dates = sorted(day_data.keys())
        if len(sorted_dates) < 200:
            print(f"  {sp}: skip ({len(sorted_dates)} days)")
            continue

        prices = [float(np.mean(day_data[d]["prices"])) for d in sorted_dates]
        ema7 = ema(prices, 7)
        ema30 = ema(prices, 30)
        rsi14 = rsi(prices, 14)

        for idx, d in enumerate(sorted_dates):
            if idx < 30: continue  # Need lookback

            dt = parse_date(d)
            di = date_idx[d]
            hol = days_to_holiday(dt)
            dd = day_data[d]

            # Time index (monotonic per group)
            if sp not in time_idx_counter:
                time_idx_counter[sp] = 0
            else:
                time_idx_counter[sp] += 1

            row = {
                # Identifiers
                "group_id": sp,
                "species_id": cfg["id"],
                "time_idx": time_idx_counter[sp],
                "date": d,

                # Target
                "price": prices[idx],

                # Known future (calendar)
                "dow": dt.weekday(),
                "month": dt.month,
                "woy": dt.isocalendar()[1],
                "is_weekend": int(dt.weekday() >= 5),
                "days_to_seollal": hol["seollal"],
                "days_to_chuseok": hol["chuseok"],

                # Observed past (price-derived)
                "ema_7": ema7[idx],
                "ema_30": ema30[idx],
                "rsi_14": rsi14[idx],
                "price_7d_avg": np.mean(prices[max(0, idx-7):idx]),
                "price_30d_avg": np.mean(prices[max(0, idx-30):idx]),
                "price_std_7d": np.std(prices[max(0, idx-7):idx]),

                # Observed past (supply)
                "n_lots": len(dd["prices"]),
                "n_origins": len(dd["origins"]),
                "quantity": dd["qty"],
                "own_qty_7d": sp_qty_7d[sp][di],
                "other_sashimi_7d": total_sashimi_7d[di] - sp_qty_7d[sp][di],
                "market_lots_7d": market_lots_7d[di],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Built TFT dataframe: {len(df):,} rows, {df['group_id'].nunique()} species")
    return df


def train_and_evaluate():
    """Train TFT and evaluate per species."""
    import pandas as pd

    df = build_tft_dataframe()

    # Train/test split by time_idx (last 20% as test per group)
    # Use each group's own 80% mark, then take the minimum to ensure all groups have test data
    group_maxes = df.groupby("group_id")["time_idx"].max()
    train_cutoff = int(group_maxes.min() * 0.8)
    print(f"  Train cutoff: time_idx={train_cutoff} (min group max={group_maxes.min()})")

    max_encoder_length = 30
    max_prediction_length = 7

    # Create TimeSeriesDataSet
    training = TimeSeriesDataSet(
        df[df["time_idx"] <= train_cutoff],
        time_idx="time_idx",
        target="price",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["group_id"],
        time_varying_known_reals=["dow", "month", "woy", "is_weekend",
                                   "days_to_seollal", "days_to_chuseok"],
        time_varying_unknown_reals=["price", "ema_7", "ema_30", "rsi_14",
                                     "price_7d_avg", "price_30d_avg", "price_std_7d",
                                     "n_lots", "n_origins", "quantity",
                                     "own_qty_7d", "other_sashimi_7d", "market_lots_7d"],
        target_normalizer=GroupNormalizer(groups=["group_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[df["time_idx"] > train_cutoff],
        predict=True,
        stop_randomization=True,
    )

    train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    print(f"\nTraining TFT: {len(training)} train samples, {len(validation)} val samples")
    print(f"Encoder length: {max_encoder_length}, Prediction length: {max_prediction_length}")

    # Create model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,  # quantiles
        loss=QuantileLoss(),
        log_interval=50,
        reduce_on_plateau_patience=4,
    )

    print(f"Model parameters: {tft.size() / 1e3:.1f}K")

    # Train
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Evaluate per species using raw predictions
    print("\n=== Evaluation ===")
    raw_preds = tft.predict(val_dataloader, mode="raw", return_x=True)

    # Collect predictions and actuals per batch
    all_preds = []
    all_actuals = []
    all_groups = []

    for x, y in val_dataloader:
        batch_preds = tft.predict(x, mode="prediction")
        # batch_preds shape: (batch, horizon) — point predictions (median)
        all_preds.append(batch_preds.cpu().numpy())
        all_actuals.append(y[0].cpu().numpy())
        # Get group ids from decoder input
        all_groups.extend(x["groups"].cpu().numpy().flatten().tolist())

    pred_arr = np.concatenate(all_preds, axis=0)
    actual_arr = np.concatenate(all_actuals, axis=0)

    # Map group indices back to species names
    group_names = training.decoded_index["group_id"].unique().tolist() if hasattr(training, "decoded_index") else [c["species"] for c in SPECIES_CONFIGS]

    # Overall MAPE (using first prediction step as point forecast)
    pred_flat = pred_arr[:, 0] if pred_arr.ndim > 1 else pred_arr
    actual_flat = actual_arr[:, 0] if actual_arr.ndim > 1 else actual_arr

    results = {}

    # Per-species using group indices
    for gi, sp_name in enumerate(group_names):
        mask = np.array(all_groups) == gi
        if mask.sum() == 0:
            continue
        sp_pred = pred_flat[mask]
        sp_actual = actual_flat[mask]
        valid = sp_actual > 0
        if valid.sum() == 0:
            continue
        mape = float(np.mean(np.abs(sp_pred[valid] - sp_actual[valid]) / sp_actual[valid])) * 100
        results[sp_name] = {"mape": round(mape, 2), "n_samples": int(mask.sum())}
        print(f"  {sp_name}: MAPE = {mape:.1f}% (n={mask.sum()})")

    # If group mapping failed, compute global MAPE
    if not results:
        valid = actual_flat > 0
        mape = float(np.mean(np.abs(pred_flat[valid] - actual_flat[valid]) / actual_flat[valid])) * 100
        results["all_species"] = {"mape": round(mape, 2), "n_samples": len(pred_flat)}
        print(f"  Global MAPE = {mape:.1f}% (n={len(pred_flat)})")

    # Feature importance via attention weights
    try:
        interpretation = tft.interpret_output(predictions.output, reduction="sum")
        print("\n=== Variable Importance (from attention) ===")
        for key in ["encoder_variables", "decoder_variables", "static_variables"]:
            if key in interpretation:
                print(f"\n  {key}:")
                imp = interpretation[key]
                if hasattr(imp, "items"):
                    for name, val in sorted(imp.items(), key=lambda x: -x[1]):
                        print(f"    {name}: {val:.3f}")
    except Exception as e:
        print(f"\n  (Interpretation skipped: {e})")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now().isoformat(),
        "model": "TFT (Temporal Fusion Transformer)",
        "device": str(tft.device),
        "parameters": tft.size(),
        "epochs": trainer.current_epoch,
        "encoder_length": max_encoder_length,
        "prediction_length": max_prediction_length,
        "results": results,
    }
    out_path = OUTPUT_DIR / "tft_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")

    return results


if __name__ == "__main__":
    print("=== TFT Training on GB10 GPU ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    results = train_and_evaluate()

    print("\n=== SUMMARY ===")
    for sp, r in results.items():
        print(f"  {sp}: {r['mape']:.1f}% MAPE")
