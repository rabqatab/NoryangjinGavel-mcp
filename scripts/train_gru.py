"""
GPU Training: VMD-GRU for Fish Price Prediction.

Per-mode GRU with optimized VMD decomposition.
Runs inside Docker container with PyTorch + CUDA.

Architecture:
  - STL decomposition: seasonal (period=7) + trend + residual
  - VMD on residual: K modes (Optuna-optimized)
  - Per-component GRU: 2 layers, 64 hidden, dropout 0.1
  - Recombine predictions by summation

Usage (inside Docker):
    python scripts/train_gru.py
"""
import json
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.seasonal import STL
from vmdpy import VMD
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
    {"species": "넙치", "state": "활", "pkg": "kg", "spec": "중", "domestic": False},
    {"species": "우럭", "state": "활", "pkg": "kg", "spec": "중", "domestic": False},
    {"species": "방어", "state": "선", "pkg": "kg", "spec": "중", "domestic": True},
    {"species": "참돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True},
    {"species": "농어", "state": "활", "pkg": "kg", "spec": "중", "domestic": True},
    {"species": "도다리", "state": "활", "pkg": "kg", "spec": "중", "domestic": False},
    {"species": "감성돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True},
]

def is_foreign(o):
    if not o: return False
    return any(kw in o for kw in FOREIGN_KW)


class PriceGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, lookback=30):
        self.X = []
        self.y = []
        for i in range(lookback, len(features)):
            self.X.append(features[i - lookback:i])
            self.y.append(targets[i])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])


def load_data():
    print("Loading parquet data...", end=" ", flush=True)
    dataset = ds.dataset(str(DATA_ROOT), format="parquet", partitioning="hive")
    cols = ["trade_date", "species", "state", "origin", "spec", "packaging", "price_avg", "quantity"]
    table = dataset.to_table(columns=cols)
    data = {col: table.column(col).to_pylist() for col in cols}
    print(f"{len(data['trade_date']):,} rows.")
    return data


def extract_daily_prices(data, n, cfg):
    day_prices = defaultdict(list)
    for i in range(n):
        if data["species"][i] != cfg["species"]: continue
        if data["state"][i] != cfg["state"]: continue
        if data["packaging"][i] != cfg["pkg"]: continue
        if data["spec"][i] != cfg["spec"]: continue
        if cfg["domestic"] and is_foreign(data["origin"][i]): continue
        day_prices[data["trade_date"][i]].append(data["price_avg"][i])
    return {d: float(np.mean(p)) for d, p in sorted(day_prices.items())}


def stl_vmd_decompose(prices, stl_period=7, vmd_K=5, vmd_alpha=2000):
    arr = np.array(prices)
    try:
        stl = STL(arr, period=stl_period, robust=True)
        result = stl.fit()
        seasonal = result.seasonal
        trend = result.trend
        residual = result.resid
    except Exception:
        trend = np.convolve(arr, np.ones(30)/30, mode="same")
        seasonal = np.zeros_like(arr)
        residual = arr - trend

    try:
        u, _, _ = VMD(residual, vmd_alpha, 0, vmd_K, 0, 1, 1e-7)
        modes = [u[k] for k in range(vmd_K)]
    except Exception:
        modes = [residual]

    return {"seasonal": seasonal, "trend": trend, "modes": modes}


def train_component_gru(component, split_idx, device, epochs=30, lr=0.001):
    features = component.reshape(-1, 1)
    train_f, val_f = features[:split_idx], features[split_idx:]
    train_t, val_t = component[:split_idx], component[split_idx:]

    train_ds = TimeSeriesDataset(train_f, train_t, lookback=30)
    val_ds = TimeSeriesDataset(val_f, val_t, lookback=30)

    if len(train_ds) < 10 or len(val_ds) < 5:
        return None, None

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = PriceGRU(input_size=1, hidden_size=64, num_layers=2, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(device)
            pred = model(x)
            preds.extend(pred.cpu().numpy().flatten())
            actuals.extend(y.numpy().flatten())

    return np.array(preds), np.array(actuals)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data = load_data()
    n = len(data["trade_date"])
    results = {}

    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        print(f"\n{'='*60}")
        print(f"  {sp}")
        print(f"{'='*60}")

        daily = extract_daily_prices(data, n, cfg)
        if len(daily) < 300:
            print(f"  Skip - {len(daily)} days")
            continue

        prices = np.array(list(daily.values()))
        print(f"  {len(prices)} days, mean={np.mean(prices):,.0f}")

        decomp = stl_vmd_decompose(prices, stl_period=7, vmd_K=5, vmd_alpha=2000)
        components = [decomp["seasonal"], decomp["trend"]] + decomp["modes"]
        print(f"  Components: seasonal + trend + {len(decomp['modes'])} VMD modes")

        split = int(len(prices) * 0.8)
        combined_preds = None
        combined_actuals = None

        for ci, component in enumerate(components):
            comp_name = ["seasonal", "trend"][ci] if ci < 2 else f"vmd_{ci-2}"
            preds, actuals = train_component_gru(component, split, device, epochs=30)

            if preds is None:
                print(f"  {comp_name}: failed")
                continue

            ml = min(len(preds), len(actuals))
            preds, actuals = preds[:ml], actuals[:ml]

            if combined_preds is None:
                combined_preds = preds.copy()
                combined_actuals = actuals.copy()
            else:
                ml2 = min(len(combined_preds), len(preds))
                combined_preds[:ml2] += preds[:ml2]
                combined_actuals[:ml2] += actuals[:ml2]

            comp_mape = float(np.mean(np.abs(preds - actuals) / np.where(np.abs(actuals) > 1, np.abs(actuals), 1))) * 100
            print(f"  {comp_name}: MAPE={comp_mape:.1f}%")

        if combined_preds is not None and len(combined_preds) > 0:
            mask = combined_actuals > 0
            mape = float(np.mean(np.abs(combined_preds[mask] - combined_actuals[mask]) / combined_actuals[mask])) * 100 if mask.any() else 999

            if len(combined_preds) > 1:
                actual_dir = combined_actuals[1:] > combined_actuals[:-1]
                pred_dir = combined_preds[1:] > combined_preds[:-1]
                dir_acc = float(np.mean(actual_dir == pred_dir)) * 100
            else:
                dir_acc = 50.0

            results[sp] = {"mape": round(mape, 2), "dir_acc": round(dir_acc, 1),
                          "n_components": len(components), "n_test": len(combined_preds)}
            print(f"\n  COMBINED: MAPE={mape:.1f}%, Dir={dir_acc:.1f}%")

    print("\n" + "=" * 60)
    print("STL-VMD-GRU RESULTS")
    print("=" * 60)
    print(f"\n  {'Species':<12} {'MAPE':>8} {'Dir%':>7}")
    for sp, r in results.items():
        print(f"  {sp:<12} {r['mape']:>7.1f}% {r['dir_acc']:>6.1f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now().isoformat(),
        "model": "STL-VMD-GRU",
        "device": device,
        "architecture": "STL(period=7) + VMD(K=5) + per-component GRU(2L, 64H)",
        "results": results,
    }
    with open(OUTPUT_DIR / "gru_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'gru_results.json'}")


if __name__ == "__main__":
    main()
