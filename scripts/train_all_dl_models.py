"""
Unified GPU Training: Deep Learning Model Comparison for Fish Price Prediction.

Trains and tests 6 DL architectures (+ TFT results loaded from separate run)
across 7 sashimi species, producing a comprehensive comparison table.

Models:
  1. GRU (2 layers, 64 hidden)
  2. LSTM (2 layers, 64 hidden)
  3. BiLSTM + Additive Attention
  4. CNN-LSTM (Conv1D + LSTM)
  5. TFT (loaded from train_tft.py results)
  6. Simplified Informer (Transformer encoder, direct multi-step decoder)
  7. PatchTST-style (patched input + Transformer encoder)

Runs inside Docker container with PyTorch + CUDA.

Usage (inside Docker):
    python scripts/train_all_dl_models.py

Usage (from host):
    docker run --gpus all -e NVIDIA_DISABLE_REQUIRE=1 --ipc=host ...
"""
import json
import math
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.dataset as ds
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "parquet" / "prices"
OUTPUT_DIR = PROJECT_ROOT / "data" / "poc_results"

# ── Constants ──────────────────────────────────────────────────────

LOOKBACK = 30
HORIZON = 7
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 10
LR = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
MIN_DAYS = 300

FOREIGN_KW = [
    "일본", "중국", "미국", "러시아", "캐나다", "노르웨이", "뉴질랜드", "대만", "칠레",
    "아르헨티나", "영국", "아일랜드", "온두라스", "북한", "(원양)", "인도", "인도네시아",
    "태국", "베트남", "필리핀", "호주", "스페인", "네덜란드", "페루", "모로코", "아프리카",
    "파키스탄", "라스팔마스", "포클랜드", "멕시코",
]

SPECIES_CONFIGS = [
    {"species": "넙치", "state": "활", "pkg": "kg", "spec": "중", "domestic": False},
    {"species": "우럭", "state": "활", "pkg": "kg", "spec": "중", "domestic": False},
    {"species": "방어", "state": "선", "pkg": "kg", "spec": "중", "domestic": True},
    {"species": "참돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True},
    {"species": "농어", "state": "활", "pkg": "kg", "spec": "중", "domestic": True},
    {"species": "도다리", "state": "활", "pkg": "kg", "spec": "중", "domestic": False},
    {"species": "감성돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True},
]

MODEL_NAMES = ["GRU", "LSTM", "BiLSTM+Attn", "CNN-LSTM", "TFT", "Transformer", "PatchTST"]


def is_foreign(origin: Optional[str]) -> bool:
    if not origin:
        return False
    return any(kw in origin for kw in FOREIGN_KW)


# ── Data Preparation ──────────────────────────────────────────────


def load_parquet_data() -> dict:
    """Load raw parquet data into column-oriented dict."""
    print("Loading parquet data...", end=" ", flush=True)
    dataset = ds.dataset(str(DATA_ROOT), format="parquet", partitioning="hive")
    cols = [
        "trade_date", "species", "state", "origin", "spec",
        "packaging", "price_avg", "quantity",
    ]
    table = dataset.to_table(columns=cols)
    data = {col: table.column(col).to_pylist() for col in cols}
    n = len(data["trade_date"])
    print(f"{n:,} rows.")
    return data


def build_species_daily_series(data: dict, cfg: dict) -> tuple[np.ndarray, list[str]]:
    """
    Extract daily average price for one species config.
    Returns gap-filled continuous daily price array and corresponding date strings.
    Forward-fills non-trading days.
    """
    n = len(data["trade_date"])
    day_prices = defaultdict(list)
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
        day_prices[data["trade_date"][i]].append(data["price_avg"][i])

    daily_avg = {d: float(np.mean(p)) for d, p in sorted(day_prices.items())}
    sorted_dates = sorted(daily_avg.keys())

    if len(sorted_dates) < MIN_DAYS:
        return np.array([]), []

    # Build continuous daily index with forward-fill
    first_dt = datetime.strptime(sorted_dates[0], "%Y.%m.%d")
    last_dt = datetime.strptime(sorted_dates[-1], "%Y.%m.%d")

    calendar_days = []
    cur = first_dt
    while cur <= last_dt:
        calendar_days.append(cur.strftime("%Y.%m.%d"))
        cur += timedelta(days=1)

    filled_prices = []
    filled_dates = []
    last_price = None
    for d in calendar_days:
        if d in daily_avg:
            last_price = daily_avg[d]
        if last_price is not None:
            filled_prices.append(last_price)
            filled_dates.append(d)

    return np.array(filled_prices, dtype=np.float64), filled_dates


def build_features(prices: np.ndarray, dates: list[str]) -> np.ndarray:
    """
    Build feature matrix from price series.
    Features per timestep:
      0: price (normalized)
      1: price_7d_avg
      2: price_30d_avg
      3-9: day_of_week one-hot (7)
      10: month_sin
      11: month_cos
    Total: 12 features
    """
    n = len(prices)

    # Simple moving averages
    sma7 = np.empty(n)
    sma30 = np.empty(n)
    for i in range(n):
        sma7[i] = np.mean(prices[max(0, i - 6):i + 1])
        sma30[i] = np.mean(prices[max(0, i - 29):i + 1])

    # Calendar features
    dow_onehot = np.zeros((n, 7))
    month_sin = np.empty(n)
    month_cos = np.empty(n)
    for i, d in enumerate(dates):
        dt = datetime.strptime(d, "%Y.%m.%d")
        dow_onehot[i, dt.weekday()] = 1.0
        month_sin[i] = math.sin(2 * math.pi * dt.month / 12)
        month_cos[i] = math.cos(2 * math.pi * dt.month / 12)

    features = np.column_stack([
        prices,
        sma7,
        sma30,
        dow_onehot,
        month_sin,
        month_cos,
    ])
    return features  # shape: (n, 12)


def normalize_features(train_features: np.ndarray, test_features: np.ndarray):
    """Per-feature z-score normalization. Returns normalized arrays and stats."""
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero
    train_norm = (train_features - mean) / std
    test_norm = (test_features - mean) / std
    return train_norm, test_norm, mean, std


# ── Dataset ───────────────────────────────────────────────────────


class SlidingWindowDataset(Dataset):
    """Sliding window dataset: (lookback, features) -> (horizon,) prices."""

    def __init__(self, features: np.ndarray, prices: np.ndarray,
                 lookback: int = LOOKBACK, horizon: int = HORIZON):
        self.X = []
        self.y = []
        for i in range(lookback, len(features) - horizon + 1):
            self.X.append(features[i - lookback:i])
            self.y.append(prices[i:i + horizon])
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])


# ── Models ────────────────────────────────────────────────────────


class GRUModel(nn.Module):
    """2-layer GRU encoder with linear decoder for multi-step output."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, horizon: int = HORIZON, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (batch, lookback, features)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])  # (batch, horizon)


class LSTMModel(nn.Module):
    """2-layer LSTM encoder with linear decoder for multi-step output."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, horizon: int = HORIZON, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM with additive attention and linear decoder."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, horizon: int = HORIZON, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        # Additive attention
        self.attn_w = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1, bias=False)
        self.fc = nn.Linear(hidden_size * 2, horizon)

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Additive attention: score = v^T tanh(W h_t)
        energy = torch.tanh(self.attn_w(lstm_out))  # (batch, seq_len, hidden)
        scores = self.attn_v(energy).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)

        # Weighted sum of encoder outputs
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden*2)
        return self.fc(context)  # (batch, horizon)


class CNNLSTMModel(nn.Module):
    """1D CNN for local pattern extraction followed by LSTM for temporal modeling."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, horizon: int = HORIZON,
                 cnn_filters: int = 32, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_filters),
        )
        self.lstm = nn.LSTM(cnn_filters, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (batch, seq_len, features)
        # Conv1d expects (batch, channels, seq_len)
        c = self.conv(x.permute(0, 2, 1))  # (batch, filters, seq_len)
        c = c.permute(0, 2, 1)  # (batch, seq_len, filters)
        out, _ = self.lstm(c)
        return self.fc(out[:, -1, :])


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # handle odd d_model
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SimpleTransformer(nn.Module):
    """
    Simplified Informer-style model.
    Standard Transformer encoder with direct multi-step linear decoder (non-autoregressive).
    """

    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, horizon: int = HORIZON, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        h = self.pos_enc(h)
        h = self.encoder(h)  # (batch, seq_len, d_model)
        return self.fc(h[:, -1, :])  # use last token


class PatchTransformer(nn.Module):
    """
    PatchTST-style model.
    Patches the input sequence into overlapping patches, then applies a Transformer encoder.
    Channel-independent: operates on the combined feature dimension per patch.
    """

    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, horizon: int = HORIZON,
                 patch_len: int = 7, stride: int = 3, seq_len: int = LOOKBACK,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        # Number of patches
        self.n_patches = (seq_len - patch_len) // stride + 1
        patch_dim = patch_len * input_size
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=self.n_patches + 10)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * self.n_patches, horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)
        patches = []
        for i in range(self.n_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :].reshape(batch_size, -1)  # (batch, patch_len * input_size)
            patches.append(patch)
        patches = torch.stack(patches, dim=1)  # (batch, n_patches, patch_dim)
        h = self.patch_proj(patches)  # (batch, n_patches, d_model)
        h = self.pos_enc(h)
        h = self.encoder(h)  # (batch, n_patches, d_model)
        h = h.reshape(batch_size, -1)  # (batch, n_patches * d_model)
        return self.fc(h)


# ── Training Utilities ────────────────────────────────────────────


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = EPOCHS, lr: float = LR, device: str = "cuda",
                patience: int = PATIENCE) -> dict:
    """
    Shared training loop with early stopping and LR scheduling.
    Returns dict with training metadata.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n_train = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            n_train += x.size(0)

        # Validate
        model.train(False)
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
                n_val += x.size(0)

        avg_train = train_loss / max(n_train, 1)
        avg_val = val_loss / max(n_val, 1)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_epoch": best_epoch, "best_val_loss": best_val_loss}


def run_test(model: nn.Module, test_loader: DataLoader, device: str,
             price_mean: float, price_std: float) -> dict:
    """
    Run model on test set and compute metrics.
    Returns MAPE (%), RMSE (original scale), and direction accuracy (%).
    price_mean/price_std are used to denormalize the price column (feature index 0).
    """
    model.train(False)
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_actuals.append(y.numpy())

    if not all_preds:
        return {"mape": 999.0, "rmse": 999.0, "dir_acc": 0.0, "n_samples": 0}

    preds = np.concatenate(all_preds, axis=0)   # (n, horizon)
    actuals = np.concatenate(all_actuals, axis=0)  # (n, horizon)

    # Denormalize: targets were normalized with price stats (feature index 0)
    preds_denorm = preds * price_std + price_mean
    actuals_denorm = actuals * price_std + price_mean

    # MAPE over all steps
    valid = actuals_denorm > 0
    if valid.any():
        mape = float(np.mean(
            np.abs(preds_denorm[valid] - actuals_denorm[valid]) / actuals_denorm[valid]
        )) * 100
    else:
        mape = 999.0

    # RMSE
    rmse = float(np.sqrt(np.mean((preds_denorm - actuals_denorm) ** 2)))

    # Direction accuracy: compare direction of day-1 prediction vs actual
    if preds_denorm.shape[0] > 1:
        # Direction: does next-day price go up or down relative to current window?
        # Use first horizon step
        pred_first = preds_denorm[:, 0]
        actual_first = actuals_denorm[:, 0]
        pred_dir = pred_first[1:] > pred_first[:-1]
        actual_dir = actual_first[1:] > actual_first[:-1]
        dir_acc = float(np.mean(pred_dir == actual_dir)) * 100
    else:
        dir_acc = 50.0

    return {
        "mape": round(mape, 2),
        "rmse": round(rmse, 0),
        "dir_acc": round(dir_acc, 1),
        "n_samples": len(preds),
    }


# ── Model Factory ─────────────────────────────────────────────────


def create_model(name: str, input_size: int) -> nn.Module:
    """Instantiate a model by name."""
    if name == "GRU":
        return GRUModel(input_size)
    elif name == "LSTM":
        return LSTMModel(input_size)
    elif name == "BiLSTM+Attn":
        return BiLSTMAttention(input_size)
    elif name == "CNN-LSTM":
        return CNNLSTMModel(input_size)
    elif name == "Transformer":
        return SimpleTransformer(input_size)
    elif name == "PatchTST":
        return PatchTransformer(input_size, seq_len=LOOKBACK)
    else:
        raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── TFT Results Loader ────────────────────────────────────────────


def load_tft_results() -> dict:
    """Load TFT results from a previous train_tft.py run, if available."""
    tft_path = OUTPUT_DIR / "tft_results.json"
    if not tft_path.exists():
        print("  [TFT] No results file found at", tft_path)
        print("  [TFT] Run 'python scripts/train_tft.py' separately first.")
        return {}
    with open(tft_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    for sp_name, sp_data in data.get("results", {}).items():
        if sp_name == "all_species":
            continue
        results[sp_name] = {
            "mape": sp_data.get("mape", 999.0),
            "rmse": 0.0,  # TFT results may not have RMSE
            "dir_acc": 0.0,
            "n_samples": sp_data.get("n_samples", 0),
        }
    return results


# ── Main Pipeline ─────────────────────────────────────────────────


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("  Unified DL Model Comparison -- Fish Price Prediction")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        mem_gb = props.total_memory / 1e9
        print(f"Memory: {mem_gb:.1f} GB")
    print(f"Lookback: {LOOKBACK}, Horizon: {HORIZON}, Batch: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}, Patience: {PATIENCE}, LR: {LR}")
    print()

    # Load data once
    data = load_parquet_data()

    # Models to train (TFT handled separately)
    trainable_models = ["GRU", "LSTM", "BiLSTM+Attn", "CNN-LSTM", "Transformer", "PatchTST"]

    # Results: {species: {model_name: {mape, rmse, dir_acc, ...}}}
    all_results = defaultdict(dict)
    timing = defaultdict(dict)

    # Load TFT results if available
    print("\n--- Loading TFT results ---")
    tft_results = load_tft_results()
    for sp_name, res in tft_results.items():
        all_results[sp_name]["TFT"] = res
        print(f"  {sp_name}: MAPE={res['mape']:.1f}%")
    if not tft_results:
        print("  (no TFT results available)")

    # Process each species
    for cfg in SPECIES_CONFIGS:
        sp = cfg["species"]
        print(f"\n{'=' * 70}")
        print(f"  Species: {sp} (state={cfg['state']}, spec={cfg['spec']})")
        print(f"{'=' * 70}")

        # Build daily series
        prices, dates = build_species_daily_series(data, cfg)
        if len(prices) < MIN_DAYS:
            print(f"  SKIP: insufficient data ({len(prices)} days < {MIN_DAYS})")
            continue

        # Build features
        features = build_features(prices, dates)
        n_features = features.shape[1]
        print(f"  Data: {len(prices)} continuous days, {n_features} features")

        # Train/test split: 80/20
        split_idx = int(len(features) * 0.8)
        train_feat = features[:split_idx]
        test_feat = features[split_idx:]
        train_prices = prices[:split_idx]
        test_prices = prices[split_idx:]

        # Normalize features
        train_norm, test_norm, feat_mean, feat_std = normalize_features(train_feat, test_feat)

        # Price stats for denormalization (feature index 0 = price)
        price_mean = feat_mean[0]
        price_std = feat_std[0]

        # Normalize targets too (using price column stats)
        train_prices_norm = (train_prices - price_mean) / price_std
        test_prices_norm = (test_prices - price_mean) / price_std

        # Create datasets
        train_ds = SlidingWindowDataset(train_norm, train_prices_norm)
        test_ds = SlidingWindowDataset(test_norm, test_prices_norm)

        if len(train_ds) < 50 or len(test_ds) < 10:
            print(f"  SKIP: insufficient samples (train={len(train_ds)}, test={len(test_ds)})")
            continue

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        print(f"  Samples: train={len(train_ds)}, test={len(test_ds)}")

        # Also create a validation split from training data for early stopping
        val_split = int(len(train_ds) * 0.9)
        train_subset = torch.utils.data.Subset(train_ds, range(val_split))
        val_subset = torch.utils.data.Subset(train_ds, range(val_split, len(train_ds)))
        train_sub_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_sub_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        for model_name in trainable_models:
            print(f"\n  [{model_name}]", end=" ", flush=True)
            t0 = time.time()

            try:
                model = create_model(model_name, n_features)
                n_params = count_parameters(model)
                print(f"({n_params:,} params)", end=" ", flush=True)

                train_info = train_model(
                    model, train_sub_loader, val_sub_loader,
                    epochs=EPOCHS, lr=LR, device=device, patience=PATIENCE,
                )
                print(f"ep={train_info['best_epoch']}", end=" ", flush=True)

                metrics = run_test(model, test_loader, device, price_mean, price_std)
                elapsed = time.time() - t0

                all_results[sp][model_name] = metrics
                timing[sp][model_name] = round(elapsed, 1)

                print(f"MAPE={metrics['mape']:.1f}% RMSE={metrics['rmse']:.0f} "
                      f"Dir={metrics['dir_acc']:.1f}% [{elapsed:.1f}s]")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAILED: {e} [{elapsed:.1f}s]")
                all_results[sp][model_name] = {
                    "mape": 999.0, "rmse": 999.0, "dir_acc": 0.0,
                    "n_samples": 0, "error": str(e),
                }
                timing[sp][model_name] = round(elapsed, 1)

            # Free GPU memory between models
            if device == "cuda":
                torch.cuda.empty_cache()

    # ── Results Summary ───────────────────────────────────────────

    print("\n")
    print("=" * 90)
    print("  RESULTS: MAPE (%) -- Lower is Better")
    print("=" * 90)

    # Header
    species_list = [cfg["species"] for cfg in SPECIES_CONFIGS if cfg["species"] in all_results]
    header = f"  {'Model':<15}"
    for sp in species_list:
        header += f" {sp:>8}"
    header += f" {'AVG':>8}"
    print(header)
    print("  " + "-" * (15 + 9 * (len(species_list) + 1)))

    # Per-model rows
    model_avgs = {}
    for model_name in MODEL_NAMES:
        row = f"  {model_name:<15}"
        mapes = []
        for sp in species_list:
            if model_name in all_results.get(sp, {}):
                mape = all_results[sp][model_name]["mape"]
                row += f" {mape:>7.1f}%"
                if mape < 900:
                    mapes.append(mape)
            else:
                row += f" {'N/A':>8}"
        avg_mape = np.mean(mapes) if mapes else 999.0
        model_avgs[model_name] = avg_mape
        row += f" {avg_mape:>7.1f}%"
        print(row)

    # Best per species
    print()
    print("  " + "-" * (15 + 9 * (len(species_list) + 1)))
    best_row = f"  {'BEST':<15}"
    for sp in species_list:
        sp_results = all_results.get(sp, {})
        if sp_results:
            best_model = min(sp_results, key=lambda m: sp_results[m].get("mape", 999))
            best_mape = sp_results[best_model]["mape"]
            best_row += f" {best_mape:>7.1f}%"
        else:
            best_row += f" {'N/A':>8}"
    best_row += f" {'':>8}"
    print(best_row)

    best_model_row = f"  {'(model)':<15}"
    for sp in species_list:
        sp_results = all_results.get(sp, {})
        if sp_results:
            best_model = min(sp_results, key=lambda m: sp_results[m].get("mape", 999))
            # Abbreviate long names
            abbrev = best_model[:8]
            best_model_row += f" {abbrev:>8}"
        else:
            best_model_row += f" {'N/A':>8}"
    best_model_row += f" {'':>8}"
    print(best_model_row)

    # Overall ranking
    print("\n")
    print("=" * 50)
    print("  OVERALL MODEL RANKING (by avg MAPE)")
    print("=" * 50)
    ranked = sorted(model_avgs.items(), key=lambda x: x[1])
    for rank, (model_name, avg) in enumerate(ranked, 1):
        marker = " <-- BEST" if rank == 1 else ""
        if avg < 900:
            print(f"  {rank}. {model_name:<15} {avg:>7.2f}%{marker}")
        else:
            print(f"  {rank}. {model_name:<15}     N/A{marker}")

    # Direction accuracy table
    print("\n")
    print("=" * 90)
    print("  RESULTS: Direction Accuracy (%) -- Higher is Better")
    print("=" * 90)
    header = f"  {'Model':<15}"
    for sp in species_list:
        header += f" {sp:>8}"
    print(header)
    print("  " + "-" * (15 + 9 * len(species_list)))
    for model_name in MODEL_NAMES:
        row = f"  {model_name:<15}"
        for sp in species_list:
            if model_name in all_results.get(sp, {}):
                da = all_results[sp][model_name].get("dir_acc", 0.0)
                row += f" {da:>7.1f}%"
            else:
                row += f" {'N/A':>8}"
        print(row)

    # Timing
    print("\n")
    print("=" * 90)
    print("  TRAINING TIME (seconds)")
    print("=" * 90)
    header = f"  {'Model':<15}"
    for sp in species_list:
        header += f" {sp:>8}"
    print(header)
    print("  " + "-" * (15 + 9 * len(species_list)))
    for model_name in trainable_models:
        row = f"  {model_name:<15}"
        for sp in species_list:
            t = timing.get(sp, {}).get(model_name, 0)
            row += f" {t:>7.1f}s"
        print(row)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "lr": LR,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "models": MODEL_NAMES,
            "species": [c["species"] for c in SPECIES_CONFIGS],
        },
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "N/A",
        "results": {
            sp: {
                model: all_results[sp][model]
                for model in all_results[sp]
            }
            for sp in all_results
        },
        "timing": dict(timing),
        "ranking": [
            {"rank": i + 1, "model": name, "avg_mape": round(avg, 2)}
            for i, (name, avg) in enumerate(ranked)
            if avg < 900
        ],
    }
    out_path = OUTPUT_DIR / "dl_comparison_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
