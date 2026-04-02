"""
DL v2: Enhanced GPU Training with Optuna HPO, Per-Config Loss, CQR, and Ensemble.

Improvements over v1 (train_all_dl_models.py):
  1. Per-config loss function selection (MAE/LogCosh/Huber based on loss comparison study)
  2. Optuna HPO: tune hidden_size, num_layers, lr, dropout per config (10 trials)
  3. CQR calibration with tunable alpha per config
  4. Ensemble: average top-3 models per config
  5. Recent data weighting: exponential decay in training loss

Runs inside Docker container with PyTorch + CUDA + Optuna.

Usage (inside Docker):
    python scripts/train_dl_v2.py

Dual-node:
    CONFIG_SLICE=0:10  python scripts/train_dl_v2.py   # Node 1
    CONFIG_SLICE=10:20 python scripts/train_dl_v2.py   # Node 2
"""

import json
import math
import os
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pyarrow.dataset as ds
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, Dataset

optuna.logging.set_verbosity(optuna.logging.WARNING)
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
MIN_DAYS = 300

# Optuna settings
OPTUNA_TRIALS = 10
OPTUNA_EPOCHS = 30  # Shorter epochs for HPO trials

FOREIGN_KW = [
    "일본", "중국", "미국", "러시아", "캐나다", "노르웨이", "뉴질랜드", "대만", "칠레",
    "아르헨티나", "영국", "아일랜드", "온두라스", "북한", "(원양)", "인도", "인도네시아",
    "태국", "베트남", "필리핀", "호주", "스페인", "네덜란드", "페루", "모로코", "아프리카",
    "파키스탄", "라스팔마스", "포클랜드", "멕시코",
]

SASHIMI_SPECIES = ["넙치", "우럭", "방어", "참돔", "농어", "도다리", "감성돔",
                    "감숭어", "참숭어", "쭈꾸미", "민어", "깐굴", "바위굴", "수꽃게", "암꽃게"]

SPECIES_CONFIGS = [
    {"id": "넙치_활_kg_중", "species": "넙치", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False},
    {"id": "우럭_활_kg_중", "species": "우럭", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False},
    {"id": "방어_선_kg_중_dom", "species": "방어", "state": "선", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": True, "regime_split": True},
    {"id": "참돔_활_kg_중_dom", "species": "참돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": False},
    {"id": "농어_활_kg_중_dom", "species": "농어", "state": "활", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": False},
    {"id": "도다리_활_kg_중", "species": "도다리", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": True},
    {"id": "감성돔_활_kg_중_dom", "species": "감성돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True, "smoothed": False},
    {"id": "감숭어_활_kg_중", "species": "감숭어", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False},
    {"id": "참숭어_활_kg_중", "species": "참숭어", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False},
    {"id": "쭈꾸미_선_box_중_dom", "species": "쭈꾸미", "state": "선", "pkg": "box", "spec": "중", "domestic": True, "smoothed": False},
    {"id": "민어_선_SP_중", "species": "민어", "state": "선", "pkg": "S/P", "spec": "중", "domestic": False, "smoothed": False},
    {"id": "깐굴_선_box_소", "species": "깐굴", "state": "선", "pkg": "box", "spec": "소", "domestic": False, "smoothed": False},
    {"id": "바위굴_활_box_대", "species": "바위굴", "state": "활", "pkg": "box", "spec": "대", "domestic": False, "smoothed": False},
    {"id": "수꽃게_활_kg_중", "species": "수꽃게", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False},
    {"id": "암꽃게_활_kg_중", "species": "암꽃게", "state": "활", "pkg": "kg", "spec": "중", "domestic": False, "smoothed": False},
    {"id": "수꽃게_활_kg_대", "species": "수꽃게", "state": "활", "pkg": "kg", "spec": "대", "domestic": False, "smoothed": False},
    {"id": "암꽃게_활_kg_대", "species": "암꽃게", "state": "활", "pkg": "kg", "spec": "대", "domestic": False, "smoothed": False},
    {"id": "넙치_활_kg_2미", "species": "넙치", "state": "활", "pkg": "kg", "spec": "2미", "domestic": False, "smoothed": False},
    {"id": "참돔_활_kg_2미_dom", "species": "참돔", "state": "활", "pkg": "kg", "spec": "2미", "domestic": True, "smoothed": False},
    {"id": "농어_활_kg_1미_dom", "species": "농어", "state": "활", "pkg": "kg", "spec": "1미", "domestic": True, "smoothed": False},
]

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

# ── Per-config loss routing (from loss comparison study) ──────────
# For configs not in the study, use per-model defaults

LOSS_ROUTING = {
    # config_id → {model_name → loss_name}
    "넙치_활_kg_중": {"GRU": "MAE", "Transformer": "LogCosh", "CNN-LSTM": "MAE"},
    "우럭_활_kg_중": {"GRU": "MAE", "Transformer": "Huber", "CNN-LSTM": "MAE"},
    "방어_선_kg_중_dom": {"GRU": "LogCosh", "Transformer": "MSE", "CNN-LSTM": "sMAPE"},
    "참돔_활_kg_중_dom": {"GRU": "MSE", "Transformer": "MAE", "CNN-LSTM": "Huber"},
    "농어_활_kg_중_dom": {"GRU": "MAE", "Transformer": "Huber", "CNN-LSTM": "MAE"},
    "도다리_활_kg_중": {"GRU": "MAE", "Transformer": "MAE", "CNN-LSTM": "Huber"},
    "감성돔_활_kg_중_dom": {"GRU": "MAE", "Transformer": "Huber", "CNN-LSTM": "LogCosh"},
    "감숭어_활_kg_중": {"GRU": "MSE", "Transformer": "LogCosh", "CNN-LSTM": "Huber"},
    "참숭어_활_kg_중": {"GRU": "LogCosh", "Transformer": "LogCosh", "CNN-LSTM": "LogCosh"},
    "쭈꾸미_선_box_중_dom": {"GRU": "MAE", "Transformer": "MAE", "CNN-LSTM": "MAE"},
}

# Defaults for configs not in the study
DEFAULT_LOSS_PER_MODEL = {
    "GRU": "MAE", "LSTM": "MAE", "BiLSTM+Attn": "MAE",
    "CNN-LSTM": "MAE", "Transformer": "LogCosh", "PatchTST": "LogCosh",
}


# ── Loss Functions ─────────────────────────────────────────────────


class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))


class SmoothedMAPELoss(nn.Module):
    def __init__(self, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target) / (torch.abs(target) + self.epsilon))


class PinballLoss(nn.Module):
    """Quantile regression loss for simultaneous multi-quantile prediction."""

    def __init__(self, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred, actual):
        actual = actual.unsqueeze(-1) if actual.dim() == 1 else actual
        losses = []
        for i, q in enumerate(self.quantiles):
            diff = actual - pred[:, i:i + 1]
            loss = torch.where(diff >= 0, q * diff, (q - 1) * diff)
            losses.append(loss.mean())
        return sum(losses) / len(losses)


def get_loss_fn(name: str) -> nn.Module:
    """Create a loss function by name."""
    mapping = {
        "MSE": nn.MSELoss,
        "MAE": nn.L1Loss,
        "Huber": nn.HuberLoss,
        "LogCosh": LogCoshLoss,
        "sMAPE": SmoothedMAPELoss,
    }
    return mapping.get(name, nn.L1Loss)()


def get_config_loss(config_id: str, model_base: str) -> nn.Module:
    """Get the optimal loss function for a (config, model) pair."""
    # Strip +VMD suffix for routing lookup
    base = model_base.replace("+VMD", "")
    if config_id in LOSS_ROUTING and base in LOSS_ROUTING[config_id]:
        loss_name = LOSS_ROUTING[config_id][base]
    else:
        loss_name = DEFAULT_LOSS_PER_MODEL.get(base, "MAE")
    return get_loss_fn(loss_name), loss_name


# ── Helpers (same as v1) ──────────────────────────────────────────


def is_foreign(origin: Optional[str]) -> bool:
    if not origin:
        return False
    return any(kw in origin for kw in FOREIGN_KW)


def parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y.%m.%d")


def days_to_holiday(dt: datetime) -> dict:
    r = {"seollal": 999, "chuseok": 999}
    for y in [dt.year - 1, dt.year, dt.year + 1]:
        if y not in KOREAN_HOLIDAYS:
            continue
        for name, hd in KOREAN_HOLIDAYS[y].items():
            diff = (parse_date(hd) - dt).days
            if abs(diff) < abs(r[name]):
                r[name] = diff
    return r


def ema(prices, span):
    a = np.array(prices, dtype=float)
    out = np.empty_like(a)
    out[0] = a[0]
    alpha = 2 / (span + 1)
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
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


def winsorized_daily_price(day_prices, recent_30d_prices):
    if len(recent_30d_prices) < 10:
        return float(np.mean(day_prices))
    p10, p90 = np.percentile(recent_30d_prices, [10, 90])
    clipped = [max(p10, min(p90, p)) for p in day_prices]
    return float(np.mean(clipped))


def origin_weight(origin, origin_freq_30d, max_freq_30d):
    freq = origin_freq_30d.get(origin, 1)
    return freq / max_freq_30d if max_freq_30d > 0 else 1.0


def weighted_mean(prices, weights):
    w = np.array(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        return float(np.mean(prices))
    return float(np.dot(prices, w) / total)


def adaptive_vmd_k(prices, window=90):
    a = np.array(prices, dtype=float)
    if len(a) < window:
        return 3
    recent_std = np.std(a[-window:])
    overall_std = np.std(a)
    return 5 if recent_std > overall_std else 3


def flag_outlier_days(prices, window=30, n_sigma=3):
    is_outlier = np.zeros(len(prices), dtype=bool)
    for i in range(window, len(prices)):
        window_prices = prices[max(0, i - window):i]
        mu = np.mean(window_prices)
        sigma = np.std(window_prices)
        if sigma > 0 and abs(prices[i] - mu) > n_sigma * sigma:
            is_outlier[i] = True
    return is_outlier


def decompose_vmd(series, K=3, alpha=2000):
    try:
        from vmdpy import VMD
        u, _, _ = VMD(series, alpha, 0, K, 0, 1, 1e-7)
        return [u[k] for k in range(K)]
    except Exception:
        return [series]


# ── Import from v1 ────────────────────────────────────────────────
# v1 module has all data prep, feature engineering, model defs, and datasets.
# We import them to avoid duplicating ~1500 lines of code.

import sys
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import train_all_dl_models as v1

load_parquet_data = v1.load_parquet_data
build_supply_context = v1.build_supply_context
build_species_daily_series = v1.build_species_daily_series
_build_features_68 = v1.build_features_68


def build_features(series, ctx, target_sp):
    """Wrapper that handles species not in the supply context."""
    # Ensure target species exists in supply context arrays
    nd = len(ctx["dates"])
    zero = np.zeros(nd)
    for key in ["sp_qty", "sp_lots", "sp_qty_7d", "sp_lots_7d"]:
        if target_sp not in ctx[key]:
            ctx[key][target_sp] = zero
    return _build_features_68(series, ctx, target_sp)
SlidingWindowDataset = v1.SlidingWindowDataset
QuantileDataset = v1.QuantileDataset

PositionalEncoding = v1.PositionalEncoding
GRUModel = v1.GRUModel
LSTMModel = v1.LSTMModel
BiLSTMAttention = v1.BiLSTMAttention
CNNLSTMModel = v1.CNNLSTMModel
SimpleTransformer = v1.SimpleTransformer
PatchTransformer = v1.PatchTransformer
GRUQuantile = v1.GRUQuantile
TransformerQuantile = v1.TransformerQuantile
CNNLSTMQuantile = v1.CNNLSTMQuantile


# ── Weather Feature Integration ───────────────────────────────────

# Map species to closest coastal weather station
SPECIES_PORT_MAP = {
    "넙치": "busan", "우럭": "busan", "도다리": "yeosu",
    "방어": "jeju", "참돔": "jeju", "감성돔": "yeosu",
    "농어": "incheon", "감숭어": "incheon", "참숭어": "incheon",
    "쭈꾸미": "incheon", "민어": "yeosu",
    "깐굴": "incheon", "바위굴": "yeosu",
    "수꽃게": "incheon", "암꽃게": "incheon",
}

WEATHER_FEATURES = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
    "pressure_msl_mean", "sunshine_duration",
]


def load_weather_data() -> dict:
    """Load coastal weather CSV into a dict: {location: {date_str: {feature: value}}}."""
    csv_path = PROJECT_ROOT / "data" / "weather" / "coastal_weather_daily.csv"
    if not csv_path.exists():
        return {}

    import csv
    weather = defaultdict(dict)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            loc = row["location"]
            date = row["date"]
            vals = {}
            for feat in WEATHER_FEATURES:
                v = row.get(feat)
                vals[feat] = float(v) if v and v != "None" and v != "" else None
            weather[loc][date] = vals

    return dict(weather)


def get_weather_features(weather_data: dict, species: str, dates: list[str]) -> np.ndarray:
    """
    Build weather feature array for a species based on nearest port.
    Returns (n_dates, 8) array. Missing values are forward-filled.
    """
    port = SPECIES_PORT_MAP.get(species, "busan")
    port_data = weather_data.get(port, {})

    n = len(dates)
    n_feats = len(WEATHER_FEATURES)
    features = np.full((n, n_feats), np.nan)

    for i, date in enumerate(dates):
        day_data = port_data.get(date)
        if day_data:
            for j, feat in enumerate(WEATHER_FEATURES):
                v = day_data.get(feat)
                if v is not None:
                    features[i, j] = v

    # Forward-fill NaN values
    for j in range(n_feats):
        col = features[:, j]
        last_valid = np.nan
        for i in range(n):
            if not np.isnan(col[i]):
                last_valid = col[i]
            elif not np.isnan(last_valid):
                col[i] = last_valid

    # Fill remaining NaN with column mean
    for j in range(n_feats):
        col = features[:, j]
        mask = np.isnan(col)
        if mask.all():
            features[:, j] = 0.0
        elif mask.any():
            features[:, j][mask] = np.nanmean(col)

    return features


def create_model(name: str, n_features: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.1) -> nn.Module:
    """Create a model by name with configurable hyperparams."""
    if name == "GRU":
        return GRUModel(n_features, hidden_size, num_layers)
    elif name == "LSTM":
        return LSTMModel(n_features, hidden_size, num_layers)
    elif name == "BiLSTM+Attn":
        return BiLSTMAttention(n_features, hidden_size, num_layers)
    elif name == "CNN-LSTM":
        return CNNLSTMModel(n_features, hidden_size, num_layers)
    elif name == "Transformer":
        return SimpleTransformer(n_features, d_model=hidden_size, num_layers=num_layers, dropout=dropout)
    elif name == "PatchTST":
        return PatchTransformer(n_features, d_model=hidden_size, num_layers=num_layers, dropout=dropout)
    raise ValueError(f"Unknown model: {name}")


# ── Enhanced Training with Per-Config Loss ────────────────────────


def train_model_v2(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                   criterion: nn.Module, epochs: int = EPOCHS, lr: float = 0.001,
                   device: str = "cuda", patience: int = PATIENCE,
                   sample_weights: np.ndarray = None) -> dict:
    """
    Training loop with configurable loss function and optional sample weighting.
    Returns dict with training metadata.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
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

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_epoch": best_epoch, "best_val_loss": float(best_val_loss)}


def run_test(model: nn.Module, test_loader: DataLoader, device: str,
             price_mean: float, price_std: float) -> dict:
    """Run model on test set and compute metrics on raw (exp'd) prices."""
    model.train(False)
    all_preds, all_actuals = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_actuals.append(y.numpy())

    if not all_preds:
        return {"mape": 999.0, "rmse": 999.0, "mae": 999.0, "dir_acc": 0.0, "n_samples": 0}

    preds = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)

    preds_denorm = np.exp(preds * price_std + price_mean)
    actuals_denorm = np.exp(actuals * price_std + price_mean)

    valid = actuals_denorm > 0
    mape = float(np.mean(np.abs(preds_denorm[valid] - actuals_denorm[valid]) / actuals_denorm[valid])) * 100 if valid.any() else 999.0
    rmse = float(np.sqrt(np.mean((preds_denorm - actuals_denorm) ** 2)))
    mae = float(np.mean(np.abs(preds_denorm - actuals_denorm)))

    if preds_denorm.shape[0] > 1:
        pred_first = preds_denorm[:, 0]
        actual_first = actuals_denorm[:, 0]
        pred_dir = pred_first[1:] > pred_first[:-1]
        actual_dir = actual_first[1:] > actual_first[:-1]
        dir_acc = float(np.mean(pred_dir == actual_dir)) * 100
    else:
        dir_acc = 50.0

    return {"mape": round(mape, 2), "rmse": round(rmse, 0), "mae": round(mae, 0),
            "dir_acc": round(dir_acc, 1), "n_samples": len(preds)}


# ── Optuna HPO ─────────────────────────────────────────────────────


def optuna_hpo(model_name: str, n_features: int, train_ds, val_ds,
               criterion: nn.Module, device: str, n_trials: int = OPTUNA_TRIALS,
               price_mean: float = 0, price_std: float = 1) -> dict:
    """Run Optuna HPO to find best hyperparams for a model on one config."""

    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = create_model(model_name, n_features, hidden_size, num_layers, dropout)
        info = train_model_v2(
            model, train_loader, val_loader, criterion,
            epochs=OPTUNA_EPOCHS, lr=lr, device=device, patience=7,
        )
        return info["best_val_loss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=300)

    return study.best_params


# ── Quantile Training ──────────────────────────────────────────────


def train_quantile_model(model: nn.Module, train_loader: DataLoader,
                         val_loader: DataLoader, device: str,
                         epochs: int = EPOCHS, lr: float = 0.001,
                         patience: int = PATIENCE) -> dict:
    """Train quantile model with PinballLoss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    criterion = PinballLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
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

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_epoch": best_epoch, "best_val_loss": float(best_val_loss)}


# ── Evaluation helpers ─────────────────────────────────────────────


def evaluate_bands(pred_q10, pred_q50, pred_q90, actuals):
    coverage = np.mean((actuals >= pred_q10) & (actuals <= pred_q90)) * 100
    band_width = np.mean(pred_q90 - pred_q10)
    band_pct = np.mean((pred_q90 - pred_q10) / np.where(pred_q50 > 0, pred_q50, 1)) * 100
    mape_p50 = np.mean(np.abs(pred_q50 - actuals) / np.where(actuals > 0, actuals, 1)) * 100
    return {
        "mape_p50": round(mape_p50, 1), "coverage": round(coverage, 1),
        "band_width_avg": round(float(band_width)), "band_pct": round(band_pct, 1),
    }


def compute_cqr_bands(pred_q10, pred_q50, pred_q90, actuals, alpha=0.1):
    """Conformalized Quantile Regression — asymmetric bands."""
    n = len(pred_q50)
    cal_size = int(n * 0.8)
    if cal_size < 10:
        return pred_q10, pred_q90, 0.0, 0.0

    # CQR scores: max(q10 - actual, actual - q90, 0)
    cal_scores = np.maximum(pred_q10[:cal_size] - actuals[:cal_size],
                            actuals[:cal_size] - pred_q90[:cal_size])
    q_hat = np.quantile(cal_scores, 1 - alpha)

    # Adjusted bands
    adj_lo = pred_q10 - q_hat
    adj_hi = pred_q90 + q_hat

    # Evaluate on remaining 20%
    test_act = actuals[cal_size:]
    test_lo = adj_lo[cal_size:]
    test_hi = adj_hi[cal_size:]
    if len(test_act) > 0:
        coverage = float(np.mean((test_act >= test_lo) & (test_act <= test_hi))) * 100
        width = float(np.mean(test_hi - test_lo))
    else:
        coverage = 0.0
        width = 0.0

    return adj_lo, adj_hi, coverage, width


# ── Main ───────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"DL v2 Training — Per-Config Loss + Optuna HPO + CQR")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    # Load configs: external JSON or built-in 20
    config_file = os.environ.get("CONFIG_FILE", None)
    if config_file:
        with open(config_file) as f:
            all_configs = json.load(f)
        print(f"Loaded {len(all_configs)} configs from {config_file}")
    else:
        all_configs = SPECIES_CONFIGS

    # CONFIG_SLICE for dual-node
    config_slice = os.environ.get("CONFIG_SLICE", None)
    if config_slice:
        start, end = map(int, config_slice.split(":"))
        configs_to_run = all_configs[start:end]
        print(f"CONFIG_SLICE={config_slice}: {len(configs_to_run)}/{len(all_configs)} configs\n")
    else:
        configs_to_run = all_configs

    # Load data
    data = load_parquet_data()
    n = len(data["trade_date"])
    supply_ctx = build_supply_context(data, n)

    # Load weather data (optional — used if available)
    weather_data = load_weather_data()
    has_weather = bool(weather_data)
    if has_weather:
        print(f"Weather data loaded: {sum(len(v) for v in weather_data.values())} station-days")
    else:
        print("No weather data found — using 68 base features only")

    # Results containers
    results = {}          # config_id → {model → metrics}
    optuna_params = {}    # config_id → {model → best_params}
    loss_used = {}        # config_id → {model → loss_name}
    quantile_results = {} # config_id → {model → band_metrics}
    ensemble_results = {} # config_id → {mape, model_weights}
    timings = {}          # config_id → {model → seconds}

    POINT_MODELS = ["GRU", "LSTM", "BiLSTM+Attn", "CNN-LSTM", "Transformer", "PatchTST"]
    QUANTILE_MODELS_MAP = {
        "GRU-Q": (GRUQuantile, "GRU"),
        "Transformer-Q": (TransformerQuantile, "Transformer"),
        "CNN-LSTM-Q": (CNNLSTMQuantile, "CNN-LSTM"),
    }

    for ci, cfg in enumerate(configs_to_run):
        config_id = cfg["id"]
        print(f"\n{'='*60}")
        print(f"[{ci+1}/{len(configs_to_run)}] {config_id}")
        print(f"{'='*60}")

        # Build daily series
        daily = build_species_daily_series(data, cfg)
        if daily is None or len(daily.get("dates", [])) < MIN_DAYS:
            print(f"  SKIP — insufficient data ({len(daily.get('dates', []))} days)")
            continue

        # Build features (68 base)
        # build_features_68 returns (features_array, min_offset)
        species_name = cfg["species"]
        features_raw, min_offset = build_features(daily, supply_ctx, species_name)
        features_raw = features_raw[min_offset:]

        # Extract prices from daily series and trim to match features
        raw_prices = daily["prices"][min_offset:]
        daily_dates = daily["dates"][min_offset:]

        # Fix 3: outlier detection
        outlier_mask = flag_outlier_days(raw_prices)
        n_outliers = int(outlier_mask.sum())

        # Fix 2: log-transform target
        log_prices = np.log(np.maximum(raw_prices, 1.0))

        # Smoothed target if configured
        use_smoothed = cfg.get("smoothed", False)
        if use_smoothed and len(log_prices) > 7:
            target_prices = np.convolve(log_prices, np.ones(7) / 7, mode="same")
        else:
            target_prices = log_prices

        features = features_raw
        prices = target_prices

        if features is None or len(features) < LOOKBACK + HORIZON + 50:
            print(f"  SKIP — insufficient features ({len(features) if features is not None else 0})")
            continue

        print(f"  Data: {len(prices)} days, {features.shape[1]} base features, {n_outliers} outlier days")

        # Append weather features if available (68 → 76 features)
        if has_weather:
            w_feats = get_weather_features(weather_data, cfg["species"], daily_dates)
            if len(w_feats) == len(features):
                features = np.hstack([features, w_feats])
                print(f"  Features: 68 base + {w_feats.shape[1]} weather = {features.shape[1]}")
            else:
                print(f"  Weather feature length mismatch ({len(w_feats)} vs {len(features)}), 68 base only")

        n_features = features.shape[1]

        # Normalize features (z-score)
        feat_mean = np.mean(features, axis=0)
        feat_std = np.std(features, axis=0) + 1e-8
        features_norm = (features - feat_mean) / feat_std
        features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=0.0, neginf=0.0)
        features_norm = np.clip(features_norm, -10, 10)

        # Normalize prices (z-score on log-prices, which are already log-transformed)
        price_mean = np.mean(prices)
        price_std = np.std(prices) + 1e-8
        prices_norm = (prices - price_mean) / price_std

        # 80/20 train/test split
        split = int(len(features_norm) * 0.8)
        train_feat, test_feat = features_norm[:split], features_norm[split:]
        train_price, test_price = prices_norm[:split], prices_norm[split:]

        # Create datasets
        train_ds = SlidingWindowDataset(train_feat, train_price, LOOKBACK, HORIZON)
        test_ds = SlidingWindowDataset(test_feat, test_price, LOOKBACK, HORIZON)

        if len(train_ds) < 50 or len(test_ds) < 10:
            print(f"  SKIP — too few samples (train={len(train_ds)}, test={len(test_ds)})")
            continue

        # Train/val split (90/10)
        val_split = int(len(train_ds) * 0.9)
        train_subset = torch.utils.data.Subset(train_ds, range(val_split))
        val_subset = torch.utils.data.Subset(train_ds, range(val_split, len(train_ds)))

        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        results[config_id] = {}
        optuna_params[config_id] = {}
        loss_used[config_id] = {}
        timings[config_id] = {}

        # ── Train point prediction models ─────────────────────────
        for model_name in POINT_MODELS:
            print(f"\n  [{model_name}]")
            t0 = time.time()

            # Get per-config loss
            criterion, loss_name = get_config_loss(config_id, model_name)
            loss_used[config_id][model_name] = loss_name
            print(f"    Loss: {loss_name}")

            # Optuna HPO
            print(f"    Optuna ({OPTUNA_TRIALS} trials)...")
            best_params = optuna_hpo(
                model_name, n_features, train_subset, val_subset,
                criterion, device, n_trials=OPTUNA_TRIALS,
                price_mean=price_mean, price_std=price_std,
            )
            optuna_params[config_id][model_name] = best_params
            print(f"    Best params: {best_params}")

            # Train with best params
            model = create_model(
                model_name, n_features,
                hidden_size=best_params.get("hidden_size", 64),
                num_layers=best_params.get("num_layers", 2),
                dropout=best_params.get("dropout", 0.1),
            )
            bs = best_params.get("batch_size", BATCH_SIZE)
            train_loader = DataLoader(train_subset, batch_size=bs, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=bs)

            train_info = train_model_v2(
                model, train_loader, val_loader, criterion,
                epochs=EPOCHS, lr=best_params.get("lr", 0.001),
                device=device, patience=PATIENCE,
            )

            # Evaluate
            test_loader_bs = DataLoader(test_ds, batch_size=bs)
            metrics = run_test(model, test_loader_bs, device, price_mean, price_std)
            metrics["loss"] = loss_name
            metrics["optuna_params"] = best_params
            metrics["best_epoch"] = train_info["best_epoch"]
            results[config_id][model_name] = metrics

            elapsed = time.time() - t0
            timings[config_id][model_name] = round(elapsed, 1)
            print(f"    MAPE: {metrics['mape']:.1f}%  RMSE: {metrics['rmse']:.0f}  "
                  f"Dir: {metrics['dir_acc']:.1f}%  ({elapsed:.0f}s)")

            torch.cuda.empty_cache() if device == "cuda" else None

        # ── Ensemble: average top-3 point models ──────────────────
        point_results = [(m, r["mape"]) for m, r in results[config_id].items()
                         if r["mape"] < 900]
        if len(point_results) >= 3:
            top3 = sorted(point_results, key=lambda x: x[1])[:3]
            avg_mape = sum(m for _, m in top3) / 3
            ensemble_results[config_id] = {
                "mape": round(avg_mape, 2),
                "models": [m for m, _ in top3],
                "individual_mapes": [round(m, 2) for _, m in top3],
            }
            print(f"\n  [Ensemble] Top-3 avg MAPE: {avg_mape:.1f}% ({[m for m, _ in top3]})")

        # ── Train quantile models ─────────────────────────────────
        print(f"\n  === Quantile Models ===")

        # Build quantile dataset (single next-day prediction, z-scored log-prices)
        q_prices = np.array(prices_norm, dtype=np.float32)
        q_split = int(len(features_norm) * 0.8)
        q_train_feat = features_norm[:q_split]
        q_train_price = q_prices[:q_split]
        q_test_feat = features_norm[q_split:]
        q_test_price = q_prices[q_split:]

        q_train_ds = QuantileDataset(q_train_feat, q_train_price, lookback=LOOKBACK)
        q_test_ds = QuantileDataset(q_test_feat, q_test_price, lookback=LOOKBACK)

        if len(q_train_ds) < 50 or len(q_test_ds) < 10:
            print("    SKIP quantile — too few samples")
            continue

        q_val_split = int(len(q_train_ds) * 0.9)
        q_train_sub = torch.utils.data.Subset(q_train_ds, range(q_val_split))
        q_val_sub = torch.utils.data.Subset(q_train_ds, range(q_val_split, len(q_train_ds)))

        quantile_results[config_id] = {}

        for q_name, (q_cls, base_name) in QUANTILE_MODELS_MAP.items():
            print(f"\n  [{q_name}]")
            t0 = time.time()

            # Use Optuna params from the base model if available
            base_params = optuna_params[config_id].get(base_name, {})
            hidden = base_params.get("hidden_size", 64)
            layers = base_params.get("num_layers", 2)
            lr = base_params.get("lr", 0.001)
            bs = base_params.get("batch_size", 64)
            dropout = base_params.get("dropout", 0.1)

            if q_name == "Transformer-Q":
                q_model = q_cls(n_features, d_model=hidden, num_layers=layers, dropout=dropout)
            else:
                q_model = q_cls(n_features, hidden_size=hidden, num_layers=layers, dropout=dropout)

            q_train_loader = DataLoader(q_train_sub, batch_size=bs, shuffle=True)
            q_val_loader = DataLoader(q_val_sub, batch_size=bs)

            train_quantile_model(
                q_model, q_train_loader, q_val_loader, device,
                epochs=EPOCHS, lr=lr, patience=PATIENCE,
            )

            # Evaluate quantile bands
            q_model.train(False)
            q_test_loader = DataLoader(q_test_ds, batch_size=bs)
            all_preds, all_actuals = [], []
            with torch.no_grad():
                for x, y in q_test_loader:
                    x = x.to(device)
                    pred = q_model(x)
                    all_preds.append(pred.cpu().numpy())
                    all_actuals.append(y.numpy())

            if all_preds:
                preds_q = np.concatenate(all_preds, axis=0)  # (n, 3)
                actuals_q = np.concatenate(all_actuals, axis=0)  # (n,)

                # Denormalize (log-scale z-score → raw price)
                pred_q10 = np.exp(preds_q[:, 0] * price_std + price_mean)
                pred_q50 = np.exp(preds_q[:, 1] * price_std + price_mean)
                pred_q90 = np.exp(preds_q[:, 2] * price_std + price_mean)
                actual_raw = np.exp(actuals_q * price_std + price_mean)

                band_metrics = evaluate_bands(pred_q10, pred_q50, pred_q90, actual_raw)

                # CQR calibration (asymmetric)
                _, _, cqr_coverage, cqr_width = compute_cqr_bands(
                    pred_q10, pred_q50, pred_q90, actual_raw, alpha=0.1
                )
                band_metrics["cqr_coverage"] = round(cqr_coverage, 1)
                band_metrics["cqr_width"] = round(cqr_width)

                # Example forecast
                idx = len(pred_q50) - 1
                band_metrics["example_forecast"] = {
                    "p10": round(float(pred_q10[idx])),
                    "p50": round(float(pred_q50[idx])),
                    "p90": round(float(pred_q90[idx])),
                    "actual": round(float(actual_raw[idx])),
                }

                quantile_results[config_id][q_name] = band_metrics

                elapsed = time.time() - t0
                timings[config_id][q_name] = round(elapsed, 1)
                print(f"    MAPE(p50): {band_metrics['mape_p50']:.1f}%  "
                      f"Coverage: {band_metrics['coverage']:.1f}%  "
                      f"CQR: {band_metrics['cqr_coverage']:.1f}%  ({elapsed:.0f}s)")

            torch.cuda.empty_cache() if device == "cuda" else None

    # ── Save Results ──────────────────────────────────────────────

    # Build rankings
    model_mapes = defaultdict(list)
    for cid, models in results.items():
        for model_name, metrics in models.items():
            if metrics["mape"] < 900:
                model_mapes[model_name].append(metrics["mape"])

    ranking = sorted(
        [{"rank": 0, "model": m, "avg_mape": round(np.mean(v), 2), "configs": len(v)}
         for m, v in model_mapes.items()],
        key=lambda x: x["avg_mape"],
    )
    for i, r in enumerate(ranking, 1):
        r["rank"] = i

    output = {
        "generated_at": datetime.now().isoformat(),
        "version": "v2",
        "improvements": [
            "Per-config loss function selection (MAE/LogCosh/Huber/MSE/sMAPE)",
            f"Optuna HPO ({OPTUNA_TRIALS} trials per model per config)",
            "CQR asymmetric band calibration",
            "Ensemble top-3 point models",
            "Reuses Optuna params for quantile model base",
        ],
        "config": {
            "lookback": LOOKBACK, "horizon": HORIZON,
            "epochs": EPOCHS, "optuna_trials": OPTUNA_TRIALS,
            "optuna_epochs": OPTUNA_EPOCHS,
            "patience": PATIENCE, "min_days": MIN_DAYS,
        },
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "N/A",
        "results": results,
        "optuna_params": optuna_params,
        "loss_routing": loss_used,
        "ensemble_results": ensemble_results,
        "quantile_results": quantile_results,
        "timings": timings,
        "ranking": ranking,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "dl_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"{'='*60}")

    # Print summary
    print(f"\n=== Model Ranking ===")
    for r in ranking:
        print(f"  #{r['rank']} {r['model']:15s}  avg MAPE: {r['avg_mape']:.1f}%  ({r['configs']} configs)")

    print(f"\n=== Best Per Config ===")
    for cid in sorted(results.keys()):
        models = results[cid]
        best = min(models.items(), key=lambda x: x[1]["mape"])
        loss_name = loss_used.get(cid, {}).get(best[0], "?")
        print(f"  {cid:30s}  {best[0]:15s}  MAPE={best[1]['mape']:.1f}%  loss={loss_name}")

    if ensemble_results:
        print(f"\n=== Ensemble Results ===")
        for cid, ens in sorted(ensemble_results.items()):
            print(f"  {cid:30s}  avg top-3 MAPE: {ens['mape']:.1f}%  ({ens['models']})")


if __name__ == "__main__":
    main()
