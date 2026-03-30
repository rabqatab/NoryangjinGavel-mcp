# Advanced Preprocessing Analysis

> Investigation into why intra-day price aggregation is noisy and how to fix it.
> Findings feed into prediction models v10+ and the DL model comparison.

## Problem: Intra-Day Price Spread is 58%

Even after filtering to the same (species, state, packaging, spec), a single day's auction lots vary by **58% median spread** for 넙치. This means the "daily average price" we feed to prediction models is a noisy aggregation of fundamentally different products.

Example: 2006.08.26 넙치 활/kg/중 — two lots: 38,000 KRW (완도) and 11,100 KRW (제주도). The simple mean is 24,550, but this number represents neither actual transaction.

## Root Cause

The `spec=중` filter captures a range of sizes within "medium" grade. A 중 넙치 from 완도 (premium origin, likely larger) commands 3x the price of 중 넙치 from 제주도 (standard). The spec classification is too coarse — "중" encompasses significant quality variation.

## Fix 1: Winsorized Mean (Rolling 30-Day Window)

Clip extreme lots to the 10th/90th percentile of the recent 30-day price distribution before computing the daily mean.

| Species | Current Lag-1 | Winsorized Lag-1 | Noise Reduction |
|---|---|---|---|
| 넙치 | 0.751 | **0.858** | **-28%** daily change |
| 우럭 | 0.597 | ~0.70 (estimated) | ~-20% |

**Why it works:** Outlier lots (data entry errors, rare premium/discount transactions) are clipped to the recent price band. The core market signal is preserved while extreme values are damped.

## Fix 2: Log-Transform Target

Predict `log(price)` instead of raw price, then `exp()` the prediction.

| Metric | Raw Price | Log Price | Box-Cox (λ=0.24) |
|---|---|---|---|
| Skewness | 0.66 | -0.22 | **0.00** |
| CV | 0.303 | 0.032 | — |

**Why it works:** Fish prices are right-skewed (occasional high prices pull the mean up). Log-transform creates a nearly symmetric distribution, which is what MSE loss assumes. This is standard practice in financial time-series prediction.

Box-Cox optimal λ=0.242 (closer to log than linear), confirming log-space is the right transformation.

## Fix 3: Outlier Day Removal

Flag and exclude days where the aggregated price changes by >3σ from the rolling 30-day mean. These represent data entry errors or extreme one-off events (e.g., a single premium lot on a low-volume day).

For 넙치: p99 daily change is 74.1%, max is 169.2%. Days above 3σ (~50% change) are likely anomalous and contaminate training data.

## Fix 4: Origin-Weighted Aggregation

Weight lots by origin trading frequency. High-frequency origins (제주도, 부산, 완도 — appearing daily) represent the stable market. Low-frequency origins (one-off appearances) are noisier.

Weight = `origin_frequency_30d / max_frequency_30d`

## Fix 5: Adaptive VMD K

Use different VMD K values based on the current price regime:
- High-volatility periods (price_std_30d > median): K=5 (more modes to capture complex dynamics)
- Low-volatility periods: K=3 (simpler decomposition)

## Aggregation Method Comparison (넙치)

| Method | Lag-1 Autocorr | Median Daily Change | Improvement |
|---|---|---|---|
| Simple Mean (current) | 0.751 | 12.7% | baseline |
| Qty-Weighted Median | 0.837 | 8.5% | +11% / -33% noise |
| **Winsorized Mean (30d)** | **0.858** | **9.2%** | **+14% / -28% noise** |
| Qty-Weighted Mean | 0.883 | 7.8% | +18% / -39% noise |

## Results: v6 → v10 (18-29% MAPE Reduction)

| Species | v6 (before) | v10 (after) | Improvement |
|---|---|---|---|
| **넙치** | 14.8% | **11.1%** | **+25%** |
| **감성돔** | 22.8% | **17.1%** | **+25%** |
| **참돔** | 26.5% | **18.9%** | **+29%** |
| **우럭** | 24.1% | **18.7%** | **+23%** |
| **농어** | 23.6% | **19.3%** | **+18%** |
| **도다리** | 26.0% | **21.1%** | **+19%** |
| **방어** | 62.6% | **49.2%** | **+21%** |

This is the single biggest improvement in the entire project — more impactful than any model architecture change (v1-v9) or DL model (TFT, Transformer, etc.).

## Price Band Prediction (Quantile + Conformal)

Beyond point predictions, v10 outputs calibrated price bands:

| Species | Point | Likely Range (p10~p90) | Conformal (80%) | Band Width |
|---|---|---|---|---|
| 넙치 | 20,950 | 19,026 ~ 21,817 | ±2,897 | 13% |
| 참돔 | 20,635 | 17,570 ~ 22,377 | ±4,430 | 23% |
| 감성돔 | 33,285 | 26,611 ~ 35,701 | ±7,103 | 27% |
| 도다리 | 14,860 | 12,570 ~ 16,596 | ±4,111 | 27% |
| 농어 | 17,455 | 15,078 ~ 20,359 | ±4,493 | 30% |
| 방어 | 3,479 | 2,981 ~ 5,561 | ±2,061 | 74% |

All conformal bands achieve exactly **80% actual coverage** — properly calibrated. Consumer output example:

```
넙치 (flatfish) 7-day forecast:
  Expected: 20,950 KRW/kg
  Likely range: 19,000 ~ 21,800 KRW/kg (80% confidence)
  Budget range: 19,000 ~ 21,800 KRW/kg (p10~p90)
```

## Impact on DL Models

The same 5 fixes applied to DL models (GRU, LSTM, Transformer, etc.) gave even larger improvements than LightGBM:

| Model | Before v10 Preproc | After v10 Preproc | Improvement |
|---|---|---|---|
| GRU | 47.6% avg | 27.2% avg | **-43%** |
| Transformer | 44.3% avg | 27.5% avg | **-38%** |
| CNN-LSTM | 43.8% avg | 28.0% avg | **-36%** |
| LSTM | 44.1% avg | 28.3% avg | **-36%** |

**Preprocessing quality matters more than model architecture** — the same 5 fixes improved DL models by 36-43%, which is larger than the difference between any two model architectures.

### Updated Best-of-Breed (with v10 preprocessing)

| Species | Best Model | MAPE |
|---|---|---|
| 넙치 | v10 LightGBM | **11.1%** |
| 우럭 | TFT | **14.7%** |
| 방어 | TFT | **15.6%** |
| 도다리 | CNN-LSTM+VMD | **16.1%** |
| 농어 | GRU | **16.5%** |
| 감성돔 | v10 LightGBM | **17.1%** |
| 참돔 | v10 LightGBM | **18.9%** |

## Implementation

All 5 fixes + quantile bands are implemented in:
- `scripts/poc_prediction_v10.py` — CPU (LightGBM) with quantile + conformal bands
- `scripts/train_all_dl_models.py` — GPU (all DL models) with v10 preprocessing + quantile bands
