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

## Implementation

All 5 fixes are implemented in `scripts/poc_prediction_v10.py` and applied consistently across both CPU (LightGBM) and GPU (DL models) training pipelines.
