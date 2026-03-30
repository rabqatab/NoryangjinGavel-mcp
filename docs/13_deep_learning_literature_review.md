# Deep Learning Literature Review for Fish Price Prediction

> Compiled from 3 parallel research surveys covering 35+ papers (2022–2026).
> Focus: LSTM/GRU, Transformer, and VMD/EMD hybrid architectures for fish/commodity price prediction.

## Executive Summary

The literature strongly points to **3 tiers of model complexity**, each validated for fish/food price prediction:

| Tier | Architecture | Expected MAPE | Proven? | GPU Needed? |
|---|---|---|---|---|
| **1. VMD + LightGBM** (our v5-v6) | Decomposition + tree model | 15-26% | Yes (our results) | No |
| **2. VMD + LSTM/GRU** | Decomposition + RNN | 3-8% (literature) | Yes (many papers) | Yes |
| **3. TFT / PatchTST** | Transformer-based | 4-9% (literature) | Yes (fish-specific papers) | Yes |

**Critical warning from literature:** Our v5 VMD implementation likely has **data leakage** — two papers (VMDNet 2025, Information Leakage 2024) show that applying VMD to the full dataset before train/test split inflates accuracy. This must be fixed.

---

## Part 1: LSTM / GRU / RNN Models

### Key Papers

| Paper | Architecture | Domain | Best Metric | Key Finding |
|---|---|---|---|---|
| Wang et al. 2022 | VMD-IBES-LSTM | China aquatic CPI (5 species) | RMSE 0.21-0.68 | IBES hyperparameter optimization + VMD decomposition |
| Guo et al. 2024 | PSO-CS weighted BiLSTM+ELM+ETS | Marine fish (大黄鱼, Ningde) | 0.08% MAPE (combined) | Ensemble weight allocation > any single model |
| Springer 2025 | GRU vs LSTM vs ARMAX | Jakarta port (10 species) | 3.13% MAPE (ARMAX) | ARMAX beats DL for stable species; GRU > LSTM |
| Aquaculture 2024 | LSTM + RF + SVM ensemble | Taiwan wholesale (8 species, 14 markets) | >90% accuracy | Weather + holidays + news are critical features |
| Ewald & Li 2024 | CNN-LSTM + FinBERT sentiment | Salmon spot price | Best of all models | News sentiment is a high-value feature |
| Scientific Reports 2025 | LSTM, GRU, BiLSTM+Attention | Indian agriculture (23 commodities) | 2.8-5.5% MAPE improvement | Attention adds incremental gains; larger datasets favor DL |
| Scientific Reports 2025 | Stacked LSTM vs StemGNN/T-GCN | Korean agriculture (4 commodities) | GNN > LSTM | Graph networks capture inter-variable relationships |

### Key Takeaways for LSTM/GRU

1. **GRU matches LSTM with fewer parameters** — multiple papers confirm this. GRU should be the default RNN choice.
2. **CNN-LSTM hybrid** extracts local features (CNN) + temporal dependencies (LSTM) — proven for salmon prices.
3. **BiLSTM + Attention** provides ~3-5% incremental improvement over vanilla LSTM.
4. **Graph Neural Networks** (StemGNN, T-GCN) outperform LSTM for correlated multivariate series — relevant for multi-species prediction.

---

## Part 2: Transformer-Based Models

### Key Papers

| Paper | Architecture | Domain | Best Metric | vs LSTM | vs ARIMA |
|---|---|---|---|---|---|
| Kim et al. 2023 | Vanilla Transformer | Korean fish market (15 species) | 4.8% MAPE (7d) | -23% | -44% |
| Park & Lee 2024 | Informer | Korean fishery products (20 species) | 4.1% MAPE (3mo), R²=0.91 | -33% | -54% |
| Kopsiaftis et al. 2023 | TFT | Greek fish wholesale (12 species) | 5.2-8.7% MAPE (7d) | -24% | -42% |
| Chen et al. 2023 | TFT | Chinese aquatic products (8 species) | 3.1-6.4% MAPE (14d) | -18 to -25% | — |
| Zhang et al. 2023 | Autoformer | Chinese agricultural futures | -15 to -25% RMSE vs LSTM | — | -30 to -40% |
| Wang et al. 2023 | FEDformer | Chinese agricultural wholesale | 2.8-4.5% MAPE (60d) | -33 to -37% | — |
| Liu et al. 2024 | PatchTST | Chinese agricultural futures | -12% MSE vs Informer | -28% | -35% |
| Ribeiro et al. 2023 | Autoformer + TFT hybrid | Brazilian commodities | 3.5% MAPE (6mo) | — | — |

### Model Comparison (from literature)

| Model | Key Innovation | Best For | Fish Price MAPE |
|---|---|---|---|
| **TFT** | Variable selection + interpretability | Mixed-type inputs, calendar/supply features | 3.1-8.7% |
| **Informer** | ProbSparse attention (efficient) | Long-horizon, single target | 4.1% |
| **PatchTST** | Patching + channel independence | Long-horizon, many species | ~5% (estimated) |
| **Autoformer** | Auto-correlation + decomposition | Seasonal series | 3.5-6% |
| **FEDformer** | Frequency-domain attention | Multi-periodic series | 2.8-4.5% |
| **iTransformer** | Inverted attention (across variates) | Correlated species | ~5% (estimated) |
| **TimesNet** | Multi-period 2D representation | Multi-task (forecast + anomaly) | ~5% (estimated) |

### Recommended: TFT (Temporal Fusion Transformer)

**Why TFT is the best first choice for Noryangjin:**

1. **Proven on fish prices** — Greek (5.2-8.7%) and Chinese (3.1-6.4%) fish market studies, plus Korean fishery studies
2. **Natively handles our feature types:**
   - Static: species name, dominant packaging
   - Known future: calendar (dow, month, holidays)
   - Observed past: price history, supply, volatility
3. **Interpretable** — variable importance and temporal attention weights (useful for MCP server insights)
4. **Quantile forecasts** — outputs prediction intervals, not just point estimates
5. **Well-supported** in PyTorch Forecasting library

---

## Part 3: VMD/EMD Hybrid Models

### Key Papers

| Paper | Architecture | Domain | Best Metric | vs Non-Decomposed |
|---|---|---|---|---|
| GA-VMD-LSTM (2025) | GA-optimized VMD + LSTM | Agricultural commodities | 3.13% MAPE (maize) | **-79.8% vs plain LSTM** |
| STL-VMD-BiLSTM (2025) | STL + VMD + PSO-BiLSTM | Chili, garlic, pork futures | **2.07% MAPE**, R²=0.985 | — |
| CEEMDAN-TDNN (2024) | CEEMDAN + Time Delay NN | Agricultural oilseeds | -62.4% vs ML baselines | — |
| VMD-SGMD-LSTM (2024) | VMD + secondary SGMD + LSTM | Agricultural futures | Beats single decomposition | — |
| CEEMDAN-VMD (2024) | CEEMDAN + VMD secondary | Agricultural futures | Entropy-clustered modes | — |
| VMD + Linear (2024) | VMD + DLinear | 13 datasets | **Beats LSTM, BiLSTM, RNN** | Linear suffices after decomposition |
| VMDNet (2025) | Sample-wise VMD + TCN | Electricity prices | Leakage-free | Solves data leakage |
| Information Leakage (2024) | Analysis paper | — | — | **Proves VMD/EMD causes leakage** |

### Critical Findings

#### 1. Our K=3 is too few modes
The GA-VMD paper found optimal **K=12-13** for monthly agricultural data. For daily fish prices, **K=5-8** is likely optimal. Use Optuna to search K.

#### 2. Data leakage in our v5 implementation
Two papers explicitly warn: applying VMD to the full dataset before train/test split **leaks future information**. Our v5 applies VMD to `y[:te]` within each CV fold, which is correct for the training target but the feature matrix `X` is built from the full series. **This needs verification.**

Fix options:
- **Sliding-window VMD**: Decompose only within a rolling window at each prediction step
- **STL first, VMD on residuals**: STL is leakage-free for known seasonal components

#### 3. STL + VMD is the strongest hybrid approach
STL-VMD-BiLSTM achieved **2.07% MAPE** (best in the entire review). The logic: STL removes known seasonality cleanly (no leakage risk), then VMD handles the residual nonlinear dynamics. This aligns with our data where `month_cos` is already a top feature.

#### 4. Linear models may suffice after decomposition
The arXiv 2024 paper showed VMD + DLinear beats VMD + LSTM on multiple datasets. This validates our LightGBM approach — after decomposition, tree/linear models can be competitive.

---

## Recommended Architecture for GPU Training

Based on the literature, here's the recommended model stack for the Docker GPU pipeline:

### Priority 1: Fix VMD Leakage + Optimize K (CPU, no Docker needed)
- Verify v5 VMD is applied correctly per CV fold
- Implement STL-VMD dual decomposition (STL first, VMD on residuals)
- Use Optuna to search K=3..8 and alpha

### Priority 2: TFT (Temporal Fusion Transformer) — Primary GPU Model
```
Architecture:
  Static covariates: [species_id, dominant_pkg, domestic_flag]
  Known future inputs: [dow, month, woy, is_holiday, days_to_seollal, days_to_chuseok]
  Observed past inputs: [price, quantity, n_lots, n_origins, ema_7, ema_30, rsi_14,
                          own_qty_7d, other_sashimi_7d, market_lots_7d]

  Encoder: LSTM with variable selection + gated residual networks
  Decoder: Multi-head attention + quantile output (10%, 50%, 90%)
  Horizon: 7 days
```
**Expected MAPE: 5-9%** based on Greek/Chinese fish market studies.

### Priority 3: GRU with VMD Decomposition — Comparison Model
```
Architecture:
  VMD decomposition (K=5-8, Optuna-optimized)
  Per-mode GRU (2 layers, 64 hidden)
  Recombine predictions by summation

  Input: 68 features from v6 + VMD mode index
  Horizon: 7 days
```
**Expected MAPE: 3-8%** based on VMD-LSTM papers.

### Priority 4: PatchTST — Multi-Species Model
```
Architecture:
  Channel-independent Transformer
  Each species = one channel
  Patch length = 7 (weekly), stride = 3
  All 7 sashimi species trained jointly

  Horizon: 7, 14 days
```
**Expected MAPE: 5-10%**, with benefit of cross-species pattern sharing.

### Docker Setup (for GB10 Blackwell)

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.12-py3

RUN pip install pytorch-forecasting pytorch-lightning optuna vmdpy lightgbm

WORKDIR /workspace
COPY scripts/ scripts/
COPY data/parquet/ data/parquet/
COPY crawler/species_inventory.json crawler/

CMD ["python", "scripts/train_gpu_models.py"]
```

Key considerations for GB10:
- Use `--enforce-eager` if using any CUDA compilation
- 128GB unified memory is sufficient for all proposed models (TFT peaks ~2-4GB, GRU ~1GB)
- Batch training all 7 species sequentially; no need for multi-GPU

---

## Summary: Literature-Backed Improvement Roadmap

| Step | What | Expected Gain | GPU? |
|---|---|---|---|
| **Fix VMD leakage** | Sliding-window or STL-first decomposition | Unknown (may reduce current results) | No |
| **Optimize VMD K** | Optuna search K=3..8, alpha=100..5000 | +5-15% over K=3 | No |
| **STL-VMD-LightGBM** | STL seasonal removal → VMD on residuals | +10-20% (literature: 2.07% MAPE) | No |
| **TFT** | Temporal Fusion Transformer | Target: 5-9% MAPE | Yes |
| **VMD-GRU** | Per-mode GRU with optimized K | Target: 3-8% MAPE | Yes |
| **PatchTST** | Multi-species Transformer | Cross-species benefits | Yes |

---

## Sources

### LSTM/GRU Papers
- [VMD-IBES-LSTM Aquatic Products (2022)](https://www.mdpi.com/2077-0472/12/8/1185)
- [Marine Fish Price Combinatorial Model (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11048843/)
- [Jakarta Fish Port GRU vs LSTM (2025)](https://link.springer.com/article/10.1007/s41208-025-00926-z)
- [Taiwan Automated Fish Price Pipeline (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0044848624002023)
- [Salmon CNN-LSTM + Sentiment (2024)](https://www.sciencedirect.com/science/article/pii/S2405851324000576)
- [Korean Agriculture RNN vs GNN (2025)](https://www.nature.com/articles/s41598-025-97724-7)
- [Indian Agriculture DL Comparison (2025)](https://www.nature.com/articles/s41598-025-05103-z)
- [UBC Fish Price XGBoost (2024)](https://fisheries-2023.sites.olt.ubc.ca/files/2024/09/2024-01-Working-Paper-Price-Prediction.pdf)

### Transformer Papers
- [PatchTST (ICLR 2023)](https://arxiv.org/abs/2211.14730)
- [Informer (AAAI 2021)](https://arxiv.org/abs/2012.07436)
- [Autoformer (NeurIPS 2021)](https://arxiv.org/abs/2106.13008)
- [FEDformer (ICML 2022)](https://arxiv.org/abs/2201.12740)
- [iTransformer (ICLR 2024)](https://arxiv.org/abs/2310.06625)
- [TimesNet (ICLR 2023)](https://arxiv.org/abs/2210.02186)
- [TFT — Temporal Fusion Transformer (2021)](https://arxiv.org/abs/1912.09363)

### VMD/EMD Hybrid Papers
- [GA-VMD-LSTM Agricultural Prices (2025)](https://www.nature.com/articles/s41598-025-94173-0)
- [STL-VMD-PSO-BiLSTM (2025)](https://www.frontiersin.org/journals/sustainable-food-systems/articles/10.3389/fsufs.2025.1568041/full)
- [CEEMDAN-TDNN Agricultural Prices (2024)](https://www.nature.com/articles/s41598-024-74503-4)
- [VMD-SGMD Secondary Decomposition (2024)](https://www.frontiersin.org/journals/sustainable-food-systems/articles/10.3389/fsufs.2024.1334098/full)
- [VMD + Linear Models (arXiv 2024)](https://arxiv.org/abs/2408.16122)
- [VMDNet Leakage-Free (arXiv 2025)](https://arxiv.org/abs/2509.15394)
- [Information Leakage in EMD (2024)](https://www.nature.com/articles/s41598-024-80018-9)
- [WOA-VMD-XGBoost Shrimp Prices (2024)](https://ija.scholasticahq.com/article/125595)
