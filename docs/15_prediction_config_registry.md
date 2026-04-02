# Prediction Config Registry

> Living document tracking all prediction configs, their data health, model status, and performance.
> Updated from `data/prediction_config_registry.json`. Regenerate with: `uv run python scripts/poc_test_mullet.py`

## Config Naming Convention

Each config is identified by: `{species}_{state}_{packaging}_{spec}[_dom]`

- `_dom` suffix = domestic-only (foreign origins filtered out)
- Same species with different configs = different market = different model

## Data Health Overview

| Config ID | Days | Rows | Lots/Day | Mean Price | CV | Lag-1 | Recent (2025+) | Date Range |
|---|---|---|---|---|---|---|---|---|
| **넙치_활_kg_중** | 5,721 | 26,707 | 4.7 | 14,883 | 0.303 | 0.751 | 2 | 2006~2025 |
| **우럭_활_kg_중** | 5,391 | 13,138 | 2.4 | 10,869 | 0.353 | 0.597 | 0 | 2006~2024 |
| **방어_선_kg_중_dom** | 1,270 | 1,625 | 1.3 | 5,430 | 1.034 | 0.300 | 2 | 2006~2025 |
| **참돔_활_kg_중_dom** | 5,448 | 15,073 | 2.8 | 17,414 | 0.352 | 0.378 | 2 | 2006~2025 |
| **농어_활_kg_중_dom** | 4,998 | 13,479 | 2.7 | 16,346 | 0.343 | 0.406 | 0 | 2006~2024 |
| **도다리_활_kg_중** | 3,055 | 6,133 | 2.0 | 14,062 | 0.497 | 0.346 | 0 | 2006~2024 |
| **감성돔_활_kg_중_dom** | 3,064 | 4,433 | 1.4 | 23,999 | 0.362 | 0.532 | 0 | 2006~2024 |
| **감숭어_활_kg_중** | 5,461 | 18,682 | 3.4 | 4,264 | 0.510 | 0.603 | 1 | 2006~2025 |
| **참숭어_활_kg_중** | 5,264 | 16,983 | 3.2 | 5,343 | 0.450 | 0.638 | 1 | 2006~2025 |
| **쭈꾸미_선_box_중_dom** | 2,097 | 3,284 | 1.6 | 36,580 | 0.432 | 0.695 | 74 | 2006~2025 |
| **민어_선_SP_중** | 4,451 | 8,642 | 1.9 | 54,046 | 0.871 | 0.451 | 4 | 2006~2025 |
| **깐굴_선_box_소** | 5,891 | 18,878 | 3.2 | 16,340 | 0.460 | 0.938 | 233 | 2006~2026 |
| **바위굴_활_box_대** | 2,253 | 4,605 | 2.0 | 15,198 | 0.238 | 0.807 | 44 | 2007~2025 |
| **수꽃게_활_kg_중** | 3,573 | 9,581 | 2.7 | 12,818 | 0.394 | 0.725 | 157 | 2004~2026 |
| **암꽃게_활_kg_중** | 3,820 | 10,270 | 2.7 | 21,404 | 0.499 | 0.865 | 182 | 2004~2025 |
| **수꽃게_활_kg_대** | 3,519 | 10,436 | 3.0 | 16,774 | 0.380 | 0.757 | 173 | 2004~2026 |
| **암꽃게_활_kg_대** | 3,816 | 12,076 | 3.2 | 26,608 | 0.459 | 0.899 | 197 | 2004~2025 |
| **넙치_활_kg_2미** | 431 | 1,182 | 2.7 | 15,966 | 0.452 | 0.644 | 295 | 2008~2026 |
| **참돔_활_kg_2미_dom** | 340 | 866 | 2.5 | 17,388 | 0.421 | 0.524 | 287 | 2009~2026 |
| **농어_활_kg_1미_dom** | 345 | 1,343 | 3.9 | 18,210 | 0.294 | 0.560 | 299 | 2009~2026 |

## Column Definitions

| Column | Description |
|---|---|
| **Days** | Total distinct trading days in the dataset |
| **Rows** | Total auction lot records (multiple lots per day) |
| **Lots/Day** | Average auction lots per trading day (more = more stable daily mean) |
| **Mean Price** | Average daily price (KRW) across all trading days |
| **CV** | Coefficient of Variation (std/mean) — higher = more volatile. >0.5 is challenging |
| **Lag-1** | Day-to-day autocorrelation — higher = more predictable. <0.4 is difficult |
| **Recent** | Trading days in 2025+ — 0 means data may have stopped |

## Model Status (Updated 2026-04-02 — DL v2 with Optuna HPO + per-config loss + weather features)

### DL v2 Improvements over v1
- **Per-config loss selection**: MAE/LogCosh/Huber/MSE based on loss comparison study (10 configs empirical, 10 defaults)
- **Optuna HPO**: 10 trials per model per config — tunes hidden_size, num_layers, lr, dropout, batch_size
- **Weather features**: 8 coastal weather features from Open-Meteo Archive (2006-2026, 5 ports) → 68+8=76 total features
- **CQR calibration**: Asymmetric conformal quantile regression for calibrated prediction bands
- **Ensemble**: Top-3 model averaging per config

### Original 20 Configs — v2 Results

| Config ID | v1 Best (DL) | v2 Best (DL) | v2 Delta | v2 Model | v2 Loss | Status |
|---|---|---|---|---|---|---|
| **바위굴_활_box_대** | 4.0% | **3.7%** | -0.4pp | GRU | MAE | **Production** |
| **넙치_활_kg_중** | 13.8% | **11.3%** | -2.5pp | PatchTST | LogCosh | **Production** |
| **쭈꾸미_선_box_중_dom** | 11.4% | **11.0%** | -0.4pp | GRU | MAE | **Production** |
| **깐굴_선_box_소** | 11.8% | **11.2%** | -0.5pp | LSTM | MAE | **Production** |
| **수꽃게_활_kg_중** | 14.7% | **13.7%** | -0.9pp | GRU | MAE | **Production** |
| **농어_활_kg_1미_dom** | 18.3% | **13.6%** | -4.7pp | GRU | MAE | **Production** |
| **암꽃게_활_kg_대** | 13.5% | **13.7%** | +0.2pp | GRU | MAE | **Production** |
| **수꽃게_활_kg_대** | 13.6% | **13.9%** | +0.3pp | GRU | MAE | **Production** |
| **참돔_활_kg_2미_dom** | 21.9% | **15.3%** | -6.6pp | Transformer | LogCosh | **Production** |
| **도다리_활_kg_중** | 16.5% | **15.2%** | -1.4pp | GRU | MAE | **Production** |
| **농어_활_kg_중_dom** | 15.9% | **15.8%** | -0.1pp | GRU | MAE | **Production** |
| **암꽃게_활_kg_중** | 16.1% | **16.2%** | +0.1pp | Transformer | LogCosh | **Production** |
| **감성돔_활_kg_중_dom** | 16.2% | **16.9%** | +0.7pp | GRU | MAE | **Production** |
| **참돔_활_kg_중_dom** | 19.3% | **19.6%** | +0.3pp | Transformer | MAE | **Production** |
| **참숭어_활_kg_중** | 21.6% | **22.7%** | +1.1pp | GRU | LogCosh | Production |
| **우럭_활_kg_중** | 24.1% | **23.1%** | -0.9pp | GRU | MAE | Production |
| **넙치_활_kg_2미** | 25.2% | **23.8%** | -1.4pp | GRU | MAE | Production |
| **감숭어_활_kg_중** | 25.4% | **26.1%** | +0.7pp | Transformer | LogCosh | Production |
| **방어_선_kg_중_dom** | 51.4% | **49.5%** | -1.9pp | CNN-LSTM | sMAPE | Directional |
| **민어_선_SP_중** | 62.8% | **68.5%** | +5.7pp | GRU | MAE | Directional |

**v2 improved 12/20 configs.** Key gains: 참돔_2미 (-6.6pp), 농어_1미 (-4.7pp), 넙치_중 (-2.5pp). GRU dominates (13/20 best). Optuna consistently favors single-layer models (num_layers=1).

### Expansion: 125 New Configs (Complete — 2026-04-02)

Full v2 pipeline on DGX Spark Node 2. Results merged into `data/poc_results/dl_v2_merged.json`.

**145 total configs (20 original + 125 new):**

| Tier | MAPE Range | Count | Description |
|---|---|---|---|
| **A** | < 10% | 22 | Production-ready, high confidence |
| **B** | 10-20% | 55 | Production-ready, good accuracy |
| **C** | 20-30% | 10 | Usable with caveats |
| **D** | 30%+ | 58 | Directional only or not viable |

**77 configs (53%) below 20% MAPE — production viable.**

### Tier A — Best Performers (MAPE < 10%)

| Config | MAPE | Model | Loss |
|---|---|---|---|
| 토바지락_활_box_대 | **0.1%** | LSTM | MAE |
| 아귀_냉_CTBT_중 | **0.2%** | CNN-LSTM | MAE |
| 쭈꾸미_선_box_대 | **2.5%** | CNN-LSTM | MAE |
| 낙지_활_그물망_20미 | **3.3%** | GRU | MAE |
| 봉바지락_활_box_대_dom | **3.5%** | BiLSTM+Attn | MAE |
| 바위굴_활_box_대 | **3.7%** | GRU | MAE |
| 멸치_선_SP_중 | **3.9%** | LSTM | MAE |
| 도루묵_선_SP_40미 | **4.2%** | GRU | MAE |
| 수꽃게_활_kg_특대 | **4.2%** | GRU | MAE |
| 오징어_선_cs상자_20미 | **4.3%** | GRU | MAE |
| 쭈꾸미_선_box_10코 | **4.8%** | GRU | MAE |
| 가리비_활_box_중 | **5.7%** | GRU | MAE |
| 깐해락_선_box_소 | **5.9%** | GRU | MAE |
| 새조개_활_box_중 | **6.2%** | GRU | MAE |
| 오징어_냉_SP_20미 | **6.5%** | GRU | MAE |
| 병어_선_SP_42미 | **6.6%** | GRU | MAE |
| 오징어_선_SP_소 | **6.9%** | GRU | MAE |
| 방어_활_미마리_중 | **7.3%** | GRU | MAE |
| 쭈꾸미_선_그물망_소 | **7.5%** | GRU | MAE |
| 오징어_선_SP_25미 | **8.6%** | GRU | MAE |
| 쭈꾸미_활_그물망_소 | **8.8%** | GRU | MAE |
| 소라_활_그물망_대 | **9.5%** | CNN-LSTM | MAE |

### Tier B — Production Ready (MAPE 10-20%)

| Config | MAPE | Model | Config | MAPE | Model |
|---|---|---|---|---|---|
| 문어_활_kg_중 | 10.3% | GRU | 깐바지락_선_box_소 | 12.6% | GRU |
| 진주담치_활_그물망_대 | 10.7% | GRU | 꼴뚜기_선_box_소 | 12.9% | GRU |
| 돌게_활_kg_중 | 10.9% | GRU | 전어_활_kg_중 | 12.9% | GRU |
| 가무락_활_그물망_대 | 10.9% | GRU | 오징어_선_SP_20미 | 13.3% | GRU |
| 쭈꾸미_선_box_중_dom | 11.0% | GRU | 수꽃게_활_kg_대중 | 13.4% | GRU |
| 깐굴_선_box_소 | 11.2% | LSTM | 농어_활_kg_1미_dom | 13.6% | GRU |
| 넙치_활_kg_중 | 11.3% | PatchTST | 암꽃게_활_kg_대 | 13.7% | GRU |
| 물바지락_활_box_2봉 | 11.4% | GRU | 수꽃게_활_kg_중 | 13.7% | GRU |
| 갑오징어_선_SP_2미 | 11.8% | GRU | 수꽃게_활_kg_대 | 13.9% | GRU |
| 수꽃게_활_kg_소 | 11.9% | GRU | 전어_선_kg_중 | 14.1% | GRU |
| 겉바지락_활_그물망_대 | 11.9% | GRU | 해삼_활_box_소 | 14.4% | GRU |
| 오징어_선_SP_중 | 12.3% | GRU | 매생이_선_box_중 | 14.7% | CNN-LSTM |
| 소라_활_box_대 | 12.4% | GRU | 소라_활_그물망_중 | 14.7% | GRU |
| 만디_선_box_2봉 | 12.5% | GRU | 물바지락_활_box_대 | 15.0% | GRU |
| 갑오징어_선_SP_3미 | 15.0% | GRU | 감성돔_활_kg_중_dom | 16.9% | GRU |
| 우럭_선_kg_중 | 15.1% | GRU | 진주담치_활_그물망_중 | 17.0% | GRU |
| 도다리_활_kg_중 | 15.2% | GRU | 분홍새우_선_SP_대 | 17.1% | PatchTST |
| 참돔_활_kg_2미_dom | 15.3% | Transformer | 깐홍합_선_box_소 | 17.2% | GRU |
| 농어_활_kg_중_dom | 15.8% | GRU | 동죽_활_box_소 | 17.6% | Transformer |
| 수꽃게_선_SP_중 | 16.0% | GRU | 메지_활_미마리_중 | 18.0% | CNN-LSTM |
| 물메기_선_kg_2미 | 16.0% | GRU | 새꼬막_활_포_소 | 18.0% | Transformer |
| 암꽃게_활_kg_중 | 16.2% | Transformer | 피꼬막_활_그물망_중 | 18.1% | Transformer |
| 칼바지락_활_box_대 | 16.6% | GRU | 대포오징어_선_SP_대 | 18.1% | GRU |
| 키조개_선_box_대 | 16.6% | GRU | 암꽃게_활_kg_소 | 18.6% | CNN-LSTM |
| 갑오징어_선_kg_중 | 18.7% | GRU | 민어_활_kg_중 | 19.0% | GRU |
| 만디_선_box_소 | 18.9% | GRU | 물메기_선_kg_3미 | 19.1% | GRU |
| 참돔_활_kg_중_dom | 19.6% | Transformer | 암꽃게_선_SP_중 | 19.6% | GRU |
| 가무락_활_그물망_중 | 19.8% | GRU | | | |

### Final Best-of-Breed (all models, all configs — v2 updated)

| Config | MAPE | Model | Pipeline |
|---|---|---|---|
| 바위굴_활_box_대 | **3.7%** | GRU | DL v2 |
| 쭈꾸미_선_box_중_dom | **11.0%** | GRU | DL v2 |
| 깐굴_선_box_소 | **11.2%** | LSTM | DL v2 |
| 넙치_활_kg_중 | **11.3%** | PatchTST | DL v2 |
| 농어_활_kg_1미_dom | **13.6%** | GRU | DL v2 |
| 수꽃게_활_kg_중 | **13.7%** | GRU | DL v2 |
| 암꽃게_활_kg_대 | **13.7%** | GRU | DL v2 |
| 수꽃게_활_kg_대 | **13.9%** | GRU | DL v2 |
| 도다리_활_kg_중 | **15.2%** | GRU | DL v2 |
| 참돔_활_kg_2미_dom | **15.3%** | Transformer | DL v2 |
| 농어_활_kg_중_dom | **15.8%** | GRU | DL v2 |
| 암꽃게_활_kg_중 | **16.2%** | Transformer | DL v2 |
| 감성돔_활_kg_중_dom | **16.9%** | GRU | DL v2 |
| 참돔_활_kg_중_dom | **19.6%** | Transformer | DL v2 |
| 참숭어_활_kg_중 | **22.7%** | GRU | DL v2 |
| 우럭_활_kg_중 | **23.1%** | GRU | DL v2 |
| 넙치_활_kg_2미 | **23.8%** | GRU | DL v2 |
| 감숭어_활_kg_중 | **26.1%** | Transformer | DL v2 |
| 방어_선_kg_중_dom | **49.5%** | CNN-LSTM | DL v2 |
| 민어_선_SP_중 | **68.5%** | GRU | DL v2 |

## Data Quality Flags

### High Quality (Lag-1 > 0.7, CV < 0.5, Days > 3000)
깐굴_선_box_소 (lag1=0.938), 암꽃게_활_kg_대 (0.899), 암꽃게_활_kg_중 (0.865), 바위굴_활_box_대 (0.807), 수꽃게_활_kg_대 (0.757), 넙치_활_kg_중 (0.751), 수꽃게_활_kg_중 (0.725)

### Challenging (Lag-1 < 0.4 or CV > 0.8)
방어_선_kg_중_dom (CV=1.034, lag1=0.300), 민어_선_SP_중 (CV=0.871, lag1=0.451), 도다리_활_kg_중 (lag1=0.346)

### Low Data Volume (< 500 days)
넙치_활_kg_2미 (431), 참돔_활_kg_2미_dom (340), 농어_활_kg_1미_dom (345) — premium grades have limited history. Models may be unstable.

### Inactive (no 2025 data)
우럭_활_kg_중, 농어_활_kg_중_dom, 도다리_활_kg_중, 감성돔_활_kg_중_dom — last traded in 2024. May need config update or data investigation.

## Guidelines for Adding New Configs

1. **Minimum requirements:** ≥200 trading days, ≥1 lot/day average
2. **Config ID format:** `{species}_{state}_{pkg}_{spec}[_dom]`
3. **Run CPU first:** `poc_test_mullet.py` for baseline (Naive/SMA/ARIMA/v10 LGBM)
4. **Run GPU if promising:** Add to `train_all_dl_models.py` SPECIES_CONFIGS
5. **Update this document** with data health stats and model results
6. **Data preprocessing:** All configs use v10 preprocessing (winsorized mean, log-target, outlier removal, origin-weight, adaptive VMD)

## Version History

| Date | Change |
|---|---|
| 2026-03-26 | Initial 7 sashimi species (v1-v6) |
| 2026-03-27 | TFT + DL model comparison (7 species) |
| 2026-03-28 | v10 preprocessing, quantile bands, fair DL comparison |
| 2026-03-29 | Added 감숭어, 참숭어, 쭈꾸미, 민어 |
| 2026-03-29 | Added 깐굴, 바위굴, 수꽃게, 암꽃게 |
| 2026-03-29 | Added premium grades: 수꽃게大, 암꽃게大, 넙치2미, 참돔2미, 농어1미 |
| 2026-03-30 | Full DL pipeline (v1) on all 20 configs |
| 2026-04-01 | DL v2: Optuna HPO + per-config loss + weather features (76 total). 12/20 improved |
| 2026-04-01 | Expansion: 125 new configs identified (Grade A: 35, Grade B: 88) |
| 2026-04-02 | Expansion complete: 145 total configs. 77 below 20% MAPE (53%). 22 below 10% |
| 2026-04-01 | Coastal weather data: 36,975 rows from Open-Meteo Archive (2006-2026, 5 ports) |
| 2026-04-01 | KHOA daily fetch: live station data for daily pipeline (4 stations) |
| 2026-04-01 | Streamlit dashboard: 5 pages (홈/시세/예측/모델/건강), all 504 species |
