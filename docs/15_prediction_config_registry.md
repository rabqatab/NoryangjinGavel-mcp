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

## Model Status

| Config ID | v10 LGBM | Best DL | Band? | Status |
|---|---|---|---|---|
| 넙치_활_kg_중 | 11.1% | GRU-Q 10.2% | Yes | **Production** |
| 우럭_활_kg_중 | 18.7% | TFT 14.7% | Yes | **Production** |
| 방어_선_kg_중_dom | 49.2% | TFT 15.6% | Yes | **Production** (TFT only) |
| 참돔_활_kg_중_dom | 18.9% | CNN-LSTM-Q 16.3% | Yes | **Production** |
| 농어_활_kg_중_dom | 19.3% | GRU-Q 12.8% | Yes | **Production** |
| 도다리_활_kg_중 | 21.1% | Transformer-Q 15.2% | Yes | **Production** |
| 감성돔_활_kg_중_dom | 17.1% | GRU-Q 12.5% | Yes | **Production** |
| 감숭어_활_kg_중 | 36.2% | — | Yes | Pending DL |
| 참숭어_활_kg_중 | 27.6% | — | Yes | Pending DL |
| 쭈꾸미_선_box_중_dom | 21.2% | — | Yes | Pending DL |
| 민어_선_SP_중 | 66.5% | — | Yes | **Directional only** |
| 깐굴_선_box_소 | 13.8% | — | Yes | Pending DL |
| 바위굴_활_box_대 | 15.9% | — (Naive 7.9%) | Yes | **Production** (Naive) |
| 수꽃게_활_kg_중 | 19.9% | — | Yes | Pending DL |
| 암꽃게_활_kg_중 | 22.2% | — | Yes | Pending DL |
| 수꽃게_활_kg_대 | 19.5% | — | Yes | Pending DL |
| 암꽃게_활_kg_대 | 19.6% | — | Yes | Pending DL |
| 넙치_활_kg_2미 | 21.2% | — | Yes | Pending DL |
| 참돔_활_kg_2미_dom | 17.9% | — (SMA-7 14.4%) | Yes | Pending DL |
| 농어_활_kg_1미_dom | 13.0% | — | Yes | Pending DL |

> DL results will be updated when the GPU training run completes (~2-3 hours).

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
| 2026-03-30 | Full DL pipeline running on all 20 configs (pending results) |
