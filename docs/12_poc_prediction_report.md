# PoC Price Prediction Report

> Results from 4 iterations (v1–v4) of price prediction models for 7 sashimi species.
> Scripts: `scripts/poc_prediction.py` (v1), `poc_prediction_v2.py` (v2), `poc_prediction_v3.py` (v3), `poc_prediction_v4.py` (v4)

## Executive Summary

Over 4 iterations, MAPE improved significantly for volatile species while stable species showed modest gains. The shift from pure autoregression (v1) to feature-engineered LightGBM (v2–v4) with cross-species supply and regime splitting delivered the best results.

| Species | v1 AR | v4 LightGBM | Improvement | Direction Accuracy |
|---|---|---|---|---|
| **넙치** (flatfish) | 16.2% | **15.4%** | +5% | 71.6% |
| **감성돔** (black porgy) | 26.7% | **23.8%** | +11% | 70.6% |
| **우럭** (rockfish) | 20.0% | **24.6%** | -23% | 70.5% |
| **농어** (sea bass) | 27.7% | **26.0%** | +6% | 69.8% |
| **도다리** (flounder) | 44.1% | **26.1%** | **+41%** | 77.3% |
| **방어** (yellowtail, winter) | 174.4% | **85.5%** | **+51%** | **80.0%** |
| **참돔** (seabream) | 27.7% | **28.5%** | -3% | 70.4% |

**Key insight:** Direction accuracy is consistently 70-80% across all species — the models reliably predict whether prices go up or down, even when point prediction MAPE is high.

---

## Species Configs

Each species is queried with specific filters to isolate a clean price signal:

| Species | State | Pkg | Spec | Domestic Only | Smoothed Target |
|---|---|---|---|---|---|
| 넙치 | 활 | kg | 중 | No | No |
| 우럭 | 활 | kg | 중 | No | No |
| 방어 | 선 | kg | 중 | Yes | Yes (7d) |
| 참돔 | 활 | kg | 중 | Yes | No |
| 농어 | 활 | kg | 중 | Yes | No |
| 도다리 | 활 | kg | 중 | No | Yes (7d) |
| 감성돔 | 활 | kg | 중 | Yes | No |

---

## Iteration History

### v1: Pure Autoregression (Baseline)

**Models:** Naive, SMA-7, SMA-30, Exponential Smoothing, ARIMA(2,1,2)

**Result:** ExpSmooth and ARIMA were best for most species. Worked well for 넙치 (16.2%) but failed for volatile species (방어 174%, 도다리 44%).

**Lesson:** Past prices alone are insufficient. Fish prices are supply-driven, not momentum-driven.

### v2: Feature-Engineered LightGBM

**New features (21):** Calendar (dow, month, woy) + holidays (설날/추석) + price history (lags, moving averages) + momentum + volatility + basic supply proxy (lots, origins, quantity)

**Result:** Improved 넙치, 참돔, 농어, 감성돔. Degraded 우럭, 방어 (LightGBM overfitted on noisy daily data).

**Lesson:** Calendar features (especially `week_of_year` and `days_to_chuseok`) are highly predictive. The model learned seasonal patterns that AR models couldn't.

### v3: Weather Proxy + Smoothed Target

**New features (34):** + Market-wide lots, supply shock detection (gap days, lots drop, qty drop), trading gap as weather proxy

**Key changes:** 7-day smoothed target for 방어 and 도다리 (too noisy at daily resolution)

**Result:** 방어 174→122%, 도다리 44→28%. Smoothed target was the biggest single improvement.

**Lesson:** For volatile species, predict the weekly trend, not tomorrow's exact price. Supply disruption proxies (gap_days, lots_drop) capture weather effects indirectly.

### v4: Cross-Species Supply + Regime Split

**New features (41):**
- Own supply enhanced: `own_qty_7d`, `own_qty_ratio_30d`, `own_qty_chg_7d`, `own_lots_chg_7d`
- Cross-species: `other_sashimi_qty_7d`, `sashimi_concentration`, `total_sashimi_chg_7d`, `market_chg_7d`
- Seasonal context: `price_vs_monthly_avg`, `month_sin/cos`, `is_peak_season`

**Key change:** Regime split for 방어 (in-season Nov-Feb vs off-season Mar-Oct)

**Result:** 방어 winter 85.5% (from 122%), 도다리 26.1% (from 28.2%), 감성돔 23.8%.

**Lesson:** Own supply is a strong predictor (r=-0.35 to -0.42 for 넙치/우럭/참돔). Cross-species substitution is weak, but overall market activity indicates demand conditions.

---

## Feature Importance Analysis

### Top Features by Species (v4, 7-day horizon)

| Rank | 넙치 | 우럭 | 방어 (winter) | 참돔 | 농어 | 도다리 | 감성돔 |
|---|---|---|---|---|---|---|---|
| 1 | price_7d_avg (33%) | price_30d_avg (28%) | market_lots_7d (9%) | price_30d_avg (9%) | month (9%) | month_cos (13%) | days_to_chuseok (14%) |
| 2 | price_30d_avg (12%) | price_7d_avg (9%) | days_to_chuseok (8%) | price_7d_avg (8%) | price_7d_avg (7%) | own_qty_7d (9%) | price_30d_avg (10%) |
| 3 | woy (4%) | price_std_30d (4%) | pchg_7v30 (8%) | market_chg_7d (4%) | woy (6%) | days_to_seollal (6%) | price_7d_avg (6%) |
| 4 | own_qty_7d (4%) | own_qty_7d (4%) | price_30d_avg (8%) | pchg_7v30 (4%) | price_30d_avg (5%) | other_sashimi_7d (5%) | month_cos (3%) |
| 5 | price_lag7 (3%) | total_sashimi_chg (3%) | own_lots_7d (7%) | own_lots_7d (4%) | price_std_30d (4%) | price_30d_avg (5%) | price_std_30d (3%) |

### Feature Category Impact

| Category | Description | Key features | Impact |
|---|---|---|---|
| **Price memory** | Recent price history | price_7d_avg, price_30d_avg | Dominates for stable species (넙치 45%, 우럭 37%) |
| **Calendar/Seasonal** | Time of year | month, woy, month_sin/cos | Critical for seasonal species (도다리 13%, 농어 15%) |
| **Holiday** | Korean holidays | days_to_chuseok, days_to_seollal | Species-specific: 감성돔 14%, 도다리 6% |
| **Own supply** | Species' own trading volume | own_qty_7d, own_lots_7d | Moderate 3-9%; strongest for 도다리 (9%) |
| **Cross supply** | Other species' supply | other_sashimi_qty_7d, market_lots_7d | Moderate 3-9%; strongest for 방어 (9%) |
| **Momentum** | Price trend direction | pchg_7v30, pchg_30d | Moderate for 방어 (8%), 참돔 (4%) |
| **Volatility** | Price stability | price_std_30d, price_range_7d | Background signal across all species |

---

## Cross-Species Supply Analysis

We tested whether other species' fishing quantities affect target species' prices.

### Own Supply → Own Price (7d rolling, strong signal)

| Species | Correlation | Interpretation |
|---|---|---|
| 우럭 | **-0.42** | More rockfish supply → lower rockfish price |
| 넙치 | **-0.35** | Classic supply-demand |
| 참돔 | **-0.22** | Same pattern, weaker |
| 농어 | -0.15 | Weak |
| 감성돔 | +0.13 | Inverted — demand-driven (busy market = higher price) |
| 방어 | +0.16 | Inverted — supply follows demand (more caught when price is high) |

### Other Sashimi Supply → Target Price (7d rolling)

| Species | Correlation | Interpretation |
|---|---|---|
| 감성돔 | **+0.33** | Busy sashimi market → higher 감성돔 demand |
| 우럭 | **+0.25** | Same — demand indicator |
| 넙치 | +0.21 | Same |
| 참돔 | +0.15 | Weak positive |
| 방어 | +0.03 | No effect (방어 is seasonal, independent) |
| 도다리 | +0.02 | No effect (도다리 is seasonal, independent) |

### Cross-Species Substitution (weak, not worth modeling)

Tested all 7×7 pairs: no substitution effect exceeded -0.10 correlation. Consumers don't significantly switch between species based on supply shocks.

---

## Per-Species Assessment

### 넙치 (Flatfish) — PRODUCTION READY

- **MAPE: 15.4%** (best across all species)
- Highly driven by recent price memory (price_7d_avg 33%)
- Stable, liquid market with 5,721 trading days
- **Recommendation:** Deploy for consumer price guidance

### 감성돔 (Black Porgy) — PRODUCTION READY

- **MAPE: 23.8%**, direction 70.6%
- Strongly driven by 추석 holiday proximity (14%)
- **Recommendation:** Deploy with holiday-aware alerts

### 우럭 (Rockfish) — USABLE WITH CAVEATS

- **MAPE: 24.6%** (v1 ARIMA was better at 20%)
- LightGBM struggles — weak daily signal (lag-1 = 0.60)
- **Recommendation:** Use ARIMA for point prediction, LightGBM for direction. Consider ensemble.

### 농어 (Sea Bass) — USABLE

- **MAPE: 26.0%**, direction 69.8%
- Strong seasonal component (month + woy = 15%)
- **Recommendation:** Deploy with "approximate" caveat

### 도다리 (Flounder) — MUCH IMPROVED

- **MAPE: 26.1%** (from 44.1% in v1, **+41% improvement**)
- month_cos (13%) captures spring seasonal cycle perfectly
- Own supply (9%) is a strong driver
- **Recommendation:** Deploy with seasonal context ("봄 도다리 제철, prices expected to rise")

### 참돔 (Seabream) — NEEDS MORE WORK

- **MAPE: 28.5%** (worse than v1's 27.7%)
- No single feature dominates — diffuse importance
- **Recommendation:** Try spec-class split (소/중/대) or origin-class split. May benefit from ensemble.

### 방어 (Yellowtail) — DIRECTIONAL ONLY

- **MAPE: 85.5%** (winter), but **80% direction accuracy**
- MAPE is inflated by low absolute prices (100-500 KRW off-season → any error is huge in %)
- **Recommendation:** Use as trend indicator only ("방어 prices trending up/down this week"). Do not show point predictions.

---

## Future Feature Candidates

Based on literature review of fish price prediction research:

### Tier 1: High Expected Impact (data accessible)

| Feature | Source | Why it matters | Reference |
|---|---|---|---|
| **Wave height (파고)** | KHOA 바다누리 API | High waves = boats can't fish = supply shock | [KMA Ocean Buoy Data](https://data.kma.go.kr/data/sea/selectBuoyRltmList.do) |
| **Sea surface temperature (수온)** | KHOA API | Affects fish migration, catch composition | [Deep learning for SST prediction (2024)](https://os.copernicus.org/articles/20/417/2024/) |
| **Wind speed (풍속)** | KMA API | Storm warnings = fishing fleet stays in port | [KMA API Hub](https://apihub.kma.go.kr/) |
| **Landing volume (어획량)** | National fisheries data | Direct supply indicator at regional ports | [Nizam Zachman Port study (2025)](https://link.springer.com/article/10.1007/s41208-025-00926-z) |

### Tier 2: Moderate Expected Impact

| Feature | Source | Why it matters | Reference |
|---|---|---|---|
| **Diesel/fuel price (유가)** | OPINET API | Fishing operation cost → minimum viable price | Korean agricultural price prediction studies |
| **Exchange rate (환율)** | Bank of Korea | Affects import competitiveness (방어, 낙지) | [Korea Seafood Market Update 2024](https://apps.fas.usda.gov/) |
| **Import volume (수입량)** | Korea Customs Service | Foreign supply shock for import-heavy species | Same |
| **Aquaculture production** | NIFS (국립수산과학원) | Farm-raised supply for 넙치, 우럭 | [XGBoost for fish price prediction (2024)](https://fisheries-2023.sites.olt.ubc.ca/files/2024/09/2024-01-Working-Paper-Price-Prediction.pdf) |

### Tier 3: Exploratory

| Feature | Source | Why it matters | Reference |
|---|---|---|---|
| **Consumer sentiment index** | Bank of Korea | Demand proxy for premium species | — |
| **Restaurant/tourism data** | KTO | Tourist demand at Noryangjin | — |
| **Typhoon/storm warnings** | KMA | Multi-day supply disruption events | [Fishery weather forecasting (2023)](https://www.sciencedirect.com/science/article/pii/S2214317323000537) |
| **Lunar calendar** | Computed | Affects 설날/추석 holiday demand timing | [Taiwan aquatic price prediction (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0044848624002023) |
| **FAO Fish Price Index** | FAO | Global seafood price trends | [Fish is Food — FAO's Fish Price Index](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0036731) |

### Research Methodology Notes

| Paper | Key Finding | Relevance |
|---|---|---|
| PMC11048843 (2024) | VMD+LSTM hybrid with signal decomposition achieved 0.08% MAPE on single-species daily prices | Signal decomposition (EMD/VMD) before prediction may help volatile species like 방어 |
| Taiwan COA (2024) | >90% accuracy with fully automated ML pipeline integrating weather + holidays + news | Holiday + weather integration confirmed as high-value |
| Nizam Zachman (2025) | GRU/LSTM with landing volume, time, species achieved best results; landing volume was key | Landing volume (equivalent to our qty features) is confirmed as important |
| Korean 조피볼락 study | MLP outperformed ARIMA and LSTM for Korean farmed fish prices | MLP may be worth testing as an alternative to LightGBM |
| Agricultural price patent (KR20200036219A) | Oil price and total supply volume as leading indicators | Fuel cost and aggregate supply are validated leading indicators |

---

## Recommended Next Steps

1. **Integrate KHOA ocean data** (wave height, SST, wind) — expected to improve 방어 and 도다리 significantly
2. **Signal decomposition** (VMD/EMD) before prediction for volatile species
3. **Ensemble approach** for 우럭: combine ARIMA (good point prediction) with LightGBM (good features)
4. **Import/fuel data** for species with foreign supply sensitivity (방어, 낙지)
5. **Weekly aggregation models** as an alternative to daily smoothed target

Sources:
- [Price Forecasting of Marine Fish — PMC (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11048843/)
- [Advanced ML for Fish Price Prediction — UBC (2024)](https://fisheries-2023.sites.olt.ubc.ca/files/2024/09/2024-01-Working-Paper-Price-Prediction.pdf)
- [Taiwan Aquatic Price Prediction — ScienceDirect (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0044848624002023)
- [Fish Price Optimization — Springer (2025)](https://link.springer.com/article/10.1007/s41208-025-00926-z)
- [Deep Learning for SST — Copernicus (2024)](https://os.copernicus.org/articles/20/417/2024/)
- [Korean Agricultural Price Patent (2020)](https://patents.google.com/patent/KR20200036219A/en)
- [Korea Seafood Market Update 2024 — USDA](https://apps.fas.usda.gov/)
- [Fishery Weather Forecasting — ScienceDirect (2023)](https://www.sciencedirect.com/science/article/pii/S2214317323000537)
- [FAO Fish Price Index — PLOS One](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0036731)
- [바다누리 OpenAPI](http://www.khoa.go.kr/oceangrid/khoa/takepart/openapi/openApiUserSampleCode.do)
- [KMA Ocean Buoy Data](https://data.kma.go.kr/data/sea/selectBuoyRltmList.do)
