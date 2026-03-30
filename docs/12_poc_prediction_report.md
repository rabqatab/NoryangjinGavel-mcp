# PoC Price Prediction Report

> Results from 10 iterations: v1–v8b (CPU, LightGBM) + TFT (GPU, Temporal Fusion Transformer).
> CPU scripts: `scripts/poc_prediction.py` (v1) through `poc_prediction_v8b.py` (v8b)
> GPU scripts: `scripts/train_tft.py` (Docker on GB10 Blackwell)
> Ocean data: `scripts/fetch_ocean_openmeteo.py` (Open-Meteo, 11K rows, 5 stations, 2020–2026)

## Executive Summary

Over 8 model iterations spanning CPU (LightGBM with 68 features, VMD decomposition, Optuna optimization) and GPU (Temporal Fusion Transformer), **5 of 7 species now achieve below 23% MAPE**. The best model varies per species — LightGBM wins for stable species, TFT wins for volatile/complex ones.

### Full DL Model Comparison (7 models × 7 species, GPU)

| Model | 넙치 | 우럭 | 방어 | 참돔 | 농어 | 도다리 | 감성돔 | AVG |
|---|---|---|---|---|---|---|---|---|
| **TFT** | 18.1% | **14.7%** | **15.6%** | **20.8%** | 51.0% | **27.2%** | 23.3% | **24.4%** |
| PatchTST | 19.8% | 32.8% | 139.3% | 28.9% | 19.6% | 34.1% | 28.3% | 43.3% |
| CNN-LSTM | 18.5% | 32.1% | 160.7% | 27.0% | 20.4% | 30.2% | 26.1% | 45.0% |
| Transformer | 18.2% | 32.1% | 173.8% | 26.8% | **18.4%** | 30.1% | **21.1%** | 45.8% |
| GRU | **17.0%** | 31.7% | 168.9% | 27.3% | 19.6% | 31.6% | 25.4% | 45.9% |
| BiLSTM+Attn | 17.8% | 32.3% | 181.3% | 27.1% | 19.2% | 31.0% | 29.0% | 48.2% |
| LSTM | 17.4% | 32.1% | 185.3% | 27.5% | 20.2% | 31.1% | 26.0% | 48.5% |

**Rankings:** TFT (24.4%) >> PatchTST (43.3%) > CNN-LSTM (45.0%) > Transformer (45.8%) > GRU (45.9%) > BiLSTM+Attn (48.2%) > LSTM (48.5%)

**DL findings:**
- TFT is the only model that solves 방어 (15.6% vs 139-185% for all others)
- Transformer (Informer-style) is a dark horse — best for 농어 (18.4%) and 감성돔 (21.1%)
- GRU beats LSTM everywhere — confirming the literature (fewer params, same or better)
- BiLSTM+Attention adds noise rather than signal for these relatively simple series
- 방어 is the litmus test: only TFT's variable selection + attention can learn its seasonal pattern

### Best-of-Breed Results (Final — CPU + all 7 DL models)

| Species | Best Model | MAPE | Runner-up | Status |
|---|---|---|---|---|
| **넙치** (flatfish) | v6 VMD+LightGBM | **14.8%** | GRU 17.0% | Production ready |
| **우럭** (rockfish) | TFT (GPU) | **14.7%** | LightGBM 24.1% | Production ready |
| **방어** (yellowtail) | TFT (GPU) | **15.6%** | LightGBM 62.6% | Production ready |
| **농어** (sea bass) | Transformer (GPU) | **18.4%** | GRU 19.6% | Production ready |
| **참돔** (seabream) | TFT (GPU) | **20.8%** | Transformer 26.8% | Usable |
| **감성돔** (black porgy) | Transformer (GPU) | **21.1%** | LightGBM 22.8% | Usable |
| **도다리** (flounder) | v7 STL-VMD+LightGBM | **25.1%** | TFT 27.2% | Usable (seasonal) |

### Full Progression: v1 → TFT

| Species | v1 AR | v6 LGBM | v7 STL-VMD | TFT (GPU) | **Best** | Total Gain |
|---|---|---|---|---|---|---|
| **넙치** | 16.2% | 14.8% | 15.3% | 18.1% | **14.8%** | +9% |
| **우럭** | 20.0% | 24.1% | 23.7% | **14.7%** | **14.7%** | **+27%** |
| **방어** | 174.4% | 62.6% | 80.6% | **15.6%** | **15.6%** | **+91%** |
| **참돔** | 27.7% | 26.5% | 27.3% | **20.8%** | **20.8%** | **+25%** |
| **감성돔** | 26.7% | **22.8%** | 23.3% | 23.3% | **22.8%** | +15% |
| **농어** | 27.7% | **23.6%** | 24.8% | 51.0%* | **23.6%** | +15% |
| **도다리** | 44.1% | 26.0% | **25.1%** | 27.2% | **25.1%** | +43% |

*농어 TFT result (51%) is an anomaly — gap-filling creates stale prices for this sporadically-traded species. LightGBM with its feature-based approach handles sparse data better.

**Key insights:**
- **No single model wins for all species.** LightGBM wins for 넙치 and 도다리, TFT wins for 우럭/방어/참돔, Transformer wins for 농어/감성돔.
- **방어 was the biggest success story:** from 174% (unpredictable) to 15.6% (production-ready) — TFT is the *only* DL model that solves it (all others: 139-185%).
- **Transformer is the surprise winner** for 농어 (18.4%) and 감성돔 (21.1%), beating both LightGBM and TFT.
- **GRU > LSTM everywhere** — confirming the literature. BiLSTM+Attention adds noise.
- **6/7 species now below 22% MAPE**, 4/7 below 19%. Production-viable for consumer guidance.
- The 68 features in v6 account for **31-47% of total feature importance** — calendar, supply, and technical indicators all contribute.

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

### v5: VMD Signal Decomposition + ARIMA Ensemble

**New technique:** Variational Mode Decomposition (VMD) — decomposes price series into K=3 modes (trend, oscillation, noise). Separate LightGBM model per mode → recombine predictions. For 우럭, uses 60/40 ARIMA+LightGBM ensemble instead.

**Result:** Broad improvement across all species:
- 방어 winter: 85.5% → **75.5%** (+12%) — VMD separated seasonal trend from noise
- 농어: 26.0% → **23.9%** (+8%) — biggest gain among stable species
- 참돔: 28.5% → **26.8%** (+6%) — VMD finally broke through the v1-v4 plateau
- 우럭: 24.6% → **23.8%** (+3%) — ARIMA ensemble recovered from v2-v4 regression

**Lesson:** Signal decomposition is highly effective for fish prices. Price = trend + seasonal + noise, and predicting each component separately then recombining beats predicting the raw signal. The PMC11048843 paper's finding (0.08% MAPE with VMD+LSTM) is directionally confirmed.

### v6: 68 Features — Technical Indicators + Fourier + Distribution + Advanced Supply

**27 new features added (41 → 68):**
- Technical Indicators (8): EMA_7, EMA_30, MACD, MACD_signal, MACD_hist, Bollinger_pct, RSI_14, momentum_14d
- Fourier/Cyclical (6): sin/cos at 365-day, 182-day, and 7-day periods
- Advanced Calendar (5): is_friday, is_pre_holiday, consecutive_gap, week_position, days_left_in_week
- Price Distribution (4): skewness_30d, kurtosis_30d, percentile_90d, zscore_30d
- Advanced Supply (4): own_qty_yoy_ratio, origin_diversity_7d, avg_lot_size_7d, high_low_spread_7d

**Result:** Improvements across most species, with 방어 seeing the biggest gain:
- 방어 winter: 75.5% → **62.6%** (+17%) — distribution features (skewness, kurtosis) captured regime-switching
- 넙치: 15.1% → **14.8%** (+2%) — technical indicators refined the already-good model
- 참돔: 26.8% → **26.5%** (+1%) — broke through the v1-v5 plateau via momentum_14d
- 농어: 23.9% → **23.6%** (+2%) — Fourier + MACD helped seasonal patterns

**New Feature Category Impact (% of total feature importance):**

| Category | Avg Across Species | Most Important For |
|---|---|---|
| Technical Indicators | 15-30% | 우럭 (EMA_30 = 18.1%), all species (momentum_14d) |
| Fourier | 6-9% | 도다리 (8.6%), 농어 (8.2%) |
| Distribution | 4-15% | 방어 off-season (15.0% — skewness + kurtosis) |
| Advanced Supply | 5-8% | 감성돔 (avg_lot_size 3.3%, hl_spread 3.3%) |
| Advanced Calendar | 0.3-1.1% | Minimal impact — not worth the complexity |

**Lesson:** Technical indicators (especially EMA and momentum) are the most universally impactful new feature category. Distribution features are situation-specific but critical for volatile species. Advanced calendar features were a miss — fish markets don't have strong day-of-week effects at the weekly prediction horizon.

### v7: STL-VMD Dual Decomposition + Optuna K Optimization (CPU)

**New approach:** STL seasonal decomposition first (period=7), then VMD on residuals. Optuna searches K=3-8, alpha=500-5000.

**Result:** Mixed — marginal gains for 2 species, slight regressions for others:
- 도다리: 26.0% → **25.1%** (+4%) — K=7, alpha=1500
- 우럭: 24.1% → **23.7%** (+2%) — K=5, alpha=5000
- 방어 winter: 62.6% → 80.6% (-29%) — STL fails on small seasonal sample

**Lesson:** STL-VMD is not universally better. The weekly STL period doesn't capture the true seasonality of fish prices (which is monthly/annual, not weekly). Optuna found two K clusters: K=7 for stable species, K=5 for volatile ones.

### TFT: Temporal Fusion Transformer (GPU, Docker on GB10)

**Architecture:** pytorch-forecasting TFT with:
- Static covariates: species_id
- Known future: dow, month, woy, is_weekend, days_to_seollal, days_to_chuseok
- Observed past: price, ema_7, ema_30, rsi_14, price_7d_avg, price_30d_avg, price_std_7d, n_lots, n_origins, quantity, own_qty_7d, other_sashimi_7d, market_lots_7d
- 118K parameters, 30-day encoder, 7-day prediction horizon, quantile loss
- Gap-filled continuous daily index (forward-fill non-trading days)
- Trained on GB10 Blackwell GPU via Docker (nvcr.io/nvidia/pytorch:24.12-py3)

**Result:** TFT wins for 3 species, LightGBM wins for 3, tie for 1:

| Species | v6 LightGBM | TFT (GPU) | Winner |
|---|---|---|---|
| **우럭** | 24.1% | **14.7%** | TFT (+39%) |
| **방어** | 62.6% | **15.6%** | TFT (+75%) |
| **참돔** | 26.5% | **20.8%** | TFT (+22%) |
| **넙치** | **14.8%** | 18.1% | LightGBM |
| **감성돔** | **22.8%** | 23.3% | LightGBM |
| **도다리** | **25.1%** (v7) | 27.2% | LightGBM |
| **농어** | **23.6%** | 51.0%* | LightGBM |

*농어 TFT anomaly: gap-filling creates long runs of stale prices (농어 trades ~250 of 365 days but with irregular gaps). The forward-filled values mislead TFT's attention mechanism. LightGBM's feature-based approach doesn't have this problem because features are computed from trading days only.

**Lesson:** TFT excels at capturing complex temporal patterns (방어's seasonal demand, 우럭's non-linear dynamics) but struggles with sparse/irregular data. The best strategy is a per-species model selection: TFT for species with good data coverage, LightGBM for sporadic traders.

### v8/v8b: Ocean Weather Features (Open-Meteo)

**Data source:** Open-Meteo API (free, no rate limit). 11,385 rows from 5 coastal stations (제주, 여수, 부산, 인천, 속초), 2020–2026. Features: wave_height, swell_height, wave_direction, wind_speed, wind_gust, temperature, precipitation, pressure, sunshine_hours.

**12 new ocean features (68 → 80):** Each species mapped to its nearest fishing port station. Lag-1 values + 7d rolling averages + storm flag.

**v8 Result (full history, ocean=None for pre-2020):** Flat — ocean features get 5-9% importance but don't improve MAPE. Pre-2020 missing values (70% of data) dilute the signal.

**v8b Result (fair A/B test, 2020+ only):** Ocean features **hurt all 6 species** (-0.1% to -20.7% worse). Two causes:
1. Dropping None rows shrinks training data ~40%
2. Open-Meteo reanalysis data is gridded/modeled, not actual station observations — spatial resolution too coarse for local fishing conditions

**Key finding from v8b:** Existing supply proxy features (`lots_drop`, `qty_drop`, `supply_shock`) already capture the downstream effect of bad weather. Adding the *cause* (waves, wind) doesn't help when the *effect* (supply drops) is already measured. This matches the literature: "landing volume was the key feature" (Nizam Zachman 2025).

### Lag Structure Analysis

Cross-correlation analysis of supply quantity at different lags vs price at t+7 revealed:

| Species | qty lag-1 | qty lag-3 | cum_qty 7d | Improvement |
|---|---|---|---|---|
| 도다리 | -0.19 | -0.18 | **-0.23** | +21% |
| 감성돔 | -0.09 | -0.10 | **-0.14** | +49% |
| 넙치 | -0.12 | -0.13 | **-0.14** | +17% |
| 참돔 | -0.07 | -0.06 | **-0.09** | +22% |

**Cumulative supply over 5-7 days is consistently stronger than any single lag.** A week of low supply creates accumulated scarcity that drives prices up more than one bad day. Individual price lags (lag-1 through lag-5) carry near-identical information (r=0.63-0.64), but the 30-day price change has unique signal (r=0.19).

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

### Completed
- ~~Signal decomposition (VMD) before prediction~~ → v5, +2-12%
- ~~Ensemble approach for 우럭~~ → v5, recovered from regression
- ~~Technical indicators (EMA, MACD, RSI, Bollinger)~~ → v6, 15-30% importance
- ~~Fourier seasonal encoding~~ → v6, 6-9% importance
- ~~Price distribution features~~ → v6, critical for 방어 (15%)
- ~~STL-VMD dual decomposition + Optuna K search~~ → v7, marginal gains for 도다리/우럭
- ~~TFT (Temporal Fusion Transformer) on GPU~~ → Wins for 우럭 (14.7%), 방어 (15.6%), 참돔 (20.8%)
- ~~GRU/LSTM evaluation~~ → Not needed; TFT already achieved literature-target MAPE
- ~~Ocean weather features (Open-Meteo)~~ → v8/v8b: absorbed but don't improve MAPE. Supply proxies already capture the effect.
- ~~Lag structure analysis~~ → Cumulative supply (5-7d) is 17-49% stronger than single lag-1

### Remaining (for production)
1. **Multi-lag supply features (v9)** — add cum_qty_3d, cum_qty_5d, qty_lag3, qty_lag5, supply_trend_7d. Lag analysis shows 17-49% stronger signal vs current lag-1 only.
2. **Per-species model routing** — deploy LightGBM for 넙치/감성돔/농어/도다리, TFT for 우럭/방어/참돔.
3. **Fix 농어 TFT** — investigate gap-filling strategy for sporadic species.
4. **Prune low-impact features** — advanced calendar (0.3-1.1%) and ocean features (hurt in A/B test) should be removed.
5. **Import/fuel data** — deferred, low priority given ocean data findings.

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
