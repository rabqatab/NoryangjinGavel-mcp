# Prediction System Design

## Overview

> **Note:** This document was the original design. The actual implementation evolved significantly through 12 iterations. See `docs/12_poc_prediction_report.md` for the final results and `docs/15_prediction_config_registry.md` for the production config list.

| Attribute | Original Design | Actual Implementation |
|-----------|----------------|----------------------|
| **Models** | Exp. Smoothing, ARIMA, Prophet | LightGBM (v10), TFT, GRU, Transformer, CNN-LSTM |
| **Libraries** | statsmodels, prophet, sklearn | lightgbm, pytorch, pytorch-forecasting, vmdpy |
| **Features** | Price history only | 68 features (calendar, supply, technical, Fourier, distribution) |
| **Preprocessing** | None planned | 5 fixes: winsorized mean, log-target, outlier removal, origin-weight, adaptive VMD |
| **Target** | Top 30 species | 20 configs across 15 species (with spec/state/origin filtering) |
| **Output** | Point predictions | Point + quantile bands (p10/p50/p90) + conformal intervals |
| **Best MAPE** | Expected 3-15% | Achieved 10.2-16.3% (sashimi), 13.8-36.2% (other) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   PREDICTION SYSTEM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Daily Prediction Pipeline (runs after 12:00 PM crawl)     │     │
│  │  ══════════════════════════════════════════════════════    │     │
│  │                                                             │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │     │
│  │  │ Short-term  │  │ Medium-term │  │ Long-term   │        │     │
│  │  │ Forecast    │  │ Forecast    │  │ Seasonal    │        │     │
│  │  │             │  │             │  │             │        │     │
│  │  │ Exp.Smooth  │  │ ARIMA       │  │ Prophet     │        │     │
│  │  │ (1-7 days)  │  │ (1-4 weeks) │  │ (monthly)   │        │     │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │     │
│  │         │                │                │                │     │
│  │         └────────────────┼────────────────┘                │     │
│  │                          ▼                                  │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │     │
│  │  │  Anomaly    │  │  Trend      │  │ Volatility  │        │     │
│  │  │  Detection  │  │  Analysis   │  │  Index      │        │     │
│  │  │             │  │             │  │             │        │     │
│  │  │ Z-score/IQR │  │ LinReg/SK   │  │ Rolling Std │        │     │
│  │  │ Iso.Forest  │  │             │  │             │        │     │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │     │
│  │         │                │                │                │     │
│  │         └────────────────┼────────────────┘                │     │
│  │                          ▼                                  │     │
│  │              ┌───────────────────────┐                     │     │
│  │              │   DuckDB Tables        │                     │     │
│  │              │   (prediction results) │                     │     │
│  │              └───────────────────────┘                     │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  MCP Server (query pre-computed predictions)               │     │
│  │  ══════════════════════════════════════════                │     │
│  │  • predict_price      • get_seasonality                    │     │
│  │  • detect_anomalies   • get_market_insight                 │     │
│  │  • get_volatility     • compare_forecast_accuracy          │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prediction Models

| Model | Library | Use Case | Horizon | Resource |
|-------|---------|----------|---------|----------|
| **Exponential Smoothing** | statsmodels | Short-term price | 1-7 days | Very Low |
| **ARIMA/SARIMA** | statsmodels | Medium-term price | 1-4 weeks | Low |
| **Prophet** | prophet | Seasonal patterns | Monthly/Yearly | Medium |
| **Linear Regression** | sklearn | Trend direction | Any | Very Low |
| **Isolation Forest** | sklearn | Anomaly detection | Historical | Low |
| **Z-score / IQR** | scipy | Anomaly detection | Real-time | Very Low |
| **Rolling Statistics** | numpy/scipy | Volatility index | Rolling window | Very Low |

### Model Selection Rationale

**Short-term (1-7 days): Exponential Smoothing**
- Captures recent trends and weekly seasonality
- Very fast, low memory
- Good for immediate price movements

**Medium-term (1-4 weeks): ARIMA**
- Handles non-stationary data (differencing)
- Captures autocorrelation patterns
- Confidence intervals built-in

**Long-term (Monthly): Prophet**
- Excellent seasonal decomposition
- Handles holidays and special events
- Interpretable components (trend, seasonality)

**Anomaly Detection: Ensemble Approach**
- Z-score for simple statistical outliers
- IQR for robust outlier detection
- Isolation Forest for complex patterns

---

## Database Tables

> Full schema details: See [`03_database_design.md`](./03_database_design.md) for complete DuckDB schema.

```sql
-- Price forecasts (pre-computed daily)
-- Stored in DuckDB: data/fish_market.duckdb
CREATE TABLE prediction_forecasts (
    id INTEGER PRIMARY KEY,
    species VARCHAR NOT NULL,             -- Species name (denormalized)
    base_date DATE NOT NULL,              -- Date prediction was made
    target_date DATE NOT NULL,            -- Date being predicted
    horizon_days INTEGER NOT NULL,        -- 1, 7, 14, 30
    predicted_price INTEGER NOT NULL,
    ci_lower INTEGER NOT NULL,            -- 80% confidence interval
    ci_upper INTEGER NOT NULL,
    model_type VARCHAR NOT NULL,          -- 'exp_smoothing', 'arima', 'prophet'
    created_at TIMESTAMP DEFAULT now(),
    UNIQUE(species, base_date, horizon_days, model_type)
);

-- Seasonal patterns (updated monthly)
CREATE TABLE prediction_seasonality (
    id INTEGER PRIMARY KEY,
    species VARCHAR NOT NULL,
    month INTEGER NOT NULL,               -- 1-12
    seasonal_index DOUBLE NOT NULL,       -- 1.0 = average, >1 = above avg
    avg_price INTEGER,
    price_std INTEGER,
    sample_years INTEGER,                 -- Years of data used
    best_week INTEGER,                    -- Best week within month (1-4)
    updated_at TIMESTAMP DEFAULT now(),
    UNIQUE(species, month)
);

-- Detected anomalies
CREATE TABLE prediction_anomalies (
    id INTEGER PRIMARY KEY,
    species VARCHAR NOT NULL,
    detected_date DATE NOT NULL,
    actual_price INTEGER NOT NULL,
    expected_price INTEGER NOT NULL,
    deviation_pct DOUBLE NOT NULL,        -- % deviation from expected
    z_score DOUBLE NOT NULL,
    anomaly_type VARCHAR,                 -- 'spike', 'drop', 'volatility'
    severity VARCHAR,                     -- 'low', 'medium', 'high'
    created_at TIMESTAMP DEFAULT now()
);

-- Market insights (daily summary per species)
CREATE TABLE prediction_insights (
    id INTEGER PRIMARY KEY,
    species VARCHAR NOT NULL,
    insight_date DATE NOT NULL,
    trend_direction VARCHAR NOT NULL,     -- 'rising', 'falling', 'stable'
    trend_strength DOUBLE,                -- 0-1 scale
    volatility_index DOUBLE,              -- 0-1 scale (0=stable, 1=volatile)
    volatility_label VARCHAR,             -- 'low', 'medium', 'high'
    price_vs_seasonal VARCHAR,            -- 'above', 'below', 'normal'
    recommendation VARCHAR,               -- 'buy', 'wait', 'hold'
    confidence DOUBLE,                    -- 0-1 scale
    summary_text VARCHAR,                 -- Human-readable insight
    updated_at TIMESTAMP DEFAULT now(),
    UNIQUE(species, insight_date)
);

-- Model performance tracking
CREATE TABLE prediction_model_accuracy (
    id INTEGER PRIMARY KEY,
    species VARCHAR NOT NULL,
    model_type VARCHAR NOT NULL,
    horizon_days INTEGER NOT NULL,
    mape DOUBLE,                          -- Mean Absolute Percentage Error
    rmse DOUBLE,                          -- Root Mean Square Error
    directional_accuracy DOUBLE,          -- % correct up/down predictions
    sample_size INTEGER,
    evaluated_at TIMESTAMP DEFAULT now(),
    UNIQUE(species, model_type, horizon_days)
);
```

### Storage Estimation

| Table | Rows (Est.) | Row Size | Total Size |
|-------|-------------|----------|------------|
| prediction_forecasts | 30 species × 4 horizons × 30 days = 3,600 | ~60 bytes | ~220 KB |
| prediction_seasonality | 30 species × 12 months = 360 | ~50 bytes | ~18 KB |
| prediction_anomalies | ~500/year | ~60 bytes | ~30 KB |
| prediction_insights | 30 species × 30 days = 900 | ~200 bytes | ~180 KB |
| prediction_model_accuracy | 30 species × 3 models × 4 horizons = 360 | ~50 bytes | ~18 KB |
| **Total** | | | **~500 KB** |

---

## MCP Prediction Tools

### Tool: `predict_price`

Predict future prices for a fish species.

**Schema:**
```json
{
    "name": "predict_price",
    "description": "Predict future fish prices using statistical/ML models. Returns predictions with confidence intervals.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "species": {
                "type": "string",
                "description": "Fish species name in Korean (e.g., '고등어')."
            },
            "horizon": {
                "type": "string",
                "enum": ["1d", "7d", "14d", "30d", "all"],
                "default": "7d",
                "description": "Prediction horizon."
            },
            "include_history": {
                "type": "boolean",
                "default": false,
                "description": "Include recent historical prices for context."
            }
        },
        "required": ["species"]
    }
}
```

**Example Response:**
```json
{
    "species": "고등어",
    "species_en": "Mackerel",
    "base_date": "2025-01-02",
    "current_price": 34500,
    "predictions": [
        {
            "horizon": "1d",
            "target_date": "2025-01-03",
            "predicted_price": 34800,
            "ci_lower": 33200,
            "ci_upper": 36400,
            "confidence": 0.80,
            "model": "exp_smoothing",
            "change_pct": 0.87
        },
        {
            "horizon": "7d",
            "target_date": "2025-01-09",
            "predicted_price": 35200,
            "ci_lower": 32100,
            "ci_upper": 38300,
            "confidence": 0.80,
            "model": "arima",
            "change_pct": 2.03
        },
        {
            "horizon": "30d",
            "target_date": "2025-02-01",
            "predicted_price": 33100,
            "ci_lower": 28500,
            "ci_upper": 37700,
            "confidence": 0.80,
            "model": "prophet",
            "change_pct": -4.06
        }
    ],
    "model_accuracy": {
        "7d_mape": 8.5,
        "directional_accuracy": 0.72
    }
}
```

---

### Tool: `get_seasonality`

Get seasonal patterns for a fish species.

**Schema:**
```json
{
    "name": "get_seasonality",
    "description": "Analyze seasonal price patterns. Returns monthly indices and best/worst months to buy.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "species": {
                "type": "string",
                "description": "Fish species name in Korean."
            },
            "include_weekly": {
                "type": "boolean",
                "default": false,
                "description": "Include day-of-week patterns."
            }
        },
        "required": ["species"]
    }
}
```

**Example Response:**
```json
{
    "species": "고등어",
    "species_en": "Mackerel",
    "data_years": 21,
    "monthly_patterns": [
        {"month": 1, "name": "January", "index": 0.92, "avg_price": 31200, "recommendation": "good_buy"},
        {"month": 2, "name": "February", "index": 0.88, "avg_price": 29800, "recommendation": "best_buy"},
        {"month": 9, "name": "September", "index": 1.31, "avg_price": 44400, "recommendation": "peak_price"}
    ],
    "summary": {
        "best_months": ["February", "January", "March"],
        "worst_months": ["September", "August", "October"],
        "price_range_pct": 48.5,
        "peak_season": "Late Summer (Aug-Oct)",
        "low_season": "Winter (Jan-Mar)"
    },
    "current_position": {
        "month": "January",
        "seasonal_index": 0.92,
        "vs_average": "8% below average",
        "recommendation": "Good time to buy - prices typically low in winter"
    }
}
```

---

### Tool: `detect_anomalies`

Detect unusual price movements.

**Schema:**
```json
{
    "name": "detect_anomalies",
    "description": "Detect unusual price movements using statistical methods.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "species": {
                "type": "string",
                "description": "Fish species name. If omitted, checks all species."
            },
            "period": {
                "type": "string",
                "enum": ["7d", "30d", "90d", "1y"],
                "default": "30d"
            },
            "min_severity": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "default": "medium"
            }
        },
        "required": []
    }
}
```

**Example Response:**
```json
{
    "period": "2024-12-01 to 2024-12-31",
    "anomalies_found": 3,
    "anomalies": [
        {
            "date": "2024-12-15",
            "species": "방어",
            "species_en": "Yellowtail",
            "type": "spike",
            "severity": "high",
            "actual_price": 45000,
            "expected_price": 28000,
            "deviation_pct": 60.7,
            "z_score": 3.2,
            "possible_cause": "Seasonal peak demand (winter yellowtail)"
        }
    ],
    "market_stability": {
        "overall": "stable",
        "volatility_trend": "decreasing"
    }
}
```

---

### Tool: `get_market_insight`

Get AI-generated market insights and recommendations.

**Schema:**
```json
{
    "name": "get_market_insight",
    "description": "Get comprehensive market insights including trend, volatility, and recommendations.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "species": {
                "type": "string",
                "description": "Fish species name. If omitted, returns overall market insight."
            },
            "insight_type": {
                "type": "string",
                "enum": ["summary", "detailed", "recommendation"],
                "default": "summary"
            }
        },
        "required": []
    }
}
```

**Example Response:**
```json
{
    "species": "고등어",
    "species_en": "Mackerel",
    "insight_date": "2025-01-02",
    "trend": {
        "direction": "rising",
        "strength": 0.65,
        "duration_days": 12,
        "description": "Moderate upward trend over past 2 weeks"
    },
    "volatility": {
        "index": 0.35,
        "label": "medium",
        "description": "Moderate price fluctuations expected"
    },
    "seasonal_context": {
        "current_month": "January",
        "seasonal_index": 0.92,
        "position": "Low season - prices typically 8% below annual average"
    },
    "recommendation": {
        "action": "buy",
        "confidence": 0.78,
        "reasoning": [
            "Currently in seasonal low period (winter)",
            "Prices 8% below annual average",
            "Short-term trend is stable"
        ]
    },
    "summary_text": "고등어 (Mackerel) is in its seasonal low period with prices 8% below average. Good buying opportunity. Recommended action: BUY with 78% confidence."
}
```

---

### Tool: `get_volatility`

Get price volatility analysis.

**Schema:**
```json
{
    "name": "get_volatility",
    "description": "Analyze price volatility. Returns volatility indices and stability rankings.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "species": {
                "type": "string",
                "description": "Fish species. If omitted, returns volatility ranking for all."
            },
            "period": {
                "type": "string",
                "enum": ["30d", "90d", "1y", "all"],
                "default": "90d"
            }
        },
        "required": []
    }
}
```

**Example Response (single species):**
```json
{
    "species": "대게",
    "species_en": "Snow Crab",
    "period": "90d",
    "volatility": {
        "index": 0.82,
        "label": "high",
        "daily_avg_change_pct": 4.2,
        "max_daily_change_pct": 18.5
    },
    "stability_rank": 68,
    "total_species": 76,
    "comparison": "More volatile than 89% of tracked species",
    "risk_assessment": {
        "level": "high",
        "description": "Significant price swings common. Consider averaging purchases."
    }
}
```

**Example Response (all species):**
```json
{
    "period": "90d",
    "most_stable": [
        {"species": "고등어", "volatility_index": 0.22, "label": "low"},
        {"species": "갈치", "volatility_index": 0.28, "label": "low"}
    ],
    "most_volatile": [
        {"species": "대게", "volatility_index": 0.82, "label": "high"},
        {"species": "방어", "volatility_index": 0.78, "label": "high"}
    ],
    "market_average": 0.45
}
```

---

## Implementation

### Project Structure

```
src/
└── prediction/
    ├── __init__.py
    ├── pipeline.py          # Main prediction pipeline
    ├── models/
    │   ├── __init__.py
    │   ├── exponential.py   # Exponential smoothing (short-term)
    │   ├── arima.py         # ARIMA (medium-term)
    │   ├── prophet_model.py # Prophet (long-term/seasonal)
    │   └── anomaly.py       # Anomaly detection
    ├── features/
    │   ├── __init__.py
    │   ├── trend.py         # Trend analysis
    │   ├── volatility.py    # Volatility calculation
    │   └── seasonality.py   # Seasonal decomposition
    └── insights/
        ├── __init__.py
        └── generator.py     # Market insight generation
```

### Main Pipeline

```python
# src/prediction/pipeline.py

import asyncio
import gc
from datetime import datetime
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class PredictionPipeline:
    """Daily prediction pipeline for fish prices."""

    # Resource optimization for t4-micro
    MAX_SPECIES = 30          # Top 30 most-traded species
    MAX_HISTORY_DAYS = 365    # 1 year of data
    BATCH_SIZE = 5            # Species per batch

    def __init__(self, duckdb_path: str, parquet_dir: str):
        self.duckdb_path = duckdb_path
        self.parquet_dir = parquet_dir

    async def run(self):
        """Run full prediction pipeline."""
        logger.info("Starting prediction pipeline...")
        species_list = await self.get_top_species(self.MAX_SPECIES)

        for i in range(0, len(species_list), self.BATCH_SIZE):
            batch = species_list[i:i + self.BATCH_SIZE]

            for species_id, species_name in batch:
                try:
                    await self.process_species(species_id, species_name)
                except Exception as e:
                    logger.error(f"Failed {species_name}: {e}")

            # Force garbage collection between batches
            gc.collect()

        logger.info("Prediction pipeline complete")

    async def process_species(self, species_id: int, species_name: str):
        """Run all predictions for a single species."""
        logger.debug(f"Processing {species_name}...")

        prices = await self.get_price_history(species_id, self.MAX_HISTORY_DAYS)

        if len(prices) < 30:
            logger.debug(f"Insufficient data for {species_name}")
            return

        # Run models
        await self.predict_exponential_smoothing(species_id, prices)
        await self.predict_arima(species_id, prices)
        await self.predict_prophet(species_id, prices)
        await self.detect_anomalies(species_id, prices)
        volatility = await self.calculate_volatility(species_id, prices)
        trend = await self.calculate_trend(species_id, prices)
        await self.generate_insight(species_id, trend, volatility)

    async def predict_exponential_smoothing(self, species_id: int, prices: np.array):
        """Short-term forecast (1-7 days) using Holt-Winters."""
        try:
            model = ExponentialSmoothing(
                prices[-90:],
                trend='add',
                seasonal='add',
                seasonal_periods=7
            ).fit(optimized=True)

            forecast = model.forecast(7)
            residuals = prices[-90:] - model.fittedvalues
            std_resid = np.std(residuals)

            base_date = datetime.now()
            for i, pred in enumerate(forecast):
                horizon = i + 1
                ci_width = 1.28 * std_resid * np.sqrt(horizon)  # 80% CI

                await self.save_forecast(
                    species_id, base_date, horizon,
                    int(pred), int(pred - ci_width), int(pred + ci_width),
                    'exp_smoothing'
                )
        except Exception as e:
            logger.warning(f"Exp smoothing failed for {species_id}: {e}")

    async def predict_arima(self, species_id: int, prices: np.array):
        """Medium-term forecast (7-30 days) using ARIMA."""
        try:
            model = ARIMA(prices[-180:], order=(2, 1, 2)).fit()
            forecast = model.get_forecast(steps=30)
            pred_mean = forecast.predicted_mean
            pred_ci = forecast.conf_int(alpha=0.20)

            for horizon in [7, 14, 30]:
                await self.save_forecast(
                    species_id, datetime.now(), horizon,
                    int(pred_mean.iloc[horizon-1]),
                    int(pred_ci.iloc[horizon-1, 0]),
                    int(pred_ci.iloc[horizon-1, 1]),
                    'arima'
                )
        except Exception as e:
            logger.warning(f"ARIMA failed for {species_id}: {e}")

    async def predict_prophet(self, species_id: int, prices: np.array):
        """Long-term forecast using Prophet."""
        try:
            df = pd.DataFrame({
                'ds': pd.date_range(end=datetime.now(), periods=len(prices)),
                'y': prices
            })

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.80
            )
            model.fit(df)

            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            row = forecast.iloc[-1]
            await self.save_forecast(
                species_id, datetime.now(), 30,
                int(row['yhat']), int(row['yhat_lower']), int(row['yhat_upper']),
                'prophet'
            )

            # Extract and save seasonality
            await self.save_seasonality_from_prophet(species_id, model, prices)

            del model
            gc.collect()

        except Exception as e:
            logger.warning(f"Prophet failed for {species_id}: {e}")

    async def detect_anomalies(self, species_id: int, prices: np.array):
        """Detect anomalies using Z-score and Isolation Forest."""
        z_scores = stats.zscore(prices)

        # IQR bounds
        q1, q3 = np.percentile(prices, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Isolation Forest
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso_labels = iso.fit_predict(prices.reshape(-1, 1))

        # Check last 30 days
        for i in range(-30, 0):
            z = z_scores[i]
            price = prices[i]
            is_iqr_anomaly = price < lower_bound or price > upper_bound
            is_iso_anomaly = iso_labels[i] == -1

            if abs(z) > 2.0 or is_iqr_anomaly or is_iso_anomaly:
                severity = 'high' if abs(z) > 3 else 'medium' if abs(z) > 2.5 else 'low'
                anomaly_type = 'spike' if z > 0 else 'drop'

                await self.save_anomaly(
                    species_id, i, int(price), int(np.mean(prices)),
                    float(z), anomaly_type, severity
                )

    async def calculate_volatility(self, species_id: int, prices: np.array) -> dict:
        """Calculate volatility metrics."""
        returns = np.diff(prices) / prices[:-1]
        rolling_std = pd.Series(prices).rolling(30).std()

        max_std = rolling_std.max()
        current_std = rolling_std.iloc[-1]
        volatility_index = current_std / max_std if max_std > 0 else 0

        if volatility_index < 0.33:
            label = 'low'
        elif volatility_index < 0.66:
            label = 'medium'
        else:
            label = 'high'

        return {
            'index': float(volatility_index),
            'label': label,
            'daily_avg_change_pct': float(np.mean(np.abs(returns)) * 100),
            'std_deviation': float(current_std) if not np.isnan(current_std) else 0
        }

    async def calculate_trend(self, species_id: int, prices: np.array) -> dict:
        """Calculate trend using linear regression."""
        X = np.arange(len(prices[-30:])).reshape(-1, 1)
        y = prices[-30:]
        reg = LinearRegression().fit(X, y)

        slope = reg.coef_[0]
        std_y = np.std(y)
        trend_strength = min(abs(slope) / std_y, 1.0) if std_y > 0 else 0

        if slope > std_y * 0.1:
            direction = 'rising'
        elif slope < -std_y * 0.1:
            direction = 'falling'
        else:
            direction = 'stable'

        return {
            'direction': direction,
            'strength': float(trend_strength),
            'slope': float(slope)
        }

    async def generate_insight(self, species_id: int, trend: dict, volatility: dict):
        """Generate market insight and recommendation."""
        # Get seasonal context
        current_month = datetime.now().month
        seasonal = await self.get_seasonality(species_id, current_month)

        # Generate recommendation
        score = 0
        reasons = []

        if seasonal and seasonal.get('index', 1.0) < 0.9:
            score += 2
            reasons.append(f"Seasonal low period ({seasonal['index']:.0%} of average)")
        elif seasonal and seasonal.get('index', 1.0) > 1.1:
            score -= 2
            reasons.append(f"Seasonal high period ({seasonal['index']:.0%} of average)")

        if trend['direction'] == 'falling':
            score += 1
            reasons.append("Prices trending downward")
        elif trend['direction'] == 'rising':
            score -= 1
            reasons.append("Prices trending upward")

        if volatility['label'] == 'high':
            score -= 1
            reasons.append("High volatility - consider waiting")

        if score >= 2:
            action = 'buy'
            confidence = min(0.5 + score * 0.1, 0.9)
        elif score <= -2:
            action = 'wait'
            confidence = min(0.5 + abs(score) * 0.1, 0.9)
        else:
            action = 'hold'
            confidence = 0.5

        await self.save_insight(
            species_id, trend, volatility, seasonal,
            action, confidence, reasons
        )
```

---

## Resource Optimization (t4-micro)

```python
OPTIMIZATION_CONFIG = {
    # Only predict for top N most-traded species
    "max_species_to_predict": 30,

    # Limit historical data loaded into memory
    "max_history_days": 365,

    # Run predictions in batches
    "batch_size": 5,

    # Prophet settings for lower memory
    "prophet_config": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,  # Disable - saves memory
        "mcmc_samples": 0,           # Disable MCMC - use MAP
    },

    # Clear model from memory after each species
    "clear_memory_after_each": True,
}
```

### Memory Usage Estimates

| Component | Peak Memory |
|-----------|-------------|
| Price data (1 year × 30 species) | ~5 MB |
| Exponential Smoothing model | ~10 MB |
| ARIMA model | ~20 MB |
| Prophet model | ~100 MB |
| Isolation Forest | ~15 MB |
| **Peak (single species)** | **~150 MB** |

With batch processing and garbage collection, peak memory stays under 200 MB.

---

## Daily Integration

Update `scripts/daily_update.py`:

```python
#!/usr/bin/env python3
"""Daily update: crawl + predictions."""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from crawler.fetcher import Fetcher
from crawler.writer import ParquetWriter
from prediction.pipeline import PredictionPipeline

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DUCKDB_PATH = DATA_DIR / "fish_market.duckdb"
PARQUET_DIR = DATA_DIR / "parquet" / "prices"

async def daily_update():
    """Run daily crawl and predictions."""
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime("%Y.%m.%d")

    # Step 1: Crawl yesterday's data
    logger.info(f"Crawling {date_str}...")
    writer = ParquetWriter(output_dir=str(PARQUET_DIR))

    async with Fetcher() as fetcher:
        records = await crawl_date(fetcher, date_str)
        if records:
            await writer.write(records, yesterday)
            logger.info(f"Saved {len(records)} records to Parquet")
        else:
            logger.info(f"No data for {date_str} (non-trading day)")

    # Step 2: Run prediction pipeline
    logger.info("Running predictions...")
    pipeline = PredictionPipeline(str(DUCKDB_PATH), str(PARQUET_DIR))
    await pipeline.run()
    logger.info("Predictions complete")

if __name__ == "__main__":
    asyncio.run(daily_update())
```

---

## Dependencies

Add to `requirements.txt`:

```txt
# Statistical/ML Libraries
statsmodels>=0.14.0
prophet>=1.1.0
scikit-learn>=1.3.0
scipy>=1.11.0
pandas>=2.0.0
numpy>=1.24.0
```

---

## Validation & Testing

### Backtesting

```python
async def backtest_model(species_id: int, model_type: str, horizon: int):
    """Backtest a model on historical data."""
    # Get historical prices
    prices = await get_full_history(species_id)

    errors = []
    for i in range(365, len(prices) - horizon):
        # Train on data up to day i
        train_data = prices[:i]

        # Predict
        prediction = await predict(train_data, model_type, horizon)

        # Compare to actual
        actual = prices[i + horizon - 1]
        error = abs(prediction - actual) / actual
        errors.append(error)

    mape = np.mean(errors) * 100
    return {
        'model': model_type,
        'horizon': horizon,
        'mape': mape,
        'sample_size': len(errors)
    }
```

### Expected Accuracy

| Model | Horizon | Expected MAPE |
|-------|---------|---------------|
| Exponential Smoothing | 1 day | 3-5% |
| Exponential Smoothing | 7 days | 6-10% |
| ARIMA | 14 days | 8-12% |
| ARIMA | 30 days | 10-15% |
| Prophet | 30 days | 10-18% |

---

## Checklist

- [ ] Install ML dependencies
- [ ] Create prediction module structure
- [ ] Implement Exponential Smoothing predictor
- [ ] Implement ARIMA predictor
- [ ] Implement Prophet predictor (with memory optimization)
- [ ] Implement anomaly detection (Z-score + IQR + Isolation Forest)
- [ ] Implement volatility calculator
- [ ] Implement trend analyzer
- [ ] Implement insight generator
- [ ] Create 5 prediction MCP tools
- [ ] Integrate with daily update script
- [ ] Backtest on historical data
- [ ] Validate prediction accuracy
- [ ] Memory profiling on t4-micro equivalent
