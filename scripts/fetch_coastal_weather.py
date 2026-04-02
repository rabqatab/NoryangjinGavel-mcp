"""
Fetch historical coastal weather data from Open-Meteo Archive API.

5 major fishing port locations × daily weather features from 2006 onwards.
No rate limits, no API key needed.

Features per location:
  - temperature_2m_max/min/mean
  - precipitation_sum
  - wind_speed_10m_max, wind_gusts_10m_max
  - pressure_msl_mean
  - sunshine_duration (hours)

Output: data/weather/coastal_weather_daily.csv

Usage:
    uv run python scripts/fetch_coastal_weather.py
    uv run python scripts/fetch_coastal_weather.py --start 2006-01-01 --end 2026-03-31
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "weather"

# 5 fishing ports supplying Noryangjin (coastal coordinates)
LOCATIONS = {
    "jeju": {"name": "제주", "lat": 33.51, "lon": 126.53},      # 방어, 참돔, 전복
    "yeosu": {"name": "여수", "lat": 34.74, "lon": 127.74},     # 삼치, 갈치, 도다리
    "busan": {"name": "부산", "lat": 35.18, "lon": 129.08},      # 넙치, 고등어, 우럭
    "incheon": {"name": "인천", "lat": 37.46, "lon": 126.59},    # 감숭어, 꽃게, 굴
    "sokcho": {"name": "속초", "lat": 38.21, "lon": 128.59},     # 오징어, 대게
}

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "pressure_msl_mean",
    "sunshine_duration",
]

CSV_HEADERS = ["date", "location", "location_name"] + DAILY_VARS


def fetch_location(loc_id: str, loc_info: dict, start_date: str, end_date: str) -> list[dict]:
    """Fetch daily weather for one location from Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": loc_info["lat"],
        "longitude": loc_info["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Seoul",
    }

    for attempt in range(5):
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 429:
            wait = 10 * (attempt + 1)
            print(f"rate limited, waiting {wait}s...", end=" ", flush=True)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    data = resp.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])

    rows = []
    for i, date in enumerate(dates):
        row = {
            "date": date.replace("-", "."),  # Match project date format YYYY.MM.DD
            "location": loc_id,
            "location_name": loc_info["name"],
        }
        for var in DAILY_VARS:
            val = daily.get(var, [None] * len(dates))[i]
            # Convert sunshine_duration from seconds to hours
            if var == "sunshine_duration" and val is not None:
                val = round(val / 3600, 2)
            row[var] = val
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Fetch coastal weather data from Open-Meteo Archive")
    parser.add_argument("--start", default="2006-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-03-31", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "coastal_weather_daily.csv"

    all_rows = []
    for loc_id, loc_info in LOCATIONS.items():
        print(f"Fetching {loc_info['name']} ({loc_id})...", end=" ", flush=True)
        rows = fetch_location(loc_id, loc_info, args.start, args.end)
        all_rows.extend(rows)
        print(f"{len(rows)} days")
        time.sleep(5)  # Respect rate limits

    # Sort by date, then location
    all_rows.sort(key=lambda r: (r["date"], r["location"]))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows to {csv_path}")
    print(f"Date range: {all_rows[0]['date']} ~ {all_rows[-1]['date']}")
    print(f"Locations: {len(LOCATIONS)}")

    # Summary stats
    for loc_id, loc_info in LOCATIONS.items():
        loc_rows = [r for r in all_rows if r["location"] == loc_id]
        null_count = sum(1 for r in loc_rows if r["temperature_2m_mean"] is None)
        print(f"  {loc_info['name']}: {len(loc_rows)} days, {null_count} missing temp")


if __name__ == "__main__":
    main()
