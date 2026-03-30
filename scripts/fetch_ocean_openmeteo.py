"""
Fetch historical ocean/weather data from Open-Meteo (free, no API key).

Downloads wave height, wind speed, temperature for coastal stations
near major Korean fishing ports. Saves to data/ocean/ocean_daily.csv.

APIs:
  - Marine: https://marine-api.open-meteo.com/v1/marine (wave height, wave period)
  - Archive: https://archive-api.open-meteo.com/v1/archive (wind, temperature)

Usage:
    uv run python scripts/fetch_ocean_openmeteo.py
    uv run python scripts/fetch_ocean_openmeteo.py --start 2020.01.01 --end 2025.12.31
"""
import argparse
import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "ocean"

# Coastal locations near major fishing ports supplying Noryangjin
LOCATIONS = {
    "jeju": {"lat": 33.25, "lon": 126.56, "name": "제주"},       # 방어, 참돔, 전복
    "yeosu": {"lat": 34.74, "lon": 127.74, "name": "여수"},      # 삼치, 갈치, 남해
    "busan": {"lat": 35.10, "lon": 129.07, "name": "부산"},      # 넙치, 고등어
    "incheon": {"lat": 37.45, "lon": 126.59, "name": "인천"},    # 서해안 어종
    "sokcho": {"lat": 38.21, "lon": 128.59, "name": "속초"},     # 오징어, 대게
}

# Open-Meteo allows max ~1 year per request for archive, unlimited for marine
MAX_DAYS_PER_REQUEST = 365


def fetch_marine(lat, lon, start, end):
    """Fetch wave data from Open-Meteo Marine API."""
    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "wave_height_max,wave_period_max",
        "start_date": start, "end_date": end,
        "timezone": "Asia/Seoul",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
        return data.get("daily", {})
    except Exception as e:
        print(f"    Marine API error: {e}")
        return {}


def fetch_weather(lat, lon, start, end):
    """Fetch wind/temperature from Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
                 "wind_speed_10m_max,wind_gusts_10m_max,"
                 "precipitation_sum",
        "start_date": start, "end_date": end,
        "timezone": "Asia/Seoul",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
        return data.get("daily", {})
    except Exception as e:
        print(f"    Weather API error: {e}")
        return {}


def fetch_all(start_date, end_date):
    """Fetch all locations and save to CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "ocean_daily.csv"

    all_rows = []

    for loc_id, loc in LOCATIONS.items():
        print(f"  Fetching {loc['name']} ({loc_id})...")

        # Split into yearly chunks
        cur_start = start_date
        while cur_start < end_date:
            cur_end = min(cur_start + timedelta(days=MAX_DAYS_PER_REQUEST - 1), end_date)
            s = cur_start.strftime("%Y-%m-%d")
            e = cur_end.strftime("%Y-%m-%d")

            marine = fetch_marine(loc["lat"], loc["lon"], s, e)
            time.sleep(0.5)
            weather = fetch_weather(loc["lat"], loc["lon"], s, e)
            time.sleep(0.5)

            # Merge by date
            dates = marine.get("time", weather.get("time", []))
            for i, d in enumerate(dates):
                row = {
                    "date": d.replace("-", "."),
                    "location": loc_id,
                    "location_name": loc["name"],
                    "wave_height_max": marine.get("wave_height_max", [None] * len(dates))[i],
                    "wave_period_max": marine.get("wave_period_max", [None] * len(dates))[i],
                    "temp_mean": weather.get("temperature_2m_mean", [None] * len(dates))[i],
                    "temp_max": weather.get("temperature_2m_max", [None] * len(dates))[i],
                    "temp_min": weather.get("temperature_2m_min", [None] * len(dates))[i],
                    "wind_speed_max": weather.get("wind_speed_10m_max", [None] * len(dates))[i],
                    "wind_gust_max": weather.get("wind_gusts_10m_max", [None] * len(dates))[i],
                    "precipitation": weather.get("precipitation_sum", [None] * len(dates))[i],
                }
                all_rows.append(row)

            print(f"    {s} ~ {e}: {len(dates)} days")
            cur_start = cur_end + timedelta(days=1)

    # Write CSV
    fieldnames = ["date", "location", "location_name",
                  "wave_height_max", "wave_period_max",
                  "temp_mean", "temp_max", "temp_min",
                  "wind_speed_max", "wind_gust_max", "precipitation"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows):,} rows to {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Fetch ocean/weather data from Open-Meteo")
    parser.add_argument("--start", default="2020.01.01", help="Start date (YYYY.MM.DD)")
    parser.add_argument("--end", default="2026.03.26", help="End date (YYYY.MM.DD)")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y.%m.%d")
    end = datetime.strptime(args.end, "%Y.%m.%d")

    print(f"Open-Meteo Ocean Data Fetch")
    print(f"  Date range: {args.start} ~ {args.end} ({(end-start).days} days)")
    print(f"  Locations: {', '.join(v['name'] for v in LOCATIONS.values())}")
    print(f"  Features: wave_height, wave_period, temperature, wind_speed, wind_gust, precipitation")
    print()

    fetch_all(start, end)


if __name__ == "__main__":
    main()
