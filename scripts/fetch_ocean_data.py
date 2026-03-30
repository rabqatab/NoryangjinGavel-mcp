"""
Fetch historical ocean observation data from KHOA (국립해양조사원) API.

Downloads daily water temperature and wind speed from tide observation stations
and saves to data/ocean/ as CSV files for integration with prediction models.

APIs used:
  - 조위관측소 실측 수온: /1192136/surveyWaterTemp/GetSurveyWaterTempApiService
  - 조위관측소 실측 풍향/풍속: /1192136/surveyWind/GetSurveyWindApiService

Usage:
    uv run python scripts/fetch_ocean_data.py                    # Fetch last 30 days
    uv run python scripts/fetch_ocean_data.py --start 2020.01.01 --end 2025.12.31
    uv run python scripts/fetch_ocean_data.py --days 365         # Fetch last year
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "ocean"
ENV_PATH = PROJECT_ROOT / ".env"

BASE_URL = "https://apis.data.go.kr/1192136"

# Stations near major fishing ports supplying Noryangjin
STATIONS = {
    "DT_0001": "인천",       # West coast — 감숭어, 꽃게
    "DT_0010": "제주",       # Jeju — 방어, 참돔, 전복
    "DT_0022": "여수",       # South coast — 삼치, 갈치
    "DT_0063": "부산",       # Southeast — 넙치, 고등어
    "DT_0049": "속초",       # East coast — 오징어, 대게
}


def load_api_key():
    if ENV_PATH.exists():
        with open(ENV_PATH) as f:
            for line in f:
                if line.startswith("mof_api_key="):
                    return line.strip().split("=", 1)[1]
    key = os.environ.get("MOF_API_KEY")
    if key:
        return key
    raise ValueError("API key not found. Set mof_api_key in .env or MOF_API_KEY env var.")


def fetch_water_temp(api_key, obs_code, date_str):
    """Fetch hourly water temperature for one station and one day."""
    url = f"{BASE_URL}/surveyWaterTemp/GetSurveyWaterTempApiService"
    params = {
        "serviceKey": api_key,
        "type": "json",
        "obsCode": obs_code,
        "reqDate": date_str,
        "min": 60,
        "numOfRows": 300,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if data.get("header", {}).get("resultCode") == "00":
            items = data.get("body", {}).get("items", {}).get("item", [])
            if isinstance(items, dict):
                items = [items]
            return items
    except Exception:
        pass
    return []


def fetch_wind(api_key, obs_code, date_str):
    """Fetch hourly wind direction/speed for one station and one day."""
    url = f"{BASE_URL}/surveyWind/GetSurveyWindApiService"
    params = {
        "serviceKey": api_key,
        "type": "json",
        "obsCode": obs_code,
        "reqDate": date_str,
        "min": 60,
        "numOfRows": 300,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if data.get("header", {}).get("resultCode") == "00":
            items = data.get("body", {}).get("items", {}).get("item", [])
            if isinstance(items, dict):
                items = [items]
            return items
    except Exception:
        pass
    return []


def aggregate_daily(hourly_items, value_key):
    """Aggregate hourly observations to daily mean/max."""
    values = [item[value_key] for item in hourly_items
              if item.get(value_key) is not None and item[value_key] != ""]
    if not values:
        return None, None
    values = [float(v) for v in values]
    return round(sum(values) / len(values), 2), round(max(values), 2)


def fetch_and_save(api_key, start_date, end_date):
    """Fetch all stations for a date range and save to CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV output
    csv_path = OUTPUT_DIR / "ocean_daily.csv"
    is_new = not csv_path.exists()

    # Load existing dates to avoid re-fetching
    existing = set()
    if csv_path.exists():
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add((row["date"], row["station_code"]))

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "date", "station_code", "station_name",
                "water_temp_avg", "water_temp_max",
                "wind_speed_avg", "wind_speed_max",
                "wind_dir_avg",
            ])

        current = start_date
        total_days = (end_date - start_date).days + 1
        fetched = 0

        while current <= end_date:
            date_str = current.strftime("%Y%m%d")
            date_dot = current.strftime("%Y.%m.%d")

            for code, name in STATIONS.items():
                if (date_dot, code) in existing:
                    continue

                # Fetch water temp
                temp_items = fetch_water_temp(api_key, code, date_str)
                temp_avg, temp_max = aggregate_daily(temp_items, "wtem")

                # Fetch wind
                wind_items = fetch_wind(api_key, code, date_str)
                wspd_avg, wspd_max = aggregate_daily(wind_items, "wspd")
                wdir_avg, _ = aggregate_daily(wind_items, "wndrct")

                writer.writerow([
                    date_dot, code, name,
                    temp_avg, temp_max,
                    wspd_avg, wspd_max,
                    wdir_avg,
                ])
                fetched += 1

                # Rate limit: 10,000 calls/day, ~2 calls per station per day
                time.sleep(0.2)

            if fetched % 50 == 0 and fetched > 0:
                days_done = (current - start_date).days + 1
                print(f"  {days_done}/{total_days} days, {fetched} records fetched...")
                f.flush()

            current += timedelta(days=1)

    print(f"\nDone: {fetched} new records saved to {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Fetch KHOA ocean observation data")
    parser.add_argument("--start", help="Start date (YYYY.MM.DD)")
    parser.add_argument("--end", help="End date (YYYY.MM.DD)")
    parser.add_argument("--days", type=int, default=30, help="Fetch last N days (default: 30)")
    args = parser.parse_args()

    api_key = load_api_key()
    print(f"API key loaded ({len(api_key)} chars)")
    print(f"Stations: {', '.join(f'{v}({k})' for k, v in STATIONS.items())}")

    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y.%m.%d")
        end = datetime.strptime(args.end, "%Y.%m.%d")
    else:
        end = datetime.now() - timedelta(days=1)
        start = end - timedelta(days=args.days - 1)

    print(f"Date range: {start.strftime('%Y.%m.%d')} ~ {end.strftime('%Y.%m.%d')}")
    print(f"Estimated API calls: {(end - start).days * len(STATIONS) * 2}")
    print()

    fetch_and_save(api_key, start, end)


if __name__ == "__main__":
    main()
