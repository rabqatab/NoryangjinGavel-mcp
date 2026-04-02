"""
Fetch today's KHOA station data (water temperature + wind) for live predictions.

Appends to data/weather/khoa_daily.csv. Designed to run daily via cron.
Only fetches yesterday's data (24 hourly obs → 1 daily record per station).

Usage:
    uv run python scripts/fetch_khoa_daily.py              # Yesterday
    uv run python scripts/fetch_khoa_daily.py --date 2025.03.30  # Specific date
"""

import argparse
import csv
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "weather"
ENV_PATH = PROJECT_ROOT / ".env"

BASE_URL = "https://apis.data.go.kr/1192136"

STATIONS = {
    "DT_0001": "인천",
    "DT_0010": "제주",
    "DT_0022": "여수",
    "DT_0063": "부산",
}

CSV_HEADERS = [
    "date", "station_code", "station_name",
    "water_temp_avg", "water_temp_max",
    "wind_speed_avg", "wind_speed_max", "wind_dir_avg",
]


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


def fetch_endpoint(api_key, endpoint, obs_code, date_str, value_key):
    url = f"{BASE_URL}/{endpoint}"
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
            values = [float(item[value_key]) for item in items
                      if item.get(value_key) is not None and item[value_key] != ""]
            if values:
                return round(sum(values) / len(values), 2), round(max(values), 2)
    except Exception:
        pass
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Date to fetch (YYYY.MM.DD), default=yesterday")
    args = parser.parse_args()

    if args.date:
        target_dt = datetime.strptime(args.date, "%Y.%m.%d")
    else:
        target_dt = datetime.now() - timedelta(days=1)

    date_dot = target_dt.strftime("%Y.%m.%d")
    date_api = target_dt.strftime("%Y%m%d")

    api_key = load_api_key()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "khoa_daily.csv"

    # Load existing to avoid duplicates
    existing = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                existing.add((row["date"], row["station_code"]))

    is_new = not csv_path.exists()
    fetched = 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if is_new:
            writer.writeheader()

        for code, name in STATIONS.items():
            if (date_dot, code) in existing:
                print(f"  {name}: already fetched")
                continue

            temp_avg, temp_max = fetch_endpoint(
                api_key, "surveyWaterTemp/GetSurveyWaterTempApiService",
                code, date_api, "wtem")
            time.sleep(0.3)

            wspd_avg, wspd_max = fetch_endpoint(
                api_key, "surveyWind/GetSurveyWindApiService",
                code, date_api, "wspd")
            time.sleep(0.3)

            wdir_avg, _ = fetch_endpoint(
                api_key, "surveyWind/GetSurveyWindApiService",
                code, date_api, "wndrct")

            writer.writerow({
                "date": date_dot,
                "station_code": code,
                "station_name": name,
                "water_temp_avg": temp_avg,
                "water_temp_max": temp_max,
                "wind_speed_avg": wspd_avg,
                "wind_speed_max": wspd_max,
                "wind_dir_avg": wdir_avg,
            })
            fetched += 1
            print(f"  {name}: temp={temp_avg}, wind={wspd_avg}")
            time.sleep(0.3)

    print(f"\n{date_dot}: {fetched} stations fetched → {csv_path}")


if __name__ == "__main__":
    main()
