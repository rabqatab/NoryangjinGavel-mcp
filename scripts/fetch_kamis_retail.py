"""
Fetch KAMIS seafood retail + wholesale prices (daily).

Appends to data/kamis/kamis_seafood_daily.csv. Incremental — skips already-fetched dates.
No API key required (works with dummy credentials).

Available items (KAMIS API limitation — only 6 seafood items):
  - 마른멸치 (638/00), 마른미역 (642/00), 굴 (644/00)
  - 홍합_안깐 (658/02), 가리비 (659/01), 건다시마 (660/01)

Usage:
    uv run python scripts/fetch_kamis_retail.py              # Fetch latest
    uv run python scripts/fetch_kamis_retail.py --backfill   # Fetch last 365 days
"""

import argparse
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "kamis"
CSV_PATH = OUT_DIR / "kamis_seafood_daily.csv"

KAMIS_URL = "https://www.kamis.or.kr/service/price/xml.do"
HEADERS = {"User-Agent": "Mozilla/5.0"}

ITEMS = [
    {"code": 638, "kc": "00", "name": "마른멸치"},
    {"code": 642, "kc": "00", "name": "마른미역"},
    {"code": 644, "kc": "00", "name": "굴"},
    {"code": 658, "kc": "02", "name": "홍합(안깐)"},
    {"code": 659, "kc": "01", "name": "가리비"},
    {"code": 660, "kc": "01", "name": "건다시마"},
]

CSV_FIELDS = ["date", "item_name", "item_code", "kind_code", "type", "price"]


def load_existing_dates() -> set:
    if not CSV_PATH.exists():
        return set()
    existing = set()
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            existing.add((row["date"], row["item_name"], row["type"]))
    return existing


def fetch_period(start: str, end: str) -> list[dict]:
    """Fetch all items for a date range."""
    existing = load_existing_dates()
    rows = []

    for item in ITEMS:
        for cls_code, label in [("01", "retail"), ("02", "wholesale")]:
            params = {
                "action": "periodProductList",
                "p_startday": start,
                "p_endday": end,
                "p_itemcategorycode": "600",
                "p_itemcode": str(item["code"]),
                "p_kindcode": item["kc"],
                "p_productrankcode": "04",
                "p_convert_kg_yn": "Y",
                "p_product_cls_code": cls_code,
                "p_cert_key": "111",
                "p_cert_id": "222",
                "p_returntype": "json",
            }
            try:
                resp = requests.get(KAMIS_URL, params=params, headers=HEADERS, timeout=15)
                d = resp.json()
                for entry in d.get("data", {}).get("item", []):
                    if entry.get("countyname") != "평균":
                        continue
                    price = entry.get("price", "-")
                    if price == "-" or not price:
                        continue
                    regday = entry.get("regday", "")
                    yyyy = entry.get("yyyy", "")
                    try:
                        dt = datetime.strptime(f"{yyyy}/{regday}", "%Y/%m/%d")
                        date_fmt = dt.strftime("%Y.%m.%d")
                    except:
                        continue

                    key = (date_fmt, item["name"], label)
                    if key in existing:
                        continue

                    rows.append({
                        "date": date_fmt,
                        "item_name": item["name"],
                        "item_code": item["code"],
                        "kind_code": item["kc"],
                        "type": label,
                        "price": price,
                    })
                    existing.add(key)
            except Exception:
                pass
            time.sleep(0.15)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true", help="Fetch last 365 days")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.backfill:
        end = datetime.now()
        start = end - timedelta(days=365)
    else:
        end = datetime.now()
        start = end - timedelta(days=7)

    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    print(f"Fetching KAMIS seafood prices: {start_str} ~ {end_str}")
    rows = fetch_period(start_str, end_str)

    if not rows:
        print("No new data.")
        return

    is_new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["item_name"], r["type"], r["date"])))

    print(f"Added {len(rows)} new records → {CSV_PATH}")


if __name__ == "__main__":
    main()
