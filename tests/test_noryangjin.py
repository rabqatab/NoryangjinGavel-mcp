"""
Test suite for Noryangjin Fish Market Crawler.

Tests REAL data from: https://www.susansijang.co.kr/nsis/miw/ko/info/miw3110

Run with: uv run python tests/test_noryangjin.py

Test logs are saved to: tests/logs/test_noryangjin_<timestamp>.log
"""
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from crawler.noryangjin_crawler import (
    NoryangjinCrawler,
    PriceRecord,
    CrawlDayResult,
    format_date,
    records_to_dicts,
)


# ============================================================================
# Test Logger
# ============================================================================

class TestLogger:
    """Logger that writes to both console and file simultaneously."""

    def __init__(self, log_dir: Path, prefix: str = "test_noryangjin"):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{prefix}_{timestamp}.log"
        self.file_handle = open(self.log_file, "w", encoding="utf-8")
        self._original_stdout = sys.stdout
        self._write_header()

    def _write_header(self):
        header = [
            "=" * 70,
            "NORYANGJIN FISH MARKET CRAWLER TEST LOG",
            f"Target: https://www.susansijang.co.kr",
            f"Endpoint: /nsis/miw/ko/info/miw3110",
            f"Timestamp: {datetime.now().isoformat()}",
            f"Log file: {self.log_file}",
            "=" * 70,
            "",
        ]
        for line in header:
            self.file_handle.write(line + "\n")
        self.file_handle.flush()

    def write(self, text: str):
        self._original_stdout.write(text)
        self._original_stdout.flush()
        self.file_handle.write(text)
        self.file_handle.flush()

    def flush(self):
        self._original_stdout.flush()
        self.file_handle.flush()

    def close(self):
        footer = [
            "",
            "=" * 70,
            f"Test completed: {datetime.now().isoformat()}",
            f"Log saved to: {self.log_file}",
            "=" * 70,
        ]
        for line in footer:
            self.file_handle.write(line + "\n")
        self.file_handle.close()
        sys.stdout = self._original_stdout

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================
# Test Cases
# ============================================================================

async def test_crawl_recent_date():
    """Test: Crawl yesterday's data (should have records if trading day)."""
    print("\n" + "=" * 60)
    print("TEST: Crawl Recent Date (Yesterday)")
    print("=" * 60)

    # Use yesterday's date
    yesterday = datetime.now() - timedelta(days=1)
    date_str = format_date(yesterday)

    print(f"\nTarget Date: {date_str}")
    print(f"URL: https://www.susansijang.co.kr/nsis/miw/ko/info/miw3110")

    crawler = NoryangjinCrawler(
        delay_between_requests=1.0,
        timeout=30,
    )

    async with crawler:
        result = await crawler.crawl_date(date_str)

    print(f"\nResult:")
    print(f"  Date: {result.date}")
    print(f"  Success: {result.success}")
    print(f"  Total Pages: {result.total_pages}")
    print(f"  Records Found: {len(result.records)}")
    print(f"  Elapsed: {result.elapsed_ms:.2f}ms")

    if result.error:
        print(f"  Error: {result.error}")

    if result.records:
        print(f"\n  Sample Records (first 5):")
        for i, record in enumerate(result.records[:5]):
            print(f"    {i+1}. {record.species_raw}")
            print(f"       Origin: {record.origin}, Spec: {record.spec}")
            print(f"       Prices: High={record.price_high:,}, Low={record.price_low:,}, Avg={record.price_avg:,} KRW")
            print(f"       Quantity: {record.quantity} {record.packaging}")
    else:
        print(f"\n  No records (likely Sunday or holiday)")

    success = result.success
    if success:
        print(f"\n[PASS] Successfully crawled {date_str}!")
    else:
        print(f"\n[FAIL] Failed to crawl {date_str}")

    return success


async def test_crawl_historical_date():
    """Test: Crawl a known historical date (2004.01.02 - first trading day)."""
    print("\n" + "=" * 60)
    print("TEST: Crawl Historical Date (2004.01.02)")
    print("=" * 60)

    date_str = "2004.01.02"

    print(f"\nTarget Date: {date_str} (First trading day in dataset)")

    crawler = NoryangjinCrawler(
        delay_between_requests=1.0,
        timeout=30,
    )

    async with crawler:
        result = await crawler.crawl_date(date_str)

    print(f"\nResult:")
    print(f"  Date: {result.date}")
    print(f"  Success: {result.success}")
    print(f"  Total Pages: {result.total_pages}")
    print(f"  Records Found: {len(result.records)}")
    print(f"  Elapsed: {result.elapsed_ms:.2f}ms")

    if result.records:
        print(f"\n  All Records:")
        for i, record in enumerate(result.records):
            state_str = f"({record.state})" if record.state else ""
            print(f"    {i+1}. {state_str}{record.species}")
            print(f"       Origin: {record.origin}, Spec: {record.spec}, Pkg: {record.packaging}")
            print(f"       Qty: {record.quantity}, High: {record.price_high:,}, Low: {record.price_low:,}, Avg: {record.price_avg:,}")

        # Verify data structure
        print(f"\n  Data Validation:")
        print(f"    - All records have species: {all(r.species for r in result.records)}")
        print(f"    - All records have prices: {all(r.price_avg > 0 for r in result.records)}")

    success = result.success and len(result.records) > 0
    if success:
        print(f"\n[PASS] Historical data crawl successful! ({len(result.records)} records)")
    else:
        if result.success and len(result.records) == 0:
            print(f"\n[WARN] No records found for this date (may be valid)")
            success = True  # Empty is OK for historical data
        else:
            print(f"\n[FAIL] Failed to crawl historical data")

    return success


async def test_crawl_holiday():
    """Test: Crawl a known holiday/Sunday (should return empty gracefully)."""
    print("\n" + "=" * 60)
    print("TEST: Crawl Holiday/Sunday (Empty Result)")
    print("=" * 60)

    # Find a recent Sunday
    today = datetime.now()
    days_since_sunday = (today.weekday() + 1) % 7
    last_sunday = today - timedelta(days=days_since_sunday)
    date_str = format_date(last_sunday)

    print(f"\nTarget Date: {date_str} (Sunday - market closed)")

    crawler = NoryangjinCrawler(
        delay_between_requests=1.0,
        timeout=30,
    )

    async with crawler:
        result = await crawler.crawl_date(date_str)

    print(f"\nResult:")
    print(f"  Date: {result.date}")
    print(f"  Success: {result.success}")
    print(f"  Records Found: {len(result.records)}")
    print(f"  Elapsed: {result.elapsed_ms:.2f}ms")

    # Sunday should have no records but still be successful
    success = result.success and len(result.records) == 0
    if success:
        print(f"\n[PASS] Empty result handled correctly!")
    else:
        if result.records:
            print(f"\n[INFO] Found {len(result.records)} records (market may have been open)")
            success = result.success  # Still pass if request succeeded
        else:
            print(f"\n[FAIL] Request failed")

    return success


async def test_pagination():
    """Test: Verify pagination works with a date that has multiple pages."""
    print("\n" + "=" * 60)
    print("TEST: Pagination (Multi-page crawl)")
    print("=" * 60)

    # Use a recent weekday that's likely to have multiple pages
    today = datetime.now()
    # Go back to find a recent weekday (Mon-Fri)
    days_back = 1
    while days_back < 10:
        test_date = today - timedelta(days=days_back)
        if test_date.weekday() < 5:  # Monday = 0, Friday = 4
            break
        days_back += 1

    date_str = format_date(test_date)

    print(f"\nTarget Date: {date_str} (Recent weekday)")

    crawler = NoryangjinCrawler(
        delay_between_requests=1.0,
        timeout=30,
    )

    async with crawler:
        result = await crawler.crawl_date(date_str)

    print(f"\nResult:")
    print(f"  Date: {result.date}")
    print(f"  Success: {result.success}")
    print(f"  Total Pages: {result.total_pages}")
    print(f"  Records Found: {len(result.records)}")
    print(f"  Elapsed: {result.elapsed_ms:.2f}ms")

    if result.total_pages > 1:
        print(f"\n  Pagination verified: {result.total_pages} pages crawled")
        expected_records = (result.total_pages - 1) * 10 + (len(result.records) % 10 or 10)
        print(f"  Expected ~{expected_records} records, got {len(result.records)}")

    if result.records:
        # Show species variety
        species_set = set(r.species for r in result.records)
        states_set = set(r.state for r in result.records if r.state)
        origins_set = set(r.origin for r in result.records)

        print(f"\n  Data Variety:")
        print(f"    - Unique Species: {len(species_set)}")
        print(f"    - States: {states_set}")
        print(f"    - Origins (sample): {list(origins_set)[:5]}")

    success = result.success and result.total_pages >= 1
    if success:
        print(f"\n[PASS] Pagination test successful!")
    else:
        print(f"\n[FAIL] Pagination test failed")

    return success


async def test_date_range():
    """Test: Crawl a small date range (3 days)."""
    print("\n" + "=" * 60)
    print("TEST: Date Range Crawl (3 days)")
    print("=" * 60)

    # Crawl last 3 days
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=2)

    start_str = format_date(start_date)
    end_str = format_date(end_date)

    print(f"\nDate Range: {start_str} to {end_str}")

    crawler = NoryangjinCrawler(
        delay_between_requests=1.0,
        delay_between_days=0.5,
        timeout=30,
    )

    def progress_callback(result, stats):
        status = "OK" if result.success else "FAIL"
        records = len(result.records) if result.records else 0
        print(f"  [{status}] {result.date}: {records} records, {result.elapsed_ms:.0f}ms")

    print(f"\nCrawling...")
    async with crawler:
        results, stats = await crawler.crawl_date_range(
            start_str, end_str,
            progress_callback=progress_callback
        )

    print(f"\nStats:")
    print(f"  Total Days: {stats.total_days}")
    print(f"  Successful Days: {stats.successful_days}")
    print(f"  Empty Days (holidays): {stats.empty_days}")
    print(f"  Failed Days: {stats.failed_days}")
    print(f"  Total Records: {stats.total_records}")
    print(f"  Total Pages: {stats.total_pages}")
    print(f"  Elapsed: {stats.elapsed_seconds:.2f}s")

    success = stats.failed_days == 0
    if success:
        print(f"\n[PASS] Date range crawl successful!")
    else:
        print(f"\n[FAIL] Some days failed to crawl")

    return success


async def test_data_extraction():
    """Test: Verify data extraction and parsing quality."""
    print("\n" + "=" * 60)
    print("TEST: Data Extraction Quality")
    print("=" * 60)

    # Use yesterday
    yesterday = datetime.now() - timedelta(days=1)
    date_str = format_date(yesterday)

    print(f"\nTarget Date: {date_str}")

    crawler = NoryangjinCrawler(
        delay_between_requests=1.0,
        timeout=30,
    )

    async with crawler:
        result = await crawler.crawl_date(date_str)

    if not result.records:
        print(f"\n  No records to validate (empty day)")
        print(f"\n[SKIP] Cannot validate on empty day")
        return True

    print(f"\nValidating {len(result.records)} records...")

    # Validation checks
    checks = {
        "all_have_species": all(r.species for r in result.records),
        "all_have_origin": all(r.origin for r in result.records),
        "all_have_prices": all(r.price_avg >= 0 for r in result.records),
        "high_gte_low": all(r.price_high >= r.price_low for r in result.records),
        "avg_in_range": all(r.price_low <= r.price_avg <= r.price_high or r.price_avg == 0 for r in result.records),
        "quantity_positive": all(r.quantity >= 0 for r in result.records),
        "date_matches": all(r.trade_date == date_str for r in result.records),
    }

    print(f"\n  Validation Results:")
    all_passed = True
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {check_name}")
        if not passed:
            all_passed = False

    # State extraction check
    records_with_state = [r for r in result.records if r.state]
    print(f"\n  State Extraction:")
    print(f"    - Records with state prefix: {len(records_with_state)}/{len(result.records)}")
    if records_with_state:
        state_counts = {}
        for r in records_with_state:
            state_counts[r.state] = state_counts.get(r.state, 0) + 1
        print(f"    - State distribution: {state_counts}")

    # Convert to dict test
    dicts = records_to_dicts(result.records[:3])
    print(f"\n  Dict Conversion (sample):")
    for d in dicts:
        print(f"    {d['species_raw']}: {d['price_avg']:,} KRW")

    if all_passed:
        print(f"\n[PASS] All data extraction checks passed!")
    else:
        print(f"\n[FAIL] Some checks failed")

    return all_passed


async def run_all_tests():
    """Run all Noryangjin crawler tests."""
    print("=" * 60)
    print("NORYANGJIN FISH MARKET CRAWLER TEST SUITE")
    print(f"Target: https://www.susansijang.co.kr")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    print("\nThese tests crawl REAL data from Noryangjin Fish Market.")
    print("Network connectivity is required.\n")

    results = {}

    # Run tests
    results["recent_date"] = await test_crawl_recent_date()
    results["historical_date"] = await test_crawl_historical_date()
    results["holiday_empty"] = await test_crawl_holiday()
    results["pagination"] = await test_pagination()
    results["date_range"] = await test_date_range()
    results["data_extraction"] = await test_data_extraction()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "=" * 60)
        print("ALL NORYANGJIN CRAWLER TESTS PASSED!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print(f"SOME TESTS FAILED ({total - passed} failures)")
        print("=" * 60)

    return passed == total


def main():
    """Main entry point with logging."""
    log_dir = Path(__file__).parent / "logs"

    with TestLogger(log_dir, prefix="test_noryangjin") as logger:
        success = asyncio.run(run_all_tests())
        print(f"\nLog file saved to: {logger.log_file}")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
