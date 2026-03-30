"""
Writer Component for Noryangjin Crawler.

Handles:
- Data persistence (Parquet, JSON, CSV)
- Hive-style partitioning (year/month)
- Batch inserts per day
- Deduplication
"""
import csv
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from .models import PriceRecord

logger = logging.getLogger(__name__)


class BaseWriter(ABC):
    """Abstract base class for data writers."""

    @abstractmethod
    def write_records(self, records: List[PriceRecord], date: str) -> bool:
        """
        Write records for a single day.

        Args:
            records: List of PriceRecords to write
            date: Trade date in YYYY.MM.DD format

        Returns:
            True if write successful
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the writer and release resources."""
        pass


class JSONWriter(BaseWriter):
    """
    Writes records to JSON files.

    One file per day: data/YYYY/MM/YYYY-MM-DD.json
    """

    def __init__(self, output_dir: str = "data/raw"):
        """
        Initialize JSON writer.

        Args:
            output_dir: Base directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, date: str) -> Path:
        """Get output file path for a date."""
        # Convert YYYY.MM.DD to YYYY/MM/YYYY-MM-DD.json
        parts = date.split(".")
        year, month, day = parts[0], parts[1], parts[2]
        dir_path = self.output_dir / year / month
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{year}-{month}-{day}.json"

    def write_records(self, records: List[PriceRecord], date: str) -> bool:
        """Write records to JSON file."""
        try:
            file_path = self._get_file_path(date)

            data = {
                "date": date,
                "crawled_at": datetime.now().isoformat(),
                "record_count": len(records),
                "records": [r.to_dict() for r in records],
            }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.debug(f"Wrote {len(records)} records to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to write JSON for {date}: {e}")
            return False

    def close(self) -> None:
        """Nothing to close for JSON writer."""
        pass


class ParquetWriter(BaseWriter):
    """
    Writes records to Parquet files with Hive-style partitioning.

    Output structure: data/parquet/prices/year=YYYY/month=MM/YYYY-MM-DD.parquet

    Features:
    - Hive-style partitioning by year/month
    - One file per day (no append issues)
    - Snappy compression (50-70% smaller than CSV)
    - Query-ready for DuckDB/Pandas
    """

    # Schema for Parquet file
    SCHEMA = pa.schema([
        ("trade_date", pa.string()),
        ("species_raw", pa.string()),
        ("species", pa.string()),
        ("state", pa.string()),
        ("origin", pa.string()),
        ("spec", pa.string()),
        ("packaging", pa.string()),
        ("quantity", pa.float64()),
        ("price_high", pa.int64()),
        ("price_low", pa.int64()),
        ("price_avg", pa.int64()),
    ]) if PYARROW_AVAILABLE else None

    def __init__(
        self,
        output_dir: str = "data/parquet/prices",
        compression: str = "snappy",
    ):
        """
        Initialize Parquet writer.

        Args:
            output_dir: Base directory for output files
            compression: Compression codec (snappy, gzip, zstd, none)
        """
        if not PYARROW_AVAILABLE:
            raise ImportError(
                "PyArrow is required for ParquetWriter. "
                "Install with: uv add pyarrow"
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression

    def _get_file_path(self, date: str) -> Path:
        """Get output file path for a date (one file per day)."""
        # date format: YYYY.MM.DD -> year=YYYY/month=MM/YYYY-MM-DD.parquet
        parts = date.split(".")
        year, month, day = parts[0], parts[1], parts[2]
        partition_path = self.output_dir / f"year={year}" / f"month={month}"
        partition_path.mkdir(parents=True, exist_ok=True)
        return partition_path / f"{year}-{month}-{day}.parquet"

    def _records_to_table(self, records: List[PriceRecord]) -> "pa.Table":
        """Convert PriceRecords to PyArrow Table."""
        data = {
            "trade_date": [r.trade_date for r in records],
            "species_raw": [r.species_raw for r in records],
            "species": [r.species for r in records],
            "state": [r.state or "" for r in records],
            "origin": [r.origin for r in records],
            "spec": [r.spec for r in records],
            "packaging": [r.packaging for r in records],
            "quantity": [r.quantity for r in records],
            "price_high": [r.price_high for r in records],
            "price_low": [r.price_low for r in records],
            "price_avg": [r.price_avg for r in records],
        }
        return pa.Table.from_pydict(data, schema=self.SCHEMA)

    def write_records(self, records: List[PriceRecord], date: str) -> bool:
        """Write records to Parquet file (one file per day)."""
        if not records:
            return True

        try:
            file_path = self._get_file_path(date)
            table = self._records_to_table(records)

            pq.write_table(
                table,
                file_path,
                compression=self.compression,
            )

            logger.debug(f"Wrote {len(records)} records to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to write Parquet for {date}: {e}")
            return False

    def close(self) -> None:
        """Nothing to close for Parquet writer."""
        pass


class CSVWriter(BaseWriter):
    """
    Writes records to a single CSV file.

    Appends new records to existing file.
    """

    HEADERS = [
        "trade_date",
        "species_raw",
        "species",
        "state",
        "origin",
        "spec",
        "packaging",
        "quantity",
        "price_high",
        "price_low",
        "price_avg",
    ]

    def __init__(self, output_path: str = "data/fish_prices.csv"):
        """
        Initialize CSV writer.

        Args:
            output_path: Path to output CSV file
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize CSV file with headers if new."""
        if not self.output_path.exists():
            with open(self.output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)

    def write_records(self, records: List[PriceRecord], date: str) -> bool:
        """Append records to CSV file."""
        try:
            with open(self.output_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for r in records:
                    writer.writerow([
                        r.trade_date,
                        r.species_raw,
                        r.species,
                        r.state or "",
                        r.origin,
                        r.spec,
                        r.packaging,
                        r.quantity,
                        r.price_high,
                        r.price_low,
                        r.price_avg,
                    ])

            logger.debug(f"Appended {len(records)} records for {date}")
            return True

        except Exception as e:
            logger.error(f"Failed to write CSV for {date}: {e}")
            return False

    def close(self) -> None:
        """Nothing to close for CSV writer."""
        pass


class MemoryWriter(BaseWriter):
    """
    In-memory writer for testing.

    Stores all records in memory.
    """

    def __init__(self):
        """Initialize memory writer."""
        self.records: Dict[str, List[PriceRecord]] = {}
        self.total_records = 0

    def write_records(self, records: List[PriceRecord], date: str) -> bool:
        """Store records in memory."""
        self.records[date] = records
        self.total_records += len(records)
        return True

    def get_records(self, date: str) -> List[PriceRecord]:
        """Get records for a specific date."""
        return self.records.get(date, [])

    def get_all_records(self) -> List[PriceRecord]:
        """Get all stored records."""
        all_records = []
        for records in self.records.values():
            all_records.extend(records)
        return all_records

    def clear(self) -> None:
        """Clear all stored records."""
        self.records.clear()
        self.total_records = 0

    def close(self) -> None:
        """Nothing to close for memory writer."""
        pass


class CompositeWriter(BaseWriter):
    """
    Writes to multiple destinations.

    Useful for writing to both JSON and CSV simultaneously.
    """

    def __init__(self, writers: List[BaseWriter]):
        """
        Initialize composite writer.

        Args:
            writers: List of writers to use
        """
        self.writers = writers

    def write_records(self, records: List[PriceRecord], date: str) -> bool:
        """Write records to all writers."""
        success = True
        for writer in self.writers:
            if not writer.write_records(records, date):
                success = False
        return success

    def close(self) -> None:
        """Close all writers."""
        for writer in self.writers:
            writer.close()
