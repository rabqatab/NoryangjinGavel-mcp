"""
Normalizer Component for Noryangjin Crawler.

Handles:
- Data cleaning and transformation
- State code mapping
- Validation and integrity checks
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import PriceRecord


class Normalizer:
    """
    Data normalizer for fish price records.

    Cleans, validates, and transforms raw records for storage.
    """

    # State prefixes for fish condition
    STATE_CODES = {
        "선": "fresh",      # 선어 - Fresh fish
        "활": "live",       # 활어 - Live fish
        "냉": "frozen",     # 냉동 - Frozen
        "가공": "processed" # 가공 - Processed
    }

    def __init__(self, include_english_state: bool = False):
        """
        Initialize normalizer.

        Args:
            include_english_state: If True, include English state translation
        """
        self.include_english_state = include_english_state

    def get_state_english(self, state: Optional[str]) -> Optional[str]:
        """
        Get English translation for state code.

        Args:
            state: Korean state code (선, 활, 냉, 가공)

        Returns:
            English translation or None
        """
        if state is None:
            return None
        return self.STATE_CODES.get(state)

    def validate_record(self, record: PriceRecord) -> bool:
        """
        Validate a price record for data integrity.

        Args:
            record: PriceRecord to validate

        Returns:
            True if record is valid
        """
        # Must have species
        if not record.species:
            return False

        # Prices must be non-negative
        if record.price_avg < 0 or record.price_high < 0 or record.price_low < 0:
            return False

        # High price should be >= low price (allow equal for single bid)
        if record.price_high < record.price_low:
            return False

        # Average should be between low and high (or 0 if no data)
        if record.price_avg > 0:
            if not (record.price_low <= record.price_avg <= record.price_high):
                # Allow some tolerance for rounding
                if record.price_avg < record.price_low * 0.99:
                    return False
                if record.price_avg > record.price_high * 1.01:
                    return False

        # Quantity should be non-negative
        if record.quantity < 0:
            return False

        return True

    def normalize_record(self, record: PriceRecord) -> Dict[str, Any]:
        """
        Normalize a raw price record for database insertion.

        Args:
            record: PriceRecord to normalize

        Returns:
            Dictionary ready for database insertion
        """
        # Parse trade date to timestamp
        try:
            trade_dt = datetime.strptime(record.trade_date, "%Y.%m.%d")
            trade_timestamp = int(trade_dt.timestamp())
        except ValueError:
            trade_timestamp = None

        normalized = {
            "trade_date": record.trade_date,
            "trade_timestamp": trade_timestamp,
            "species_raw": record.species_raw,
            "species": record.species,
            "state": record.state,
            "origin": record.origin if record.origin else None,
            "spec": record.spec if record.spec else None,
            "packaging": record.packaging if record.packaging else None,
            "quantity": record.quantity,
            "price_high": record.price_high,
            "price_low": record.price_low,
            "price_avg": record.price_avg,
        }

        if self.include_english_state:
            normalized["state_en"] = self.get_state_english(record.state)

        return normalized

    def normalize_records(
        self,
        records: List[PriceRecord],
        validate: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Normalize a list of price records.

        Args:
            records: List of PriceRecords to normalize
            validate: If True, skip invalid records

        Returns:
            List of normalized dictionaries
        """
        normalized = []
        for record in records:
            if validate and not self.validate_record(record):
                continue
            normalized.append(self.normalize_record(record))
        return normalized

    def compute_daily_summary(self, records: List[PriceRecord]) -> Dict[str, Any]:
        """
        Compute summary statistics for a day's records.

        Args:
            records: List of PriceRecords for a single day

        Returns:
            Summary statistics dictionary
        """
        if not records:
            return {
                "total_records": 0,
                "unique_species": 0,
                "states": {},
                "avg_price": 0,
                "total_quantity": 0,
            }

        species_set = set(r.species for r in records)
        state_counts = {}
        for r in records:
            if r.state:
                state_counts[r.state] = state_counts.get(r.state, 0) + 1

        total_price = sum(r.price_avg for r in records)
        total_quantity = sum(r.quantity for r in records)

        return {
            "total_records": len(records),
            "unique_species": len(species_set),
            "states": state_counts,
            "avg_price": total_price / len(records) if records else 0,
            "total_quantity": total_quantity,
        }
