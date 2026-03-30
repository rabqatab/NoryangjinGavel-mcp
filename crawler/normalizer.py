"""
Normalizer Component for Noryangjin Crawler.

Handles:
- Data cleaning and transformation
- State code mapping
- Species name unification
- Spec zero-padding normalization
- Empty string coercion
- Price sanity fixes
- Validation and integrity checks

See docs/08_data_preprocessing.md for full rationale and affected row counts.
"""
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .models import PriceRecord

logger = logging.getLogger(__name__)

_INVENTORY_PATH = Path(__file__).parent / "species_inventory.json"


# Precompiled pattern for zero-padded numeric specs like "08미"
_ZERO_PADDED_SPEC = re.compile(r"^0(\d+미)$")


class Normalizer:
    """
    Data normalizer for fish price records.

    Cleans, validates, and transforms raw records for storage.
    """

    # State prefixes for fish condition
    STATE_CODES = {
        "선": "fresh",           # 선어 - Fresh fish
        "활": "live",            # 활어 - Live fish
        "냉": "frozen",          # 냉동 - Frozen
        "가공": "processed",     # 가공 - Processed
        "냉건": "frozen_dried",  # 냉동건조 - Frozen then dried
        "건": "dried",           # 건조 - Dried
    }

    # Species name aliases: variant → canonical
    # Unifies typos, variant spellings, and truncated names found in source data.
    SPECIES_ALIASES = {
        "고둥 갯고동": "고등 갯고동",    # 고둥→고등 typo (standard fisheries term)
        "망둑어": "망둥어",              # 표준국어대사전: 망둥어
        "쭈구미": "쭈꾸미",             # 국립국어원 standard spelling
        "학공치": "학꽁치",              # standard spelling
        "깐우렁": "깐우렁이",           # truncation → full form
        "갑오징어기타": "갑오징어 기타",  # missing space (consistent with other 기타 entries)
    }

    def __init__(self, include_english_state: bool = False):
        """
        Initialize normalizer.

        Args:
            include_english_state: If True, include English state translation
        """
        self.include_english_state = include_english_state
        self._known_species: Optional[Set[str]] = None
        self._new_species: Set[str] = set()

    # ------------------------------------------------------------------
    # Species inventory
    # ------------------------------------------------------------------

    @staticmethod
    def load_inventory(path: Path = _INVENTORY_PATH) -> Set[str]:
        """Load the canonical species set from the inventory JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return set(data["species"].keys())

    @staticmethod
    def save_inventory(
        species_counts: Dict[str, int],
        path: Path = _INVENTORY_PATH,
    ) -> None:
        """Write an updated species inventory JSON."""
        inventory = {
            "_meta": {
                "description": "Canonical species inventory for Noryangjin Fish Market data",
                "generated_from": "data/parquet/prices/**/*.parquet",
                "total_species": len(species_counts),
                "total_records": sum(species_counts.values()),
            },
            "species": dict(sorted(species_counts.items())),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(inventory, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def ensure_inventory(self) -> Set[str]:
        """Load the inventory lazily on first access."""
        if self._known_species is None:
            try:
                self._known_species = self.load_inventory()
            except FileNotFoundError:
                logger.warning(
                    "Species inventory not found at %s — "
                    "new species detection disabled. "
                    "Run: uv run python scripts/update_species_inventory.py",
                    _INVENTORY_PATH,
                )
                self._known_species = set()
        return self._known_species

    def check_new_species(self, species: str) -> bool:
        """
        Check if a species name is new (not in the inventory).

        Logs a warning on first occurrence and collects new names
        in self._new_species for later retrieval.

        Returns True if the species is new.
        """
        known = self.ensure_inventory()
        if not known:
            return False
        if species not in known and species not in self._new_species:
            self._new_species.add(species)
            logger.warning("New species detected: '%s'", species)
            return True
        return species in self._new_species

    def get_new_species(self) -> Set[str]:
        """Return all new species detected during this session."""
        return set(self._new_species)

    # ------------------------------------------------------------------
    # Field-level normalization (classmethod so batch scripts can use them)
    # ------------------------------------------------------------------

    @classmethod
    def normalize_species(cls, name: str) -> str:
        """Normalize a species name using the alias mapping."""
        return cls.SPECIES_ALIASES.get(name, name)

    @staticmethod
    def normalize_spec(spec: str) -> str:
        """
        Normalize a spec value.

        Strips leading zeros from single-number specs (e.g. "08미" → "8미").
        Preserves range specs like "09/10" unchanged.
        """
        m = _ZERO_PADDED_SPEC.match(spec)
        if m:
            return m.group(1)
        return spec

    @staticmethod
    def empty_to_none(value: str) -> Optional[str]:
        """Coerce empty strings to None."""
        return value if value else None

    @staticmethod
    def fix_price_avg(price_high: int, price_low: int, price_avg: int) -> int:
        """
        Fix price_avg when it is 0 but high/low are both positive.

        Recalculates as integer mean of high and low.
        """
        if price_avg == 0 and price_high > 0 and price_low > 0:
            return (price_high + price_low) // 2
        return price_avg

    # ------------------------------------------------------------------
    # Record-level normalization
    # ------------------------------------------------------------------

    @classmethod
    def normalize_price_record(cls, record: PriceRecord) -> None:
        """
        Apply all normalization rules to a PriceRecord in place.

        This is the single entry point used by both the live crawl pipeline
        and the batch transformation script.

        Applies (in order):
          1. Species alias mapping
          2. Spec zero-padding removal
          3. Empty string → None for state, origin, spec, packaging
          4. price_avg fix when 0 with positive high/low
        """
        record.species = cls.normalize_species(record.species)
        record.spec = cls.normalize_spec(record.spec)
        record.state = cls.empty_to_none(record.state)
        record.origin = cls.empty_to_none(record.origin)
        record.spec = cls.empty_to_none(record.spec)
        record.packaging = cls.empty_to_none(record.packaging)
        record.price_avg = cls.fix_price_avg(
            record.price_high, record.price_low, record.price_avg,
        )

    def get_state_english(self, state: Optional[str]) -> Optional[str]:
        """
        Get English translation for state code.

        Args:
            state: Korean state code (선, 활, 냉, 가공, 냉건, 건)

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
            "species": self.normalize_species(record.species),
            "state": self.empty_to_none(record.state),
            "origin": self.empty_to_none(record.origin),
            "spec": self.empty_to_none(self.normalize_spec(record.spec or "")),
            "packaging": self.empty_to_none(record.packaging),
            "quantity": record.quantity,
            "price_high": record.price_high,
            "price_low": record.price_low,
            "price_avg": self.fix_price_avg(
                record.price_high, record.price_low, record.price_avg,
            ),
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
