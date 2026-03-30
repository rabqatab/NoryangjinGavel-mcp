"""
Rebuild the canonical species inventory from current Parquet data.

Scans all Hive-partitioned Parquet files and writes the species name → count
mapping to crawler/species_inventory.json.

Run this after:
  - Initial historical crawl
  - Adding new species aliases to Normalizer.SPECIES_ALIASES
  - Periodically to pick up legitimately new species from live data

Usage:
    uv run python scripts/update_species_inventory.py
"""
import sys
from collections import Counter
from pathlib import Path

import pyarrow.dataset as ds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from crawler.normalizer import Normalizer

PARQUET_ROOT = PROJECT_ROOT / "data" / "parquet" / "prices"
INVENTORY_PATH = PROJECT_ROOT / "crawler" / "species_inventory.json"


def main():
    # Load current inventory for diff
    try:
        old_species = Normalizer.load_inventory(INVENTORY_PATH)
    except FileNotFoundError:
        old_species = set()

    # Scan all parquet files
    dataset = ds.dataset(str(PARQUET_ROOT), format="parquet", partitioning="hive")
    table = dataset.to_table(columns=["species"])
    counts = Counter(table.column("species").to_pylist())

    new_species = set(counts.keys()) - old_species
    removed_species = old_species - set(counts.keys())

    # Write updated inventory
    Normalizer.save_inventory(dict(counts), INVENTORY_PATH)

    print(f"Species inventory updated: {INVENTORY_PATH}")
    print(f"  Total species: {len(counts)}")
    print(f"  Total records: {sum(counts.values()):,}")

    if new_species:
        print(f"\n  New species added ({len(new_species)}):")
        for name in sorted(new_species):
            print(f"    + {name} ({counts[name]:,} records)")

    if removed_species:
        print(f"\n  Species removed ({len(removed_species)}):")
        for name in sorted(removed_species):
            print(f"    - {name}")

    if not new_species and not removed_species:
        print("\n  No changes from previous inventory.")


if __name__ == "__main__":
    main()
