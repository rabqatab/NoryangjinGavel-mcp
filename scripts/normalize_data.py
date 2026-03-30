"""
Batch normalize existing Parquet files.

Applies all preprocessing rules from docs/08_data_preprocessing.md:
  1. Species name aliases (variant → canonical)
  2. Spec zero-padding removal (08미 → 8미)
  3. Empty string → null for state, origin, spec, packaging
  4. price_avg fix (0 → recalculated when high/low are positive)

State codes (Fix 2) don't need column changes — they already store the
Korean value correctly. The only change is that STATE_CODES now maps them
for English translation, which is a code-only fix.

Usage:
    uv run python scripts/normalize_data.py              # dry-run (default)
    uv run python scripts/normalize_data.py --apply      # overwrite in place
"""
import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Allow importing from the crawler package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crawler.normalizer import Normalizer

PARQUET_ROOT = Path(__file__).resolve().parent.parent / "data" / "parquet" / "prices"

_ZERO_PADDED_SPEC = re.compile(r"^0(\d+미)$")


@dataclass
class FileStats:
    """Change counts for a single parquet file."""
    species: int = 0
    spec: int = 0
    empty_str: int = 0
    price_avg: int = 0

    @property
    def total(self) -> int:
        return self.species + self.spec + self.empty_str + self.price_avg

    def __iadd__(self, other: "FileStats") -> "FileStats":
        self.species += other.species
        self.spec += other.spec
        self.empty_str += other.empty_str
        self.price_avg += other.price_avg
        return self


def normalize_parquet(parquet_path: Path, apply: bool) -> FileStats:
    """
    Apply all normalization rules to a single Parquet file.

    Returns per-fix change counts.
    """
    pf = pq.ParquetFile(parquet_path)
    table = pf.read()
    stats = FileStats()
    columns_to_update: dict[str, pa.Array] = {}

    # --- Fix 1: Species aliases ---
    species_list = table.column("species").to_pylist()
    new_species = []
    for name in species_list:
        canonical = Normalizer.SPECIES_ALIASES.get(name, name)
        if canonical != name:
            stats.species += 1
        new_species.append(canonical)
    if stats.species:
        columns_to_update["species"] = pa.array(new_species, type=pa.string())

    # --- Fix 3: Spec zero-padding ---
    spec_list = table.column("spec").to_pylist()
    new_spec = []
    for val in spec_list:
        m = _ZERO_PADDED_SPEC.match(val) if val else None
        if m:
            stats.spec += 1
            new_spec.append(m.group(1))
        else:
            new_spec.append(val)
    if stats.spec:
        columns_to_update["spec"] = pa.array(new_spec, type=pa.string())

    # --- Fix 4: Empty string → null ---
    for col_name in ("state", "origin", "spec", "packaging"):
        col_list = columns_to_update.get(col_name)
        if col_list is not None:
            # Already have a pending update array — work on that
            vals = col_list.to_pylist()
        else:
            vals = table.column(col_name).to_pylist()

        new_vals = []
        col_changed = 0
        for v in vals:
            if v == "":
                new_vals.append(None)
                col_changed += 1
            else:
                new_vals.append(v)

        if col_changed:
            stats.empty_str += col_changed
            columns_to_update[col_name] = pa.array(new_vals, type=pa.string())

    # --- Fix 5: price_avg = 0 fix ---
    high_list = table.column("price_high").to_pylist()
    low_list = table.column("price_low").to_pylist()
    avg_list = table.column("price_avg").to_pylist()
    new_avg = []
    for h, lo, a in zip(high_list, low_list, avg_list):
        if a == 0 and h > 0 and lo > 0:
            stats.price_avg += 1
            new_avg.append((h + lo) // 2)
        else:
            new_avg.append(a)
    if stats.price_avg:
        columns_to_update["price_avg"] = pa.array(new_avg, type=pa.int64())

    # --- Write if anything changed ---
    if columns_to_update and apply:
        for col_name, new_array in columns_to_update.items():
            idx = table.schema.get_field_index(col_name)
            table = table.set_column(idx, col_name, new_array)
        pq.write_table(table, parquet_path, compression="snappy")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Batch normalize Parquet data (see docs/08_data_preprocessing.md)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually overwrite files (default is dry-run)",
    )
    args = parser.parse_args()

    print("Normalization rules:")
    print(f"  Fix 1: Species aliases ({len(Normalizer.SPECIES_ALIASES)} rules)")
    print(f"  Fix 3: Spec zero-padding (0N미 → N미)")
    print(f"  Fix 4: Empty string → null (state, origin, spec, packaging)")
    print(f"  Fix 5: price_avg=0 recalculation")
    print(f"  (Fix 2: State codes — code-only, no data change needed)")
    print()

    if not args.apply:
        print("DRY RUN — no files will be modified. Use --apply to write changes.\n")

    parquet_files = sorted(PARQUET_ROOT.rglob("*.parquet"))
    print(f"Scanning {len(parquet_files)} Parquet files under {PARQUET_ROOT}\n")

    totals = FileStats()
    files_affected = 0

    for path in parquet_files:
        stats = normalize_parquet(path, args.apply)
        if stats.total > 0:
            totals += stats
            files_affected += 1
            rel = path.relative_to(PARQUET_ROOT)
            parts = []
            if stats.species:
                parts.append(f"species={stats.species}")
            if stats.spec:
                parts.append(f"spec={stats.spec}")
            if stats.empty_str:
                parts.append(f"empty={stats.empty_str}")
            if stats.price_avg:
                parts.append(f"price_avg={stats.price_avg}")
            print(f"  {rel}: {', '.join(parts)}")

    print(f"\n{'=' * 50}")
    print(f"Files affected: {files_affected}")
    print(f"  Fix 1 (species):   {totals.species:>8,} rows")
    print(f"  Fix 3 (spec):      {totals.spec:>8,} rows")
    print(f"  Fix 4 (empty str): {totals.empty_str:>8,} rows")
    print(f"  Fix 5 (price_avg): {totals.price_avg:>8,} rows")
    print(f"  Total:             {totals.total:>8,} rows")

    if args.apply:
        print("\nAll changes written.")
    else:
        print("\nDry-run complete. Use --apply to write changes.")


if __name__ == "__main__":
    main()
