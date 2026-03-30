"""Shared helpers for aggregation EDA notebook."""
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

DATA_ROOT = Path(__file__).parent.parent / "data" / "parquet" / "prices"

EDA_COLUMNS = [
    "trade_date", "species", "state", "origin", "spec",
    "packaging", "quantity", "price_high", "price_low", "price_avg",
]


def load_all_data() -> dict[str, list]:
    """Load all parquet data as a dict of column lists."""
    dataset = ds.dataset(str(DATA_ROOT), format="parquet", partitioning="hive")
    table = dataset.to_table(columns=EDA_COLUMNS)
    return {col: table.column(col).to_pylist() for col in EDA_COLUMNS}


def load_top_species(data: dict, n: int = 10) -> list[str]:
    """Return top N species by row count."""
    from collections import Counter
    counts = Counter(data["species"])
    return [name for name, _ in counts.most_common(n)]


def cv(values: list[float]) -> float:
    """Coefficient of variation. Returns 0 if mean is 0."""
    arr = np.array(values, dtype=float)
    mean = np.mean(arr)
    if mean == 0:
        return 0.0
    return float(np.std(arr) / mean)


def weighted_avg(prices: list[int], quantities: list[float]) -> float:
    """Quantity-weighted average price."""
    p = np.array(prices, dtype=float)
    q = np.array(quantities, dtype=float)
    total_q = q.sum()
    if total_q == 0:
        return float(np.mean(p))
    return float(np.dot(p, q) / total_q)


def pearson_corr(a: list[float], b: list[float]) -> float:
    """Pearson correlation coefficient."""
    arr_a = np.array(a)
    arr_b = np.array(b)
    if len(arr_a) < 3:
        return 0.0
    corr_matrix = np.corrcoef(arr_a, arr_b)
    return float(corr_matrix[0, 1])


def lag1_autocorr(values: list[float]) -> float:
    """Lag-1 autocorrelation of a time series."""
    arr = np.array(values, dtype=float)
    if len(arr) < 3:
        return 0.0
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


_SPEC_SIZE_GRADE = re.compile(r"^(특대|대|중|소)$")
_SPEC_COUNT = re.compile(r"^\d+미$")
_SPEC_WEIGHT_RANGE = re.compile(r"^\d+/\d+$")
_SPEC_COUNT_RANGE = re.compile(r"^\d+/\d+미$")


def classify_spec(spec: Optional[str]) -> str:
    """Classify a spec value into a category."""
    if spec is None:
        return "null"
    if _SPEC_SIZE_GRADE.match(spec):
        return "size_grade"
    if _SPEC_COUNT.match(spec):
        return "count"
    if _SPEC_COUNT_RANGE.match(spec):
        return "count_range"
    if _SPEC_WEIGHT_RANGE.match(spec):
        return "weight_range"
    return "other"
