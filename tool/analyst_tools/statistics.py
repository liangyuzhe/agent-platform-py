"""Descriptive statistics computation for SQL result data."""

from __future__ import annotations

import math
from typing import Any


def _to_float_list(values: list[Any]) -> list[float]:
    """Attempt to coerce every element of *values* to ``float``.

    Non-numeric values are silently skipped.
    """
    out: list[float] = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _percentile(sorted_data: list[float], p: float) -> float:
    """Return the *p*-th percentile (0-100) of an already-sorted list."""
    if not sorted_data:
        raise ValueError("Cannot compute percentile of empty data")
    n = len(sorted_data)
    k = (p / 100) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def compute_statistics(data: dict[str, Any]) -> dict[str, Any]:
    """Compute descriptive statistics for every numeric column in *data*.

    Expected input formats
    ----------------------
    * ``{"columns": [...], "rows": [[...], ...]}``  (pipe-table style)
    * ``{"data": [[...], ...]}``  (JSON list-of-lists, first row = header)
    * ``{"data": [{"col": val, ...}, ...]}``  (JSON list-of-dicts)

    Returns
    -------
    dict
        ``{"<column_name>": {"mean": ..., "median": ..., ...}, ...}``
        Only columns whose values are predominantly numeric are included.
    """
    columns: list[str] = []
    rows: list[list[Any]] = []

    if "columns" in data and "rows" in data:
        columns = data["columns"]
        rows = data["rows"]
    elif "data" in data:
        payload = data["data"]
        if not payload:
            return {}
        if isinstance(payload[0], dict):
            columns = list(payload[0].keys())
            rows = [[row.get(c) for c in columns] for row in payload]
        elif isinstance(payload[0], list):
            columns = [str(i) for i in range(len(payload[0]))] if len(payload) > 0 else []
            rows = payload
        else:
            return {}

    if not columns or not rows:
        return {}

    result: dict[str, Any] = {}
    for col_idx, col_name in enumerate(columns):
        col_values = [row[col_idx] for row in rows if col_idx < len(row)]
        nums = _to_float_list(col_values)
        if len(nums) < 2:
            continue  # not enough data to compute meaningful stats

        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        mean = sum(nums_sorted) / n
        median = _percentile(nums_sorted, 50)
        variance = sum((x - mean) ** 2 for x in nums_sorted) / n
        stddev = math.sqrt(variance)

        result[col_name] = {
            "count": n,
            "mean": round(mean, 6),
            "median": round(median, 6),
            "stddev": round(stddev, 6),
            "min": nums_sorted[0],
            "max": nums_sorted[-1],
            "q1": round(_percentile(nums_sorted, 25), 6),
            "q3": round(_percentile(nums_sorted, 75), 6),
        }

    return result
