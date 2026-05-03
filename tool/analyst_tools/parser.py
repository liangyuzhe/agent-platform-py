"""SQL result parser -- converts raw query output into structured dicts."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_sql_result(result: str) -> dict[str, Any]:
    """Parse a raw SQL result string into a structured dictionary.

    The function tries two strategies in order:

    1. **JSON** -- if *result* is valid JSON (or contains a JSON array/object),
       it is parsed directly.
    2. **Pipe-delimited table** -- if the string looks like a ``| col1 | col2 |``
       formatted table, rows are extracted into ``columns`` and ``rows`` keys.

    Parameters
    ----------
    result:
        Raw string returned by an SQL executor.

    Returns
    -------
    dict
        Always contains a ``"format"`` key (``"json"`` | ``"table"`` | ``"raw"``).
        For JSON: the parsed object is returned under ``"data"``.
        For table: ``"columns"`` (list[str]) and ``"rows"`` (list[list[str]]).
        Fallback: ``"raw"`` holds the original string.
    """
    stripped = result.strip()
    if not stripped:
        return {"format": "raw", "raw": result, "data": None}

    # --- Strategy 1: JSON ---------------------------------------------------
    try:
        data = json.loads(stripped)
        return {"format": "json", "data": data, "raw": result}
    except (json.JSONDecodeError, ValueError):
        pass

    # Some MCP servers wrap the JSON in markdown code fences.
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", stripped, re.DOTALL)
    if fence_match:
        try:
            data = json.loads(fence_match.group(1).strip())
            return {"format": "json", "data": data, "raw": result}
        except (json.JSONDecodeError, ValueError):
            pass

    # --- Strategy 2: pipe-delimited table -----------------------------------
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    pipe_lines = [ln for ln in lines if "|" in ln]

    if len(pipe_lines) >= 2:
        return _parse_pipe_table(pipe_lines, raw=result)

    # --- Fallback -----------------------------------------------------------
    return {"format": "raw", "raw": result, "data": None}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_pipe_table(lines: list[str], *, raw: str) -> dict[str, Any]:
    """Parse a pipe-delimited ASCII table."""
    rows: list[list[str]] = []
    for line in lines:
        # Skip separator lines like  +----+----+
        if re.match(r"^[+\-|]+$", line.replace(" ", "")):
            continue
        cells = [cell.strip() for cell in line.split("|")]
        # Drop empty leading/trailing cells from edge pipes.
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        rows.append(cells)

    if not rows:
        return {"format": "raw", "raw": raw, "data": None}

    columns = rows[0]
    data_rows = rows[1:]
    return {
        "format": "table",
        "columns": columns,
        "rows": data_rows,
        "raw": raw,
    }
