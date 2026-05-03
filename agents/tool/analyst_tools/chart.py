"""Chart configuration generator that produces ECharts option dicts."""

from __future__ import annotations

from typing import Any


def recommend_chart_type(data: dict[str, Any]) -> str:
    """Recommend a chart type based on the shape of *data*.

    Parameters
    ----------
    data:
        A dict with ``"columns"`` and ``"rows"`` keys (table format) **or**
        ``"data"`` (list-of-dicts or list-of-lists).

    Returns
    -------
    str
        One of ``"table"``, ``"line"``, ``"bar"``, ``"pie"``.
    """
    columns, rows = _extract_columns_rows(data)
    if not columns or not rows:
        return "table"

    n_cols = len(columns)
    n_rows = len(rows)

    # Single categorical column + single value column => pie or bar
    if n_cols == 2 and n_rows <= 10:
        return "pie"

    if n_cols == 2 and n_rows <= 30:
        return "bar"

    # Many rows with >=2 columns => line chart
    if n_rows > 5 and n_cols >= 2:
        return "line"

    # Default
    if n_rows > 20:
        return "table"

    return "bar"


def generate_chart_config(data: dict[str, Any]) -> dict[str, Any]:
    """Generate an ECharts-compatible configuration dict from *data*.

    Parameters
    ----------
    data:
        Same structure as accepted by :func:`recommend_chart_type`.

    Returns
    -------
    dict
        A full ECharts ``option`` object ready for ``echarts.setOption()``.
    """
    chart_type = recommend_chart_type(data)
    columns, rows = _extract_columns_rows(data)

    if chart_type == "table" or not columns or not rows:
        return _build_table(columns, rows)

    dispatch = {
        "line": _build_line,
        "bar": _build_bar,
        "pie": _build_pie,
    }
    builder = dispatch.get(chart_type, _build_bar)
    return builder(columns, rows)


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _extract_columns_rows(data: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """Normalise *data* into ``(columns, rows)``."""
    if "columns" in data and "rows" in data:
        return list(data["columns"]), list(data["rows"])

    payload = data.get("data")
    if not payload:
        return [], []

    if isinstance(payload[0], dict):
        columns = list(payload[0].keys())
        rows = [[row.get(c) for c in columns] for row in payload]
        return columns, rows

    if isinstance(payload[0], list):
        columns = [f"col_{i}" for i in range(len(payload[0]))]
        return columns, payload

    return [], []


def _x_axis_data(columns: list[str], rows: list[list[Any]]) -> list[str]:
    """First column values used as x-axis labels."""
    return [str(row[0]) if row else "" for row in rows]


def _series_values(col_idx: int, rows: list[list[Any]]) -> list[Any]:
    """Extract numeric values for column *col_idx*."""
    vals: list[Any] = []
    for row in rows:
        if col_idx < len(row):
            v = row[col_idx]
            try:
                v = float(v)
            except (TypeError, ValueError):
                pass
            vals.append(v)
    return vals


def _build_line(columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    x_data = _x_axis_data(columns, rows)
    series = []
    for i in range(1, len(columns)):
        series.append(
            {
                "name": columns[i],
                "type": "line",
                "data": _series_values(i, rows),
                "smooth": True,
            }
        )
    return {
        "title": {"text": "", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0},
        "xAxis": {"type": "category", "data": x_data},
        "yAxis": {"type": "value"},
        "series": series,
    }


def _build_bar(columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    x_data = _x_axis_data(columns, rows)
    series = []
    for i in range(1, len(columns)):
        series.append(
            {
                "name": columns[i],
                "type": "bar",
                "data": _series_values(i, rows),
            }
        )
    return {
        "title": {"text": "", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0},
        "xAxis": {"type": "category", "data": x_data},
        "yAxis": {"type": "value"},
        "series": series,
    }


def _build_pie(columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    pie_data = []
    for row in rows:
        name = str(row[0]) if row else ""
        value = 0
        if len(row) > 1:
            try:
                value = float(row[1])
            except (TypeError, ValueError):
                value = 0
        pie_data.append({"name": name, "value": value})
    return {
        "title": {"text": "", "left": "center"},
        "tooltip": {"trigger": "item"},
        "legend": {"bottom": 0},
        "series": [
            {
                "type": "pie",
                "radius": "50%",
                "data": pie_data,
            }
        ],
    }


def _build_table(columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    """Return a pseudo-ECharts config that the frontend renders as a table."""
    return {
        "title": {"text": "Data Table", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "dataset": {
            "source": {
                "columns": columns,
                "rows": rows,
            },
        },
        "renderAs": "table",
    }
