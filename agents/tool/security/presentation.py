from __future__ import annotations

import json
import re
from typing import Any


_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_EXECUTION_TIME_RE = re.compile(
    r"\s*Query execution time:\s*[\d.]+\s*ms\s*$",
    re.IGNORECASE,
)


def build_column_display_map(
    columns: list[str],
    *,
    semantic_model: dict | None = None,
    table_names: list[str] | None = None,
    sql: str = "",
) -> dict[str, str]:
    display_map: dict[str, str] = {}
    semantic_model = semantic_model or {}
    ordered_tables = _ordered_tables(semantic_model, table_names)

    for column in columns:
        column_name = str(column)
        if _contains_chinese(column_name):
            display_map[column_name] = column_name
            continue

        alias = _chinese_alias_for_column(column_name, sql)
        if alias:
            display_map[column_name] = alias
            continue

        display_map[column_name] = (
            _display_name_from_semantic_model(column_name, semantic_model, ordered_tables)
            or column_name
        )

    return display_map


def format_result_for_user(
    result: str,
    *,
    semantic_model: dict | None = None,
    table_names: list[str] | None = None,
    sql: str = "",
) -> tuple[str, dict]:
    parsed = _parse_result_payload(result)
    if parsed is _UNPARSEABLE:
        return (
            f"查询已执行完成。\n{result}",
            {"display_columns": {}, "row_count": None},
        )

    rows, row_count = _normalize_rows(parsed)
    if rows is None:
        return (
            "查询已执行完成。\n结果：无数据",
            {"display_columns": {}, "row_count": row_count},
        )

    columns = _columns_from_rows(rows)
    display_columns = build_column_display_map(
        columns,
        semantic_model=semantic_model,
        table_names=table_names,
        sql=sql,
    )

    if not rows:
        return (
            "查询已执行完成。\n未查询到符合条件的数据。",
            {"display_columns": display_columns, "row_count": row_count},
        )

    text = _format_rows(rows, display_columns)
    return text, {"display_columns": display_columns, "row_count": row_count}


def _ordered_tables(semantic_model: dict, table_names: list[str] | None) -> list[str]:
    ordered: list[str] = []
    for table_name in table_names or []:
        if table_name in semantic_model and table_name not in ordered:
            ordered.append(table_name)
    for table_name in semantic_model:
        if table_name not in ordered:
            ordered.append(table_name)
    return ordered


def _contains_chinese(value: str) -> bool:
    return bool(_CHINESE_RE.search(value or ""))


def _chinese_alias_for_column(column: str, sql: str) -> str | None:
    if not sql:
        return None
    identifier = re.escape(column)
    column_ref = rf"(?:[`\"]?\w+[`\"]?\.)?[`\"]?{identifier}[`\"]?"
    alias_re = re.compile(
        rf"(?<![\w.]){column_ref}\s+(?:AS\s+)?[`\"]?"
        rf"(?P<alias>[\u4e00-\u9fff][^`\",\s;)]*)[`\"]?",
        re.IGNORECASE,
    )
    for match in alias_re.finditer(sql):
        alias = match.group("alias").strip()
        if alias:
            return alias
    return None


def _display_name_from_semantic_model(
    column: str,
    semantic_model: dict,
    ordered_tables: list[str],
) -> str | None:
    for table_name in ordered_tables:
        table_meta = semantic_model.get(table_name) or {}
        column_meta = table_meta.get(column)
        if not isinstance(column_meta, dict):
            continue
        business_name = str(column_meta.get("business_name") or "").strip()
        if business_name:
            return business_name
        column_comment = str(column_meta.get("column_comment") or "").strip()
        if column_comment:
            return column_comment
    return None


_UNPARSEABLE = object()


def _parse_result_payload(result: str) -> Any:
    clean = _EXECUTION_TIME_RE.sub("", result or "").strip()
    try:
        return json.loads(clean)
    except Exception:
        return _UNPARSEABLE


def _normalize_rows(payload: Any) -> tuple[list[Any] | None, int | None]:
    if payload is None:
        return None, None

    rows = payload
    row_count = None
    if isinstance(payload, dict):
        wrapper_count = _wrapper_row_count(payload)
        for key in ("rows", "data", "result", "items"):
            if key in payload:
                rows = payload[key]
                row_count = wrapper_count
                break
        else:
            rows = [payload]

    if isinstance(rows, list):
        return rows, row_count if row_count is not None else len(rows)
    if isinstance(rows, dict):
        return [rows], row_count if row_count is not None else 1
    return [rows], None


def _wrapper_row_count(payload: dict) -> int | None:
    for key in ("row_count", "total", "count"):
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


def _columns_from_rows(rows: list[Any]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for column in row:
            column_name = str(column)
            if column_name not in columns:
                columns.append(column_name)
    return columns


def _format_rows(rows: list[Any], display_columns: dict[str, str]) -> str:
    if len(rows) == 1 and isinstance(rows[0], dict):
        return "查询已执行完成。\n" + "\n".join(
            _format_row_fields(rows[0], display_columns)
        )
    if len(rows) == 1:
        return f"查询已执行完成。\n结果：{_format_value(rows[0])}"

    preview_lines = []
    for row in rows[:5]:
        if isinstance(row, dict):
            preview_lines.append("，".join(_format_row_fields(row, display_columns)))
        else:
            preview_lines.append(_format_value(row))

    suffix = "\n仅展示前 5 条。" if len(rows) > 5 else ""
    return f"查询已执行完成。\n共返回 {len(rows)} 条记录。\n" + "\n".join(preview_lines) + suffix


def _format_row_fields(row: dict, display_columns: dict[str, str]) -> list[str]:
    return [
        f"{display_columns.get(str(key), str(key))}：{_format_value(value)}"
        for key, value in row.items()
    ]


def _format_value(value: Any) -> str:
    if value is None:
        return "无数据"
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value)
