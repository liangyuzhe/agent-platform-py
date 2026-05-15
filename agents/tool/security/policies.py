from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class SecurityContext:
    user_id: str = ""
    username: str = ""
    role_ids: list[str] = field(default_factory=list)
    department_ids: list[int] = field(default_factory=list)
    company_id: int | None = None
    data_scopes: dict[str, list[Any]] = field(default_factory=dict)
    allowed_tables: list[str] | None = None
    denied_tables: list[str] = field(default_factory=list)
    column_policies: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: "SecurityContext | Mapping[str, Any] | None",
    ) -> "SecurityContext":
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        data = dict(data)
        return cls(
            user_id=data.get("user_id", "") or "",
            username=data.get("username", "") or "",
            role_ids=list(data.get("role_ids") or []),
            department_ids=list(data.get("department_ids") or []),
            company_id=data.get("company_id"),
            data_scopes=dict(data.get("data_scopes") or {}),
            allowed_tables=(
                list(data["allowed_tables"])
                if data.get("allowed_tables") is not None
                else None
            ),
            denied_tables=list(data.get("denied_tables") or []),
            column_policies=dict(data.get("column_policies") or {}),
        )


@dataclass
class TableAuthorizationResult:
    allowed: bool
    allowed_tables: list[str]
    denied_tables: list[str]
    display_denied_tables: list[str]
    message: str = ""
    stage: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "allowed_tables": self.allowed_tables,
            "denied_tables": self.denied_tables,
            "display_denied_tables": self.display_denied_tables,
            "message": self.message,
            "stage": self.stage,
        }


def display_name_for_table(
    table_name: str,
    table_metadata: dict[str, str] | None = None,
) -> str:
    if table_metadata:
        display_name = table_metadata.get(table_name)
        if display_name:
            return display_name
    return table_name


def authorize_tables(
    tables: list[str],
    context: SecurityContext | dict | None,
    table_metadata: dict[str, str] | None = None,
    stage: str = "selected_tables",
) -> TableAuthorizationResult:
    unique_tables = _dedupe(tables)
    if context is None:
        return TableAuthorizationResult(
            allowed=True,
            allowed_tables=unique_tables,
            denied_tables=[],
            display_denied_tables=[],
            stage=stage,
        )

    security_context = SecurityContext.from_dict(context)
    allowed_lookup = (
        set(security_context.allowed_tables)
        if security_context.allowed_tables is not None
        else None
    )
    denied_lookup = set(security_context.denied_tables)

    allowed_tables: list[str] = []
    denied_tables: list[str] = []
    for table in unique_tables:
        is_denied = table in denied_lookup
        is_not_allowed = allowed_lookup is not None and table not in allowed_lookup
        if is_denied or is_not_allowed:
            denied_tables.append(table)
        else:
            allowed_tables.append(table)

    display_denied_tables = [
        display_name_for_table(table, table_metadata) for table in denied_tables
    ]
    message = ""
    if denied_tables:
        joined_names = "、".join(display_denied_tables)
        message = (
            f"当前问题需要访问「{joined_names}」相关数据，但你暂无该数据权限。"
            "请联系管理员开通后再查询。"
        )

    return TableAuthorizationResult(
        allowed=not denied_tables,
        allowed_tables=allowed_tables,
        denied_tables=denied_tables,
        display_denied_tables=display_denied_tables,
        message=message,
        stage=stage,
    )


def build_audit_event(
    event_type: str,
    *,
    query: str = "",
    context: SecurityContext | dict | None = None,
    selected_tables: list[str] | None = None,
    denied_tables: list[str] | None = None,
    display_tables: list[str] | None = None,
    status: str = "",
    error: str = "",
    extra: dict | None = None,
) -> dict:
    security_context = (
        context if isinstance(context, SecurityContext) else SecurityContext.from_dict(context)
    )
    event = {
        "event_type": event_type,
        "query": query,
        "user_id": security_context.user_id,
        "username": security_context.username,
        "role_ids": list(security_context.role_ids),
        "department_ids": list(security_context.department_ids),
        "company_id": security_context.company_id,
        "selected_tables": list(selected_tables or []),
        "denied_tables": list(denied_tables or []),
        "display_tables": list(display_tables or []),
        "status": status,
        "error": error,
        "extra": dict(extra or {}),
    }
    return event


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
