"""Database-backed complex-query route rules.

Rules are optional semantic signals for broad NL2SQL routing. The code owns
only table DDL and match mechanics; keywords, regex patterns, priorities and
route labels live in MySQL/Admin.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from agents.config.settings import settings

logger = logging.getLogger(__name__)

_TABLE_NAME = "t_query_route_rule"
_MATCH_TYPES = {"contains", "exact", "regex"}
_ROUTE_SIGNALS = {"analysis", "report", "comparison", "detail", "export", "sensitive", "ambiguous"}


@dataclass(frozen=True)
class QueryRouteRuleDecision:
    route_signal: str
    confidence: float
    rule_id: int
    rule_name: str
    priority: int
    match_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_signal": self.route_signal,
            "confidence": self.confidence,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "priority": self.priority,
            "match_type": self.match_type,
        }


def _get_conn():
    import pymysql

    return pymysql.connect(
        host=settings.mysql.host,
        port=settings.mysql.port,
        user=settings.mysql.username,
        password=settings.mysql.password,
        database=settings.mysql.database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def ensure_query_route_rule_table() -> None:
    """Ensure the configurable complex-query route rule table exists."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(128) NOT NULL,
                    route_signal VARCHAR(64) NOT NULL,
                    match_type VARCHAR(32) NOT NULL DEFAULT 'contains',
                    pattern TEXT NOT NULL,
                    priority INT NOT NULL DEFAULT 100,
                    confidence DECIMAL(4,3) NOT NULL DEFAULT 0.900,
                    enabled TINYINT(1) NOT NULL DEFAULT 1,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_enabled_priority (enabled, priority)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='复杂查询路由规则'
            """)
        conn.commit()
        logger.info("Ensured %s table exists", _TABLE_NAME)
    except Exception as e:
        logger.warning("Failed to create %s: %s", _TABLE_NAME, e)
    finally:
        conn.close()


def list_query_route_rules(enabled_only: bool = False) -> list[dict[str, Any]]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            if enabled_only:
                cur.execute(f"SELECT * FROM {_TABLE_NAME} WHERE enabled=1 ORDER BY priority DESC, id ASC")
            else:
                cur.execute(f"SELECT * FROM {_TABLE_NAME} ORDER BY priority DESC, id ASC")
            return [_normalize_row(row) for row in cur.fetchall()]
    finally:
        conn.close()


def upsert_query_route_rule(rule: dict[str, Any]) -> int:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            rule_id = rule.get("id")
            values = (
                str(rule.get("name", "")).strip(),
                _normalize_route_signal(str(rule.get("route_signal", "ambiguous"))),
                _normalize_match_type(str(rule.get("match_type", "contains"))),
                rule.get("pattern", ""),
                int(rule.get("priority", 100)),
                float(rule.get("confidence", 0.9)),
                1 if rule.get("enabled", True) else 0,
                rule.get("description", ""),
            )
            if rule_id:
                cur.execute(
                    f"""UPDATE {_TABLE_NAME}
                        SET name=%s, route_signal=%s, match_type=%s, pattern=%s,
                            priority=%s, confidence=%s, enabled=%s, description=%s
                        WHERE id=%s""",
                    (*values, int(rule_id)),
                )
                saved_id = int(rule_id)
            else:
                cur.execute(
                    f"""INSERT INTO {_TABLE_NAME}
                        (name, route_signal, match_type, pattern, priority,
                         confidence, enabled, description)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    values,
                )
                saved_id = int(cur.lastrowid)
        conn.commit()
        return saved_id
    finally:
        conn.close()


def delete_query_route_rule(rule_id: int) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {_TABLE_NAME} WHERE id=%s", (rule_id,))
        conn.commit()
    finally:
        conn.close()


async def evaluate_query_route_rules(query: str) -> QueryRouteRuleDecision | None:
    """Evaluate enabled DB rules and return the strongest route signal."""
    if not query:
        return None

    def _evaluate() -> QueryRouteRuleDecision | None:
        try:
            rules = list_query_route_rules(enabled_only=True)
        except Exception as e:
            logger.warning("Query route rule load failed: %s", e)
            return None

        for rule in rules:
            if _matches(query, rule):
                return QueryRouteRuleDecision(
                    route_signal=str(rule.get("route_signal") or ""),
                    confidence=float(rule.get("confidence") or 0),
                    rule_id=int(rule.get("id") or 0),
                    rule_name=str(rule.get("name") or ""),
                    priority=int(rule.get("priority") or 0),
                    match_type=str(rule.get("match_type") or ""),
                )
        return None

    return await asyncio.to_thread(_evaluate)


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if isinstance(normalized.get("confidence"), Decimal):
        normalized["confidence"] = float(normalized["confidence"])
    normalized["enabled"] = bool(normalized.get("enabled"))
    normalized["route_signal"] = _normalize_route_signal(str(normalized.get("route_signal") or "ambiguous"))
    return normalized


def _normalize_match_type(value: str) -> str:
    match_type = (value or "contains").strip().lower()
    if match_type not in _MATCH_TYPES:
        raise ValueError(f"Unsupported query route rule match_type: {value}")
    return match_type


def _normalize_route_signal(value: str) -> str:
    signal = (value or "ambiguous").strip().lower()
    if signal not in _ROUTE_SIGNALS:
        raise ValueError(f"Unsupported route signal: {value}")
    return signal


def _matches(query: str, rule: dict[str, Any]) -> bool:
    pattern = str(rule.get("pattern") or "")
    if not pattern:
        return False

    match_type = _normalize_match_type(str(rule.get("match_type") or "contains"))
    if match_type == "exact":
        return query == pattern
    if match_type == "contains":
        return pattern.lower() in query.lower()

    try:
        return re.search(pattern, query, flags=re.IGNORECASE) is not None
    except re.error as e:
        logger.warning("Invalid query route rule regex #%s: %s", rule.get("id"), e)
        return False
