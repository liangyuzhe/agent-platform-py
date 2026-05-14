"""Complex NL2SQL routing and plan validation helpers.

This module is intentionally structural. Runtime business keywords belong in
metadata, database-backed rules, or LLM arbitration, not in Python constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SINGLE_SQL_TABLE_LIMIT = 8
PRIMARY_SELECTED_TABLE_LIMIT = 5
MAX_COMPLEX_PLAN_STEPS = 5
MAX_TABLES_PER_SQL_STEP = 5

RouteMode = Literal["single_sql", "single_sql_with_strict_checks", "complex_plan", "clarify"]
RouteSignal = Literal["analysis", "report", "comparison", "detail", "export", "sensitive", "ambiguous"]
VALID_STEP_TYPES = {"sql", "python_merge", "report"}


@dataclass(frozen=True)
class ComplexityDecision:
    route_mode: RouteMode
    selected_tables_count: int
    relationship_count: int
    estimated_join_count: int
    query_intent_complexity: str
    reason: str


def classify_query_complexity(
    query: str,
    selected_tables: list[str],
    relationships: list[dict],
    route_signal: RouteSignal | str | None = None,
) -> ComplexityDecision:
    """Choose the SQL route from structure plus optional semantic signal."""
    del query  # Query semantics are supplied by route_signal, not parsed here.

    selected_count = len(dict.fromkeys(selected_tables))
    relationship_count = len(relationships or [])
    estimated_join_count = max(0, selected_count - 1)

    if selected_count <= PRIMARY_SELECTED_TABLE_LIMIT:
        return ComplexityDecision(
            route_mode="single_sql",
            selected_tables_count=selected_count,
            relationship_count=relationship_count,
            estimated_join_count=estimated_join_count,
            query_intent_complexity="normal",
            reason="selected table count is within primary budget",
        )

    if selected_count <= SINGLE_SQL_TABLE_LIMIT:
        return ComplexityDecision(
            route_mode="single_sql_with_strict_checks",
            selected_tables_count=selected_count,
            relationship_count=relationship_count,
            estimated_join_count=estimated_join_count,
            query_intent_complexity="medium",
            reason="selected table count is within single SQL budget",
        )

    if route_signal in {"analysis", "report", "comparison"}:
        return ComplexityDecision(
            route_mode="complex_plan",
            selected_tables_count=selected_count,
            relationship_count=relationship_count,
            estimated_join_count=estimated_join_count,
            query_intent_complexity=str(route_signal),
            reason="broad query has a configured/LLM analysis route signal",
        )

    if route_signal in {"detail", "export", "sensitive"}:
        return ComplexityDecision(
            route_mode="clarify",
            selected_tables_count=selected_count,
            relationship_count=relationship_count,
            estimated_join_count=estimated_join_count,
            query_intent_complexity=str(route_signal),
            reason="broad query has a configured/LLM clarify route signal",
        )

    return ComplexityDecision(
        route_mode="clarify",
        selected_tables_count=selected_count,
        relationship_count=relationship_count,
        estimated_join_count=estimated_join_count,
        query_intent_complexity="ambiguous",
        reason="query exceeds single SQL table budget but lacks a clear analysis goal",
    )


def validate_complex_plan(plan: dict, allowed_tables: set[str]) -> tuple[bool, str]:
    """Validate a planner output before any SQL generation or execution."""
    if not isinstance(plan, dict):
        return False, "plan must be an object"

    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return False, "plan.steps must be a non-empty list"
    if len(steps) > MAX_COMPLEX_PLAN_STEPS:
        return False, f"plan has too many steps: {len(steps)}"

    seen_steps = set()
    has_sql = False
    for item in steps:
        if not isinstance(item, dict):
            return False, "each step must be an object"

        step_no = item.get("step")
        step_type = item.get("type")
        if not isinstance(step_no, int):
            return False, "each step must have integer step"
        if step_no in seen_steps:
            return False, f"duplicate step: {step_no}"
        seen_steps.add(step_no)

        if step_type not in VALID_STEP_TYPES:
            return False, f"unsupported step type: {step_type}"
        if not item.get("goal"):
            return False, f"step {step_no} missing goal"

        tables = item.get("tables") or []
        if step_type == "sql":
            has_sql = True
            if not tables:
                return False, f"sql step {step_no} missing tables"
            if len(tables) > MAX_TABLES_PER_SQL_STEP:
                return False, f"sql step {step_no} has too many tables"
            unknown = set(tables) - allowed_tables
            if unknown:
                return False, f"step {step_no} uses unknown table(s): {sorted(unknown)}"

        depends_on = item.get("depends_on") or []
        if any(dep not in seen_steps for dep in depends_on):
            return False, f"step {step_no} depends on unknown or future step"
        if step_type == "python_merge" and not item.get("merge_keys"):
            return False, f"python_merge step {step_no} missing merge_keys"

    if not has_sql:
        return False, "plan must include at least one sql step"
    return True, ""
