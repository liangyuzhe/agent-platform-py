"""Offline NL2SQL end-to-end evaluation helpers.

This module evaluates recorded or pre-generated NL2SQL cases without calling
the live agent by default. It is intended as the stable report layer that a
future online replay runner can feed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.eval.reporting import percentile
from agents.model.format_tool import normalize_sql_answer


@dataclass
class NL2SQLCaseResult:
    query: str
    generated_sql: str = ""
    expected_sql: str = ""
    actual_result: Any = None
    expected_result: Any = None
    metrics: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    first_token_latency_ms: float | None = None
    error: str = ""


def _load_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _canonical_result(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        marker = "Query execution time:"
        if marker in text:
            text = text.split(marker, 1)[0].strip()
        try:
            value = json.loads(text)
        except Exception:
            return text
    if isinstance(value, dict):
        for key in ("rows", "data", "result", "items"):
            if key in value:
                return _canonical_result(value[key])
        return {str(k): _canonical_result(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        normalized = [_canonical_result(v) for v in value]
        if all(isinstance(v, dict) for v in normalized):
            return sorted(normalized, key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True))
        return normalized
    return value


def _result_matches(actual: Any, expected: Any) -> float:
    if actual is None or expected is None:
        return 0.0
    return 1.0 if _canonical_result(actual) == _canonical_result(expected) else 0.0


def evaluate_nl2sql_case(item: dict) -> NL2SQLCaseResult:
    """Evaluate one recorded NL2SQL case without executing external services."""
    sql = item.get("generated_sql") or item.get("sql") or ""
    expected_sql = item.get("expected_sql", "")
    t0 = time.monotonic()
    normalized_sql, sql_ok, _ = normalize_sql_answer(sql) if sql else ("", False, "missing_sql")
    latency_ms = float(item.get("latency_ms") or 0.0)
    if not latency_ms:
        latency_ms = (time.monotonic() - t0) * 1000

    error = str(item.get("error") or "")
    actual_result = item.get("actual_result", item.get("result"))
    expected_result = item.get("expected_result")
    has_result_label = expected_result is not None

    metrics = {
        "sql_valid": 1.0 if sql_ok else 0.0,
        "execution_success": 0.0 if error else 1.0,
    }
    if has_result_label:
        metrics["result_exact_match"] = _result_matches(actual_result, expected_result)

    return NL2SQLCaseResult(
        query=item.get("query", ""),
        generated_sql=normalized_sql or sql,
        expected_sql=expected_sql,
        actual_result=actual_result,
        expected_result=expected_result,
        metrics=metrics,
        latency_ms=latency_ms,
        first_token_latency_ms=item.get("first_token_latency_ms"),
        error=error,
    )


def _aggregate_case_metrics(results: list[NL2SQLCaseResult]) -> dict[str, float]:
    if not results:
        return {}
    keys = sorted({key for result in results for key in result.metrics})
    aggregate = {}
    for key in keys:
        values = [result.metrics[key] for result in results if key in result.metrics]
        if values:
            aggregate[key] = sum(values) / len(values)
    return aggregate


def build_nl2sql_report(results: list[NL2SQLCaseResult], dataset_path: str | Path = "") -> dict:
    """Build a durable NL2SQL report payload."""
    latencies = [r.latency_ms for r in results]
    first_token_latencies = [r.first_token_latency_ms for r in results if r.first_token_latency_ms is not None]
    return {
        "report_type": "nl2sql_end_to_end",
        "dataset_path": str(dataset_path),
        "num_queries": len(results),
        "metrics": {k: round(v, 4) for k, v in _aggregate_case_metrics(results).items()},
        "latency": {
            "avg_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "p50_ms": round(percentile(latencies, 50), 1),
            "p95_ms": round(percentile(latencies, 95), 1),
        },
        "first_token_latency": {
            "avg_ms": round(sum(first_token_latencies) / len(first_token_latencies), 1)
            if first_token_latencies else None,
            "p50_ms": round(percentile(first_token_latencies, 50), 1)
            if first_token_latencies else None,
            "p95_ms": round(percentile(first_token_latencies, 95), 1)
            if first_token_latencies else None,
        },
        "results": [
            {
                "query": r.query,
                "generated_sql": r.generated_sql,
                "expected_sql": r.expected_sql,
                "actual_result": r.actual_result,
                "expected_result": r.expected_result,
                "metrics": {k: round(v, 4) for k, v in r.metrics.items()},
                "latency_ms": round(r.latency_ms, 1),
                "first_token_latency_ms": r.first_token_latency_ms,
                "error": r.error,
            }
            for r in results
        ],
    }


def run_nl2sql_evaluation(
    dataset_path: str | Path,
    output_path: str | Path = "nl2sql_eval_report.json",
) -> dict:
    """Run offline NL2SQL evaluation from a JSONL dataset and save report."""
    items = _load_jsonl(dataset_path)
    results = [evaluate_nl2sql_case(item) for item in items]
    report = build_nl2sql_report(results, dataset_path=dataset_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
