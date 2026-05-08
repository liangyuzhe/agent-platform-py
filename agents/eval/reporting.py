"""Evaluation report serialization and summary helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def percentile(values: list[float], p: float) -> float:
    """Return the nearest-rank percentile for a non-empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, round((p / 100) * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def build_report_payload(
    reports: list[Any],
    dataset_path: str | Path,
    k_values: list[int] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build a durable, UI-friendly evaluation report payload."""
    k_values = k_values or [1, 3, 5, 10]
    strategies = []

    for report in reports:
        latencies = [r.latency_ms for r in report.results]
        first_token_latencies = [
            getattr(r, "first_token_latency_ms", None)
            for r in report.results
            if getattr(r, "first_token_latency_ms", None) is not None
        ]

        strategies.append({
            "strategy": report.config.name,
            "description": report.config.description,
            "relevant_field": getattr(report.config, "relevant_field", "relevant_doc_ids"),
            "num_queries": len(report.results),
            "metrics": {k: round(v, 4) for k, v in report.aggregate.items()},
            "latency": {
                "avg_ms": round(report.avg_latency_ms, 1),
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
                    "relevant_doc_ids": sorted(r.relevant_doc_ids),
                    "retrieved_doc_ids": r.retrieved_doc_ids,
                    "metrics": {k: round(v, 4) for k, v in r.metrics.items()},
                    "latency_ms": round(r.latency_ms, 1),
                    "first_token_latency_ms": getattr(r, "first_token_latency_ms", None),
                }
                for r in report.results
            ],
        })

    return {
        "run_id": run_id or str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "k_values": k_values,
        "report_type": "retrieval",
        "strategies": strategies,
    }


def compact_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a small summary for report lists and dashboard cards."""
    strategies = payload.get("strategies", [])
    best = None
    if strategies:
        best = max(
            strategies,
            key=lambda s: (
                s.get("metrics", {}).get("accuracy@5", 0),
                s.get("metrics", {}).get("recall@5", 0),
                -s.get("latency", {}).get("p95_ms", 0),
            ),
        )
    return {
        "run_id": payload.get("run_id"),
        "created_at": payload.get("created_at"),
        "dataset_path": payload.get("dataset_path"),
        "report_type": payload.get("report_type", "retrieval"),
        "strategy_count": len(strategies),
        "best_strategy": best.get("strategy") if best else None,
        "best_metrics": best.get("metrics", {}) if best else {},
        "best_latency": best.get("latency", {}) if best else {},
    }
