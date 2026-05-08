"""Unit tests for evaluation report serialization."""

from types import SimpleNamespace

from agents.eval.reporting import build_report_payload, compact_report, percentile


def _result(query, latency, metrics):
    return SimpleNamespace(
        query=query,
        relevant_doc_ids={"schema_a"},
        retrieved_doc_ids=["schema_a", "schema_b"],
        metrics=metrics,
        latency_ms=latency,
    )


def test_percentile_nearest_rank():
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile([10, 20, 30, 40], 95) == 40
    assert percentile([], 95) == 0.0


def test_build_report_payload_contains_summary_and_traceable_results():
    report = SimpleNamespace(
        config=SimpleNamespace(name="hybrid", description="Hybrid retrieval"),
        results=[
            _result("q1", 10.0, {"accuracy@5": 1.0, "recall@5": 1.0}),
            _result("q2", 30.0, {"accuracy@5": 0.0, "recall@5": 0.5}),
        ],
        aggregate={"accuracy@5": 0.5, "recall@5": 0.75},
        avg_latency_ms=20.0,
    )

    payload = build_report_payload([report], "data/eval/dataset.jsonl", run_id="run-1")

    assert payload["run_id"] == "run-1"
    strategy = payload["strategies"][0]
    assert strategy["relevant_field"] == "relevant_doc_ids"
    assert strategy["metrics"] == {"accuracy@5": 0.5, "recall@5": 0.75}
    assert strategy["latency"] == {"avg_ms": 20.0, "p50_ms": 10.0, "p95_ms": 30.0}
    assert strategy["first_token_latency"] == {"avg_ms": None, "p50_ms": None, "p95_ms": None}
    assert strategy["results"][0]["query"] == "q1"
    assert strategy["results"][0]["relevant_doc_ids"] == ["schema_a"]


def test_compact_report_selects_best_strategy_by_accuracy_recall_latency():
    payload = {
        "run_id": "run-1",
        "created_at": "2026-05-08T00:00:00Z",
        "dataset_path": "dataset.jsonl",
        "strategies": [
            {"strategy": "slow", "metrics": {"accuracy@5": 0.8, "recall@5": 0.9}, "latency": {"p95_ms": 500}},
            {"strategy": "fast", "metrics": {"accuracy@5": 0.8, "recall@5": 0.9}, "latency": {"p95_ms": 100}},
        ],
    }

    summary = compact_report(payload)

    assert summary["best_strategy"] == "fast"
    assert summary["best_metrics"]["accuracy@5"] == 0.8
