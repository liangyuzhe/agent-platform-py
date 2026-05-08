"""Unit tests for offline NL2SQL evaluation."""

from __future__ import annotations

import json

from agents.eval.nl2sql_runner import (
    _canonical_result,
    evaluate_nl2sql_case,
    run_nl2sql_evaluation,
    write_nl2sql_template,
)


def test_canonical_result_ignores_execution_time_and_row_order():
    left = '[{"b":2,"a":1},{"a":3,"b":4}]Query execution time: 10.0 ms'
    right = [{"a": 3, "b": 4}, {"a": 1, "b": 2}]

    assert _canonical_result(left) == _canonical_result(right)


def test_evaluate_nl2sql_case_computes_core_metrics():
    result = evaluate_nl2sql_case({
        "query": "去年亏损多少",
        "generated_sql": "SELECT 1;",
        "actual_result": [{"loss_amount": "100.00"}],
        "expected_result": [{"loss_amount": "100.00"}],
        "latency_ms": 25.0,
        "first_token_latency_ms": 12.0,
    })

    assert result.metrics["sql_valid"] == 1.0
    assert result.metrics["execution_success"] == 1.0
    assert result.metrics["result_exact_match"] == 1.0
    assert result.latency_ms == 25.0
    assert result.first_token_latency_ms == 12.0


def test_run_nl2sql_evaluation_writes_report(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    output = tmp_path / "report.json"
    dataset.write_text(
        json.dumps({
            "query": "查数据",
            "generated_sql": "SELECT 1;",
            "actual_result": [{"x": 1}],
            "expected_result": [{"x": 1}],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = run_nl2sql_evaluation(dataset, output)

    assert report["report_type"] == "nl2sql_end_to_end"
    assert report["metrics"]["result_exact_match"] == 1.0
    assert json.loads(output.read_text(encoding="utf-8"))["num_queries"] == 1


def test_write_nl2sql_template_creates_jsonl(tmp_path):
    output = tmp_path / "cases.jsonl"

    write_nl2sql_template(output)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["query"]
    assert "generated_sql" in rows[0]
    assert "actual_result" in rows[0]
    assert "expected_result" in rows[0]
