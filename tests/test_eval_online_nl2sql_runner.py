"""Unit tests for online NL2SQL evaluation runner."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents.eval.online_nl2sql_runner import (
    run_online_nl2sql_case,
    run_online_nl2sql_evaluation_async,
    write_online_nl2sql_template,
)


class _FakeGraph:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, payload, config=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "__interrupt__": [
                    SimpleNamespace(value={"sql": "SELECT 1 AS x;", "message": "approve?"})
                ]
            }
        return {
            "sql": "SELECT 1 AS x;",
            "result": [{"x": 1}],
            "answer": "x：1",
            "status": "completed",
        }


@pytest.mark.asyncio
async def test_online_case_auto_approve_reaches_execution():
    result = await run_online_nl2sql_case(
        {"query": "查 x", "expected_result": [{"x": 1}]},
        auto_approve_sql=True,
        graph_factory=_FakeGraph,
        config_factory=lambda session_id: {"configurable": {"thread_id": session_id}},
    )

    assert result.status == "completed"
    assert result.generated_sql == "SELECT 1 AS x;"
    assert result.actual_result == [{"x": 1}]
    assert result.metrics["result_exact_match"] == 1.0
    assert result.metrics["completed"] == 1.0


@pytest.mark.asyncio
async def test_online_case_without_auto_approve_stops_at_interrupt():
    result = await run_online_nl2sql_case(
        {"query": "查 x"},
        auto_approve_sql=False,
        graph_factory=_FakeGraph,
        config_factory=lambda session_id: {"configurable": {"thread_id": session_id}},
    )

    assert result.status == "pending_approval"
    assert result.generated_sql == "SELECT 1 AS x;"
    assert result.metrics["sql_valid"] == 1.0
    assert result.metrics["execution_success"] == 0.0


@pytest.mark.asyncio
async def test_run_online_evaluation_writes_report(tmp_path):
    dataset = tmp_path / "online_cases.jsonl"
    output = tmp_path / "online_report.json"
    dataset.write_text(
        json.dumps({"query": "查 x", "expected_result": [{"x": 1}]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = await run_online_nl2sql_evaluation_async(
        dataset,
        output,
        auto_approve_sql=True,
        graph_factory=_FakeGraph,
        config_factory=lambda session_id: {"configurable": {"thread_id": session_id}},
    )

    assert report["report_type"] == "online_nl2sql_end_to_end"
    assert report["metrics"]["result_exact_match"] == 1.0
    assert report["results"][0]["sql_rounds"] == ["SELECT 1 AS x;"]
    assert json.loads(output.read_text(encoding="utf-8"))["num_queries"] == 1


def test_write_online_nl2sql_template_creates_jsonl(tmp_path):
    output = tmp_path / "online_cases.jsonl"

    write_online_nl2sql_template(output)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["query"]
    assert "expected_result" in rows[0]
