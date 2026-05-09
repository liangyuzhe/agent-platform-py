"""Online NL2SQL evaluation runner.

This runner replays natural-language cases through the live LangGraph agent.
It can stop at the SQL approval interrupt, or auto-approve safe SQL so the
case reaches SQL execution and result reflection.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langgraph.types import Command

from agents.eval.nl2sql_runner import (
    NL2SQLCaseResult,
    build_nl2sql_report,
    evaluate_nl2sql_case,
)
from agents.tool.trace.tracing import get_trace_callbacks


ONLINE_NL2SQL_TEMPLATE_CASES = [
    {
        "query": "去年亏损多少",
        "expected_result": [{"loss_amount": "10000.00"}],
        "tags": ["profit_loss", "multi_turn_baseline"],
    }
]


@dataclass
class OnlineNL2SQLCaseResult:
    """Raw online run detail before converting to report metrics."""

    query: str
    session_id: str
    generated_sql: str = ""
    sql_rounds: list[str] = field(default_factory=list)
    actual_result: Any = None
    expected_result: Any = None
    expected_sql: str = ""
    answer: str = ""
    status: str = "completed"
    error: str = ""
    latency_ms: float = 0.0
    first_response_latency_ms: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)


def _load_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_online_nl2sql_template(output_path: str | Path) -> Path:
    """Write a starter JSONL dataset for online NL2SQL replay."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for item in ONLINE_NL2SQL_TEMPLATE_CASES:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output


def _make_config(session_id: str) -> dict:
    callbacks = get_trace_callbacks()
    config: dict[str, Any] = {"configurable": {"thread_id": session_id}}
    if callbacks:
        config["callbacks"] = callbacks
    return config


def _extract_interrupt(result: dict) -> dict | None:
    interrupts = result.get("__interrupt__", []) if isinstance(result, dict) else []
    if not interrupts:
        return None
    interrupt = interrupts[0] if isinstance(interrupts, list) else interrupts
    value = interrupt.value if hasattr(interrupt, "value") else interrupt
    if isinstance(value, list) and value:
        value = value[0]
    return value if isinstance(value, dict) else None


def _to_metric_case(raw: OnlineNL2SQLCaseResult) -> NL2SQLCaseResult:
    metric_item = {
        "query": raw.query,
        "generated_sql": raw.generated_sql,
        "expected_sql": raw.expected_sql,
        "actual_result": raw.actual_result,
        "expected_result": raw.expected_result,
        "latency_ms": raw.latency_ms,
        "first_token_latency_ms": raw.first_response_latency_ms,
        "error": raw.error,
    }
    result = evaluate_nl2sql_case(metric_item)
    result.metrics.update(raw.metrics)
    return result


async def run_online_nl2sql_case(
    item: dict,
    *,
    case_index: int = 0,
    run_id: str | None = None,
    session_prefix: str = "eval-online-nl2sql",
    auto_approve_sql: bool = False,
    max_approval_rounds: int = 2,
    force_sql_intent: bool = True,
    graph_factory: Callable[[], Any] | None = None,
    config_factory: Callable[[str], dict] = _make_config,
) -> OnlineNL2SQLCaseResult:
    """Run one NL2SQL case through the live graph.

    ``auto_approve_sql=False`` records the first generated SQL and stops at the
    approval interrupt. Set it to true for full execution/result-reflection
    evaluation.
    """
    from agents.flow.dispatcher import build_final_graph

    query = item.get("query", "")
    run_id = run_id or uuid.uuid4().hex[:8]
    session_id = item.get("session_id") or f"{session_prefix}-{run_id}-{case_index}"
    graph = (graph_factory or build_final_graph)()
    config = config_factory(session_id)
    t0 = time.monotonic()
    first_response_latency_ms: float | None = None
    sql_rounds: list[str] = []

    state: dict[str, Any] = {
        "query": query,
        "session_id": session_id,
        "chat_history": item.get("chat_history", []),
    }
    if force_sql_intent:
        state["intent"] = item.get("intent", "sql_query")
        state["rewritten_query"] = item.get("rewritten_query", query)
    else:
        if item.get("intent"):
            state["intent"] = item["intent"]
        if item.get("rewritten_query"):
            state["rewritten_query"] = item["rewritten_query"]

    status = "completed"
    error = ""
    answer = ""
    actual_result: Any = None
    generated_sql = ""
    final_result: dict[str, Any] = {}

    try:
        result = await graph.ainvoke(state, config=config)
        interrupt_val = _extract_interrupt(result)
        if interrupt_val and first_response_latency_ms is None:
            first_response_latency_ms = (time.monotonic() - t0) * 1000

        approval_round = 0
        while interrupt_val:
            sql = interrupt_val.get("sql", "")
            if sql:
                sql_rounds.append(sql)
                generated_sql = sql
            if not auto_approve_sql:
                status = "pending_approval"
                error = "pending_approval"
                break
            if approval_round >= max_approval_rounds:
                status = "max_approval_rounds_reached"
                error = status
                break

            approval_round += 1
            result = await graph.ainvoke(
                Command(resume={"approved": True, "feedback": ""}),
                config=config,
            )
            interrupt_val = _extract_interrupt(result)
            if interrupt_val and first_response_latency_ms is None:
                first_response_latency_ms = (time.monotonic() - t0) * 1000

        if not interrupt_val:
            final_result = result if isinstance(result, dict) else {}
            generated_sql = final_result.get("sql", generated_sql)
            actual_result = final_result.get("result")
            answer = final_result.get("answer", "")
            status = final_result.get("status", status)
            error = str(final_result.get("error") or "")
            if not error and isinstance(actual_result, str) and actual_result.startswith("SQL 执行失败"):
                error = actual_result
            if not error and isinstance(answer, str) and answer.startswith("SQL 执行失败"):
                error = answer
    except Exception as e:
        status = "error"
        error = str(e)

    latency_ms = (time.monotonic() - t0) * 1000
    if first_response_latency_ms is None:
        first_response_latency_ms = latency_ms

    raw = OnlineNL2SQLCaseResult(
        query=query,
        session_id=session_id,
        generated_sql=generated_sql,
        sql_rounds=sql_rounds,
        actual_result=actual_result,
        expected_result=item.get("expected_result"),
        expected_sql=item.get("expected_sql", ""),
        answer=answer,
        status=status,
        error=error,
        latency_ms=latency_ms,
        first_response_latency_ms=first_response_latency_ms,
        metrics={
            "completed": 1.0 if status == "completed" and not error else 0.0,
            "auto_approved": 1.0 if auto_approve_sql else 0.0,
        },
    )
    raw.metrics.update(_to_metric_case(raw).metrics)
    return raw


def _build_online_report(raw_results: list[OnlineNL2SQLCaseResult], dataset_path: str | Path) -> dict:
    metric_results = [_to_metric_case(raw) for raw in raw_results]
    report = build_nl2sql_report(metric_results, dataset_path=dataset_path)
    report["report_type"] = "online_nl2sql_end_to_end"
    for idx, raw in enumerate(raw_results):
        report["results"][idx].update({
            "session_id": raw.session_id,
            "sql_rounds": raw.sql_rounds,
            "answer": raw.answer,
            "status": raw.status,
            "first_response_latency_ms": round(raw.first_response_latency_ms or 0.0, 1),
        })
    return report


async def run_online_nl2sql_evaluation_async(
    dataset_path: str | Path,
    output_path: str | Path = "data/eval/online_nl2sql_eval_report.json",
    *,
    auto_approve_sql: bool = False,
    max_approval_rounds: int = 2,
    force_sql_intent: bool = True,
    session_prefix: str = "eval-online-nl2sql",
    graph_factory: Callable[[], Any] | None = None,
    config_factory: Callable[[str], dict] = _make_config,
) -> dict:
    """Run online NL2SQL evaluation and save a report."""
    items = _load_jsonl(dataset_path)
    run_id = uuid.uuid4().hex[:8]
    raw_results = []
    for idx, item in enumerate(items):
        raw_results.append(
            await run_online_nl2sql_case(
                item,
                case_index=idx,
                run_id=run_id,
                session_prefix=session_prefix,
                auto_approve_sql=auto_approve_sql,
                max_approval_rounds=max_approval_rounds,
                force_sql_intent=force_sql_intent,
                graph_factory=graph_factory,
                config_factory=config_factory,
            )
        )

    report = _build_online_report(raw_results, dataset_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def run_online_nl2sql_evaluation(
    dataset_path: str | Path,
    output_path: str | Path = "data/eval/online_nl2sql_eval_report.json",
    *,
    auto_approve_sql: bool = False,
    max_approval_rounds: int = 2,
    force_sql_intent: bool = True,
    session_prefix: str = "eval-online-nl2sql",
) -> dict:
    """Sync wrapper for CLI usage."""
    import asyncio

    return asyncio.run(
        run_online_nl2sql_evaluation_async(
            dataset_path,
            output_path,
            auto_approve_sql=auto_approve_sql,
            max_approval_rounds=max_approval_rounds,
            force_sql_intent=force_sql_intent,
            session_prefix=session_prefix,
        )
    )
