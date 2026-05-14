"""Evaluation retrieval strategies aligned with the NL2SQL pipeline."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class StrategyRunResult:
    retrieved_doc_ids: list[str]
    latency_ms: float


def _tables_to_doc_ids(tables: list[str]) -> list[str]:
    return [f"schema_{table}" for table in tables if table]


def _docs_to_ids(docs: list[Document]) -> list[str]:
    ids = []
    for doc in docs:
        doc_id = doc.metadata.get("doc_id") or doc.metadata.get("term") or ""
        if doc_id:
            ids.append(str(doc_id))
    return ids


async def run_preselect_pipeline(query: str) -> StrategyRunResult:
    """Run recall_evidence -> query_enhance -> select_tables.

    This evaluates the table-selection path as it is used before SQL
    generation, including business knowledge driven query enhancement.
    """
    from agents.model.chat_model import init_chat_models
    from agents.flow.sql_react import recall_evidence, query_enhance, select_tables

    init_chat_models()
    state = {"query": query, "rewritten_query": query}
    t0 = time.monotonic()
    evidence_update = await recall_evidence(state)
    state.update(evidence_update)
    enhance_update = await query_enhance(state)
    state.update(enhance_update)
    table_update = await select_tables(state)
    latency_ms = (time.monotonic() - t0) * 1000
    return StrategyRunResult(
        retrieved_doc_ids=_tables_to_doc_ids(table_update.get("selected_tables", [])),
        latency_ms=latency_ms,
    )


async def run_business_knowledge_recall(query: str, top_k: int = 10) -> StrategyRunResult:
    """Run business knowledge recall and return retrieved doc IDs."""
    from agents.rag.retriever import recall_business_knowledge

    t0 = time.monotonic()
    docs = await asyncio.to_thread(recall_business_knowledge, query, top_k)
    latency_ms = (time.monotonic() - t0) * 1000
    return StrategyRunResult(retrieved_doc_ids=_docs_to_ids(docs), latency_ms=latency_ms)


async def run_agent_knowledge_recall(query: str, top_k: int = 10) -> StrategyRunResult:
    """Run SQL few-shot knowledge recall and return retrieved doc IDs."""
    from agents.rag.retriever import recall_agent_knowledge

    t0 = time.monotonic()
    docs = await asyncio.to_thread(recall_agent_knowledge, query, top_k)
    latency_ms = (time.monotonic() - t0) * 1000
    return StrategyRunResult(retrieved_doc_ids=_docs_to_ids(docs), latency_ms=latency_ms)


def route_accuracy(results: list[tuple[str, str]]) -> float:
    """Compute exact route-mode accuracy for complex query routing."""
    if not results:
        return 0.0
    return sum(1 for actual, expected in results if actual == expected) / len(results)


def run_complex_route_eval_case(case: dict) -> dict:
    """Evaluate one complex-query routing case without calling external LLMs."""
    from agents.flow.complex_query import classify_query_complexity

    decision = classify_query_complexity(
        query=case.get("query", ""),
        selected_tables=case.get("tables", []),
        relationships=case.get("relationships", []),
        route_signal=case.get("route_signal"),
    )
    expected = case.get("expected_route", "")
    return {
        "query": case.get("query", ""),
        "actual_route": decision.route_mode,
        "expected_route": expected,
        "passed": decision.route_mode == expected,
        "reason": decision.reason,
    }
