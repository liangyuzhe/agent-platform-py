"""Run retrieval evaluation across multiple strategies.

Compares different retrieval configurations against a labeled dataset
and produces a comparison report.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from agents.eval.metrics import evaluate_single, aggregate_metrics
from agents.eval.reporting import build_report_payload

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for a retrieval strategy variant."""
    name: str
    description: str
    # HybridRetriever params
    retrieve_k: int = 20
    reranker_model: str | None = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 10
    rerank_threshold: float = 0.1
    # Which retriever class to use
    mode: str = "schema_lexical"  # schema_lexical, schema_table_name, traditional, parent, vector_only, es_only


@dataclass
class EvalResult:
    """Result of evaluating one strategy on one query."""
    query: str
    relevant_doc_ids: set[str]
    retrieved_doc_ids: list[str]
    metrics: dict[str, float]
    latency_ms: float


@dataclass
class StrategyReport:
    """Aggregated report for one strategy."""
    config: StrategyConfig
    results: list[EvalResult] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)
    avg_latency_ms: float = 0.0


class _VectorOnlyRetriever:
    """Wrapper that only uses Milvus vector search (no ES BM25)."""

    def __init__(self, retrieve_k: int = 20, reranker_top_k: int = 10):
        from agents.rag.retriever import build_milvus_retriever
        self._retriever = build_milvus_retriever(
            search_kwargs={"search_type": "similarity", "k": retrieve_k},
        )
        self._top_k = reranker_top_k

    def retrieve(self, query: str) -> list[Document]:
        docs = self._retriever.invoke(query)
        return docs[: self._top_k]


class _ESOnlyRetriever:
    """Wrapper that only uses Elasticsearch BM25 (no vector search)."""

    def __init__(self, retrieve_k: int = 20, reranker_top_k: int = 10):
        from agents.rag.retriever import build_es_retriever
        self._retriever = build_es_retriever(
            search_kwargs={"search_type": "similarity", "k": retrieve_k},
        )
        self._top_k = reranker_top_k

    def retrieve(self, query: str) -> list[Document]:
        docs = self._retriever.invoke(query)
        return docs[: self._top_k]


def _tokenize_for_eval(text: str) -> set[str]:
    """Tokenize Chinese/English schema text for lightweight local evaluation."""
    normalized = text.lower()
    tokens = set(re.findall(r"[a-z0-9_]+", normalized))
    tokens.update(ch for ch in normalized if "\u4e00" <= ch <= "\u9fff")
    for size in (2, 3):
        for i in range(0, max(0, len(normalized) - size + 1)):
            chunk = normalized[i : i + size]
            if any("\u4e00" <= ch <= "\u9fff" for ch in chunk):
                tokens.add(chunk)
    return tokens


class _SchemaMetadataRetriever:
    """Local schema retriever based on t_semantic_model-rendered documents.

    This mirrors the current NL2SQL architecture more closely than the legacy
    Milvus ``mysql_schema`` retrieval path: schema authority is MySQL/Redis,
    and evaluation labels are ``schema_<table>`` doc IDs.
    """

    def __init__(self, include_fields: bool = True, top_k: int = 10):
        from agents.eval.dataset_generator import _fetch_all_schema_docs

        self._top_k = top_k
        self._docs = _fetch_all_schema_docs()
        self._indexed = []
        for doc in self._docs:
            table_name = doc.get("table_name", "")
            text = doc.get("text", "") if include_fields else table_name
            self._indexed.append((doc, _tokenize_for_eval(f"{table_name}\n{text}")))

    def retrieve(self, query: str) -> list[Document]:
        query_terms = _tokenize_for_eval(query)
        ranked = []
        for doc, terms in self._indexed:
            overlap = query_terms & terms
            if not overlap:
                continue
            score = len(overlap) / max(len(query_terms), 1)
            ranked.append((score, doc))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            Document(
                page_content=doc.get("text", ""),
                metadata={
                    "doc_id": doc.get("doc_id", doc.get("pk", "")),
                    "table_name": doc.get("table_name", ""),
                    "retriever_source": "schema_metadata",
                },
            )
            for _, doc in ranked[: self._top_k]
        ]


def _build_retriever(config: StrategyConfig):
    """Build a retriever instance from strategy config."""
    if config.mode == "schema_lexical":
        return _SchemaMetadataRetriever(include_fields=True, top_k=config.reranker_top_k)
    elif config.mode == "schema_table_name":
        return _SchemaMetadataRetriever(include_fields=False, top_k=config.reranker_top_k)
    elif config.mode == "parent":
        from agents.rag.parent_retriever import ParentDocumentRetriever
        return ParentDocumentRetriever(reranker_top_k=config.reranker_top_k)
    elif config.mode == "vector_only":
        return _VectorOnlyRetriever(
            retrieve_k=config.retrieve_k,
            reranker_top_k=config.reranker_top_k,
        )
    elif config.mode == "es_only":
        return _ESOnlyRetriever(
            retrieve_k=config.retrieve_k,
            reranker_top_k=config.reranker_top_k,
        )
    else:
        from agents.rag.retriever import HybridRetriever
        return HybridRetriever(
            retrieve_k=config.retrieve_k,
            reranker_model=config.reranker_model,
            reranker_top_k=config.reranker_top_k,
            rerank_threshold=config.rerank_threshold,
        )


def _extract_doc_ids(docs: list[Document]) -> list[str]:
    """Extract doc_id from retrieved documents."""
    return [d.metadata.get("doc_id", "") for d in docs]


def run_single_query(
    retriever,
    query: str,
    relevant_ids: set[str],
    k_values: list[int],
) -> EvalResult:
    """Run retrieval for a single query and compute metrics."""
    t0 = time.monotonic()
    docs = retriever.retrieve(query)
    latency = (time.monotonic() - t0) * 1000

    retrieved_ids = _extract_doc_ids(docs)
    metrics = evaluate_single(retrieved_ids, relevant_ids, k_values)

    return EvalResult(
        query=query,
        relevant_doc_ids=relevant_ids,
        retrieved_doc_ids=retrieved_ids,
        metrics=metrics,
        latency_ms=latency,
    )


def evaluate_strategy(
    config: StrategyConfig,
    dataset: list[dict],
    k_values: list[int] | None = None,
) -> StrategyReport:
    """Evaluate a single strategy against the full dataset.

    Parameters
    ----------
    config:
        Strategy configuration.
    dataset:
        List of {"query": str, "relevant_doc_ids": list[str]}.
    k_values:
        Cutoff values for Recall@K and NDCG@K.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    logger.info("Evaluating strategy: %s (%d queries)", config.name, len(dataset))
    retriever = _build_retriever(config)

    report = StrategyReport(config=config)

    for i, item in enumerate(dataset):
        query = item["query"]
        relevant_ids = set(item["relevant_doc_ids"])

        try:
            result = run_single_query(retriever, query, relevant_ids, k_values)
            report.results.append(result)
        except Exception as e:
            logger.warning("Query %d failed for %s: %s", i, config.name, e)
            continue

        if (i + 1) % 10 == 0:
            logger.info("  %s: %d/%d queries done", config.name, i + 1, len(dataset))

    if report.results:
        all_metrics = [r.metrics for r in report.results]
        report.aggregate = aggregate_metrics(all_metrics)
        report.avg_latency_ms = sum(r.latency_ms for r in report.results) / len(report.results)

    return report


def run_evaluation(
    dataset_path: str | Path = "eval_dataset.jsonl",
    strategies: list[StrategyConfig] | None = None,
    k_values: list[int] | None = None,
    output_path: str | Path = "eval_report.json",
) -> list[StrategyReport]:
    """Run full evaluation across all strategies.

    Parameters
    ----------
    dataset_path:
        Path to the JSONL evaluation dataset.
    strategies:
        List of strategy configs to compare. Uses defaults if None.
    k_values:
        Cutoff values for metrics.
    output_path:
        Where to save the JSON report.
    """
    # Load dataset
    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))

    if not dataset:
        logger.error("Empty dataset: %s", dataset_path)
        return []

    logger.info("Loaded %d evaluation queries from %s", len(dataset), dataset_path)

    # Default strategies to compare. Schema datasets use schema_<table> labels,
    # so default to current NL2SQL schema metadata retrieval instead of legacy
    # document-vector retrievers that may load heavy rerankers.
    if strategies is None:
        strategies = [
            StrategyConfig(
                name="schema_lexical",
                description="Local lexical schema retrieval over t_semantic_model fields/business metadata",
                retrieve_k=20,
                reranker_top_k=10,
                mode="schema_lexical",
            ),
            StrategyConfig(
                name="schema_table_name",
                description="Local lexical schema retrieval over table names only",
                retrieve_k=20,
                reranker_top_k=10,
                mode="schema_table_name",
            ),
        ]
        if os.environ.get("ENABLE_LEGACY_EVAL_RETRIEVERS") == "1":
            strategies.extend([
                StrategyConfig(
                    name="hybrid_no_rerank",
                    description="Legacy document Hybrid (Milvus + ES BM25) + RRF, no reranker",
                    retrieve_k=20,
                    reranker_model=None,
                    reranker_top_k=10,
                    mode="traditional",
                ),
                StrategyConfig(
                    name="vector_only",
                    description="Legacy Milvus vector search only (dense)",
                    retrieve_k=20,
                    reranker_model=None,
                    reranker_top_k=10,
                    mode="vector_only",
                ),
                StrategyConfig(
                    name="es_only",
                    description="Legacy Elasticsearch BM25 only (sparse)",
                    retrieve_k=20,
                    reranker_model=None,
                    reranker_top_k=10,
                    mode="es_only",
                ),
            ])

    # Run evaluation for each strategy
    reports = []
    for config in strategies:
        report = evaluate_strategy(config, dataset, k_values)
        reports.append(report)

    # Save report
    _save_report(reports, output_path, dataset_path)

    # Print summary table
    _print_summary(reports)

    return reports


def _save_report(reports: list[StrategyReport], output_path: str | Path, dataset_path: str | Path = "") -> None:
    """Save evaluation report as JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = build_report_payload(reports, dataset_path=dataset_path)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("Report saved to %s", output)


def _print_summary(reports: list[StrategyReport]) -> None:
    """Print a comparison table to stdout."""
    if not reports:
        return

    # Collect all metric keys
    metric_keys = list(reports[0].aggregate.keys())
    if not metric_keys:
        print("No results to report.")
        return

    # Header
    col_w = 18
    name_w = 22
    header = f"{'Strategy':<{name_w}}" + "".join(f"{k:>{col_w}}" for k in metric_keys) + f"{'Latency(ms)':>{col_w}}"
    print("\n" + "=" * len(header))
    print("RAG Retrieval Evaluation Report")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Rows
    for report in reports:
        row = f"{report.config.name:<{name_w}}"
        for k in metric_keys:
            val = report.aggregate.get(k, 0)
            row += f"{val:>{col_w}.4f}"
        row += f"{report.avg_latency_ms:>{col_w}.1f}"
        print(row)

    print("-" * len(header))
    print(f"{'Queries evaluated':<{name_w}}", len(reports[0].results) if reports else 0)
    print()


def format_detail_report(reports: list[StrategyReport], top_n: int = 5) -> str:
    """Format a detailed report showing worst-performing queries per strategy."""
    lines = []
    for report in reports:
        lines.append(f"\n--- {report.config.name}: {report.config.description} ---")

        # Show queries with worst recall@5
        sorted_results = sorted(report.results, key=lambda r: r.metrics.get("recall@5", 0))
        lines.append(f"Worst {top_n} queries (by recall@5):")
        for r in sorted_results[:top_n]:
            lines.append(f"  Query: {r.query}")
            lines.append(f"    Relevant: {r.relevant_doc_ids}")
            lines.append(f"    Retrieved: {r.retrieved_doc_ids[:5]}")
            lines.append(f"    recall@5={r.metrics.get('recall@5', 0):.2f}, mrr={r.metrics.get('mrr', 0):.2f}")

    return "\n".join(lines)
