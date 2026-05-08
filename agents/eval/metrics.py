"""Evaluation metrics for retrieval and NL2SQL quality reports."""

from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@K: fraction of relevant docs found in top-K results.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved document IDs (ranked by relevance).
    relevant_ids:
        Set of ground-truth relevant document IDs.
    k:
        Cutoff rank.

    Returns
    -------
    float in [0, 1]
    """
    if not relevant_ids:
        return 1.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Precision@K: fraction of top-K retrieved docs that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def accuracy_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Accuracy@K: 1 if all relevant docs are found in top-K, else 0.

    This is stricter than Recall@K and useful as a user-facing "query passed"
    indicator for retrieval tasks.
    """
    if not relevant_ids:
        return 1.0
    return 1.0 if relevant_ids.issubset(set(retrieved_ids[:k])) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant doc.

    Returns 0 if no relevant doc is found.
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain@K.

    Uses binary relevance (1 if relevant, 0 otherwise).
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG: all relevant docs at the top
    ideal_k = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))

    if idcg == 0:
        return 1.0  # No relevant docs, perfect score
    return dcg / idcg


def evaluate_single(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute all metrics for a single query.

    Returns a dict with keys like "recall@5", "mrr", "ndcg@5", etc.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results = {"mrr": mrr(retrieved_ids, relevant_ids)}
    for k in k_values:
        results[f"accuracy@{k}"] = accuracy_at_k(retrieved_ids, relevant_ids, k)
        results[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        results[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)

    return results


def aggregate_metrics(all_results: list[dict[str, float]]) -> dict[str, float]:
    """Average metrics across all queries."""
    if not all_results:
        return {}

    keys = all_results[0].keys()
    return {k: sum(r[k] for r in all_results) / len(all_results) for k in keys}
