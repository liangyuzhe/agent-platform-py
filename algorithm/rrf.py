"""Reciprocal Rank Fusion (RRF) for merging multiple ranked lists.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods", SIGIR 2009.
"""

from __future__ import annotations

from langchain_core.documents import Document


def reciprocal_rank_fusion(
    doc_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """Fuse multiple ranked document lists using Reciprocal Rank Fusion.

    For each document that appears in any input list, an RRF score is
    computed as::

        score(d) = SUM over all lists L containing d: 1 / (k + rank_L(d) + 1)

    where ``rank_L(d)`` is the 0-based rank of *d* in list *L*.

    Documents are then returned sorted by descending RRF score.  When the
    same document (identified by ``page_content``) appears in multiple lists,
    the metadata from the **highest-scoring** occurrence is kept.

    Parameters
    ----------
    doc_lists:
        A list of ranked document lists (each list is sorted by relevance,
        most relevant first).
    k:
        RRF smoothing constant.  Larger values reduce the influence of rank
        differences.  Default ``60``.

    Returns
    -------
    list[Document]
        Documents sorted by descending RRF score.
    """
    # Accumulate scores keyed by document content (used as identity)
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            # Keep the document instance from the highest-scoring occurrence
            if key not in doc_map:
                doc_map[key] = doc

    # Attach fusion scores to metadata
    for key, score in scores.items():
        doc_map[key].metadata["rrf_score"] = score

    # Sort by RRF score descending
    sorted_keys = sorted(scores, key=lambda k_: scores[k_], reverse=True)
    return [doc_map[key] for key in sorted_keys]
