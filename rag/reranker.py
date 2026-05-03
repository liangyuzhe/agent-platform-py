"""Cross-Encoder reranker for reordering retrieved documents."""

from __future__ import annotations

from langchain_core.documents import Document


class CrossEncoderReranker:
    """Rerank a list of documents given a query using a Cross-Encoder model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to ``"BAAI/bge-reranker-v2-m3"``.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``, etc.).
        When *None* the library picks the best available device.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            model_name,
            max_length=512,
            **({"device": device} if device else {}),
        )

    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """Score and reorder *docs* by relevance to *query*.

        Parameters
        ----------
        query:
            The search query.
        docs:
            Candidate documents to rerank.
        top_k:
            If provided, only the top *top_k* documents are returned.
            When *None*, all documents are returned in reranked order.

        Returns
        -------
        list[Document]
            Documents sorted by descending relevance score.
        """
        if not docs:
            return []

        # Build (query, passage) pairs for the Cross-Encoder
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs)

        # Attach scores to metadata and sort descending
        scored: list[tuple[float, Document]] = []
        for score, doc in zip(scores, docs):
            doc.metadata["rerank_score"] = float(score)
            scored.append((float(score), doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = [doc for _, doc in scored]
        if top_k is not None:
            results = results[:top_k]

        return results
