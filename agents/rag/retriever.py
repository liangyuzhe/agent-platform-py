"""Hybrid retrieval: Milvus (dense) + Elasticsearch BM25 (sparse) + RRF fusion + rerank."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_elasticsearch import ElasticsearchStore
from langchain_milvus import Milvus

from agents.config.settings import settings
from agents.algorithm.rrf import reciprocal_rank_fusion
from agents.rag.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pymilvus compatibility patch
# ---------------------------------------------------------------------------

_milvus_patched = False


def _patch_milvus_connections() -> None:
    """Register MilvusClient connections in the global pymilvus registry.

    langchain-milvus 0.3.x creates a MilvusClient internally, but pymilvus
    2.6.x's MilvusClient no longer registers its connection in the global
    ``pymilvus.connections`` registry.  ``Collection(alias=...)`` then fails
    with *ConnectionNotExistException*.  This one-time patch wraps
    ``MilvusClient.__init__`` to auto-register the handler.
    """
    global _milvus_patched
    if _milvus_patched:
        return

    from pymilvus import MilvusClient, connections

    _orig_init = MilvusClient.__init__

    def _wrapped_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        alias = self._using
        if not connections.has_connection(alias):
            connections._alias_handlers[alias] = self._handler

    MilvusClient.__init__ = _wrapped_init
    _milvus_patched = True


# ---------------------------------------------------------------------------
# Embedding helper (shared with indexing)
# ---------------------------------------------------------------------------

def _get_embeddings():
    """Return an embedding model instance based on the configured provider."""
    provider = settings.embedding_model_type
    if provider == "ark":
        from langchain_community.embeddings import VolcengineEmbeddings
        return VolcengineEmbeddings(
            ark_api_key=settings.ark.key,
            model=settings.ark.embedding_model,
        )
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            openai_api_key=settings.openai.key,
            model=settings.openai.embedding_model,
        )
    if provider == "qwen":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            openai_api_key=settings.qwen.key,
            openai_api_base=settings.qwen.base_url,
            model=settings.qwen.embedding_model,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
        )
    raise ValueError(f"Unsupported embedding_model_type: {provider!r}")


# ---------------------------------------------------------------------------
# Individual retriever builders
# ---------------------------------------------------------------------------

def build_milvus_retriever(
    milvus_uri: str | None = None,
    collection: str | None = None,
    search_kwargs: dict | None = None,
) -> BaseRetriever:
    """Build a dense vector retriever backed by Milvus.

    Parameters
    ----------
    milvus_uri:
        Milvus connection URI.  Falls back to ``settings.milvus.addr`` or
        ``"http://localhost:19530"``.
    collection:
        Milvus collection name.  Defaults to ``settings.milvus.collection_name``.
    search_kwargs:
        Extra keyword arguments forwarded to ``as_retriever``.
    """
    uri = milvus_uri or f"http://{settings.milvus.addr}"
    coll = collection or settings.milvus.collection_name
    embeddings = _get_embeddings()

    # Workaround: langchain-milvus 0.3.x + pymilvus 2.6.x has a bug where
    # MilvusClient's internal connection is not registered in the global
    # pymilvus connections registry, causing Collection() to fail.
    # Monkey-patch MilvusClient to auto-register its handler on init.
    _patch_milvus_connections()

    store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": uri},
        collection_name=coll,
    )
    kwargs = search_kwargs or {"search_type": "similarity", "k": 20}
    return store.as_retriever(**kwargs)


def build_es_retriever(
    es_url: str | None = None,
    index: str | None = None,
    search_kwargs: dict | None = None,
) -> BaseRetriever:
    """Build a sparse (BM25) retriever backed by Elasticsearch.

    Uses ``BM25Strategy`` so that queries are matched purely by keyword
    relevance -- no dense vector search.

    Parameters
    ----------
    es_url:
        Elasticsearch connection URL.  Falls back to ``settings.es.address``.
    index:
        Elasticsearch index name.  Defaults to ``settings.es.index``.
    search_kwargs:
        Extra keyword arguments forwarded to ``as_retriever``.
    """
    from langchain_elasticsearch import BM25Strategy

    url = es_url or settings.es.address
    idx = index or settings.es.index
    embeddings = _get_embeddings()

    store = ElasticsearchStore(
        es_url=url,
        index_name=idx,
        embedding=embeddings,
        strategy=BM25Strategy(),
    )
    kwargs = search_kwargs or {"search_type": "similarity", "k": 20}
    return store.as_retriever(**kwargs)


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Retrieve from Milvus (dense) and Elasticsearch BM25 (sparse) in
    parallel, fuse results with Reciprocal Rank Fusion, then optionally
    rerank with a Cross-Encoder model.

    Parameters
    ----------
    milvus_uri:
        Milvus connection URI.
    milvus_collection:
        Milvus collection name.  Defaults to ``settings.milvus.collection_name``.
    es_url:
        Elasticsearch connection URL.
    es_index:
        Elasticsearch index name.  Defaults to ``settings.es.index``.
    retrieve_k:
        Number of candidates to fetch from each retriever (Milvus / ES).
    reranker_model:
        Sentence-Transformers Cross-Encoder model name for reranking.
        Set to *None* to skip reranking entirely (faster).
    reranker_top_k:
        Default number of results to keep after reranking.
    """

    # Class-level cache for reranker (heavy to load)
    _reranker_cache: dict[str, CrossEncoderReranker] = {}

    def __init__(
        self,
        milvus_uri: str | None = None,
        milvus_collection: str | None = None,
        es_url: str | None = None,
        es_index: str | None = None,
        retrieve_k: int = 5,
        reranker_model: str | None = "BAAI/bge-reranker-v2-m3",
        reranker_top_k: int = 5,
        rerank_threshold: float = 0.1,
    ) -> None:
        search_kwargs = {"search_type": "similarity", "k": retrieve_k}
        self._milvus = build_milvus_retriever(
            milvus_uri=milvus_uri,
            collection=milvus_collection,
            search_kwargs=search_kwargs,
        )
        self._es = build_es_retriever(
            es_url=es_url,
            index=es_index,
            search_kwargs=search_kwargs,
        )
        # Cache the reranker to avoid reloading the model on every request
        self._reranker = None
        if reranker_model:
            if reranker_model not in self._reranker_cache:
                self._reranker_cache[reranker_model] = CrossEncoderReranker(
                    model_name=reranker_model
                )
            self._reranker = self._reranker_cache[reranker_model]
        self._reranker_top_k = reranker_top_k
        self._rerank_threshold = rerank_threshold

    # -- internal helpers ---------------------------------------------------

    def _retrieve_milvus(self, query: str) -> list[Document]:
        import time
        t0 = time.monotonic()
        try:
            docs = self._milvus.invoke(query)
            for d in docs:
                d.metadata["retriever_source"] = "milvus"
            elapsed = time.monotonic() - t0
            logger.info("Milvus retrieve: %d docs in %.2fs", len(docs), elapsed)
            return docs
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.warning("Milvus retrieve failed after %.2fs: %s", elapsed, e)
            return []

    def _retrieve_es(self, query: str) -> list[Document]:
        import time
        t0 = time.monotonic()
        try:
            docs = self._es.invoke(query)
            for d in docs:
                d.metadata["retriever_source"] = "es"
            elapsed = time.monotonic() - t0
            logger.info("ES retrieve: %d docs in %.2fs", len(docs), elapsed)
            return docs
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.warning("ES retrieve failed after %.2fs: %s", elapsed, e)
            return []

    # -- public API ---------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Run hybrid retrieval and return the top *top_k* documents.

        Steps:
        1. Retrieve from Milvus (dense) and ES BM25 (sparse) **in parallel**.
        2. Fuse the two ranked lists with Reciprocal Rank Fusion (RRF).
        3. (Optional) Rerank the fused list with a Cross-Encoder.
        4. Return the top *top_k* results.

        Parameters
        ----------
        query:
            The search query string.
        top_k:
            Number of final results to return.  Defaults to the value passed
            at construction time (``reranker_top_k``).
        """
        k = top_k or self._reranker_top_k

        # 1. Parallel retrieval
        import time as _time
        t0 = _time.monotonic()
        doc_lists: list[list[Document]] = [None, None]  # type: ignore[list-item]

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(self._retrieve_milvus, query): 0,
                pool.submit(self._retrieve_es, query): 1,
            }
            for future in as_completed(futures, timeout=10):
                idx = futures[future]
                try:
                    doc_lists[idx] = future.result(timeout=5)
                except Exception as e:
                    logger.warning("Retriever[%d] exception: %s", idx, e)
                    doc_lists[idx] = []

        # Handle any futures that didn't complete
        for idx, docs in enumerate(doc_lists):
            if docs is None:
                logger.warning("Retriever[%d] did not complete (timeout)", idx)
                doc_lists[idx] = []

        elapsed_total = _time.monotonic() - t0
        logger.info("Hybrid retrieve total: %.2fs", elapsed_total)

        # 2. RRF fusion
        milvus_count = len(doc_lists[0]) if doc_lists[0] else 0
        es_count = len(doc_lists[1]) if doc_lists[1] else 0
        logger.info(
            "Hybrid retrieve: milvus=%d, es=%d docs",
            milvus_count, es_count,
        )
        fused = reciprocal_rank_fusion(doc_lists, k=60)

        # 3. Cross-Encoder rerank (optional) + threshold filter
        if self._reranker:
            reranked = self._reranker.rerank(query, fused, top_k=k)
            results = [
                d for d in reranked
                if d.metadata.get("rerank_score", 0) >= self._rerank_threshold
            ]
            logger.info(
                "After rerank (threshold=%.2f): %d docs kept — %s",
                self._rerank_threshold,
                len(results),
                [
                    (d.metadata.get("table_name", "?"),
                     d.metadata.get("retriever_source", "?"),
                     round(d.metadata.get("rerank_score", 0), 4))
                    for d in results
                ],
            )
            return results

        return fused[:k]


# Module-level singleton to avoid reconnecting Milvus/ES on every request
_retriever_instance: HybridRetriever | None = None


def get_hybrid_retriever(
    milvus_collection: str | None = None,
    es_index: str | None = None,
    retrieve_k: int = 5,
    reranker_model: str | None = "BAAI/bge-reranker-v2-m3",
    reranker_top_k: int = 5,
    rerank_threshold: float = 0.1,
) -> HybridRetriever:
    """Return a cached HybridRetriever singleton."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever(
            milvus_collection=milvus_collection,
            es_index=es_index,
            retrieve_k=retrieve_k,
            reranker_model=reranker_model,
            reranker_top_k=reranker_top_k,
            rerank_threshold=rerank_threshold,
        )
    return _retriever_instance
