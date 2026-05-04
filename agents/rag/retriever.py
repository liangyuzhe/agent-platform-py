"""Hybrid retrieval: Milvus (dense) + Elasticsearch BM25 (sparse) + RRF fusion + rerank."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_elasticsearch import ElasticsearchStore
from langchain_milvus import Milvus

from agents.config.settings import settings
from agents.algorithm.rrf import reciprocal_rank_fusion
from agents.rag.reranker import CrossEncoderReranker


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

    # -- internal helpers ---------------------------------------------------

    def _retrieve_milvus(self, query: str) -> list[Document]:
        return self._milvus.invoke(query)

    def _retrieve_es(self, query: str) -> list[Document]:
        return self._es.invoke(query)

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
                except Exception:
                    doc_lists[idx] = []

        # 2. RRF fusion
        fused = reciprocal_rank_fusion(doc_lists, k=60)

        # 3. Cross-Encoder rerank (optional)
        if self._reranker:
            return self._reranker.rerank(query, fused, top_k=k)

        return fused[:k]


# Module-level singleton to avoid reconnecting Milvus/ES on every request
_retriever_instance: HybridRetriever | None = None


def get_hybrid_retriever(
    milvus_collection: str | None = None,
    es_index: str | None = None,
    retrieve_k: int = 5,
    reranker_model: str | None = "BAAI/bge-reranker-v2-m3",
    reranker_top_k: int = 5,
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
        )
    return _retriever_instance
