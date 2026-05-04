"""Parent Document Retriever: retrieve child chunks, expand to parent chunks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_milvus import Milvus

from agents.config.settings import settings
from agents.algorithm.rrf import reciprocal_rank_fusion
from agents.rag.reranker import CrossEncoderReranker


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


class ParentDocumentRetriever:
    """Retrieve via child chunks (small, precise), then expand to parent chunks
    (large, rich context).

    Index structure:
    - Children stored in Milvus + ES with `parent_id` metadata
    - Parents stored in a separate Milvus collection (`rag_parents`)

    Query flow:
    1. Hybrid retrieval on child chunks (Milvus + ES BM25 + RRF)
    2. Cross-Encoder rerank on children
    3. Deduplicate parent IDs from top children
    4. Fetch parent chunks from parent collection
    5. Return parent chunks (ordered by best child relevance)
    """

    def __init__(
        self,
        milvus_uri: str | None = None,
        child_collection: str = "rag_children",
        parent_collection: str = "rag_parents",
        es_url: str | None = None,
        es_index: str = "rag_children",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        reranker_top_k: int = 5,
    ) -> None:
        uri = milvus_uri or getattr(settings, "milvus_uri", None) or "http://localhost:19530"
        url = es_url or getattr(settings, "es_url", None) or "http://localhost:9200"
        embeddings = _get_embeddings()

        self._child_milvus = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": uri},
            collection_name=child_collection,
        )
        self._child_es = ElasticsearchStore(
            es_url=url,
            index_name=es_index,
            embedding=embeddings,
        )
        self._parent_milvus = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": uri},
            collection_name=parent_collection,
        )
        self._reranker = CrossEncoderReranker(model_name=reranker_model)
        self._reranker_top_k = reranker_top_k

    def _retrieve_children_milvus(self, query: str) -> list[Document]:
        retriever = self._child_milvus.as_retriever(
            search_type="similarity", k=20
        )
        return retriever.invoke(query)

    def _retrieve_children_es(self, query: str) -> list[Document]:
        from langchain_elasticsearch import BM25Strategy

        store = ElasticsearchStore(
            es_url=self._child_es.es_url,
            index_name=self._child_es.index_name,
            embedding=self._child_es.embedding,
            strategy=BM25Strategy(),
        )
        retriever = store.as_retriever(search_type="similarity", k=20)
        return retriever.invoke(query)

    def _fetch_parents_by_ids(self, parent_ids: list[str]) -> dict[str, Document]:
        """Fetch parent chunks from the parent Milvus collection by doc_id."""
        if not parent_ids:
            return {}
        # Use Milvus filter to fetch parent docs by their doc_id
        results = self._parent_milvus.similarity_search(
            query="",  # dummy query, we filter by ID
            k=len(parent_ids),
            expr=f'doc_id in {parent_ids}',
        )
        return {doc.metadata.get("doc_id", ""): doc for doc in results}

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Run parent document retrieval.

        Steps:
        1. Parallel retrieval of child chunks from Milvus + ES
        2. RRF fusion on children
        3. Cross-Encoder rerank on children
        4. Extract unique parent IDs from top children
        5. Fetch parent chunks and return them ordered by child relevance
        """
        k = top_k or self._reranker_top_k

        # 1. Parallel child retrieval
        child_lists: list[list[Document]] = [None, None]  # type: ignore[list-item]

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(self._retrieve_children_milvus, query): 0,
                pool.submit(self._retrieve_children_es, query): 1,
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    child_lists[idx] = future.result()
                except Exception:
                    child_lists[idx] = []

        # 2. RRF fusion on children
        fused_children = reciprocal_rank_fusion(child_lists, k=60)

        # 3. Rerank children
        reranked_children = self._reranker.rerank(query, fused_children, top_k=k)

        # 4. Extract parent IDs (preserve order, deduplicate)
        seen_parent_ids: list[str] = []
        seen: set[str] = set()
        for child in reranked_children:
            pid = child.metadata.get("parent_id", child.metadata.get("doc_id", ""))
            if pid and pid not in seen:
                seen.add(pid)
                seen_parent_ids.append(pid)

        if not seen_parent_ids:
            # Fallback: no parent mapping, return children as-is
            return reranked_children

        # 5. Fetch parent chunks
        parent_map = self._fetch_parents_by_ids(seen_parent_ids)

        # Return parents ordered by best child relevance
        parents = []
        for pid in seen_parent_ids:
            if pid in parent_map:
                parents.append(parent_map[pid])

        return parents
