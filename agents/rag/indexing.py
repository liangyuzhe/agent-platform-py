"""Document indexing pipeline: load -> preprocess -> split -> store to Milvus + Elasticsearch."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_milvus import Milvus

from agents.config.settings import settings

# ---------------------------------------------------------------------------
# Loader map: file extension -> LangChain document loader class
# ---------------------------------------------------------------------------

LOADER_MAP: dict[str, type] = {}

try:
    from langchain_community.document_loaders import TextLoader
    LOADER_MAP[".txt"] = TextLoader
except ImportError:
    pass

try:
    from langchain_community.document_loaders import PyPDFLoader
    LOADER_MAP[".pdf"] = PyPDFLoader
except ImportError:
    pass

try:
    from langchain_community.document_loaders import UnstructuredHTMLLoader
    LOADER_MAP[".html"] = UnstructuredHTMLLoader
except ImportError:
    pass

try:
    from langchain_community.document_loaders import Docx2txtLoader
    LOADER_MAP[".docx"] = Docx2txtLoader
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Embedding helper
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
            chunk_size=10,
        )
    raise ValueError(f"Unsupported embedding_model_type: {provider!r}")


# ---------------------------------------------------------------------------
# Load & split
# ---------------------------------------------------------------------------

def load_document(file_path: str) -> list[Document]:
    """Load a single document file and return a list of LangChain Documents."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(
            f"Unsupported file extension {ext!r}. "
            f"Supported: {', '.join(sorted(LOADER_MAP))}"
        )

    loader = loader_cls(file_path)
    return loader.load()


def split_documents(
    docs: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """Split documents into smaller chunks for indexing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# Indexing graph (with LLM preprocessing)
# ---------------------------------------------------------------------------

# 长文本阈�值：超过此字数的文档使用父子分块
_LONG_DOC_THRESHOLD = 3000


def build_indexing_graph(
    milvus_uri: str | None = None,
    milvus_collection: str | None = None,
    es_url: str | None = None,
    es_index: str = "knowledge_base",
    source: str = "user_document",
    session_id: str = "",
) -> Callable[[str], Any]:
    """Return an async callable that indexes a document with LLM preprocessing.

    Flow: load -> preprocess (LLM) -> save metadata (MySQL) -> split -> enrich chunks -> store

    Parameters
    ----------
    source:
        Knowledge source type. ``user_document`` uses parent-child chunking for long docs.
    session_id:
        User session ID for document isolation (user_document only).
    """
    _milvus_uri = milvus_uri or f"http://{settings.milvus.addr}"
    _milvus_collection = milvus_collection or settings.milvus.collection_name
    _es_url = es_url or settings.es.address
    embeddings = _get_embeddings()

    milvus_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": _milvus_uri},
        collection_name=_milvus_collection,
    )

    es_store = ElasticsearchStore(
        es_url=_es_url,
        index_name=es_index,
        embedding=embeddings,
    )

    async def _index(file_path: str) -> dict[str, Any]:
        """Load, preprocess, split, and persist a document.

        Returns
        -------
        dict
            ``{"doc_ids": [...], "chunk_count": int}``
        """
        from agents.rag.doc_preprocessor import preprocess_document, enrich_chunk_content
        from agents.tool.storage.doc_metadata import save_doc_metadata

        raw_docs = load_document(file_path)
        full_text = "\n".join(d.page_content for d in raw_docs)
        base_name = os.path.basename(file_path)

        # 1. LLM 预处理：提取元数据、摘要、假设性问题
        preprocess_result = await preprocess_document(full_text, base_name)

        # 2. 保存元数据到 MySQL
        doc_id_prefix = os.path.splitext(base_name)[0]
        save_doc_metadata(
            doc_id=doc_id_prefix,
            filename=base_name,
            source=source,
            session_id=session_id,
            category=preprocess_result.category,
            tags=preprocess_result.tags,
            entities=preprocess_result.entities,
            summary=preprocess_result.summary,
            hypothetical_questions=preprocess_result.hypothetical_questions,
            key_facts=preprocess_result.key_facts,
        )

        # 3. 切块
        chunks = split_documents(raw_docs)

        # 4. 用摘要和假设性问题丰富每个 chunk
        doc_ids = [f"{base_name}_{i}" for i in range(len(chunks))]
        for chunk, cid in zip(chunks, doc_ids):
            chunk.page_content = enrich_chunk_content(
                chunk.page_content,
                preprocess_result.summary,
                preprocess_result.hypothetical_questions,
            )
            chunk.metadata["doc_id"] = cid
            chunk.metadata["source"] = source
            chunk.metadata["file_path"] = file_path
            chunk.metadata["table_name"] = ""
            chunk.metadata["session_id"] = session_id

        # 5. 存储到 Milvus + ES
        milvus_store.add_documents(chunks, ids=doc_ids)
        es_store.add_documents(chunks, ids=doc_ids)

        return {"doc_ids": doc_ids, "chunk_count": len(chunks)}

    return _index


def build_parent_indexing_graph(
    milvus_uri: str | None = None,
    child_collection: str | None = None,
    parent_collection: str | None = None,
    es_url: str | None = None,
    es_index: str = "knowledge_base",
    parent_chunk_size: int = 2048,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 512,
    child_chunk_overlap: int = 64,
    source: str = "user_document",
    session_id: str = "",
) -> Callable[[str], Any]:
    """Return an async callable that indexes with parent-child chunking + LLM preprocessing.

    Long documents get parent-child chunking; short documents fall back to regular chunking.
    """
    _milvus_uri = milvus_uri or f"http://{settings.milvus.addr}"
    _child_collection = child_collection or settings.milvus.collection_name
    _parent_collection = parent_collection or settings.milvus.collection_name
    _es_url = es_url or settings.es.address
    embeddings = _get_embeddings()

    child_milvus = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": _milvus_uri},
        collection_name=_child_collection,
    )
    parent_milvus = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": _milvus_uri},
        collection_name=_parent_collection,
    )
    es_store = ElasticsearchStore(
        es_url=_es_url,
        index_name=es_index,
        embedding=embeddings,
    )

    async def _index(file_path: str) -> dict[str, Any]:
        """Load, preprocess, split (parent→child), and persist a document.

        Returns
        -------
        dict
            ``{"doc_ids": [...], "chunk_count": int, "parent_count": int}``
        """
        from agents.rag.doc_preprocessor import preprocess_document, enrich_chunk_content
        from agents.tool.storage.doc_metadata import save_doc_metadata

        raw_docs = load_document(file_path)
        full_text = "\n".join(d.page_content for d in raw_docs)
        base_name = os.path.basename(file_path)

        # 1. LLM 预处理
        preprocess_result = await preprocess_document(full_text, base_name)

        # 2. 保存元数据到 MySQL
        doc_id_prefix = os.path.splitext(base_name)[0]
        save_doc_metadata(
            doc_id=doc_id_prefix,
            filename=base_name,
            source=source,
            session_id=session_id,
            category=preprocess_result.category,
            tags=preprocess_result.tags,
            entities=preprocess_result.entities,
            summary=preprocess_result.summary,
            hypothetical_questions=preprocess_result.hypothetical_questions,
            key_facts=preprocess_result.key_facts,
        )

        # 3. 判断是否为长文本
        total_chars = sum(len(d.page_content) for d in raw_docs)
        use_parent = total_chars > _LONG_DOC_THRESHOLD

        if use_parent:
            # 父子分块
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=parent_chunk_size,
                chunk_overlap=parent_chunk_overlap,
                length_function=len,
                add_start_index=True,
            )
            parent_chunks = parent_splitter.split_documents(raw_docs)

            parent_ids = [f"{base_name}_parent_{i}" for i in range(len(parent_chunks))]
            for chunk, pid in zip(parent_chunks, parent_ids):
                chunk.metadata["doc_id"] = pid
                chunk.metadata["source"] = source
                chunk.metadata["file_path"] = file_path
                chunk.metadata["table_name"] = ""
                chunk.metadata["session_id"] = session_id

            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=child_chunk_size,
                chunk_overlap=child_chunk_overlap,
                length_function=len,
                add_start_index=True,
            )

            all_children: list[Document] = []
            all_child_ids: list[str] = []
            child_idx = 0
            for parent_chunk, pid in zip(parent_chunks, parent_ids):
                children = child_splitter.split_documents([parent_chunk])
                for child in children:
                    child_id = f"{base_name}_child_{child_idx}"
                    # 用摘要和假设性问题丰富子 chunk
                    child.page_content = enrich_chunk_content(
                        child.page_content,
                        preprocess_result.summary,
                        preprocess_result.hypothetical_questions,
                    )
                    child.metadata["parent_id"] = pid
                    child.metadata["doc_id"] = child_id
                    child.metadata["source"] = source
                    child.metadata["file_path"] = file_path
                    child.metadata["table_name"] = ""
                    child.metadata["session_id"] = session_id
                    all_children.append(child)
                    all_child_ids.append(child_id)
                    child_idx += 1

            parent_milvus.add_documents(parent_chunks, ids=parent_ids)
            child_milvus.add_documents(all_children, ids=all_child_ids)
            es_store.add_documents(all_children, ids=all_child_ids)

            return {
                "doc_ids": all_child_ids,
                "chunk_count": len(all_children),
                "parent_count": len(parent_chunks),
            }
        else:
            # 短文本：普通分块
            chunks = split_documents(raw_docs)
            doc_ids = [f"{base_name}_{i}" for i in range(len(chunks))]
            for chunk, cid in zip(chunks, doc_ids):
                chunk.page_content = enrich_chunk_content(
                    chunk.page_content,
                    preprocess_result.summary,
                    preprocess_result.hypothetical_questions,
                )
                chunk.metadata["doc_id"] = cid
                chunk.metadata["source"] = source
                chunk.metadata["file_path"] = file_path
                chunk.metadata["table_name"] = ""
                chunk.metadata["session_id"] = session_id

            child_milvus.add_documents(chunks, ids=doc_ids)
            es_store.add_documents(chunks, ids=doc_ids)

            return {
                "doc_ids": doc_ids,
                "chunk_count": len(chunks),
                "parent_count": 0,
            }

    return _index
