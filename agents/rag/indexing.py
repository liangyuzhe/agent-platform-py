"""Document indexing pipeline: load -> split -> store to Milvus + Elasticsearch."""

from __future__ import annotations

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
        )
    raise ValueError(f"Unsupported embedding_model_type: {provider!r}")


# ---------------------------------------------------------------------------
# Load & split
# ---------------------------------------------------------------------------

def load_document(file_path: str) -> list[Document]:
    """Load a single document file and return a list of LangChain Documents.

    Parameters
    ----------
    file_path:
        Absolute path to the document.  The file extension determines which
        loader is used (see ``LOADER_MAP``).

    Returns
    -------
    list[Document]
        Parsed document pages/sections.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If *file_path* does not exist.
    """
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
    """Split documents into smaller chunks for indexing.

    Parameters
    ----------
    docs:
        Documents returned by :func:`load_document`.
    chunk_size:
        Maximum number of characters per chunk.
    chunk_overlap:
        Number of overlapping characters between consecutive chunks.

    Returns
    -------
    list[Document]
        Chunked documents with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# Indexing graph
# ---------------------------------------------------------------------------

def build_indexing_graph(
    milvus_uri: str | None = None,
    milvus_collection: str = "rag_chunks",
    es_url: str | None = None,
    es_index: str = "rag_chunks",
) -> Callable[[str], dict[str, Any]]:
    """Return a callable that indexes a document into Milvus and Elasticsearch.

    The returned function accepts a single ``file_path`` argument and returns a
    dict with ``doc_ids`` (list[str]) and ``chunk_count`` (int).

    Parameters
    ----------
    milvus_uri:
        Milvus connection URI.  Defaults to ``settings.milvus_uri`` or
        ``"http://localhost:19530"``.
    milvus_collection:
        Name of the Milvus collection.
    es_url:
        Elasticsearch connection URL.  Defaults to ``settings.es_url`` or
        ``"http://localhost:9200"``.
    es_index:
        Name of the Elasticsearch index.
    """
    _milvus_uri = milvus_uri or getattr(settings, "milvus_uri", None) or "http://localhost:19530"
    _es_url = es_url or getattr(settings, "es_url", None) or "http://localhost:9200"
    embeddings = _get_embeddings()

    milvus_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": _milvus_uri},
        collection_name=milvus_collection,
    )

    es_store = ElasticsearchStore(
        es_url=_es_url,
        index_name=es_index,
        embedding=embeddings,
    )

    def _index(file_path: str) -> dict[str, Any]:
        """Load, split, and persist a document to Milvus and ES.

        Returns
        -------
        dict
            ``{"doc_ids": [...], "chunk_count": int}``
        """
        raw_docs = load_document(file_path)
        chunks = split_documents(raw_docs)

        # Generate deterministic IDs from file path + chunk index
        doc_ids = [
            f"{os.path.basename(file_path)}_{i}"
            for i in range(len(chunks))
        ]
        for chunk, cid in zip(chunks, doc_ids):
            chunk.metadata["doc_id"] = cid
            chunk.metadata["source"] = file_path

        # Store in both vector store (Milvus) and keyword store (ES)
        milvus_store.add_documents(chunks, ids=doc_ids)
        es_store.add_documents(chunks, ids=doc_ids)

        return {"doc_ids": doc_ids, "chunk_count": len(chunks)}

    return _index


def build_parent_indexing_graph(
    milvus_uri: str | None = None,
    child_collection: str = "rag_children",
    parent_collection: str = "rag_parents",
    es_url: str | None = None,
    es_index: str = "rag_children",
    parent_chunk_size: int = 2048,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 512,
    child_chunk_overlap: int = 64,
) -> Callable[[str], dict[str, Any]]:
    """Return a callable that indexes a document using parent-child chunking.

    Splits documents into large parent chunks first, then splits each parent
    into small child chunks. Children are stored in Milvus + ES for retrieval
    (with ``parent_id`` metadata). Parents are stored in a separate Milvus
    collection for context expansion.

    Parameters
    ----------
    milvus_uri:
        Milvus connection URI.
    child_collection:
        Milvus collection for child chunks (used for retrieval).
    parent_collection:
        Milvus collection for parent chunks (used for context expansion).
    es_url:
        Elasticsearch connection URL.
    es_index:
        Elasticsearch index name for child chunks.
    parent_chunk_size:
        Size of parent chunks (large, for context).
    parent_chunk_overlap:
        Overlap between parent chunks.
    child_chunk_size:
        Size of child chunks (small, for precise retrieval).
    child_chunk_overlap:
        Overlap between child chunks.
    """
    _milvus_uri = milvus_uri or getattr(settings, "milvus_uri", None) or "http://localhost:19530"
    _es_url = es_url or getattr(settings, "es_url", None) or "http://localhost:9200"
    embeddings = _get_embeddings()

    child_milvus = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": _milvus_uri},
        collection_name=child_collection,
    )
    parent_milvus = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": _milvus_uri},
        collection_name=parent_collection,
    )
    es_store = ElasticsearchStore(
        es_url=_es_url,
        index_name=es_index,
        embedding=embeddings,
    )

    def _index(file_path: str) -> dict[str, Any]:
        """Load, split (parent→child), and persist a document.

        Returns
        -------
        dict
            ``{"doc_ids": [...], "chunk_count": int, "parent_count": int}``
        """
        raw_docs = load_document(file_path)

        # Split into parent chunks (large)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        parent_chunks = parent_splitter.split_documents(raw_docs)

        # Assign parent IDs
        base_name = os.path.basename(file_path)
        parent_ids = [f"{base_name}_parent_{i}" for i in range(len(parent_chunks))]
        for chunk, pid in zip(parent_chunks, parent_ids):
            chunk.metadata["doc_id"] = pid
            chunk.metadata["source"] = file_path

        # Split each parent into child chunks (small)
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
                child.metadata["parent_id"] = pid
                child.metadata["doc_id"] = child_id
                child.metadata["source"] = file_path
                all_children.append(child)
                all_child_ids.append(child_id)
                child_idx += 1

        # Store parents in parent collection
        parent_milvus.add_documents(parent_chunks, ids=parent_ids)

        # Store children in child collection (Milvus + ES)
        child_milvus.add_documents(all_children, ids=all_child_ids)
        es_store.add_documents(all_children, ids=all_child_ids)

        return {
            "doc_ids": all_child_ids,
            "chunk_count": len(all_children),
            "parent_count": len(parent_chunks),
        }

    return _index
