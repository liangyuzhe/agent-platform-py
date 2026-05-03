"""Document indexing pipeline: load -> split -> store to Milvus + Elasticsearch."""

from __future__ import annotations

import os
from typing import Any, Callable

from langchain.text_splitter import RecursiveCharacterTextSplitter
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
            ark_api_key=settings.ark.api_key,
            model=settings.ark.embedding_model,
        )
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            openai_api_key=settings.openai.api_key,
            model=settings.openai.embedding_model,
        )
    if provider == "qwen":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            openai_api_key=settings.qwen.api_key,
            openai_api_base=settings.qwen.base_url,
            model=settings.qwen.embedding_model,
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
