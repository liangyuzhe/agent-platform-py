"""Document parser that dispatches loading + splitting by file extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.schema import Document

from agents.tool.document.loader import get_loader
from agents.tool.document.splitter import get_splitter


def parse_document(
    file_path: str | Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Load a document, split it into chunks, and return a list of ``Document`` objects.

    Parameters
    ----------
    file_path:
        Path to a single file or directory.  The loader is chosen
        automatically based on the file extension (see :mod:`loader`).
    chunk_size:
        Override the default splitter chunk size (1000).
    chunk_overlap:
        Override the default splitter overlap (200).

    Returns
    -------
    list[Document]
        A flat list of document chunks, each with ``page_content`` and ``metadata``.
    """
    loader = get_loader(file_path)
    raw_docs: list[Document] = loader.load()

    kwargs: dict[str, Any] = {}
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        kwargs["chunk_overlap"] = chunk_overlap

    splitter = get_splitter(**kwargs)
    chunks = splitter.split_documents(raw_docs)

    # Ensure every chunk carries the source path in metadata.
    source = str(Path(file_path).resolve())
    for doc in chunks:
        doc.metadata.setdefault("source", source)

    return chunks
