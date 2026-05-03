"""Text splitter factory for chunking documents."""

from __future__ import annotations

from langchain.text_splitter import RecursiveCharacterTextSplitter

_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 200


def get_splitter(
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Return a ``RecursiveCharacterTextSplitter`` configured for general use.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk (default 1000).
    chunk_overlap:
        Number of overlapping characters between consecutive chunks (default 200).

    Returns
    -------
    RecursiveCharacterTextSplitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
