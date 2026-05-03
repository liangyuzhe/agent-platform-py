"""Document loader wrapper mapping file extensions to LangChain loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)

# Maps file extension -> loader factory.
# Each value is a callable that accepts a file path and returns a BaseLoader.
LOADER_MAP: dict[str, Any] = {
    ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
    ".md": lambda p: UnstructuredMarkdownLoader(str(p)),
    ".pdf": lambda p: PyPDFLoader(str(p)),
    ".csv": lambda p: CSVLoader(str(p), encoding="utf-8"),
    ".json": lambda p: JSONLoader(str(p), jq_schema=".", text_content=False),
    ".jsonl": lambda p: JSONLoader(str(p), jq_schema=".", text_content=False),
    ".doc": lambda p: UnstructuredWordDocumentLoader(str(p)),
    ".docx": lambda p: UnstructuredWordDocumentLoader(str(p)),
}


def get_loader(file_path: str | Path) -> Any:
    """Return an appropriate LangChain document loader for *file_path*.

    Parameters
    ----------
    file_path:
        Path to a single file **or** a directory.  For directories a
        ``DirectoryLoader`` is returned that recursively loads ``.txt`` files
        by default.

    Returns
    -------
    A LangChain ``BaseLoader`` instance.

    Raises
    ------
    ValueError
        If the file extension has no registered loader.
    FileNotFoundError
        If *file_path* does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_dir():
        return DirectoryLoader(str(path), glob="**/*", show_progress=True)

    ext = path.suffix.lower()
    factory = LOADER_MAP.get(ext)
    if factory is None:
        raise ValueError(
            f"No loader registered for extension '{ext}'. "
            f"Supported extensions: {', '.join(sorted(LOADER_MAP))}"
        )
    return factory(path)
