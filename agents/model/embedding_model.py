"""Embedding model factory with registry pattern."""

from __future__ import annotations

from typing import Callable, Optional

from langchain_core.embeddings import Embeddings

from agents.config.settings import settings

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_embedding_model_registry: dict[str, Callable[[], Embeddings]] = {}


def register_embedding_model(name: str, factory: Callable[[], Embeddings]) -> None:
    """Register an embedding model factory under *name*."""
    _embedding_model_registry[name] = factory


def get_embedding_model(name: Optional[str] = None) -> Embeddings:
    """Return an embedding model instance.

    If *name* is ``None`` the default embedding model type from settings is
    used.
    """
    if name is None:
        name = settings.embedding_model_type
    factory = _embedding_model_registry.get(name)
    if factory is None:
        raise ValueError(f"Unsupported embedding model type: {name}")
    return factory()


# ---------------------------------------------------------------------------
# Provider initialisation
# ---------------------------------------------------------------------------

def init_embedding_models() -> None:
    """Register all configured embedding-model providers."""
    from agents.model.providers import ark, qwen  # noqa: F401

    ark.init_embedding()
    qwen.init_embedding()
