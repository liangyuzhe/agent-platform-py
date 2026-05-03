"""Chat model factory with registry pattern."""

from __future__ import annotations

from typing import Callable, Optional

from langchain_core.language_models import BaseChatModel

from agents.config.settings import settings

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_chat_model_registry: dict[str, Callable[[], BaseChatModel]] = {}


def register_chat_model(name: str, factory: Callable[[], BaseChatModel]) -> None:
    """Register a chat model factory under *name*."""
    _chat_model_registry[name] = factory


def get_chat_model(name: Optional[str] = None) -> BaseChatModel:
    """Return a chat model instance.

    If *name* is ``None`` the default model type from settings is used.
    """
    if name is None:
        name = settings.chat_model_type
    factory = _chat_model_registry.get(name)
    if factory is None:
        raise ValueError(f"Unsupported chat model type: {name}")
    return factory()


# ---------------------------------------------------------------------------
# Provider initialisation
# ---------------------------------------------------------------------------

def init_chat_models() -> None:
    """Register all configured chat-model providers."""
    from agents.model.providers import ark, deepseek, gemini, openai, qwen  # noqa: F401

    ark.init()
    openai.init()
    deepseek.init()
    qwen.init()
    gemini.init()
