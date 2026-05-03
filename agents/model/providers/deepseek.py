"""DeepSeek provider for chat models."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from agents.config.settings import settings
from agents.model.chat_model import register_chat_model


_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


def _create_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.deepseek.chat_model,
        api_key=settings.deepseek.key,
        base_url=_DEEPSEEK_BASE_URL,
    )


def init() -> None:
    """Register the DeepSeek chat model."""
    register_chat_model("deepseek", _create_chat_model)
