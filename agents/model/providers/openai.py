"""OpenAI provider for chat and embedding models."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from agents.config.settings import settings
from agents.model.chat_model import register_chat_model


def _create_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai.chat_model,
        api_key=settings.openai.api_key,
    )


def init() -> None:
    """Register the OpenAI chat model."""
    register_chat_model("openai", _create_chat_model)
