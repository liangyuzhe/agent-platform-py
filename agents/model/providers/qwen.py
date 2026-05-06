"""Qwen (DashScope) provider for chat and embedding models."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from agents.config.settings import settings
from agents.model.chat_model import register_chat_model
from agents.model.embedding_model import register_embedding_model


def _create_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.qwen.chat_model,
        api_key=settings.qwen.key,
        base_url=settings.qwen.base_url,
        request_timeout=60,
        max_retries=2,
    )


def _create_embedding_model():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.qwen.embedding_model,
        api_key=settings.qwen.key,
        base_url=settings.qwen.base_url,
    )


def init() -> None:
    """Register the Qwen chat model."""
    register_chat_model("qwen", _create_chat_model)


def init_embedding() -> None:
    """Register the Qwen embedding model."""
    register_embedding_model("qwen", _create_embedding_model)
