"""Ark (Volcengine) provider for chat and embedding models."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from agents.config.settings import settings
from agents.model.chat_model import register_chat_model
from agents.model.embedding_model import register_embedding_model


_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


def _create_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.ark.chat_model,
        api_key=settings.ark.key,
        base_url=_ARK_BASE_URL,
        request_timeout=60,
        max_retries=2,
    )


def _create_embedding_model():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.ark.embedding_model,
        api_key=settings.ark.key,
        base_url=_ARK_BASE_URL,
    )


def init() -> None:
    """Register the Ark chat model."""
    register_chat_model("ark", _create_chat_model)


def init_embedding() -> None:
    """Register the Ark embedding model."""
    register_embedding_model("ark", _create_embedding_model)
