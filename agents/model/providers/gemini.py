"""Gemini provider for chat models.

Tries to use ``langchain_google_genai.ChatGoogleGenerativeAI`` first.  Falls
back to ``ChatOpenAI`` with a Gemini-compatible base URL if the Google package
is not installed.
"""

from __future__ import annotations

from agents.config.settings import settings
from agents.model.chat_model import register_chat_model


def _create_chat_model():
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini.chat_model,
            google_api_key=settings.gemini.key,
        )
    except ImportError:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.gemini.chat_model,
            api_key=settings.gemini.key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            request_timeout=60,
            max_retries=2,
        )


def init() -> None:
    """Register the Gemini chat model."""
    register_chat_model("gemini", _create_chat_model)
