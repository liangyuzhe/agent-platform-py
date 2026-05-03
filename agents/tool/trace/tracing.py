"""Tracing initialization for LangSmith and CozeLoop.

Sets up environment variables for LangSmith and creates CozeLoop callback
handlers that can be attached to LangChain/LangGraph calls.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from agents.config.settings import settings

logger = logging.getLogger(__name__)


def init_langsmith() -> None:
    """Enable LangSmith tracing via environment variables.

    LangChain automatically reads these env vars and starts tracing.
    """
    if not settings.langsmith.tracing or not settings.langsmith.api_key:
        logger.info("LangSmith tracing disabled")
        return

    os.environ["LANGSMITH_API_KEY"] = settings.langsmith.api_key
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith.url

    logger.info("LangSmith tracing enabled (endpoint: %s)", settings.langsmith.url)


def get_cozeloop_handler() -> Any | None:
    """Return a CozeLoop callback handler, or None if disabled.

    Usage::

        handler = get_cozeloop_handler()
        if handler:
            result = await graph.ainvoke(input, config={"callbacks": [handler]})
    """
    if not settings.cozeloop.tracing or not settings.cozeloop.api_key:
        logger.info("CozeLoop tracing disabled")
        return None

    try:
        from cozeloop.integration.langchain import CozeLoopCallbackHandler

        handler = CozeLoopCallbackHandler(
            api_key=settings.cozeloop.api_key,
            endpoint=settings.cozeloop.endpoint,
        )
        logger.info("CozeLoop tracing enabled (endpoint: %s)", settings.cozeloop.endpoint)
        return handler
    except ImportError:
        logger.warning(
            "CozeLoop tracing requested but 'cozeloop' package not installed. "
            "Install with: pip install cozeloop"
        )
        return None
    except Exception as e:
        logger.warning("Failed to initialize CozeLoop: %s", e)
        return None


def get_trace_callbacks() -> list[Any]:
    """Return all enabled trace callback handlers."""
    callbacks = []

    handler = get_cozeloop_handler()
    if handler:
        callbacks.append(handler)

    return callbacks


def init_tracing() -> None:
    """Initialize all tracing systems."""
    init_langsmith()
    # CozeLoop is lazy-loaded per request via get_trace_callbacks()
    if settings.cozeloop.tracing:
        logger.info("CozeLoop tracing configured (will attach per-request)")
