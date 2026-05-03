"""Tracing initialization for LangSmith and CozeLoop.

Creates callback handlers that are attached to every LangChain/LangGraph
invocation via get_trace_callbacks().
"""

from __future__ import annotations

import logging
import os
from typing import Any

from agents.config.settings import settings

logger = logging.getLogger(__name__)

# CozeLoop client singleton (lazily created, cleaned up on shutdown)
_cozeloop_client: Any = None


def init_langsmith() -> None:
    """Set env vars for LangSmith (also used by LangChainTracer)."""
    if not settings.langsmith.tracing or not settings.langsmith.api_key:
        logger.info("LangSmith tracing disabled")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith.api_key
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith.url

    # Show LangSmith trace send/flush logs in server output
    for name in ("langsmith", "langsmith.client", "langsmith._internal"):
        logging.getLogger(name).setLevel(logging.INFO)

    logger.info("LangSmith tracing enabled (endpoint: %s)", settings.langsmith.url)


def _get_langsmith_handler() -> Any | None:
    """Return a LangChainTracer, or None if disabled."""
    if not settings.langsmith.tracing or not settings.langsmith.api_key:
        return None

    try:
        from langchain_core.tracers.langchain import LangChainTracer
        tracer = LangChainTracer()
        return tracer
    except Exception as e:
        logger.warning("Failed to create LangChainTracer: %s", e)
        return None


def _set_cozeloop_env() -> None:
    """Push CozeLoop settings into env vars so the SDK can read them."""
    cfg = settings.cozeloop
    if cfg.workspace_id:
        os.environ["COZELOOP_WORKSPACE_ID"] = cfg.workspace_id
    if cfg.api_base_url:
        os.environ["COZELOOP_API_BASE_URL"] = cfg.api_base_url
    if cfg.jwt_oauth_client_id:
        os.environ["COZELOOP_JWT_OAUTH_CLIENT_ID"] = cfg.jwt_oauth_client_id
    if cfg.jwt_oauth_private_key:
        # Fix PEM: collapse double newlines that break cryptography parser
        key = cfg.jwt_oauth_private_key.replace("\n\n", "\n")
        os.environ["COZELOOP_JWT_OAUTH_PRIVATE_KEY"] = key
    if cfg.jwt_oauth_public_key_id:
        os.environ["COZELOOP_JWT_OAUTH_PUBLIC_KEY_ID"] = cfg.jwt_oauth_public_key_id


def get_cozeloop_handler() -> Any | None:
    """Return a CozeLoop LangChain callback handler, or None if disabled."""
    global _cozeloop_client

    if not settings.cozeloop.tracing or not settings.cozeloop.jwt_oauth_client_id:
        return None

    try:
        import cozeloop
        from cozeloop.integration.langchain.trace_callback import LoopTracer
    except ImportError:
        logger.warning(
            "CozeLoop tracing requested but 'cozeloop' package not installed. "
            "Install with: pip install cozeloop"
        )
        return None

    try:
        if _cozeloop_client is None:
            _set_cozeloop_env()
            _cozeloop_client = cozeloop.new_client()
            logger.info("CozeLoop client initialized (JWT OAuth)")

        handler = LoopTracer.get_callback_handler(_cozeloop_client)
        return handler
    except Exception as e:
        logger.warning("Failed to initialize CozeLoop: %s", e)
        return None


def get_trace_callbacks() -> list[Any]:
    """Return all enabled trace callback handlers."""
    callbacks = []

    # LangSmith
    langsmith = _get_langsmith_handler()
    if langsmith:
        callbacks.append(langsmith)

    # CozeLoop
    cozelop = get_cozeloop_handler()
    if cozelop:
        callbacks.append(cozelop)

    return callbacks


def close_cozeloop() -> None:
    """Shut down the CozeLoop client (call on app shutdown)."""
    global _cozeloop_client
    if _cozeloop_client is not None:
        try:
            _cozeloop_client.close()
        except Exception:
            pass
        _cozeloop_client = None


def init_tracing() -> None:
    """Initialize all tracing systems."""
    init_langsmith()
    if settings.cozeloop.tracing and settings.cozeloop.jwt_oauth_client_id:
        logger.info("CozeLoop tracing configured (JWT OAuth, will attach per-request)")
    elif settings.cozeloop.tracing:
        logger.warning("CozeLoop tracing enabled but jwt_oauth_client_id is empty")
