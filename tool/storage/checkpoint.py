"""LangGraph Redis checkpointer factory."""

from __future__ import annotations

import logging
from typing import Optional

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from agents.tool.storage.redis_client import get_redis

logger = logging.getLogger(__name__)

_checkpointer: Optional[AsyncRedisSaver] = None


def get_checkpointer() -> AsyncRedisSaver:
    """Return a LangGraph async Redis checkpointer backed by the shared client.

    The checkpointer is a module-level singleton. It reuses the async Redis
    connection managed by :mod:`agents.tool.storage.redis_client`.

    Returns:
        An ``AsyncRedisSaver`` instance ready for use with LangGraph.
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    client = get_redis()
    _checkpointer = AsyncRedisSaver(redis_client=client)
    logger.info("LangGraph AsyncRedisSaver checkpointer created")
    return _checkpointer
