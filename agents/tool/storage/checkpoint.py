"""LangGraph checkpointer factory.

Creates a checkpointer for LangGraph interrupt/resume support.

Set ``REDIS_CHECKPOINTER_ENABLED=true`` to use Redis-backed persistent
checkpointing (requires RedisJSON).  Falls back to in-memory ``MemorySaver``
if Redis is unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

_checkpointer: Optional[BaseCheckpointSaver] = None


def get_checkpointer() -> BaseCheckpointSaver:
    """Return a checkpointer based on configuration.

    If ``settings.redis.checkpointer_enabled`` is True and Redis is reachable,
    returns an ``AsyncRedisSaver`` for persistent checkpointing across restarts.
    Otherwise returns an in-memory ``MemorySaver``.
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    from agents.config.settings import settings

    if settings.redis.checkpointer_enabled:
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver
            from agents.tool.storage.redis_client import get_redis

            client = get_redis()
            _checkpointer = AsyncRedisSaver(redis_client=client)
            logger.info("LangGraph AsyncRedisSaver checkpointer created")
            return _checkpointer
        except Exception as e:
            logger.warning("Redis checkpointer unavailable (%s), falling back to MemorySaver", e)

    _checkpointer = MemorySaver()
    logger.info("LangGraph MemorySaver checkpointer created")
    return _checkpointer
