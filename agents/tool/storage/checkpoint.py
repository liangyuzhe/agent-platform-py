"""LangGraph checkpointer factory.

Creates a checkpointer for LangGraph interrupt/resume support.

To use persistent Redis-backed checkpointing, install RedisJSON:
  docker run -p 6379:6379 redis/redis-stack-server:latest
Then set ``use_redis=True`` or call :func:`get_redis_checkpointer`.
"""

from __future__ import annotations

import logging
from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

_checkpointer: Optional[BaseCheckpointSaver] = None


def get_checkpointer() -> BaseCheckpointSaver:
    """Return an in-memory checkpointer (default).

    Suitable for single-process deployments.  For persistent checkpointing
    across restarts, use :func:`get_redis_checkpointer`.
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    _checkpointer = MemorySaver()
    logger.info("LangGraph MemorySaver checkpointer created")
    return _checkpointer


def get_redis_checkpointer() -> BaseCheckpointSaver:
    """Return a Redis-backed checkpointer (requires RedisJSON module).

    Falls back to ``MemorySaver`` if Redis or RedisJSON is unavailable.
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    try:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver
        from agents.tool.storage.redis_client import get_redis

        client = get_redis()
        _checkpointer = AsyncRedisSaver(redis_client=client)
        logger.info("LangGraph AsyncRedisSaver checkpointer created")
    except Exception as e:
        logger.warning("Redis checkpointer unavailable (%s), using MemorySaver", e)
        _checkpointer = MemorySaver()

    return _checkpointer
