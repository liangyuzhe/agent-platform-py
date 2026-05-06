"""Async Redis client with lazy initialization."""

from __future__ import annotations

import logging
from typing import Optional

import redis.asyncio as aioredis

from agents.config.settings import get_settings

logger = logging.getLogger(__name__)

_client: Optional[aioredis.Redis] = None


def _build_url() -> str:
    """Build Redis connection URL from application settings."""
    settings = get_settings()
    addr = settings.redis.addr
    host, port = addr.rsplit(":", 1) if ":" in addr else (addr, "6379")
    db = settings.redis.db
    password = settings.redis.password
    if password:
        return f"redis://:{password}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"


async def init_redis() -> aioredis.Redis:
    """Initialize and return the async Redis client.

    Creates a connection pool and verifies connectivity with a PING.
    The client is stored as a module-level singleton for reuse.
    """
    global _client
    if _client is not None:
        return _client

    url = _build_url()
    _client = aioredis.from_url(
        url,
        decode_responses=True,
        max_connections=10,
        socket_timeout=3,
        socket_connect_timeout=5,
        retry_on_timeout=True,
    )

    try:
        await _client.ping()
        logger.info("Async Redis client connected to %s", url.split("@")[-1])
    except Exception as e:
        logger.warning("Failed to connect to Redis: %s (using in-memory fallback)", e)
        await close_redis()
        # Don't raise — allow server to start without Redis

    return _client


def get_redis() -> Optional[aioredis.Redis]:
    """Return the initialized async Redis client, or None if unavailable."""
    return _client


async def close_redis() -> None:
    """Gracefully close the Redis connection pool."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception as e:
            logger.warning("Error closing Redis client: %s", e)
        _client = None
        logger.info("Async Redis client closed")
