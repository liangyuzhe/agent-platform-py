"""Session store with Redis primary and in-memory fallback."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from agents.config.settings import settings
from agents.tool.memory.session import Session

logger = logging.getLogger(__name__)

# Redis key configuration
SESSION_KEY_PREFIX = "session:memory:"
SESSION_TTL = 24 * 60 * 60  # 24 hours in seconds


class SessionStore:
    """Session store backed by Redis with in-memory fallback."""

    def __init__(self) -> None:
        self._fallback: dict[str, Session] = {}
        self._use_fallback: bool = True
        self._redis = None

    def _get_redis(self):
        """Attempt to get a Redis client, falling back to memory if unavailable."""
        if self._redis is not None:
            return self._redis
        try:
            import redis

            addr = settings.redis.addr
            host, port = addr.rsplit(":", 1) if ":" in addr else (addr, "6379")
            self._redis = redis.Redis(
                host=host,
                port=int(port),
                db=settings.redis.db,
                password=settings.redis.password or None,
                decode_responses=True,
                socket_timeout=3,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            self._redis.ping()
            self._use_fallback = False
            logger.info("Session store connected to Redis")
            return self._redis
        except Exception as e:
            logger.warning("Redis unavailable, using in-memory session store: %s", e)
            self._use_fallback = True
            return None

    def _make_key(self, session_id: str) -> str:
        return f"{SESSION_KEY_PREFIX}{session_id}"

    def get(self, session_id: str) -> Session:
        """Retrieve a session by ID. Returns a new session if not found."""
        if self._use_fallback:
            if session_id in self._fallback:
                return self._fallback[session_id]
            return Session(id=session_id)

        try:
            client = self._get_redis()
            if client is None:
                return self.get(session_id)  # retry with fallback
            data = client.get(self._make_key(session_id))
            if data is None:
                return Session(id=session_id)
            return Session.model_validate_json(data)
        except Exception as e:
            logger.error("Redis get failed, falling back to memory: %s", e)
            self._use_fallback = True
            return self.get(session_id)

    def save(self, session_id: str, session: Session) -> None:
        """Save a session. Updates the timestamp before saving."""
        session.updated_at = datetime.now()

        if self._use_fallback:
            self._fallback[session_id] = session
            return

        try:
            client = self._get_redis()
            if client is None:
                self.save(session_id, session)  # retry with fallback
                return
            data = session.model_dump_json()
            client.set(self._make_key(session_id), data, ex=SESSION_TTL)
        except Exception as e:
            logger.error("Redis save failed, falling back to memory: %s", e)
            self._use_fallback = True
            self.save(session_id, session)

    def delete(self, session_id: str) -> None:
        """Delete a session by ID."""
        if self._use_fallback:
            self._fallback.pop(session_id, None)
            return

        try:
            client = self._get_redis()
            if client is not None:
                client.delete(self._make_key(session_id))
        except Exception as e:
            logger.error("Redis delete failed: %s", e)
            self._fallback.pop(session_id, None)


# ---------------------------------------------------------------------------
# Module-level convenience functions (singleton store)
# ---------------------------------------------------------------------------

_store: Optional[SessionStore] = None


def _get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store


def get_session(session_id: str) -> Session:
    """Retrieve a session by ID using the module-level store."""
    return _get_store().get(session_id)


def save_session(session_id: str, session: Session) -> None:
    """Save a session using the module-level store."""
    _get_store().save(session_id, session)
