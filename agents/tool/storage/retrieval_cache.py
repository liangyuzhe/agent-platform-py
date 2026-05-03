"""Retrieval result cache with SHA256-keyed Redis storage and in-memory fallback."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional, Sequence

from agents.tool.storage.redis_client import get_redis

logger = logging.getLogger(__name__)

# Key prefixes and TTL
EMBEDDING_PREFIX = "embedding"
RETRIEVAL_PREFIX = "retrieval"
CACHE_TTL = 60 * 60  # 1 hour in seconds


def _hash_key(text: str) -> str:
    """Compute a SHA256 hex digest for cache keying."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RetrievalCache:
    """Cache for retrieval documents and embedding vectors.

    Uses Redis when available; falls back to an in-memory dict on failure.
    """

    def __init__(self) -> None:
        self._fallback_emb: dict[str, list[float]] = {}
        self._fallback_doc: dict[str, list[dict]] = {}
        self._use_fallback: bool = True

    def _ensure_redis(self) -> bool:
        """Check if Redis is reachable; switch to fallback if not."""
        try:
            client = get_redis()
            client.ping()
            self._use_fallback = False
            return True
        except Exception:
            self._use_fallback = True
            return False

    # ---- Document retrieval cache ----

    def get_retrieval(self, query: str) -> Optional[list[dict]]:
        """Get cached retrieval documents for *query*.

        Returns:
            A list of document dicts, or ``None`` on cache miss.
        """
        h = _hash_key(query)

        if self._use_fallback:
            return self._fallback_doc.get(h)

        try:
            client = get_redis()
            data = client.get(f"{RETRIEVAL_PREFIX}:{h}")
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.warning("Retrieval cache get failed, using fallback: %s", e)
            self._use_fallback = True
            return self._fallback_doc.get(h)

    def set_retrieval(self, query: str, documents: Sequence[dict]) -> None:
        """Cache retrieval documents for *query*."""
        h = _hash_key(query)

        if self._use_fallback:
            self._fallback_doc[h] = list(documents)
            return

        try:
            client = get_redis()
            client.set(
                f"{RETRIEVAL_PREFIX}:{h}",
                json.dumps(list(documents), ensure_ascii=False),
                ex=CACHE_TTL,
            )
        except Exception as e:
            logger.warning("Retrieval cache set failed, using fallback: %s", e)
            self._use_fallback = True
            self._fallback_doc[h] = list(documents)

    # ---- Embedding vector cache ----

    def get_embedding(self, text: str) -> Optional[list[float]]:
        """Get cached embedding vector for *text*.

        Returns:
            A list of floats, or ``None`` on cache miss.
        """
        h = _hash_key(text)

        if self._use_fallback:
            return self._fallback_emb.get(h)

        try:
            client = get_redis()
            data = client.get(f"{EMBEDDING_PREFIX}:{h}")
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.warning("Embedding cache get failed, using fallback: %s", e)
            self._use_fallback = True
            return self._fallback_emb.get(h)

    def set_embedding(self, text: str, vector: Sequence[float]) -> None:
        """Cache embedding vector for *text*."""
        h = _hash_key(text)

        if self._use_fallback:
            self._fallback_emb[h] = list(vector)
            return

        try:
            client = get_redis()
            client.set(
                f"{EMBEDDING_PREFIX}:{h}",
                json.dumps(list(vector)),
                ex=CACHE_TTL,
            )
        except Exception as e:
            logger.warning("Embedding cache set failed, using fallback: %s", e)
            self._use_fallback = True
            self._fallback_emb[h] = list(vector)

    # ---- Housekeeping ----

    def clear(self) -> None:
        """Clear all in-memory fallback caches."""
        self._fallback_emb.clear()
        self._fallback_doc.clear()
