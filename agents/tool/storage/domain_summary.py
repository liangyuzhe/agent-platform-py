"""Domain summary persistence: MySQL + Redis + in-memory cache.

Stores a compact LLM-generated summary of all database table schemas,
used by the intent classifier to avoid hardcoded prompts.

Three-tier lookup: in-memory → Redis → MySQL.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from agents.config.settings import settings

logger = logging.getLogger(__name__)

_REDIS_KEY = "domain:summary"
_REDIS_TTL = 86400  # 24 hours

# In-memory cache (module-level singleton)
_summary_text: str | None = None


# ---------------------------------------------------------------------------
# MySQL DDL / CRUD via MCP
# ---------------------------------------------------------------------------

def _parse_json_result(raw: str) -> list[dict]:
    """Parse MCP query result JSON, stripping trailing timing info."""
    clean = raw
    idx = raw.rfind("]Query execution time:")
    if idx != -1:
        clean = raw[: idx + 1]
    return json.loads(clean)


async def ensure_domain_summary_table() -> None:
    """Create the domain_summary table if it doesn't exist."""
    from agents.tool.sql_tools.mcp_client import execute_sql

    ddl = (
        "CREATE TABLE IF NOT EXISTS domain_summary ("
        "  id INT PRIMARY KEY DEFAULT 1,"
        "  summary_text TEXT NOT NULL,"
        "  updated_at DATETIME NOT NULL"
        ")"
    )
    await execute_sql(ddl)
    logger.info("domain_summary table ensured")


async def load_domain_summary() -> str | None:
    """Load domain summary from MySQL. Returns None if not found."""
    from agents.tool.sql_tools.mcp_client import execute_sql

    try:
        raw = await execute_sql(
            "SELECT summary_text FROM domain_summary WHERE id = 1"
        )
        rows = _parse_json_result(raw)
        if rows:
            return rows[0].get("summary_text") or rows[0].get("SUMMARY_TEXT")
    except Exception as e:
        logger.warning("Failed to load domain summary from MySQL: %s", e)
    return None


async def save_domain_summary(text: str) -> None:
    """Save domain summary to MySQL + Redis + in-memory."""
    global _summary_text
    _summary_text = text

    # MySQL (upsert)
    from agents.tool.sql_tools.mcp_client import execute_sql

    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        sql = (
            f"INSERT INTO domain_summary (id, summary_text, updated_at) "
            f"VALUES (1, '{text.replace(chr(39), chr(39)+chr(39))}', '{now}') "
            f"ON DUPLICATE KEY UPDATE summary_text = VALUES(summary_text), updated_at = VALUES(updated_at)"
        )
        await execute_sql(sql)
        logger.info("Domain summary saved to MySQL (%d chars)", len(text))
    except Exception as e:
        logger.warning("Failed to save domain summary to MySQL: %s", e)

    # Redis
    try:
        from agents.tool.storage.redis_client import get_redis
        redis = get_redis()
        await redis.set(_REDIS_KEY, text, ex=_REDIS_TTL)
        logger.info("Domain summary cached in Redis")
    except Exception as e:
        logger.warning("Failed to cache domain summary in Redis: %s", e)


async def get_domain_summary() -> str:
    """Get domain summary with three-tier lookup: memory → Redis → MySQL.

    Returns empty string if no summary exists anywhere.
    """
    global _summary_text

    # 1. In-memory
    if _summary_text:
        return _summary_text

    # 2. Redis
    try:
        from agents.tool.storage.redis_client import get_redis
        redis = get_redis()
        cached = await redis.get(_REDIS_KEY)
        if cached:
            _summary_text = cached
            logger.info("Domain summary loaded from Redis")
            return _summary_text
    except Exception:
        pass

    # 3. MySQL
    text = await load_domain_summary()
    if text:
        _summary_text = text
        # Backfill Redis
        try:
            from agents.tool.storage.redis_client import get_redis
            redis = get_redis()
            await redis.set(_REDIS_KEY, text, ex=_REDIS_TTL)
        except Exception:
            pass
        logger.info("Domain summary loaded from MySQL")
        return _summary_text

    return ""
