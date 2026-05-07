"""MCP (Model Context Protocol) client for MySQL.

Connects to ``mcp-server-mysql`` via stdio transport and exposes helper
functions for listing tables and executing queries.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters, stdio_client

from agents.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_session: ClientSession | None = None
_read_stream: Any = None
_write_stream: Any = None
_cm: Any = None  # context manager handle


async def _connect() -> ClientSession:
    """Establish a stdio connection to ``mcp-server-mysql`` and return a session."""
    global _session, _read_stream, _write_stream, _cm

    if _session is not None:
        return _session

    # mcp-server-mysql reads config from env vars, not CLI args
    mcp_env = {
        "MYSQL_HOST": os.environ.get("MCP_MYSQL_HOST", settings.mysql.host),
        "MYSQL_PORT": os.environ.get("MCP_MYSQL_PORT", str(settings.mysql.port)),
        "MYSQL_USER": os.environ.get("MCP_MYSQL_USER", settings.mysql.username),
        "MYSQL_PASS": os.environ.get("MCP_MYSQL_PASSWORD", settings.mysql.password),
        "MYSQL_DB": os.environ.get("MCP_MYSQL_DATABASE", settings.mysql.database),
        "MYSQL_CHARSET": os.environ.get("MCP_MYSQL_CHARSET", "utf8mb4_0900_ai_ci"),
    }

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "mcp-server-mysql"],
        env={**os.environ, **mcp_env},
    )

    _cm = stdio_client(server_params)
    _read_stream, _write_stream = await _cm.__aenter__()
    _session = ClientSession(_read_stream, _write_stream)
    await _session.__aenter__()
    await _session.initialize()
    logger.info("MCP MySQL session established.")
    return _session


async def init_mcp_tools() -> ClientSession:
    """Public async initializer -- call once at application startup."""
    return await _connect()


async def execute_sql(sql: str, timeout: float | None = None) -> str:
    """Execute an arbitrary SQL statement via the MCP ``mysql_query`` tool.

    Parameters
    ----------
    sql:
        The SQL statement to execute.
    timeout:
        Maximum seconds to wait for the MCP server to respond.

    Returns
    -------
    str
        The result payload returned by the MCP server (JSON-encoded).

    Raises
    ------
    Exception
        Re-raises after logging if the MCP call fails or returns an error.
    """
    session = await _connect()
    effective_timeout = timeout if timeout is not None else settings.resilience.sql_execution_timeout
    logger.info("MCP execute_sql: %s", sql[:200])
    result = await asyncio.wait_for(
        session.call_tool("mysql_query", {"sql": sql}),
        timeout=effective_timeout,
    )
    # ``result.content`` is a list of content blocks; extract text.
    parts = []
    for block in result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    raw = "".join(parts) if parts else str(result)

    # Check if MCP returned an error
    if result.isError:
        logger.error("MCP execute_sql error: %s", raw[:500])
        raise RuntimeError(raw)

    logger.info("MCP execute_sql OK: %s", raw[:200])
    return raw


async def list_tables() -> str:
    """Return a JSON string listing all tables in the connected database."""
    return await execute_sql("SHOW TABLES")


async def close_mcp() -> None:
    """Gracefully shut down the MCP session and its stdio transport."""
    global _session, _read_stream, _write_stream, _cm

    if _session is not None:
        try:
            await _session.__aexit__(None, None, None)
        except Exception:
            logger.exception("Error closing MCP session")
        _session = None

    if _cm is not None:
        try:
            await _cm.__aexit__(None, None, None)
        except Exception:
            logger.exception("Error closing MCP stdio transport")
        _cm = None
        _read_stream = None
        _write_stream = None

    logger.info("MCP MySQL session closed.")
