"""SQL executor wrapper providing a synchronous-friendly interface over the MCP client."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from agents.tool.sql_tools.mcp_client import execute_sql, list_tables

logger = logging.getLogger(__name__)


def _run_async(coro):  # type: ignore[no-untyped-def]
    """Run *coro* on the current event loop, or create one if necessary."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an already-running loop (e.g. Jupyter, async agent).
        import nest_asyncio  # type: ignore[import-untyped]

        nest_asyncio.apply()
        return loop.run_until_complete(coro)

    return asyncio.run(coro)


class SQLExecutor:
    """High-level SQL execution helper.

    Uses the MCP MySQL client under the hood but offers a simpler, more
    synchronous API for callers that don't want to deal with async/await.
    """

    # ------------------------------------------------------------------
    # Public sync helpers
    # ------------------------------------------------------------------

    def run(self, sql: str) -> dict[str, Any]:
        """Execute *sql* and return parsed result as a dict.

        Returns
        -------
        dict
            A dictionary with at least ``"raw"`` (the raw string from MCP)
            and ``"data"`` (parsed JSON if possible, else ``None``).
        """
        raw: str = _run_async(execute_sql(sql))
        data: Any = None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
        return {"raw": raw, "data": data}

    def tables(self) -> dict[str, Any]:
        """List all tables in the connected database."""
        return self.run("SHOW TABLES")

    def describe(self, table: str) -> dict[str, Any]:
        """Return the column metadata for *table*."""
        return self.run(f"DESCRIBE `{table}`")

    def count(self, table: str) -> dict[str, Any]:
        """Return the row count for *table*."""
        return self.run(f"SELECT COUNT(*) AS cnt FROM `{table}`")

    def select(
        self,
        table: str,
        columns: str = "*",
        where: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Convenience SELECT builder.

        Parameters
        ----------
        table:
            Target table name.
        columns:
            Comma-separated column list (default ``*``).
        where:
            Optional WHERE clause **without** the ``WHERE`` keyword.
        limit:
            Maximum rows to return (default 100).
        """
        sql = f"SELECT {columns} FROM `{table}`"
        if where:
            sql += f" WHERE {where}"
        sql += f" LIMIT {int(limit)}"
        return self.run(sql)
