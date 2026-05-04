"""SQL execution tool registered in the Tool Registry."""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from agents.tool.registry import register

logger = logging.getLogger(__name__)


@tool
async def execute_query(sql: str) -> str:
    """Execute a SQL query against the MySQL database and return results as JSON.

    Only SELECT queries are allowed. DDL/DML (DROP, DELETE, UPDATE, INSERT) is blocked.

    Args:
        sql: The SQL SELECT query to execute.
    """
    from agents.tool.sql_tools.mcp_client import execute_sql

    try:
        result = await execute_sql(sql)
        return result
    except Exception as e:
        logger.warning("execute_query failed: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# Register at module level
register("sql")(execute_query)
