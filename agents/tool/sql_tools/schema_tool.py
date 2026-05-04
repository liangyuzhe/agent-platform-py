"""Schema inspection tools registered in the Tool Registry."""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from agents.tool.registry import register

logger = logging.getLogger(__name__)


@tool
async def list_tables() -> str:
    """List all tables in the connected MySQL database.

    Returns a JSON array of table names.
    """
    from agents.tool.sql_tools.mcp_client import execute_sql

    try:
        result = await execute_sql("SHOW TABLES")
        return result
    except Exception as e:
        logger.warning("list_tables failed: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
async def describe_table(table_name: str) -> str:
    """Describe the columns of a MySQL table (name, type, key, comment).

    Args:
        table_name: The name of the table to describe.
    """
    from agents.tool.sql_tools.mcp_client import execute_sql

    try:
        result = await execute_sql(
            "SELECT column_name, column_type, is_nullable, column_key, column_comment "
            "FROM information_schema.columns "
            f"WHERE table_schema = DATABASE() AND table_name = '{table_name}' "
            "ORDER BY ordinal_position"
        )
        return result
    except Exception as e:
        logger.warning("describe_table failed: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# Register at module level
register("sql")(list_tables)
register("sql")(describe_table)
