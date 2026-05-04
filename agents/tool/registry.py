"""Unified Tool Registry for LLM tool calling.

Tools are registered by category (e.g. "sql", "finance", "knowledge").
Graph nodes retrieve tools by category and bind them to the LLM.
"""

from __future__ import annotations

import logging
from typing import Callable

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

_tools: dict[str, list[BaseTool]] = {}


def register(category: str) -> Callable[[BaseTool], BaseTool]:
    """Decorator to register a tool under a category.

    Usage::

        @register("sql")
        @tool
        def execute_query(sql: str) -> str:
            \"\"\"Execute a SQL query.\"\"\"
            ...
    """

    def decorator(t: BaseTool) -> BaseTool:
        _tools.setdefault(category, []).append(t)
        logger.debug("Registered tool '%s' in category '%s'", t.name, category)
        return t

    return decorator


def register_tool(category: str, tool: BaseTool) -> None:
    """Register a tool instance directly (non-decorator form)."""
    _tools.setdefault(category, []).append(tool)
    logger.debug("Registered tool '%s' in category '%s'", tool.name, category)


def get_tools(*categories: str) -> list[BaseTool]:
    """Return all tools in the given categories.

    If no categories are specified, returns all registered tools.
    """
    if not categories:
        result = []
        for tools in _tools.values():
            result.extend(tools)
        return result

    result = []
    for cat in categories:
        result.extend(_tools.get(cat, []))
    return result


def list_categories() -> list[str]:
    """Return all registered category names."""
    return list(_tools.keys())


def clear() -> None:
    """Clear all registered tools (useful for testing)."""
    _tools.clear()
