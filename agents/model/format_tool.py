"""Structured-output tool using Pydantic and LangChain ``@tool`` decorator."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field


class FormatOutput(BaseModel):
    """Schema for the format-response tool."""

    answer: str = Field(description="The direct answer to the question")
    is_sql: bool = Field(description="Whether the answer is a SQL query")
    needs_more_tables: bool = Field(
        default=False,
        description="Set to true if key tables are missing and you need more table schemas to generate correct SQL",
    )
    missing_tables: list[str] = Field(
        default_factory=list,
        description="List of table names you need but are not in the provided schemas (e.g. ['t_user', 't_user_role'])",
    )

    @classmethod
    def json_schema(cls) -> dict:
        """Return a JSON-schema dict suitable for tool calling."""
        return cls.model_json_schema()


@tool
def sql_format_response(answer: str, is_sql: bool, needs_more_tables: bool = False, missing_tables: list[str] | None = None) -> dict:
    """Format the SQL output answer and indicate whether it is a SQL query.

    Args:
        answer: The direct answer to the question.
        is_sql: Whether the answer is a SQL query.
        needs_more_tables: Set to true if you need more table schemas to generate correct SQL.
        missing_tables: List of table names you need but are not in the provided schemas.
    """
    return {
        "answer": answer,
        "is_sql": is_sql,
        "needs_more_tables": needs_more_tables,
        "missing_tables": missing_tables or [],
    }


def create_format_tool() -> BaseTool:
    """Create and return the SQL format-response tool."""
    return sql_format_response
