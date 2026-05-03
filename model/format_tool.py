"""Structured-output tool using Pydantic and LangChain ``@tool`` decorator."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field


class FormatOutput(BaseModel):
    """Schema for the format-response tool."""

    answer: str = Field(description="The direct answer to the question")
    is_sql: bool = Field(description="Whether the answer is a SQL query")

    @classmethod
    def json_schema(cls) -> dict:
        """Return a JSON-schema dict suitable for tool calling."""
        return cls.model_json_schema()


@tool
def format_response(answer: str, is_sql: bool) -> dict:
    """Format the output answer and indicate whether it is a SQL query.

    Args:
        answer: The direct answer to the question.
        is_sql: Whether the answer is a SQL query.
    """
    return {"answer": answer, "is_sql": is_sql}


def create_format_tool() -> BaseTool:
    """Create and return the format-response tool."""
    return format_response
