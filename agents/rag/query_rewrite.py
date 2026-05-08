"""Query rewriting with conversation context."""

from __future__ import annotations

from typing import Union

from agents.config.settings import settings
from agents.model.chat_model import get_chat_model
from agents.tool.trace.tracing import child_trace_config


_REWRITE_SYSTEM_PROMPT = """\
You are a query rewriting assistant.  Given a conversation summary, recent \
chat history, and a new user query, produce a single **standalone search \
query** that captures the user's true information need.

Rules:
- Output ONLY the rewritten query -- no explanations, no punctuation noise.
- Incorporate relevant context from the history (pronouns, references, etc.).
- Keep it concise (one or two sentences max).
- If the query is already self-contained, return it unchanged.
"""


def _format_history(history: Union[str, list[dict]]) -> str:
    """Format history into a string for the prompt."""
    if isinstance(history, str):
        return history
    lines = []
    for h in history:
        role = h.get("role", "user")
        content = h.get("content", "")
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


async def rewrite_query(
    summary: str,
    history: Union[str, list[dict]],
    query: str,
    config: dict | None = None,
) -> str:
    """Rewrite *query* into a standalone search query using conversation context.

    Parameters
    ----------
    summary:
        A condensed summary of the entire conversation so far.
    history:
        The most recent chat history as a string or list of message dicts.
    query:
        The latest user query that may contain ambiguous references.

    Returns
    -------
    str
        A rewritten, self-contained query suitable for retrieval.
    """
    llm = get_chat_model(settings.chat_model_type)

    history_text = _format_history(history)
    user_message = (
        f"## Conversation summary\n{summary}\n\n"
        f"## Recent history\n{history_text}\n\n"
        f"## New query\n{query}"
    )

    messages = [
        ("system", _REWRITE_SYSTEM_PROMPT),
        ("human", user_message),
    ]

    response = await llm.ainvoke(
        messages,
        config=child_trace_config(
            config,
            "query_rewrite.llm",
            tags=["llm", "query_rewrite"],
        ),
    )
    return response.content.strip()
