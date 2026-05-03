"""Query rewriting with conversation context using a Qwen model."""

from __future__ import annotations

from agents.config.settings import settings


def _get_qwen_llm():
    """Return a ChatOpenAI-compatible LLM instance for Qwen."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        openai_api_key=settings.qwen.key,
        openai_api_base=settings.qwen.base_url,
        model_name=settings.qwen.chat_model,
        temperature=0.3,
    )


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


def rewrite_query(summary: str, history: str, query: str) -> str:
    """Rewrite *query* into a standalone search query using conversation context.

    Parameters
    ----------
    summary:
        A condensed summary of the entire conversation so far.
    history:
        The most recent chat history (e.g. last N turns as formatted text).
    query:
        The latest user query that may contain ambiguous references.

    Returns
    -------
    str
        A rewritten, self-contained query suitable for retrieval.
    """
    llm = _get_qwen_llm()

    user_message = (
        f"## Conversation summary\n{summary}\n\n"
        f"## Recent history\n{history}\n\n"
        f"## New query\n{query}"
    )

    messages = [
        ("system", _REWRITE_SYSTEM_PROMPT),
        ("human", user_message),
    ]

    response = llm.invoke(messages)
    return response.content.strip()
