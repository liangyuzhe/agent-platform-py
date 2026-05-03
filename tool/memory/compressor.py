"""LLM-based history compression for session memory."""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from agents.tool.memory.session import Message, Session

logger = logging.getLogger(__name__)

# Maximum number of history messages before compression triggers
DEFAULT_MAX_HISTORY_LEN = 20

# Number of recent messages to keep after compression
KEEP_RECENT = 6

SUMMARY_PROMPT = """你是一个记忆管理助手。
请将 <previous_summary> 和 <older_messages> 合并为一个新的、精炼的摘要。
要求：
1. 保留核心事实、用户的偏好、以及尚未解决的问题。
2. 丢弃寒暄和冗余的中间过程。
3. 保持摘要的连贯性。

<previous_summary>: {previous_summary}
<older_messages>: {older_messages}
新的摘要："""


def _format_messages(messages: list[Message]) -> str:
    """Format messages into a readable text block."""
    lines = []
    for m in messages:
        lines.append(f"[{m.role}]: {m.content}")
    return "\n".join(lines)


async def compress_session(
    session: Session,
    llm: BaseChatModel,
    max_history_len: int = DEFAULT_MAX_HISTORY_LEN,
    keep_recent: int = KEEP_RECENT,
) -> None:
    """Compress session history using LLM summarization.

    When the history length exceeds *max_history_len*, the older messages are
    merged with the existing summary via the LLM and the history is trimmed to
    the most recent *keep_recent* messages.

    Args:
        session: The session to compress (modified in place).
        llm: A LangChain chat model instance for generating summaries.
        max_history_len: Trigger compression when history exceeds this length.
        keep_recent: Number of recent messages to retain after compression.
    """
    if len(session.history) <= max_history_len:
        return

    # Split: older messages to compress, recent messages to keep
    to_compress = session.history[:-keep_recent]
    session.history = session.history[-keep_recent:]

    # Format older messages for the prompt
    older_text = _format_messages(to_compress)

    prompt = SUMMARY_PROMPT.format(
        previous_summary=session.summary or "(无)",
        older_messages=older_text,
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        session.summary = response.content
        logger.info(
            "Compressed %d messages for session %s, new summary length: %d",
            len(to_compress),
            session.id,
            len(session.summary),
        )
    except Exception as e:
        logger.error("Failed to compress session %s: %s", session.id, e)
        # Restore compressed messages to history on failure
        session.history = to_compress + session.history
        raise
