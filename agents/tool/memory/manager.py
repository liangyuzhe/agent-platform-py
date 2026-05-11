"""Layered memory maintenance helpers."""

from __future__ import annotations

import asyncio
import logging

from langchain_core.language_models import BaseChatModel

from agents.config.settings import settings
from agents.model.chat_model import get_chat_model
from agents.tool.memory.compressor import compress_session
from agents.tool.memory.store import get_session, save_session
from agents.tool.memory.vector_store import index_long_term_memory

logger = logging.getLogger(__name__)


async def maintain_session_memory(
    session_id: str,
    llm: BaseChatModel | None = None,
    max_history_len: int | None = None,
    keep_recent: int | None = None,
) -> None:
    """Maintain short/medium/long-term memory for one session.

    Short-term memory is the recent history kept in ``session.history``.
    Medium-term memory is ``session.summary`` generated from older turns.
    Long-term memory stores compressed archives in the vector collection.
    """
    session = get_session(session_id)
    threshold = max_history_len or settings.memory.summary_max_history_len
    recent_count = keep_recent or settings.memory.summary_keep_recent
    if len(session.history) <= threshold:
        save_session(session_id, session)
        return

    model = llm or get_chat_model(settings.chat_model_type)
    archived_messages = await compress_session(
        session,
        llm=model,
        max_history_len=threshold,
        keep_recent=recent_count,
    )

    if archived_messages:
        doc_id = await asyncio.to_thread(
            index_long_term_memory,
            session_id,
            archived_messages,
            session.summary,
        )
        if doc_id:
            session.preferences["_has_long_term_memory"] = "1"

    save_session(session_id, session)
    logger.info(
        "Maintained memory for session %s: history=%d, summary_chars=%d, archived=%d",
        session_id,
        len(session.history),
        len(session.summary or ""),
        len(archived_messages),
    )


def schedule_memory_maintenance(session_id: str) -> None:
    """Run memory maintenance in the background when an event loop is active."""
    try:
        asyncio.get_running_loop().create_task(maintain_session_memory(session_id))
    except RuntimeError:
        save_session(session_id, get_session(session_id))
