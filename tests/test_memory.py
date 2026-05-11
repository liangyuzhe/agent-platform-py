"""Tests for memory system (session store, compressor, knowledge)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agents.tool.memory.session import Session, Message
from agents.tool.memory.store import SessionStore, get_session, save_session


class TestSession:
    """Test Session model."""

    def test_create_empty_session(self):
        session = Session(id="test")
        assert session.id == "test"
        assert session.history == []
        assert session.summary == ""
        assert session.entities == {}
        assert session.facts == []
        assert session.preferences == {}

    def test_session_with_history(self):
        messages = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there"),
        ]
        session = Session(id="test", history=messages)
        assert len(session.history) == 2
        assert session.history[0].role == "user"

    def test_session_model_dump(self):
        session = Session(id="test", summary="test summary")
        data = session.model_dump()
        assert data["id"] == "test"
        assert data["summary"] == "test summary"

    def test_session_model_validate(self):
        data = {
            "id": "test",
            "history": [{"role": "user", "content": "hello"}],
            "summary": "",
            "entities": {},
            "facts": [],
            "preferences": {},
        }
        session = Session.model_validate(data)
        assert session.id == "test"
        assert len(session.history) == 1


class TestSessionStore:
    """Test SessionStore with in-memory fallback."""

    def test_get_new_session(self):
        store = SessionStore()
        session = store.get("new_user")
        assert session.id == "new_user"
        assert session.history == []

    def test_save_and_get(self):
        store = SessionStore()
        session = Session(id="user1", summary="test")
        store.save("user1", session)

        retrieved = store.get("user1")
        assert retrieved.id == "user1"
        assert retrieved.summary == "test"

    def test_delete(self):
        store = SessionStore()
        session = Session(id="user1")
        store.save("user1", session)
        store.delete("user1")

        # After delete, should return new empty session
        retrieved = store.get("user1")
        assert retrieved.history == []

    def test_convenience_functions(self):
        """Test module-level get_session and save_session."""
        session = Session(id="test_user", summary="hello")
        save_session("test_user", session)

        retrieved = get_session("test_user")
        assert retrieved.id == "test_user"
        assert retrieved.summary == "hello"


@pytest.mark.asyncio
async def test_compress_session_returns_archived_messages():
    from agents.tool.memory.compressor import compress_session

    session = Session(
        id="s1",
        history=[Message(role="user", content=f"msg-{i}") for i in range(8)],
    )
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="摘要"))

    archived = await compress_session(session, llm=llm, max_history_len=6, keep_recent=2)

    assert [m.content for m in archived] == [f"msg-{i}" for i in range(6)]
    assert [m.content for m in session.history] == ["msg-6", "msg-7"]
    assert session.summary == "摘要"


@pytest.mark.asyncio
async def test_maintain_session_memory_indexes_compressed_archive(monkeypatch):
    from agents.tool.memory.manager import maintain_session_memory

    session = Session(
        id="s1",
        history=[Message(role="user", content=f"msg-{i}") for i in range(8)],
    )
    saved = {}
    indexed = {}
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="摘要"))

    monkeypatch.setattr("agents.tool.memory.manager.get_session", lambda session_id: session)
    monkeypatch.setattr(
        "agents.tool.memory.manager.save_session",
        lambda session_id, sess: saved.update({session_id: sess}),
    )

    async def fake_to_thread(fn, *args, **kwargs):
        indexed["args"] = (fn, args, kwargs)
        return fn(*args, **kwargs)

    monkeypatch.setattr("agents.tool.memory.manager.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr(
        "agents.tool.memory.manager.index_long_term_memory",
        lambda session_id, messages, summary: indexed.update({
            "session_id": session_id,
            "messages": messages,
            "summary": summary,
        }),
    )

    await maintain_session_memory("s1", llm=llm, max_history_len=6, keep_recent=2)

    assert saved["s1"].summary == "摘要"
    assert [m.content for m in saved["s1"].history] == ["msg-6", "msg-7"]
    assert indexed["session_id"] == "s1"
    assert [m.content for m in indexed["messages"]] == [f"msg-{i}" for i in range(6)]
