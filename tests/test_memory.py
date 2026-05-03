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
