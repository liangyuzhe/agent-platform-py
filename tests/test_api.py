"""Tests for API endpoints using FastAPI TestClient."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked lifespan."""
    from agents.api.app import app
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestChatEndpoints:
    """Test /api/chat endpoints."""

    @patch("agents.api.routers.chat.get_chat_model")
    def test_chat_test_returns_answer(self, mock_get_model, client):
        mock_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Hello!"
        mock_model.ainvoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        resp = client.post("/api/chat/test", json={
            "question": "hi",
            "history": [],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["question"] == "hi"
        assert data["answer"] == "Hello!"

    @patch("agents.api.routers.chat.get_chat_model")
    def test_chat_test_with_history(self, mock_get_model, client):
        mock_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "I remember."
        mock_model.ainvoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        resp = client.post("/api/chat/test", json={
            "question": "remember me?",
            "history": [{"role": "user", "content": "I am Alice"}],
        })
        assert resp.status_code == 200
        assert resp.json()["answer"] == "I remember."

    def test_chat_test_missing_question(self, client):
        resp = client.post("/api/chat/test", json={"history": []})
        assert resp.status_code == 422


class TestRAGEndpoints:
    """Test /api/rag endpoints."""

    @patch("agents.api.routers.rag.build_rag_chat_graph")
    def test_rag_ask_returns_answer(self, mock_build_graph, client):
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {"answer": "doc says hello"}
        mock_build_graph.return_value = mock_graph

        resp = client.post("/api/rag/ask", json={
            "query": "what does the doc say?",
            "session_id": "test",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "doc says hello"
        assert data["session_id"] == "test"

    @patch("agents.api.routers.rag.build_rag_chat_graph")
    def test_rag_ask_with_rag_mode(self, mock_build_graph, client):
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {"answer": "ok"}
        mock_build_graph.return_value = mock_graph

        resp = client.post("/api/rag/ask", json={
            "query": "test",
            "session_id": "s1",
            "rag_mode": "parent",
        })
        assert resp.status_code == 200
        call_args = mock_graph.ainvoke.call_args[0][0]
        assert call_args["input"]["rag_mode"] == "parent"

    def test_rag_ask_missing_query(self, client):
        resp = client.post("/api/rag/ask", json={"session_id": "s1"})
        assert resp.status_code == 422

    @patch("agents.api.routers.rag.build_rag_chat_graph")
    def test_rag_stream_returns_sse(self, mock_build_graph, client):
        """SSE stream should return events."""
        mock_graph = AsyncMock()

        async def mock_stream_events(input, version):
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": MagicMock(content="Hello")},
            }
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": MagicMock(content=" World")},
            }

        mock_graph.astream_events = mock_stream_events
        mock_build_graph.return_value = mock_graph

        with client.stream("POST", "/api/rag/chat/stream", json={
            "query": "test",
            "session_id": "s1",
        }) as resp:
            assert resp.status_code == 200
            text = ""
            for line in resp.iter_lines():
                text += line
            assert "Hello" in text or "data" in text


class TestDocumentEndpoint:
    """Test /api/document endpoints."""

    def test_document_insert_no_file(self, client):
        resp = client.post("/api/document/insert")
        assert resp.status_code == 422


class TestStaticFiles:
    """Test static file serving."""

    def test_index_html_served(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Financial Copilot" in resp.text
