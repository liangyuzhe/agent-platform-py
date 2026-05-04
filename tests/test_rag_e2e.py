"""End-to-end tests for RAG chat flow with mocked dependencies.

Tests the full graph: preprocess → rewrite → retrieve → construct_messages → chat
All external services (LLM, Milvus, ES, Redis, reranker) are mocked.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import AIMessage, HumanMessage

from agents.tool.memory.session import Session, Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(session_id="test", history=None, summary=""):
    """Return a real Session object."""
    return Session(
        id=session_id,
        history=history or [],
        summary=summary,
    )


def _mock_doc(text="test document content"):
    """Return a mock Document."""
    doc = MagicMock()
    doc.page_content = text
    doc.metadata = {"source": "test.txt"}
    return doc


def _mock_llm_response(content="Hello! I'm an AI assistant."):
    """Return a real AIMessage as LLM response."""
    return AIMessage(content=content)


# ---------------------------------------------------------------------------
# Full graph integration tests
# ---------------------------------------------------------------------------

class TestRAGChatGraphE2E:
    """Test the full RAG chat graph with mocked dependencies."""

    @pytest.mark.asyncio
    @patch("agents.rag.query_rewrite.get_chat_model")
    @patch("agents.flow.rag_chat.get_chat_model")
    @patch("agents.flow.rag_chat.get_session")
    @patch("agents.flow.rag_chat.save_session")
    @patch("agents.flow.rag_chat.get_hybrid_retriever")
    async def test_full_graph_produces_answer(
        self, mock_retriever_cls, mock_save, mock_get_session, mock_get_model,
        mock_rewrite_model,
    ):
        """Full graph run should produce an answer."""
        from agents.flow.rag_chat import build_rag_chat_graph

        mock_get_session.return_value = _make_session()

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_mock_doc()]
        mock_retriever_cls.return_value = mock_retriever

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=_mock_llm_response("Test answer"))
        mock_model.invoke = MagicMock(return_value=_mock_llm_response("rewritten"))
        mock_get_model.return_value = mock_model
        mock_rewrite_model.return_value = mock_model

        graph = build_rag_chat_graph()
        result = await graph.ainvoke({
            "input": {"session_id": "s1", "query": "hello"},
        })

        assert "answer" in result
        assert result["answer"] == "Test answer"

    @pytest.mark.asyncio
    @patch("agents.rag.query_rewrite.get_chat_model")
    @patch("agents.flow.rag_chat.get_chat_model")
    @patch("agents.flow.rag_chat.get_session")
    @patch("agents.flow.rag_chat.save_session")
    @patch("agents.flow.rag_chat.get_hybrid_retriever")
    async def test_graph_with_empty_retrieval(
        self, mock_retriever_cls, mock_save, mock_get_session, mock_get_model,
        mock_rewrite_model,
    ):
        """Graph should still produce answer even if retrieval returns no docs."""
        from agents.flow.rag_chat import build_rag_chat_graph

        mock_get_session.return_value = _make_session()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever_cls.return_value = mock_retriever

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=_mock_llm_response("No docs answer"))
        mock_model.invoke = MagicMock(return_value=_mock_llm_response("rewritten"))
        mock_get_model.return_value = mock_model
        mock_rewrite_model.return_value = mock_model

        graph = build_rag_chat_graph()
        result = await graph.ainvoke({
            "input": {"session_id": "s1", "query": "hello"},
        })

        assert result["answer"] == "No docs answer"

    @pytest.mark.asyncio
    @patch("agents.rag.query_rewrite.get_chat_model")
    @patch("agents.flow.rag_chat.get_chat_model")
    @patch("agents.flow.rag_chat.get_session")
    @patch("agents.flow.rag_chat.save_session")
    @patch("agents.flow.rag_chat.get_hybrid_retriever")
    async def test_graph_with_history(
        self, mock_retriever_cls, mock_save, mock_get_session, mock_get_model,
        mock_rewrite_model,
    ):
        """Graph should include history in context."""
        from agents.flow.rag_chat import build_rag_chat_graph

        session = _make_session(
            history=[
                Message(role="user", content="what is Python?"),
                Message(role="assistant", content="Python is a programming language."),
            ],
            summary="User is learning Python",
        )
        mock_get_session.return_value = session

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_mock_doc("Python docs")]
        mock_retriever_cls.return_value = mock_retriever

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=_mock_llm_response("Follow-up answer"))
        mock_model.invoke = MagicMock(return_value=_mock_llm_response("rewritten query"))
        mock_get_model.return_value = mock_model
        mock_rewrite_model.return_value = mock_model

        graph = build_rag_chat_graph()
        result = await graph.ainvoke({
            "input": {"session_id": "s1", "query": "tell me more"},
        })

        assert result["answer"] == "Follow-up answer"
        # Verify the model was called with context including history
        call_args = mock_model.ainvoke.call_args[0][0]
        context = call_args[0].content
        assert "Python" in context

    @pytest.mark.asyncio
    @patch("agents.rag.query_rewrite.get_chat_model")
    @patch("agents.flow.rag_chat.get_chat_model")
    @patch("agents.flow.rag_chat.get_session")
    @patch("agents.flow.rag_chat.save_session")
    @patch("agents.flow.rag_chat.get_hybrid_retriever")
    async def test_graph_llm_error_propagates(
        self, mock_retriever_cls, mock_save, mock_get_session, mock_get_model,
        mock_rewrite_model,
    ):
        """When LLM fails, the error should propagate."""
        from agents.flow.rag_chat import build_rag_chat_graph

        mock_get_session.return_value = _make_session()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever_cls.return_value = mock_retriever

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(side_effect=ConnectionError("API down"))
        mock_model.invoke = MagicMock(return_value=_mock_llm_response("rewritten"))
        mock_get_model.return_value = mock_model
        mock_rewrite_model.return_value = mock_model

        graph = build_rag_chat_graph()

        with pytest.raises(ConnectionError, match="API down"):
            await graph.ainvoke({"input": {"session_id": "s1", "query": "hello"}})

    @pytest.mark.asyncio
    @patch("agents.rag.query_rewrite.get_chat_model")
    @patch("agents.flow.rag_chat.get_chat_model")
    @patch("agents.flow.rag_chat.get_session")
    @patch("agents.flow.rag_chat.save_session")
    @patch("agents.flow.rag_chat.get_hybrid_retriever")
    async def test_graph_rag_mode_parent(
        self, mock_retriever_cls, mock_save, mock_get_session, mock_get_model,
        mock_rewrite_model,
    ):
        """Graph with rag_mode=parent should use ParentDocumentRetriever."""
        from agents.flow.rag_chat import build_rag_chat_graph

        mock_get_session.return_value = _make_session()

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=_mock_llm_response("parent answer"))
        mock_model.invoke = MagicMock(return_value=_mock_llm_response("rewritten"))
        mock_get_model.return_value = mock_model
        mock_rewrite_model.return_value = mock_model

        with patch("agents.rag.parent_retriever.ParentDocumentRetriever") as mock_parent:
            mock_parent_instance = MagicMock()
            mock_parent_instance.retrieve.return_value = [_mock_doc("parent doc")]
            mock_parent.return_value = mock_parent_instance

            graph = build_rag_chat_graph()
            result = await graph.ainvoke({
                "input": {"session_id": "s1", "query": "hello", "rag_mode": "parent"},
            })

        assert result["answer"] == "parent answer"


# ---------------------------------------------------------------------------
# Individual node tests
# ---------------------------------------------------------------------------

class TestChatNode:
    """Test the chat node in isolation."""

    @pytest.mark.asyncio
    @patch("agents.flow.rag_chat.get_chat_model")
    @patch("agents.flow.rag_chat.asyncio.create_task")
    async def test_chat_returns_answer(self, mock_task, mock_get_model):
        from agents.flow.rag_chat import chat

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=_mock_llm_response("Hello!"))
        mock_get_model.return_value = mock_model

        state = {
            "messages": [HumanMessage(content="hi")],
            "query": "hi",
            "session_id": "s1",
            "session": _make_session().model_dump(),
        }

        result = await chat(state)

        assert result["answer"] == "Hello!"
        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    @patch("agents.flow.rag_chat.get_chat_model")
    @patch("agents.flow.rag_chat.asyncio.create_task")
    async def test_chat_updates_session_history(self, mock_task, mock_get_model):
        from agents.flow.rag_chat import chat

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=_mock_llm_response("response"))
        mock_get_model.return_value = mock_model

        session = _make_session().model_dump()
        state = {
            "messages": [HumanMessage(content="question")],
            "query": "question",
            "session_id": "s1",
            "session": session,
        }

        await chat(state)

        assert len(session["history"]) == 2
        assert session["history"][0]["role"] == "user"
        assert session["history"][0]["content"] == "question"
        assert session["history"][1]["role"] == "assistant"
        assert session["history"][1]["content"] == "response"

    @pytest.mark.asyncio
    @patch("agents.flow.rag_chat.get_chat_model")
    async def test_chat_llm_error_propagates(self, mock_get_model):
        from agents.flow.rag_chat import chat

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(side_effect=RuntimeError("LLM crashed"))
        mock_get_model.return_value = mock_model

        state = {
            "messages": [HumanMessage(content="hi")],
            "query": "hi",
            "session_id": "s1",
            "session": _make_session().model_dump(),
        }

        with pytest.raises(RuntimeError, match="LLM crashed"):
            await chat(state)


class TestConstructMessagesNode:
    """Test construct_messages node."""

    @pytest.mark.asyncio
    async def test_construct_with_docs_and_summary(self):
        from agents.flow.rag_chat import construct_messages

        state = {
            "query": "what is Python?",
            "session": {
                "summary": "User learning programming",
                "history": [
                    {"role": "user", "content": "hi"},
                ],
            },
            "docs": [_mock_doc("Python is a language."), _mock_doc("Python was created by Guido.")],
        }

        result = await construct_messages(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        content = result["messages"][0].content
        assert "Python" in content
        assert "learning programming" in content

    @pytest.mark.asyncio
    async def test_construct_without_docs(self):
        from agents.flow.rag_chat import construct_messages

        state = {
            "query": "hello",
            "session": {"summary": "", "history": []},
            "docs": [],
        }

        result = await construct_messages(state)
        assert len(result["messages"]) == 1
        assert "hello" in result["messages"][0].content


class TestRetrieveNode:
    """Test retrieve node."""

    @pytest.mark.asyncio
    @patch("agents.flow.rag_chat.get_hybrid_retriever")
    async def test_retrieve_traditional(self, mock_retriever_cls):
        from agents.flow.rag_chat import retrieve

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_mock_doc("doc1"), _mock_doc("doc2")]
        mock_retriever_cls.return_value = mock_retriever

        state = {"rewritten_query": "test query", "query": "test query", "rag_mode": "traditional"}

        result = await retrieve(state)

        assert len(result["docs"]) == 2
        mock_retriever.retrieve.assert_called_with("test query")

    @pytest.mark.asyncio
    @patch("agents.flow.rag_chat.get_hybrid_retriever")
    async def test_retrieve_falls_back_to_original_query(self, mock_retriever_cls):
        """If no rewritten_query, should use original query."""
        from agents.flow.rag_chat import retrieve

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever_cls.return_value = mock_retriever

        state = {"query": "original", "rag_mode": "traditional"}

        result = await retrieve(state)

        mock_retriever.retrieve.assert_called_with("original")


class TestPreprocessNode:
    """Test preprocess node."""

    @pytest.mark.asyncio
    @patch("agents.flow.rag_chat.get_session")
    async def test_preprocess_loads_session(self, mock_get_session):
        from agents.flow.rag_chat import preprocess

        session = _make_session(session_id="s1")
        mock_get_session.return_value = session

        state = {"input": {"session_id": "s1", "query": "hello"}}
        result = await preprocess(state)

        assert result["session"]["id"] == "s1"
        assert result["query"] == "hello"
        assert result["session_id"] == "s1"
