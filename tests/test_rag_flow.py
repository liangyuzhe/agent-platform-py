"""Tests for RAG chat flow."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.flow.state import RAGChatState
from agents.tool.memory.session import Session, Message


class TestRAGChatPreprocess:
    """Test preprocess node in RAG chat flow."""

    @pytest.mark.asyncio
    async def test_preprocess_loads_session(self):
        """Test that preprocess correctly loads session (sync, not async)."""
        from agents.flow.rag_chat import preprocess

        state = {
            "input": {"session_id": "test_user", "query": "hello"},
        }

        result = await preprocess(state)

        assert result["query"] == "hello"
        assert result["session_id"] == "test_user"
        assert "session" in result
        assert isinstance(result["session"], dict)

    @pytest.mark.asyncio
    async def test_preprocess_passes_rag_mode(self):
        """Test that preprocess passes rag_mode from input."""
        from agents.flow.rag_chat import preprocess

        state = {
            "input": {
                "session_id": "test_user",
                "query": "hello",
                "rag_mode": "parent",
            },
        }

        result = await preprocess(state)
        assert result["rag_mode"] == "parent"

    @pytest.mark.asyncio
    async def test_preprocess_default_rag_mode(self):
        """Test that preprocess uses default rag_mode when not specified."""
        from agents.flow.rag_chat import preprocess

        state = {
            "input": {"session_id": "test_user", "query": "hello"},
        }

        result = await preprocess(state)
        assert result["rag_mode"] in ("traditional", "parent")


class TestRAGChatRewrite:
    """Test rewrite node."""

    @pytest.mark.asyncio
    async def test_rewrite_no_history(self):
        """Test rewrite returns original query when no history."""
        from agents.flow.rag_chat import rewrite

        state = {
            "query": "what is AI?",
            "session": {"history": [], "summary": ""},
        }

        result = await rewrite(state)
        assert result["rewritten_query"] == "what is AI?"


class TestRAGChatRetrieve:
    """Test retrieve node."""

    @pytest.mark.asyncio
    async def test_retrieve_traditional_mode(self):
        """Test retrieve uses HybridRetriever in traditional mode."""
        from agents.flow.rag_chat import retrieve

        with patch("agents.flow.rag_chat.get_hybrid_retriever") as mock_retriever:
            mock_instance = MagicMock()
            mock_instance.retrieve.return_value = []
            mock_retriever.return_value = mock_instance

            state = {
                "query": "test query",
                "rewritten_query": "test query",
                "rag_mode": "traditional",
            }

            result = await retrieve(state)
            mock_retriever.assert_called_once()
            assert result["docs"] == []

    @pytest.mark.asyncio
    async def test_retrieve_passes_trace_callbacks(self):
        """Retriever calls should inherit callbacks from graph config."""
        from agents.flow.rag_chat import retrieve

        with patch("agents.flow.rag_chat.get_hybrid_retriever") as mock_retriever:
            mock_instance = MagicMock()
            mock_instance.retrieve.return_value = []
            mock_retriever.return_value = mock_instance

            state = {
                "query": "test query",
                "rewritten_query": "test query",
                "rag_mode": "traditional",
            }

            await retrieve(state, config={"callbacks": ["trace-handler"]})
            assert mock_instance.retrieve.call_args.kwargs["callbacks"] == ["trace-handler"]

    @pytest.mark.asyncio
    async def test_retrieve_parent_mode(self):
        """Test retrieve uses ParentDocumentRetriever in parent mode."""
        from agents.flow.rag_chat import retrieve

        with patch("agents.rag.parent_retriever.ParentDocumentRetriever") as mock_retriever:
            mock_instance = MagicMock()
            mock_instance.retrieve.return_value = []
            mock_retriever.return_value = mock_instance

            state = {
                "query": "test query",
                "rewritten_query": "test query",
                "rag_mode": "parent",
            }

            result = await retrieve(state)
            mock_retriever.assert_called_once()
            assert result["docs"] == []


class TestRAGChatConstructMessages:
    """Test construct_messages node."""

    @pytest.mark.asyncio
    async def test_construct_messages_basic(self):
        """Test message construction with basic query."""
        from agents.flow.rag_chat import construct_messages

        state = {
            "query": "hello",
            "session": {"history": [], "summary": ""},
            "docs": [],
        }

        result = await construct_messages(state)
        assert "messages" in result
        assert len(result["messages"]) == 2  # SystemMessage + HumanMessage
        assert "hello" in result["messages"][1].content

    @pytest.mark.asyncio
    async def test_construct_messages_with_summary(self):
        """Test message construction includes summary."""
        from agents.flow.rag_chat import construct_messages

        state = {
            "query": "hello",
            "session": {"history": [], "summary": "User likes Python"},
            "docs": [],
        }

        result = await construct_messages(state)
        assert "User likes Python" in result["messages"][1].content

    @pytest.mark.asyncio
    async def test_construct_messages_with_docs(self):
        """Test message construction includes retrieved docs."""
        from agents.flow.rag_chat import construct_messages
        from langchain_core.documents import Document

        state = {
            "query": "hello",
            "session": {"history": [], "summary": ""},
            "docs": [Document(page_content="AI is artificial intelligence")],
        }

        result = await construct_messages(state)
        assert "AI is artificial intelligence" in result["messages"][1].content


class TestRAGChatChat:
    """Test chat node."""

    @pytest.mark.asyncio
    @patch("agents.flow.rag_chat.save_session")
    @patch("agents.flow.rag_chat.compress_session")
    @patch("agents.flow.rag_chat.get_chat_model")
    async def test_chat_passes_trace_callbacks(self, mock_get_model, mock_compress, mock_save):
        from agents.flow.rag_chat import chat
        from langchain_core.messages import HumanMessage

        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="answer"))
        mock_get_model.return_value = mock_model

        await chat(
            {
                "messages": [HumanMessage(content="hello")],
                "session": {"id": "s1", "history": []},
                "session_id": "s1",
                "query": "hello",
            },
            config={"callbacks": ["trace-handler"]},
        )

        call_config = mock_model.ainvoke.call_args.kwargs["config"]
        assert call_config["callbacks"] == ["trace-handler"]
        assert call_config["run_name"] == "rag_chat.chat.llm"


class TestBuildRAGChatGraph:
    """Test graph construction."""

    def test_build_graph(self):
        from agents.flow.rag_chat import build_rag_chat_graph
        graph = build_rag_chat_graph()
        assert graph is not None
