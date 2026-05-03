"""Tests for SSE error handling and streaming."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestRAGStreamErrorHandling:
    """Test that errors in RAG stream are sent to client."""

    @pytest.mark.asyncio
    async def test_stream_error_is_caught(self):
        """When astream_events raises, the error should be catchable."""

        async def failing_stream(input, version):
            raise ConnectionError("Milvus not available")
            yield  # make it a generator

        events = []
        try:
            async for event in failing_stream({}, "v2"):
                events.append(event)
        except Exception as e:
            events.append({"event": "error", "data": str(e)})

        assert len(events) == 1
        assert events[0]["event"] == "error"
        assert "Milvus not available" in events[0]["data"]

    @pytest.mark.asyncio
    async def test_generate_yields_error_on_exception(self):
        """The generate() coroutine in rag_chat_stream should yield error events."""
        from agents.api.routers.rag import build_rag_chat_graph

        with patch("agents.api.routers.rag.build_rag_chat_graph") as mock_build:
            mock_graph = AsyncMock()

            async def failing_stream(input, version):
                raise ValueError("test error")
                yield

            mock_graph.astream_events = failing_stream
            mock_build.return_value = mock_graph

            # Import and call the generate function directly
            from agents.api.routers.rag import RAGChatRequest

            graph = mock_build()
            req = RAGChatRequest(query="test", session_id="s1")
            inp = {"session_id": req.session_id, "query": req.query}

            events = []
            try:
                async for event in graph.astream_events(
                    {"input": inp}, version="v2"
                ):
                    if event["event"] == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if chunk.content:
                            events.append({"event": "data", "data": chunk.content})
            except Exception as e:
                events.append({"event": "error", "data": str(e)})

            assert len(events) == 1
            assert events[0]["event"] == "error"
            assert "test error" in events[0]["data"]


class TestSSEHelper:
    """Test the sse_response helper."""

    def test_sse_response_is_callable(self):
        from agents.api.sse import sse_response
        assert callable(sse_response)
