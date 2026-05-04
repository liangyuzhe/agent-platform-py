"""Tests for LangGraph checkpointer factory."""

import pytest
from unittest.mock import patch, MagicMock


class TestGetCheckpointer:
    """Test get_checkpointer returns the right checkpointer type."""

    def setup_method(self):
        """Reset singleton before each test."""
        import agents.tool.storage.checkpoint as cp
        cp._checkpointer = None

    def test_returns_memory_saver(self):
        """Default should return MemorySaver."""
        from agents.tool.storage.checkpoint import get_checkpointer
        from langgraph.checkpoint.memory import MemorySaver

        cp = get_checkpointer()
        assert isinstance(cp, MemorySaver)

    def test_returns_singleton(self):
        """Should return the same instance on repeated calls."""
        from agents.tool.storage.checkpoint import get_checkpointer

        cp1 = get_checkpointer()
        cp2 = get_checkpointer()
        assert cp1 is cp2


class TestGetRedisCheckpointer:
    """Test get_redis_checkpointer."""

    def setup_method(self):
        """Reset singleton before each test."""
        import agents.tool.storage.checkpoint as cp
        cp._checkpointer = None

    @patch("langgraph.checkpoint.redis.aio.AsyncRedisSaver")
    @patch("agents.tool.storage.redis_client.get_redis")
    def test_returns_redis_checkpointer(self, mock_redis, mock_saver):
        """Should return AsyncRedisSaver when Redis is available."""
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        mock_instance = MagicMock()
        mock_saver.return_value = mock_instance

        from agents.tool.storage.checkpoint import get_redis_checkpointer
        result = get_redis_checkpointer()

        assert result is mock_instance
        mock_saver.assert_called_once_with(redis_client=mock_client)

    @patch("agents.tool.storage.redis_client.get_redis", side_effect=Exception("Redis down"))
    def test_fallback_to_memory(self, mock_redis):
        """Should fall back to MemorySaver on error."""
        from agents.tool.storage.checkpoint import get_redis_checkpointer
        from langgraph.checkpoint.memory import MemorySaver

        result = get_redis_checkpointer()
        assert isinstance(result, MemorySaver)

    @patch("langgraph.checkpoint.redis.aio.AsyncRedisSaver")
    @patch("agents.tool.storage.redis_client.get_redis")
    def test_returns_singleton(self, mock_redis, mock_saver):
        """Should return the same instance on repeated calls."""
        mock_redis.return_value = MagicMock()
        mock_saver.return_value = MagicMock()

        from agents.tool.storage.checkpoint import get_redis_checkpointer

        cp1 = get_redis_checkpointer()
        cp2 = get_redis_checkpointer()
        assert cp1 is cp2
