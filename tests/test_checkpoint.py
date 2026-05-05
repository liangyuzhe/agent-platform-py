"""Tests for LangGraph checkpointer factory."""

import pytest
from unittest.mock import patch, MagicMock


class TestGetCheckpointer:
    """Test get_checkpointer returns the right checkpointer type."""

    def setup_method(self):
        """Reset singleton before each test."""
        import agents.tool.storage.checkpoint as cp
        cp._checkpointer = None

    def test_returns_memory_saver_by_default(self):
        """Default (checkpointer_enabled=False) should return MemorySaver."""
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

    @patch("langgraph.checkpoint.redis.aio.AsyncRedisSaver")
    @patch("agents.tool.storage.redis_client.get_redis")
    @patch("agents.config.settings.settings")
    def test_returns_redis_when_enabled(self, mock_settings, mock_redis, mock_saver):
        """Should return AsyncRedisSaver when checkpointer_enabled=True."""
        mock_settings.redis.checkpointer_enabled = True
        mock_redis.return_value = MagicMock()
        mock_saver.return_value = MagicMock()

        from agents.tool.storage.checkpoint import get_checkpointer
        result = get_checkpointer()

        assert result is mock_saver.return_value

    @patch("agents.tool.storage.redis_client.get_redis", side_effect=Exception("Redis down"))
    @patch("agents.config.settings.settings")
    def test_fallback_to_memory_on_redis_error(self, mock_settings, mock_redis):
        """Should fall back to MemorySaver when Redis is enabled but unavailable."""
        mock_settings.redis.checkpointer_enabled = True

        from agents.tool.storage.checkpoint import get_checkpointer
        from langgraph.checkpoint.memory import MemorySaver

        result = get_checkpointer()
        assert isinstance(result, MemorySaver)
