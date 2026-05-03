"""Tests for tracing initialization."""

import pytest
from unittest.mock import patch, MagicMock


class TestLangSmithInit:
    """Test LangSmith tracing initialization."""

    @patch("agents.tool.trace.tracing.settings")
    def test_langsmith_disabled_by_default(self, mock_settings):
        mock_settings.langsmith.tracing = False
        mock_settings.langsmith.api_key = ""

        from agents.tool.trace.tracing import init_langsmith
        # Should not raise
        init_langsmith()

    @patch("agents.tool.trace.tracing.settings")
    def test_langsmith_sets_env_vars(self, mock_settings):
        import os
        mock_settings.langsmith.tracing = True
        mock_settings.langsmith.api_key = "test-key"
        mock_settings.langsmith.url = "https://test.langchain.com"

        from agents.tool.trace.tracing import init_langsmith
        init_langsmith()

        assert os.environ.get("LANGSMITH_API_KEY") == "test-key"
        assert os.environ.get("LANGSMITH_TRACING_V2") == "true"
        assert os.environ.get("LANGSMITH_ENDPOINT") == "https://test.langchain.com"

        # Cleanup
        os.environ.pop("LANGSMITH_API_KEY", None)
        os.environ.pop("LANGSMITH_TRACING_V2", None)
        os.environ.pop("LANGSMITH_ENDPOINT", None)


class TestCozeLoopInit:
    """Test CozeLoop tracing initialization."""

    @patch("agents.tool.trace.tracing.settings")
    def test_cozeloop_disabled_by_default(self, mock_settings):
        mock_settings.cozeloop.tracing = False
        mock_settings.cozeloop.api_key = ""

        from agents.tool.trace.tracing import get_cozeloop_handler
        result = get_cozeloop_handler()
        assert result is None

    @patch("agents.tool.trace.tracing.settings")
    def test_cozeloop_returns_none_without_package(self, mock_settings):
        mock_settings.cozeloop.tracing = True
        mock_settings.cozeloop.api_key = "test-key"
        mock_settings.cozeloop.endpoint = "https://api.coze.com"

        from agents.tool.trace.tracing import get_cozeloop_handler
        # Should return None if cozeloop package is not installed
        result = get_cozeloop_handler()
        # Either returns a handler or None (if package not installed)
        assert result is None or hasattr(result, 'on_llm_start')


class TestTraceCallbacks:
    """Test get_trace_callbacks."""

    @patch("agents.tool.trace.tracing.settings")
    def test_returns_empty_list_when_disabled(self, mock_settings):
        mock_settings.cozeloop.tracing = False
        mock_settings.cozeloop.api_key = ""

        from agents.tool.trace.tracing import get_trace_callbacks
        callbacks = get_trace_callbacks()
        assert callbacks == []
