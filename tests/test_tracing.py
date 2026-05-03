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

        assert os.environ.get("LANGCHAIN_API_KEY") == "test-key"
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_ENDPOINT") == "https://test.langchain.com"

        # Cleanup
        os.environ.pop("LANGCHAIN_API_KEY", None)
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGCHAIN_ENDPOINT", None)


class TestCozeLoopInit:
    """Test CozeLoop tracing initialization with JWT OAuth."""

    @patch("agents.tool.trace.tracing.settings")
    def test_cozeloop_disabled_by_default(self, mock_settings):
        mock_settings.cozeloop.tracing = False
        mock_settings.cozeloop.jwt_oauth_client_id = ""

        from agents.tool.trace.tracing import get_cozeloop_handler
        result = get_cozeloop_handler()
        assert result is None

    @patch("agents.tool.trace.tracing.settings")
    def test_cozeloop_returns_none_without_client_id(self, mock_settings):
        mock_settings.cozeloop.tracing = True
        mock_settings.cozeloop.jwt_oauth_client_id = ""

        from agents.tool.trace.tracing import get_cozeloop_handler
        result = get_cozeloop_handler()
        assert result is None

    @patch("agents.tool.trace.tracing.settings")
    def test_cozeloop_returns_none_without_package(self, mock_settings):
        mock_settings.cozeloop.tracing = True
        mock_settings.cozeloop.jwt_oauth_client_id = "test-client-id"
        mock_settings.cozeloop.jwt_oauth_private_key = "test-key"
        mock_settings.cozeloop.jwt_oauth_public_key_id = "test-key-id"
        mock_settings.cozeloop.workspace_id = "test-workspace"
        mock_settings.cozeloop.api_base_url = ""

        from agents.tool.trace.tracing import get_cozeloop_handler
        # Should return None if cozeloop package is not installed
        result = get_cozeloop_handler()
        assert result is None

    @patch("agents.tool.trace.tracing.settings")
    def test_cozeloop_sets_env_vars(self, mock_settings):
        import os
        mock_settings.cozeloop.tracing = True
        mock_settings.cozeloop.jwt_oauth_client_id = "test-client-id"
        mock_settings.cozeloop.jwt_oauth_private_key = "test-private-key"
        mock_settings.cozeloop.jwt_oauth_public_key_id = "test-public-id"
        mock_settings.cozeloop.workspace_id = "test-workspace"
        mock_settings.cozeloop.api_base_url = ""

        from agents.tool.trace.tracing import _set_cozeloop_env
        _set_cozeloop_env()

        assert os.environ.get("COZELOOP_WORKSPACE_ID") == "test-workspace"
        assert os.environ.get("COZELOOP_JWT_OAUTH_CLIENT_ID") == "test-client-id"
        assert os.environ.get("COZELOOP_JWT_OAUTH_PRIVATE_KEY") == "test-private-key"
        assert os.environ.get("COZELOOP_JWT_OAUTH_PUBLIC_KEY_ID") == "test-public-id"

        # Cleanup
        for key in [
            "COZELOOP_WORKSPACE_ID", "COZELOOP_JWT_OAUTH_CLIENT_ID",
            "COZELOOP_JWT_OAUTH_PRIVATE_KEY", "COZELOOP_JWT_OAUTH_PUBLIC_KEY_ID",
            "COZELOOP_API_BASE_URL",
        ]:
            os.environ.pop(key, None)


class TestTraceCallbacks:
    """Test get_trace_callbacks."""

    @patch("agents.tool.trace.tracing.settings")
    def test_returns_empty_list_when_disabled(self, mock_settings):
        mock_settings.cozeloop.tracing = False
        mock_settings.cozeloop.jwt_oauth_client_id = ""

        from agents.tool.trace.tracing import get_trace_callbacks
        callbacks = get_trace_callbacks()
        assert callbacks == []
