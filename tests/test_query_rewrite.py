"""Tests for query rewrite module."""

import pytest
from unittest.mock import patch, MagicMock

from agents.rag.query_rewrite import _format_history, rewrite_query


class TestFormatHistory:
    """Test _format_history helper."""

    def test_string_passthrough(self):
        """String input is returned as-is."""
        assert _format_history("hello") == "hello"

    def test_empty_string(self):
        assert _format_history("") == ""

    def test_list_of_dicts(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = _format_history(history)
        assert "[user]: hello" in result
        assert "[assistant]: hi there" in result

    def test_empty_list(self):
        assert _format_history([]) == ""

    def test_missing_role(self):
        history = [{"content": "hello"}]
        result = _format_history(history)
        assert "[user]: hello" in result

    def test_missing_content(self):
        history = [{"role": "user"}]
        result = _format_history(history)
        assert "[user]: " in result


class TestRewriteQuery:
    """Test rewrite_query with mocked LLM."""

    @patch("agents.rag.query_rewrite.get_chat_model")
    def test_rewrite_calls_llm(self, mock_get_model):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "rewritten query"
        mock_llm.invoke.return_value = mock_response
        mock_get_model.return_value = mock_llm

        result = rewrite_query(
            summary="user likes Python",
            history=[{"role": "user", "content": "what is list comprehension?"}],
            query="its advantages?",
        )

        assert result == "rewritten query"
        mock_llm.invoke.assert_called_once()
        messages = mock_llm.invoke.call_args[0][0]
        assert len(messages) == 2
        assert messages[0][0] == "system"
        assert messages[1][0] == "human"
        assert "user likes Python" in messages[1][1]
        assert "its advantages?" in messages[1][1]

    @patch("agents.rag.query_rewrite.get_chat_model")
    def test_rewrite_with_string_history(self, mock_get_model):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "standalone query"
        mock_llm.invoke.return_value = mock_response
        mock_get_model.return_value = mock_llm

        result = rewrite_query(summary="", history="[user]: hi", query="test")
        assert result == "standalone query"

    @patch("agents.rag.query_rewrite.get_chat_model")
    def test_rewrite_strips_whitespace(self, mock_get_model):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "  cleaned query  "
        mock_llm.invoke.return_value = mock_response
        mock_get_model.return_value = mock_llm

        result = rewrite_query(summary="", history="", query="test")
        assert result == "cleaned query"

    @patch("agents.rag.query_rewrite.get_chat_model")
    def test_rewrite_uses_configured_model(self, mock_get_model):
        """Verify it uses get_chat_model, not hardcoded Qwen."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "result"
        mock_llm.invoke.return_value = mock_response
        mock_get_model.return_value = mock_llm

        rewrite_query(summary="", history="", query="test")

        from agents.config.settings import settings
        mock_get_model.assert_called_with(settings.chat_model_type)
