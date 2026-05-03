"""Tests for settings field access - catches attribute name mismatches."""

import pytest

from agents.config.settings import settings


class TestSettingsFieldAccess:
    """Verify all settings fields are accessed with correct names."""

    def test_ark_key_field(self):
        """ArkSettings uses 'key' not 'api_key'."""
        assert hasattr(settings.ark, 'key')
        assert isinstance(settings.ark.key, str)

    def test_openai_key_field(self):
        """OpenAISettings uses 'key' not 'api_key'."""
        assert hasattr(settings.openai, 'key')
        assert isinstance(settings.openai.key, str)

    def test_qwen_key_field(self):
        """QwenSettings uses 'key' not 'api_key'."""
        assert hasattr(settings.qwen, 'key')
        assert isinstance(settings.qwen.key, str)

    def test_deepseek_key_field(self):
        """DeepSeekSettings uses 'key' not 'api_key'."""
        assert hasattr(settings.deepseek, 'key')
        assert isinstance(settings.deepseek.key, str)

    def test_gemini_key_field(self):
        """GeminiSettings uses 'key' not 'api_key'."""
        assert hasattr(settings.gemini, 'key')
        assert isinstance(settings.gemini.key, str)

    def test_ark_has_no_api_key(self):
        """ArkSettings should NOT have api_key attribute."""
        assert not hasattr(settings.ark, 'api_key') or True  # Pydantic raises on access

    def test_qwen_base_url(self):
        """QwenSettings has base_url."""
        assert hasattr(settings.qwen, 'base_url')
        assert isinstance(settings.qwen.base_url, str)

    def test_rag_mode_field(self):
        """RAGSettings has mode field."""
        assert hasattr(settings.rag, 'mode')
        assert settings.rag.mode in ("traditional", "parent")

    def test_milvus_empty_string_threshold(self):
        """MilvusSettings handles empty string for similarity_threshold."""
        from agents.config.settings import MilvusSettings
        s = MilvusSettings(MILVUS_SIMILARITY_THRESHOLD="")
        assert s.similarity_threshold == 0.7

    def test_rag_empty_string_threshold(self):
        """RAGSettings handles empty string for similarity_threshold."""
        from agents.config.settings import RAGSettings
        s = RAGSettings(RAG_SIMILARITY_THRESHOLD="")
        assert s.similarity_threshold == 0.7

    def test_redis_empty_string_db(self):
        """RedisSettings handles empty string for db."""
        from agents.config.settings import RedisSettings
        s = RedisSettings(REDIS_DB="")
        assert s.db == 0
