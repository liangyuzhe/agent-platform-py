"""Application configuration via Pydantic Settings.

All values are loaded from environment variables (or a .env file).
Nested config groups mirror the Go project's config.Config layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env file into environment BEFORE creating any Settings instances
# This ensures nested sub-configs can read the values
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Sub-configs (each becomes its own section in the .env)
# ---------------------------------------------------------------------------


class ArkSettings(BaseSettings):
    """Ark (Doubao / Volcengine) model configuration."""

    model_config = SettingsConfigDict(env_prefix="ARK_")

    key: str = Field(default="", description="Ark API key")
    chat_model: str = Field(
        default="doubao-seed-2-0-code-preview-260215",
        description="Ark chat model name",
    )
    embedding_model: str = Field(
        default="",
        description="Ark embedding model endpoint ID",
    )


class OpenAISettings(BaseSettings):
    """OpenAI model configuration."""

    model_config = SettingsConfigDict(env_prefix="OPENAI_")

    key: str = Field(default="", description="OpenAI API key")
    chat_model: str = Field(default="gpt-4", description="OpenAI chat model")
    embedding_model: str = Field(default="", description="OpenAI embedding model")


class QwenSettings(BaseSettings):
    """Qwen (DashScope) model configuration."""

    model_config = SettingsConfigDict(env_prefix="QWEN_")

    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="DashScope compatible endpoint",
    )
    key: str = Field(default="", description="DashScope API key")
    chat_model: str = Field(default="", description="Qwen chat model")
    embedding_model: str = Field(
        default="text-embedding-v3",
        description="Qwen embedding model",
    )


class DeepSeekSettings(BaseSettings):
    """DeepSeek model configuration."""

    model_config = SettingsConfigDict(env_prefix="DEEPSEEK_")

    base_url: str = Field(default="https://api.deepseek.com", description="DeepSeek API base URL")
    key: str = Field(default="", description="DeepSeek API key")
    chat_model: str = Field(default="", description="DeepSeek chat model")
    embedding_model: str = Field(default="", description="DeepSeek embedding model")
    timeout: int = Field(default=30, description="Request timeout in seconds")

    @model_validator(mode="before")
    @classmethod
    def _coerce_empty_strings(cls, data: dict) -> dict:
        return {k: v for k, v in data.items() if v != ""}


class GeminiSettings(BaseSettings):
    """Google Gemini model configuration."""

    model_config = SettingsConfigDict(env_prefix="GEMINI_")

    key: str = Field(default="", description="Gemini API key")
    chat_model: str = Field(default="", description="Gemini chat model")
    embedding_model: str = Field(default="", description="Gemini embedding model")


class MilvusSettings(BaseSettings):
    """Milvus vector database configuration."""

    model_config = SettingsConfigDict(env_prefix="MILVUS_")

    addr: str = Field(default="localhost:19530", description="Milvus address")
    username: str = Field(default="root", description="Milvus username")
    password: str = Field(default="milvus", description="Milvus password")
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score to keep a result",
    )
    collection_name: str = Field(
        default="GoAgent",
        description="Default Milvus collection",
    )
    top_k: int = Field(default=5, description="Number of results to retrieve")

    @model_validator(mode="before")
    @classmethod
    def _coerce_empty_strings(cls, data: dict) -> dict:
        return {k: v for k, v in data.items() if v != ""}


class ElasticSearchSettings(BaseSettings):
    """Elasticsearch configuration."""

    model_config = SettingsConfigDict(env_prefix="ES_")

    address: str = Field(
        default="http://localhost:9200",
        description="Comma-separated ES addresses",
    )
    username: str = Field(default="", description="ES username")
    password: str = Field(default="", description="ES password")
    index: str = Field(default="go_agent_docs", description="Default ES index name")

    @property
    def addresses(self) -> list[str]:
        """Return the address list (split on commas)."""
        return [a.strip() for a in self.address.split(",") if a.strip()]


class RedisSettings(BaseSettings):
    """Redis configuration (checkpointer / cache)."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    addr: str = Field(default="localhost:6379", description="Redis address host:port")
    password: str = Field(default="", description="Redis password")
    db: int = Field(default=0, description="Redis database number")

    @model_validator(mode="before")
    @classmethod
    def _coerce_empty_strings(cls, data: dict) -> dict:
        return {k: v for k, v in data.items() if v != ""}


class MySQLSettings(BaseSettings):
    """MySQL configuration (audit log)."""

    model_config = SettingsConfigDict(env_prefix="MYSQL_")

    host: str = Field(default="localhost", description="MySQL host")
    port: int = Field(default=3307, description="MySQL port")
    username: str = Field(default="root", description="MySQL username")
    password: str = Field(default="", description="MySQL password")
    database: str = Field(default="go_agent_audit", description="MySQL database name")

    @property
    def url(self) -> str:
        return (
            f"mysql+pymysql://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @model_validator(mode="before")
    @classmethod
    def _coerce_empty_strings(cls, data: dict) -> dict:
        return {k: v for k, v in data.items() if v != ""}


class LangSmithSettings(BaseSettings):
    """LangSmith tracing configuration."""

    model_config = SettingsConfigDict(env_prefix="LANGSMITH_")

    api_key: str = Field(default="", description="LangSmith API key")
    url: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API URL",
    )
    project: str = Field(default="", description="LangSmith project name")
    tracing: bool = Field(default=False, description="Enable LangSmith tracing")


class CozeLoopSettings(BaseSettings):
    """CozeLoop tracing configuration (JWT OAuth)."""

    model_config = SettingsConfigDict(env_prefix="COZELOOP_")

    workspace_id: str = Field(default="", description="CozeLoop workspace ID")
    api_base_url: str = Field(
        default="",
        description="CozeLoop API base URL (leave empty for default)",
    )
    jwt_oauth_client_id: str = Field(default="", description="JWT OAuth client ID")
    jwt_oauth_private_key: str = Field(default="", description="JWT OAuth private key (PEM)")
    jwt_oauth_public_key_id: str = Field(default="", description="JWT OAuth public key ID")
    tracing: bool = Field(default=False, description="Enable CozeLoop tracing")

    @model_validator(mode="before")
    @classmethod
    def _coerce_empty_strings(cls, data: dict) -> dict:
        return {k: v for k, v in data.items() if v != ""}


# ---------------------------------------------------------------------------
# RAG & Memory parameters
# ---------------------------------------------------------------------------


class RAGSettings(BaseSettings):
    """Retrieval-Augmented Generation parameters."""

    model_config = SettingsConfigDict(env_prefix="RAG_")

    mode: Literal["traditional", "parent"] = Field(
        default="traditional",
        description="RAG retrieval mode: traditional or parent document",
    )
    chunk_size: int = Field(default=1024, description="Text chunk size in tokens")
    chunk_overlap: int = Field(default=128, description="Overlap between chunks")
    top_k: int = Field(default=5, description="Default number of chunks to retrieve")
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity for a chunk to be kept",
    )
    # Reranker params
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Cross-Encoder reranker model name, empty string to disable",
    )
    reranker_top_k: int = Field(default=5, description="Number of results after reranking")
    rerank_threshold: float = Field(default=0.1, description="Minimum rerank score to keep")
    # Parent Document RAG params
    parent_chunk_size: int = Field(
        default=2048, description="Parent chunk size for parent document RAG"
    )
    parent_chunk_overlap: int = Field(
        default=200, description="Overlap between parent chunks"
    )
    child_chunk_size: int = Field(
        default=512, description="Child chunk size for parent document RAG"
    )
    child_chunk_overlap: int = Field(
        default=64, description="Overlap between child chunks"
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_empty_strings(cls, data: dict) -> dict:
        # Keep RAG_RERANKER_MODEL as-is (empty string = disable reranker)
        return {
            k: v for k, v in data.items()
            if v != "" or k == "reranker_model"
        }


class MemorySettings(BaseSettings):
    """Agent conversation memory parameters."""

    model_config = SettingsConfigDict(env_prefix="MEMORY_")

    max_tokens: int = Field(
        default=4000,
        description="Max tokens to keep in conversation history",
    )
    summary_threshold: int = Field(
        default=3000,
        description="Token count at which summarisation kicks in",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_empty_strings(cls, data: dict) -> dict:
        return {k: v for k, v in data.items() if v != ""}


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Root application settings.

    Loads from environment variables and ``.env`` in the project root.
    Sub-configs are accessible as attributes, e.g. ``settings.ark.key``.
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- model type selectors ---
    chat_model_type: Literal["ark", "openai", "qwen", "gemini", "deepseek"] = (
        Field(default="ark", description="Provider to use for chat LLM")
    )
    intent_model_type: Literal["ark", "openai", "qwen", "gemini", "deepseek"] = (
        Field(default="ark", description="Provider to use for intent detection LLM")
    )
    embedding_model_type: Literal["ark", "openai", "qwen", "gemini", "deepseek"] = (
        Field(default="qwen", description="Provider to use for embeddings")
    )
    vector_db_type: Literal["MILVUS", "ELASTICSEARCH"] = Field(
        default="MILVUS",
        description="Which vector store backend to use",
    )

    # --- API server ---
    api_host: str = Field(default="0.0.0.0", description="API listen host")
    api_port: int = Field(default=8080, description="API listen port")

    # --- nested provider configs (each reads its own env prefix) ---
    ark: ArkSettings = Field(default_factory=ArkSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    qwen: QwenSettings = Field(default_factory=QwenSettings)
    deepseek: DeepSeekSettings = Field(default_factory=DeepSeekSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)

    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    es: ElasticSearchSettings = Field(default_factory=ElasticSearchSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    mysql: MySQLSettings = Field(default_factory=MySQLSettings)

    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)
    cozeloop: CozeLoopSettings = Field(default_factory=CozeLoopSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached ``Settings`` instance (created once)."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()
    return _settings


# Module-level singleton for convenient imports: ``from agents.config.settings import settings``
settings = get_settings()
