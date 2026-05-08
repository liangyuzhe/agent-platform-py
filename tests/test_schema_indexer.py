"""Tests for legacy MySQL schema indexing helpers."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TABLES_RAW = json.dumps([{"TABLE_NAME": "t_user"}]) + "Query execution time: 5.0 ms"

_COLUMNS_RAW = json.dumps([
    {"COLUMN_NAME": "id", "COLUMN_TYPE": "int", "IS_NULLABLE": "NO", "COLUMN_KEY": "PRI", "COLUMN_COMMENT": "ID"},
    {"COLUMN_NAME": "username", "COLUMN_TYPE": "varchar(50)", "IS_NULLABLE": "NO", "COLUMN_KEY": "UNI", "COLUMN_COMMENT": ""},
]) + "Query execution time: 10.0 ms"


# ---------------------------------------------------------------------------
# _parse_json_result
# ---------------------------------------------------------------------------

class TestParseJsonResult:
    """Test JSON parsing with timing suffix."""

    def test_parse_with_timing(self):
        from agents.rag.schema_indexer import _parse_json_result
        raw = '[{"TABLE_NAME":"t_user"}]Query execution time: 5.0 ms'
        result = _parse_json_result(raw)
        assert result == [{"TABLE_NAME": "t_user"}]

    def test_parse_without_timing(self):
        from agents.rag.schema_indexer import _parse_json_result
        raw = '[{"TABLE_NAME":"t_user"}]'
        result = _parse_json_result(raw)
        assert result == [{"TABLE_NAME": "t_user"}]

    def test_parse_invalid_json_raises(self):
        from agents.rag.schema_indexer import _parse_json_result
        with pytest.raises(json.JSONDecodeError):
            _parse_json_result("not json")


# ---------------------------------------------------------------------------
# _fetch_table_schemas
# ---------------------------------------------------------------------------

class TestFetchTableSchemas:
    """Test fetching schemas from MySQL via MCP."""

    @pytest.mark.asyncio
    @patch("agents.rag.schema_indexer.execute_sql", new_callable=AsyncMock)
    async def test_fetch_single_table(self, mock_exec):
        """Should return one Document per table with columns."""
        from agents.rag.schema_indexer import _fetch_table_schemas

        mock_exec.side_effect = [_TABLES_RAW, _COLUMNS_RAW]

        docs = await _fetch_table_schemas()

        assert len(docs) == 1
        assert docs[0].metadata["table_name"] == "t_user"
        assert docs[0].metadata["doc_id"] == "schema_t_user"
        assert docs[0].metadata["source"] == "mysql_schema"
        assert "表名: t_user" in docs[0].page_content
        assert "id int PRIMARY KEY" in docs[0].page_content
        assert "username varchar(50) UNIQUE" in docs[0].page_content

    @pytest.mark.asyncio
    @patch("agents.rag.schema_indexer.execute_sql", new_callable=AsyncMock)
    async def test_fetch_multiple_tables(self, mock_exec):
        """Should handle multiple tables."""
        from agents.rag.schema_indexer import _fetch_table_schemas

        tables_raw = json.dumps([
            {"TABLE_NAME": "t_user"},
            {"TABLE_NAME": "t_order"},
        ]) + "Query execution time: 5.0 ms"

        columns_user = json.dumps([
            {"COLUMN_NAME": "id", "COLUMN_TYPE": "int", "IS_NULLABLE": "NO", "COLUMN_KEY": "PRI", "COLUMN_COMMENT": ""},
        ]) + "Query execution time: 5.0 ms"

        columns_order = json.dumps([
            {"COLUMN_NAME": "order_id", "COLUMN_TYPE": "bigint", "IS_NULLABLE": "NO", "COLUMN_KEY": "PRI", "COLUMN_COMMENT": ""},
        ]) + "Query execution time: 5.0 ms"

        mock_exec.side_effect = [tables_raw, columns_user, columns_order]

        docs = await _fetch_table_schemas()

        assert len(docs) == 2
        assert docs[0].metadata["table_name"] == "t_user"
        assert docs[1].metadata["table_name"] == "t_order"

    @pytest.mark.asyncio
    @patch("agents.rag.schema_indexer.execute_sql", new_callable=AsyncMock)
    async def test_fetch_no_tables(self, mock_exec):
        """Should return empty list when no tables exist."""
        from agents.rag.schema_indexer import _fetch_table_schemas

        mock_exec.return_value = '[]' + "Query execution time: 1.0 ms"

        docs = await _fetch_table_schemas()

        assert docs == []

    @pytest.mark.asyncio
    @patch("agents.rag.schema_indexer.execute_sql", new_callable=AsyncMock)
    async def test_fetch_invalid_json_returns_empty(self, mock_exec):
        """Should return empty list when result is not valid JSON."""
        from agents.rag.schema_indexer import _fetch_table_schemas

        mock_exec.return_value = "not json"

        docs = await _fetch_table_schemas()

        assert docs == []

    @pytest.mark.asyncio
    @patch("agents.rag.schema_indexer.execute_sql", new_callable=AsyncMock)
    async def test_fetch_column_error_skips_table(self, mock_exec):
        """Should skip table if column query fails."""
        from agents.rag.schema_indexer import _fetch_table_schemas

        tables_raw = json.dumps([{"TABLE_NAME": "t_user"}]) + "Query execution time: 5.0 ms"
        mock_exec.side_effect = [tables_raw, Exception("Column query failed")]

        docs = await _fetch_table_schemas()

        assert docs == []

    @pytest.mark.asyncio
    @patch("agents.rag.schema_indexer.execute_sql", new_callable=AsyncMock)
    async def test_fetch_column_json_error_skips_table(self, mock_exec):
        """Should skip table if column result is invalid JSON."""
        from agents.rag.schema_indexer import _fetch_table_schemas

        tables_raw = json.dumps([{"TABLE_NAME": "t_user"}]) + "Query execution time: 5.0 ms"
        mock_exec.side_effect = [tables_raw, "not json"]

        docs = await _fetch_table_schemas()

        assert docs == []


# ---------------------------------------------------------------------------
# _store_schema_docs
# ---------------------------------------------------------------------------

class TestStoreSchemaDocs:
    """Test storing docs into Milvus + ES."""

    @patch("agents.rag.schema_indexer.Elasticsearch")
    @patch("agents.rag.schema_indexer.MilvusClient")
    @patch("agents.rag.schema_indexer._get_embeddings")
    def test_store_single_doc(self, mock_get_emb, MockMilvusClient, MockES):
        """Should embed and store docs in Milvus and ES."""
        from agents.rag.schema_indexer import _store_schema_docs

        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 1024]
        mock_get_emb.return_value = mock_embeddings

        mock_milvus = MagicMock()
        MockMilvusClient.return_value = mock_milvus

        mock_es = MagicMock()
        MockES.return_value = mock_es

        docs = [Document(
            page_content="表名: t_user\n字段:\n  id int PRIMARY KEY",
            metadata={"source": "mysql_schema", "table_name": "t_user", "doc_id": "schema_t_user"},
        )]

        result = _store_schema_docs(docs)

        assert result["chunk_count"] == 1
        mock_milvus.insert.assert_called_once()
        mock_es.index.assert_called_once()

    @patch("agents.rag.schema_indexer._get_embeddings")
    def test_store_empty_docs(self, mock_get_emb):
        """Should return chunk_count=0 for empty docs."""
        from agents.rag.schema_indexer import _store_schema_docs

        result = _store_schema_docs([])

        assert result["chunk_count"] == 0
        mock_get_emb.assert_not_called()

    @patch("agents.rag.schema_indexer.Elasticsearch")
    @patch("agents.rag.schema_indexer.MilvusClient")
    @patch("agents.rag.schema_indexer._get_embeddings")
    def test_store_deletes_old_docs_first(self, mock_get_emb, MockMilvusClient, MockES):
        """Should delete old schema docs before inserting."""
        from agents.rag.schema_indexer import _store_schema_docs

        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 1024]
        mock_get_emb.return_value = mock_embeddings

        mock_milvus = MagicMock()
        MockMilvusClient.return_value = mock_milvus

        mock_es = MagicMock()
        MockES.return_value = mock_es

        docs = [Document(
            page_content="test",
            metadata={"source": "mysql_schema", "table_name": "t", "doc_id": "schema_t"},
        )]

        _store_schema_docs(docs)

        # Verify delete was called before insert
        mock_milvus.delete.assert_called_once()
        mock_milvus.insert.assert_called_once()

    @patch("agents.rag.schema_indexer.Elasticsearch")
    @patch("agents.rag.schema_indexer.MilvusClient")
    @patch("agents.rag.schema_indexer._get_embeddings")
    def test_store_creates_index_and_loads(self, mock_get_emb, MockMilvusClient, MockES):
        """Should create Milvus index and load collection after insert."""
        from agents.rag.schema_indexer import _store_schema_docs

        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 1024]
        mock_get_emb.return_value = mock_embeddings

        mock_milvus = MagicMock()
        MockMilvusClient.return_value = mock_milvus

        mock_es = MagicMock()
        MockES.return_value = mock_es

        docs = [Document(
            page_content="test",
            metadata={"source": "mysql_schema", "table_name": "t", "doc_id": "schema_t"},
        )]

        _store_schema_docs(docs)

        mock_milvus.create_index.assert_called_once()
        mock_milvus.load_collection.assert_called_once()

    @patch("agents.rag.schema_indexer.Elasticsearch")
    @patch("agents.rag.schema_indexer.MilvusClient")
    @patch("agents.rag.schema_indexer._get_embeddings")
    def test_store_es_failure_is_non_fatal(self, mock_get_emb, MockMilvusClient, MockES):
        """ES failure should not raise — it's non-fatal."""
        from agents.rag.schema_indexer import _store_schema_docs

        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 1024]
        mock_get_emb.return_value = mock_embeddings

        mock_milvus = MagicMock()
        MockMilvusClient.return_value = mock_milvus

        MockES.side_effect = Exception("ES down")

        docs = [Document(
            page_content="test",
            metadata={"source": "mysql_schema", "table_name": "t", "doc_id": "schema_t"},
        )]

        # Should not raise
        result = _store_schema_docs(docs)
        assert result["chunk_count"] == 1


# ---------------------------------------------------------------------------
# index_mysql_schemas (main entry point)
# ---------------------------------------------------------------------------

class TestIndexMysqlSchemas:
    """Test the main entry point."""

    @pytest.mark.asyncio
    async def test_index_disabled_by_default(self):
        """Legacy schema indexing should not run unless explicitly enabled."""
        from agents.rag.schema_indexer import index_mysql_schemas

        result = await index_mysql_schemas()

        assert result == {"chunk_count": 0, "disabled": True}

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"ENABLE_LEGACY_SCHEMA_INDEX": "1"})
    @patch("agents.rag.schema_indexer._store_schema_docs")
    @patch("agents.rag.schema_indexer._fetch_table_schemas")
    async def test_index_success(self, mock_fetch, mock_store):
        """Should fetch and store schemas when legacy indexing is enabled."""
        from agents.rag.schema_indexer import index_mysql_schemas

        mock_docs = [MagicMock()]
        mock_fetch.return_value = mock_docs
        mock_store.return_value = {"chunk_count": 1}

        result = await index_mysql_schemas()

        assert result["chunk_count"] == 1
        mock_fetch.assert_called_once()
        mock_store.assert_called_once_with(mock_docs)

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"ENABLE_LEGACY_SCHEMA_INDEX": "1"})
    @patch("agents.rag.schema_indexer._fetch_table_schemas")
    async def test_index_no_tables(self, mock_fetch):
        """Should return chunk_count=0 when no tables found."""
        from agents.rag.schema_indexer import index_mysql_schemas

        mock_fetch.return_value = []

        result = await index_mysql_schemas()

        assert result["chunk_count"] == 0

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"ENABLE_LEGACY_SCHEMA_INDEX": "1"})
    @patch("agents.rag.schema_indexer._fetch_table_schemas")
    async def test_index_fetch_error_returns_zero(self, mock_fetch):
        """Should return chunk_count=0 when fetch raises."""
        from agents.rag.schema_indexer import index_mysql_schemas

        mock_fetch.side_effect = Exception("MySQL down")

        result = await index_mysql_schemas()

        assert result["chunk_count"] == 0
