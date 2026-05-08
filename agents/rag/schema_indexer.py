"""Legacy MySQL schema indexing into Milvus + ES.

Current NL2SQL schema retrieval reads Redis/MySQL semantic metadata directly.
This module is kept only for explicit legacy maintenance and tests. It is not
called by seed scripts or app startup, and ``index_mysql_schemas`` is disabled
unless ``ENABLE_LEGACY_SCHEMA_INDEX=1`` is set.
"""

from __future__ import annotations

import json
import logging
import os

from langchain_core.documents import Document
from pymilvus import MilvusClient

from agents.config.settings import settings
from agents.tool.sql_tools.mcp_client import execute_sql

try:
    from elasticsearch import Elasticsearch
except ImportError:
    Elasticsearch = None

logger = logging.getLogger(__name__)

_SCHEMA_SOURCE = "mysql_schema"


def _get_embeddings():
    """Return an embedding model instance based on the configured provider."""
    provider = settings.embedding_model_type
    if provider == "ark":
        from langchain_community.embeddings import VolcengineEmbeddings
        return VolcengineEmbeddings(
            ark_api_key=settings.ark.key,
            model=settings.ark.embedding_model,
        )
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            openai_api_key=settings.openai.key,
            model=settings.openai.embedding_model,
        )
    if provider == "qwen":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            openai_api_key=settings.qwen.key,
            openai_api_base=settings.qwen.base_url,
            model=settings.qwen.embedding_model,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
            chunk_size=10,
        )
    raise ValueError(f"Unsupported embedding_model_type: {provider!r}")


def _parse_json_result(raw: str) -> list[dict]:
    """Parse MCP query result, stripping trailing timing info."""
    clean = raw
    idx = raw.rfind("]Query execution time:")
    if idx != -1:
        clean = raw[: idx + 1]
    return json.loads(clean)


async def _load_semantic_model() -> dict[str, dict[str, dict]]:
    """Load semantic model from MySQL, returns {table_name: {column_name: {...}}}."""
    import pymysql

    try:
        conn = pymysql.connect(
            host=settings.mysql.host,
            port=settings.mysql.port,
            user=settings.mysql.username,
            password=settings.mysql.password,
            database=settings.mysql.database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT table_name, column_name, business_name, synonyms, business_description FROM t_semantic_model")
            rows = cur.fetchall()
        conn.close()

        result: dict[str, dict[str, dict]] = {}
        for row in rows:
            tbl = row["table_name"]
            col = row["column_name"]
            result.setdefault(tbl, {})[col] = row
        logger.info("Loaded semantic model: %d tables, %d entries", len(result), len(rows))
        return result
    except Exception as e:
        logger.info("Semantic model not available (%s), using raw schema only", e)
        return {}


async def _fetch_table_schemas() -> list[Document]:
    """Connect to MySQL via MCP and fetch all table schemas via information_schema."""
    docs: list[Document] = []

    tables_raw = await execute_sql(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = DATABASE()"
    )
    logger.info("Raw tables result: %s", tables_raw[:200])

    try:
        tables_data = _parse_json_result(tables_raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse tables result as JSON: %s", tables_raw[:200])
        return docs

    table_names = [row.get("TABLE_NAME") or row.get("table_name") for row in tables_data]
    table_names = [n for n in table_names if n]

    if not table_names:
        logger.info("No tables found in MySQL database")
        return docs

    logger.info("Found %d tables: %s", len(table_names), table_names)

    # Load semantic model for enrichment
    semantic_map = await _load_semantic_model()

    for table_name in table_names:
        try:
            columns_raw = await execute_sql(
                "SELECT column_name, column_type, is_nullable, column_key, column_comment "
                "FROM information_schema.columns "
                "WHERE table_schema = DATABASE() AND table_name = %s "
                "ORDER BY ordinal_position" % repr(table_name)
            )

            try:
                columns_data = _parse_json_result(columns_raw)
            except json.JSONDecodeError:
                logger.warning("Failed to parse columns for %s", table_name)
                continue

            table_semantic = semantic_map.get(table_name, {})
            lines = [f"表名: {table_name}", "字段:"]
            for col in columns_data:
                name = col.get("COLUMN_NAME") or col.get("column_name", "?")
                col_type = col.get("COLUMN_TYPE") or col.get("column_type", "?")
                nullable = col.get("IS_NULLABLE") or col.get("is_nullable", "?")
                key = col.get("COLUMN_KEY") or col.get("column_key", "")
                comment = col.get("COLUMN_COMMENT") or col.get("column_comment", "")

                sem = table_semantic.get(name, {})
                biz_name = sem.get("business_name", "")
                synonyms = sem.get("synonyms", "")
                biz_desc = sem.get("business_description", "")

                parts = [f"  {name} {col_type}"]
                if key == "PRI":
                    parts.append("PRIMARY KEY")
                elif key == "UNI":
                    parts.append("UNIQUE")
                if nullable == "NO":
                    parts.append("NOT NULL")
                if biz_name:
                    parts.append(f"-- {biz_name}")
                if comment and comment != biz_name:
                    parts.append(f"COMMENT '{comment}'")
                lines.append(" ".join(parts))
                if synonyms:
                    lines.append(f"    同义词: {synonyms}")
                if biz_desc:
                    lines.append(f"    描述: {biz_desc}")

            content = "\n".join(lines)

            doc = Document(
                page_content=content,
                metadata={
                    "source": _SCHEMA_SOURCE,
                    "table_name": table_name,
                    "doc_id": f"schema_{table_name}",
                },
            )
            docs.append(doc)
            logger.info("Indexed schema for table: %s (%d columns)", table_name, len(columns_data))

        except Exception as e:
            logger.warning("Failed to fetch schema for table %s: %s", table_name, e)

    return docs


def _store_schema_docs(docs: list[Document]) -> dict:
    """Store schema documents into Milvus and Elasticsearch."""
    if not docs:
        return {"chunk_count": 0}

    embeddings = _get_embeddings()
    doc_ids = [d.metadata["doc_id"] for d in docs]

    # --- Milvus ---
    milvus_uri = (
        f"http://{settings.milvus.addr}"
        if not settings.milvus.addr.startswith("http")
        else settings.milvus.addr
    )

    milvus_client = MilvusClient(uri=milvus_uri)

    # Delete old schema docs (idempotent re-index)
    try:
        milvus_client.delete(
            collection_name=settings.milvus.collection_name,
            ids=doc_ids,
        )
    except Exception:
        pass

    # Embed and insert (batch to avoid embedding API limits)
    texts = [d.page_content for d in docs]
    vectors = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        vectors.extend(embeddings.embed_documents(texts[i : i + batch_size]))

    records = []
    for doc, doc_id, vector in zip(docs, doc_ids, vectors):
        records.append({
            "pk": doc_id,
            "text": doc.page_content,
            "vector": vector,
            "source": _SCHEMA_SOURCE,
            "table_name": doc.metadata.get("table_name", ""),
            "doc_id": doc_id,
        })

    milvus_client.insert(
        collection_name=settings.milvus.collection_name,
        data=records,
    )

    # Ensure index exists and collection is loaded for search
    try:
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        milvus_client.create_index(
            collection_name=settings.milvus.collection_name,
            index_params=index_params,
        )
    except Exception:
        pass  # index may already exist

    try:
        milvus_client.load_collection(settings.milvus.collection_name)
    except Exception:
        pass  # may already be loaded

    logger.info("Stored %d schema docs in Milvus", len(docs))

    # --- Elasticsearch ---
    es_url = (
        settings.es.address
        if settings.es.address.startswith("http")
        else f"http://{settings.es.address}"
    )

    try:
        es = Elasticsearch(es_url)
        for doc, doc_id in zip(docs, doc_ids):
            es.index(
                index=settings.es.index,
                id=doc_id,
                document={
                    "text": doc.page_content,
                    "metadata": {
                        "source": _SCHEMA_SOURCE,
                        "table_name": doc.metadata.get("table_name", ""),
                        "doc_id": doc_id,
                    },
                },
            )
        logger.info("Stored %d schema docs in Elasticsearch", len(docs))
    except Exception as e:
        logger.warning("ES indexing failed (non-fatal): %s", e)

    return {"chunk_count": len(docs)}


async def generate_domain_summary(docs: list[Document]) -> str:
    """Compatibility wrapper; use domain_summary_builder instead."""
    from agents.rag.domain_summary_builder import generate_domain_summary as _generate

    return await _generate(docs)


async def index_mysql_schemas() -> dict:
    """Legacy entry point: fetch MySQL schemas and index them.

    Disabled by default because current SQL React uses Redis/MySQL semantic
    metadata instead of Milvus/ES ``source=mysql_schema`` documents. Set
    ``ENABLE_LEGACY_SCHEMA_INDEX=1`` only when intentionally rebuilding the old
    schema vector index for compatibility checks.

    Returns
    -------
    dict
        ``{"chunk_count": int}`` on legacy success, or
        ``{"chunk_count": 0, "disabled": True}`` when disabled.
    """
    if os.environ.get("ENABLE_LEGACY_SCHEMA_INDEX") != "1":
        logger.info("Legacy mysql_schema indexing disabled")
        return {"chunk_count": 0, "disabled": True}

    try:
        logger.info("Starting legacy MySQL schema indexing...")
        docs = await _fetch_table_schemas()
        if not docs:
            logger.info("No table schemas to index")
            return {"chunk_count": 0}

        result = _store_schema_docs(docs)
        logger.info("MySQL schema indexing complete: %s", result)

        # Generate domain summary from all schemas
        try:
            await generate_domain_summary(docs)
        except Exception as e:
            logger.warning("Domain summary generation failed: %s", e)

        return result
    except Exception as e:
        logger.warning("MySQL schema auto-indexing failed: %s", e)
        return {"chunk_count": 0}
