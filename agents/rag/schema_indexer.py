"""Auto-index MySQL table schemas into Milvus + ES at startup.

Connects to MySQL via MCP, reads all table schemas from information_schema,
and indexes them as documents for the SQL React graph's retrieval step.
"""

from __future__ import annotations

import json
import logging

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
        )
    raise ValueError(f"Unsupported embedding_model_type: {provider!r}")


def _parse_json_result(raw: str) -> list[dict]:
    """Parse MCP query result, stripping trailing timing info."""
    clean = raw
    idx = raw.rfind("]Query execution time:")
    if idx != -1:
        clean = raw[: idx + 1]
    return json.loads(clean)


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

            lines = [f"表名: {table_name}", "字段:"]
            for col in columns_data:
                name = col.get("COLUMN_NAME") or col.get("column_name", "?")
                col_type = col.get("COLUMN_TYPE") or col.get("column_type", "?")
                nullable = col.get("IS_NULLABLE") or col.get("is_nullable", "?")
                key = col.get("COLUMN_KEY") or col.get("column_key", "")
                comment = col.get("COLUMN_COMMENT") or col.get("column_comment", "")

                parts = [f"  {name} {col_type}"]
                if key == "PRI":
                    parts.append("PRIMARY KEY")
                elif key == "UNI":
                    parts.append("UNIQUE")
                if nullable == "NO":
                    parts.append("NOT NULL")
                if comment:
                    parts.append(f"COMMENT '{comment}'")
                lines.append(" ".join(parts))

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

    # Embed and insert
    texts = [d.page_content for d in docs]
    vectors = embeddings.embed_documents(texts)

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
                    "source": _SCHEMA_SOURCE,
                    "table_name": doc.metadata.get("table_name", ""),
                    "doc_id": doc_id,
                },
            )
        logger.info("Stored %d schema docs in Elasticsearch", len(docs))
    except Exception as e:
        logger.warning("ES indexing failed (non-fatal): %s", e)

    return {"chunk_count": len(docs)}


async def generate_domain_summary(docs: list[Document]) -> str:
    """Use LLM to generate a compact domain summary from all schema docs.

    The summary is a short text describing what tables exist and their
    purpose, used by the intent classifier instead of a hardcoded prompt.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from agents.model.chat_model import get_chat_model
    from agents.tool.storage.domain_summary import save_domain_summary

    schemas_text = "\n\n".join(d.page_content for d in docs)

    model = get_chat_model(settings.chat_model_type)
    response = await model.ainvoke([
        SystemMessage(content=(
            "你是一个数据库架构分析专家。请根据以下所有表结构信息，生成一段简洁的领域摘要。"
            "摘要需要说明：\n"
            "1. 这个数据库管理的是什么业务领域\n"
            "2. 包含哪些核心实体（表）及它们的关系\n"
            "3. 可以回答哪些类型的问题\n\n"
            "要求：控制在 500 字以内，使用中文，语言简洁专业。"
        )),
        HumanMessage(content=f"以下是所有表结构：\n\n{schemas_text}"),
    ])

    summary = response.content.strip()
    await save_domain_summary(summary)
    logger.info("Domain summary generated (%d chars)", len(summary))
    return summary


async def index_mysql_schemas() -> dict:
    """Main entry point: fetch MySQL table schemas and index them.

    Call this at application startup (as a background task so it
    doesn't block the server from accepting requests).

    Returns
    -------
    dict
        ``{"chunk_count": int}`` on success, or ``{"chunk_count": 0}``
        if no tables were found or an error occurred.
    """
    try:
        logger.info("Starting MySQL schema auto-indexing...")
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
