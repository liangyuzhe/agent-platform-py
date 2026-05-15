from __future__ import annotations

import logging
from typing import Any

import pymysql

from agents.config.settings import settings

logger = logging.getLogger(__name__)


def reindex_business_knowledge_documents() -> int:
    rows = _fetch_rows(
        "SELECT term, formula, synonyms, related_tables FROM t_business_knowledge"
    )
    docs = []
    for row in rows:
        content = f"术语: {row['term']}\n公式: {row['formula']}"
        if row.get("synonyms"):
            content += f"\n同义词: {row['synonyms']}"
        if row.get("related_tables"):
            content += f"\n关联表: {row['related_tables']}"
        docs.append({
            "content": content,
            "doc_id": f"bk_{row['term']}",
            "metadata": {
                "source": "business_knowledge",
                "term": row["term"],
                "doc_id": f"bk_{row['term']}",
            },
        })
    return _index_documents(docs, source="business_knowledge")


def reindex_agent_knowledge_documents() -> int:
    rows = _fetch_rows(
        "SELECT id, question, sql_text, description, category FROM t_agent_knowledge"
    )
    docs = []
    for row in rows:
        doc_id = f"ak_{row['id']}"
        content = f"问题: {row['question']}\nSQL: {row['sql_text']}"
        if row.get("description"):
            content += f"\n说明: {row['description']}"
        docs.append({
            "content": content,
            "doc_id": doc_id,
            "metadata": {
                "source": "agent_knowledge",
                "question": row["question"],
                "category": row.get("category", ""),
                "doc_id": doc_id,
            },
        })
    return _index_documents(docs, source="agent_knowledge")


def _get_conn():
    return pymysql.connect(
        host=settings.mysql.host,
        port=settings.mysql.port,
        user=settings.mysql.username,
        password=settings.mysql.password,
        database=settings.mysql.database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def _fetch_rows(sql: str) -> list[dict[str, Any]]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return list(cur.fetchall())
    finally:
        conn.close()


def _index_documents(docs: list[dict[str, Any]], *, source: str) -> int:
    if not docs:
        return 0

    from agents.rag.retriever import _get_embeddings
    from pymilvus import MilvusClient

    embeddings = _get_embeddings()
    doc_ids = [doc["doc_id"] for doc in docs]
    texts = [doc["content"] for doc in docs]

    vectors = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        vectors.extend(embeddings.embed_documents(texts[i : i + batch_size]))

    milvus_uri = f"http://{settings.milvus.addr}"
    client = MilvusClient(uri=milvus_uri)
    try:
        try:
            client.delete(collection_name=settings.milvus.collection_name, ids=doc_ids)
        except Exception:
            logger.debug("Ignoring missing existing %s vectors during reindex", source, exc_info=True)

        client.insert(
            collection_name=settings.milvus.collection_name,
            data=[
                {
                    "pk": doc["doc_id"],
                    "text": doc["content"],
                    "vector": vector,
                    "source": source,
                    "table_name": "",
                    "doc_id": doc["doc_id"],
                }
                for doc, vector in zip(docs, vectors)
            ],
        )
    finally:
        client.close()

    _index_documents_to_es(docs)
    return len(docs)


def _index_documents_to_es(docs: list[dict[str, Any]]) -> None:
    try:
        from elasticsearch import Elasticsearch

        es_url = (
            settings.es.address
            if settings.es.address.startswith("http")
            else f"http://{settings.es.address}"
        )
        es = Elasticsearch(es_url)
        for doc in docs:
            es.index(
                index=settings.es.index,
                id=doc["doc_id"],
                document={
                    "text": doc["content"],
                    "metadata": doc["metadata"],
                },
            )
    except Exception as exc:
        logger.warning("ES knowledge indexing failed: %s", exc)
