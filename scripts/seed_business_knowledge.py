"""Seed business knowledge (terms, formulas, synonyms) into MySQL + Milvus.

Usage:
    python -m scripts.seed_business_knowledge

Optionally override the seed file:
    BUSINESS_KNOWLEDGE_SEED_FILE=/path/to/business_knowledge.json python -m scripts.seed_business_knowledge
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pymysql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.config.settings import settings

_DEFAULT_SEED_FILE = Path(__file__).resolve().parents[1] / "data" / "business_knowledge_seed.json"


def get_conn():
    return pymysql.connect(
        host=settings.mysql.host,
        port=settings.mysql.port,
        user=settings.mysql.username,
        password=settings.mysql.password,
        database=settings.mysql.database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS t_business_knowledge (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                term VARCHAR(128) NOT NULL COMMENT '业务术语',
                formula TEXT NOT NULL COMMENT '公式/定义',
                synonyms TEXT COMMENT '同义词，逗号分隔',
                related_tables TEXT COMMENT '关联表名，逗号分隔',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uk_term (term)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='业务知识：术语、公式、同义词'
        """)
    conn.commit()
    print("t_business_knowledge table created")


def load_seed_records() -> list[dict]:
    seed_file = Path(os.getenv("BUSINESS_KNOWLEDGE_SEED_FILE", str(_DEFAULT_SEED_FILE))).expanduser()
    with seed_file.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Business knowledge seed file must contain a list: {seed_file}")
    return records


def seed_data(conn):
    records = load_seed_records()

    with conn.cursor() as cur:
        for item in records:
            term = item["term"]
            formula = item["formula"]
            synonyms = item.get("synonyms", "")
            related_tables = item.get("related_tables", "")
            cur.execute(
                """INSERT INTO t_business_knowledge (term, formula, synonyms, related_tables)
                   VALUES (%s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                       formula=VALUES(formula),
                       synonyms=VALUES(synonyms),
                       related_tables=VALUES(related_tables)""",
                (term, formula, synonyms, related_tables),
            )
    conn.commit()
    print(f"Seeded {len(records)} business knowledge entries")


async def index_to_milvus():
    """Vectorize business knowledge and store in Milvus."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT term, formula, synonyms, related_tables FROM t_business_knowledge")
        rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No business knowledge to index")
        return

    from agents.rag.retriever import _get_embeddings
    from pymilvus import MilvusClient

    docs = []
    for row in rows:
        content = f"术语: {row['term']}\n公式: {row['formula']}"
        if row.get("synonyms"):
            content += f"\n同义词: {row['synonyms']}"
        if row.get("related_tables"):
            content += f"\n关联表: {row['related_tables']}"
        docs.append({
            "content": content,
            "term": row["term"],
            "doc_id": f"bk_{row['term']}",
        })

    embeddings = _get_embeddings()
    milvus_uri = f"http://{settings.milvus.addr}"
    client = MilvusClient(uri=milvus_uri)

    doc_ids = [d["doc_id"] for d in docs]
    try:
        client.delete(collection_name=settings.milvus.collection_name, ids=doc_ids)
    except Exception:
        pass

    texts = [d["content"] for d in docs]
    vectors = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        vectors.extend(embeddings.embed_documents(texts[i : i + batch_size]))

    records = []
    for doc, doc_id, vector in zip(docs, doc_ids, vectors):
        records.append({
            "pk": doc_id,
            "text": doc["content"],
            "vector": vector,
            "source": "business_knowledge",
            "table_name": "",
            "doc_id": doc_id,
        })

    client.insert(collection_name=settings.milvus.collection_name, data=records)
    client.close()
    print(f"Indexed {len(docs)} business knowledge entries into Milvus")

    # --- Elasticsearch BM25 ---
    try:
        from elasticsearch import Elasticsearch
        es_url = settings.es.address if settings.es.address.startswith("http") else f"http://{settings.es.address}"
        es = Elasticsearch(es_url)
        for doc, doc_id in zip(docs, doc_ids):
            es.index(
                index=settings.es.index,
                id=doc_id,
                document={
                    "text": doc["content"],
                    "metadata": {
                        "source": "business_knowledge",
                        "term": doc["term"],
                        "doc_id": doc_id,
                    },
                },
            )
        print(f"Indexed {len(docs)} business knowledge entries into Elasticsearch")
    except Exception as e:
        print(f"ES indexing failed (non-fatal): {e}")


def main():
    conn = get_conn()
    try:
        create_table(conn)
        seed_data(conn)
    finally:
        conn.close()

    asyncio.run(index_to_milvus())


if __name__ == "__main__":
    main()
