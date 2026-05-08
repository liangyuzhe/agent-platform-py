"""Remove legacy MySQL schema documents from vector/search indexes.

Current NL2SQL schema retrieval reads Redis/MySQL t_semantic_model directly.
The old Milvus/Elasticsearch documents with source=mysql_schema are no longer
used and can be deleted safely.

Usage:
    python -m scripts.cleanup_schema_indexes
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.config.settings import settings

_TIMEOUT_SECONDS = 15


def cleanup_milvus() -> None:
    from pymilvus import MilvusClient

    milvus_uri = (
        f"http://{settings.milvus.addr}"
        if not settings.milvus.addr.startswith("http")
        else settings.milvus.addr
    )
    print(f"Connecting Milvus: {milvus_uri}", flush=True)
    client = MilvusClient(uri=milvus_uri, timeout=_TIMEOUT_SECONDS)
    try:
        collection = settings.milvus.collection_name
        if collection not in client.list_collections(timeout=_TIMEOUT_SECONDS):
            print(f"Milvus collection not found: {collection}")
            return
        print(f"Deleting Milvus docs where source=mysql_schema from {collection}", flush=True)
        result = client.delete(
            collection_name=collection,
            filter='source == "mysql_schema"',
            timeout=_TIMEOUT_SECONDS,
        )
        print(f"Milvus deleted legacy mysql_schema docs: {result}")
    finally:
        client.close()


def cleanup_elasticsearch() -> None:
    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        print("Elasticsearch package is not installed, skipped")
        return

    es_url = (
        settings.es.address
        if settings.es.address.startswith("http")
        else f"http://{settings.es.address}"
    )
    print(f"Connecting Elasticsearch: {es_url}", flush=True)
    es = Elasticsearch(es_url, request_timeout=_TIMEOUT_SECONDS)
    if not es.indices.exists(index=settings.es.index):
        print(f"Elasticsearch index not found: {settings.es.index}")
        return
    print(f"Deleting Elasticsearch docs where metadata.source=mysql_schema from {settings.es.index}", flush=True)
    result = es.delete_by_query(
        index=settings.es.index,
        query={"term": {"metadata.source.keyword": "mysql_schema"}},
        conflicts="proceed",
        refresh=True,
        request_timeout=_TIMEOUT_SECONDS,
    )
    print(f"Elasticsearch deleted legacy mysql_schema docs: {result}")


def main() -> None:
    cleanup_milvus()
    cleanup_elasticsearch()


if __name__ == "__main__":
    main()
