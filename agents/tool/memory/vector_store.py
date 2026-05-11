"""Long-term vector memory storage.

Short-term memory stays in the session sliding window, medium-term memory is
the session summary, and archived turns are stored here for semantic recall.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime

from langchain_core.documents import Document

from agents.config.settings import settings
from agents.tool.memory.session import Message

logger = logging.getLogger(__name__)

_SOURCE = "conversation_memory"


def _format_messages(messages: list[Message]) -> str:
    return "\n".join(f"[{m.role}]: {m.content}" for m in messages)


def index_long_term_memory(session_id: str, messages: list[Message], summary: str = "") -> str:
    """Persist archived conversation messages into the vector collection.

    This is best-effort by design. If Milvus or embeddings are unavailable the
    online chat flow should still complete using short-term and summary memory.
    """
    if not settings.memory.enable_long_term_vector or not messages:
        return ""

    content = _format_messages(messages).strip()
    if not content:
        return ""

    archive_text = content
    if summary:
        archive_text = f"[滚动摘要]\n{summary}\n\n[归档对话]\n{content}"

    digest = hashlib.sha1(f"{session_id}:{content}".encode("utf-8")).hexdigest()[:16]
    doc_id = f"memory_{session_id}_{digest}"

    try:
        from agents.rag.retriever import _get_embeddings
        from pymilvus import MilvusClient

        vector = _get_embeddings().embed_query(archive_text)
        client = MilvusClient(uri=f"http://{settings.milvus.addr}")
        try:
            client.delete(collection_name=settings.milvus.collection_name, ids=[doc_id])
        except Exception:
            pass
        client.insert(
            collection_name=settings.milvus.collection_name,
            data=[{
                "pk": doc_id,
                "text": archive_text,
                "vector": vector,
                "source": _SOURCE,
                "table_name": "",
                "doc_id": doc_id,
                "session_id": session_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }],
        )
        client.close()
        logger.info("Indexed long-term memory %s for session %s", doc_id, session_id)
        return doc_id
    except Exception as e:
        logger.warning("Long-term memory indexing failed for session %s: %s", session_id, e)
        return ""


def recall_long_term_memory(session_id: str, query: str, top_k: int | None = None) -> list[Document]:
    """Recall archived semantic memories for one session."""
    if not settings.memory.enable_long_term_vector or not session_id or not query:
        return []

    limit = top_k or settings.memory.long_term_top_k
    try:
        from agents.rag.retriever import _get_embeddings
        from pymilvus import MilvusClient

        vector = _get_embeddings().embed_query(query)
        client = MilvusClient(uri=f"http://{settings.milvus.addr}")
        results = client.search(
            collection_name=settings.milvus.collection_name,
            data=[vector],
            limit=limit,
            filter=f'source == "{_SOURCE}" and session_id == "{session_id}"',
            output_fields=["text", "doc_id", "session_id"],
        )
        client.close()
        docs: list[Document] = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            docs.append(Document(
                page_content=entity.get("text", ""),
                metadata={
                    "source": _SOURCE,
                    "doc_id": entity.get("doc_id", ""),
                    "session_id": entity.get("session_id", session_id),
                    "score": hit.get("distance", 0),
                    "retriever_source": "milvus",
                },
            ))
        return docs
    except Exception as e:
        logger.warning("Long-term memory recall failed for session %s: %s", session_id, e)
        return []
