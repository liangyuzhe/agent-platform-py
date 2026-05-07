"""文档元数据 MySQL 存储。

提供 t_document_metadata 表的 CRUD 操作。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from agents.config.settings import settings

logger = logging.getLogger(__name__)

_TABLE_NAME = "t_document_metadata"


def _get_conn():
    import pymysql
    return pymysql.connect(
        host=settings.mysql.host,
        port=settings.mysql.port,
        user=settings.mysql.username,
        password=settings.mysql.password,
        database=settings.mysql.database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def ensure_doc_metadata_table():
    """确保 t_document_metadata 表存在。"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    doc_id VARCHAR(255) NOT NULL,
                    filename VARCHAR(500) DEFAULT '',
                    source VARCHAR(100) DEFAULT '',
                    session_id VARCHAR(255) DEFAULT '',
                    category VARCHAR(100) DEFAULT '',
                    tags JSON,
                    entities JSON,
                    summary TEXT,
                    hypothetical_questions JSON,
                    key_facts JSON,
                    chunk_count INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_source (source),
                    INDEX idx_session (session_id),
                    INDEX idx_doc_id (doc_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        conn.commit()
        logger.info("Ensured %s table exists", _TABLE_NAME)
    except Exception as e:
        logger.warning("Failed to create %s: %s", _TABLE_NAME, e)
    finally:
        conn.close()


def save_doc_metadata(
    doc_id: str,
    filename: str = "",
    source: str = "",
    session_id: str = "",
    category: str = "",
    tags: list[str] | None = None,
    entities: list[str] | None = None,
    summary: str = "",
    hypothetical_questions: list[str] | None = None,
    key_facts: list[str] | None = None,
    chunk_count: int = 0,
) -> None:
    """保存文档元数据到 MySQL。"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO {_TABLE_NAME}
                (doc_id, filename, source, session_id, category, tags, entities, summary, hypothetical_questions, key_facts, chunk_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    doc_id,
                    filename,
                    source,
                    session_id,
                    category,
                    json.dumps(tags or [], ensure_ascii=False),
                    json.dumps(entities or [], ensure_ascii=False),
                    summary,
                    json.dumps(hypothetical_questions or [], ensure_ascii=False),
                    json.dumps(key_facts or [], ensure_ascii=False),
                    chunk_count,
                ),
            )
        conn.commit()
        logger.info("Saved doc metadata: %s (%s)", doc_id, category)
    except Exception as e:
        logger.warning("Failed to save doc metadata for %s: %s", doc_id, e)
    finally:
        conn.close()
