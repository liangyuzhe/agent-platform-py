"""Admin endpoints: schema refresh, cache management, semantic model CRUD."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Semantic Model CRUD
# ---------------------------------------------------------------------------

class SemanticModelItem(BaseModel):
    table_name: str
    column_name: str
    business_name: str = ""
    synonyms: str = ""
    business_description: str = ""


class SemanticModelBatchImport(BaseModel):
    items: list[SemanticModelItem]


class RefreshResponse(BaseModel):
    success: bool
    message: str
    chunk_count: int = 0


@router.post("/refresh-schemas", response_model=RefreshResponse)
async def refresh_schemas():
    """Full re-index of MySQL table schemas + regenerate domain summary."""
    try:
        from agents.rag.schema_indexer import index_mysql_schemas

        result = await index_mysql_schemas()
        return RefreshResponse(
            success=True,
            message=f"Schema 索引完成，共 {result['chunk_count']} 个表",
            chunk_count=result["chunk_count"],
        )
    except Exception as e:
        logger.warning("Schema refresh failed: %s", e)
        return RefreshResponse(
            success=False,
            message=f"Schema 刷新失败: {e}",
        )


def _get_semantic_conn():
    import pymysql
    from agents.config.settings import settings

    return pymysql.connect(
        host=settings.mysql.host,
        port=settings.mysql.port,
        user=settings.mysql.username,
        password=settings.mysql.password,
        database=settings.mysql.database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


@router.get("/semantic-model")
async def list_semantic_model(table_name: str = None):
    """List all semantic model entries, optionally filtered by table_name."""
    conn = _get_semantic_conn()
    try:
        with conn.cursor() as cur:
            if table_name:
                cur.execute(
                    "SELECT * FROM t_semantic_model WHERE table_name=%s ORDER BY table_name, column_name",
                    (table_name,),
                )
            else:
                cur.execute("SELECT * FROM t_semantic_model ORDER BY table_name, column_name")
            rows = cur.fetchall()
        return {"items": rows, "count": len(rows)}
    finally:
        conn.close()


@router.post("/semantic-model")
async def upsert_semantic_model(item: SemanticModelItem):
    """Create or update a semantic model entry."""
    conn = _get_semantic_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO t_semantic_model (table_name, column_name, business_name, synonyms, business_description)
                   VALUES (%s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                       business_name=VALUES(business_name),
                       synonyms=VALUES(synonyms),
                       business_description=VALUES(business_description)""",
                (item.table_name, item.column_name, item.business_name, item.synonyms, item.business_description),
            )
        conn.commit()
        return {"success": True, "message": f"Saved {item.table_name}.{item.column_name}"}
    finally:
        conn.close()


@router.post("/semantic-model/batch")
async def batch_import_semantic_model(body: SemanticModelBatchImport):
    """Batch import semantic model entries."""
    conn = _get_semantic_conn()
    try:
        with conn.cursor() as cur:
            for item in body.items:
                cur.execute(
                    """INSERT INTO t_semantic_model (table_name, column_name, business_name, synonyms, business_description)
                       VALUES (%s, %s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE
                           business_name=VALUES(business_name),
                           synonyms=VALUES(synonyms),
                           business_description=VALUES(business_description)""",
                    (item.table_name, item.column_name, item.business_name, item.synonyms, item.business_description),
                )
        conn.commit()
        return {"success": True, "message": f"Imported {len(body.items)} entries"}
    finally:
        conn.close()


@router.delete("/semantic-model/{table_name}/{column_name}")
async def delete_semantic_model(table_name: str, column_name: str):
    """Delete a semantic model entry."""
    conn = _get_semantic_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM t_semantic_model WHERE table_name=%s AND column_name=%s",
                (table_name, column_name),
            )
        conn.commit()
        return {"success": True, "message": f"Deleted {table_name}.{column_name}"}
    finally:
        conn.close()
