"""Admin endpoints: schema refresh, cache management."""

import logging
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


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
