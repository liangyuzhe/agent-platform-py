"""Admin endpoints: semantic model & business knowledge CRUD."""

import logging
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.tool.knowledge_indexing import (
    reindex_agent_knowledge_documents,
    reindex_business_knowledge_documents,
)

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


# ---------------------------------------------------------------------------
# Business Knowledge CRUD
# ---------------------------------------------------------------------------

class BusinessKnowledgeItem(BaseModel):
    term: str
    formula: str
    synonyms: str = ""
    related_tables: str = ""


class BusinessKnowledgeBatchImport(BaseModel):
    items: list[BusinessKnowledgeItem]


@router.get("/business-knowledge")
async def list_business_knowledge():
    """List all business knowledge entries."""
    conn = _get_semantic_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM t_business_knowledge ORDER BY term")
            rows = cur.fetchall()
        return {"items": rows, "count": len(rows)}
    finally:
        conn.close()


@router.post("/business-knowledge")
async def upsert_business_knowledge(item: BusinessKnowledgeItem):
    """Create or update a business knowledge entry."""
    conn = _get_semantic_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO t_business_knowledge (term, formula, synonyms, related_tables)
                   VALUES (%s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                       formula=VALUES(formula),
                       synonyms=VALUES(synonyms),
                       related_tables=VALUES(related_tables)""",
                (item.term, item.formula, item.synonyms, item.related_tables),
            )
        conn.commit()
        return {"success": True, "message": f"Saved {item.term}"}
    finally:
        conn.close()


@router.post("/business-knowledge/batch")
async def batch_import_business_knowledge(body: BusinessKnowledgeBatchImport):
    """Batch import business knowledge entries."""
    conn = _get_semantic_conn()
    try:
        with conn.cursor() as cur:
            for item in body.items:
                cur.execute(
                    """INSERT INTO t_business_knowledge (term, formula, synonyms, related_tables)
                       VALUES (%s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE
                           formula=VALUES(formula),
                           synonyms=VALUES(synonyms),
                           related_tables=VALUES(related_tables)""",
                    (item.term, item.formula, item.synonyms, item.related_tables),
                )
        conn.commit()
        return {"success": True, "message": f"Imported {len(body.items)} entries"}
    finally:
        conn.close()


@router.delete("/business-knowledge/{term}")
async def delete_business_knowledge(term: str):
    """Delete a business knowledge entry."""
    conn = _get_semantic_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM t_business_knowledge WHERE term=%s", (term,))
        conn.commit()
        return {"success": True, "message": f"Deleted {term}"}
    finally:
        conn.close()


@router.post("/business-knowledge/reindex")
async def reindex_business_knowledge():
    """Re-index business knowledge into Milvus/ES."""
    try:
        count = await asyncio.to_thread(reindex_business_knowledge_documents)
        return {
            "success": True,
            "message": "Business knowledge re-indexed into Milvus/ES",
            "count": count,
        }
    except Exception as e:
        logger.warning("Business knowledge reindex failed: %s", e)
        return {"success": False, "message": f"Reindex failed: {e}"}


# ---------------------------------------------------------------------------
# Agent Knowledge CRUD (SQL Q&A few-shot)
# ---------------------------------------------------------------------------

class AgentKnowledgeItem(BaseModel):
    question: str
    sql_text: str
    description: str = ""
    category: str = ""


class AgentKnowledgeBatchImport(BaseModel):
    items: list[AgentKnowledgeItem]


@router.get("/agent-knowledge")
async def list_agent_knowledge():
    """List all agent knowledge entries."""
    conn = _get_semantic_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM t_agent_knowledge ORDER BY category, id")
            rows = cur.fetchall()
        return {"items": rows, "count": len(rows)}
    finally:
        conn.close()


@router.post("/agent-knowledge")
async def upsert_agent_knowledge(item: AgentKnowledgeItem):
    """Create or update an agent knowledge entry."""
    conn = _get_semantic_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO t_agent_knowledge (question, sql_text, description, category)
                   VALUES (%s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                       sql_text=VALUES(sql_text),
                       description=VALUES(description),
                       category=VALUES(category)""",
                (item.question, item.sql_text, item.description, item.category),
            )
        conn.commit()
        return {"success": True, "message": f"Saved: {item.question[:50]}"}
    finally:
        conn.close()


@router.post("/agent-knowledge/batch")
async def batch_import_agent_knowledge(body: AgentKnowledgeBatchImport):
    """Batch import agent knowledge entries."""
    conn = _get_semantic_conn()
    try:
        with conn.cursor() as cur:
            for item in body.items:
                cur.execute(
                    """INSERT INTO t_agent_knowledge (question, sql_text, description, category)
                       VALUES (%s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE
                           sql_text=VALUES(sql_text),
                           description=VALUES(description),
                           category=VALUES(category)""",
                    (item.question, item.sql_text, item.description, item.category),
                )
        conn.commit()
        return {"success": True, "message": f"Imported {len(body.items)} entries"}
    finally:
        conn.close()


@router.delete("/agent-knowledge/{knowledge_id}")
async def delete_agent_knowledge(knowledge_id: int):
    """Delete an agent knowledge entry."""
    conn = _get_semantic_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM t_agent_knowledge WHERE id=%s", (knowledge_id,))
        conn.commit()
        return {"success": True, "message": f"Deleted agent knowledge #{knowledge_id}"}
    finally:
        conn.close()


@router.post("/agent-knowledge/reindex")
async def reindex_agent_knowledge():
    """Re-index agent knowledge into Milvus/ES."""
    try:
        count = await asyncio.to_thread(reindex_agent_knowledge_documents)
        return {
            "success": True,
            "message": "Agent knowledge re-indexed into Milvus/ES",
            "count": count,
        }
    except Exception as e:
        logger.warning("Agent knowledge reindex failed: %s", e)
        return {"success": False, "message": f"Reindex failed: {e}"}


# ---------------------------------------------------------------------------
# Intent Rule CRUD
# ---------------------------------------------------------------------------

class IntentRuleItem(BaseModel):
    id: int | None = None
    name: str
    target_intent: str
    match_type: str = "contains"
    pattern: str
    rewrite_template: str = ""
    priority: int = 100
    confidence: float = 0.9
    enabled: bool = True
    description: str = ""


class QueryRouteRuleItem(BaseModel):
    id: int | None = None
    name: str
    route_signal: str
    match_type: str = "contains"
    pattern: str
    priority: int = 100
    confidence: float = 0.9
    enabled: bool = True
    description: str = ""


@router.get("/intent-rules")
async def list_intent_rule_items():
    """List configurable intent rules."""
    try:
        from agents.tool.storage.intent_rules import ensure_intent_rule_table, list_intent_rules

        ensure_intent_rule_table()
        items = list_intent_rules(enabled_only=False)
        return {"items": items, "count": len(items)}
    except Exception as e:
        logger.warning("List intent rules failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/intent-rules")
async def upsert_intent_rule_item(item: IntentRuleItem):
    """Create or update a configurable intent rule."""
    try:
        from agents.tool.storage.intent_rules import ensure_intent_rule_table, upsert_intent_rule

        ensure_intent_rule_table()
        rule_id = upsert_intent_rule(item.model_dump())
        return {"success": True, "id": rule_id, "message": f"Saved intent rule #{rule_id}"}
    except Exception as e:
        logger.warning("Save intent rule failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete("/intent-rules/{rule_id}")
async def delete_intent_rule_item(rule_id: int):
    """Delete a configurable intent rule."""
    try:
        from agents.tool.storage.intent_rules import delete_intent_rule

        delete_intent_rule(rule_id)
        return {"success": True, "message": f"Deleted intent rule #{rule_id}"}
    except Exception as e:
        logger.warning("Delete intent rule failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Complex Query Route Rule CRUD
# ---------------------------------------------------------------------------

@router.get("/query-route-rules")
async def list_query_route_rule_items():
    """List configurable complex-query route rules."""
    try:
        from agents.tool.storage.query_route_rules import ensure_query_route_rule_table, list_query_route_rules

        ensure_query_route_rule_table()
        items = list_query_route_rules(enabled_only=False)
        return {"items": items, "count": len(items)}
    except Exception as e:
        logger.warning("List query route rules failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/query-route-rules")
async def upsert_query_route_rule_item(item: QueryRouteRuleItem):
    """Create or update a configurable complex-query route rule."""
    try:
        from agents.tool.storage.query_route_rules import ensure_query_route_rule_table, upsert_query_route_rule

        ensure_query_route_rule_table()
        rule_id = upsert_query_route_rule(item.model_dump())
        return {"success": True, "id": rule_id, "message": f"Saved query route rule #{rule_id}"}
    except Exception as e:
        logger.warning("Save query route rule failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete("/query-route-rules/{rule_id}")
async def delete_query_route_rule_item(rule_id: int):
    """Delete a configurable complex-query route rule."""
    try:
        from agents.tool.storage.query_route_rules import delete_query_route_rule

        delete_query_route_rule(rule_id)
        return {"success": True, "message": f"Deleted query route rule #{rule_id}"}
    except Exception as e:
        logger.warning("Delete query route rule failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
