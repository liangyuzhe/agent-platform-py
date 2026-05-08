"""Generate evaluation dataset from semantic schema metadata.

Uses an LLM to produce (query, relevant_doc_ids) pairs from MySQL
``t_semantic_model``. Each query simulates a realistic user question that
would require specific tables to answer.

Output format (JSONL):
{"query": "查询所有员工的薪资", "relevant_doc_ids": ["schema_t_employee", "schema_t_salary"]}
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from agents.config.settings import settings
from agents.model.chat_model import get_chat_model, init_chat_models

logger = logging.getLogger(__name__)

_DEFAULT_EXCLUDED_TABLES = {
    "domain_summary",
    "t_semantic_model",
    "t_business_knowledge",
    "t_agent_knowledge",
    "t_document_metadata",
}


def _excluded_tables() -> set[str]:
    """Tables excluded from business evaluation datasets.

    Internal metadata/knowledge tables are useful for the application runtime
    but should not become NL2SQL business evaluation targets.
    """
    configured = os.getenv("EVAL_EXCLUDE_TABLES", "")
    extra = {item.strip() for item in configured.split(",") if item.strip()}
    return _DEFAULT_EXCLUDED_TABLES | extra


def _schema_rows_to_docs(rows: list[dict]) -> list[dict]:
    """Render t_semantic_model rows into schema-doc-like dictionaries."""
    excluded = _excluded_tables()
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        table_name = row.get("table_name")
        if table_name and table_name not in excluded:
            grouped.setdefault(table_name, []).append(row)

    docs: list[dict] = []
    for table_name, columns in grouped.items():
        lines = [f"表名: {table_name}", "字段:"]
        for col in columns:
            parts = [f"  {col.get('column_name', '')}"]
            if col.get("column_type"):
                parts.append(str(col["column_type"]))
            if col.get("is_pk"):
                parts.append("PRIMARY KEY")
            if col.get("is_fk") and col.get("ref_table"):
                parts.append(f"REFERENCES {col['ref_table']}({col.get('ref_column', '')})")
            if col.get("column_comment"):
                parts.append(f"-- {col['column_comment']}")
            if col.get("business_name"):
                parts.append(f"[业务名: {col['business_name']}]")
            if col.get("synonyms"):
                parts.append(f"[同义词: {col['synonyms']}]")
            if col.get("business_description"):
                parts.append(f"[描述: {col['business_description']}]")
            lines.append(" ".join(parts))

        doc_id = f"schema_{table_name}"
        docs.append({
            "pk": doc_id,
            "text": "\n".join(lines),
            "table_name": table_name,
            "doc_id": doc_id,
        })

    return docs


def _parse_eval_items(raw: str) -> list[dict]:
    """Parse JSONL-ish LLM output into evaluation items."""
    items = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            if line.startswith("{"):
                item = json.loads(line)
            else:
                start = line.find("{")
                end = line.rfind("}") + 1
                if start < 0 or end <= start:
                    continue
                item = json.loads(line[start:end])
            if "query" in item and "relevant_doc_ids" in item:
                items.append(item)
        except json.JSONDecodeError:
            continue
    return items


def _fetch_all_schema_docs() -> list[dict]:
    """Fetch schema documents from MySQL t_semantic_model as seed material."""
    import pymysql

    conn = pymysql.connect(
        host=settings.mysql.host,
        port=settings.mysql.port,
        user=settings.mysql.username,
        password=settings.mysql.password,
        database=settings.mysql.database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name, column_name, column_type, column_comment, "
                "is_pk, is_fk, ref_table, ref_column, "
                "business_name, synonyms, business_description "
                "FROM t_semantic_model "
                "ORDER BY table_name, column_name"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return _schema_rows_to_docs(rows)


async def generate_eval_dataset(
    num_queries_per_table: int = 3,
    output_path: str | Path = "eval_dataset.jsonl",
) -> list[dict]:
    """Generate evaluation dataset using LLM.

    For each table schema, generate `num_queries_per_table` queries that
    would require that table (and potentially related tables) to answer.

    Returns the dataset as a list of dicts and writes to JSONL file.
    """
    docs = _fetch_all_schema_docs()
    if not docs:
        logger.error("No schema metadata found in t_semantic_model")
        return []

    logger.info("Found %d semantic schema documents", len(docs))

    # Build schema text for LLM context
    schema_parts = []
    doc_id_map = {}  # table_name -> doc_id
    for doc in docs:
        table_name = doc.get("table_name", "")
        doc_id = doc.get("doc_id", doc.get("pk", ""))
        text = doc.get("text", "")
        schema_parts.append(text)
        if table_name:
            doc_id_map[table_name] = doc_id

    all_schemas = "\n\n".join(schema_parts)
    table_list = ", ".join(sorted(doc_id_map.keys()))

    init_chat_models()
    model = get_chat_model(settings.chat_model_type)

    system = SystemMessage(content=(
        "你是一个数据集标注专家。你的任务是根据数据库表结构，生成用于评估检索系统的测试数据集。\n\n"
        "规则：\n"
        "1. 每条数据包含一个 query（模拟真实用户问题）和对应的 relevant_doc_ids（回答该问题必须查询的表）\n"
        "2. query 应该是自然的中文问题，模拟业务人员的实际提问方式\n"
        "3. relevant_doc_ids 必须使用 schema_{表名} 的格式\n"
        "4. 包含单表查询和多表关联查询\n"
        "5. 问题类型多样化：简单查询、统计分析、条件筛选、排序等\n"
        "6. 不要生成与数据库内容无关的问题\n\n"
        "输出格式：每行一个 JSON 对象，不要输出其他内容。\n"
        '{"query": "问题内容", "relevant_doc_ids": ["schema_t_xxx", "schema_t_yyy"]}'
    ))

    # Generate in batches per table to ensure coverage
    all_items = []

    for table_name, doc_id in sorted(doc_id_map.items()):
        human = HumanMessage(content=(
            f"以下是数据库中所有表的结构：\n\n{all_schemas}\n\n"
            f"当前表: {table_name} (doc_id: {doc_id})\n"
            f"所有可用表: {table_list}\n\n"
            f"请为表 {table_name} 生成 {num_queries_per_table} 个评估问题。\n"
            f"其中至少 1 个是单表查询（只涉及 {table_name}），"
            f"至少 1 个是多表关联查询（涉及 {table_name} 和其他表）。\n\n"
            f"每行输出一个 JSON 对象，格式：\n"
            f'{{"query": "问题", "relevant_doc_ids": ["schema_t_xxx"]}}'
        ))

        response = await model.ainvoke([system, human])
        raw = response.content.strip()

        all_items.extend(_parse_eval_items(raw))

    # Deduplicate by query text
    seen = set()
    unique_items = []
    for item in all_items:
        if item["query"] not in seen:
            seen.add(item["query"])
            unique_items.append(item)

    # Write to file
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for item in unique_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        "Generated %d evaluation queries (deduplicated from %d), saved to %s",
        len(unique_items), len(all_items), output,
    )
    return unique_items
