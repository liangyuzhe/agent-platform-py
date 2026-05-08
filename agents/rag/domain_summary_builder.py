"""Build and persist a compact domain summary from semantic schema metadata."""

from __future__ import annotations

import logging

from langchain_core.documents import Document

from agents.rag.retriever import get_semantic_model_by_tables, load_full_table_metadata

logger = logging.getLogger(__name__)


def build_schema_docs_from_semantic_model() -> list[Document]:
    """Build schema summary documents from Redis/MySQL semantic metadata."""
    metadata = load_full_table_metadata()
    if not metadata:
        return []

    table_names = [m["table_name"] for m in metadata if m.get("table_name")]
    semantic = get_semantic_model_by_tables(table_names)

    docs: list[Document] = []
    for table in metadata:
        table_name = table.get("table_name", "")
        if not table_name:
            continue

        lines = [f"表名: {table_name}"]
        if table.get("table_comment"):
            lines[0] += f" -- {table['table_comment']}"

        columns = semantic.get(table_name, {})
        if columns:
            lines.append("字段:")
            for col_name, meta in columns.items():
                parts = [f"  {col_name}"]
                if meta.get("column_type"):
                    parts.append(str(meta["column_type"]))
                if meta.get("column_comment"):
                    parts.append(f"-- {meta['column_comment']}")
                if meta.get("business_name"):
                    parts.append(f"[业务名: {meta['business_name']}]")
                if meta.get("business_description"):
                    parts.append(f"[描述: {meta['business_description']}]")
                lines.append(" ".join(parts))

        docs.append(Document(page_content="\n".join(lines), metadata={"table_name": table_name}))

    return docs


async def generate_domain_summary(docs: list[Document] | None = None) -> str:
    """Use an LLM to generate and persist a compact domain summary."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from agents.model.chat_model import get_chat_model
    from agents.tool.storage.domain_summary import save_domain_summary

    schema_docs = docs if docs is not None else build_schema_docs_from_semantic_model()
    schemas_text = "\n\n".join(d.page_content for d in schema_docs)
    if not schemas_text.strip():
        logger.info("No schema metadata available for domain summary")
        return ""

    model = get_chat_model()
    response = await model.ainvoke([
        SystemMessage(content=(
            "你是一个数据库架构分析专家。请根据以下所有表结构信息，生成一段简洁的领域摘要。"
            "摘要需要说明：\n"
            "1. 这个数据库管理的是什么业务领域\n"
            "2. 包含哪些核心实体（表）及它们的关系\n"
            "3. 可以回答哪些类型的问题\n"
            "4. 适合哪些业务场景（如：数据查询、异常分析、资金核对、报告生成、审计追踪）\n\n"
            "要求：控制在 600 字以内，使用中文，语言简洁专业。"
        )),
        HumanMessage(content=f"以下是所有表结构：\n\n{schemas_text}"),
    ])

    summary = response.content.strip()
    await save_domain_summary(summary)
    logger.info("Domain summary generated (%d chars)", len(summary))
    return summary
