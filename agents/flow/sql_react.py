"""SQL React 图：自然语言 -> SQL -> 审批 -> 执行，支持自动纠错重试。"""

import asyncio
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langchain_core.messages import HumanMessage, SystemMessage

from agents.flow.state import SQLReactState
from agents.model.chat_model import get_chat_model
from agents.model.format_tool import create_format_tool
from agents.tool.sql_tools.safety import SQLSafetyChecker
from agents.rag.retriever import get_hybrid_retriever
from agents.tool.storage.checkpoint import get_checkpointer
from agents.config.settings import settings

try:
    from elasticsearch import Elasticsearch
except ImportError:
    Elasticsearch = None

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


async def sql_retrieve(state: SQLReactState) -> dict:
    """检索表结构信息。"""
    retriever = get_hybrid_retriever()
    docs = await asyncio.to_thread(retriever.retrieve, state["query"])
    return {"docs": docs}


async def check_docs(state: SQLReactState) -> dict:
    """检查是否检索到相关表结构。

    如果精确检索没有命中（如查询人名等非表结构关键词），回退到
    获取全部已索引的 schema，让 LLM 自行判断使用哪个表。
    """
    docs = state.get("docs", [])
    if not docs:
        # 回退：直接从 ES 拉取所有 schema 文档（不做相似度过滤）
        try:
            from agents.rag.schema_indexer import _SCHEMA_SOURCE

            es_url = settings.es.address
            es = Elasticsearch(es_url)
            resp = es.search(
                index=settings.es.index,
                query={"term": {"source": _SCHEMA_SOURCE}},
                size=50,
            )
            from langchain_core.documents import Document
            docs = [
                Document(
                    page_content=hit["_source"]["text"],
                    metadata={"source": _SCHEMA_SOURCE, "table_name": hit["_source"].get("table_name", "")},
                )
                for hit in resp["hits"]["hits"]
            ]
            if docs:
                logger.info("check_docs: fallback retrieved %d schema docs", len(docs))
                return {"docs": docs}
        except Exception as e:
            logger.warning("check_docs fallback failed: %s", e)

        return {
            "answer": "未找到相关的数据库表结构信息，无法生成 SQL。请先上传数据库表结构文档。",
            "is_sql": False,
        }
    return {}


async def sql_generate(state: SQLReactState) -> dict:
    """LLM 生成 SQL。"""
    model = get_chat_model(settings.chat_model_type)
    model_with_tools = model.bind_tools([create_format_tool()])

    docs_text = "\n".join([d.page_content for d in state.get("docs", [])])

    # 如果有修改意见，加入上下文
    refine_context = ""
    if state.get("refine_feedback"):
        refine_context = f"\n修改意见: {state['refine_feedback']}"

    # 如果有执行历史（纠错场景），加入上下文
    history_context = ""
    execution_history = state.get("execution_history", [])
    if execution_history:
        history_lines = []
        for i, h in enumerate(execution_history, 1):
            entry = f"第{i}次尝试: SQL={h['sql']}"
            if h.get("error"):
                entry += f"\n  错误: {h['error']}"
            elif h.get("result"):
                entry += f"\n  结果: {h['result'][:200]}"
            history_lines.append(entry)
        history_context = f"\n执行历史:\n" + "\n".join(history_lines)

    messages = [
        SystemMessage(content=f"""你是一个 SQL 专家。根据用户的问题和数据库表结构信息，生成正确的 SQL 查询。

表结构信息:
        {docs_text}{refine_context}{history_context}

要求：
1. 使用 MySQL 语法
2. 只生成 SELECT 查询（禁止 DROP/DELETE/TRUNCATE 等危险操作）
3. 如果有执行历史和错误信息，请分析错误原因并生成修正后的 SQL
4. 使用 format_response 工具输出结果"""),
        HumanMessage(content=state["query"]),
    ]

    response = await model_with_tools.ainvoke(messages)

    # 解析结构化输出
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        args = tool_call["args"]
        answer_text = args.get("answer", "")
        is_sql = args.get("is_sql", False)
        return {
            "answer": answer_text,
            "sql": answer_text if is_sql else answer_text,
            "is_sql": is_sql,
            "error": None,  # 清除之前的错误
        }

    return {"answer": response.content, "sql": response.content, "is_sql": False, "error": None}


async def safety_check(state: SQLReactState) -> dict:
    """SQL 安全分析。"""
    if not state.get("is_sql"):
        return {"safety_report": None}

    checker = SQLSafetyChecker()
    report = checker.check(state["sql"])

    if not report.is_safe:
        return {
            "safety_report": {
                "risks": report.risks,
                "estimated_rows": report.estimated_rows,
                "required_permissions": report.required_permissions,
            },
            "answer": f"SQL 安全检查未通过: {', '.join(report.risks)}",
            "is_sql": False,
        }

    return {"safety_report": None}


def approve(state: SQLReactState) -> dict:
    """人工审批 SQL。使用 interrupt 暂停图执行，等待用户确认。"""
    result = interrupt({
        "sql": state["sql"],
        "message": "请确认是否执行以上 SQL？",
    })

    if result.get("approved"):
        return {"approved": True}

    return {
        "approved": False,
        "answer": result.get("feedback", "SQL 已被拒绝。"),
        "is_sql": False,
    }


async def execute_sql(state: SQLReactState) -> dict:
    """通过 MCP 执行 SQL。捕获错误用于自动纠错。"""
    from agents.tool.sql_tools.mcp_client import execute_sql as mcp_execute

    sql = state["sql"]
    execution_history = list(state.get("execution_history", []))

    try:
        result = await mcp_execute(sql)
        execution_history.append({"sql": sql, "result": result, "error": None})
        return {
            "result": result,
            "answer": result,
            "error": None,
            "execution_history": execution_history,
        }
    except Exception as e:
        error_msg = str(e)
        logger.warning("SQL execution failed (retry %d/%d): %s", state.get("retry_count", 0), _MAX_RETRIES, error_msg)
        execution_history.append({"sql": sql, "result": None, "error": error_msg})
        return {
            "result": f"SQL 执行失败: {error_msg}",
            "answer": f"SQL 执行失败: {error_msg}",
            "error": error_msg,
            "execution_history": execution_history,
        }


async def error_analysis(state: SQLReactState) -> dict:
    """分析 SQL 执行错误，生成修正建议。"""
    model = get_chat_model(settings.chat_model_type)

    docs_text = "\n".join([d.page_content for d in state.get("docs", [])])
    last_error = state.get("error", "")
    last_sql = state.get("sql", "")
    retry_count = state.get("retry_count", 0)

    response = await model.ainvoke([
        SystemMessage(content=f"""你是一个 SQL 调试专家。以下 SQL 执行失败，请分析错误原因并给出修正建议。

表结构信息:
{docs_text}

失败的 SQL:
{last_sql}

错误信息:
{last_error}

请简要分析错误原因（1-2 句话），并给出修正建议。"""),
        HumanMessage(content=f"这是第 {retry_count} 次重试，请分析错误并给出修正建议。"),
    ])

    return {
        "refine_feedback": response.content.strip(),
        "retry_count": retry_count + 1,
    }


def build_sql_react_graph():
    """构建 SQL React 图，支持自动纠错重试。

    流程: retrieve → check_docs → generate → safety_check → approve → execute
                                                              ↓
                                                         error_analysis (失败时)
                                                              ↓
                                                          generate (重试，最多3次)
    """
    graph = StateGraph(SQLReactState)

    graph.add_node("sql_retrieve", sql_retrieve)
    graph.add_node("check_docs", check_docs)
    graph.add_node("sql_generate", sql_generate)
    graph.add_node("safety_check", safety_check)
    graph.add_node("approve", approve)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("error_analysis", error_analysis)

    graph.add_edge(START, "sql_retrieve")
    graph.add_edge("sql_retrieve", "check_docs")

    def route_after_check(state: SQLReactState) -> str:
        if state.get("is_sql") is False and state.get("answer"):
            return END
        return "sql_generate"

    graph.add_conditional_edges("check_docs", route_after_check)
    graph.add_edge("sql_generate", "safety_check")

    def route_after_safety(state: SQLReactState) -> str:
        if state.get("is_sql"):
            return "approve"
        return END

    graph.add_conditional_edges("safety_check", route_after_safety)

    def route_after_approve(state: SQLReactState) -> str:
        if state.get("approved"):
            return "execute_sql"
        return END

    graph.add_conditional_edges("approve", route_after_approve)

    def route_after_execute(state: SQLReactState) -> str:
        # 执行成功
        if not state.get("error"):
            return END
        # 重试次数未达上限，进入错误分析
        if state.get("retry_count", 0) < _MAX_RETRIES:
            return "error_analysis"
        # 超过最大重试次数
        return END

    graph.add_conditional_edges("execute_sql", route_after_execute)
    graph.add_edge("error_analysis", "sql_generate")

    checkpointer = get_checkpointer()
    return graph.compile(checkpointer=checkpointer)
