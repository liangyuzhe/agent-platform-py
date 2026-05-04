"""SQL React 图：自然语言 -> SQL -> 审批 -> 执行。"""

import asyncio
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langchain_core.messages import HumanMessage, SystemMessage

from agents.flow.state import SQLReactState
from agents.model.chat_model import get_chat_model
from agents.model.format_tool import create_format_tool
from agents.tool.sql_tools.safety import SQLSafetyChecker
from agents.rag.retriever import HybridRetriever
from agents.tool.storage.checkpoint import get_checkpointer
from agents.config.settings import settings

logger = logging.getLogger(__name__)


async def sql_retrieve(state: SQLReactState) -> dict:
    """检索表结构信息。"""
    retriever = HybridRetriever()
    docs = await asyncio.to_thread(retriever.retrieve, state["query"])
    return {"docs": docs}


async def check_docs(state: SQLReactState) -> dict:
    """检查是否检索到相关表结构。"""
    docs = state.get("docs", [])
    if not docs:
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
        refine_context = f"\n用户修改意见: {state['refine_feedback']}"

    messages = [
        SystemMessage(content=f"""你是一个 SQL 专家。根据用户的问题和数据库表结构信息，生成正确的 SQL 查询。

表结构信息:
{docs_text}{refine_context}

要求：
1. 使用 MySQL 语法
2. 只生成 SELECT 查询（禁止 DROP/DELETE/TRUNCATE 等危险操作）
3. 使用 format_response 工具输出结果"""),
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
        }

    return {"answer": response.content, "sql": response.content, "is_sql": False}


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
    """通过 MCP 执行 SQL。"""
    from agents.tool.sql_tools.mcp_client import execute_sql as mcp_execute
    try:
        result = await mcp_execute(state["sql"])
    except Exception as e:
        logger.warning("SQL execution failed: %s", e)
        result = f"SQL 执行失败: {e}"
    return {"result": result, "answer": result}


def build_sql_react_graph():
    """构建 SQL React 图。

    流程: retrieve → check_docs → generate → safety_check → approve → execute
    """
    graph = StateGraph(SQLReactState)

    graph.add_node("sql_retrieve", sql_retrieve)
    graph.add_node("check_docs", check_docs)
    graph.add_node("sql_generate", sql_generate)
    graph.add_node("safety_check", safety_check)
    graph.add_node("approve", approve)
    graph.add_node("execute_sql", execute_sql)

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
    graph.add_edge("execute_sql", END)

    checkpointer = get_checkpointer()
    return graph.compile(checkpointer=checkpointer)
