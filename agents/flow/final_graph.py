"""主调度图：意图分类 -> SQL 或 Chat 分发。"""

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from agents.flow.state import FinalGraphState
from agents.model.chat_model import get_chat_model
from agents.tool.storage.checkpoint import get_checkpointer
from agents.config.settings import settings


async def classify_intent(state: FinalGraphState) -> dict:
    """意图分类：SQL 还是 Chat？用小模型省成本。"""
    model = get_chat_model(settings.chat_model_type)

    response = await model.ainvoke([
        HumanMessage(content=f"""请判断以下用户问题的意图类型，只回答 "SQL" 或 "Chat"：

- SQL：需要查询数据库、统计数据、生成报表
- Chat：普通对话、知识问答、闲聊

用户问题: {state['query']}
""")
    ])

    intent = response.content.strip().upper()
    return {"intent": "SQL" if "SQL" in intent else "Chat"}


async def sql_react(state: FinalGraphState, config=None) -> dict:
    """SQL React 子图。"""
    from agents.flow.sql_react import build_sql_react_graph
    sql_graph = build_sql_react_graph()
    result = await sql_graph.ainvoke(
        {"query": state["query"]},
        config=config,
    )
    return {
        "sql": result.get("sql", ""),
        "result": result.get("result", ""),
        "answer": result.get("answer", ""),
        "status": "completed",
    }


async def chat_direct(state: FinalGraphState) -> dict:
    """普通对话，接入 RAG Chat 子图。"""
    from agents.flow.rag_chat import build_rag_chat_graph
    rag_graph = build_rag_chat_graph()
    result = await rag_graph.ainvoke({
        "input": {"session_id": state["session_id"], "query": state["query"]},
    })
    return {
        "answer": result.get("answer", ""),
        "status": "completed",
    }


def route_intent(state: FinalGraphState) -> str:
    """条件路由：根据意图分发。"""
    if state.get("intent") == "SQL":
        return "sql_react"
    return "chat_direct"


def build_final_graph():
    """构建主调度图。"""
    graph = StateGraph(FinalGraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("sql_react", sql_react)
    graph.add_node("chat_direct", chat_direct)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_intent)
    graph.add_edge("sql_react", END)
    graph.add_edge("chat_direct", END)

    checkpointer = get_checkpointer()
    return graph.compile(checkpointer=checkpointer)
