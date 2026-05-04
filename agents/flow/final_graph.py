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
        HumanMessage(content=f"""请判断以下用户问题的意图类型，只回答 "SQL" 或 "Chat"。

数据库中包含以下表：
- t_user: 用户信息（username, real_name, email, phone, gender, status）
- t_department: 部门信息（name, manager, parent_id, phone）
- t_role: 角色信息（name, code, description）
- t_user_role: 用户-角色关联
- t_user_department: 用户-部门关联

判断规则：
- SQL：涉及查询数据库中存储的信息，包括但不限于：用户信息、部门结构、角色权限、人员统计、关联关系等。例如："zhangsan是谁"、"有哪些部门"、"技术部有多少人"、"谁是管理员"
- Chat：与数据库无关的对话，如：天气、闲聊、通用知识问答

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
