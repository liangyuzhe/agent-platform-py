"""主调度图：多场景意图分类 -> 子图分发。"""

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from agents.flow.state import FinalGraphState
from agents.model.chat_model import get_chat_model
from agents.tool.storage.checkpoint import get_checkpointer
from agents.tool.storage.domain_summary import get_domain_summary
from agents.config.settings import settings

# 意图类别
_INTENT_SQL = "sql_query"
_INTENT_ANOMALY = "anomaly_detect"
_INTENT_RECONCILIATION = "reconciliation"
_INTENT_REPORT = "report"
_INTENT_AUDIT = "audit"
_INTENT_KNOWLEDGE = "knowledge"
_INTENT_CHAT = "chat"

_INTENTS = [
    _INTENT_SQL, _INTENT_ANOMALY, _INTENT_RECONCILIATION,
    _INTENT_REPORT, _INTENT_AUDIT, _INTENT_KNOWLEDGE, _INTENT_CHAT,
]


async def classify_intent(state: FinalGraphState) -> dict:
    """意图分类：根据用户问题路由到对应子图。"""
    model = get_chat_model(settings.chat_model_type)

    domain = await get_domain_summary()

    # 构建对话历史上下文
    chat_history = state.get("chat_history", [])
    history_context = ""
    if chat_history:
        recent = chat_history[-6:]  # 最近 3 轮
        lines = [f"[{h['role']}]: {h['content']}" for h in recent]
        history_context = "\n\n最近对话历史:\n" + "\n".join(lines)

    response = await model.ainvoke([
        HumanMessage(content=f"""请判断以下用户问题的意图类型，只回答意图类别名称。

数据库领域摘要：
{domain if domain else "（暂无领域摘要）"}

意图类别：
- sql_query：查询数据库中的信息，如用户、部门、角色、余额、交易记录等
- anomaly_detect：分析数据异常波动、异常归因，如"为什么 USDT 流水突然涨了"
- reconciliation：多表/多系统资金明细核对，如"核对链上链下资金是否一致"
- report：生成报告，如日报、周报、月报、季报
- audit：审计追踪、凭证查询，如"查一下凭证 TX001 的流转"
- knowledge：财务制度、会计准则、合规规则查询
- chat：闲聊、通用问答、与上述无关的问题
{history_context}

用户问题: {state['query']}

只回答一个意图类别名称，不要解释。""")
    ])

    intent = response.content.strip().lower()
    # 匹配最接近的意图
    matched = _INTENT_CHAT
    for valid_intent in _INTENTS:
        if valid_intent in intent:
            matched = valid_intent
            break

    return {"intent": matched}


async def sql_react(state: FinalGraphState, config=None) -> dict:
    """SQL React 子图。"""
    from agents.flow.sql_react import build_sql_react_graph
    sql_graph = build_sql_react_graph()
    result = await sql_graph.ainvoke(
        {
            "query": state["query"],
            "chat_history": state.get("chat_history", []),
        },
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
    """条件路由：根据意图分发。

    Phase 1: sql_query -> sql_react, 其他 -> chat_direct
    Phase 2: anomaly_detect -> anomaly_graph, report -> report_graph, 等
    """
    intent = state.get("intent", _INTENT_CHAT)
    if intent == _INTENT_SQL:
        return "sql_react"
    # Phase 2 逐步替换为专用子图
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
