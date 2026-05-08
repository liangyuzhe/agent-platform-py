"""意图调度图：意图分类 + 上下文重写 -> 子图分发。

/classify 端点同时完成意图识别和查询重写，返回 intent + rewritten_query。
invoke 端点接收预分类结果，跳过重复 LLM 调用。
"""

import json
import logging

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

from agents.flow.state import FinalGraphState
from agents.model.chat_model import get_chat_model
from agents.tool.storage.checkpoint import get_checkpointer
from agents.tool.storage.domain_summary import get_domain_summary
from agents.config.settings import settings

logger = logging.getLogger(__name__)

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

_CLASSIFY_SYSTEM_PROMPT = """你是一个智能助手，同时完成两个任务：

1. **意图分类**：根据数据库领域摘要和用户问题，判断意图类别
2. **查询重写**：结合对话历史，将代词化/省略的查询重写为独立完整的查询

意图类别说明：
- sql_query：用户想查询数据库中存储的结构化数据（必须与数据库领域摘要中的表/字段相关）
- anomaly_detect：分析数据异常波动、异常归因
- reconciliation：多表/多系统资金明细核对
- report：生成报告，如日报、周报、月报、季报
- audit：审计追踪、凭证查询
- knowledge：财务制度、会计准则、合规规则等知识查询
- chat：闲聊、通用问答、或问题与数据库领域无关

重要判断原则：
- 只有当问题明确指向数据库中的数据时，才归类为 sql_query
- 如果问题涉及的是公开信息、通用知识、股市行情等非数据库内容，应归类为 chat
- 结合对话历史重写查询时，只补充对话中明确提到的上下文，不要添加对话中没有的信息

请严格按以下 JSON 格式返回，不要有其他内容：
{{"intent": "意图类别", "rewritten_query": "重写后的独立查询"}}"""


async def classify_intent(state: FinalGraphState) -> dict:
    """意图分类 + 上下文重写：返回 intent 和 rewritten_query。

    如果 state 中已有 intent 和 rewritten_query（外部预分类），直接返回。
    """
    # 外部已传入 intent 且有 rewritten_query，跳过 LLM
    existing_intent = state.get("intent", "")
    existing_rewrite = state.get("rewritten_query", "")
    if existing_intent and existing_intent in _INTENTS and existing_rewrite:
        return {"intent": existing_intent, "rewritten_query": existing_rewrite}

    # 只有 intent 没有 rewritten_query（兼容旧版），跳过分类但不跳过重写
    if existing_intent and existing_intent in _INTENTS and not existing_rewrite:
        return {"intent": existing_intent, "rewritten_query": state.get("query", "")}

    model = get_chat_model(settings.chat_model_type)
    domain = await get_domain_summary()

    # 构建对话历史上下文
    chat_history = state.get("chat_history", [])
    history_context = ""
    if chat_history:
        recent = chat_history[-6:]  # 最近 3 轮
        lines = [f"[{h['role']}]: {h['content']}" for h in recent]
        history_context = "\n\n最近对话历史:\n" + "\n".join(lines)

    user_msg = f"""数据库领域摘要：
{domain if domain else "（暂无领域摘要）"}
{history_context}

用户问题: {state['query']}"""

    response = await model.ainvoke([
        SystemMessage(content=_CLASSIFY_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    raw = response.content.strip()

    # 解析 JSON 响应
    intent = _INTENT_CHAT
    rewritten_query = state.get("query", "")

    try:
        # 处理可能的 markdown 代码块包裹
        clean = raw
        if clean.startswith("```"):
            lines = clean.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.strip() == "```" and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            clean = "\n".join(json_lines)

        data = json.loads(clean)
        raw_intent = data.get("intent", "").strip().lower()
        rewritten_query = data.get("rewritten_query", "").strip() or state.get("query", "")

        # 匹配意图
        for valid_intent in _INTENTS:
            if valid_intent in raw_intent:
                intent = valid_intent
                break
    except (json.JSONDecodeError, AttributeError):
        # JSON 解析失败，尝试旧版纯文本匹配
        for valid_intent in _INTENTS:
            if valid_intent in raw.lower():
                intent = valid_intent
                break

    logger.info("classify_intent: intent=%s, query='%s' -> '%s'", intent, state["query"], rewritten_query)
    return {"intent": intent, "rewritten_query": rewritten_query}


async def sql_react(state: FinalGraphState, config=None) -> dict:
    """SQL React 子图。"""
    from agents.flow.sql_react import build_sql_react_graph
    sql_graph = build_sql_react_graph()
    query = state.get("query", "")
    result = await sql_graph.ainvoke(
        {
            "query": query,
            "rewritten_query": state.get("rewritten_query", ""),
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
    rewritten = state.get("rewritten_query", "")
    result = await rag_graph.ainvoke({
        "input": {
            "session_id": state["session_id"],
            "query": rewritten or state["query"],
            "rewritten_query": rewritten,
        },
    })
    return {
        "answer": result.get("answer", ""),
        "status": "completed",
    }


def route_intent(state: FinalGraphState) -> str:
    """条件路由：根据意图分发。"""
    intent = state.get("intent", _INTENT_CHAT)
    if intent == _INTENT_SQL:
        return "sql_react"
    return "chat_direct"


def build_final_graph():
    """构建意图调度图。"""
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
