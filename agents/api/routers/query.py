"""查询端点：意图分类 + 子图调用 + SQL 审批。"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel

from langgraph.types import Command

from agents.flow.dispatcher import build_final_graph
from agents.tool.trace.tracing import get_trace_callbacks
from agents.tool.memory.store import get_session, save_session
from agents.tool.memory.session import Message

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_user"
    intent: str = ""  # 前端预分类的意图，非空时跳过 LLM 分类
    rewritten_query: str = ""  # 前端预重写的查询，非空时跳过上下文重写


class QueryResponse(BaseModel):
    query: str
    answer: str
    status: str
    session_id: str
    pending_approval: bool = False
    sql: str = ""


class ClassifyResponse(BaseModel):
    intent: str
    rewritten_query: str  # 重写后的独立查询
    session_id: str


class ApproveRequest(BaseModel):
    session_id: str = "default_user"
    approved: bool = True
    feedback: str = ""


def _load_chat_history(session_id: str) -> list[dict]:
    """从 session store 加载对话历史。"""
    session = get_session(session_id)
    history = [{"role": m.role, "content": m.content} for m in session.history]
    if session.summary:
        history.insert(0, {"role": "system", "content": f"[对话摘要] {session.summary}"})
    logger.info("Loaded chat history for session %s: %d messages", session_id, len(history))
    return history


def _save_qa_to_session(session_id: str, query: str, answer: str) -> None:
    """将本轮 Q&A 保存到 session store。"""
    try:
        session = get_session(session_id)
        session.history.append(Message(role="user", content=query))
        session.history.append(Message(role="assistant", content=answer))
        save_session(session_id, session)
        logger.info("Saved Q&A to session %s, history length: %d", session_id, len(session.history))
    except Exception as e:
        logger.warning("Failed to save Q&A to session: %s", e)


def _save_pending_query(session_id: str, query: str) -> None:
    """中断时暂存原始 query，供 approve 后恢复。"""
    try:
        session = get_session(session_id)
        session.preferences["_pending_query"] = query
        save_session(session_id, session)
    except Exception as e:
        logger.warning("Failed to save pending query: %s", e)


def _pop_pending_query(session_id: str) -> str:
    """取出中断时暂存的 query。"""
    try:
        session = get_session(session_id)
        query = session.preferences.pop("_pending_query", "")
        if query:
            save_session(session_id, session)
        return query
    except Exception:
        return ""


def _make_config(session_id: str) -> dict:
    """构建包含 thread_id 和 trace callbacks 的 config。"""
    callbacks = get_trace_callbacks()
    config = {
        "configurable": {"thread_id": session_id},
    }
    if callbacks:
        config["callbacks"] = callbacks
    return config


def _extract_interrupt(result: dict) -> dict | None:
    """从结果中提取 interrupt 信息。"""
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None
    interrupt = interrupts[0] if isinstance(interrupts, list) else interrupts
    value = interrupt.value if hasattr(interrupt, "value") else interrupt
    if isinstance(value, list) and value:
        value = value[0]
    return value if isinstance(value, dict) else None


@router.post("/classify", response_model=ClassifyResponse)
async def classify_intent_endpoint(req: QueryRequest):
    """意图分类（非流式），前端据此选择流式端点。"""
    from agents.flow.dispatcher import classify_intent
    chat_history = _load_chat_history(req.session_id)
    result = await classify_intent({"query": req.query, "chat_history": chat_history})
    return ClassifyResponse(
        intent=result.get("intent", "chat"),
        rewritten_query=result.get("rewritten_query", req.query),
        session_id=req.session_id,
    )


@router.post("/invoke", response_model=QueryResponse)
async def query_invoke(req: QueryRequest):
    """查询调用：传入 intent 时跳过分类，直接路由到子图。"""
    graph = build_final_graph()
    config = _make_config(req.session_id)
    chat_history = _load_chat_history(req.session_id)

    initial_state = {
        "query": req.query,
        "session_id": req.session_id,
        "chat_history": chat_history,
    }
    # 前端已分类+重写，传入后跳过 LLM 调用
    if req.intent:
        initial_state["intent"] = req.intent
    if req.rewritten_query:
        initial_state["rewritten_query"] = req.rewritten_query

    try:
        result = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        logger.error("query_invoke failed: %s", e, exc_info=True)
        return QueryResponse(
            query=req.query,
            answer=f"系统错误: {e}",
            status="error",
            session_id=req.session_id,
        )

    # 检查是否被 interrupt（等待审批）
    interrupt_val = _extract_interrupt(result)
    if interrupt_val:
        # 暂存原始 query，供 approve 后恢复
        _save_pending_query(req.session_id, req.query)
        return QueryResponse(
            query=req.query,
            answer=interrupt_val.get("message", "请确认是否执行该 SQL"),
            status="pending_approval",
            session_id=req.session_id,
            pending_approval=True,
            sql=interrupt_val.get("sql", ""),
        )

    answer = result.get("answer", "")
    # 保存本轮 Q&A 到 session
    if answer:
        _save_qa_to_session(req.session_id, req.query, answer)

    return QueryResponse(
        query=req.query,
        answer=answer,
        status=result.get("status", "completed"),
        session_id=req.session_id,
    )


@router.post("/approve")
async def approve_sql(req: ApproveRequest):
    """审批 SQL：继续执行被中断的图。"""
    graph = build_final_graph()
    config = _make_config(req.session_id)

    try:
        result = await graph.ainvoke(
            Command(resume={
                "approved": req.approved,
                "feedback": req.feedback,
            }),
            config=config,
        )
    except Exception as e:
        logger.error("approve_sql failed: %s", e, exc_info=True)
        return QueryResponse(
            query="",
            answer=f"系统错误: {e}",
            status="error",
            session_id=req.session_id,
        )

    # 审批后可能再次 interrupt（理论上不会，但防御性处理）
    interrupt_val = _extract_interrupt(result)
    if interrupt_val:
        return QueryResponse(
            query="",
            answer=interrupt_val.get("message", "请确认是否执行该 SQL"),
            status="pending_approval",
            session_id=req.session_id,
            pending_approval=True,
            sql=interrupt_val.get("sql", ""),
        )

    answer = result.get("answer", "")
    # 恢复中断时暂存的原始 query
    original_query = _pop_pending_query(req.session_id) or result.get("query", "")
    # 保存本轮 Q&A 到 session
    if answer and original_query:
        _save_qa_to_session(req.session_id, original_query, answer)

    return QueryResponse(
        query=original_query,
        answer=answer,
        status=result.get("status", "completed"),
        session_id=req.session_id,
    )
