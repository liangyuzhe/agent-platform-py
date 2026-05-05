"""Final Graph 端点：支持中断/恢复的主调度。"""

import logging
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import AsyncGenerator

from langgraph.types import Command

from agents.flow.final_graph import build_final_graph
from agents.api.sse import sse_response
from agents.tool.trace.tracing import get_trace_callbacks

logger = logging.getLogger(__name__)

router = APIRouter()


class FinalRequest(BaseModel):
    query: str
    session_id: str = "default_user"


class FinalResponse(BaseModel):
    query: str
    answer: str
    status: str
    session_id: str
    pending_approval: bool = False
    sql: str = ""


class ClassifyResponse(BaseModel):
    intent: str
    session_id: str


class ApproveRequest(BaseModel):
    session_id: str = "default_user"
    approved: bool = True
    feedback: str = ""


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
async def classify_intent_endpoint(req: FinalRequest):
    """意图分类（非流式），前端据此选择流式端点。"""
    from agents.flow.final_graph import classify_intent
    result = await classify_intent({"query": req.query})
    return ClassifyResponse(
        intent=result.get("intent", "chat"),
        session_id=req.session_id,
    )


@router.post("/invoke", response_model=FinalResponse)
async def final_invoke(req: FinalRequest):
    """非流式 Final Graph 调用。"""
    graph = build_final_graph()
    config = _make_config(req.session_id)

    result = await graph.ainvoke({
        "query": req.query,
        "session_id": req.session_id,
    }, config=config)

    # 检查是否被 interrupt（等待审批）
    interrupt_val = _extract_interrupt(result)
    if interrupt_val:
        return FinalResponse(
            query=req.query,
            answer=interrupt_val.get("message", "请确认是否执行该 SQL"),
            status="pending_approval",
            session_id=req.session_id,
            pending_approval=True,
            sql=interrupt_val.get("sql", ""),
        )

    return FinalResponse(
        query=req.query,
        answer=result.get("answer", ""),
        status=result.get("status", "completed"),
        session_id=req.session_id,
    )


@router.post("/approve")
async def approve_sql(req: ApproveRequest):
    """审批 SQL：继续执行被中断的图。"""
    graph = build_final_graph()
    config = _make_config(req.session_id)

    result = await graph.ainvoke(
        Command(resume={
            "approved": req.approved,
            "feedback": req.feedback,
        }),
        config=config,
    )

    # 审批后可能再次 interrupt（理论上不会，但防御性处理）
    interrupt_val = _extract_interrupt(result)
    if interrupt_val:
        return FinalResponse(
            query="",
            answer=interrupt_val.get("message", "请确认是否执行该 SQL"),
            status="pending_approval",
            session_id=req.session_id,
            pending_approval=True,
            sql=interrupt_val.get("sql", ""),
        )

    return FinalResponse(
        query="",
        answer=result.get("answer", ""),
        status=result.get("status", "completed"),
        session_id=req.session_id,
    )


@router.post("/invoke/stream")
async def final_invoke_stream(req: FinalRequest, request: Request):
    """SSE 流式 Final Graph 调用。"""
    graph = build_final_graph()
    config = _make_config(req.session_id)

    async def generate() -> AsyncGenerator[dict, None]:
        yield {"event": "start", "data": ""}
        final_result = None
        in_intermediate_node = False

        try:
            async for event in graph.astream_events(
                {"query": req.query, "session_id": req.session_id},
                version="v2",
                config=config,
            ):
                evt_type = event["event"]
                evt_name = event.get("name", "")

                # Track intermediate graph nodes to filter their LLM output
                if evt_type == "on_chain_start" and evt_name == "classify_intent":
                    in_intermediate_node = True
                elif evt_type == "on_chain_end" and evt_name == "classify_intent":
                    in_intermediate_node = False

                # Only emit LLM chunks from final nodes (sql_react / chat_direct)
                if evt_type == "on_chat_model_stream" and not in_intermediate_node:
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        yield {"event": "data", "data": chunk.content}
                elif evt_type == "on_chain_end" and evt_name == "LangGraph":
                    final_result = event["data"].get("output")
        except Exception as e:
            logger.exception("Final graph stream error")
            yield {"event": "error", "data": str(e)}

        # Check if graph paused for SQL approval (interrupt)
        if final_result:
            interrupt_val = _extract_interrupt(final_result)
            if interrupt_val:
                import json
                yield {"event": "approval", "data": json.dumps({
                    "sql": interrupt_val.get("sql", ""),
                    "message": interrupt_val.get("message", "请确认是否执行该 SQL"),
                })}

        yield {"event": "end", "data": ""}

    return await sse_response(generate(), request)
