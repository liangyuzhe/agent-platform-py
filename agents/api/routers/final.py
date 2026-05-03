"""Final Graph 端点：支持中断/恢复的主调度。"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import AsyncGenerator

from agents.flow.final_graph import build_final_graph
from agents.api.sse import sse_response
from agents.tool.trace.tracing import get_trace_callbacks

router = APIRouter()

# 审批会话状态（内存存储，生产环境应存 Redis）
_sessions: dict[str, dict] = {}


class FinalRequest(BaseModel):
    query: str
    session_id: str = "default_user"


class FinalResponse(BaseModel):
    query: str
    answer: str
    status: str
    session_id: str


@router.post("/invoke", response_model=FinalResponse)
async def final_invoke(req: FinalRequest):
    """非流式 Final Graph 调用。"""
    graph = build_final_graph()
    callbacks = get_trace_callbacks()
    config = {"callbacks": callbacks} if callbacks else {}

    result = await graph.ainvoke({
        "query": req.query,
        "session_id": req.session_id,
    }, config=config)

    return FinalResponse(
        query=req.query,
        answer=result.get("answer", ""),
        status=result.get("status", "completed"),
        session_id=req.session_id,
    )


@router.post("/invoke/stream")
async def final_invoke_stream(req: FinalRequest, request: Request):
    """SSE 流式 Final Graph 调用。"""
    graph = build_final_graph()
    callbacks = get_trace_callbacks()
    config = {"callbacks": callbacks} if callbacks else {}

    async def generate() -> AsyncGenerator[dict, None]:
        yield {"event": "start", "data": ""}
        async for event in graph.astream_events(
            {"query": req.query, "session_id": req.session_id},
            version="v2",
            **config,
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield {"event": "data", "data": chunk.content}
        yield {"event": "end", "data": ""}

    return await sse_response(generate(), request)
