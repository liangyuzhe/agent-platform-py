"""RAG Chat 端点。"""

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import AsyncGenerator, Literal

from agents.flow.rag_chat import build_rag_chat_graph
from agents.api.sse import sse_response
from agents.tool.trace.tracing import get_trace_callbacks

logger = logging.getLogger(__name__)
router = APIRouter()


class RAGChatRequest(BaseModel):
    query: str
    session_id: str = "default_user"
    rag_mode: Literal["traditional", "parent"] | None = None


class RAGAskResponse(BaseModel):
    answer: str
    session_id: str


def _build_input(req: RAGChatRequest) -> dict:
    """Build graph input dict, including optional rag_mode override."""
    inp = {"session_id": req.session_id, "query": req.query}
    if req.rag_mode is not None:
        inp["rag_mode"] = req.rag_mode
    return inp


@router.post("/ask", response_model=RAGAskResponse)
async def rag_ask(req: RAGChatRequest):
    """非流式 RAG 问答。"""
    graph = build_rag_chat_graph()
    callbacks = get_trace_callbacks()
    config = {"callbacks": callbacks} if callbacks else {}
    result = await graph.ainvoke({"input": _build_input(req)}, config=config)
    return RAGAskResponse(answer=result.get("answer", ""), session_id=req.session_id)


@router.post("/chat/stream")
async def rag_chat_stream(req: RAGChatRequest, request: Request):
    """SSE 流式 RAG 对话 (POST)。"""
    return await _stream_rag_chat(_build_input(req), request)


@router.get("/chat/stream")
async def rag_chat_stream_get(
    query: str,
    session_id: str = "default_user",
    rag_mode: Literal["traditional", "parent"] | None = None,
    request: Request = None,
):
    """SSE 流式 RAG 对话 (GET, for EventSource)."""
    inp = {"session_id": session_id, "query": query}
    if rag_mode is not None:
        inp["rag_mode"] = rag_mode
    return await _stream_rag_chat(inp, request)


@router.get("/test/stream")
async def rag_test_stream(request: Request):
    """简单 SSE 测试端点。"""
    import asyncio

    async def generate():
        for i in range(5):
            yield {"event": "message", "data": f"chunk-{i}"}
            await asyncio.sleep(0.3)
        yield {"event": "done", "data": "[DONE]"}

    return await sse_response(generate(), request)


async def _stream_rag_chat(inp: dict, request: Request):
    """Shared streaming logic."""
    graph = build_rag_chat_graph()
    callbacks = get_trace_callbacks()
    config = {"callbacks": callbacks} if callbacks else {}

    async def generate() -> AsyncGenerator[dict, None]:
        try:
            async for event in graph.astream_events(
                {"input": inp},
                version="v2",
                config=config,
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        yield {"event": "message", "data": chunk.content}
        except Exception as e:
            logger.exception("RAG chat stream error")
            yield {"event": "error", "data": str(e)}
        finally:
            yield {"event": "done", "data": "[DONE]"}

    return await sse_response(generate(), request)
