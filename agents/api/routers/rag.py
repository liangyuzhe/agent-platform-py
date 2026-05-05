"""RAG Chat 端点。"""

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Literal

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
    graph = build_rag_chat_graph()
    callbacks = get_trace_callbacks()
    config = {"callbacks": callbacks} if callbacks else {}

    async def generate():
        try:
            async for event in graph.astream_events(
                {"input": _build_input(req)},
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
