"""RAG Chat 端点。"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import AsyncGenerator, Literal

from agents.flow.rag_chat import build_rag_chat_graph
from agents.api.sse import sse_response

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
    result = await graph.ainvoke({"input": _build_input(req)})
    return RAGAskResponse(answer=result.get("answer", ""), session_id=req.session_id)


@router.post("/chat/stream")
async def rag_chat_stream(req: RAGChatRequest, request: Request):
    """SSE 流式 RAG 对话。"""
    graph = build_rag_chat_graph()

    async def generate() -> AsyncGenerator[dict, None]:
        yield {"event": "start", "data": ""}
        async for event in graph.astream_events(
            {"input": _build_input(req)},
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield {"event": "data", "data": chunk.content}
        yield {"event": "end", "data": ""}

    return await sse_response(generate(), request)
