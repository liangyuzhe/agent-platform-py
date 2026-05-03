"""Chat 测试端点。"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import AsyncGenerator

from agents.model.chat_model import get_chat_model
from agents.api.sse import sse_response
from agents.config.settings import settings

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

router = APIRouter()


class ChatTestRequest(BaseModel):
    question: str
    history: list[dict] = []  # [{"role": "user/assistant", "content": "..."}]


class ChatTestResponse(BaseModel):
    question: str
    answer: str


@router.post("/test", response_model=ChatTestResponse)
async def chat_generate(req: ChatTestRequest):
    """非流式 Chat 测试。"""
    model = get_chat_model(settings.chat_model_type)

    messages = [SystemMessage(content="你是一个有帮助的AI助手。")]
    for msg in req.history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=req.question))

    response = await model.ainvoke(messages)
    return ChatTestResponse(question=req.question, answer=response.content)


@router.post("/test/stream")
async def chat_stream(req: ChatTestRequest, request: Request):
    """SSE 流式 Chat 测试。"""
    model = get_chat_model(settings.chat_model_type)

    messages = [SystemMessage(content="你是一个有帮助的AI助手。")]
    for msg in req.history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=req.question))

    async def generate() -> AsyncGenerator[dict, None]:
        yield {"event": "start", "data": ""}
        async for chunk in model.astream(messages):
            if chunk.content:
                yield {"event": "data", "data": chunk.content}
        yield {"event": "end", "data": ""}

    return await sse_response(generate(), request)
