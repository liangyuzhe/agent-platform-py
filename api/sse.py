"""SSE 流式响应工具。"""

from typing import AsyncGenerator
from sse_starlette.sse import EventSourceResponse
from fastapi import Request


async def sse_response(generator: AsyncGenerator[dict, None], request: Request):
    """SSE 流式响应。"""

    async def event_generator():
        async for chunk in generator:
            if await request.is_disconnected():
                break
            yield chunk

    return EventSourceResponse(event_generator())
