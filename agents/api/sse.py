"""SSE 流式响应工具。"""

from typing import AsyncGenerator
from sse_starlette.sse import EventSourceResponse
from fastapi import Request


async def sse_response(generator: AsyncGenerator[dict, None], request: Request):
    """SSE 流式响应。"""
    return EventSourceResponse(generator)
