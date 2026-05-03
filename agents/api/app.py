"""FastAPI 应用入口。"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agents.api.routers import chat, rag, final, document
from agents.config.settings import settings

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化所有组件，关闭时清理。"""
    from agents.tool.storage.redis_client import init_redis, close_redis
    from agents.model.chat_model import init_chat_models
    from agents.model.embedding_model import init_embedding_models
    from agents.tool.trace.tracing import init_tracing, close_cozeloop

    # 初始化基础设施
    await init_redis()

    # 初始化模型
    init_chat_models()
    init_embedding_models()

    # 初始化链路追踪
    init_tracing()

    yield

    # 清理
    close_cozeloop()
    await close_redis()


app = FastAPI(
    title="Agents-Py",
    description="AI Agent Platform built with LangChain and LangGraph",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由注册
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(rag.router, prefix="/api/rag", tags=["rag"])
app.include_router(final.router, prefix="/api/final", tags=["final"])
app.include_router(document.router, prefix="/api/document", tags=["document"])


@app.get("/health")
async def health():
    return {"status": "ok"}


# 静态文件
try:
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
except Exception:
    pass  # static 目录不存在时忽略
