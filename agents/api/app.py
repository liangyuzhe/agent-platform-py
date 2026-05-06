"""FastAPI 应用入口。"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agents.api.routers import chat, rag, final, document, admin
from agents.config.settings import settings

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化所有组件，关闭时清理。"""
    import asyncio
    import logging

    from agents.tool.storage.redis_client import init_redis, close_redis
    from agents.model.chat_model import init_chat_models
    from agents.model.embedding_model import init_embedding_models
    from agents.tool.trace.tracing import init_tracing, close_cozeloop

    logger = logging.getLogger(__name__)

    # 初始化基础设施
    await init_redis()

    # 初始化模型
    init_chat_models()
    init_embedding_models()

    # 注册工具（import 触发 @register）
    import agents.tool.sql_tools  # noqa: F401

    # 初始化链路追踪
    init_tracing()

    # 后台异步：检查领域摘要，按需索引 MySQL 表结构（不阻塞服务启动）
    asyncio.create_task(_index_schemas_background(logger))

    yield

    # 清理
    close_cozeloop()
    await close_redis()


async def _index_schemas_background(logger):
    """后台任务：连接 MySQL 并将表结构向量化到 Milvus + ES。

    如果 domain_summary 表中已有摘要，跳过全量索引，直接加载摘要。
    """
    try:
        from agents.tool.storage.domain_summary import (
            ensure_domain_summary_table,
            get_domain_summary,
        )
        from agents.rag.schema_indexer import index_mysql_schemas

        await ensure_domain_summary_table()

        existing = await get_domain_summary()
        if existing:
            logger.info("Domain summary found in cache/DB (%d chars), skipping schema re-index", len(existing))
            return

        result = await index_mysql_schemas()
        logger.info("Schema auto-indexing done: %s", result)
    except Exception as e:
        logger.warning("Schema auto-indexing failed: %s", e)


app = FastAPI(
    title="Financial Copilot",
    description="Financial Copilot Platform built with LangChain and LangGraph",
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
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])


@app.get("/health")
async def health():
    return {"status": "ok"}


# 静态文件
try:
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
except Exception:
    pass  # static 目录不存在时忽略
