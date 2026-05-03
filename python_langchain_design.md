# Python 版本技术设计文档：LangChain + LangGraph 实现

## 目录

1. [项目概述](#1-项目概述)
2. [技术栈映射](#2-技术栈映射)
3. [项目结构设计](#3-项目结构设计)
4. [核心模块设计](#4-核心模块设计)
5. [Flow 编排层：LangGraph 实现](#5-flow-编排层langgraph-实现)
6. [RAG 管线设计](#6-rag-管线设计)
7. [记忆系统设计](#7-记忆系统设计)
8. [模型抽象层设计](#8-模型抽象层设计)
9. [工具层设计](#9-工具层设计)
10. [SFT 数据管线设计](#10-sft-数据管线设计)
11. [存储层设计](#11-存储层设计)
12. [API 层设计](#12-api-层设计)
13. [配置与部署](#13-配置与部署)
14. [Go -> Python 映射速查表](#14-go---python-映射速查表)

---

## 1. 项目概述

### 1.1 目标

将 Go-Agent 项目完整移植到 Python 生态，使用 **LangChain** 作为 LLM 抽象层，**LangGraph** 作为图编排引擎。在保持原有功能的基础上，利用 Python 生态的优势（更丰富的 AI 库、更快的原型开发）并引入 Go 版本中已识别的优化方案。

### 1.2 核心能力（与 Go 版本一致）

| 能力 | 说明 |
|------|------|
| RAG 对话 | 文档索引 + 混合检索（向量 + BM25） + RRF 融合 + LLM 生成 |
| SQL 生成与执行 | 自然语言 -> SQL -> 人工审批 -> MCP 执行 |
| 数据分析 | SQL 结果 -> 统计分析 -> 图表生成 -> 文字报告 |
| 对话记忆 | 三级记忆（工作记忆 + 摘要记忆 + 知识记忆） |
| SFT 数据采集 | 自动收集 + 教师标注 + JSONL 导出 |

### 1.3 相比 Go 版本的改进

| 改进项 | Go 版本现状 | Python 版本方案 |
|--------|-----------|----------------|
| 混合检索 | ES 用向量模式，BM25 未使用 | ES 用 BM25 + Milvus 向量，真正混合 |
| 记忆系统 | 2 级（History + Summary） | 3 级（+ Knowledge Memory） |
| Token 管理 | 无 | 显式 Token 计数 + 上下文窗口管理 |
| 模型切换 | 改代码 | 配置驱动 + 降级链 |
| SQL 安全 | 无 | 静态分析 + 风险评估 |
| 检索质量 | 无重排序 | Cross-Encoder Reranker |

---

## 2. 技术栈映射

### 2.1 Go -> Python 组件对照

| Go 组件 | Python 替代 | 选择理由 |
|---------|------------|---------|
| CloudWeGo Eino | **LangGraph** | 同为图编排引擎，LangGraph 是 Python 生态标准，支持中断/恢复 |
| Eino compose.Graph | **langgraph.graph.StateGraph** | 状态图模型，节点间通过 TypedDict 共享状态 |
| Eino ChatModel | **langchain_core.language_models.BaseChatModel** | LangChain 统一模型接口 |
| Eino Embedder | **langchain_core.embeddings.Embeddings** | LangChain 统一 Embedding 接口 |
| Eino Retriever | **langchain_core.retrievers.BaseRetriever** | LangChain 统一检索器接口 |
| Eino Indexer | **langchain.indexes.RecordManager** + VectorStore | LangChain 索引管理 |
| Gin | **FastAPI** | 异步原生，自动 OpenAPI 文档，SSE 原生支持 |
| Milvus SDK | **pymilvus** + **langchain_milvus** | 官方 SDK + LangChain 集成 |
| ES8 Client | **elasticsearch** + **langchain_elasticsearch** | 官方 SDK + LangChain 集成 |
| Redis | **redis-py** + **langchain_redis** | 官方 SDK |
| MCP (Go SDK) | **mcp** Python SDK | Anthropic 官方 Python MCP SDK |
| godotenv | **python-dotenv** | 等价替代 |
| sync.Once | **functools.lru_cache** / 单例模式 | Python 惯用方式 |

### 2.2 新增依赖

| 库 | 用途 | Go 版本对应 |
|----|------|------------|
| `langchain` | LLM 抽象层核心 | 无（Go 版自建抽象） |
| `langgraph` | 图编排引擎 | CloudWeGo Eino |
| `langchain-openai` | OpenAI/兼容模型接入 | eino-ext openai |
| `langchain-anthropic` | Claude 模型接入 | 无 |
| `langchain-milvus` | Milvus 向量存储 | eino-ext milvus |
| `langchain-elasticsearch` | ES 检索器 | eino-ext es8 |
| `tiktoken` | Token 计数 | 无（Go 版缺失） |
| `sentence-transformers` | Cross-Encoder 重排序 | 无（Go 版缺失） |
| `pandas` + `numpy` | 数据分析 | 手动实现 |
| `pydantic` | 数据校验 + 配置 | 手动 struct |
| `uvicorn` + `fastapi` | HTTP 服务 | Gin |
| `sse-starlette` | SSE 流式 | 手动 SSE |

---

## 3. 项目结构设计

```
agents-py/
├── pyproject.toml                  # 项目配置 + 依赖
├── .env                            # 环境变量
├── docker-compose.yaml             # 基础设施（Milvus、ES、Redis）
│
├── agents/                         # 主包
│   ├── __init__.py
│   ├── main.py                     # 入口，初始化编排
│   │
│   ├── config/                     # 配置层
│   │   ├── __init__.py
│   │   └── settings.py             # Pydantic Settings，从 .env 加载
│   │
│   ├── api/                        # API 层（FastAPI）
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI 应用 + 路由注册
│   │   ├── routers/
│   │   │   ├── chat.py             # /api/chat/* 路由
│   │   │   ├── rag.py              # /api/rag/* 路由
│   │   │   ├── final.py            # /api/final/* 路由
│   │   │   └── document.py         # /api/document/* 路由
│   │   └── sse.py                  # SSE 流式响应工具
│   │
│   ├── flow/                       # Flow 编排层（LangGraph）
│   │   ├── __init__.py
│   │   ├── rag_chat.py             # RAG Chat 图
│   │   ├── sql_react.py            # SQL React 图
│   │   ├── analyst.py              # Analyst 图
│   │   ├── final_graph.py          # 主调度图
│   │   └── state.py                # 共享状态定义（TypedDict）
│   │
│   ├── model/                      # 模型抽象层
│   │   ├── __init__.py
│   │   ├── chat_model.py           # Chat Model 工厂 + 路由
│   │   ├── embedding_model.py      # Embedding Model 工厂
│   │   ├── providers/              # 各提供商实现
│   │   │   ├── ark.py              # 火山引擎 Ark (豆包)
│   │   │   ├── openai.py           # OpenAI
│   │   │   ├── deepseek.py         # DeepSeek
│   │   │   ├── qwen.py             # 通义千问
│   │   │   └── gemini.py           # Google Gemini
│   │   └── format_tool.py          # 结构化输出工具
│   │
│   ├── rag/                        # RAG 管线
│   │   ├── __init__.py
│   │   ├── indexing.py             # 索引图（Loader -> Parser -> Splitter -> Store）
│   │   ├── retriever.py            # 检索图（双路检索 + RRF + Reranker）
│   │   ├── query_rewrite.py        # 查询重写
│   │   └── reranker.py             # Cross-Encoder 重排序
│   │
│   ├── tool/                       # 工具层
│   │   ├── __init__.py
│   │   ├── memory/                 # 记忆系统
│   │   │   ├── __init__.py
│   │   │   ├── session.py          # Session 数据模型
│   │   │   ├── store.py            # Session 存储（内存 / Redis）
│   │   │   ├── compressor.py       # LLM 摘要压缩
│   │   │   └── knowledge.py        # 知识记忆（实体/事实/偏好）
│   │   │
│   │   ├── storage/                # 存储层
│   │   │   ├── __init__.py
│   │   │   ├── redis_client.py     # Redis 连接管理
│   │   │   ├── checkpoint.py       # LangGraph Checkpointer (Redis)
│   │   │   ├── retrieval_cache.py  # 检索结果缓存
│   │   │   └── session_store.py    # 审批会话存储
│   │   │
│   │   ├── document/               # 文档处理
│   │   │   ├── __init__.py
│   │   │   ├── loader.py           # 文件加载器
│   │   │   ├── parser.py           # 按扩展名分发的解析器
│   │   │   ├── splitter.py         # 文本分块器
│   │   │   └── docx_parser.py      # DOCX 解析
│   │   │
│   │   ├── sql_tools/              # SQL 工具
│   │   │   ├── __init__.py
│   │   │   ├── mcp_client.py       # MCP 连接管理
│   │   │   ├── executor.py         # SQL 执行
│   │   │   └── safety.py           # SQL 安全分析（新增）
│   │   │
│   │   ├── analyst_tools/          # 数据分析工具
│   │   │   ├── __init__.py
│   │   │   ├── parser.py           # SQL 结果解析
│   │   │   ├── statistics.py       # 统计计算
│   │   │   └── chart.py            # 图表生成
│   │   │
│   │   ├── sft/                    # SFT 数据管线
│   │   │   ├── __init__.py
│   │   │   ├── callback.py         # LangChain CallbackHandler
│   │   │   ├── annotator.py        # 教师模型标注
│   │   │   ├── speculative.py      # 推测解码
│   │   │   └── storage.py          # Sample 存储 + JSONL 导出
│   │   │
│   │   └── token_counter.py        # Token 计数器（新增）
│   │
│   ├── algorithm/                  # 算法
│   │   ├── __init__.py
│   │   ├── bm25.py                 # BM25 实现
│   │   └── rrf.py                  # RRF 融合
│   │
│   └── trace/                      # 可观测性
│       ├── __init__.py
│       ├── langsmith.py            # LangSmith 集成
│       └── callback.py             # 自定义 CallbackHandler
│
├── static/                         # 前端静态文件
│   └── final_graph.html
│
├── tests/                          # 测试
│   ├── test_rag_chat.py
│   ├── test_sql_react.py
│   ├── test_memory.py
│   └── test_retriever.py
│
└── data/                           # 数据目录
    └── sft/                        # SFT 训练数据
```

---

## 4. 核心模块设计

### 4.1 配置管理（`config/settings.py`）

使用 Pydantic Settings 替代 Go 版的 godotenv + 手动结构体。

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class ArkConfig(BaseSettings):
    key: str = Field(default="", alias="ARK_KEY")
    chat_model: str = Field(default="doubao-seed-2-0", alias="ARK_CHAT_MODEL")
    embedding_model: str = Field(default="", alias="ARK_EMBEDDING_MODEL")

class OpenAIConfig(BaseSettings):
    key: str = Field(default="", alias="OPENAI_KEY")
    chat_model: str = Field(default="gpt-4o", alias="OPENAI_CHAT_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")

class QwenConfig(BaseSettings):
    base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1", alias="QWEN_BASE_URL")
    key: str = Field(default="", alias="QWEN_KEY")
    chat_model: str = Field(default="", alias="QWEN_CHAT_MODEL")
    embedding_model: str = Field(default="text-embedding-v3", alias="QWEN_EMBEDDING_MODEL")

class MilvusConfig(BaseSettings):
    addr: str = Field(default="localhost:19530", alias="MILVUS_ADDR")
    username: str = Field(default="root", alias="MILVUS_USERNAME")
    password: str = Field(default="milvus", alias="MILVUS_PASSWORD")
    collection_name: str = Field(default="goagent2", alias="MILVUS_COLLECTION_NAME")
    similarity_threshold: float = Field(default=0.0, alias="MILVUS_SIMILARITY_THRESHOLD")
    top_k: int = Field(default=5, alias="TOPK")

class ESConfig(BaseSettings):
    address: str = Field(default="http://localhost:9200", alias="ES_ADDRESS")
    username: str = Field(default="", alias="ES_USERNAME")
    password: str = Field(default="", alias="ES_PASSWORD")
    index: str = Field(default="go_agent_docs", alias="ES_INDEX")

class RedisConfig(BaseSettings):
    addr: str = Field(default="localhost:6379", alias="REDIS_ADDR")
    password: str = Field(default="", alias="REDIS_PASSWORD")
    db: int = Field(default=0, alias="REDIS_DB")

class MySQLConfig(BaseSettings):
    host: str = Field(default="localhost", alias="MYSQL_HOST")
    port: int = Field(default=3306, alias="MYSQL_PORT")
    username: str = Field(default="root", alias="MYSQL_USERNAME")
    password: str = Field(default="", alias="MYSQL_PASSWORD")
    database: str = Field(default="", alias="MYSQL_DATABASE")


class Settings(BaseSettings):
    # 模型选择
    chat_model_type: str = Field(default="ark", alias="CHAT_MODEL_TYPE")
    embedding_model_type: str = Field(default="qwen", alias="EMBEDDING_MODEL_TYPE")
    vector_db_type: str = Field(default="MILVUS", alias="VECTOR_DB_TYPE")

    # 各提供商配置
    ark: ArkConfig = ArkConfig()
    openai: OpenAIConfig = OpenAIConfig()
    qwen: QwenConfig = QwenConfig()
    milvus: MilvusConfig = MilvusConfig()
    es: ESConfig = ESConfig()
    redis: RedisConfig = RedisConfig()
    mysql: MySQLConfig = MySQLConfig()

    # RAG 参数
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # 记忆参数
    max_history_len: int = 3
    summary_model_type: str = "ark"  # 摘要用的模型

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
```

**对比 Go 版改进：**
- Pydantic 自动校验类型和必填字段
- 嵌套配置结构更清晰
- 模型切换只需改环境变量，不改代码

### 4.2 入口（`main.py`）

```python
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

from agents.config.settings import settings
from agents.tool.storage.redis_client import init_redis
from agents.model.chat_model import init_chat_models
from agents.model.embedding_model import init_embedding_models
from agents.rag.indexing import build_indexing_graph
from agents.flow.rag_chat import build_rag_chat_graph
from agents.flow.final_graph import build_final_graph
from agents.tool.sql_tools.mcp_client import init_mcp_tools


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化所有组件，关闭时清理。"""
    # 初始化基础设施
    await init_redis()

    # 初始化模型
    init_chat_models()
    init_embedding_models()

    # 初始化 MCP 工具
    await init_mcp_tools()

    # 预编译图（LangGraph 会在首次调用时编译，这里做预热）
    app.state.rag_chat_graph = build_rag_chat_graph()
    app.state.final_graph = build_final_graph()
    app.state.indexing_graph = build_indexing_graph()

    yield

    # 清理
    from agents.tool.storage.redis_client import close_redis
    await close_redis()


app = FastAPI(lifespan=lifespan)
```

**对比 Go 版改进：**
- 异步初始化，启动更快
- `lifespan` 管理生命周期，优雅关闭
- 组件挂载在 `app.state` 上，依赖注入更自然

---

## 5. Flow 编排层：LangGraph 实现

### 5.1 状态定义（`flow/state.py`）

LangGraph 的核心是 **StateGraph**——节点间通过共享的 TypedDict 状态通信。

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from langchain_core.documents import Document


class RAGChatState(TypedDict):
    """RAG Chat 图的状态。"""
    input: dict                          # {"session_id": str, "query": str}
    session_id: str
    query: str
    session: dict                        # Session 数据
    rewritten_query: str                 # 重写后的查询
    docs: list[Document]                 # 检索到的文档
    messages: Annotated[list[BaseMessage], add_messages]  # 对话消息（自动追加）
    answer: str                          # 最终回答


class SQLReactState(TypedDict):
    """SQL React 图的状态。"""
    query: str
    docs: list[Document]                 # 检索到的表结构
    sql: str                             # 生成的 SQL
    is_sql: bool                         # 是否为 SQL 输出
    answer: str                          # 非 SQL 回答
    approved: bool                       # 是否已审批
    refine_feedback: str                 # 修改意见
    result: str                          # SQL 执行结果


class AnalystState(TypedDict):
    """数据分析图的状态。"""
    sql_result: str
    parsed_data: dict                    # ParsedData
    statistics: dict                     # Statistics
    text_analysis: str
    chart_config: dict
    analysis_result: dict                # AnalysisResult


class FinalGraphState(TypedDict):
    """主调度图的状态。"""
    query: str
    session_id: str
    intent: str                          # "SQL" or "Chat"
    sql: str
    result: str
    answer: str
    status: str                          # "pending" | "approved" | "rejected" | "completed"
```

**对比 Go 版改进：**
- `Annotated[list[BaseMessage], add_messages]` 是 LangGraph 的内置消息追加语义，比 Go 版手动 append 更安全
- 状态定义集中管理，一目了然

### 5.2 RAG Chat 图（`flow/rag_chat.py`）

```python
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from agents.flow.state import RAGChatState
from agents.tool.memory.store import get_session, save_session
from agents.tool.memory.compressor import compress_session
from agents.rag.query_rewrite import rewrite_query
from agents.rag.retriever import build_retriever_graph
from agents.model.chat_model import get_chat_model
from agents.tool.token_counter import TokenCounter


async def preprocess(state: RAGChatState) -> dict:
    """加载 Session。"""
    session = await get_session(state["session_id"])
    return {"session": session, "query": state["input"]["query"]}


async def rewrite(state: RAGChatState) -> dict:
    """查询重写：利用记忆上下文化查询。"""
    session = state["session"]
    if not session.get("history") and not session.get("summary"):
        return {"rewritten_query": state["query"]}

    rewritten = await rewrite_query(
        summary=session.get("summary", ""),
        history=session.get("history", []),
        query=state["query"],
    )
    return {"rewritten_query": rewritten}


async def retrieve(state: RAGChatState) -> dict:
    """双路检索 + RRF 融合 + Cross-Encoder 重排序。"""
    retriever_graph = build_retriever_graph()
    docs = await retriever_graph.ainvoke({"query": state["rewritten_query"]})
    return {"docs": docs}


async def construct_messages(state: RAGChatState) -> dict:
    """组装最终 Prompt。"""
    counter = TokenCounter()
    budget = 4096  # 预留给响应

    parts = []
    session = state["session"]

    # 1. 摘要记忆
    if session.get("summary"):
        parts.append(f"背景摘要: {session['summary']}")

    # 2. 工作记忆（历史消息）
    for msg in session.get("history", []):
        parts.append(f"[{msg['role']}]: {msg['content']}")

    # 3. 检索文档（带 Token 预算）
    doc_texts = []
    for doc in state.get("docs", []):
        doc_texts.append(doc.page_content)
    docs_text = "\n---\n".join(doc_texts)
    parts.append(f"参考知识:\n{docs_text}")

    # 4. 当前查询
    parts.append(state["query"])

    # Token 预算裁剪
    fitted = counter.fit_to_budget(parts, budget)
    context = "\n\n".join(fitted)

    messages = [HumanMessage(content=context)]
    return {"messages": messages}


async def chat(state: RAGChatState) -> dict:
    """LLM 生成响应。"""
    model = get_chat_model("ark")
    response = await model.ainvoke(state["messages"])

    # 异步更新记忆
    session_id = state["session_id"]
    session = state["session"]
    session["history"] = session.get("history", []) + [
        {"role": "user", "content": state["query"]},
        {"role": "assistant", "content": response.content},
    ]

    # 异步压缩 + 保存（不阻塞响应）
    import asyncio
    asyncio.create_task(_compress_and_save(session_id, session))

    return {"answer": response.content, "messages": [response]}


async def _compress_and_save(session_id: str, session: dict):
    """后台任务：压缩记忆并保存。"""
    await compress_session(session)
    await save_session(session_id, session)


def build_rag_chat_graph() -> StateGraph:
    """构建 RAG Chat 图。"""
    graph = StateGraph(RAGChatState)

    graph.add_node("preprocess", preprocess)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("construct_messages", construct_messages)
    graph.add_node("chat", chat)

    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "construct_messages")
    graph.add_edge("construct_messages", "chat")
    graph.add_edge("chat", END)

    return graph.compile()
```

**对比 Go 版改进：**
- `TokenCounter` 集成到 `construct_messages`，防止超出上下文窗口
- `asyncio.create_task` 异步压缩，比 Go 版 goroutine 更安全（有异常传播）
- `fit_to_budget` 自动裁剪低优先级内容

### 5.3 SQL React 图（`flow/sql_react.py`）

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agents.flow.state import SQLReactState
from agents.model.chat_model import get_chat_model
from agents.model.format_tool import create_format_tool
from agents.tool.sql_tools.safety import SQLSafetyChecker


async def sql_retrieve(state: SQLReactState) -> dict:
    """检索表结构信息。"""
    retriever_graph = build_retriever_graph()
    docs = await retriever_graph.ainvoke({"query": state["query"]})
    return {"docs": docs}


async def sql_generate(state: SQLReactState) -> dict:
    """LLM 生成 SQL。"""
    model = get_chat_model("ark").bind_tools([create_format_tool()])

    docs_text = "\n".join([d.page_content for d in state["docs"]])
    messages = [
        SystemMessage(content=f"""你是一个 SQL 专家。根据用户的问题和数据库表结构信息，生成正确的 SQL 查询。

表结构信息:
{docs_text}

要求：
1. 使用 MySQL 语法
2. 只生成 SELECT 查询
3. 使用 format_response 工具输出结果"""),
        HumanMessage(content=state["query"]),
    ]

    response = await model.ainvoke(messages)

    # 解析结构化输出
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        args = tool_call["args"]
        return {"sql": args.get("answer", ""), "is_sql": args.get("is_sql", False)}

    return {"answer": response.content, "is_sql": False}


async def safety_check(state: SQLReactState) -> dict:
    """SQL 安全分析。"""
    if not state.get("is_sql"):
        return {}

    checker = SQLSafetyChecker()
    report = checker.check(state["sql"])

    if report.risks:
        # 高风险操作，标记需要额外确认
        return {"safety_report": report.to_dict()}

    return {}


async def approve(state: SQLReactState) -> Command:
    """Human-in-the-Loop 审批。"""
    # LangGraph 的 interrupt 机制：暂停图执行，等待外部输入
    safety_info = ""
    if state.get("safety_report"):
        report = state["safety_report"]
        safety_info = f"\n⚠️ 安全分析：风险={report['risks']}, 预计影响行数={report['estimated_rows']}"

    user_decision = interrupt({
        "type": "approval_request",
        "sql": state["sql"],
        "safety_info": safety_info,
        "message": f"请审批以下 SQL：\n{state['sql']}{safety_info}\n\n回复 'YES' 执行，或提供修改意见。",
    })

    if user_decision.upper() in ("YES", "执行", "批准执行"):
        return Command(update={"approved": True}, goto="execute_sql")
    else:
        return Command(
            update={"approved": False, "refine_feedback": user_decision},
            goto="sql_generate",  # 回到生成节点，带修改意见重新生成
        )


async def execute_sql(state: SQLReactState) -> dict:
    """通过 MCP 执行 SQL。"""
    from agents.tool.sql_tools.executor import execute_sql as mcp_execute
    result = await mcp_execute(state["sql"])
    return {"result": result, "status": "completed"}


def build_sql_react_graph() -> StateGraph:
    """构建 SQL React 图。"""
    graph = StateGraph(SQLReactState)

    graph.add_node("sql_retrieve", sql_retrieve)
    graph.add_node("sql_generate", sql_generate)
    graph.add_node("safety_check", safety_check)
    graph.add_node("approve", approve)
    graph.add_node("execute_sql", execute_sql)

    graph.add_edge(START, "sql_retrieve")
    graph.add_edge("sql_retrieve", "sql_generate")
    graph.add_edge("sql_generate", "safety_check")
    graph.add_conditional_edges(
        "safety_check",
        lambda state: "approve" if state.get("is_sql") else END,
    )
    # approve 节点通过 Command 动态决定 goto

    return graph.compile(checkpointer=RedisCheckpointer())
```

**对比 Go 版改进：**
- `interrupt()` 是 LangGraph 原生的 Human-in-the-Loop 机制，比 Go 版的 `compose.Interrupt()` 更成熟
- `Command(update=..., goto=...)` 实现动态路由，审批不通过直接回到生成节点
- `SQLSafetyChecker` 在审批前做静态安全分析
- `checkpointer=RedisCheckpointer()` 支持跨进程的中断/恢复

### 5.4 Analyst 图（`flow/analyst.py`）

```python
from langgraph.graph import StateGraph, START, END
from agents.flow.state import AnalystState
from agents.tool.analyst_tools.parser import parse_sql_result
from agents.tool.analyst_tools.statistics import compute_statistics
from agents.tool.analyst_tools.chart import generate_chart_config
from agents.model.chat_model import get_chat_model


async def parse_data(state: AnalystState) -> dict:
    """解析 SQL 结果。"""
    parsed = parse_sql_result(state["sql_result"])
    return {"parsed_data": parsed}


async def analyze_data(state: AnalystState) -> dict:
    """计算统计指标。"""
    stats = compute_statistics(state["parsed_data"])
    return {"statistics": stats}


async def generate_report(state: AnalystState) -> dict:
    """LLM 生成文字分析报告。"""
    model = get_chat_model("ark")
    data = state["parsed_data"]
    stats = state["statistics"]

    response = await model.ainvoke([HumanMessage(content=f"""
请基于以下数据生成一段简洁的分析报告（200字以内）：

列: {data['columns']}
样本数据: {data['sample_rows']}
统计指标: {stats}
""")])
    return {"text_analysis": response.content}


async def generate_chart(state: AnalystState) -> dict:
    """生成 ECharts 配置。"""
    chart_config = generate_chart_config(state["parsed_data"])
    return {"chart_config": chart_config}


async def merge_result(state: AnalystState) -> dict:
    """合并分析结果。"""
    return {
        "analysis_result": {
            "text_analysis": state["text_analysis"],
            "chart_config": state["chart_config"],
            "statistics": state["statistics"],
        }
    }


def build_analyst_graph() -> StateGraph:
    """构建 Analyst 图。"""
    graph = StateGraph(AnalystState)

    graph.add_node("parse_data", parse_data)
    graph.add_node("analyze_data", analyze_data)
    graph.add_node("generate_report", generate_report)
    graph.add_node("generate_chart", generate_chart)
    graph.add_node("merge_result", merge_result)

    graph.add_edge(START, "parse_data")
    graph.add_edge("parse_data", "analyze_data")
    graph.add_edge("analyze_data", "generate_report")
    graph.add_edge("analyze_data", "generate_chart")  # 并行分支
    graph.add_edge("generate_report", "merge_result")
    graph.add_edge("generate_chart", "merge_result")
    graph.add_edge("merge_result", END)

    return graph.compile()
```

**LangGraph 并行执行说明：** 当两个节点没有依赖关系时（如 `generate_report` 和 `generate_chart`），LangGraph 会自动并行执行它们。

### 5.5 主调度图（`flow/final_graph.py`）

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage

from agents.flow.state import FinalGraphState
from agents.flow.sql_react import build_sql_react_graph
from agents.model.chat_model import get_chat_model


async def classify_intent(state: FinalGraphState) -> dict:
    """意图分类：SQL 还是 Chat？"""
    model = get_chat_model("qwen")  # 用小模型做分类，省成本

    response = await model.ainvoke([HumanMessage(content=f"""
请判断以下用户问题的意图类型，只回答 "SQL" 或 "Chat"：

- SQL：需要查询数据库、统计数据、生成报表
- Chat：普通对话、知识问答、闲聊

用户问题: {state['query']}
""")])

    intent = response.content.strip().upper()
    return {"intent": "SQL" if "SQL" in intent else "Chat"}


async def sql_react(state: FinalGraphState) -> dict:
    """SQL React 子图。"""
    sql_graph = build_sql_react_graph()
    result = await sql_graph.ainvoke({
        "query": state["query"],
        "session_id": state["session_id"],
    })
    return {"sql": result.get("sql", ""), "result": result.get("result", "")}


async def chat_direct(state: FinalGraphState) -> dict:
    """普通对话（直接调用 RAG Chat）。"""
    from agents.flow.rag_chat import build_rag_chat_graph
    rag_graph = build_rag_chat_graph()
    result = await rag_graph.ainvoke({
        "input": {"session_id": state["session_id"], "query": state["query"]},
    })
    return {"answer": result.get("answer", ""), "status": "completed"}


def route_intent(state: FinalGraphState) -> str:
    """条件路由。"""
    if state.get("intent") == "SQL":
        return "sql_react"
    return "chat_direct"


def build_final_graph() -> StateGraph:
    """构建主调度图。"""
    graph = StateGraph(FinalGraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("sql_react", sql_react)
    graph.add_node("chat_direct", chat_direct)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_intent)
    graph.add_edge("sql_react", END)
    graph.add_edge("chat_direct", END)

    return graph.compile(checkpointer=RedisCheckpointer())
```

**对比 Go 版改进：**
- 修复了 Go 版 `Condition == "SOL"` 的拼写错误
- Chat 路径正确接入 RAG Chat 子图（Go 版 Chat 路径直接结束）
- 意图分类用小模型（qwen），主生成用大模型（ark），成本优化

---

## 6. RAG 管线设计

### 6.1 索引管线（`rag/indexing.py`）

```python
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document

from agents.config.settings import settings
from agents.model.embedding_model import get_embedding_model


# 文件扩展名 -> Loader 映射
LOADER_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".docx": Docx2txtLoader,
}


def load_document(file_path: str) -> list[Document]:
    """根据文件扩展名选择 Loader。"""
    import os
    ext = os.path.splitext(file_path)[1].lower()
    loader_cls = LOADER_MAP.get(ext, TextLoader)
    loader = loader_cls(file_path)
    return loader.load()


def split_documents(docs: list[Document]) -> list[Document]:
    """文本分块。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(docs)


def build_indexing_graph():
    """构建索引管线（同步函数，供 API 调用）。"""
    def index_document(file_path: str) -> dict:
        # 1. 加载
        docs = load_document(file_path)

        # 2. 分块
        chunks = split_documents(docs)

        # 3. 并行写入 Milvus + ES
        embedding = get_embedding_model()

        # Milvus
        milvus_store = Milvus.from_documents(
            documents=chunks,
            embedding=embedding,
            connection_args={
                "host": settings.milvus.addr.split(":")[0],
                "port": settings.milvus.addr.split(":")[1],
            },
            collection_name=settings.milvus.collection_name,
        )

        # Elasticsearch
        es_store = ElasticsearchStore.from_documents(
            documents=chunks,
            embedding=embedding,
            es_url=settings.es.address,
            index_name=settings.es.index,
        )

        return {
            "chunk_count": len(chunks),
            "doc_ids": [c.metadata.get("id", "") for c in chunks],
        }

    return index_document
```

### 6.2 检索管线（`rag/retriever.py`）

```python
from langchain_milvus import Milvus
from langchain_elasticsearch import ElasticsearchStore, BM25Strategy
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from agents.config.settings import settings
from agents.model.embedding_model import get_embedding_model
from agents.algorithm.rrf import reciprocal_rank_fusion
from agents.rag.reranker import CrossEncoderReranker


def build_milvus_retriever() -> BaseRetriever:
    """Milvus 向量检索器。"""
    embedding = get_embedding_model()
    store = Milvus(
        embedding_function=embedding,
        connection_args={
            "host": settings.milvus.addr.split(":")[0],
            "port": settings.milvus.addr.split(":")[1],
        },
        collection_name=settings.milvus.collection_name,
    )
    return store.as_retriever(search_kwargs={"k": settings.milvus.top_k})


def build_es_retriever() -> BaseRetriever:
    """ES BM25 关键词检索器（真正的关键词检索，不是向量检索）。"""
    store = ElasticsearchStore(
        es_url=settings.es.address,
        index_name=settings.es.index,
        strategy=BM25Strategy(),  # 关键：使用 BM25 策略而非向量
    )
    return store.as_retriever(search_kwargs={"k": settings.milvus.top_k})


class HybridRetriever:
    """混合检索器：Milvus(向量) + ES(BM25) + RRF + Cross-Encoder。"""

    def __init__(self):
        self.milvus_retriever = build_milvus_retriever()
        self.es_retriever = build_es_retriever()
        self.reranker = CrossEncoderReranker()

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        # 1. 双路并行检索
        import asyncio
        milvus_task = asyncio.to_thread(self.milvus_retriever.invoke, query)
        es_task = asyncio.to_thread(self.es_retriever.invoke, query)
        milvus_docs, es_docs = await asyncio.gather(milvus_task, es_task)

        # 2. RRF 融合
        fused = reciprocal_rank_fusion([milvus_docs, es_docs], k=60)

        # 3. Cross-Encoder 重排序
        reranked = self.reranker.rerank(query, fused[:top_k * 2])

        return reranked[:top_k]


def build_retriever_graph():
    """返回混合检索器实例。"""
    return HybridRetriever()
```

**对比 Go 版关键改进：**
- ES 使用 `BM25Strategy()`，真正的关键词检索（Go 版 ES 用的是向量检索）
- 添加 `CrossEncoderReranker`，RRF 后再精排
- 双路检索真正并行（`asyncio.gather`）

### 6.3 Cross-Encoder 重排序（`rag/reranker.py`）

```python
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


class CrossEncoderReranker:
    """Cross-Encoder 重排序器。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list[Document], top_k: int = None) -> list[Document]:
        if not docs:
            return []

        # 构造 (query, doc) 对
        pairs = [(query, doc.page_content) for doc in docs]

        # 计算相关性分数
        scores = self.model.predict(pairs)

        # 按分数排序
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        result = [doc for _, doc in scored_docs]
        if top_k:
            result = result[:top_k]
        return result
```

### 6.4 查询重写（`rag/query_rewrite.py`）

```python
from langchain_core.messages import HumanMessage
from agents.model.chat_model import get_chat_model


async def rewrite_query(summary: str, history: list[dict], query: str) -> str:
    """利用记忆上下文重写查询，消除指代消解。"""
    model = get_chat_model("qwen")  # 用小模型做重写

    history_text = "\n".join([f"[{m['role']}]: {m['content']}" for m in history])

    prompt = f"""请将用户的最新问题重写为一个独立的、不依赖上下文的搜索查询。

背景摘要: {summary or '无'}

最近对话:
{history_text or '无'}

用户最新问题: {query}

要求：
1. 保留用户的核心意图
2. 将代词替换为具体实体
3. 只输出重写后的查询，不要解释"""

    response = await model.ainvoke([HumanMessage(content=prompt)])
    return response.content.strip()
```

---

## 7. 记忆系统设计

### 7.1 三级记忆架构

```
┌─────────────────────────────────────────┐
│  L1: 工作记忆 (Working Memory)           │
│  最近 3-5 轮对话原文                      │
│  存储：内存 / Redis                       │
├─────────────────────────────────────────┤
│  L2: 摘要记忆 (Summary Memory)           │
│  LLM 生成的对话摘要                       │
│  存储：Redis                              │
├─────────────────────────────────────────┤
│  L3: 知识记忆 (Knowledge Memory)          │
│  结构化实体/事实/偏好                      │
│  存储：Redis Hash + 向量数据库             │
└─────────────────────────────────────────┘
```

### 7.2 Session 数据模型（`tool/memory/session.py`）

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class Message(BaseModel):
    role: str          # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class Entity(BaseModel):
    name: str
    type: str          # "person" | "place" | "product" | ...
    attributes: dict[str, str] = {}
    last_update: datetime = Field(default_factory=datetime.now)


class Fact(BaseModel):
    content: str
    source: str        # 来自哪轮对话
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence: float = 1.0


class Session(BaseModel):
    id: str
    history: list[Message] = []           # L1: 工作记忆
    summary: str = ""                     # L2: 摘要记忆
    entities: dict[str, Entity] = {}      # L3: 知识记忆 - 实体
    facts: list[Fact] = []               # L3: 知识记忆 - 事实
    preferences: dict[str, str] = {}     # L3: 知识记忆 - 偏好
    updated_at: datetime = Field(default_factory=datetime.now)
```

### 7.3 Session 存储（`tool/memory/store.py`）

```python
import json
from typing import Optional
from agents.tool.storage.redis_client import get_redis
from agents.tool.memory.session import Session


class SessionStore:
    """Session 存储，支持内存和 Redis 双模式。"""

    PREFIX = "session:memory:"

    def __init__(self):
        self._memory: dict[str, Session] = {}

    async def get(self, session_id: str) -> Session:
        # 优先查 Redis
        redis = get_redis()
        if redis:
            data = await redis.get(self.PREFIX + session_id)
            if data:
                return Session.model_validate_json(data)

        # 降级到内存
        if session_id not in self._memory:
            self._memory[session_id] = Session(id=session_id)
        return self._memory[session_id]

    async def save(self, session_id: str, session: Session):
        session.updated_at = datetime.now()
        data = session.model_dump_json()

        redis = get_redis()
        if redis:
            await redis.set(self.PREFIX + session_id, data, ex=86400)  # TTL 24h

        self._memory[session_id] = session


# 单例
_store = SessionStore()

async def get_session(session_id: str) -> Session:
    return await _store.get(session_id)

async def save_session(session_id: str, session: Session):
    await _store.save(session_id, session)
```

### 7.4 记忆压缩（`tool/memory/compressor.py`）

```python
from langchain_core.messages import HumanMessage, SystemMessage
from agents.model.chat_model import get_chat_model
from agents.tool.memory.session import Session, Message
from agents.config.settings import settings


SUMMARY_PROMPT = """你是一个对话摘要助手。请将以下旧对话与已有摘要合并为一段简洁的新摘要。

要求：
1. 保留核心事实、用户偏好、未解决的问题
2. 去掉寒暄和重复的中间步骤
3. 保持连贯性

已有摘要:
{previous_summary}

旧对话:
{older_messages}

请输出合并后的新摘要:"""


async def compress_session(session: Session) -> None:
    """压缩 Session 历史。"""
    max_history = settings.max_history_len

    if len(session.history) <= max_history:
        return  # 不需要压缩

    # 分割：旧消息 vs 最近消息
    to_compress = session.history[:-max_history]
    session.history = session.history[-max_history:]

    # 格式化旧消息
    older_text = "\n".join([f"[{m.role}]: {m.content}" for m in to_compress])

    # LLM 压缩
    model = get_chat_model(settings.summary_model_type)
    response = await model.ainvoke([
        HumanMessage(content=SUMMARY_PROMPT.format(
            previous_summary=session.summary or "（首次对话，无已有摘要）",
            older_messages=older_text,
        ))
    ])

    session.summary = response.content
    session.updated_at = datetime.now()
```

### 7.5 知识记忆提取（`tool/memory/knowledge.py`）

```python
import json
from langchain_core.messages import HumanMessage
from agents.model.chat_model import get_chat_model
from agents.tool.memory.session import Session, Entity, Fact


EXTRACT_PROMPT = """从以下对话中提取结构化信息。

对话:
{conversation}

请输出 JSON 格式:
{{
  "entities": [
    {{"name": "实体名", "type": "类型", "attributes": {{"属性": "值"}}}}
  ],
  "facts": [
    {{"content": "事实描述", "confidence": 0.9}}
  ],
  "preferences": {{
    "偏好名": "偏好值"
  }}
}}

只输出 JSON，不要解释。"""


async def extract_knowledge(session: Session, new_messages: list[Message]) -> None:
    """从新对话中提取知识，更新 L3 记忆。"""
    if not new_messages:
        return

    model = get_chat_model("qwen")  # 用小模型做提取

    conversation = "\n".join([f"[{m.role}]: {m.content}" for m in new_messages])
    response = await model.ainvoke([
        HumanMessage(content=EXTRACT_PROMPT.format(conversation=conversation))
    ])

    try:
        data = json.loads(response.content)

        # 更新实体
        for e in data.get("entities", []):
            name = e["name"]
            if name in session.entities:
                session.entities[name].attributes.update(e.get("attributes", {}))
            else:
                session.entities[name] = Entity(**e)

        # 更新事实
        for f in data.get("facts", []):
            session.facts.append(Fact(content=f["content"], confidence=f.get("confidence", 0.9)))

        # 更新偏好
        session.preferences.update(data.get("preferences", {}))

    except json.JSONDecodeError:
        pass  # 解析失败静默忽略
```

---

## 8. 模型抽象层设计

### 8.1 Chat Model 工厂（`model/chat_model.py`）

```python
from typing import Protocol
from langchain_core.language_models import BaseChatModel
from agents.config.settings import settings


# 工厂注册表
_registry: dict[str, callable] = {}


def register_chat_model(name: str, factory: callable):
    _registry[name] = factory


def get_chat_model(name: str = None) -> BaseChatModel:
    """获取 Chat Model 实例。"""
    name = name or settings.chat_model_type
    if name not in _registry:
        raise ValueError(f"Chat model '{name}' not registered")
    return _registry[name]()


def init_chat_models():
    """初始化所有 Chat Model（注册工厂）。"""
    from agents.model.providers import ark, openai, deepseek, qwen, gemini
    # 各 provider 的 init 函数会调用 register_chat_model
    ark.init()
    openai.init()
    deepseek.init()
    qwen.init()
    gemini.init()
```

### 8.2 Provider 实现示例（`model/providers/ark.py`）

```python
from langchain_openai import ChatOpenAI
from agents.config.settings import settings
from agents.model.chat_model import register_chat_model


def _create_ark_model() -> ChatOpenAI:
    """创建火山引擎 Ark 模型（OpenAI 兼容接口）。"""
    return ChatOpenAI(
        model=settings.ark.chat_model,
        openai_api_key=settings.ark.key,
        openai_api_base="https://ark.cn-beijing.volces.com/api/v3",
        streaming=True,
    )


def init():
    register_chat_model("ark", _create_ark_model)
```

**为什么用 ChatOpenAI 而不是专用 SDK？**
- Ark、Qwen、DeepSeek 都兼容 OpenAI API 格式
- 统一用 `ChatOpenAI` 减少依赖，简化代码
- 只有 Gemini 和 Anthropic 需要专用 Chat Model

### 8.3 Token 计数器（`tool/token_counter.py`）

```python
import tiktoken


class TokenCounter:
    """Token 计数器，用于上下文窗口管理。"""

    def __init__(self, model: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(model)

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def fit_to_budget(self, parts: list[str], max_tokens: int) -> list[str]:
        """贪心算法：按优先级逐个添加，直到达到 Token 预算。"""
        total = 0
        result = []
        for part in parts:
            count = self.count(part)
            if total + count > max_tokens:
                # 尝试截断当前部分
                remaining = max_tokens - total
                if remaining > 100:  # 至少保留 100 tokens
                    truncated = self._truncate_to_tokens(part, remaining)
                    result.append(truncated)
                break
            result.append(part)
            total += count
        return result

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """截断文本到指定 Token 数。"""
        tokens = self.encoding.encode(text)[:max_tokens]
        return self.encoding.decode(tokens)
```

---

## 9. 工具层设计

### 9.1 MCP SQL 客户端（`tool/sql_tools/mcp_client.py`）

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

from agents.config.settings import settings

_session: ClientSession = None
_exit_stack: AsyncExitStack = None


async def init_mcp_tools():
    """初始化 MCP 连接。"""
    global _session, _exit_stack

    _exit_stack = AsyncExitStack()

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "mcp-server-mysql"],
        env={
            "MYSQL_HOST": settings.mysql.host,
            "MYSQL_PORT": str(settings.mysql.port),
            "MYSQL_USER": settings.mysql.username,
            "MYSQL_PASS": settings.mysql.password,
            "MYSQL_DB": settings.mysql.database,
        },
    )

    read_stream, write_stream = await _exit_stack.enter_async_context(
        stdio_client(server_params)
    )
    _session = await _exit_stack.enter_async_context(
        ClientSession(read_stream, write_stream)
    )
    await _session.initialize()


async def execute_sql(sql: str) -> str:
    """通过 MCP 执行 SQL。"""
    result = await _session.call_tool("mysql_query", {"sql": sql})
    return result.content[0].text


async def list_tables() -> str:
    """列出所有表。"""
    result = await _session.call_tool("list_tables", {})
    return result.content[0].text


async def close_mcp():
    """关闭 MCP 连接。"""
    if _exit_stack:
        await _exit_stack.aclose()
```

### 9.2 SQL 安全分析（`tool/sql_tools/safety.py`）

```python
import re
from dataclasses import dataclass, field


@dataclass
class SafetyReport:
    risks: list[str] = field(default_factory=list)
    estimated_rows: int = 0
    required_permissions: list[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return len(self.risks) == 0


class SQLSafetyChecker:
    """SQL 静态安全分析器。"""

    DANGEROUS_PATTERNS = [
        (r"\bDROP\s+TABLE\b", "DROP TABLE"),
        (r"\bDROP\s+DATABASE\b", "DROP DATABASE"),
        (r"\bTRUNCATE\b", "TRUNCATE"),
        (r"\bDELETE\b.*\bWHERE\b.*1\s*=\s*1", "DELETE with always-true WHERE"),
        (r"\bDELETE\b(?!\s.*\bWHERE\b)", "DELETE without WHERE"),
        (r"\bUPDATE\b.*\bSET\b.*\bWHERE\b.*1\s*=\s*1", "UPDATE with always-true WHERE"),
        (r"\bALTER\s+TABLE\b", "ALTER TABLE"),
        (r"\bGRANT\b", "GRANT"),
        (r"\bREVOKE\b", "REVOKE"),
    ]

    def check(self, sql: str) -> SafetyReport:
        report = SafetyReport()
        sql_upper = sql.upper()

        for pattern, risk_name in self.DANGEROUS_PATTERNS:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                report.risks.append(risk_name)

        # 估算影响行数
        if "LIMIT" not in sql_upper:
            report.estimated_rows = -1  # 无 LIMIT，未知

        # 权限检查
        if any(r in ("DROP TABLE", "DROP DATABASE", "ALTER TABLE") for r in report.risks):
            report.required_permissions.append("DDL")
        elif "DELETE" in sql_upper or "UPDATE" in sql_upper:
            report.required_permissions.append("DML_WRITE")
        else:
            report.required_permissions.append("DML_READ")

        return report
```

### 9.3 文档处理（`tool/document/`）

```python
# tool/document/loader.py
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
)

LOADER_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".docx": Docx2txtLoader,
}

def get_loader(file_path: str):
    import os
    ext = os.path.splitext(file_path)[1].lower()
    return LOADER_MAP.get(ext, TextLoader)(file_path)
```

```python
# tool/document/splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agents.config.settings import settings

def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )
```

---

## 10. SFT 数据管线设计

### 10.1 Callback Handler（`tool/sft/callback.py`）

```python
from typing import Any, Optional
from uuid import uuid4
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from agents.tool.sft.storage import save_sample
from agents.tool.sft.annotator import annotate


class SFTCallbackHandler(BaseCallbackHandler):
    """自动采集 ChatModel 调用数据的 Callback Handler。"""

    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self._inputs: dict[str, Any] = {}

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id, **kwargs):
        self._inputs[str(run_id)] = {
            "prompts": prompts,
            "model": serialized.get("name", "unknown"),
            "timestamp": datetime.now().isoformat(),
        }

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs):
        import asyncio

        run_id_str = str(run_id)
        input_data = self._inputs.pop(run_id_str, None)
        if not input_data:
            return

        sample = {
            "id": str(uuid4()),
            "agent_id": self.agent_id,
            "model": input_data["model"],
            "input": input_data["prompts"],
            "output": [g.text for g in response.generations[0]],
            "timestamp": input_data["timestamp"],
            "label": 0,  # 未标注
        }

        # 异步保存 + 标注
        asyncio.create_task(_save_and_annotate(sample))


async def _save_and_annotate(sample: dict):
    """异步保存并标注。"""
    await save_sample(sample)
    await annotate(sample)
    await save_sample(sample)  # 保存标注后的版本
```

### 10.2 教师模型标注（`tool/sft/annotator.py`）

```python
import json
from langchain_core.messages import HumanMessage
from agents.model.chat_model import get_chat_model


ANNOTATE_PROMPT = """你是一个 AI 回答质量评估专家。请评估以下对话中 AI 的回答质量。

用户问题:
{question}

AI 回答:
{answer}

请评估并输出 JSON:
{{
  "score": 0.0-1.0,
  "reasoning": "评估理由",
  "corrected": "如果回答有误，请提供修正后的正确回答"
}}

只输出 JSON，不要解释。"""


async def annotate(sample: dict) -> None:
    """使用教师模型标注样本。"""
    model = get_chat_model("deepseek")  # 用 DeepSeek 做教师

    question = sample["input"][-1] if sample["input"] else ""
    answer = sample["output"][0] if sample["output"] else ""

    response = await model.ainvoke([
        HumanMessage(content=ANNOTATE_PROMPT.format(question=question, answer=answer))
    ])

    try:
        data = json.loads(response.content)
        sample["score"] = data.get("score", 0)
        sample["reasoning"] = data.get("reasoning", "")
        sample["corrected"] = data.get("corrected", "")
        sample["is_annotated"] = True
    except json.JSONDecodeError:
        pass
```

---

## 11. 存储层设计

### 11.1 Redis 客户端（`tool/storage/redis_client.py`）

```python
import redis.asyncio as redis
from typing import Optional
from agents.config.settings import settings

_client: Optional[redis.Redis] = None


async def init_redis():
    global _client
    _client = redis.Redis(
        host=settings.redis.addr.split(":")[0],
        port=int(settings.redis.addr.split(":")[1]),
        password=settings.redis.password or None,
        db=settings.redis.db,
        max_connections=10,
        decode_responses=True,
    )


def get_redis() -> Optional[redis.Redis]:
    return _client


async def close_redis():
    if _client:
        await _client.close()
```

### 11.2 LangGraph Checkpointer（`tool/storage/checkpoint.py`）

```python
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from agents.config.settings import settings


async def get_checkpointer() -> AsyncRedisSaver:
    """获取 Redis Checkpointer，用于 LangGraph 的中断/恢复。"""
    return AsyncRedisSaver.from_conn_string(
        redis_url=f"redis://{settings.redis.addr}/{settings.redis.db}"
    )
```

**说明：** LangGraph 提供了官方的 Redis Checkpointer（`langgraph-checkpoint-redis`），直接替代 Go 版手动实现的 `RedisCheckPointStore`。

### 11.3 检索缓存（`tool/storage/retrieval_cache.py`）

```python
import hashlib
import json
from typing import Optional
from langchain_core.documents import Document

from agents.tool.storage.redis_client import get_redis


class RetrievalCache:
    """检索结果缓存。"""

    EMBEDDING_PREFIX = "cache:embedding:"
    RETRIEVAL_PREFIX = "cache:retrieval:"
    TTL = 3600  # 1 hour

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def get_retrieval(self, query: str) -> Optional[list[Document]]:
        redis = get_redis()
        if not redis:
            return None

        key = self.RETRIEVAL_PREFIX + self._hash(query)
        data = await redis.get(key)
        if data:
            return [Document(**d) for d in json.loads(data)]
        return None

    async def set_retrieval(self, query: str, docs: list[Document]):
        redis = get_redis()
        if not redis:
            return

        key = self.RETRIEVAL_PREFIX + self._hash(query)
        data = json.dumps([{"page_content": d.page_content, "metadata": d.metadata} for d in docs])
        await redis.set(key, data, ex=self.TTL)
```

---

## 12. API 层设计

### 12.1 FastAPI 应用（`api/app.py`）

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agents.api.routers import chat, rag, final, document
from agents.main import lifespan


app = FastAPI(title="Go-Agent Python", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/chat")
app.include_router(rag.router, prefix="/api/rag")
app.include_router(final.router, prefix="/api/final")
app.include_router(document.router, prefix="/api/document")

app.mount("/", StaticFiles(directory="static", html=True))
```

### 12.2 SSE 流式响应（`api/sse.py`）

```python
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
```

### 12.3 RAG Chat 流式路由（`api/routers/rag.py`）

```python
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import AsyncGenerator

from agents.flow.rag_chat import build_rag_chat_graph
from agents.api.sse import sse_response

router = APIRouter()


class RAGChatRequest(BaseModel):
    query: str
    session_id: str = "default_user"


@router.post("/chat/stream")
async def rag_chat_stream(req: RAGChatRequest, request: Request):
    """RAG Chat 流式端点。"""
    graph = build_rag_chat_graph()

    async def generate() -> AsyncGenerator[dict, None]:
        async for event in graph.astream_events(
            {"input": {"session_id": req.session_id, "query": req.query}},
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield {"event": "data", "data": chunk.content}

        yield {"event": "end", "data": ""}

    return await sse_response(generate(), request)
```

### 12.4 Final Graph 路由（`api/routers/final.py`）

```python
from fastapi import APIRouter, Request
from pydantic import BaseModel
from langgraph.types import Command

from agents.flow.final_graph import build_final_graph
from agents.tool.storage.checkpoint import get_checkpointer
from agents.api.sse import sse_response

router = APIRouter()

# 审批会话状态（生产环境应存 Redis）
_sessions: dict[str, dict] = {}


class FinalRequest(BaseModel):
    query: str
    session_id: str = "default_user"


@router.post("/invoke")
async def final_invoke(req: FinalRequest, request: Request):
    """Final Graph 端点（支持中断/恢复）。"""
    graph = build_final_graph()
    checkpointer = await get_checkpointer()

    # 检查是否有待审批的会话
    session_state = _sessions.get(req.session_id)

    if session_state and session_state.get("waiting_approval"):
        # 用户回复审批决定
        config = {"configurable": {"thread_id": req.session_id}}
        async for event in graph.astream_events(
            Command(resume=req.query),
            config=config,
            version="v2",
        ):
            # 流式返回
            ...
        _sessions.pop(req.session_id, None)
    else:
        # 新请求
        config = {"configurable": {"thread_id": req.session_id}}
        async for event in graph.astream_events(
            {"query": req.query, "session_id": req.session_id},
            config=config,
            version="v2",
        ):
            if event["event"] == "on_chain_end" and "interrupt" in str(event.get("data", {})):
                # 遇到中断，记录等待审批状态
                _sessions[req.session_id] = {"waiting_approval": True}
                yield {"event": "data", "data": "SQL 已生成，等待审批..."}
                break
            # ... 流式输出
```

---

## 13. 配置与部署

### 13.1 pyproject.toml

```toml
[project]
name = "agents-py"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # LangChain 核心
    "langchain>=0.3",
    "langchain-core>=0.3",
    "langchain-community>=0.3",
    "langchain-openai>=0.2",
    "langchain-anthropic>=0.3",

    # LangGraph
    "langgraph>=0.2",
    "langgraph-checkpoint-redis>=0.1",

    # 向量数据库
    "langchain-milvus>=0.1",
    "pymilvus>=2.4",

    # Elasticsearch
    "langchain-elasticsearch>=0.3",
    "elasticsearch>=8.12",

    # 文档处理
    "langchain-text-splitters>=0.3",
    "pypdf>=4.0",
    "docx2txt>=0.8",
    "unstructured>=0.15",

    # 模型
    "tiktoken>=0.7",

    # 重排序
    "sentence-transformers>=3.0",

    # 数据分析
    "pandas>=2.0",
    "numpy>=1.24",

    # 存储
    "redis>=5.0",

    # MCP
    "mcp>=1.0",

    # API
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "sse-starlette>=2.0",

    # 配置
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "httpx>=0.27",
]
```

### 13.2 docker-compose.yaml

```yaml
version: "3.8"
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.18
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision
      ETCD_QUOTA_BACKEND_BYTES: "4294967296"

  minio:
    image: minio/minio:RELEASE.2023-03-20
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /minio_data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"

  milvus:
    image: milvusdb/milvus:v2.5.10
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    ports:
      - "19530:19530"
    depends_on: [etcd, minio]

  elasticsearch:
    image: elasticsearch:8.12.0
    environment:
      discovery.type: single-node
      xpack.security.enabled: "false"
      ES_JAVA_OPTS: "-Xms512m -Xmx512m"
    ports:
      - "9200:9200"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### 13.3 .env.example

```bash
# 模型配置
CHAT_MODEL_TYPE=ark
EMBEDDING_MODEL_TYPE=qwen

# Ark
ARK_KEY=your-ark-key
ARK_CHAT_MODEL=doubao-seed-2-0-code-preview-260215
ARK_EMBEDDING_MODEL=ep-xxx

# Qwen
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_KEY=your-qwen-key
QWEN_EMBEDDING_MODEL=text-embedding-v3

# Milvus
MILVUS_ADDR=localhost:19530
MILVUS_USERNAME=root
MILVUS_PASSWORD=milvus
MILVUS_COLLECTION_NAME=agents_py
TOPK=5

# Elasticsearch
ES_ADDRESS=http://localhost:9200
ES_INDEX=agents_py_docs

# Redis
REDIS_ADDR=localhost:6379

# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USERNAME=root
MYSQL_PASSWORD=your-password
MYSQL_DATABASE=your-database
```

---

## 14. Go -> Python 映射速查表

| Go 文件 | Python 文件 | 说明 |
|---------|------------|------|
| `main.go` | `main.py` + `api/app.py` | 入口 + FastAPI 应用 |
| `config/config.go` | `config/settings.py` | Pydantic Settings |
| `api/router.go` | `api/app.py` | 路由注册 |
| `api/chat.go` | `api/routers/chat.py` | Chat 端点 |
| `api/rag_chat_stream.go` | `api/routers/rag.py` | RAG 流式端点 |
| `api/final_graph.go` | `api/routers/final.py` | Final 端点 + 中断/恢复 |
| `flow/rag_chat.go` | `flow/rag_chat.py` | LangGraph StateGraph |
| `flow/sql_react.go` | `flow/sql_react.py` | LangGraph + interrupt |
| `flow/analyst_graph.go` | `flow/analyst.py` | LangGraph 并行分支 |
| `flow/final_graph.go` | `flow/final_graph.py` | 意图路由 + 条件边 |
| `model/chat_model/chat_model.go` | `model/chat_model.py` | 工厂注册模式 |
| `model/chat_model/ark.go` | `model/providers/ark.py` | ChatOpenAI 包装 |
| `model/embedding_model/qwen.go` | `model/providers/qwen.py` | OpenAI Embeddings 包装 |
| `model/format_tool/ark_format.go` | `model/format_tool.py` | Pydantic + with_structured_output |
| `rag/rag_flow/index.go` | `rag/indexing.py` | Loader/Splitter/VectorStore |
| `rag/rag_flow/retriever.go` | `rag/retriever.py` | 混合检索 + RRF + Reranker |
| `rag/rag_tools/query.go` | `rag/query_rewrite.py` | LLM 查询重写 |
| `tool/memory/session.go` | `tool/memory/session.py` | Pydantic BaseModel |
| `tool/memory/store.go` | `tool/memory/store.py` | Redis + 内存双模式 |
| `tool/memory/compressor.go` | `tool/memory/compressor.py` | LLM 摘要压缩 |
| `tool/storage/redis.go` | `tool/storage/redis_client.py` | redis.asyncio |
| `tool/storage/checkpoint_store.go` | `tool/storage/checkpoint.py` | langgraph-checkpoint-redis |
| `tool/storage/retrieval_cache.go` | `tool/storage/retrieval_cache.py` | Redis 缓存 |
| `tool/document/parser.go` | `tool/document/loader.py` | LangChain DocumentLoader |
| `tool/document/splitter.go` | `tool/document/splitter.py` | RecursiveCharacterTextSplitter |
| `tool/sql_tools/exectue.go` | `tool/sql_tools/mcp_client.py` | MCP Python SDK |
| `tool/analyst_tools/tools.go` | `tool/analyst_tools/parser.py` | pandas 替代手动解析 |
| `tool/sft/callback.go` | `tool/sft/callback.py` | BaseCallbackHandler |
| `tool/sft/annotator.go` | `tool/sft/annotator.py` | 教师模型标注 |
| `tool/sft/speculative.go` | `tool/sft/speculative.py` | 推测解码 |
| `algorithm/bm25.go` | `algorithm/bm25.py` | rank_bm25 库 |
| `algorithm/rrf.go` | `algorithm/rrf.py` | RRF 融合算法 |
| `docker-compose.yaml` | `docker-compose.yaml` | +Redis 容器 |
