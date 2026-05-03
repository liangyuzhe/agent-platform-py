# Agent Platform Python 技术设计文档

## 目录

1. [项目概述](#1-项目概述)
2. [技术栈](#2-技术栈)
3. [项目结构](#3-项目结构)
4. [核心模块设计](#4-核心模块设计)
5. [Flow 编排层：LangGraph 实现](#5-flow-编排层langgraph-实现)
6. [RAG 管线设计](#6-rag-管线设计)
7. [记忆系统设计](#7-记忆系统设计)
8. [模型抽象层设计](#8-模型抽象层设计)
9. [工具层设计](#9-工具层设计)
10. [API 层设计](#10-api-层设计)
11. [配置与部署](#11-配置与部署)

---

## 1. 项目概述

### 1.1 目标

构建一个基于 **LangChain + LangGraph** 的 AI Agent 平台，提供 RAG 对话、SQL 生成与执行、数据分析、对话记忆等核心能力。

### 1.2 核心能力

| 能力 | 说明 |
|------|------|
| RAG 对话 | 文档索引 + 混合检索（向量 + BM25）+ RRF 融合 + Cross-Encoder 重排序 |
| SQL 生成与执行 | 自然语言 → SQL → 人工审批 → MCP 执行 |
| 数据分析 | SQL 结果 → 统计分析 → 图表生成 → 文字报告 |
| 对话记忆 | 三级记忆（工作记忆 + 摘要记忆 + 知识记忆） |
| SFT 数据采集 | 自动收集 + 教师标注 + JSONL 导出 |

### 1.3 设计亮点

| 特性 | 说明 |
|------|------|
| 混合检索 | Milvus 向量 + ES BM25 真正混合 |
| 三级记忆 | 工作记忆 + 摘要记忆 + 知识记忆 |
| Token 管理 | 显式 Token 计数 + 上下文窗口管理 |
| 模型切换 | 配置驱动 + 多 provider 支持 |
| SQL 安全 | 静态分析 + 风险评估 |
| 检索质量 | Cross-Encoder Reranker 重排序 |

---

## 2. 技术栈

### 2.1 核心依赖

| 库 | 用途 |
|----|------|
| `langchain` + `langgraph` | LLM 抽象 + 图编排 |
| `langchain-milvus` | Milvus 向量存储 |
| `langchain-elasticsearch` | ES 检索 |
| `sentence-transformers` | Cross-Encoder 重排序 |
| `tiktoken` | Token 计数 |
| `fastapi` + `uvicorn` | HTTP 服务 |
| `sse-starlette` | SSE 流式响应 |
| `mcp` | MCP 协议（SQL 执行） |
| `pydantic-settings` | 配置管理 |
| `redis` | 缓存 + CheckPoint |

### 2.2 模型支持

| Provider | Chat Model | Embedding Model |
|----------|-----------|-----------------|
| Ark（豆包） | doubao-seed-2-0 | - |
| OpenAI | gpt-4 | text-embedding-3-small |
| DeepSeek | deepseek-chat | - |
| 通义千问 | qwen-max | text-embedding-v3 |
| Gemini | gemini-pro | - |

---

## 3. 项目结构

```
agents/
├── main.py                     # 入口
├── config/                     # 配置层
│   └── settings.py             # Pydantic Settings
│
├── api/                        # API 层
│   ├── app.py                  # FastAPI 应用
│   ├── sse.py                  # SSE 流式响应
│   └── routers/                # 路由
│       ├── chat.py             # Chat 测试
│       ├── rag.py              # RAG 对话
│       ├── final.py            # 主调度
│       └── document.py         # 文档上传
│
├── flow/                       # LangGraph 图编排
│   ├── state.py                # 共享状态定义
│   ├── rag_chat.py             # RAG Chat 图
│   ├── sql_react.py            # SQL React 图
│   ├── analyst.py              # 数据分析图
│   └── final_graph.py          # 主调度图
│
├── model/                      # 模型抽象层
│   ├── chat_model.py           # Chat Model 工厂
│   ├── embedding_model.py      # Embedding Model 工厂
│   ├── format_tool.py          # 结构化输出工具
│   └── providers/              # 各提供商实现
│       ├── ark.py
│       ├── openai.py
│       ├── deepseek.py
│       ├── qwen.py
│       └── gemini.py
│
├── rag/                        # RAG 管线
│   ├── indexing.py             # 文档索引
│   ├── retriever.py            # 混合检索
│   ├── parent_retriever.py     # Parent Document RAG
│   ├── reranker.py             # Cross-Encoder 重排序
│   └── query_rewrite.py        # 查询重写
│
├── tool/                       # 工具层
│   ├── memory/                 # 三级记忆系统
│   │   ├── session.py          # Session 数据模型
│   │   ├── store.py            # Session 存储
│   │   ├── compressor.py       # LLM 摘要压缩
│   │   └── knowledge.py        # 知识记忆提取
│   ├── storage/                # 存储层
│   │   ├── redis_client.py     # Redis 连接
│   │   └── checkpoint.py       # LangGraph Checkpointer
│   ├── document/               # 文档处理
│   │   ├── loader.py           # 文件加载器
│   │   ├── parser.py           # 文档解析
│   │   └── splitter.py         # 文本分块
│   ├── sql_tools/              # SQL 工具
│   │   ├── mcp_client.py       # MCP 客户端
│   │   └── safety.py           # SQL 安全分析
│   ├── analyst_tools/          # 数据分析
│   │   ├── parser.py           # SQL 结果解析
│   │   ├── statistics.py       # 统计计算
│   │   └── chart.py            # ECharts 图表生成
│   ├── sft/                    # SFT 数据管线
│   │   ├── callback.py         # 数据采集 Callback
│   │   ├── annotator.py        # 教师模型标注
│   │   └── storage.py          # 样本存储 + JSONL 导出
│   └── token_counter.py        # Token 计数器
│
├── algorithm/                  # 算法
│   ├── bm25.py                 # BM25 实现
│   └── rrf.py                  # RRF 融合
│
└── static/                     # 前端
    └── index.html              # Chat UI
```

---

## 4. 核心模块设计

### 4.1 配置管理

使用 Pydantic Settings 管理所有配置，支持环境变量和 `.env` 文件：

```python
class Settings(BaseSettings):
    chat_model_type: Literal["ark", "openai", "qwen", "gemini", "deepseek"] = "ark"
    embedding_model_type: Literal["ark", "openai", "qwen"] = "qwen"

    # 嵌套配置
    ark: ArkSettings
    openai: OpenAISettings
    qwen: QwenSettings
    milvus: MilvusSettings
    es: ElasticSearchSettings
    redis: RedisSettings
    rag: RAGSettings
    memory: MemorySettings
```

### 4.2 模型工厂

统一的模型创建接口，支持多 provider：

```python
# Chat Model
model = get_chat_model("ark")  # 返回 BaseChatModel

# Embedding Model
embeddings = get_embedding_model("qwen")  # 返回 Embeddings
```

---

## 5. Flow 编排层：LangGraph 实现

### 5.1 RAG Chat 图

```python
class RAGChatState(TypedDict):
    input: dict
    session_id: str
    query: str
    rag_mode: str
    session: dict
    rewritten_query: str
    docs: list[Document]
    messages: Annotated[list[BaseMessage], add_messages]
    answer: str

# 构建图
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
```

### 5.2 SQL React 图（Human-in-the-Loop）

```python
# LangGraph interrupt 机制
user_decision = interrupt({
    "sql": "SELECT * FROM orders",
    "message": "请审批此 SQL",
})

if user_decision == "YES":
    return Command(goto="execute_sql")
else:
    return Command(goto="sql_generate", update={"refine_feedback": user_decision})
```

### 5.3 主调度图

根据意图自动路由到 RAG Chat 或 SQL React：

```python
graph = StateGraph(FinalGraphState)
graph.add_node("intent_detect", detect_intent)
graph.add_node("rag_chat", rag_chat_subgraph)
graph.add_node("sql_react", sql_react_subgraph)

# 条件路由
graph.add_conditional_edges(
    "intent_detect",
    lambda state: "sql" if state["intent"] == "SQL" else "chat",
    {"sql": "sql_react", "chat": "rag_chat"},
)
```

---

## 6. RAG 管线设计

### 6.1 混合检索架构

```
Query → [Milvus(向量), ES(BM25)] → RRF 融合 → Cross-Encoder 重排序 → Top-K
```

- **Milvus**：稠密向量相似度检索
- **Elasticsearch**：BM25 关键词检索
- **RRF**：Reciprocal Rank Fusion，无需调参
- **Cross-Encoder**：`BAAI/bge-reranker-v2-m3`，精排序

### 6.2 Parent Document RAG

索引时使用父子分块：
- 父块（大）：存储完整上下文
- 子块（小）：用于精确检索

检索时：
1. 子块检索（Milvus + ES BM25 + RRF）
2. Cross-Encoder 重排序
3. 获取父块（保留上下文完整性）

### 6.3 查询重写

利用对话上下文消解指代：

```python
rewritten = await rewrite_query(
    summary="用户询问 Python 相关问题",
    history=[{"role": "user", "content": "什么是列表推导式？"}],
    query="它的优点是什么？",  # → "列表推导式的优点是什么？"
)
```

---

## 7. 记忆系统设计

### 7.1 三级记忆架构

| 层级 | 存储 | 内容 | 生命周期 |
|------|------|------|---------|
| L1 工作记忆 | Session.History | 最近 N 轮对话原文 | 压缩触发前 |
| L2 摘要记忆 | Session.Summary | LLM 生成的对话摘要 | 跨压缩周期累积 |
| L3 知识记忆 | Session.Entities/Facts/Preferences | 结构化实体、事实、偏好 | 持久累积 |

### 7.2 压缩流程

1. 历史超过阈值 → 取出旧消息
2. LLM 合并旧消息 + 已有摘要 → 新摘要
3. 保留最近 N 轮作为工作记忆

### 7.3 知识提取

每轮对话后异步提取：
- 实体（Entity）：人名、地点、组织
- 事实（Fact）：用户陈述的事实
- 偏好（Preference）：用户偏好

---

## 8. 模型抽象层设计

### 8.1 Chat Model 工厂

```python
# 注册 provider
register_chat_model("ark", lambda: ChatOpenAI(...))
register_chat_model("openai", lambda: ChatOpenAI(...))

# 使用
model = get_chat_model("ark")
response = await model.ainvoke([HumanMessage(content="你好")])
```

### 8.2 Embedding Model 工厂

```python
register_embedding_model("qwen", lambda: OpenAIEmbeddings(...))

embeddings = get_embedding_model("qwen")
vector = embeddings.embed_query("Hello World")
```

---

## 9. 工具层设计

### 9.1 SQL 安全分析

```python
checker = SQLSafetyChecker()
report = checker.check("DELETE FROM users WHERE 1=1")
# report.risks = ["DELETE with always-true WHERE"]
# report.is_safe = False
```

检测模式：DROP TABLE、DELETE without WHERE、TRUNCATE、UPDATE with always-true WHERE 等。

### 9.2 Token 预算管理

```python
counter = TokenCounter()
parts = [summary, history, docs, query]
fitted = counter.fit_to_budget(parts, max_tokens=28672)
# 自动裁剪低优先级内容，防止超出上下文窗口
```

### 9.3 SFT 数据飞轮

```
ChatModel 调用 → SFTHandler 自动采集 → 教师模型（DeepSeek）评分/修正 → JSONL 导出
```

- 每次 LLM 调用自动记录输入输出
- 教师模型评估质量并提供修正答案
- 导出为训练格式，用于微调小模型

---

## 10. API 层设计

### 10.1 FastAPI 应用

```python
app = FastAPI(title="Agents-Py")

# 路由注册
app.include_router(chat.router, prefix="/api/chat")
app.include_router(rag.router, prefix="/api/rag")
app.include_router(final.router, prefix="/api/final")
app.include_router(document.router, prefix="/api/document")
```

### 10.2 SSE 流式响应

```python
@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    async def generate():
        async for event in graph.astream_events(...):
            if event["event"] == "on_chat_model_stream":
                yield {"event": "data", "data": chunk.content}

    return await sse_response(generate(), request)
```

### 10.3 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/chat/test` | POST | Chat 测试 |
| `/api/chat/test/stream` | POST | Chat 流式测试 |
| `/api/rag/ask` | POST | RAG 问答 |
| `/api/rag/chat/stream` | POST | RAG 流式对话 |
| `/api/rag/insert` | POST | 文档上传索引 |
| `/api/final/invoke` | POST | 主调度（意图路由） |
| `/health` | GET | 健康检查 |

---

## 11. 配置与部署

### 11.1 环境变量

```bash
# 模型配置
CHAT_MODEL_TYPE=ark          # ark / openai / deepseek / qwen / gemini
EMBEDDING_MODEL_TYPE=qwen    # ark / openai / qwen

# RAG 配置
RAG_MODE=traditional         # traditional / parent
RAG_CHUNK_SIZE=1024
RAG_TOP_K=5

# 基础设施
MILVUS_ADDR=localhost:19530
ES_ADDRESS=http://localhost:9200
REDIS_ADDR=localhost:6379
```

### 11.2 Docker 部署

```bash
# 启动基础设施
docker-compose up -d

# 启动服务
python -m agents.main
```

### 11.3 依赖安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## License

MIT
