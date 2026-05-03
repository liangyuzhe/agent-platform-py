# Agent Platform Py

基于 **LangChain + LangGraph** 构建的 AI Agent 平台，提供 RAG 对话、SQL 生成与执行、数据分析、对话记忆等核心能力。

## 核心特性

- **RAG 对话**：文档索引 + 混合检索（向量 + BM25）+ RRF 融合 + Cross-Encoder 重排序
- **SQL 生成与执行**：自然语言 → SQL → 人工审批 → MCP 执行，含 SQL 安全分析
- **数据分析**：SQL 结果 → 统计分析 → 图表生成（ECharts）+ 文字报告
- **三级记忆系统**：工作记忆 + 摘要记忆 + 知识记忆（实体/事实/偏好）
- **SFT 数据管线**：自动采集训练数据 + 教师模型标注 + JSONL 导出
- **多模型支持**：Ark（豆包）、OpenAI、DeepSeek、通义千问、Gemini

## 架构概览

```
┌─────────────────────────────────────────────────┐
│  API 层 (FastAPI)                                │
│  路由、SSE 流式、请求校验                          │
├─────────────────────────────────────────────────┤
│  Flow 编排层 (LangGraph)                         │
│  RAG Chat / SQL React / Analyst / Final Graph    │
├─────────────────────────────────────────────────┤
│  能力层                                          │
│  Model（LLM 工厂） / RAG（检索管线） / Tool（工具）│
├─────────────────────────────────────────────────┤
│  基础设施层                                       │
│  Config / Storage（Redis） / Algorithm（BM25/RRF）│
└─────────────────────────────────────────────────┘
```

## 项目结构

```
agent-platform-py/
├── pyproject.toml                  # 项目配置 + 依赖
├── .env.example                    # 环境变量模板
├── docker-compose.yaml             # 基础设施（Milvus、ES、Redis）
│
├── agents/                         # 主包
│   ├── main.py                     # 入口
│   ├── config/                     # 配置层
│   │   └── settings.py             # Pydantic Settings
│   │
│   ├── api/                        # API 层
│   │   ├── app.py                  # FastAPI 应用
│   │   ├── sse.py                  # SSE 流式响应
│   │   └── routers/                # 路由
│   │       ├── chat.py             # Chat 测试
│   │       ├── rag.py              # RAG 对话
│   │       ├── final.py            # 主调度（支持中断/恢复）
│   │       └── document.py         # 文档上传
│   │
│   ├── flow/                       # LangGraph 图编排
│   │   ├── state.py                # 共享状态定义
│   │   ├── rag_chat.py             # RAG Chat 图
│   │   ├── sql_react.py            # SQL React 图（含 Human-in-the-Loop）
│   │   ├── analyst.py              # 数据分析图
│   │   └── final_graph.py          # 主调度图
│   │
│   ├── model/                      # 模型抽象层
│   │   ├── chat_model.py           # Chat Model 工厂
│   │   ├── embedding_model.py      # Embedding Model 工厂
│   │   ├── format_tool.py          # 结构化输出工具
│   │   └── providers/              # 各提供商实现
│   │       ├── ark.py              # 火山引擎 Ark（豆包）
│   │       ├── openai.py           # OpenAI
│   │       ├── deepseek.py         # DeepSeek
│   │       ├── qwen.py             # 通义千问
│   │       └── gemini.py           # Google Gemini
│   │
│   ├── rag/                        # RAG 管线
│   │   ├── indexing.py             # 文档索引（Loader → Splitter → Store）
│   │   ├── retriever.py            # 混合检索（Milvus + ES BM25 + RRF）
│   │   ├── parent_retriever.py     # Parent Document RAG
│   │   ├── reranker.py             # Cross-Encoder 重排序
│   │   └── query_rewrite.py        # 查询重写（指代消解）
│   │
│   ├── tool/                       # 工具层
│   │   ├── memory/                 # 三级记忆系统
│   │   │   ├── session.py          # Session 数据模型
│   │   │   ├── store.py            # Session 存储（Redis + 内存）
│   │   │   ├── compressor.py       # LLM 摘要压缩
│   │   │   └── knowledge.py        # 知识记忆提取
│   │   ├── storage/                # 存储层
│   │   │   ├── redis_client.py     # Redis 连接
│   │   │   ├── checkpoint.py       # LangGraph Checkpointer
│   │   │   └── retrieval_cache.py  # 检索结果缓存
│   │   ├── document/               # 文档处理
│   │   │   ├── loader.py           # 文件加载器
│   │   │   ├── parser.py           # 文档解析
│   │   │   └── splitter.py         # 文本分块
│   │   ├── sql_tools/              # SQL 工具
│   │   │   ├── mcp_client.py       # MCP 客户端
│   │   │   ├── executor.py         # SQL 执行器
│   │   │   └── safety.py           # SQL 安全分析
│   │   ├── analyst_tools/          # 数据分析
│   │   │   ├── parser.py           # SQL 结果解析
│   │   │   ├── statistics.py       # 统计计算
│   │   │   └── chart.py            # ECharts 图表生成
│   │   ├── sft/                    # SFT 数据管线
│   │   │   ├── callback.py         # 数据采集 Callback
│   │   │   ├── annotator.py        # 教师模型标注
│   │   │   └── storage.py          # 样本存储 + JSONL 导出
│   │   ├── trace/                  # 可观测性
│   │   │   └── callback.py         # 追踪 Callback
│   │   └── token_counter.py        # Token 计数器
│   │
│   ├── algorithm/                  # 算法
│   │   ├── bm25.py                 # BM25 实现
│   │   └── rrf.py                  # RRF 融合
│   │
│   └── static/                     # 前端
│       └── index.html              # Chat + SQL Agent + 文档上传 UI
│
├── tests/                          # 测试
├── docs/                           # 技术文档
│   └── python_langchain_design.md  # Python 版设计文档
│
└── data/                           # 数据目录
    └── sft/                        # SFT 训练数据
```

## 快速开始

### 1. 环境准备

```bash
# Python 3.11+
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 如需链路追踪（CozeLoop）
pip install cozeloop
```

### 2. 启动基础设施

```bash
docker-compose up -d
```

启动以下服务：
| 服务 | 端口 | 说明 |
|------|------|------|
| Milvus | 19530 | 向量数据库 |
| Attu | 8000 | Milvus Web UI |
| Elasticsearch | 9200 | 全文检索 |
| Redis | 6379 | 缓存 + CheckPoint |
| MinIO | 9000/9001 | Milvus 对象存储 |

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

必须配置的变量：

```bash
# 模型（至少配置一个）
CHAT_MODEL_TYPE=ark
ARK_KEY=your-ark-key
ARK_CHAT_MODEL=doubao-seed-2-0-code-preview-260215

# Embedding
EMBEDDING_MODEL_TYPE=qwen
QWEN_KEY=your-qwen-key
QWEN_EMBEDDING_MODEL=text-embedding-v3

# 向量数据库
MILVUS_ADDR=localhost:19530

# ES
ES_ADDRESS=http://localhost:9200

# Redis
REDIS_ADDR=localhost:6379
```

### 4. 启动服务

```bash
# 方式 1：直接运行
python -m agents.main

# 方式 2：uvicorn
uvicorn agents.api.app:app --host 0.0.0.0 --port 8080 --reload
```

服务启动后访问：

| 页面 | 路径 | 说明 |
|------|------|------|
| Chat UI | http://localhost:8080/ | RAG 对话（Tab 1） |
| SQL Agent | http://localhost:8080/ | 意图自动路由：SQL 查询 / 普通对话（Tab 2） |
| 文档上传 | http://localhost:8080/ | 上传文档并索引到 RAG（Tab 3） |
| API 文档 | http://localhost:8080/docs | Swagger UI |
| 健康检查 | http://localhost:8080/health | 服务状态 |

## API 接口

### Chat 测试

```bash
# 非流式
curl -X POST http://localhost:8080/api/chat/test \
  -H "Content-Type: application/json" \
  -d '{"question": "你好", "history": []}'

# 流式（SSE）
curl -X POST http://localhost:8080/api/chat/test/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "介绍一下自己"}'
```

### RAG 对话

```bash
# 文档索引
curl -X POST http://localhost:8080/api/rag/insert \
  -F "file=@document.pdf"

# RAG 问答
curl -X POST http://localhost:8080/api/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "文档中提到了什么？", "session_id": "user1"}'

# RAG 流式对话
curl -X POST http://localhost:8080/api/rag/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "继续刚才的话题", "session_id": "user1"}'
```

### 主调度（意图路由）

```bash
# 自动分类意图：SQL 查询 vs 普通对话
curl -X POST http://localhost:8080/api/final/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "查询最近 7 天的订单数量", "session_id": "user1"}'
```

### 文档上传

```bash
# 上传文档并索引（默认 RAG 模式）
curl -X POST http://localhost:8080/api/document/insert \
  -F "file=@document.pdf"

# 指定 Parent Document RAG 模式
curl -X POST http://localhost:8080/api/document/insert \
  -F "file=@document.pdf" \
  -F "rag_mode=parent"
```

## 核心设计

### 1. LangGraph 图编排

所有业务流程用 **StateGraph** 定义，节点间通过 TypedDict 共享状态：

```python
# 定义状态
class RAGChatState(TypedDict):
    query: str
    docs: list[Document]
    messages: Annotated[list[BaseMessage], add_messages]
    answer: str

# 构建图
graph = StateGraph(RAGChatState)
graph.add_node("retrieve", retrieve)
graph.add_node("chat", chat)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "chat")
graph.add_edge("chat", END)

app = graph.compile()
result = await app.ainvoke({"query": "..."})
```

**优势：**
- 节点自动并行执行（无依赖的节点）
- 原生支持中断/恢复（Human-in-the-Loop）
- 状态流转清晰，可调试

### 2. 混合检索 + 重排序

```
Query → [Milvus(向量), ES(BM25)] → RRF 融合 → Cross-Encoder 重排序 → Top-K
```

- **Milvus**：稠密向量相似度检索
- **Elasticsearch**：BM25 关键词检索（不是向量！）
- **RRF**：Reciprocal Rank Fusion，无需调参
- **Cross-Encoder**：`BAAI/bge-reranker-v2-m3`，精排序

### 3. 三级记忆系统

| 层级 | 存储 | 内容 | 生命周期 |
|------|------|------|---------|
| L1 工作记忆 | Session.History | 最近 3 轮对话原文 | 压缩触发前 |
| L2 摘要记忆 | Session.Summary | LLM 生成的对话摘要 | 跨压缩周期累积 |
| L3 知识记忆 | Session.Entities/Facts/Preferences | 结构化实体、事实、偏好 | 持久累积 |

**压缩流程：**
1. 历史超过阈值 → 取出旧消息
2. LLM 合并旧消息 + 已有摘要 → 新摘要
3. 保留最近 N 轮作为工作记忆

**知识提取：**
- 每轮对话后异步提取实体、事实、偏好
- 用于 Query 重写和 Prompt 增强

### 4. SQL 安全分析

```python
checker = SQLSafetyChecker()
report = checker.check("DELETE FROM users WHERE 1=1")
# report.risks = ["DELETE with always-true WHERE"]
# report.is_safe = False
```

检测模式：DROP TABLE、DELETE without WHERE、TRUNCATE、UPDATE with always-true WHERE 等。

### 5. Human-in-the-Loop 审批

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

审批不通过时，用户可提供修改意见，系统自动回到 SQL 生成节点重新生成。

### 6. Token 预算管理

```python
counter = TokenCounter()
parts = [summary, history, docs, query]
fitted = counter.fit_to_budget(parts, max_tokens=28672)
# 自动裁剪低优先级内容，防止超出上下文窗口
```

### 7. SFT 数据飞轮

```
ChatModel 调用 → SFTHandler 自动采集 → 教师模型（DeepSeek）评分/修正 → JSONL 导出
```

- 每次 LLM 调用自动记录输入输出
- 教师模型评估质量并提供修正答案
- 导出为训练格式，用于微调小模型

## 配置说明

### 模型配置

```bash
# 主模型（用于生成、SQL、分析）
CHAT_MODEL_TYPE=ark          # ark / openai / deepseek / qwen / gemini

# Embedding 模型（用于向量化）
EMBEDDING_MODEL_TYPE=qwen    # ark / openai / qwen / gemini
```

### RAG 参数

```bash
CHUNK_SIZE=1000              # 分块大小（字符数）
CHUNK_OVERLAP=200            # 分块重叠
TOP_K=5                      # 检索返回文档数
```

### 记忆参数

```bash
MAX_HISTORY_LEN=3            # 保留最近 N 轮对话
```

### 链路追踪

LangSmith 通过环境变量自动启用，CozeLoop 需额外安装：

```bash
pip install cozeloop
```

`.env` 配置：

```bash
# LangSmith
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key

# CozeLoop (JWT OAuth)
COZELOOP_TRACING=true
COZELOOP_WORKSPACE_ID=your-workspace-id
COZELOOP_JWT_OAUTH_CLIENT_ID=your-client-id
COZELOOP_JWT_OAUTH_PRIVATE_KEY=your-private-key
COZELOOP_JWT_OAUTH_PUBLIC_KEY_ID=your-public-key-id
```

## Docker 部署

```bash
# 启动所有基础设施
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f milvus
```

## 技术文档

- [Python 版设计文档](python_langchain_design.md)

## 依赖说明

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

## License

MIT
