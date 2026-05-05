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
12. [财务 Copilot 演进路线](#12-财务-copilot-演进路线)
13. [RAG 检索质量评估框架](#13-rag-检索质量评估框架)

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
| 检索评估 | LLM 自动生成评测集 + 多策略对比（Recall/MRR/NDCG） |

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
│   ├── registry.py             # 统一 Tool Registry（按分类注册 @tool）
│   ├── memory/                 # 三级记忆系统
│   │   ├── session.py          # Session 数据模型
│   │   ├── store.py            # Session 存储
│   │   ├── compressor.py       # LLM 摘要压缩
│   │   └── knowledge.py        # 知识记忆提取
│   ├── storage/                # 存储层
│   │   ├── redis_client.py     # Redis 连接
│   │   ├── checkpoint.py       # LangGraph Checkpointer
│   │   ├── domain_summary.py   # 领域摘要持久化（MySQL + Redis）
│   │   └── retrieval_cache.py  # 检索结果缓存
│   ├── document/               # 文档处理
│   │   ├── loader.py           # 文件加载器
│   │   ├── parser.py           # 文档解析
│   │   └── splitter.py         # 文本分块
│   ├── sql_tools/              # SQL 工具
│   │   ├── mcp_client.py       # MCP 客户端
│   │   ├── execute_tool.py     # @tool: execute_query
│   │   ├── schema_tool.py      # @tool: list_tables, describe_table
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

### 5.2 SQL React 图（ReAct 自纠错 + 自动补表 + Human-in-the-Loop）

SQL React 支持三层自纠错：

1. **执行错误重试**：SQL 执行失败 → LLM 分析错误 → 重新生成（最多 3 次）
2. **自动补表发现**：LLM 缺少关键表 → 自动 re-retrieve → 合并 schema → 重新生成（最多 3 轮）
3. **人工审批**：SQL 通过安全检查后 interrupt 等待用户确认

```python
# 流程：retrieve → check_docs → generate → safety → approve → execute
#                            ↑    ↓                       |
#                            | needs_more_tables?         |
#                            |  → re-retrieve missing     |
#                            |  → merge docs              |
#                            ←──┘                         |
#                                                  ↑      |
#                                                  error_analysis ← FAIL
#                                                  ↓       retry (max 3)
#                                                 END (success)

# State 新增字段
class SQLReactState(TypedDict):
    error: str | None          # SQL 执行错误信息
    retry_count: int           # 重试次数
    execution_history: list    # [{sql, result, error}]
```

**自动补表机制**（`sql_format_response` 工具）：

LLM 可通过工具参数声明缺少表，触发自动 re-retrieve：

```python
# LLM 调用 sql_format_response 时设置：
{
    "answer": "",
    "is_sql": False,
    "needs_more_tables": True,       # 声明缺少表
    "missing_tables": ["t_user"],    # 需要的表名
}

# 系统自动 re-retrieve → 合并 schema → 重新调 LLM
# 最多 3 轮，避免无限循环
```

人工审批使用 LangGraph interrupt 机制：

```python
user_decision = interrupt({
    "sql": "SELECT * FROM orders",
    "message": "请审批此 SQL",
})

if user_decision == "YES":
    return Command(goto="execute_sql")
else:
    return Command(goto="sql_generate", update={"refine_feedback": user_decision})
```

### 5.3 主调度图（多场景意图路由）

支持 7 种意图自动路由：

| 意图 | 路由目标 | 说明 |
|------|---------|------|
| `sql_query` | SQL React | 数据库查询 |
| `anomaly_detect` | Chat (暂代) | 异常归因 |
| `reconciliation` | Chat (暂代) | 资金核对 |
| `report` | Chat (暂代) | 报告生成 |
| `audit` | Chat (暂代) | 审计追踪 |
| `knowledge` | RAG Chat | 知识库问答 |
| `chat` | RAG Chat | 闲聊 |

```python
graph = StateGraph(FinalGraphState)
graph.add_node("classify_intent", classify_intent)
graph.add_node("sql_react", sql_react)
graph.add_node("chat_direct", chat_direct)

graph.add_conditional_edges("classify_intent", route_intent)
# route_intent 根据 intent 字段分发到对应子图
```

意图分类使用 LLM + 动态领域摘要（从 MySQL/Redis 加载，非硬编码）。

**各意图处理流程**：

**sql_query — 数据库查询**
```
用户输入 → classify_intent → sql_query
  → sql_retrieve (vector-only)
  → check_docs (schema 充足？)
    → 不足 → re-retrieve 缺失表 → 合并 → 重新生成
  → sql_generate (LLM 生成 SQL)
  → safety_check (安全审查)
    → 危险 → 拒绝
  → approve (interrupt 等待人工审批)
    → 拒绝 → 反馈 → 重新生成
    → 通过 → execute_sql
      → 失败 → error_analysis → retry (最多 3 次)
      → 成功 → 返回结果
```

**knowledge — 知识库问答**
```
用户输入 → classify_intent → knowledge
  → preprocess (加载 session)
  → rewrite (查询重写，消解指代)
  → retrieve (hybrid: Milvus + ES BM25 + RRF)
  → construct_messages (拼装 prompt + token 预算)
  → chat (LLM 生成回答)
  → 保存记忆 → 返回
```

**chat — 闲聊**
```
用户输入 → classify_intent → chat
  → 同 knowledge 流程（复用 RAG Chat 图）
```

**anomaly_detect — 异常归因（Phase 2）**
```
用户输入 → classify_intent → anomaly_detect
  → 指标查询 (SQL: 同比/环比)
  → 波动检测 (阈值/Z-score)
  → 归因分析 (LLM: 维度下钻)
  → 生成报告
```

**reconciliation — 资金核对（Phase 2）**
```
用户输入 → classify_intent → reconciliation
  → 多源拉取 (内部账本 + 链上/银行流水)
  → 逐条比对 (金额/时间/对方)
  → 差异标记
  → LLM 解释差异原因
  → 生成核对报告
```

**report — 报告生成（Phase 2）**
```
用户输入 → classify_intent → report
  → 意图解析 (日报/周报/月报？哪些指标？)
  → 数据聚合 (SQL: SUM/AVG/GROUP BY)
  → 趋势分析 (环比/同比)
  → LLM 生成文字报告
  → ECharts 生成图表
  → 组装输出
```

**audit — 审计追踪（Phase 2）**
```
用户输入 → classify_intent → audit
  → 目标识别 (哪笔交易/哪个科目？)
  → 凭证链查询 (凭证号 → 分录 → 发票 → 审批流)
  → 时间线构建
  → LLM 摘要审计要点
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

### 6.4 Reranker 配置与可观测性

**配置项**（`.env` 或环境变量）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-Encoder 模型名，空字符串禁用 rerank |
| `RAG_RERANKER_TOP_K` | `5` | Rerank 后保留的文档数 |
| `RAG_RERANK_THRESHOLD` | `0.1` | 最低 rerank 分数，低于此值的文档被过滤 |

**LangSmith 可观测性**：

- Milvus / ES 各自独立的 trace span（通过 callback 传播）
- Cross-Encoder Rerank 独立 span（`@traceable` 装饰器）
- 检索阶段日志记录耗时和文档数量，便于定位瓶颈

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

### 9.1 统一 Tool Registry

所有 LLM 可调用的工具通过 `ToolRegistry` 统一注册和管理，按分类组织：

```python
from agents.tool.registry import register, get_tools

# 注册工具到分类
@register("sql")
@tool
def execute_query(sql: str) -> str:
    """Execute a SQL query."""
    ...

# 按分类获取工具
sql_tools = get_tools("sql")        # [execute_query, list_tables, describe_table]
all_tools = get_tools()              # 所有已注册工具
```

工具分类：
- `sql`: execute_query, list_tables, describe_table
- `finance`（扩展）: get_balance, get_exchange_rate, get_kyc_info
- `knowledge`（扩展）: search_regulations, search_accounting_standards

### 9.2 SQL 安全分析

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

前端默认使用流式模式，LLM 输出逐字显示：

```python
@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    async def generate():
        async for event in graph.astream_events(..., config=config):
            if event["event"] == "on_chat_model_stream":
                yield {"event": "data", "data": chunk.content}
    return await sse_response(generate(), request)
```

SQL Agent 流式端点额外支持审批事件：流结束后检测 interrupt，emit `approval` 事件。

### 10.3 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/chat/test` | POST | Chat 测试 |
| `/api/chat/test/stream` | POST | Chat 流式测试 |
| `/api/rag/ask` | POST | RAG 问答（非流式） |
| `/api/rag/chat/stream` | POST | RAG 流式对话（前端默认） |
| `/api/document/insert` | POST | 文档上传索引 |
| `/api/final/invoke` | POST | 主调度（非流式） |
| `/api/final/invoke/stream` | POST | 主调度流式（前端默认，含审批事件） |
| `/api/final/approve` | POST | 审批恢复（SQL/工单） |
| `/api/admin/refresh-schemas` | POST | 全量刷新 Schema + 领域摘要 |
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
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3   # 空字符串禁用 rerank
RAG_RERANKER_TOP_K=5
RAG_RERANK_THRESHOLD=0.1

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

## 12. 财务 Copilot 演进路线

### 12.1 目标场景

| 场景 | 意图 | 子图 | 状态 |
|------|------|------|------|
| SQL 自动生成与纠错 | `sql_query` | SQL React (ReAct) | ✅ 已实现 |
| 财务异常波动归因 | `anomaly_detect` | Anomaly Graph | Phase 2 |
| 多表资金明细核对 | `reconciliation` | Reconciliation Graph | Phase 2 |
| 自动生成日报/周报/月报 | `report` | Report Graph | Phase 2 |
| 审计问答与凭证追踪 | `audit` | Audit Graph | Phase 2 |
| 财务制度/会计准则查询 | `knowledge` | RAG Chat | Phase 2 |
| 跨系统自动执行 | `agent_task` | Agent Graph | Phase 3 |

### 12.2 Phase 1 已完成

1. **统一 Tool Registry** — `agents/tool/registry.py`，工具按分类注册，LLM 动态选择
2. **SQL ReAct 自纠错** — 执行失败时 LLM 分析错误并重试（最多 3 次）
3. **多场景意图路由** — 7 种意图分类，Phase 2 逐步替换为专用子图

### 12.3 Phase 2 计划

- 异常归因子图：检测 → 上下文查询 → 原因分析 → 报告
- 资金核对子图：多源拉取 → 逐条比对 → 差异解释 → 报告
- 报告生成子图：指标聚合 → 趋势分析 → LLM 生成 → 图表
- 审计追踪子图：目标识别 → 凭证链查询 → 摘要

### 12.4 Phase 3 计划

- 链上余额查询（Moralis/Alchemy/TronGrid）
- KYC/用户标签拉取
- USDT/USD 汇率换算
- 结算单自动生成
- 异常工单自动发起

---

## 13. RAG 检索质量评估框架

### 13.1 概述

为量化比较不同检索策略的效果，构建了自动化评估框架。通过 LLM 自动生成评测数据集，对多种检索配置进行 Recall@K、MRR、NDCG@K 等指标的对比评估。

### 13.2 评测数据集生成

使用 LLM 从 Milvus 中已索引的 schema 文档自动生成 `(query, relevant_doc_ids)` 标注对：

```bash
python -m agents.eval.cli generate --num-per-table 3 --output eval_dataset.jsonl
```

**生成策略**：
- 遍历每个已索引的表 schema
- LLM 为每个表生成 3 条 query：至少 1 条单表查询 + 至少 1 条多表关联查询
- query 模拟真实业务人员的自然语言提问方式
- 自动去重

**数据格式**（JSONL）：
```json
{"query": "查询所有会计科目的余额方向", "relevant_doc_ids": ["schema_t_account"]}
{"query": "查看2025年每个成本中心的预算和实际发生额对比", "relevant_doc_ids": ["schema_t_budget", "schema_t_cost_center"]}
```

**当前数据集规模**：37 条 query，覆盖 15 张财务业务表。

### 13.3 评估指标

| 指标 | 核心拷问 | 关注点 | 优缺点 | 公式 |
| :--- | :--- | :--- | :--- | :--- |
| Recall@K | 找到了没有？ | 覆盖率（找得全不全） | 简单直观，但忽略了排序先后 | `hits_in_top_k / total_relevant` |
| MRR | 第一个对的答案来得早不早？ | 首位精准度（找得快不快） | 极其看重第一名，但忽略了后面还有没有其他好答案 | `1 / rank_of_first_relevant` |
| NDCG@K | 整体排得好不好？ | 排序质量（强相关的在前面吗） | 最全面、最符合人类直觉，但计算相对复杂 | `DCG@K / IDCG@K` |

**如何选择指标**：
- **快速筛查**：看 Recall@K — 相关文档有没有被捞出来
- **用户体验**：看 MRR — 用户第一个看到的答案是否正确
- **综合排序**：看 NDCG@K — 强相关文档是否排在前面，兼顾了排序质量

评估函数实现：`agents/eval/metrics.py`

```python
from agents.eval.metrics import evaluate_single, aggregate_metrics

# 单条 query 评估
result = evaluate_single(
    retrieved_ids=["schema_t_account", "schema_t_budget"],
    relevant_ids={"schema_t_account"},
    k_values=[1, 3, 5, 10],
)
# {"mrr": 1.0, "recall@1": 1.0, "recall@5": 1.0, "ndcg@5": 1.0, ...}
```

### 13.4 评测策略配置

对比 5 种检索策略变体：

| 策略 | mode | Reranker | 阈值 | 说明 |
|------|------|----------|------|------|
| `hybrid_rerank` | traditional | bge-reranker-v2-m3 | 0.01~0.3 | 当前默认，混合 + 重排序 |
| `hybrid_no_rerank` | traditional | None | - | 混合检索，无重排序 |
| `vector_only` | vector_only | None | - | 仅 Milvus 向量检索 |
| `es_only` | es_only | None | - | 仅 ES BM25 关键词检索 |
| `parent_doc` | parent | bge-reranker-v2-m3 | - | 子块检索 → 父块扩展 |

运行评估：
```bash
python -m agents.eval.cli run --dataset eval_dataset.jsonl --output eval_report.json --detail
```

### 13.5 评估结果（2025-05-05）

**测试环境**：15 张财务业务表 schema，37 条 LLM 生成的评测 query。

#### 第一轮：默认阈值（threshold=0.3）

| 策略 | MRR | Recall@1 | Recall@5 | NDCG@5 | 延迟 |
|------|-----|----------|----------|--------|------|
| hybrid_rerank (t=0.3) | 0.73 | 0.50 | 0.57 | 0.61 | 1010ms |
| hybrid_no_rerank | **0.96** | 0.60 | **0.94** | **0.92** | 289ms |
| vector_only | **0.97** | **0.62** | **0.94** | **0.93** | 225ms |
| es_only | 0.00 | 0.00 | 0.00 | 0.00 | 216ms |
| parent_doc | 0.00 | 0.00 | 0.00 | 0.00 | 21ms |

#### 第二轮：调整 Reranker 阈值

| 策略 | MRR | Recall@1 | Recall@5 | NDCG@5 | 延迟 |
|------|-----|----------|----------|--------|------|
| hybrid_rerank (t=0.01) | 0.89 | 0.60 | 0.81 | 0.82 | 934ms |
| hybrid_rerank (t=0.05) | 0.81 | 0.56 | 0.67 | 0.70 | 807ms |
| hybrid_no_rerank | 0.96 | 0.60 | 0.94 | 0.92 | 291ms |
| vector_only | **0.97** | **0.62** | **0.94** | **0.93** | 196ms |

### 13.6 关键发现

1. **Reranker 在 schema 检索场景是负优化**
   - `bge-reranker-v2-m3` 对「自然语言 → DDL schema」匹配给出的绝对分数很低（最相关文档仅 ~0.14）
   - 默认阈值 0.3 过滤掉所有结果；即使降到 0.01，recall@5 仍从 0.94 降至 0.81
   - 延迟增加 4 倍（934ms vs 196ms）
   - **原因**：Cross-Encoder 模型对中文 schema DDL 格式的语义理解不足

2. **纯向量检索效果最佳**
   - `vector_only` 的 MRR=0.97、Recall@5=0.94，均优于所有混合策略
   - 对于「自然语言问题 → 表结构」这种语义匹配场景，dense embedding 足够
   - BM25 关键词匹配未带来额外收益

3. **ES BM25 对 schema 检索无贡献**
   - ES 存储时直接使用 `es.index()` API，metadata 未按 langchain `ElasticsearchStore` 格式写入
   - 检索返回的 Document metadata 为空，导致 doc_id 无法匹配
   - 需修复 ES 存储格式或适配 metadata 读取

4. **Parent Document RAG 需要独立索引**
   - schema 文档仅索引到主集合（`GoAgent`），未索引到 `rag_children` / `rag_parents`
   - 该策略适用于文件类文档，不适用于 schema 检索

### 13.7 优化建议

| 优化项 | 建议 | 预期效果 |
|--------|------|---------|
| Schema 检索默认策略 | 改为 `vector_only`，禁用 reranker | Recall@5 提升至 0.94，延迟降至 ~200ms |
| Reranker 使用场景 | 仅在文件文档检索（非 schema）中启用 | 避免 schema 检索被误伤 |
| ES metadata 修复 | 改用 `ElasticsearchStore.add_documents()` 写入 | 修复 ES 检索的 doc_id 匹配问题 |
| 阈值调优 | 若必须用 reranker，阈值设为 0.01 | 保留更多正确结果 |

### 13.8 文件清单

| 文件 | 说明 |
|------|------|
| `agents/eval/__init__.py` | 模块入口 |
| `agents/eval/dataset_generator.py` | LLM 自动生成评测数据集 |
| `agents/eval/metrics.py` | Recall@K、MRR、NDCG@K 指标计算 |
| `agents/eval/runner.py` | 多策略评估运行器 + 报告生成 |
| `agents/eval/cli.py` | CLI 入口（generate / run / detail） |
| `tests/test_eval_metrics.py` | 指标单元测试（18 个） |
| `scripts/seed_financial.py` | 财务测试数据种子脚本 |
| `eval_dataset.jsonl` | 评测数据集（37 条） |
| `eval_report.json` | 评估报告 JSON |

---

## License

MIT
