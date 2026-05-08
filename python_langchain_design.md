# Agent Platform Python 技术设计文档

本文档描述当前项目的最新实现，重点覆盖 NL2SQL Agent、RAG、LangGraph 编排、Human-in-the-Loop 审批、异常结果反思与 SSE 前端反馈。

相关文档：

- [README](README.md)
- [迭代优化记录](docs/iterations.md)
- [熔断降级与 Fallback 设计](docs/resilience_design.md)

## 1. 项目定位

项目是一个基于 **LangChain + LangGraph + FastAPI** 的财务 Copilot 平台，核心目标是把自然语言问题转成可信的结构化数据查询，并在执行前后加入安全、审批、修正和用户反馈。

当前主要能力：

| 能力 | 当前实现 |
|------|----------|
| NL2SQL | `Final Graph -> SQL React`，支持业务知识增强、语义模型选表、SQL 生成、安全审查、人工审批、执行与反思 |
| RAG 问答 | Milvus 向量 + ES BM25 + RRF + Cross-Encoder rerank |
| 业务知识 | `t_business_knowledge` 存储术语、公式、同义词，支持 MySQL lexical fallback |
| 语义模型 | `t_semantic_model` 统一存储物理字段、业务名、同义词、描述、主外键信息 |
| 审批体验 | LangGraph interrupt 暂停，`/api/query/approve/stream` 用 SSE 展示执行和反思过程 |
| 记忆系统 | Session 历史、摘要、实体/事实/偏好 |
| 稳定性 | 超时、错误分类、可配置重试、SQL 格式化、LangGraph 状态 reducer |

## 2. 分层架构

```text
┌─────────────────────────────────────────────────┐
│ API 层 FastAPI                                   │
│ query / rag / document / admin / static          │
├─────────────────────────────────────────────────┤
│ Flow 编排层 LangGraph                            │
│ Final Graph / SQL React / RAG Chat / Analyst     │
├─────────────────────────────────────────────────┤
│ 能力层                                           │
│ Model Factory / RAG Retriever / Tool Registry    │
├─────────────────────────────────────────────────┤
│ 数据与基础设施层                                  │
│ MySQL / Redis Checkpoint / Milvus / ES / MCP      │
└─────────────────────────────────────────────────┘
```

关键目录：

| 路径 | 说明 |
|------|------|
| `agents/api/routers/query.py` | 查询入口、意图分类、invoke、approve、approve SSE |
| `agents/flow/dispatcher.py` | Final Graph，负责意图分类和子图分发 |
| `agents/flow/sql_react.py` | NL2SQL 主流程 |
| `agents/flow/state.py` | LangGraph 状态定义和 reducer |
| `agents/rag/retriever.py` | 混合检索、业务知识召回、语义模型加载 |
| `agents/model/format_tool.py` | LLM 结构化输出工具和 SQL 格式化校验 |
| `agents/tool/sql_tools/` | MCP SQL 执行、安全检查、错误分类 |
| `agents/static/index.html` | Chat、SQL Agent、审批与 SSE 进度 UI |
| `scripts/seed_business_knowledge.py` | 业务知识种子数据 |

## 3. API 调用设计

### 3.1 查询入口

| Endpoint | 用途 |
|----------|------|
| `POST /api/query/classify` | 只做意图分类和查询重写，返回 `intent + rewritten_query` |
| `POST /api/query/invoke` | 执行主图。若 SQL 需要审批，返回 `pending_approval=true` |
| `POST /api/query/approve` | 非流式审批恢复 |
| `POST /api/query/approve/stream` | 流式审批恢复，向前端推送执行和反思状态 |

`invoke` 初始状态：

```python
{
    "query": req.query,
    "session_id": req.session_id,
    "chat_history": chat_history,
    "intent": req.intent,                 # 可选，前端预分类时传入
    "rewritten_query": req.rewritten_query # 可选，前端预重写时传入
}
```

SQL 审批中断时，API 会把原始 query 暂存到 Session preference 的 `_pending_query`。approve 完成后恢复保存本轮 Q&A。

### 3.2 approve 恢复原则

当前 approve 只通过 `Command(resume=...)` 恢复 interrupt：

```python
Command(resume={
    "approved": req.approved,
    "feedback": req.feedback,
})
```

不再通过 `Command.update` 写回 `query`。原因是父图和 SQL 子图在恢复时可能同一步合并状态，重复写不可聚合字段会触发 LangGraph `INVALID_CONCURRENT_GRAPH_UPDATE`。

## 4. LangGraph 状态设计

### 4.1 FinalGraphState

`FinalGraphState` 负责主调度状态：

```python
class FinalGraphState(TypedDict):
    query: Annotated[str, keep_existing_query]
    session_id: str
    chat_history: list[dict]
    intent: str
    sql: str
    result: str
    answer: str
    status: str
```

### 4.2 SQLReactState

`SQLReactState` 负责 NL2SQL 子图状态：

```python
class SQLReactState(TypedDict):
    query: Annotated[str, keep_existing_query]
    rewritten_query: str
    enhanced_query: str
    evidence: list[str]
    few_shot_examples: list[str]
    selected_tables: list[str]
    table_relationships: list[dict]
    docs: list[Document]
    semantic_model: dict
    sql: str
    is_sql: bool
    approved: bool
    refine_feedback: str
    result: str
    error: str | None
    retry_count: int
    execution_history: list[dict]
    reflection_notice: str
```

### 4.3 query reducer

`query` 是稳定输入字段，父图和子图都可能携带。为了避免 approve/resume 后同一步重复写入报错，`query` 使用 reducer：

```python
def keep_existing_query(current: str | None, incoming: str | None) -> str:
    if current:
        return current
    return incoming or ""
```

该 reducer 只用于保持原始用户问题稳定，不承担查询增强职责。查询增强仍由 `rewritten_query` 和 `enhanced_query` 表达。

## 5. Final Graph 设计

```text
START
  -> classify_intent
  -> route_intent
       -> sql_react
       -> chat_direct
  -> END
```

`classify_intent` 一次 LLM 调用同时完成两件事：

1. 根据数据库领域摘要和用户问题判断 intent。
2. 根据对话历史把省略/指代问题重写成独立问题。

当请求已传入 `intent + rewritten_query` 时，节点跳过 LLM，避免前后端重复分类。

当前路由策略：

| intent | 目标 |
|--------|------|
| `sql_query` | SQL React |
| 其他意图 | RAG Chat / chat_direct |

## 6. SQL React 设计

### 6.1 主流程

```text
contextualize_query
  -> recall_evidence
  -> query_enhance
  -> select_tables
  -> sql_retrieve
  -> check_docs
  -> sql_generate
  -> safety_check
  -> approve
  -> execute_sql
  -> END
```

异常分支：

```text
execute_sql -- retryable error --> error_analysis -> sql_generate
execute_sql -- suspicious result --> result_reflection -> safety_check -> approve -> execute_sql
```

### 6.2 节点职责

| 节点 | 说明 |
|------|------|
| `contextualize_query` | 若外层已有 `rewritten_query` 则复用，否则结合历史做指代消解 |
| `recall_evidence` | 并行召回业务知识和 SQL few-shot 示例 |
| `query_enhance` | 基于召回 evidence 翻译业务术语，LLM 空响应时使用 evidence 驱动 fallback |
| `select_tables` | 从 `t_semantic_model` 加载表名和描述，让 LLM 精选相关表 |
| `sql_retrieve` | 根据表名加载完整语义模型并构建 schema docs |
| `check_docs` | 无 schema 时直接返回用户友好错误 |
| `sql_generate` | 用 LLM tool 生成 SQL，支持最多 3 轮自动补表 |
| `safety_check` | 只允许安全 SELECT/WITH 查询继续审批 |
| `approve` | LangGraph interrupt，等待用户确认 |
| `execute_sql` | 通过 MCP MySQL 执行 SQL |
| `error_analysis` | 对可重试执行错误生成修正反馈，再回到 `sql_generate` |
| `result_reflection` | 对空集、NULL 等异常成功结果直接生成修正 SQL |

### 6.3 query_enhance 设计

`query_enhance` 不做 query-specific 硬编码。它只使用召回到的业务知识：

```text
术语: 净利润
公式: 收入 - 成本 - 费用；亏损表示净利润 < 0
同义词: 净收益, 盈利, 亏损, 净亏损, 赔钱, 赚钱
```

增强策略：

1. 优先让 LLM 根据 evidence 输出增强后的 query。
2. 如果 LLM 返回空或失败，使用 `_heuristic_enhance_query()` 从 evidence 中解析 `术语/公式/同义词` 做确定性增强。
3. 如果没有 evidence，保持原 query，不臆造业务口径。

业务词新增方式是维护 `t_business_knowledge.term/synonyms/formula`，不是改代码。

### 6.4 SQL 输出格式化

LLM 通过 `sql_format_response` tool 返回：

```python
{
    "answer": "SELECT ...",
    "is_sql": True,
    "needs_more_tables": False,
    "missing_tables": []
}
```

本地 `normalize_sql_answer()` 会再次清洗和校验：

- 去除 `<text_never_used_...>` 这类异常 token。
- 去除 Markdown SQL fence。
- 截取解释文本后的 `SELECT/WITH`。
- 拦截尾部截断关键字，例如 `HAVIN`、`WHERE`、`AND`。
- 拦截括号不匹配。
- 拦截非 `SELECT/WITH` 开头的 SQL。
- 统一单个结尾分号。

`sql_generate` 和 `result_reflection` 都必须经过该格式化器，不能只依赖 LLM tool schema。

### 6.5 执行结果反思

SQL 执行成功不等于结果可信。`_result_anomaly_reason()` 会识别：

- 裸 `[]`
- `{"rows": []}`、`{"data": []}`、`{"result": []}`、`{"items": []}`
- `{"columns": [...], "rows": []}`
- `null` / `None`
- 结构化结果中所有字段均为 `NULL` 或空字符串
- 原始字符串中出现 `null` 或 `[]` 的可疑信号

命中异常且未超过 `max_sql_retries` 时进入：

```text
execute_sql -> result_reflection -> safety_check -> approve -> execute_sql
```

`result_reflection` 的职责是直接输出修正后的 SQL，不再把建议交回 `sql_generate` 二次生成。

## 7. Human-in-the-Loop 与 SSE

### 7.1 初次审批

`approve` 节点 interrupt payload：

```python
{
    "sql": state["sql"],
    "message": "请确认是否执行以上 SQL？",
    "reflection": False,
}
```

前端收到 `pending_approval=true` 后展示 SQL 审批卡片。

### 7.2 反思后审批

如果 SQL 是 `result_reflection` 生成的修正 SQL，payload 会包含：

```python
{
    "message": "上次执行结果疑似异常，系统已反思并生成修正后的 SQL。请确认是否执行修正后的 SQL？",
    "reflection": True,
}
```

### 7.3 SSE 进度

`/api/query/approve/stream` 会推送：

```text
status: 已确认，正在执行 SQL...
status: 如果执行结果异常，系统会自动反思并生成修正 SQL...
status: 检测到执行结果异常，已完成反思并生成修正 SQL。
status: 请确认是否执行修正后的 SQL。
result: QueryResponse JSON
done: [DONE]
```

这样用户能理解“为什么又需要确认一条 SQL”，不会误以为系统重复审批同一条 SQL。

## 8. RAG 设计

RAG 检索管线：

```text
Query
  -> Milvus vector search
  -> Elasticsearch BM25 search
  -> RRF fusion
  -> Cross-Encoder rerank
  -> Top-K docs
```

RAG Chat 流程：

```text
preprocess -> rewrite -> retrieve -> construct_messages -> chat -> END
```

支持传统 RAG 和 Parent Document RAG。查询重写使用 Session 历史和摘要做指代消解。

## 9. 业务知识与语义模型

### 9.1 t_business_knowledge

用于表达业务口径：

| 字段 | 作用 |
|------|------|
| `term` | 业务术语，例如净利润 |
| `definition/formula` | 定义和计算口径 |
| `synonyms` | 同义词和口语表达，例如亏损、净亏损 |
| `related_tables` | 相关表提示 |

召回顺序：

1. Milvus 向量召回。
2. ES BM25 召回。
3. RRF 融合和过滤。
4. MySQL `term/synonyms` lexical fallback，补齐口语化查询。

### 9.2 t_semantic_model

用于 NL2SQL schema 理解：

| 信息 | 用途 |
|------|------|
| 表名、表描述 | `select_tables` 候选表选择 |
| 字段名、类型、注释 | SQL 物理字段生成 |
| 业务名、同义词、描述 | 业务含义理解 |
| PK/FK、逻辑外键 | JOIN 关系提示 |

`sql_retrieve` 从该表构建 schema docs，不再依赖 Milvus schema 索引。

## 10. 稳定性设计

| 风险 | 当前处理 |
|------|----------|
| LLM 空响应 | query_enhance 使用 evidence fallback |
| SQL 格式异常 | `normalize_sql_answer()` 清洗和拦截 |
| SQL 危险操作 | `SQLSafetyChecker` 拦截 |
| SQL 执行失败 | `is_retryable()` 决定是否 `error_analysis -> sql_generate` |
| SQL 执行成功但结果异常 | `_result_anomaly_reason()` 触发 `result_reflection` |
| approve 后 query 丢失 | Session `_pending_query` 用于最终 Q&A 保存 |
| approve 后并发写 query | `query: Annotated[..., keep_existing_query]` |
| 检索/LLM 慢或失败 | resilience 配置的 timeout 和 retry |

## 11. 时序图

```mermaid
sequenceDiagram
    participant U as 用户/前端
    participant API as Query API
    participant G as Final Graph
    participant S as SQL React
    participant LLM as LLM
    participant DB as MCP MySQL

    U->>API: POST /api/query/classify
    API->>G: classify_intent(query, history)
    G-->>API: intent + rewritten_query

    U->>API: POST /api/query/invoke
    API->>G: initial_state
    G->>S: SQL React subgraph
    S->>S: contextualize_query
    S->>S: recall_evidence + query_enhance
    S->>S: select_tables + sql_retrieve
    S->>LLM: sql_generate
    LLM-->>S: tool call SQL
    S->>S: normalize_sql_answer + safety_check
    S-->>API: interrupt(sql)
    API-->>U: pending_approval

    U->>API: POST /api/query/approve/stream
    API-->>U: SSE status 执行中
    API->>G: Command(resume)
    G->>S: approve -> execute_sql
    S->>DB: execute SQL
    DB-->>S: result

    alt result normal
        S-->>API: completed answer
        API-->>U: SSE result completed
    else result suspicious
        S->>LLM: result_reflection
        LLM-->>S: corrected SQL
        S->>S: normalize_sql_answer + safety_check
        S-->>API: interrupt(corrected sql)
        API-->>U: SSE result pending_approval
    end
```

## 12. 测试覆盖

当前关键测试：

| 文件 | 覆盖 |
|------|------|
| `tests/test_sql_react.py` | query_enhance、SQL 格式化、结果异常检测、result_reflection、图节点、query reducer |
| `tests/test_final_api.py` | query invoke、approve resume、approve SSE |
| `tests/test_rag_e2e.py` | RAG 端到端流程 |
| `tests/test_api.py` | API 基础行为 |

推荐变更后至少执行：

```bash
pytest tests/test_final_api.py tests/test_sql_react.py -q
```

