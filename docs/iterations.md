# 迭代优化记录

## 迭代 1：Schema 召回策略优化（参考 DataAgent）

### 为什么优化

对比阿里 DataAgent 项目，发现我们的 schema 召回策略存在根本性问题：

| 对比项 | DataAgent | 我们（优化前） |
|--------|-----------|----------------|
| Schema 召回方式 | LLM 选表 → metadata 精确过滤 | 向量相似度检索（模糊匹配） |
| 语义模型 | 业务名称 + 同义词 + 业务描述 | 只有 information_schema 原始字段 |
| 列信息丰富度 | 表/列分开存储，列有 samples、foreignKey | 扁平文本，一个表一个 doc |

**核心问题**：表名和列名是确定性的，不需要"语义模糊匹配"。用户问"查询各部门费用"，向量检索可能召回不相关的表，而 LLM 直接看表名列表就能判断需要用 `t_cost_center` 和 `t_journal_item`。

### 优化了什么

将 schema 召回从"向量相似度检索"改为"LLM 选表 + metadata 过滤精确拉取"两步走：

**Step 1: 表名列表召回**（轻量级）
- 启动时从 MySQL 拉取所有表名，缓存为表名列表
- 用户提问时，LLM 从表名列表中选择相关的表

**Step 2: Schema 精确拉取**（metadata 过滤）
- 按选中的表名，从 Milvus 用 metadata 过滤精确拉取 schema 文档
- 不依赖向量相似度，100% 准确

### 怎么优化的

#### 新增节点：`select_tables`

```
用户问题 + 表名列表 → LLM 判断需要哪些表 → 返回 table_names
```

- 从 Milvus 中获取所有 schema 文档的 table_name（去重）
- LLM 看到表名列表 + 用户问题，输出需要的表名
- 比向量检索更准确：LLM 理解"费用"对应 `t_expense_claim`，而向量检索可能匹配不到

#### 改造 `sql_retrieve`

```
优化前: query → vector similarity search → docs
优化后: query → select_tables (LLM) → metadata filter by table_name → docs
```

向量检索保留为 fallback：当 LLM 无法判断需要哪些表时（如模糊查询），回退到向量检索。

### 提升预期

| 指标 | 优化前（向量检索） | 优化后（LLM 选表 + 精确过滤） |
|------|-------------------|-------------------------------|
| 召回准确率 | 依赖 embedding 质量 | LLM 直接判断，理论上接近 100% |
| 延迟 | 向量检索 ~200ms | LLM 调用 ~500ms（多一次调用） |
| 可解释性 | 低（黑盒相似度） | 高（LLM 给出选表理由） |

**取舍**：用一次额外的 LLM 调用换取更高的召回准确率。对于 SQL 场景，错误的 schema 会导致生成错误 SQL，准确率比延迟更重要。

> **后续优化**：迭代 5 进一步优化为"向量粗筛 top-10 + LLM 精选"两阶段方案，避免全量表名发 LLM 的 token 浪费。

---

## 迭代 2：语义模型（字段级业务映射）

### 为什么优化

现在 schema 文档只包含 information_schema 的原始信息（字段名、类型、COMMENT）。用户说"查记账金额"，LLM 看到的是 `amount decimal(18,2)`，无法确定哪个字段对应"记账金额"。

DataAgent 的语义模型为每个字段维护：
- `business_name`：业务名称（如"记账金额"）
- `synonyms`：同义词（如"交易金额, 发生额"）
- `business_description`：业务描述（解释枚举值、状态码等）

这些信息注入 SQL 生成 prompt 后，LLM 能准确映射业务语言到物理字段。

### 优化了什么

1. MySQL 新建 `t_semantic_model` 表，存储字段级业务映射
2. Admin API 支持 CRUD 语义模型配置
3. `schema_indexer` 索引时 JOIN 语义模型，丰富 schema 文档内容
4. `sql_generate` prompt 自动带上增强后的 schema（含业务名称和同义词）

### 怎么优化的

#### 数据模型

```sql
CREATE TABLE t_semantic_model (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    table_name VARCHAR(128) NOT NULL,
    column_name VARCHAR(128) NOT NULL,
    business_name VARCHAR(256) COMMENT '业务名称',
    synonyms TEXT COMMENT '同义词，逗号分隔',
    business_description TEXT COMMENT '业务描述',
    UNIQUE KEY uk_table_col (table_name, column_name)
);
```

#### Schema 文档增强

索引时，对每个字段查找语义模型，丰富 page_content：

```
优化前:
  表名: t_journal_item
  字段:
    amount decimal(18,2) COMMENT '金额'

优化后:
  表名: t_journal_item
  字段:
    amount decimal(18,2) -- 记账金额
      同义词: 交易金额, 发生额, 借贷金额
      描述: 凭证行的借方或贷方金额，正值表示借方，负值表示贷方
```

### 提升预期

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| "查记账金额" | LLM 可能猜错字段 | 直接映射到 amount |
| "交易金额是多少" | 需要 LLM 理解 COMMENT | 同义词直接命中 |
| 枚举值查询 | LLM 不知道 status=1 含义 | 业务描述解释清楚 |

---

## 迭代 3：业务知识配置 ✅

### 为什么优化

用户问"毛利率是多少"，LLM 不知道"毛利率 = (收入 - 成本) / 收入 * 100"，也无法知道这个公式关联 `t_journal_item` 和 `t_account` 表。业务知识是"不存在于数据库 schema 中的计算逻辑和领域定义"。

DataAgent 的 BusinessKnowledge 模块存储业务术语 + 公式 + 同义词，向量检索后注入 SQL 生成 prompt。

### 优化了什么

1. MySQL 新建 `t_business_knowledge` 表，存储业务术语、公式、同义词
2. 向量化存入 Milvus（metadata.source = "business_knowledge"）
3. `sql_react` 图新增 `recall_evidence` 节点，向量检索业务知识
4. 检索结果注入 `sql_generate` prompt
5. Admin API 新增业务知识 CRUD（GET/POST/DELETE/batch/reindex）

### 怎么优化的

#### 数据模型

```sql
CREATE TABLE t_business_knowledge (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    term VARCHAR(128) NOT NULL COMMENT '业务术语',
    formula TEXT NOT NULL COMMENT '公式/定义',
    synonyms TEXT COMMENT '同义词，逗号分隔',
    related_tables TEXT COMMENT '关联表名，逗号分隔',
    UNIQUE KEY uk_term (term)
);
```

#### 图流程变更

```
优化前: START → load_table_names → select_tables → sql_retrieve → ...
优化后: START → load_table_names → select_tables → recall_evidence → sql_retrieve → ...
```

#### 消费路径

```
recall_evidence:
  用户问题 → 向量检索 t_business_knowledge (score > 0.3) → 匹配的业务知识
  ↓
sql_generate:
  prompt += "业务知识:\n毛利率 = (收入-成本)/收入*100\n预算执行率 = 实际/预算*100"
```

### 提升预期

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| "毛利率是多少" | LLM 不知道公式，无法生成 SQL | 注入公式 + 关联表，直接生成 |
| "预算执行率" | LLM 可能误解为普通字段查询 | 注入"实际/预算*100"公式 |
| 术语同义词匹配 | 无法识别 | 向量检索匹配同义词 |

## 迭代 4：SQL 领域智能体知识库 ✅

### 为什么优化

用户问"查询各部门费用汇总"，LLM 需要从零开始构造 SQL，可能遗漏 JOIN 条件、GROUP BY 逻辑。如果有相似问题的 SQL 示例（few-shot），LLM 可以参考模式生成更准确的 SQL。

DataAgent 的 AgentKnowledge 模块存储 Q&A 对，向量检索后注入 prompt 作为 few-shot 示例。

### 优化了什么

1. MySQL 新建 `t_agent_knowledge` 表，存储问题、SQL、说明、分类
2. 向量化存入 Milvus（metadata.source = "agent_knowledge"）
3. `recall_evidence` 节点同时检索业务知识 + 智能体知识库
4. 检索结果作为 few-shot 示例注入 `sql_generate` prompt
5. Admin API 新增智能体知识库 CRUD（GET/POST/DELETE/batch/reindex）
6. 种子数据：12 个常见财务 SQL Q&A 对

### 怎么优化的

#### 数据模型

```sql
CREATE TABLE t_agent_knowledge (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    question TEXT NOT NULL COMMENT '用户问题',
    sql_text TEXT NOT NULL COMMENT '参考 SQL',
    description TEXT COMMENT '说明',
    category VARCHAR(64) COMMENT '分类: query/report/analysis',
    UNIQUE KEY uk_question (question(128))
);
```

#### 图流程

```
recall_evidence:
  用户问题 → 向量检索 business_knowledge (score > 0.3) → 业务知识
  用户问题 → 向量检索 agent_knowledge (score > 0.3) → SQL Q&A few-shot
  ↓
sql_generate:
  prompt += "业务知识:\n毛利率 = ..."
  prompt += "相似问题参考:\n问题: 查询所有科目余额\nSQL: SELECT ..."
```

### 提升预期

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| "各部门费用汇总" | LLM 从零构造 SQL | 参考相似 Q&A，JOIN 逻辑更准确 |
| "预算执行情况" | LLM 可能忘记计算公式 | 注入 few-shot + 业务知识双保险 |
| 复杂多表查询 | LLM 容易遗漏关联条件 | 参考已有示例模式 |

---

## 迭代 5：表选择两阶段优化（向量粗筛 + LLM 精选）✅

### 为什么优化

迭代 1 把表选择从"向量检索"改为"LLM 选表"，但问题是把**所有表名**都发给 LLM。表少时没问题，表多（50+）时浪费 token。

对比 DataAgent 的方案：先向量检索 top-10 候选表，再让 LLM 从候选中精选。两阶段组合兼顾效率和准确率。

### 优化了什么

1. `select_tables` 节点改为两阶段：向量粗筛 → LLM 精选
2. 新增 `search_schema_tables` 函数：向量检索 schema 文档，返回 top-K 候选表名
3. 候选 ≤ 3 个时直接使用，省一次 LLM 调用
4. 向量检索失败时 fallback 到全量表名 + LLM 选表
5. 移除 `load_table_names` 独立节点（合并到 `select_tables` 内部）

### 怎么优化的

#### 图流程变更

```
优化前: START → load_table_names（全量） → select_tables（LLM 从全部选） → ...
优化后: START → select_tables（向量 top-10 → LLM 精选） → recall_evidence → ...
```

#### select_tables 逻辑

```python
# Stage 1: 向量粗筛
candidate_tables = search_schema_tables(query, top_k=10)

# Stage 2: 候选少，直接用
if len(candidate_tables) <= 3:
    return candidate_tables

# Stage 2: 候选多，LLM 精选
response = llm.invoke(f"从 {candidate_tables} 中选出需要的表")
```

### 提升预期

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| 50+ 表库 | 全量表名发 LLM（浪费 token） | 向量 top-10 → LLM 只看 10 个 |
| 3 个表命中 | LLM 调用（不必要） | 直接使用，省一次 LLM |
| 向量检索失败 | 报错 | fallback 到全量 + LLM |

### 待优化（后续迭代）

- **外键扩展**：向量粗筛后，自动补全外键关联的缺失表（需外键元数据）

---

## 迭代 6：上下文记忆系统 ✅

### 为什么优化

用户先问"zhangsan01是谁"（SQL 正确返回），接着问"他在哪个部门"时，系统无法解析"他"指的是 zhangsan01，生成的 SQL 缺少真实用户名。

**根因**：`FinalGraphState` 和 `SQLReactState` 没有 `chat_history` 字段，API 端点不传对话历史，SQL 生成流水线每次都是独立执行，没有上下文。

### 优化了什么

1. `FinalGraphState` 和 `SQLReactState` 新增 `chat_history` 和 `rewritten_query` 字段
2. API 端点（classify/invoke/approve）从 session store 加载对话历史，注入 graph state
3. SQL React 图新增 `contextualize_query` 入口节点，调用 `rewrite_query` 将代词化查询重写为独立查询
4. 意图分类（`classify_intent`）注入最近 3 轮对话历史，帮助理解代词
5. 下游节点（`select_tables`/`recall_evidence`/`sql_generate`）使用重写后的查询
6. SQL 审批中断时暂存原始 query，approve 后正确恢复并保存 Q&A

### 怎么优化的

#### 状态变更

```python
class FinalGraphState(TypedDict):
    query: str
    session_id: str
    chat_history: list[dict]     # 新增：对话历史
    intent: str
    ...

class SQLReactState(TypedDict):
    query: str
    rewritten_query: str         # 新增：上下文化后的独立问题
    chat_history: list[dict]     # 新增：对话历史
    ...
```

#### 图流程变更

```
优化前: START → select_tables → recall_evidence → sql_retrieve → ...
优化后: START → contextualize_query → select_tables → recall_evidence → sql_retrieve → ...
```

#### contextualize_query 逻辑

```python
async def contextualize_query(state):
    chat_history = state.get("chat_history", [])
    if not chat_history:
        return {"rewritten_query": state["query"]}  # 无历史，原样返回

    rewritten = await rewrite_query(
        summary=summary,
        history=chat_history[-6:],  # 最近 3 轮
        query=state["query"],
    )
    return {"rewritten_query": rewritten}
```

#### Session 持久化流程

```
invoke 端点:
  1. _load_chat_history(session_id) → 从 session store 加载历史
  2. graph.ainvoke({query, chat_history})
  3. _save_qa_to_session(session_id, query, answer) → 保存本轮 Q&A

approve 端点:
  1. invoke 中断时: _save_pending_query(session_id, query) → 暂存 query
  2. approve 恢复时: _pop_pending_query(session_id) → 取出原始 query
  3. _save_qa_to_session(session_id, original_query, answer) → 保存完整 Q&A
```

### Bug 修复

| Bug | 原因 | 修复 |
|-----|------|------|
| Q&A 未保存到 session | `compress_session` 是 async 函数，但在同步函数中直接调用未 await | 移除 compress_session 调用（历史 < 20 条不需要压缩） |
| approve 后 query 为空 | graph 恢复后 state 中 query 字段可能丢失 | 中断时暂存 query 到 session preferences，approve 后恢复 |
| Redis 未启动导致服务不可用 | `init_redis()` 连接失败直接 raise，阻断服务启动 | 改为 warning 日志，允许无 Redis 环境启动（session store 有内存 fallback） |

### 提升预期

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| "zhangsan01是谁" → "他在哪个部门" | "他"无法解析，SQL 缺少条件 | 重写为"zhangsan01在哪个部门" |
| "查一下这个月的费用" → "按部门分呢" | "按部门分"无法理解上下文 | 重写为"这个月的费用按部门分组" |
| 意图分类带代词 | "他"可能被分类为 chat | 注入历史后正确分类为 sql_query |

---

## 迭代 7：熔断降级与超时保护 ✅

### 为什么优化

项目依赖 Milvus、MySQL、Redis、LLM API、Elasticsearch 5 个外部服务。分析发现 Milvus 直接查询（5 个函数）、LLM API（8+ 调用点）、MySQL MCP 三个关键路径**无超时**，一旦下游挂起，整个请求链阻塞。

select_tables 节点在 Milvus 不可用时无限阻塞，是最先暴露的问题。

### 优化了什么

**Phase 1：超时保护（本次实施）**

| 文件 | 改动 |
|------|------|
| `agents/rag/retriever.py` | 5 个 Milvus 直接查询函数加 try/except，失败返回空 |
| `agents/flow/sql_react.py` | select_tables/recall_evidence/sql_retrieve/contextualize_query 加 `asyncio.wait_for` 超时 |
| `agents/model/providers/*.py` | 所有 5 个 provider 加 `request_timeout=60, max_retries=2` |
| `agents/tool/sql_tools/mcp_client.py` | execute_sql 加 `asyncio.wait_for(timeout=15)` |
| `agents/flow/rag_chat.py` | rewrite/chat/compress_and_save 加超时 |
| `agents/tool/storage/redis_client.py` | init_redis 失败不再 raise，允许无 Redis 启动 |

**Phase 2：Fallback 降级（已内置）**

| 组件 | 降级策略 |
|------|----------|
| Milvus 向量检索超时 | 返回空列表，跳过该路召回 |
| LLM 重写超时 | 使用原始 query |
| LLM 选表超时 | 使用向量检索结果 |
| MySQL 执行超时 | 进入 error_analysis 重试 |
| Redis 不可用 | session store 使用内存 dict |
| Redis checkpointer 不可用 | 使用 MemorySaver |

### 超时配置汇总

| 调用 | 超时 | 降级行为 |
|------|------|----------|
| Milvus 向量检索 | 8s | 返回空 |
| Milvus metadata 查询 | 10s | 返回空 |
| LLM request_timeout | 60s | 自动重试 2 次 |
| LLM rewrite/compress | 15s/30s | 使用原始值 |
| MySQL MCP execute_sql | 15s | 返回错误 |
| Milvus HybridRetriever | 30s 外层 | 返回空 |

### DataAgent 对比

对比本地 DataAgent 项目（`/Users/a0000/project/DataAgent`），熔断降级方面：

| 能力 | DataAgent | 我们 | 状态 |
|------|-----------|------|------|
| SQL 执行重试 | LLM 引导，最多 10 次 | LLM 引导，最多 5 次（可配置） | ✅ 已有 |
| DB 连接重试 | 3 次 + 线性退避 | MCP 长连接，无重试 | ⚠️ 后续补齐 |
| SQL 执行超时 | 30s | 15s（可配置） | ✅ 已有 |
| LLM 超时 | 仅图表生成 3s | 所有调用点 15~60s（可配置） | ✅ 超越 |
| LLM 重试 | ❌ 无 | max_retries=2 | ✅ 超越 |
| 向量检索降级 | catch → 返回空 | catch → 返回空 | ✅ 已有 |
| 知识入库降级 | mark FAILED | log only | ⚠️ 后续补齐 |
| 错误码映射 | 20+ SQLState | 16 种 SQLState + is_retryable() | ✅ 已有 |
| 可配置重试次数 | 配置文件 | ResilienceSettings（环境变量） | ✅ 已有 |
| 熔断器 | ❌ 无 | ❌ 无 | 双方都无 |

**核心结论**：DataAgent 也没有熔断器，其容错依赖"超时 + LLM 引导重试 + 节点级降级"三板斧。我们已在这三方面达到或超越 DataAgent。错误码映射和可配置重试已补齐。

### 详细设计

见 `docs/resilience_design.md`，包含 DataAgent 对比分析、三层方案、错误码映射设计。

---

## 迭代 8：错误码分类 + 可配置重试 ✅

### 为什么优化

迭代 7 实现了超时保护和基础重试，但存在两个问题：

1. **重试不区分错误类型**：语法错误（表不存在、列不存在）和连接错误（连接中断）同样重试，浪费 LLM 调用
2. **重试次数硬编码**：`_MAX_RETRIES = 3` 无法通过配置调整，需要改代码

DataAgent 有 16 种 SQLState 映射（`ErrorCodeEnum`）和可配置重试次数（`DataAgentProperties`），我们需要补齐。

### 优化了什么

1. 新建 `agents/tool/sql_tools/error_codes.py`，定义 16 种 SQLState 错误码分类 + `is_retryable()` 函数
2. 新增 `ResilienceSettings` 到 `agents/config/settings.py`，支持环境变量配置重试次数和超时
3. `sql_react.py` 的 `route_after_execute` 使用 `is_retryable()` 判断是否重试
4. 所有超时值改为从 `settings.resilience` 读取，不再硬编码

### 怎么优化的

#### 错误码分类

```python
# agents/tool/sql_tools/error_codes.py
SQL_ERROR_CODES = {
    "08001": ("连接建立失败", True),    # 可重试
    "08S01": ("连接中断", True),        # 可重试
    "28P01": ("密码错误", False),       # 不可重试
    "42S02": ("表不存在", False),       # 不可重试
    "42S22": ("列不存在", False),       # 不可重试
    # ... 共 16 种
}

def is_retryable(error_msg: str) -> bool:
    """连接类错误重试，语法/权限类不重试。未知错误默认不重试。"""
```

#### 可配置超时

```python
# agents/config/settings.py
class ResilienceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RESILIENCE_")
    max_sql_retries: int = 5            # RESILIENCE_MAX_SQL_RETRIES
    sql_execution_timeout: float = 15   # RESILIENCE_SQL_EXECUTION_TIMEOUT
    milvus_timeout: float = 8           # RESILIENCE_MILVUS_TIMEOUT
    llm_timeout: float = 60             # RESILIENCE_LLM_TIMEOUT
    llm_rewrite_timeout: float = 15     # RESILIENCE_LLM_REWRITE_TIMEOUT
```

#### 条件路由

```python
def route_after_execute(state):
    if not state.get("error"):
        return END
    if not is_retryable(state["error"]):  # 语法/权限错误不重试
        return END
    if state.get("retry_count", 0) < settings.resilience.max_sql_retries:
        return "error_analysis"
    return END
```

### 提升预期

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| "查不存在的表" | 重试 3 次 LLM（浪费 token） | 直接结束，返回"表不存在" |
| "密码错误" | 重试 3 次（无意义） | 直接结束，返回认证错误 |
| 连接超时 | 重试 3 次（正确） | 重试 5 次（可配置） |
| 生产环境调优 | 改代码重新部署 | 改环境变量重启 |

### 涉及文件

| 文件 | 改动 |
|------|------|
| 新建 `agents/tool/sql_tools/error_codes.py` | 16 种 SQLState + `is_retryable()` |
| `agents/config/settings.py` | 新增 `ResilienceSettings`，5 个可配置参数 |
| `agents/flow/sql_react.py` | `route_after_execute` 使用 `is_retryable`，超时从配置读取 |
| `agents/flow/rag_chat.py` | 超时从 `settings.resilience` 读取 |
| `agents/tool/sql_tools/mcp_client.py` | SQL 执行超时从配置读取 |

---

## Bug 修复记录

### Bug 1：ConnectionNotExistException 文件上传失败

**出现什么问题**：文件上传到 Milvus 时报 `ConnectionNotExistException`，langchain-milvus 内部调用 `Collection(alias=...)` 找不到连接。

**什么原因**：pymilvus 2.6.x 的 `MilvusClient` 不再自动注册到全局 `pymilvus.connections` 注册表。langchain-milvus 0.3.x 内部使用 `Collection(alias=...)` 时依赖全局注册表，导致找不到连接句柄。

**怎么解决的**：在 `agents/rag/retriever.py` 中实现 `_patch_milvus_connections()` 猴子补丁，拦截 `MilvusClient.__init__`，在创建后自动调用 `connections.add_connection()` 注册到全局注册表。补丁在应用启动时 (`app.py` 的 `_init_milvus()`) 执行一次。

### Bug 2：DataNotMatchException Insert missed field table_name

**出现什么问题**：用户文档上传时报 `DataNotMatchException: Insert missed field table_name`，Goagent2 集合的 `table_name` 字段为 NOT NULL。

**什么原因**：旧集合 Goagent2 的 schema 定义了 `table_name` 为非空字段，但用户上传的文档（PDF/TXT）没有这个字段。schema 文档有 `table_name`，用户文档没有。

**怎么解决的**：在 `agents/rag/indexing.py` 中，对所有非 schema 文档设置 `chunk.metadata["table_name"] = ""`。同时统一使用 `knowledge_base` 集合，通过 `source` 字段区分文档来源（mysql_schema / user_document / business_knowledge / agent_knowledge）。

### Bug 3：MCP MySQL Server 只读，无法写入 domain_summary

**出现什么问题**：`domain_summary` 表创建和写入操作静默失败，启动时无法保存领域摘要。

**什么原因**：MCP MySQL Server 配置为只读模式（`--read-only`），所有 DDL/DML 操作都被拒绝。之前所有数据库操作都通过 MCP，写操作无法执行。

**怎么解决的**：将 `agents/tool/storage/domain_summary.py` 中所有写操作改为 pymysql 直连（`ensure_domain_summary_table`、`save_domain_summary`），读操作保持通过 MCP。

### Bug 4：启动时 schema 索引被错误跳过

**出现什么问题**：启动后 Milvus 中没有 schema 文档，但 `_index_schemas_background` 认为已有数据而跳过索引。

**什么原因**：使用 `get_collection_stats().row_count` 判断是否有数据，但 Milvus 删除数据后 `row_count` 不立即更新（compaction 前显示旧值）。`domain_summary` 表中也有旧数据，导致双重误判。

**怎么解决的**：改为直接查询 `source == "mysql_schema"` 的文档是否存在，而不是依赖 `row_count`。同时检查 `domain_summary` 表中的摘要是否真正存在。

### Bug 5：Qwen Embedding 批量大小超限

**出现什么问题**：上传 PDF 文件时报 `Error code: 400 - batch size is invalid, it should not be larger than 10`。

**什么原因**：Qwen Embedding API 限制每次请求最多处理 10 个文本。langchain_milvus 的 `add_documents()` 内部调用 `embed_documents()` 时默认批量大小为 32 或更大，超出 API 限制。

**怎么解决的**：在所有 4 个 `_get_embeddings()` 函数中（`indexing.py`、`retriever.py`、`schema_indexer.py`、`parent_retriever.py`），为 Qwen provider 添加 `chunk_size=10` 参数传递给 `OpenAIEmbeddings`，强制限制批量大小。

---

## 迭代 9：RAG 知识体系重构 ✅

### 为什么优化

1. 意图识别做了两次（API `/classify` + Graph `classify_intent` 节点），浪费 LLM 调用
2. `final_graph.py` 命名不清晰，实际是意图调度器
3. 意图提示词中 `sql_query` 描述硬编码，不随数据库变化
4. 文档入库无 LLM 预处理，直接切块向量化，信息密度低
5. `recall_evidence` 串行检索，`select_tables` 输出冗余 `table_names`

### 优化了什么

#### 9.1 意图去重 + 文件重命名
- `final_graph.py` → `dispatcher.py`，`final.py` → `query.py`
- `classify_intent` 检测到 state 中已有 intent 时跳过 LLM 调用
- 前端 invoke 请求携带 intent 参数，Graph 直接路由
- API 路径 `/api/final/*` → `/api/query/*`

#### 9.2 动态意图提示词
- `sql_query` 描述从硬编码改为引用 `domain_summary`
- 数据库表结构变化时，意图分类自动适应

#### 9.3 LLM 文档预处理
- 新建 `agents/rag/doc_preprocessor.py`：DocumentPreprocessor 类
- 预处理流程：提取元数据（分类、标签、实体）→ 生成摘要 → 假设性问题 → 关键事实
- 新建 `t_document_metadata` MySQL 表存储元数据
- 每个 chunk 的 page_content 组合：`[摘要] + [原文] + [相关问题]`

#### 9.4 user_document 父子分块 + session_id 隔离
- 长文本（>3000 字）自动使用父子分块，短文本用普通分块
- Milvus schema 新增 `session_id` 字段
- 检索时按 `session_id` 过滤，用户文档互相隔离
- `get_hybrid_retriever` 对 session-scoped 检索单独创建实例（不走单例）

#### 9.5 recall_evidence 并行化
- `recall_business_knowledge` 和 `recall_agent_knowledge` 改为 `asyncio.gather` 并行
- 耗时从 sum(两个) 降为 max(单个)

#### 9.6 select_tables 精简
- 移除 `select_tables` 对 `table_names` 的输出，不再覆盖 state 中的全量表名

### 涉及文件

| 文件 | 改动 |
|------|------|
| `agents/flow/final_graph.py` → `dispatcher.py` | 重命名 + 跳过逻辑 + 动态 prompt |
| `agents/api/routers/final.py` → `query.py` | 重命名 + intent 参数 |
| `agents/rag/doc_preprocessor.py` | 新建：LLM 文档预处理 |
| `agents/tool/storage/doc_metadata.py` | 新建：MySQL 元数据 CRUD |
| `agents/rag/indexing.py` | 集成预处理 + 父子分块阈值判断 |
| `agents/api/routers/document.py` | session_id 参数 + await 异步 |
| `agents/flow/sql_react.py` | select_tables 精简 + recall_evidence 并行 |
| `agents/flow/rag_chat.py` | session_id 检索过滤 |
| `agents/rag/retriever.py` | session_id_filter 支持 |
| `agents/api/app.py` | 路由注册 + schema 加 session_id + doc_metadata 表初始化 |
| `agents/static/index.html` | API 路径更新 + invoke 带 intent |
| `tests/test_final_api.py` | 适配新模块名 |
| `tests/test_imports.py` | 适配新模块名 |

---

## 迭代 10：意图识别 + 上下文重写合并 ✅

### 为什么优化

原有流程需要 **3 次 LLM 调用**：

```
/api/query/classify → LLM 1: 意图分类
/api/query/invoke   → LLM 2: 上下文重写（contextualize_query）
                    → LLM 3: SQL 生成（sql_generate）
```

问题：
1. 意图分类和上下文重写是独立的 LLM 调用，浪费 token
2. 重写后的查询可能改变意图（如"一季度营收多少"→"贵州茅台2026年第一季度营收多少"），但重写在分类之后，意图已经定了
3. 意图 prompt 中硬编码了示例（如 sql_query 的示例），导致 LLM 倾向于返回特定意图，而非根据实际数据库领域判断

### 优化了什么

**合并意图分类 + 上下文重写为一次 LLM 调用**，返回 JSON 结构：

```json
{
  "intent": "sql_query",
  "rewritten_query": "贵州茅台2026年第一季度营收多少"
}
```

### 怎么优化的

#### 1. 新 prompt 设计

```
你是一个智能助手，同时完成两个任务：

1. **意图分类**：根据数据库领域摘要和用户问题，判断意图类别
2. **查询重写**：结合对话历史，将代词化/省略的查询重写为独立完整的查询

意图类别说明：
- sql_query：用户想查询数据库中存储的结构化数据（必须与数据库领域摘要中的表/字段相关）
- chat：闲聊、通用问答、或问题与数据库领域无关

重要判断原则：
- 只有当问题明确指向数据库中的数据时，才归类为 sql_query
- 如果问题涉及的是公开信息、通用知识、股市行情等非数据库内容，应归类为 chat
- 结合对话历史重写查询时，只补充对话中明确提到的上下文，不要添加对话中没有的信息
```

**关键改进**：
- 移除硬编码示例，意图判断完全由 LLM 根据 domain_summary 决定
- 明确 sql_query 的边界：只有指向数据库的问题才是 sql_query
- 添加"不要添加对话中没有的信息"约束，防止 LLM 过度推断

#### 2. 返回 JSON 结构

classify 端点返回 `{intent, rewritten_query}`，前端捕获后：
- SQL 路径：invoke 请求带上 `{intent, rewritten_query}`
- Chat 路径：stream 请求直接用 `rewritten_query` 替代原始 query

#### 3. 下游节点跳过 LLM

- `classify_intent`：检测到 state 中有 intent + rewritten_query，跳过 LLM
- `contextualize_query`：检测到 state 中有 rewritten_query，跳过 LLM

#### 4. 优化效果

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| LLM 调用次数 | 3 次（classify + rewrite + generate） | 2 次（classify+rewrite 合并 + generate） |
| 意图准确性 | 硬编码示例导致偏见 | 纯 LLM 判断，结合 domain_summary |
| 上下文重写 | 重写在分类之后，意图可能不准 | 重写和分类同时完成，意图基于重写后的查询 |
| "一季度营收多少" | 可能误判为 sql_query | LLM 看到 domain_summary 中无此数据，归类为 chat |

### 涉及文件

| 文件 | 改动 |
|------|------|
| `agents/flow/dispatcher.py` | classify_intent 合并重写，返回 JSON，prompt 去硬编码 |
| `agents/api/routers/query.py` | ClassifyResponse 加 rewritten_query，QueryRequest 加 rewritten_query |
| `agents/flow/sql_react.py` | contextualize_query 跳过逻辑 |
| `agents/static/index.html` | 捕获 rewritten_query，传给 invoke 和 stream |

---

## 迭代 11：下游节点去除对话历史依赖 ✅

### 为什么优化

迭代 10 将意图分类 + 上下文重写合并为一次 LLM 调用，返回 `rewritten_query`。但发现：

1. `dispatcher.py` 的 `sql_react` 节点仍然将 `chat_history` 传递给 SQL React 子图，子图中 `contextualize_query` 虽然检测到 `rewritten_query` 后跳过 LLM，但 `chat_history` 作为 state 字段被白白传递
2. `chat_direct` 节点将 `rewritten_query` 作为 `query` 传给 RAG Chat 子图，但未传递 `rewritten_query` 标记，导致 RAG Chat 的 `rewrite` 节点再次调用 LLM 重写（浪费 token）
3. 对话历史只在意图分析和查询重写两个阶段有价值，之后的节点（表选择、证据检索、SQL 生成、RAG 对话）都应该直接使用重写后的查询

### 优化了什么

**核心原则**：对话历史只在最外层 `classify_intent` 使用一次，之后所有下游节点只使用 `rewritten_query`。

1. `dispatcher.py` 的 `sql_react` 节点：移除 `chat_history` 传递，只传 `query` + `rewritten_query`
2. `dispatcher.py` 的 `chat_direct` 节点：将 `rewritten_query` 传入 RAG Chat 的 input dict
3. `rag_chat.py` 的 `preprocess` 节点：从 input 中读取 `rewritten_query`
4. `rag_chat.py` 的 `rewrite` 节点：检测到 `rewritten_query` 已存在时跳过 LLM 调用

### 怎么优化的

#### dispatcher.py 变更

```python
# 优化前：传递 chat_history
result = await sql_graph.ainvoke({
    "query": state["query"],
    "rewritten_query": state.get("rewritten_query", ""),
    "chat_history": state.get("chat_history", []),  # 多余
})

# 优化后：只传必要字段
result = await sql_graph.ainvoke({
    "query": state["query"],
    "rewritten_query": state.get("rewritten_query", ""),
})
```

```python
# 优化前：rag_chat 没有 rewritten_query 标记
result = await rag_graph.ainvoke({
    "input": {"session_id": ..., "query": rewritten or state["query"]},
})

# 优化后：明确传递 rewritten_query
result = await rag_graph.ainvoke({
    "input": {
        "session_id": ...,
        "query": rewritten or state["query"],
        "rewritten_query": rewritten,  # 标记已重写
    },
})
```

#### rag_chat.py 变更

```python
# preprocess: 读取 rewritten_query
return {
    "session": session.model_dump(),
    "query": inp["query"],
    "rewritten_query": inp.get("rewritten_query", ""),  # 新增
    ...
}

# rewrite: 跳过逻辑
async def rewrite(state):
    existing = state.get("rewritten_query", "")
    if existing:
        return {"rewritten_query": existing}  # 跳过 LLM
    # ... 原有重写逻辑
```

### 优化效果

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| SQL 路径 LLM 调用 | 3 次（classify+rewrite + contextualize 跳过 + generate） | 2 次（classify+rewrite + generate） |
| Chat 路径 LLM 调用 | 3 次（classify+rewrite + rag_rewrite + chat） | 2 次（classify+rewrite + chat） |
| chat_history 传递 | 从 API → dispatcher → sql_react 全链路携带 | 只在 API → dispatcher 使用，下游不传 |
| Token 消耗 | chat_history 占用 prompt token（3 轮 ≈ 500 token） | 下游节点无历史，prompt 更精简 |

### 涉及文件

| 文件 | 改动 |
|------|------|
| `agents/flow/dispatcher.py` | sql_react 移除 chat_history 传递；chat_direct 传递 rewritten_query |
| `agents/flow/rag_chat.py` | preprocess 读取 rewritten_query；rewrite 跳过逻辑 |
| `agents/flow/sql_react.py` | 无变更（contextualize_query 跳过逻辑已有） |

---

## 迭代 12：recall_evidence 混合检索 + 质量过滤 ✅

### 为什么优化

`recall_evidence` 节点检索业务知识和智能体知识库，但存在两个问题：

1. **只用向量检索，不用 ES BM25**：`recall_business_knowledge` 和 `recall_agent_knowledge` 只查 Milvus 向量，缺少关键词精确匹配。当用户用精确业务术语提问时（如"预算执行率"），向量检索可能召回语义相似但内容无关的文档。
2. **无质量过滤**：智能体知识库（agent_knowledge）的核心价值是 SQL 示例（few-shot），但向量检索可能召回没有 SQL 的纯文本描述。业务知识（business_knowledge）的核心价值是公式和术语定义，但可能召回无公式的一般性描述。
3. **seed 脚本不索引到 ES**：`seed_business_knowledge.py` 和 `seed_agent_knowledge.py` 只存 Milvus，ES 中没有这些数据，BM25 检索无结果。

### 优化了什么

1. `recall_business_knowledge` 和 `recall_agent_knowledge` 改为混合检索：向量 + ES BM25 + RRF 融合
2. 新增质量过滤：`_filter_has_sql`（agent_knowledge 必须含 SQL）、`_filter_has_business_term`（business_knowledge 必须含公式/术语）
3. seed 脚本新增 ES 索引，Milvus + ES 双写
4. 抽取 `_milvus_vector_search` 和 `_es_bm25_search` 公共函数

### 怎么优化的

#### 混合检索流程

```
recall_business_knowledge / recall_agent_knowledge:
  query → Milvus 向量检索 (source filter) → vector_docs
  query → ES BM25 关键词检索 (metadata.source filter) → es_docs
  ↓
  RRF 融合 [vector_docs, es_docs] → fused_docs
  ↓
  质量过滤 → filtered_docs
```

#### ES BM25 检索

```python
def _es_bm25_search(query, source, top_k=10):
    body = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [{"match": {"text": query}}],
                "filter": [{"term": {"metadata.source": source}}],
            }
        },
    }
    resp = es.search(index=settings.es.index, body=body)
```

使用 raw ES client（与 schema_indexer 和 seed 脚本格式一致），搜索 `text` 字段，过滤 `metadata.source`。

#### 质量过滤

```python
def _filter_has_sql(docs):
    """agent_knowledge 必须包含 SELECT/INSERT/UPDATE 等 SQL 关键词。"""
    sql_keywords = ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER")
    return [d for d in docs if any(kw in d.page_content.upper() for kw in sql_keywords)]

def _filter_has_business_term(docs):
    """business_knowledge 必须包含公式/定义/术语关系。"""
    formula_indicators = ("=", "/", "*", "SUM", "COUNT", "公式", "定义", "计算", "比率", "率")
    return [d for d in docs if any(ind in d.page_content for ind in formula_indicators)]
```

#### Seed 脚本 ES 索引

```python
# seed_agent_knowledge.py / seed_business_knowledge.py
# 新增 ES 索引（与 Milvus 并行）
es = Elasticsearch(es_url)
for doc, doc_id in zip(docs, doc_ids):
    es.index(
        index=settings.es.index,
        id=doc_id,
        document={
            "text": doc["content"],
            "metadata": {"source": "agent_knowledge", "doc_id": doc_id},
        },
    )
```

### 优化效果

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| "预算执行率怎么算" | 向量检索可能召回相关但无公式的描述 | BM25 精确匹配"预算执行率" + 向量语义匹配，RRF 融合 |
| agent_knowledge 召回 | 可能召回无 SQL 的纯文本描述（浪费 token） | 质量过滤丢弃无 SQL 结果，只保留 few-shot 示例 |
| business_knowledge 召回 | 可能召回无公式的一般性描述 | 质量过滤丢弃无公式/术语结果，只保留定义和公式 |
| seed 数据 ES 检索 | ES 中无 business/agent knowledge 数据 | seed 脚本双写 Milvus + ES |

### 涉及文件

| 文件 | 改动 |
|------|------|
| `agents/rag/retriever.py` | 新增 `_milvus_vector_search`、`_es_bm25_search`、`_filter_has_sql`、`_filter_has_business_term`；重写 `recall_business_knowledge`、`recall_agent_knowledge` |
| `scripts/seed_business_knowledge.py` | 新增 ES 索引 |
| `scripts/seed_agent_knowledge.py` | 新增 ES 索引 |

---

## 迭代 13：SQL React 图流程重构（证据前置 + 查询增强 + 语义模型） ✅

### 为什么优化

原流程 `select_tables → recall_evidence → sql_retrieve`，业务知识在选表之后才召回，导致：

1. **选表不准确**：用户问"GMV是多少"，向量检索 schema 文档时"GMV"无法匹配到 `orders` 表，因为 schema 中没有"GMV"这个词。但业务知识中有"GMV = 已支付订单总额"，如果先召回业务知识，就能用"已支付订单总额"去匹配 schema。
2. **缺少查询增强**：用户用业务术语提问（如"华东区GMV"），但数据库字段是物理名（如 `region`, `amount`），向量检索匹配度低。
3. **语义模型只在索引时使用**：`t_semantic_model` 的字段业务映射在 schema_indexer 索引时嵌入文档，但查询时无法直接访问结构化的语义模型数据来辅助 SQL 生成。

### 优化了什么

1. **流程重排**：`recall_evidence` 移到 `select_tables` 之前，业务知识先于选表
2. **新增 `query_enhance` 节点**：用证据翻译业务术语，增强向量检索命中率
3. **语义模型查询时加载**：`sql_retrieve` 阶段从 MySQL 加载选中表的语义模型，注入 `sql_generate` prompt

### 怎么优化的

#### 新流程

```
优化前: START → contextualize_query → select_tables → recall_evidence → sql_retrieve → ...
优化后: START → contextualize_query → recall_evidence → query_enhance → select_tables → sql_retrieve (+ semantic model) → ...
```

#### 新增 `query_enhance` 节点

```
输入: rewritten_query + evidence + few_shot_examples
输出: enhanced_query

示例:
  Query: "华东区上月GMV是多少"
  Evidence: "GMV = 已支付订单总额", "华东包含上海、江苏、浙江..."
  Enhanced: "华东区（上海、江苏、浙江）上月已支付订单总额是多少"
```

- 无证据时跳过（返回原查询）
- LLM 失败时 graceful degradation（返回原查询）
- 超时复用 `llm_rewrite_timeout`（15s）

#### `select_tables` 查询源变更

```python
# 优化前
query = state.get("rewritten_query") or state.get("query", "")

# 优化后
query = state.get("enhanced_query") or state.get("rewritten_query") or state.get("query", "")
```

#### `sql_retrieve` 扩展：加载语义模型

```python
# 新增：从 MySQL 加载选中表的字段业务映射
semantic = await asyncio.wait_for(
    asyncio.to_thread(get_semantic_model_by_tables, selected),
    timeout=settings.resilience.milvus_timeout,
)
return {"docs": docs, "semantic_model": semantic}
```

#### `sql_generate` prompt 增强

```python
# 语义模型文本格式
语义模型（字段业务映射）:
表 t_orders:
  amount | 业务名: 订单金额 | 同义词: 交易金额, GMV | 描述: 已支付订单的总金额
  region | 业务名: 区域 | 同义词: 地区 | 描述: 订单所属区域
```

prompt 新增要求："语义模型中提供了字段的业务名称和同义词，生成 SQL 时优先使用物理字段名，但可参考业务名称理解字段含义"

### 优化效果

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| "GMV是多少" | 向量检索匹配不到相关表 | 业务知识翻译 → "已支付订单总额" → 精准匹配 |
| "华东区费用" | 向量检索"华东"匹配度低 | 查询增强补充省份列表 → 更好的匹配 |
| SQL 字段映射 | LLM 只看 schema 文档中的 COMMENT | 语义模型提供结构化的业务名、同义词、描述 |
| 无业务知识 | 正常工作 | query_enhance 跳过，降级为原流程 |

### 涉及文件

| 文件 | 改动 |
|------|------|
| `agents/flow/state.py` | SQLReactState 新增 `enhanced_query`、`semantic_model` |
| `agents/flow/sql_react.py` | 新增 `query_enhance` 节点；修改 `select_tables`/`sql_retrieve`/`sql_generate`/`_build_sql_messages`；更新图拓扑 |
| `agents/rag/retriever.py` | 新增 `get_semantic_model_by_tables()` |
| `tests/test_sql_react.py` | 新增 `query_enhance` 测试；更新 graph 节点断言 |

## 迭代 14：表描述 + 统一语义模型 + 表关系

### 为什么优化

1. **select_tables 只传英文表名**：LLM 看到 `t_order, t_user, t_payment` 无法判断哪个表与"订单金额"相关
2. **schema docs 与 semantic_model 重复**：Milvus 向量检索的 schema 文档和 MySQL 的 semantic_model 包含重叠信息
3. **缺少表关系信息**：sql_generate 不知道表之间如何 JOIN

### 优化了什么

**1. select_tables 表名带描述**

加载 `information_schema.tables` 的 TABLE_COMMENT，LLM prompt 格式：
```
候选表名:
- t_order: 订单主表
- t_user: 用户信息表
- t_payment: 支付记录表
```

**2. 统一到 t_semantic_model，去掉 Milvus schema 向量检索**

- 扩展 `t_semantic_model` 表，新增字段：`column_type`, `column_comment`, `is_pk`, `is_fk`, `ref_table`, `ref_column`
- 种子脚本自动从 `information_schema.columns` + `information_schema.key_column_usage` 同步技术 schema
- `sql_retrieve` 改为只查 MySQL `t_semantic_model`，不再从 Milvus 拉取 schema docs
- 从 semantic_model 构建 schema 文档（`_build_schema_docs_from_semantic`）

**3. select_tables 返回表关系**

- 新增 `get_table_relationships()` 函数，从 MySQL `information_schema.key_column_usage` 提取外键关系
- `select_tables` 返回 `selected_tables` + `table_relationships`
- `sql_generate` prompt 中加入表关系信息，帮助 LLM 生成正确的 JOIN 条件

**4. 关键词过滤字段**

- 新增 `_extract_keywords(query)` 使用 Jieba 分词提取关键词
- 新增 `_filter_columns_by_keywords()` 根据关键词过滤 schema 文档中的字段
- 保留匹配字段 + 时间字段 + PK/FK，精简 prompt

### 涉及文件

| 文件 | 改动 |
|------|------|
| `agents/flow/state.py` | SQLReactState 新增 `table_relationships` |
| `agents/flow/sql_react.py` | `select_tables` 加载表描述+表关系；`sql_retrieve` 改为 MySQL-only；新增 `_build_schema_docs_from_semantic`、`_extract_keywords`、`_filter_columns_by_keywords` |
| `agents/rag/retriever.py` | `get_semantic_model_by_tables` 返回完整 schema 字段；新增 `get_table_relationships()` |
| `scripts/seed_semantic_model.py` | 扩展表结构；新增 `sync_schema_from_information_schema()` |
| `pyproject.toml` | 添加 `jieba>=0.42` 依赖 |
| `tests/test_sql_react.py` | 更新 `test_generate_retrieves_missing_tables` 使用 semantic_model mock |

### 优化效果

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| 表选择准确度 | LLM 只看英文表名 | 表名+描述，更易判断 |
| schema 信息来源 | Milvus 向量检索 + MySQL semantic_model | 统一 MySQL semantic_model |
| 表关联 | LLM 自己猜 JOIN 条件 | 提供外键关系信息 |
| prompt 大小 | 全量字段 | 关键词过滤后只保留 5-10 个核心字段 |

## 迭代 15：t_semantic_model 自动同步（全量初始化 + binlog 增量）

### 为什么优化

手动执行 `seed_semantic_model.py` 容易遗忘，新增表或字段后 semantic_model 不会自动更新。

### 优化了什么

新增 `agents/init/schema_sync.py` 模块，应用启动时自动同步：

**1. 全量初始化**
- 启动时检查 `t_semantic_model` 是否有数据
- 无数据则自动全量同步：从 `information_schema.tables` + `information_schema.columns` + `information_schema.key_column_usage` 读取所有表结构

**2. binlog 增量同步**
- 使用 `python-mysql-replication` 监听 MySQL binlog
- 检测 DDL 事件（CREATE TABLE / ALTER TABLE / DROP TABLE / RENAME TABLE）
- 自动增量更新 `t_semantic_model`

**3. 定时轮询 fallback**
- binlog 不可用时（权限、配置等），每 5 分钟轮询 `information_schema`
- 检测新增/删除的表，增量同步

### 涉及文件

| 文件 | 改动 |
|------|------|
| `agents/init/__init__.py` | 新建 init 模块 |
| `agents/init/schema_sync.py` | 全量同步 + binlog 监听 + 轮询 fallback |
| `agents/api/app.py` | lifespan 中启动 schema_sync 后台任务 |
| `pyproject.toml` | 添加 `python-mysql-replication>=1.0` |

## 迭代 16：Bug 修复 + Jieba 过滤踩坑回顾

### Bug 1：ES BM25 检索返回无关结果

**现象**：`recall_agent_knowledge` 搜索 "毛利率" 时，ES 返回的全是财务报告摘要（贵州茅台一季报），而非 SQL 示例。

**原因**：ES 索引中 `metadata.source` 是 `text` 类型，`term` 查询无法精确匹配。财务报告被错误索引为 `agent_knowledge`，BM25 匹配到 "毛利" 关键词后返回。

**修复**：`_es_bm25_search` 的过滤条件从 `metadata.source` 改为 `metadata.source.keyword`（keyword 子字段支持精确匹配）。

### Bug 2：few_shot_examples 检索不到

**现象**：用户问 "查询上周毛利率" 时，`few_shot_examples` 返回空。

**原因**：
1. ES 返回无关结果（Bug 1）
2. `_recall_agent` 的 `top_k=3` 太小，向量检索返回的 3 个结果中，财务报告占了位置，过滤后无 SQL 示例

**修复**：
1. 修复 ES 精确匹配（Bug 1）
2. `_recall_agent` 的 `top_k` 从 3 增加到 10，确保过滤后仍有足够 SQL 示例

### Bug 3：逻辑外键未同步

**现象**：`t_journal_item.account_code` 在语义模型中 `is_fk=0`，schema 文档中没有 `REFERENCES` 标记。

**原因**：`sync_schema_from_information_schema` 只从 `information_schema.key_column_usage` 同步 FK，但数据库中未定义外键约束（业务逻辑上的关联，非数据库 FK）。

**修复**：在 `seed_semantic_model.py` 新增 `seed_logical_foreign_keys()` 函数，手动更新 6 个逻辑外键：
- `t_journal_item.entry_id` → `t_journal_entry.id`
- `t_journal_item.account_code` → `t_account.account_code`
- `t_journal_item.cost_center_id` → `t_cost_center.id`
- `t_budget.cost_center_id` → `t_cost_center.id`
- `t_budget.account_code` → `t_account.account_code`
- `t_expense_claim.cost_center_id` → `t_cost_center.id`

### Bug 4：MySQL 8.0 SHOW MASTER STATUS 废弃

**现象**：binlog 监听启动时报 SQL 语法错误。

**原因**：MySQL 8.0.22+ 废弃了 `SHOW MASTER STATUS`，改用 `SHOW BINARY LOG STATUS`。

**修复**：先尝试 `SHOW BINARY LOG STATUS`，失败则 fallback 到 `SHOW MASTER STATUS`。

### Bug 5：服务用系统 Python 启动

**现象**：`mysql-replication not installed, binlog listener disabled`。

**原因**：服务用 `/opt/homebrew/Cellar/python@3.14/...` 启动（系统 Python），但包装在 venv 里。

**修复**：用 venv Python 启动：`/path/to/.venv/bin/python -m agents.main`。

### 踩坑：Jieba 关键词过滤字段（已废弃）

**方案**：用 Jieba 分词提取查询关键词，过滤 schema 文档中不匹配的字段，精简 prompt。

**实现**：
```python
def _extract_keywords(query: str) -> list[str]:
    import jieba
    words = jieba.cut(query)
    return [w.strip() for w in words if len(w.strip()) >= 2]

def _filter_columns_by_keywords(docs, semantic_model, keywords):
    # 只保留匹配字段 + 时间字段 + PK/FK
    ...
```

**遇到的问题**：

1. **派生指标无法匹配**：用户问 "上周毛利率"，Jieba 切成 `["上周", "毛利率"]`。但 `毛利率` 的计算公式是 `(SUM(credit_amount) - SUM(debit_amount)) / SUM(credit_amount)`，字段名 `credit_amount` 和 `debit_amount` 不包含 "毛利率" 关键词，被错误过滤掉。

2. **业务术语到物理字段的映射只有业务知识能桥接**：关键词匹配是字符串层面的，无法理解 "毛利率 = (借方 - 贷方) / 借方" 这种业务定义。

**结论**：参考 DataAgent 项目，**不做字段级过滤**。DataAgent 的做法是：
- 选中表后返回全部字段（最多 50 列/表）
- 靠业务知识（evidence）+ 语义模型（semantic_model）+ SQL 生成 LLM 自己判断用哪些列
- 字段选择的负担放在 LLM 上，而非关键词匹配

**最终方案**：删除 `_extract_keywords` 和 `_filter_columns_by_keywords`，移除 `jieba` 依赖。`sql_retrieve` 直接返回选中表的全部字段。

### 涉及文件

| 文件 | 改动 |
|------|------|
| `agents/flow/sql_react.py` | 删除 `_extract_keywords`、`_filter_columns_by_keywords`；`_recall_agent` top_k 3→10 |
| `agents/rag/retriever.py` | `_es_bm25_search` 过滤条件改用 `metadata.source.keyword` |
| `agents/init/schema_sync.py` | binlog 兼容 MySQL 8.0.22+；线程池执行阻塞操作 |
| `scripts/seed_semantic_model.py` | 新增 `seed_logical_foreign_keys()`；修复 DictCursor |
| `pyproject.toml` | 移除 `jieba>=0.42`；修正 `mysql-replication>=1.0` 包名 |
| `tests/test_sql_react.py` | 新增 `TestContextualizeQuery` 测试 |

### 教训总结

| 问题 | 教训 |
|------|------|
| Jieba 过滤字段 | 关键词匹配无法处理派生指标，业务术语到物理字段的映射需要业务知识桥接 |
| ES 精确匹配 | `text` 字段用 `term` 查询会匹配失败，要用 `keyword` 子字段 |
| top_k 太小 | 有质量过滤时，top_k 要放大，否则过滤后可能为空 |
| MySQL 版本兼容 | `SHOW MASTER STATUS` 在 8.0.22+ 废弃，用 `SHOW BINARY LOG STATUS` |
| 系统 Python vs venv | 包装在 venv 里但用系统 Python 启动会找不到包 |

---

## 迭代 17：去除 Milvus Schema 索引依赖 ✅

### 为什么优化

统一 `t_semantic_model` 后，`sql_retrieve` 已经只从 MySQL 加载 schema，但 `select_tables` 仍然依赖 Milvus 做表发现（向量检索 schema 文档提取表名）。同时 admin 页面的"刷新 Schema"按钮、启动时的 `_index_schemas_background` 自动索引、`schema_indexer.py` 的 Milvus+ES 双写都已不再需要。

**核心矛盾**：schema 数据已经统一到 MySQL `t_semantic_model`，但表发现仍在绕道 Milvus，增加了不必要的依赖和延迟。

### 优化了什么

**1. select_tables 去除 Milvus 依赖**

- 移除 `search_schema_tables`（Milvus 向量检索 schema 文档返回候选表名）
- 移除 `get_schema_table_names`（Milvus metadata 查询所有表名）
- 移除 `load_table_names` 节点（已不在图中，但函数定义仍存在）
- `select_tables` 改为直接从 `load_full_table_metadata()` 加载表名+描述（MySQL `information_schema.tables`）

```
优化前: query → Milvus 向量检索 top-10 schema docs → 提取候选表名 → LLM 精选
优化后: query → MySQL information_schema.tables 加载全量表名+描述 → LLM 精选
```

**2. 移除 admin 刷新 Schema 功能**

- 删除 `POST /api/admin/refresh-schemas` 端点（调用 `schema_indexer.index_mysql_schemas`）
- 删除前端 Admin Tab 的"刷新 Schema"按钮和 `refreshSchemas()` 函数
- Admin Tab 改为提示用户使用 seed 脚本管理语义模型

**3. 移除启动时 schema 自动索引**

- 删除 `_index_schemas_background`（向量化 schema 到 Milvus + ES）
- 替换为 `_ensure_domain_summary`（仅在 domain_summary 为空时，从 MySQL 表元数据生成领域摘要）
- 领域摘要仍用于意图分类，但不再依赖 Milvus schema 文档

**4. 清理死代码**

- 删除 `get_schema_table_names`（retriever.py）
- 删除 `search_schema_tables`（retriever.py）
- 删除 `get_schema_docs_by_tables`（retriever.py，零调用方）

**保留的代码**

- `schema_indexer.py`：保留，seed 脚本仍在使用（`seed_financial.py`、`seed_all.py`）
- `VectorOnlyRetriever`：保留，eval runner 仍在使用
- Milvus + ES：保留用于业务知识、智能体知识、用户文档的向量检索

### 涉及文件

| 文件 | 改动 |
|------|------|
| `agents/flow/sql_react.py` | `select_tables` 改为 MySQL-only；移除 `load_table_names`；移除 `search_schema_tables`/`get_schema_table_names` 导入 |
| `agents/rag/retriever.py` | 删除 `get_schema_table_names`、`search_schema_tables`、`get_schema_docs_by_tables` |
| `agents/api/routers/admin.py` | 删除 `POST /refresh-schemas` 端点和 `RefreshResponse` 模型 |
| `agents/api/app.py` | `_index_schemas_background` → `_ensure_domain_summary` |
| `agents/static/index.html` | 删除 Admin Tab 的刷新 Schema 按钮和 JS |

### 优化效果

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| 表发现 | Milvus 向量检索 + embedding 计算（~200ms） | MySQL 直查（~10ms） |
| 外部依赖 | select_tables 依赖 Milvus 可用 | select_tables 只依赖 MySQL |
| 启动索引 | 启动时向量化所有表 schema 到 Milvus + ES | 仅生成 domain_summary（如缺失） |
| Admin 功能 | 手动刷新 schema 索引 | 不再需要（schema 自动同步） |
| 代码量 | 3 个 Milvus schema 函数 + 启动索引 + admin 端点 | 全部移除 |

---

## Iteration 18：Schema 元数据 Redis 缓存

### 出现了什么问题

`load_full_table_metadata` 和 `get_semantic_model_by_tables` 每次请求都查 MySQL。高频调用时（如 SQL Agent 每次对话都触发 select_tables → sql_retrieve），产生不必要的 DB 压力。

### 为什么要解决

- 表元数据变化频率低（仅 DDL 时变更），适合缓存
- `start_schema_sync` 已有全量同步 + binlog 增量机制，可作为缓存维护者
- Redis 读取延迟 ~1ms vs MySQL ~10ms，减少 SQL Agent 响应时间

### 怎么解决

**Redis Key 设计**

| Key | 类型 | 内容 | TTL |
|-----|------|------|-----|
| `schema:table_metadata` | string (JSON) | `[{table_name, table_comment}, ...]` | 无（由 sync 任务维护） |
| `schema:semantic_model:{table}` | string (JSON) | `{column_name: {col_type, is_pk, ...}}` | 无 |

**Cache-Aside 模式**

1. 查询时：Redis → MySQL fallback → 回填 Redis
2. 同步时：更新 MySQL → 刷新 Redis

**涉及文件**

| 文件 | 改动 |
|------|------|
| `agents/rag/retriever.py` | 新增 `_get_sync_redis()` + Redis 常量；`load_full_table_metadata` 改为 Redis→MySQL→回填；`get_semantic_model_by_tables` 改为 Redis pipeline→MySQL→回填 |
| `agents/init/schema_sync.py` | 新增 `_refresh_redis_cache()`；全量同步后刷新 Redis；binlog 增量同步后更新指定表缓存；轮询检测到新增/删除表后更新缓存 |

**关键实现**

- Sync Redis 客户端（非 async），因为 `retriever.py` 的函数通过 `asyncio.to_thread` 调用
- Redis pipeline 批量操作（多个 `get` 或 `set` 一次 round-trip）
- Per-table key 便于增量失效（binlog 只刷新受影响的表）
- Redis 不可用时 graceful fallback 到 MySQL（try/except 兜底）

### 优化效果

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| load_full_table_metadata | 每次查 MySQL information_schema (~10ms) | Redis hit ~1ms，miss 时回填 |
| get_semantic_model_by_tables | 每次查 MySQL t_semantic_model (~5ms×N) | Redis pipeline 一次性取 (~1ms) |
| DDL 变更后 | 立即生效（直查 MySQL） | binlog → 更新 MySQL → 刷新 Redis（秒级延迟） |
| Redis 不可用 | N/A | 自动 fallback 到 MySQL，无影响 |

---

## Iteration 19：MCP 错误日志 + API 异常捕获

### 出现了什么问题

SQL Agent 执行 SQL 返回 HTTP 500，但服务器日志中看不到任何错误信息。错误被静默吞掉，无法排查。

### 为什么要解决

- 无日志 = 无法排查。500 可能来自 MCP 执行失败、LLM 调用超时、图执行异常等多种原因
- 需要在关键路径记录错误，快速定位问题

### 怎么解决

**涉及文件**

| 文件 | 改动 |
|------|------|
| `agents/tool/sql_tools/mcp_client.py` | `execute_sql` 记录入参 SQL + 返回结果；检测 `result.isError` 并 `raise RuntimeError` |
| `agents/api/routers/query.py` | `query_invoke` / `approve_sql` 添加 `try/except` + `logger.error(exc_info=True)`，返回错误而非 500 |

**关键改动**

- MCP 返回 `isError=True` 时，之前静默返回错误文本 → 现在 `logger.error` + `raise`
- API 端点无 try/except → 现在捕获异常并返回 `"系统错误: ..."` + 完整 traceback 日志
