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

## Iteration 42：移除 SQL 子图二次查询重写节点

### 背景

`classify_intent` 已经一次性完成意图分类和查询重写，返回 `intent + rewritten_query`。SQL React 子图中的 `contextualize_query` 原本用于兼容直接调用 SQL 子图且没有传入 `rewritten_query` 的旧路径，但当前产品链路已经明确不支持绕过 `classify_intent` 直接进入 SQL 子图。

保留该节点会带来三个问题：

- 链路追踪中多一个“看起来会 rewrite”的节点，但正常请求只是透传，容易误导排障。
- 文档和图节点职责重复，让人以为 SQL 子图还会做第二次查询重写。
- 旧兼容逻辑可能重新读取历史消息并产生新的 rewritten query，和外层分类节点的稳定输入原则冲突。

### 方案

- 删除 `agents/flow/sql_react.py` 中的 `contextualize_query` 节点和 `rewrite_query` 依赖。
- SQL React 子图入口从 `START -> contextualize_query -> recall_evidence` 改为 `START -> recall_evidence`。
- 保留 `query/rewritten_query` 作为稳定 state 字段，下游节点继续按 `enhanced_query -> rewritten_query -> query` 的优先级读取。
- README 和技术设计文档同步更新，明确查询重写只在外层 `classify_intent` 完成。

### TDD 验证

先新增图结构测试，要求：

- SQL React 图中不再包含 `contextualize_query` 节点。
- `__start__` 直接连接 `recall_evidence`。

红灯结果：

```bash
.venv/bin/python -m pytest tests/test_sql_react.py::TestBuildSqlReactGraph::test_graph_has_all_nodes tests/test_sql_react.py::TestBuildSqlReactGraph::test_graph_starts_at_recall_evidence -q
# 2 failed，原因是图中仍存在 contextualize_query，START 仍连向 contextualize_query
```

实现后再跑 SQL React 相关测试，保证结构变更没有破坏现有节点行为。
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

---

## Iteration 20：MCP MySQL Collation 冲突修复（二）

### 出现了什么问题

上次修复将 MCP MySQL charset 设为 `utf8mb4`，解决了 `utf8mb4_unicode_ci` 冲突。但 `utf8mb4` 在 mysql2 中默认 collation 是 `utf8mb4_general_ci`，与 MySQL 8.0 的 `utf8mb4_0900_ai_ci` 仍然不同，导致：

```
Illegal mix of collations (utf8mb4_0900_ai_ci,IMPLICIT) and (utf8mb4_general_ci,IMPLICIT) for operation '='
```

### 为什么要解决

- Collation 不一致导致 JOIN / WHERE 中的字符串比较失败
- `utf8mb4` 只指定字符集，不指定 collation，mysql2 会用旧版默认值

### 怎么解决

**涉及文件**

| 文件 | 改动 |
|------|------|
| mcp-server-mysql config | `charset` 默认值改为 `utf8mb4_0900_ai_ci` |
| `agents/tool/sql_tools/mcp_client.py` | `mcp_env` 中增加 `MYSQL_CHARSET=utf8mb4_0900_ai_ci`，不依赖 npx 缓存 |

**关键点**

- mysql2 的 `charset` 选项同时控制字符集和 collation
- 设置 `MYSQL_CHARSET` 环境变量确保即使 npx 缓存清除后修复仍生效

---

## Iteration 21：NL2SQL 增强稳定性 + 异常结果反思

### 出现了什么问题

SQL Agent 在处理类似“去年亏损”的查询时暴露了几类问题：

1. `query_enhance` 依赖 LLM 输出，LLM 空响应时直接退回原 query，业务术语增强不稳定。
2. 业务知识召回只依赖向量/BM25，口语化同义词（如“亏损”）召回不到时，增强链路无法使用 `t_business_knowledge`。
3. LLM 生成的 SQL 可能带有异常 token、Markdown 代码块、尾部截断关键字（如 `HAVIN`）或多余分号。
4. SQL 执行成功但结果异常（空集、NULL、包装结构中的 `rows: []` 等）时，原流程直接结束，无法自我修正。
5. 结果异常后反思生成修正 SQL，会再次触发审批；如果前端没有明确过程提示，用户会以为同一条 SQL 被重复审批。
6. approve 恢复父图时，缺失 `query` 会报 `KeyError: 'query'`；补回 `query` 后若节点同一步再次写入 `query`，会触发 LangGraph `INVALID_CONCURRENT_GRAPH_UPDATE`。

### 为什么要解决

- NL2SQL 的错误不只来自 SQL 执行异常，也可能来自“结果可执行但语义不可信”。
- 业务口径应来自可配置业务知识，而不是在代码里硬编码某个 query 的词表。
- 审批是用户交互节点，自动反思和二次确认必须让用户看见过程，否则体验上像“重复弹窗”。
- LangGraph 恢复时状态字段要稳定，避免 approve 后在父图/子图之间丢上下文。

### 怎么解决

**1. 业务知识驱动的 query_enhance**

- 移除 `_PROFIT_LOSS_HINTS`、`去年` 等 case 级硬编码。
- `query_enhance` 只解析召回到的业务知识 evidence：`术语`、`公式/定义`、`同义词`。
- `recall_business_knowledge` 增加 MySQL 词典兜底：当 Milvus/ES 未召回足够结果时，从 `t_business_knowledge.term/synonyms` 做通用同义词匹配。
- 业务词扩展通过维护 `t_business_knowledge` 完成，不再改代码。

**2. SQL 输出格式化与校验**

- 新增 `normalize_sql_answer()`，统一处理：
  - `<text_never_used_...>` / `</text_never_used_...>` 异常 token
  - Markdown SQL 代码块
  - SQL 前的解释性文本
  - 尾部多余分号
  - 截断关键字（如 `HAVIN`、`WHERE`、`AND`、`GROUP BY`）
  - 括号不匹配、非 `SELECT/WITH` 开头
- `sql_generate` 和 `result_reflection` 都显式调用本地 formatter，不能只依赖 LLM tool schema。
- 格式不合法时返回 `is_sql=False`，不进入审批/执行。

**3. 执行结果异常检测 + 反思修正**

- 新增 `_result_anomaly_reason()`，识别：
  - 裸 `[]`
  - `{"rows": []}`、`{"data": []}`、`{"result": []}`、`{"items": []}`
  - `{"columns": [...], "rows": []}`
  - `null` / `None`
  - 结构化结果中全字段为 `NULL` 或空字符串
  - 原始结果字符串中包含 `null` 或 `[]` 的可疑信号
- 新增 `result_reflection` 节点：执行成功但结果异常时，LLM 直接反思并生成修正后的 SQL。
- 反思后的 SQL 不再回到 `sql_generate`，而是走：

```text
execute_sql -> result_reflection -> safety_check -> approve -> execute_sql
```

这样避免“反思节点给建议，再让 sql_generate 再生成一次”的重复生成。

**4. 审批与 SSE 体验**

- `approve` 对反思后的 SQL 使用不同 interrupt 文案：

```text
上次执行结果疑似异常，系统已反思并生成修正后的 SQL。请确认是否执行修正后的 SQL？
```

- 新增 `POST /api/query/approve/stream`：
  - 审批后推送“已确认，正在执行 SQL...”
  - 推送“如果执行结果异常，系统会自动反思并生成修正 SQL...”
  - 若再次进入审批，推送“检测到执行结果异常，已完成反思并生成修正 SQL”
  - 最后通过 `result` 事件返回 `QueryResponse`
- 前端 SQL Agent 审批按钮改为读取 SSE，显示带动画的进度行，再渲染最终结果或修正 SQL 审批卡片。

**5. approve 恢复状态修复**

- SQL 审批中断时仍暂存原始 query。
- approve 恢复时只发送 `Command(resume=...)`：

```python
Command(resume={
    "approved": req.approved,
    "feedback": req.feedback,
})
```

- `sql_react` 子图不再返回 `query`，减少父图/子图状态合并时的重复写入。
- `FinalGraphState.query` 和 `SQLReactState.query` 增加 `Annotated[..., keep_existing_query]` reducer。父图和 SQL 子图在 approve/resume 后若同一步携带 `query`，保留已有原始 query，避免 LangGraph `INVALID_CONCURRENT_GRAPH_UPDATE`。
- approve 完成后仍通过 session preference 中暂存的 `_pending_query` 恢复原始问题，用于保存本轮 Q&A；不再把 query 写入 graph update。

### 涉及文件

| 文件 | 改动 |
|------|------|
| `agents/flow/sql_react.py` | `query_enhance` 改为 evidence 驱动；新增结果异常检测和 `result_reflection`；反思 SQL 直接走 `safety_check`；approve 文案区分修正 SQL |
| `agents/rag/retriever.py` | 业务知识召回增加 MySQL term/synonyms 兜底 |
| `agents/model/format_tool.py` | 新增 `normalize_sql_answer()`；format tool 内部也执行 SQL 清洗校验 |
| `agents/api/routers/query.py` | 新增 `/approve/stream`；approve 恢复只使用 `Command(resume=...)`；pending query 仅用于最终 Q&A 保存 |
| `agents/static/index.html` | SQL 审批改为 SSE 进度展示；反思后修正 SQL 显示明确状态 |
| `agents/flow/state.py` | 新增 `reflection_notice` 状态字段；`query` 增加 `keep_existing_query` reducer |
| `scripts/seed_business_knowledge.py` | 补充“净利润”业务知识及口语化同义词 |
| `tests/test_sql_react.py` | 覆盖 SQL formatter、空/NULL 结果异常、result_reflection 直接生成 SQL |
| `tests/test_final_api.py` | 覆盖 approve 恢复状态和 approve SSE 流式事件 |

### 优化效果

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| LLM query_enhance 空响应 | 退回原 query | 使用召回到的业务知识做确定性增强 |
| 业务同义词召回失败 | 只能依赖向量/BM25 | MySQL `term/synonyms` 兜底召回 |
| SQL 带异常 token | 可能进入审批/执行 | formatter 清理或拦截 |
| SQL 尾部截断 | 可能执行失败或报错不清晰 | 本地识别为 invalid SQL |
| SQL 执行返回 `[]` / `NULL` | 直接结束 | 进入 `result_reflection` 生成修正 SQL |
| 反思后 SQL | 先给建议再进 `sql_generate` | 直接生成修正 SQL，走安全检查和审批 |
| 二次审批体验 | 用户只看到又弹出 SQL | SSE 展示执行、异常检测、反思、修正 SQL 确认 |
| approve 恢复缺 query | 报 `KeyError: 'query'` | approve resume 不依赖写回 query；最终保存从 `_pending_query` 取原始问题 |
| 重复写 query | 触发 LangGraph 并发写错误 | 子图不返回 `query`，且 `query` 字段使用 reducer 保留已有值 |

---

## Iteration 22：追问场景 SQL 口径继承

### 出现了什么问题

连续追问时，第二轮 SQL 可能和第一轮 SQL 口径不一致。例如第一轮问“去年亏损”，第二轮问“亏损多少”：

- 第一轮可能使用 `je.status = '已过账'`
- 第二轮可能改成 `je.status IN ('已审核','已过账')`
- 第一轮可能用简单借贷发生额差额
- 第二轮可能改用 `balance_direction` 公式
- 第一轮字段别名是“净利润/盈亏状态”
- 第二轮又回到“去年净利润”

这些 SQL 都可能能执行，但语义口径已经漂移，用户看到的结果会互相矛盾。

### 根因

- 外层 `classify_intent` 读取了 session history，但 `dispatcher.sql_react` 只把 `query` 和 `rewritten_query` 传给 SQL React 子图，没有传 `chat_history`。
- SQL 执行完成后只保存了自然语言 answer，没有保存上一轮 SQL 的时间范围、状态过滤、JOIN、指标公式、排除条件等口径信息。
- 后续追问进入 `sql_generate` 时缺少上一轮 SQL 上下文，LLM 会重新推断口径，因此发生状态过滤和公式漂移。

### 解决方案

**1. 保存最近一次 SQL 口径**

SQL 执行完成后，将以下内容写入 session preference 的 `_last_sql_context`：

```text
用户问题
生成 SQL
展示结果
```

这不是对业务问题硬编码，而是保存当前会话中已经确认执行过的 SQL 口径。

**2. 加载会话时注入 SQL 上下文**

`_load_chat_history()` 会把 `_last_sql_context` 注入为 system message：

```text
[上一轮SQL上下文]
用户问题: ...
生成SQL:
...
展示结果: ...
```

**3. SQL 子图接收 chat_history**

`dispatcher.sql_react` 调用 SQL React 子图时传入 `chat_history`，让 SQL 生成节点能看到上一轮 SQL 口径。

**4. Prompt 明确追问继承规则**

`sql_generate` prompt 增加规则：

- 如果上下文中提供了上一轮 SQL，且用户是在追问或省略表达，必须沿用上一轮的时间范围、状态过滤、表连接、指标计算口径和排除条件。
- 除非用户明确要求变更口径。

### 涉及文件

| 文件 | 改动 |
|------|------|
| `agents/api/routers/query.py` | 保存 `_last_sql_context`；加载 history 时注入上一轮 SQL 上下文 |
| `agents/flow/dispatcher.py` | 调用 SQL React 子图时传入 `chat_history` |
| `agents/flow/sql_react.py` | SQL 生成 prompt 加入追问继承规则；把上一轮 SQL 上下文放入生成上下文 |
| `tests/test_final_api.py` | 覆盖 SQL 上下文保存与加载 |
| `tests/test_sql_react.py` | 覆盖 prompt 约束和 SQL 上下文注入 |

### 优化效果

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| “去年亏损”后追问“亏损多少” | 第二轮重新推断 SQL 口径 | 第二轮沿用上一轮时间范围、状态过滤和指标公式 |
| 状态过滤 | 可能从 `已过账` 漂移到 `已审核/已过账` | 默认继承上一轮状态过滤 |
| 指标公式 | 可能换公式 | 默认继承上一轮指标计算口径 |
| 用户明确改口径 | 不确定 | 用户明确要求时允许变更 |

---

## Iteration 23：上一年度亏损测试数据补齐

### 出现了什么问题

用户执行“去年亏损”相关 SQL 时，最终结果仍然是 0：

- `net_profit = 0.00`
- `profit_status = 不盈不亏`
- `loss_amount = 0`

这不是 SQL 执行异常，而是测试数据缺口。

### 根因

- `scripts.seed_financial` 只生成最近 6 个月数据，按当前日期查询上一年度时可能没有完整年度数据。
- 随机记账凭证只从前 15 个科目里抽样，里面没有 `6001`、`6401`、`5401` 等损益类科目。
- 因此按 `status = '已过账'`、`account_type = '损益'`、上一年度期间过滤时，收入、成本、费用都可能没有发生额，结果自然为 0。

### 解决方案

在 `scripts.seed_financial` 中补充可重复刷新的上一年度亏损场景：

- 每次 seed 先删除同年度 `LOSS-YYYY-*` 测试凭证，避免重复累加。
- 重新插入上一年度 12 个月已过账凭证。
- 每月包含主营业务收入、主营业务成本、期间费用三类分录。
- 每张凭证借贷平衡，损益科目与银行存款科目配平。
- 年度合计保证成本和费用大于收入，使净利润为负、亏损金额为正。

这是测试数据造数，不是运行时 query 规则硬编码。SQL Agent 仍然通过 schema、语义模型、业务知识和对话上下文生成 SQL。

### 涉及文件

| 文件 | 改动 |
|------|------|
| `scripts/seed_financial.py` | 新增 `seed_prior_year_loss_scenario()`，写入稳定的上一年度亏损凭证 |
| `README.md` | 初始化脚本教程说明 `seed_financial` 包含可重复刷新的亏损测试数据 |

### 验证口径

执行 `python -m scripts.seed_financial` 后，用上一年度、已过账、损益类科目过滤，收入减成本减费用应返回负数；“亏损多少”应返回正的亏损金额。

---

## Iteration 24：清理遗留 mysql_schema 向量索引

### 出现了什么问题

Milvus 中仍然存在 `source=mysql_schema` 的历史 schema 文档。当前 SQL Agent 的候选表选择和 schema 加载已经改为 Redis/MySQL：

- 候选表：`load_full_table_metadata()`，优先 Redis，miss 后查 MySQL `information_schema.tables`
- 字段 schema：`get_semantic_model_by_tables()`，优先 Redis，miss 后查 MySQL `t_semantic_model`

因此这些旧向量记录不再参与 SQL 生成，但会造成维护和排查上的混淆。

### 解决方案

- `scripts.seed_financial` 不再执行 schema re-index。
- `scripts.seed_all` 不再执行 `schema_indexer.index_mysql_schemas()`。
- 新增 `scripts.cleanup_schema_indexes`，一次性删除 Milvus/ES 中 `source=mysql_schema` 的历史记录。
- README 初始化脚本说明同步更新：schema 数据统一由 MySQL/Redis 提供，Milvus/ES 只保留业务知识、SQL few-shot 和用户文档等非结构化检索数据。

### 检索分工

| 数据类型 | 权威来源/缓存 | 检索方式 | 原因 |
|----------|---------------|----------|------|
| 表名、表注释 | MySQL `information_schema.tables`；Redis `schema:table_metadata` 缓存 | 精确加载全量表元数据，再让 LLM 精选候选表 | 表元数据必须完整实时，向量召回可能漏表 |
| 字段 schema、业务名、同义词、字段描述 | MySQL `t_semantic_model`；Redis `schema:semantic_model:<table>` 缓存 | 按选中表精确加载 | SQL 生成需要精确字段、类型、PK/FK 和业务描述 |
| 表关系/JOIN 关系 | `information_schema.key_column_usage` + `t_semantic_model` 逻辑外键 | 按表名精确查询 | JOIN 条件不能靠语义相似度猜测 |
| 业务知识 | MySQL `t_business_knowledge` + Milvus `business_knowledge` + ES `business_knowledge` | Milvus 向量 + ES BM25 + RRF；MySQL term/synonyms 字符匹配兜底 | 业务表达有同义词和口语化说法，需要语义召回和关键词召回结合 |
| SQL few-shot | MySQL `t_agent_knowledge` + Milvus `agent_knowledge` + ES `agent_knowledge` | Milvus 向量 + ES BM25 + RRF，过滤无 SQL 内容 | 用相似问题和关键词命中补充 SQL 写法示例 |
| 用户上传文档 | Milvus/ES 文档索引 | 向量/关键词检索 + 重排 | 文档是非结构化文本，适合 RAG 检索 |
| 会话、checkpoint、schema 缓存、领域摘要缓存 | Redis | key-value 精确读写 | 高频状态数据需要低延迟缓存 |
| 业务明细数据 | MySQL 业务表 | 审批后的 SELECT SQL 执行 | MySQL 是业务事实数据的权威来源 |

### 涉及文件

| 文件 | 改动 |
|------|------|
| `scripts/seed_financial.py` | 去掉 schema re-index 调用 |
| `scripts/seed_all.py` | 去掉第 5 步 schema re-index |
| `scripts/cleanup_schema_indexes.py` | 新增历史 `mysql_schema` 索引清理脚本 |
| `README.md` | 更新 seed 流程和清理脚本说明 |

---

## Iteration 25：旧 schema 向量索引入口降级

### 出现了什么问题

Iteration 24 已经停止 seed 流程重建 `source=mysql_schema` 数据，但代码里仍有两个容易误用的入口：

- `agents.rag.schema_indexer.index_mysql_schemas()` 仍可直接把 MySQL schema 写入 Milvus/ES。
- `agents.eval.dataset_generator` 仍从 Milvus `source=mysql_schema` 生成评测数据。

这会让新架构和旧评测/维护脚本出现口径不一致：线上 NL2SQL 从 Redis/MySQL 读 schema，但评测和手工脚本仍可能依赖旧向量记录。

### 解决方案

- 新增 `agents/rag/domain_summary_builder.py`，领域摘要改为基于 Redis/MySQL 语义模型生成，不再依赖 `schema_indexer`。
- `agents/api/app.py` 启动时调用新的领域摘要生成模块。
- `schema_indexer.index_mysql_schemas()` 默认禁用，只有显式设置 `ENABLE_LEGACY_SCHEMA_INDEX=1` 才会运行旧版 schema 向量索引。
- `agents/eval/dataset_generator.py` 改为从 MySQL `t_semantic_model` 生成评测数据，不再查询 Milvus `source=mysql_schema`。
- README 和技术设计文档标明 `schema_indexer` 是 legacy 兼容入口，当前 schema 权威来源是 MySQL/Redis。

### 影响

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| app 启动生成领域摘要 | 间接复用 `schema_indexer.generate_domain_summary` | 使用 `domain_summary_builder`，来源统一为语义模型 |
| 手工调用 `index_mysql_schemas()` | 默认写入 Milvus/ES `mysql_schema` | 默认返回 disabled，避免误建旧索引 |
| 评测数据生成 | 依赖 Milvus 旧 schema 文档 | 依赖 MySQL `t_semantic_model` |
| 架构说明 | schema_indexer 容易被理解为当前链路 | 明确为旧版兼容入口 |

---

## Iteration 26：评测报告与页面可视化

### 背景

生产环境需要用量化指标指导迭代，而不是只靠单条 case 观察。当前已有 `agents.eval` CLI，但报告偏命令行，缺少：

- 可视化展示
- per-query 回溯明细
- 业界常用指标的统一口径
- 首字延迟等端到端指标的扩展位置

### 解决方案

- `agents.eval.metrics` 新增 `accuracy@K`、`precision@K`，保留 `recall@K`、`MRR`、`NDCG@K`。
- 新增 `agents.eval.reporting`，统一报告 JSON 格式，包含：
  - `run_id`
  - `created_at`
  - `dataset_path`
  - 策略级指标
  - 平均/P50/P95 延迟
  - 首字延迟预留字段
  - per-query 明细
- `agents.eval.runner` 输出新的可回溯报告格式。
- 新增 `/api/eval/reports`、`/api/eval/reports/latest`。
- 前端新增 `Evaluation` tab，展示最佳策略、Accuracy@5、Recall@5、P95 延迟、策略对比和 query 明细。
- 新增 `docs/evaluation_design.md` 说明指标、报告格式、API、页面和后续端到端评测计划。

### 后续

低优先级可接入 LLM 对失败样本给优化建议，但建议只作为辅助分析，不自动改生产配置。

---

## Iteration 27：线上预选链路评测补齐

### 背景

Iteration 26 先补齐了报告格式和页面，但默认评测仍偏 schema metadata 基线。实际 NL2SQL 流程里，选表前还有两类关键输入：

- `business_knowledge_recall`：召回业务术语、公式、口径。
- `agent_knowledge_recall`：召回 SQL few-shot 示例。

这些证据会进入 `query_enhance`，再由 `select_tables` 选择表。单独评测本地 schema metadata 有价值，但不能完整回答“线上这条 query 经过证据召回和增强后，最终能不能选对表”。

### 解决方案

- 新增 `agents.eval.strategies`：
  - `run_preselect_pipeline(query)`：执行 `recall_evidence -> query_enhance -> select_tables`，输出 `schema_<table_name>`。
  - `run_business_knowledge_recall(query)`：只评测业务知识召回结果。
  - `run_agent_knowledge_recall(query)`：只评测 SQL few-shot 召回结果。
- `agents.eval.runner` 支持策略直接返回 `retrieved_doc_ids`，并继续复用统一指标计算。
- 默认评测策略扩展为：
  - `schema_lexical`
  - `schema_table_name`
  - `business_knowledge_recall`
  - `agent_knowledge_recall`
- `preselect_pipeline` 支持通过 `--include-online-pipeline` 显式开启。
- 对业务知识和 Agent 知识召回采用独立标注字段：
  - `relevant_business_doc_ids`
  - `relevant_agent_doc_ids`

### 设计取舍

`business_knowledge_recall` 和 `agent_knowledge_recall` 发生在选表之前，它们召回的是公式、业务定义和 few-shot 示例，不是 schema 表。因此 runner 不会拿它们和 `relevant_doc_ids` 强行对比；只有数据集显式包含对应标注字段时才计算指标，没有标注时报告显示 `num_queries = 0`。

`preselect_pipeline` 才是线上表选择链路评测：它消费前置证据和 query 增强结果，最后把 `select_tables` 产出的表名转为 `schema_<table_name>`，再和 `relevant_doc_ids` 对比。由于该链路会调用 LLM 节点，默认不启用，避免普通本地评测消耗 token；需要真实线上链路指标时显式加 `--include-online-pipeline`。

### 验证

新增单元测试覆盖：

- 预选链路按 `recall_evidence -> query_enhance -> select_tables` 顺序执行。
- 业务知识召回只使用 `relevant_business_doc_ids` 计算。
- Agent 知识召回只使用 `relevant_agent_doc_ids` 计算。
- 没有对应标注的数据集行会跳过，不污染指标。
- 默认 `run_evaluation` 报告包含新增策略。

本地验证命令：

```bash
pytest tests/test_eval_pipeline_strategies.py tests/test_eval_runner_schema.py tests/test_eval_metrics.py tests/test_eval_reporting.py tests/test_eval_dataset_generator.py -q
```

结果：`39 passed`。完整回归 `tests/test_eval_pipeline_strategies.py tests/test_eval_runner_schema.py tests/test_eval_metrics.py tests/test_eval_reporting.py tests/test_eval_dataset_generator.py tests/test_imports.py tests/test_api.py` 为 `76 passed`。

---

## Iteration 28：知识召回标注与 NL2SQL 离线端到端评测

### 背景

Iteration 27 已经能分别评测 schema、业务知识和 Agent few-shot 召回，但数据集默认只有 `relevant_doc_ids`，导致业务知识和 few-shot 策略经常显示 `num_queries = 0`。同时生产环境最终关心的不只是召回，还包括 SQL 是否规范、是否执行成功、执行结果是否符合预期、端到端延迟和首字延迟。

### 解决方案

- `generate` 默认基于本地知识表补充可选标注：
  - `relevant_business_doc_ids`
  - `relevant_agent_doc_ids`
- 知识标注不调用 LLM：
  - 业务知识用 `term` / `synonyms` 命中 query。
  - Agent few-shot 用 query 与 `question` / `description` / `category` 词法重叠匹配。
- 增加 `--no-knowledge-labels`，必要时只生成 schema 表标注。
- 新增 `agents.eval.nl2sql_runner` 和 CLI：
  - `python -m agents.eval.cli run-nl2sql --dataset ... --output ...`
- NL2SQL 离线报告包含：
  - `sql_valid`
  - `execution_success`
  - `result_exact_match`
  - P50/P95 延迟
  - 首字延迟
  - per-query 明细
- 前端 Evaluation 页面支持：
  - retrieval 报告按策略切换 query 明细。
  - NL2SQL 端到端报告独立展示核心指标。

### 设计取舍

`run-nl2sql` 默认只评测已记录样本，不调用 Agent，不执行数据库。这保证本地 TDD 和 CI 不依赖外部 LLM token、MySQL 数据状态或人工审批。后续接生产回放时，只需要把线上生成的 `generated_sql`、`actual_result`、`latency_ms`、`first_token_latency_ms` 写成 JSONL，即可复用同一套报告和页面。

### 验证

```bash
pytest -q
```

结果：`234 passed`。测试过程中 LangSmith 网络上报失败为本地网络限制，不影响测试结论。

---

## Iteration 29：评测报告历史回溯

### 背景

Evaluation 页面只能读取 latest 报告，不方便对比和回溯历史评测。生产迭代时需要能查看某一次 run 的完整报告，否则指标变化无法定位到具体数据集和策略结果。

### 解决方案

- `GET /api/eval/reports` 返回每个报告的 `name`，作为前端选择历史报告的稳定标识。
- 新增 `GET /api/eval/reports/{name}`，只允许读取已发现报告文件名，拒绝路径穿越。
- Evaluation 页面增加报告下拉菜单：
  - 默认展示最新报告。
  - 可切换历史 retrieval 或 NL2SQL 端到端报告。
  - 切换后保留原有策略明细和 NL2SQL 明细展示能力。

### 验证

新增 API 测试覆盖：

- 报告列表按更新时间返回，并包含 `name`。
- 可按报告文件名读取完整报告。
- 路径穿越请求返回 404。

完整回归：

```bash
pytest -q
```

结果：`236 passed`。LangSmith 网络上报失败为本地网络限制，不影响测试结论。

---

## Iteration 30：NL2SQL 离线评测 CLI 体验修复

### 背景

`run-nl2sql` 需要读取已经记录好的回放样本，但文档示例里的 `data/eval/nl2sql_cases.jsonl` 不一定存在。直接运行会触发 `FileNotFoundError` traceback，用户无法判断是命令错误、代码错误，还是缺少输入样本。

### 解决方案

- `run-nl2sql` 在 dataset 缺失时输出明确提示并以退出码 2 结束，不再展示 Python traceback。
- 新增 `--init-template`，可生成一份 JSONL 样本模板：

```bash
python -m agents.eval.cli run-nl2sql \
  --dataset data/eval/nl2sql_cases.jsonl \
  --init-template
```

- README 和评测设计文档改为先初始化模板，再填写真实 `generated_sql` / `actual_result` / `expected_result`，最后运行评测。

### 验证

- 缺失 dataset 场景返回可读错误和模板生成命令。
- `--init-template` 可生成 JSONL 模板。
- 使用模板可成功生成 `nl2sql_eval_report.json`。

---

## Iteration 31：链路追踪子调用细化

### 背景

LangSmith / CozeLoop 能看到 LangGraph 节点，但部分节点内部的 LLM 调用、Milvus 向量检索、Elasticsearch BM25、Redis/MySQL 元数据加载没有形成子 span。排查问题时只能看到节点耗时，无法判断时间花在 LLM、向量库、关键词检索还是 schema 元数据读取。

### 解决方案

- 新增 tracing helper：
  - `child_trace_config`：从 LangGraph config 继承 callbacks，传给内部 LLM/Runnable。
  - `traced_retriever_call`：把裸 Milvus/ES 检索包装成 retriever span。
  - `traced_tool_call` / `traced_async_tool_call`：把 Redis/MySQL 等非 Runnable IO 包装成 tool span。
- SQL React 细化：
  - `contextualize_query`
  - `recall_evidence`
  - `query_enhance`
  - `select_tables`
  - `sql_retrieve`
  - `sql_generate`
  - `error_analysis`
  - `result_reflection`
- RAG Chat / Dispatcher / Analyst 细化：
  - query rewrite、retrieve、chat LLM、intent classify、domain summary load、analyst report LLM。
- 业务知识和 Agent 知识召回细化：
  - Milvus vector search
  - ES BM25 search
  - MySQL lexical fallback

### 验证

新增/更新测试覆盖 callbacks 继承和手动 span 触发：

- 内部 LLM 调用会收到 graph callbacks。
- knowledge retriever 会收到 graph callbacks。
- `traced_retriever_call` 会触发 retriever start/end。
- `traced_tool_call` 会触发 tool start/end。

---

## Iteration 32：在线 NL2SQL 评测 Runner 与微调方案

### 背景

离线 `run-nl2sql` 只能评测已记录的 SQL 和结果，不会调用真实 Agent，也不会覆盖审批中断、SQL 执行、异常结果反思等线上链路。生产迭代需要能用同一批 query 回放当前 Agent，并把生成 SQL、执行结果、延迟和失败原因写成可回溯报告。

同时，后续准备微调 SQL 生成模型，需要明确模型选择、样本来源和数据飞轮，避免只拿公开 Text-to-SQL 数据训练，导致财务口径和项目 schema 不稳定。

### 解决方案

- 新增 `agents.eval.online_nl2sql_runner`：
  - `run_online_nl2sql_case`：单条 query 真实调用 LangGraph Agent。
  - `run_online_nl2sql_evaluation_async`：批量回放 JSONL 数据集并写报告。
  - `write_online_nl2sql_template`：生成在线评测模板。
- 新增 CLI：

```bash
python -m agents.eval.cli run-online-nl2sql \
  --dataset data/eval/online_nl2sql_cases.jsonl \
  --output data/eval/online_nl2sql_eval_report.json
```

- 默认停在 SQL 审批中断，只记录生成 SQL 和首次响应延迟。
- 增加 `--auto-approve-sql`，可在测试库中自动恢复审批中断并执行 SQL，覆盖结果反思后的二次 SQL。
- 增加 `--full-dispatch`，可选择是否把意图分类也纳入评测。
- 新增 `docs/evaluation_user_guide.md`，整理已完成评测能力的使用手册。
- 新增 `docs/sql_finetuning_plan.md`：
  - 推荐 `Qwen/Qwen2.5-Coder-7B-Instruct` 作为 7B code 基座。
  - 明确公开数据只做泛化补充，主数据来自本项目 schema、业务知识、few-shot、线上日志、失败修正和人工黄金集。

### 验证

- 在线 Runner 单测覆盖：
  - 自动审批后可进入执行结果。
  - 不自动审批时停在 `pending_approval`。
  - 批量评测会写 `online_nl2sql_end_to_end` 报告。
  - 模板命令可生成 JSONL。

---

## Iteration 33：修复跨轮 Query 被 Checkpoint 旧状态污染

### 背景

同一前端会话连续提问时，`thread_id` 直接使用 `session_id`，LangGraph checkpoint 会把上一轮 SQL 图状态带到下一轮。例如用户新问“第一季度员工工资”，`classify_intent.llm` 已返回：

```json
{"intent": "sql_query", "rewritten_query": "我们公司第一季度的员工工资情况"}
```

但进入 `route_intent` / `sql_react` 时，状态里的 `query` 仍可能是上一轮“我们公司去年亏损”，导致后续 SQL 全部沿用旧问题。

### 原因

- `FinalGraphState.query` 使用的 reducer 是“已有就保留”，新一轮输入无法覆盖旧 checkpoint query。
- `FinalGraphState` 没声明 `rewritten_query`，前端预分类传入的 rewritten query 可能无法稳定进入主图状态。
- graph checkpoint 以会话为粒度复用，上一轮 `sql`、`result`、`answer` 等状态也存在污染下一轮的风险。

### 解决方案

- 将 query reducer 改为 `latest_non_empty`：新输入覆盖旧值，approve resume 没有新值时保留当前值。
- `SQLReactState.rewritten_query` 和 `FinalGraphState.rewritten_query` 使用同样 reducer，保证预分类结果可传入子图。
- `/api/query/invoke` 每个新 query 生成独立 graph thread id：

```text
{session_id}:turn:{uuid}
```

- SQL 审批中断时暂存 `_pending_thread_id`，`approve` 使用该 thread 恢复图执行。
- 聊天历史仍然通过 session store 维护，不再依赖 LangGraph checkpoint 跨轮保存业务状态。

### 验证

- 新增 dispatcher 回归：同一 `thread_id` 连续两次输入不同 query，第二次进入 SQL 子图时必须使用新 query 和新 rewritten query。
- 新增 API 回归：`invoke` 使用单轮 graph thread，`approve` 使用 pending graph thread。
- 完整回归：

```bash
.venv/bin/python -m pytest -q
```

结果：`259 passed`。

---

## Iteration 34：公司财务查数意图防误判

> 后续已由 Iteration 36 替换为“数据库可配置规则 + LLM + 仲裁”的通用方案，避免把业务关键词写死在代码中。

### 背景

用户问“去年亏损”时，`classify_intent.llm` 在有历史对话干扰的情况下返回：

```json
{"intent": "chat", "rewritten_query": "我们公司去年是否亏损"}
```

但该问题属于当前企业财务数据库可回答的结构化查数问题，应该进入 `sql_query`。历史中曾出现“参考知识中未提供...”这类 RAG 回答，会误导 LLM 把本公司财务数据问题当成普通 chat/knowledge。

### 解决方案

- 在 `classify_intent` 解析 LLM 输出后增加 deterministic guard：
  - 当前 query 或 rewritten query 命中本公司/企业财务数据特征时，强制归为 `sql_query`。
  - 覆盖关键词包括亏损、盈利、利润、收入、成本、费用、工资、薪酬、余额、发生额、预算、报销、发票、应收应付、凭证、资产、折旧等。
  - 时间口径包括去年、今年、季度、本月、上月、具体年份/期间等。
- 明确排除外部公开公司问题，例如“贵州茅台去年亏损情况”，避免把公开知识问题强行路由到本地 SQL。

### 验证

- 新增回归：LLM 返回 `chat` 且 rewritten query 为“我们公司去年是否亏损”时，最终 intent 必须是 `sql_query`。
- 新增回归：外部公司“去年贵州茅台的亏损情况”仍可保持 `chat`。
- 完整回归：

```bash
.venv/bin/python -m pytest -q
```

结果：`262 passed`。

---

## Iteration 35：审批后 SQL 执行失败自动修复

### 背景

用户审批 SQL 后，执行阶段可能出现由 LLM 生成 SQL 导致的错误，例如：

```text
Error: Unknown column 'a.account_type' in 'field list'
```

这类错误通常是 SQL 作用域、字段别名、子查询外层引用内层表别名、嵌套聚合等生成问题，应该进入自动修复流程，而不是直接把执行失败返回给前端。

### 解决方案

- 新增 `_should_repair_sql_error`：
  - `Unknown column`
  - `SQL syntax`
  - `Invalid use of group function`
  - `42S02` / `42S22` / `42000`
  - `1054` / `1064` / `1111`
  - `ambiguous`、`GROUP BY`、子查询返回列数等常见 SQL 生成错误
- 权限、认证、密码、连接等不可由 SQL 改写修复的错误不进入 LLM 修复。
- `route_after_execute` 遇到可修复 SQL 错误时进入：

```text
execute_sql -> error_analysis -> sql_generate -> safety_check -> approve
```

- `sql_generate` prompt 增加约束：
  - 不要在同一层 SELECT 中嵌套聚合函数。
  - 外层查询不能引用内层表别名，只能引用子查询输出列。
- 二次审批文案改为：

```text
上次 SQL 执行失败，系统已分析错误并生成修正后的 SQL。请确认是否执行修正后的 SQL？
```

- SSE 进度文案同步覆盖“执行失败或结果异常”两类自动修复。

### 验证

- `Unknown column 'a.account_type' in 'field list'` 被判定为可修复。
- 权限错误不会进入 LLM SQL 修复。
- 修复后 SQL 的 approve interrupt 使用用户友好的执行失败修正文案。

---

## Iteration 36：意图规则配置化与 LLM 仲裁

### 背景

Iteration 34 为了解决“去年亏损”被历史 RAG 回答误导成 `chat`，在 `dispatcher.py` 中加入了本公司、时间、财务指标、外部公司等关键词 guard。该方式能解决单个问题，但会带来两个明显风险：

- 业务词、主体词和时间词写在代码里，后续换查询会不断增加 hardcode。
- 外部主体问题可能被“时间 + 财务词”误判为本地 SQL 查询。

### 解决方案

- 移除 `dispatcher.py` 中的业务关键词常量和 deterministic keyword guard。
- 新增 `t_intent_rule`，规则字段包括 `target_intent`、`match_type`、`pattern`、`rewrite_template`、`priority`、`confidence`、`enabled`。
- 新增 `agents.tool.storage.intent_rules`：
  - 只负责规则表 DDL、CRUD、匹配算法。
  - 不内置任何业务关键词或公开公司名单。
  - MySQL 不可用时返回空规则，不影响 LLM 意图识别。
- 新增 `data/intent_rules_seed.json` 和 `scripts.seed_intent_rules`，用于把默认规则作为数据写入 MySQL，而不是写在 `dispatcher.py`。
- `classify_intent` 改为并行执行：

```text
用户问题
  ├─ LLM 意图识别 + 查询重写
  └─ 数据库规则引擎匹配
        ↓
      仲裁器
        ↓
  intent + rewritten_query
```

- 仲裁策略：
  - 没有规则命中时，使用 LLM 意图。
  - 有规则命中且目标意图合法时，使用规则意图。
  - 有规则命中且配置了 `rewrite_template` 时，用数据里的模板补齐查询主体，例如把“第一季度毛利率”补齐为“公司第一季度毛利率”。
  - 规则内容由 Admin 页面维护，而不是写入代码。
- Admin 页面新增意图规则维护入口，可新增、编辑、删除、启停规则和维护重写模板。
- 移除“只有 intent 没有 rewritten_query 时兼容旧版跳过分类”的分支；只有同时传入 `intent` 和 `rewritten_query` 才视为前端已完成预分类，避免旧状态或旧客户端把后续问题强行路由到错误意图。

### 验证

- 新增回归：无规则命中时，LLM 返回 `chat` 的问题保持 `chat`。
- 新增回归：规则引擎可通过数据库规则把 LLM 的 `chat` 仲裁为 `sql_query`。
- 新增回归：规则引擎可通过 `rewrite_template` 把“第一季度毛利率”重写为“公司第一季度毛利率”。
- 新增回归：只有 `intent` 没有 `rewritten_query` 时必须重新走当前轮 LLM 分类。
- 新增 API 回归：`/api/admin/intent-rules` 支持列表和保存。
- 局部回归：

```bash
.venv/bin/python -m pytest tests/test_dispatcher.py tests/test_intent_rules.py tests/test_api.py -q
```

结果：`19 passed`。

---

## Iteration 37：分层记忆体系细化

### 背景

原记忆实现把历史、摘要、SQL 上下文都放在 session 里，读取时容易把过多旧消息直接塞给 LLM。这样有三个问题：

- 短期上下文太粗，旧问题容易干扰当前意图识别。
- 中期摘要只在 RAG Chat 后台压缩，SQL Query 路径追加历史后不会触发摘要。
- 长期历史没有进入向量库，压缩掉的对话无法按语义回溯。

### 解决方案

- 短期记忆：`_load_chat_history` 只注入最近 `MEMORY_SHORT_WINDOW_MESSAGES` 条消息，SQL 上下文和摘要仍以 system message 形式置顶。
- 中期记忆：新增 `agents.tool.memory.manager`，统一维护 session 记忆；SQL Query 与 RAG Chat 完成后都会异步触发压缩。
- 长期记忆：新增 `agents.tool.memory.vector_store`，把被压缩归档的旧消息写入 Milvus，`source=conversation_memory`，并按 `session_id` 隔离。
- 召回策略：只有 session 标记了 `_has_long_term_memory` 时才按当前 query 检索长期记忆，避免新会话每轮都访问向量库。
- 配置项：
  - `MEMORY_SHORT_WINDOW_MESSAGES`
  - `MEMORY_SUMMARY_MAX_HISTORY_LEN`
  - `MEMORY_SUMMARY_KEEP_RECENT`
  - `MEMORY_LONG_TERM_TOP_K`
  - `MEMORY_ENABLE_LONG_TERM_VECTOR`

### 验证

- 新增回归：加载 chat history 时只保留短期滑动窗口。
- 新增回归：`compress_session` 返回被归档消息，便于长期记忆索引。
- 新增回归：memory manager 压缩后会把归档消息写入长期向量记忆接口。
- 新增回归：存在长期记忆标记时，加载 chat history 会按 query 注入 `[长期记忆]`。

```bash
.venv/bin/python -m pytest tests/test_memory.py tests/test_final_api.py::TestSessionSqlContext -q
```

结果：`13 passed`。扩展回归：

```bash
.venv/bin/python -m pytest tests/test_memory.py tests/test_final_api.py tests/test_rag_flow.py tests/test_token_counter.py tests/test_imports.py tests/test_api.py -q
```

结果：`84 passed`。

---

## Iteration 38：管理表选表召回与逻辑外键补全

### 背景

管理类查询（用户、角色、部门、用户角色绑定、用户部门归属）在 `select_tables` 阶段长期排不进 Top5。典型问题包括：

- “查询所有用户的真实姓名以及他们被分配的角色名称”需要 `t_user + t_user_role + t_role`。
- “查询公司各部门的负责人姓名以及对应的部门名称”需要 `t_department + t_user_department + t_user`。
- “查询所有拥有财务审核角色的用户分别属于哪个部门”需要 `t_user_role + t_role + t_user + t_user_department + t_department`。

优化前管理表专项评测表现为：管理表能在 Top10 附近出现，但经常被财务核心表挤到 6-10 位，导致 `Recall@5 = 0%`，`Recall@10 = 96.67%`。这说明问题不是“完全找不到表”，而是排序和关联补表链路不稳定。

### 根因

1. **业务知识 evidence 污染选表**
   `business_knowledge` 中的 `related_tables` 会无条件并入候选结果。即使当前 query 没命中该业务术语，也可能把财务表推到管理表前面。

2. **LLM 输出顺序被直接信任**
   `select_tables` 让 LLM 从全量表名/表描述中精选表，但后续没有用本地语义模型做稳定重排。管理表虽然被选中，仍可能排在 Top5 之外。

3. **桥接表和端点表缺失**
   多表查询经常需要关系表，例如 `t_user_department`、`t_user_role`。如果 LLM 只选了两个端点表，SQL 生成阶段就缺少 JOIN 桥接表；如果 LLM 只选了绑定表，又可能缺少被引用的端点实体表。

4. **表关系只读物理外键**
   `get_table_relationships` 只查 `information_schema.key_column_usage`。当前业务库很多关系是逻辑外键，已维护在 `t_semantic_model.is_fk/ref_table/ref_column` 中，但没有注入给 SQL 生成。

5. **管理表缺少“表级可见语义”**
   财务核心表天然有更丰富的字段名、业务名和描述，LLM 更容易判断；管理表如果只有英文表名或弱表注释，就会输给财务表。

### 解决方案

#### 1. 管理表语义补齐

在 `scripts.seed_semantic_model` 中补充管理表的表级描述和字段级业务语义：

- `t_user`：用户/员工账号信息，真实姓名、联系电话、邮箱、注册时间、账号状态。
- `t_role`：系统角色信息，角色名称、角色编码、角色状态。
- `t_user_role`：用户角色绑定关系。
- `t_department`：组织部门信息，部门名称、上级部门、部门负责人、联系电话、状态。
- `t_user_department`：用户部门归属关系，是否部门负责人。

同时补充逻辑外键：

- `t_user_role.user_id -> t_user.id`
- `t_user_role.role_id -> t_role.id`
- `t_user_department.user_id -> t_user.id`
- `t_user_department.department_id -> t_department.id`
- `t_department.parent_id -> t_department.id`
- `t_cost_center.department_id -> t_department.id`
- `t_expense_claim.department_id -> t_department.id`

seed 脚本会清理 Redis `schema:table_metadata` 缓存，避免旧表注释继续影响 `select_tables`。

#### 2. 业务 evidence 改为 query-aware

`_related_tables_from_business_evidence` 不再无条件使用 `related_tables`。只有当前 query 命中 evidence 的 `term` 或 `synonyms` 时，才把对应关联表合并进选表结果。

这样可以保留“净利润/亏损/预算”等业务知识对财务查询的帮助，同时避免未命中的财务术语把管理表 query 污染成财务表优先。

#### 3. select_tables 增加本地语义重排

`select_tables` 在 LLM 精选后加载候选表的 `t_semantic_model`，基于以下信息做本地重排：

- 表名
- 表注释
- 字段名
- 字段注释
- `business_name`
- `synonyms`
- `business_description`

匹配策略不是写死“用户/角色/部门”等业务关键词，而是读取数据库中的语义模型。短语命中（例如“真实姓名”“角色名称”“部门负责人”）权重高于孤立字符重叠。

#### 4. 基于语义外键补桥接表和端点表

新增 `_expand_selected_tables_by_semantic_relationships`：

- 如果已选关系表，则补齐其引用的端点表。
- 如果两个已选端点表被某个未选桥接表同时引用，则补齐桥接表。

示例：

```text
t_department + t_user
  -> 根据 t_user_department.department_id/user_id
  -> 自动补 t_user_department

t_user_role + t_user_department + t_role
  -> 根据逻辑外键
  -> 自动补 t_user、t_department
```

该逻辑仍然是数据驱动的：只依赖 `t_semantic_model` 中的 `is_fk/ref_table/ref_column`，不是运行时代码硬编码业务表名。

#### 5. 表关系读取合并逻辑外键

`get_table_relationships` 现在合并两类来源：

- 物理外键：`information_schema.key_column_usage`
- 逻辑外键：`t_semantic_model.is_fk/ref_table/ref_column`

如果逻辑外键查询失败，会降级保留已查到的物理外键，避免老库或不完整环境直接丢失关系信息。

#### 6. online 评测初始化模型注册

`preselect_pipeline` 是线上选表前置链路评测，会真实执行：

```text
recall_evidence -> query_enhance -> select_tables
```

该链路会调用 embedding、ES/Milvus、`query_enhance` LLM 和 `select_tables` LLM。评测入口补充 `init_chat_models()`，避免直接运行 CLI 时模型注册未初始化。

### 评测结果

使用管理表专项数据集：

```bash
.venv/bin/python -m agents.eval.cli run \
  --dataset data/eval/management_eval_dataset.jsonl \
  --output data/eval/management_preselect_report.json \
  --include-online-pipeline
```

`preselect_pipeline` 线上链路结果：

| 指标 | 结果 |
| --- | ---: |
| 样本数 | 12 |
| MRR | 1.0000 |
| Accuracy@5 | 1.0000 |
| Recall@5 | 1.0000 |
| NDCG@5 | 1.0000 |
| 平均延迟 | 8389.2 ms |
| P50 延迟 | 6575.3 ms |
| P95 延迟 | 10593.5 ms |

典型 Top5 结果：

| Query | Top5 召回 |
| --- | --- |
| 查询所有用户的真实姓名以及他们被分配的角色名称 | `t_role, t_user, t_user_role` |
| 查询公司各部门的负责人姓名以及对应的部门名称 | `t_department, t_user_department, t_user` |
| 查询所有拥有财务审核角色的用户分别属于哪个部门 | `t_user_role, t_department, t_user_department, t_role, t_user` |

该 `Recall@5=100%` 是管理表专项评测结果，不能直接等同于全量业务评测结果。随后使用全量 `eval_dataset.jsonl` 重跑线上预选链路：

```bash
.venv/bin/python -m agents.eval.cli run \
  --dataset data/eval/eval_dataset.jsonl \
  --output data/eval/eval_report.json \
  --include-online-pipeline
```

全量评测结果（45 条，2026-05-13）：

| 策略 | MRR | Accuracy@5 | Precision@5 | Recall@5 | NDCG@5 | 平均延迟 | P50 延迟 | P95 延迟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `schema_lexical` | 96.67% | 77.78% | 35.56% | 90.63% | 90.03% | 0.1 ms | 0.0 ms | 0.1 ms |
| `preselect_pipeline` | 96.67% | 88.89% | 84.93% | 94.07% | 94.13% | 7545.6 ms | 7426.8 ms | 10541.4 ms |

结论：管理表专项已经修复到 Top5 全召回；全量线上预选链路 `Recall@5` 从旧报告的 67.59% 提升到 94.07%，但还未超过 95%。剩余缺口需要继续从失败明细中定位是标注粒度、query_enhance 扩表，还是 `select_tables` 过度/不足补表。

### 验证

新增/更新回归：

- 未命中的业务 knowledge 不再把财务表塞进管理表 query。
- 本地语义重排能把管理表排到 LLM 原始输出前面。
- 已选端点表能通过语义外键补桥接表。
- 已选关系表能补齐被引用端点表。
- `get_table_relationships` 能合并物理外键与 `t_semantic_model` 逻辑外键。
- 逻辑外键查询失败时保留物理外键降级。
- 兼容 `pymysql.fetchall()` 返回 tuple 的情况，避免逻辑外键合并时报 `'tuple' object has no attribute 'extend'`。
- `preselect_pipeline` CLI 运行前初始化 chat model registry。

局部回归：

```bash
.venv/bin/python -m pytest \
  tests/test_seed_semantic_model.py \
  tests/test_sql_react.py::TestSelectTables \
  tests/test_eval_pipeline_strategies.py \
  tests/test_retriever_relationships.py
```

结果：`17 passed`。

格式检查：

```bash
git diff --check
```

结果：通过。

## Iteration 39：复杂多表查询的单 SQL 与计划模式切换

### 背景

当前 `select_tables` 默认围绕 Top5 表做评测和选表，逻辑外键补表后可能超过 5 张。这个默认值对大多数财务 NL2SQL 是合理的：

- “去年亏损”：通常需要 `t_journal_entry + t_journal_item + t_account`。
- “第一季度员工工资”：通常需要凭证主表、分录表、会计科目，必要时补部门/成本中心。
- “用户角色部门”：通常需要 `t_user + t_role + t_user_role + t_department + t_user_department`。

问题在于：如果逻辑外键补表后超过 8 张表，继续把所有 schema 直接塞给单个 `sql_generate` 节点，会带来三个风险：

1. **上下文膨胀**：字段、关系和业务知识会占用大量 token，压缩 SQL 生成可用上下文。
2. **JOIN 幻觉**：LLM 更容易生成错误 JOIN、重复 JOIN 或漏掉关键过滤条件。
3. **业务目标不清**：很多超过 8 张表的问题，本质上不是一条明细 SQL，而是多个指标、多个业务域的分析任务。

早期想法是“超过 8 张表就提示用户缩小范围或拆成多步查询”。这个策略安全，但过于保守，会把一部分真实复杂分析需求挡掉。

### DataAgent 参考

DataAgent 的处理不是简单扩大单条 SQL 的表数量，也不是超过阈值直接拒绝：

- `TableRelationNode` 先基于 query/evidence 构建初始 schema，用 LLM 精选相关表，再通过外键/逻辑外键找到缺失表，并加载语义模型。
- `PlannerNode` 基于 evidence、schema 和语义模型生成执行计划。
- `PlanExecutorNode` 按计划逐步路由到 SQL、Python、Report 等节点。
- SQL/Python 执行后再回到 step select，最终生成报告。

这说明复杂问题应该进入 **Planner 多步执行模式**，但不等同于“LLM 自动并行拆很多 SQL”。是否拆分、怎么合并、是否需要用户确认，都需要计划层治理。

### 方案取舍

#### 方案 A：超过 8 张表直接提示用户缩小范围

优点：

- 实现简单，风险低。
- 不会生成非常复杂、不可读、不可审计的 SQL。
- 适合明细查询和权限敏感场景。

缺点：

- 对真实复杂分析问题不友好。
- 用户明明希望系统完成分析，却被迫手动拆问题。
- 不能充分发挥 Agent 的计划和工具编排能力。

#### 方案 B：LLM 拆分任务，多步查询后综合

优点：

- 能处理多指标、多业务域、多阶段分析。
- 每个子 SQL 可控制在 3-5 张表，降低单 SQL 复杂度。
- 可引入 Python/本地代码做合并、同比环比、排序、异常归因。

缺点：

- 需要计划校验、依赖管理、结果合并和失败恢复。
- 并行拆分不一定安全；有些步骤必须串行。
- 如果没有稳定 join key 或业务口径，LLM 硬拆会产生错误结论。

### 推荐策略：8 张表作为“计划模式切换阈值”

不把 8 张表作为拒绝阈值，而作为从 **单 SQL 模式** 切换到 **Complex Query Planner 模式** 的阈值。

```text
候选召回层：Top10-20，用于保证召回，不直接全部给 SQL 生成
LLM 主动选表：默认最多 5 张主表
逻辑外键补表后：允许最多 8 张表进入单 SQL
超过 8 张表：进入 Complex Query Planner，先判断是否可拆
```

#### 单 SQL 模式

适用条件：

- 最终表数不超过 8 张。
- 查询目标明确，能用一条聚合或明细 SQL 回答。
- JOIN 路径能由物理外键或逻辑外键解释。
- 不需要多个业务域结果再二次合并。

处理方式：

```text
select_tables -> sql_retrieve -> sql_generate -> safety_check -> approve -> execute_sql
```

#### Complex Query Planner 模式

适用条件：

- 逻辑外键补表后超过 8 张。
- 用户问题包含多个指标、多个业务域或分析型诉求。
- 每个子任务可拆成 3-5 张表的独立查询。
- 子任务之间有清晰的公共维度，例如期间、部门、项目、客户、供应商。
- 最终目标是汇总、对比、分析、报告，而不是一张强一致明细表。

Planner 需要输出结构化计划：

```json
{
  "mode": "complex_plan",
  "reason": "涉及多个业务域和超过 8 张表，拆分为多个可执行 SQL 子任务",
  "steps": [
    {
      "step": 1,
      "type": "sql",
      "goal": "查询 2025 年各月收入和成本费用",
      "tables": ["t_journal_entry", "t_journal_item", "t_account"],
      "depends_on": [],
      "merge_keys": ["period"]
    },
    {
      "step": 2,
      "type": "sql",
      "goal": "查询 2025 年各部门预算执行情况",
      "tables": ["t_budget", "t_cost_center", "t_department"],
      "depends_on": [],
      "merge_keys": ["period", "department_id"]
    },
    {
      "step": 3,
      "type": "python_merge",
      "goal": "按期间和部门合并结果，生成综合分析",
      "depends_on": [1, 2],
      "merge_keys": ["period", "department_id"]
    }
  ],
  "requires_user_confirmation": true
}
```

#### 需要澄清或拒绝自动拆分的场景

以下场景不应该直接让 LLM 自动拆分执行：

- 用户要的是强一致明细表，必须跨很多表精确 JOIN。
- 子任务之间没有稳定 join key。
- 用户问题过宽泛，例如“分析公司所有经营情况”。
- 涉及工资、个人信息、银行账号等敏感字段。
- 需要事务一致性或同一时间点快照，但系统无法保证。
- Planner 无法说明每一步的表、目标、依赖和合并键。

这类请求应该返回澄清问题，或者先给前端一个计划预览，让用户确认范围后再执行。

### 迭代开发拆分

#### Task 1：复杂度评分与路由决策

目标：在 `select_tables` 之后增加复杂度判定，不改变现有 SQL 生成行为。

输出：

- `selected_tables_count`
- `relationship_count`
- `estimated_join_count`
- `query_intent_complexity`
- `route_mode`: `single_sql | complex_plan | clarify`

路由原则：

```text
selected_tables <= 5: single_sql
6 <= selected_tables <= 8: single_sql_with_strict_checks
selected_tables > 8: 进入复杂查询路由仲裁
```

复杂查询路由仲裁不能在 Python 代码里硬编码“分析、明细、敏感字段”等业务关键词。可执行方案是：

1. **结构信号走本地代码**：表数、关系数、估算 JOIN 数、是否有逻辑外键路径，这些是稳定工程规则。
2. **语义信号走配置或模型**：分析型、报表型、明细导出型、敏感数据型等判断由数据库规则表或 LLM 仲裁给出，不把关键词写死在代码里。
3. **默认保守**：超过 8 张表但无法确认可拆分、无法给出公共合并维度时，返回 `clarify`，要求用户缩小范围。

验证：

- 财务核心三表查询仍走 `single_sql`。
- 用户/角色/部门五表查询仍走 `single_sql`。
- 超过 8 张且规则/LLM 仲裁为可拆分析问题时进入 `complex_plan`。
- 超过 8 张且规则/LLM 仲裁为明细导出、敏感数据或不可稳定合并时进入 `clarify`。

#### Task 2：Complex Query Planner 节点

目标：新增计划节点，只生成结构化计划，不执行 SQL。

计划必须包含：

- 每步 `type`: `sql | python_merge | report`
- 每步 `goal`
- 每步 `tables`
- 每步 `depends_on`
- 每步 `merge_keys`
- 是否需要用户确认

约束：

- SQL 子任务表数建议不超过 5。
- 无法给出 merge key 时必须转 `clarify`。
- 涉及敏感字段时必须转 `clarify` 或要求权限确认。

#### Task 3：计划校验与用户确认

目标：防止 LLM 生成不可执行或不可审计的计划。

校验规则：

- 所有表必须来自当前 schema 候选。
- `depends_on` 必须引用已存在步骤。
- `python_merge` 必须至少依赖一个 SQL 步骤。
- 有多个 SQL 步骤需要合并时必须提供 `merge_keys`。
- 步骤数设置上限，例如 5 步。

前端/SSE 展示：

```text
检测到这是复杂多表分析问题，系统已拆分为 3 个步骤：
1. 查询收入和成本费用
2. 查询预算执行情况
3. 按期间和部门合并分析
请确认是否按该计划执行？
```

#### Task 4：多 SQL 执行与结果聚合

目标：复用现有 `sql_generate -> safety_check -> approve -> execute_sql`，逐步执行计划。

执行策略：

- 无依赖 SQL 步骤可以并行，但初期建议先串行，降低状态复杂度。
- 每个 SQL 仍必须 safety_check 和 approve。
- 每步结果写入 `plan_execution_results[step_id]`。
- `python_merge` 或本地聚合节点只消费已完成步骤结果。

#### Task 5：评测与回归

新增复杂查询专项数据集：

- 3-5 张表单 SQL 样本。
- 6-8 张表单 SQL 样本。
- 超过 8 张且可拆分析样本。
- 超过 8 张但应该澄清的明细/敏感样本。

指标：

- route accuracy：路由模式是否正确。
- plan validity：计划结构是否合法。
- step success rate：每步执行成功率。
- final answer correctness：最终回答是否符合预期。
- latency P50/P95：复杂计划整体耗时。

### 当前结论

Top5 仍然适合作为默认选表和评测指标，因为绝大多数 NL2SQL 问题不应依赖很多表。

8 张表不应作为拒绝阈值，而应作为从单 SQL 模式切换到复杂计划模式的阈值。

复杂多表查询的关键不是“给 LLM 更多表”，而是“让 Agent 先规划、再分步执行、最后可审计地合并结果”。

### 开发计划链接

详细 TDD 拆分见：[Complex Query Planner Implementation Plan](superpowers/plans/2026-05-14-complex-query-planner.md)。

## Iteration 40：轻量表路由画像与链式补表优化

### 背景

全量线上预选链路中，“查询各个部门的年度预算总金额”暴露出两个问题：

- `select_tables` prompt 只给表名和表注释，LLM 看到“预算”后容易只选 `t_budget`，看不到 `t_budget.cost_center_id` 同义词“部门”、`t_cost_center.annual_budget` 业务名“年度预算”等字段级证据。
- 语义外键补表是一轮扫描，`t_budget -> t_cost_center -> t_department` 这类多跳关系可能受字典顺序或缓存状态影响，补表不稳定。

同时不能把完整 schema/字段列表全部放进 `select_tables`，否则会把 SQL 生成阶段的 token 压力提前到选表阶段。

### 方案

- 新增轻量 table routing profile：每张候选表只展示表说明，以及与当前 query 命中的少量字段提示。
- 字段提示由 `t_semantic_model` 的 `business_name / column_comment / synonyms / ref_table` 动态计算，不在代码里写业务关键词。
- 字段提示只用于选表；完整字段类型和完整 schema 仍然只在 `sql_retrieve -> sql_generate` 阶段按已选表加载。
- 语义外键补表改成闭包式扩展，并结合 query 相关性判断，避免只补一跳或过度补无关外键。
- `scripts.seed_semantic_model` 清理 Redis `schema:table_metadata` 与 `schema:semantic_model:*` 缓存，避免逻辑外键和表注释更新后线上仍读旧缓存。

### 实际案例

Query：

```text
查询各个部门的年度预算总金额
```

优化前 `select_tables` 只给：

```text
- t_budget: 预算管理表
- t_cost_center: 成本中心表
- t_department: 组织部门信息表，包含部门名称
```

优化后会给轻量画像：

```text
- t_budget: 预算管理表 | 匹配字段: budget_amount(预算金额/预算额度), budget_year(预算年度), cost_center_id(成本中心ID/部门/-> t_cost_center.id)
- t_cost_center: 成本中心表 | 匹配字段: annual_budget(年度预算/全年预算), department_id(关联部门ID/-> t_department.id), center_code(成本中心编码/部门编码)
- t_department: 组织部门信息表，包含部门名称 | 匹配字段: name(部门名称/组织名称/部门), manager(部门负责人/负责人/部门经理)
```

如果 LLM 仍然只返回 `t_budget`，链式补表会继续补出：

```text
t_budget -> t_cost_center -> t_department
```

这样 Top5 不再依赖 LLM 一次性选全维表。

### 验证

- 新增测试覆盖链式外键补表不依赖字典顺序。
- 新增测试覆盖 `select_tables` 只暴露命中的轻量字段画像，不包含无关字段如 `created_at`。
- 本地评测样本 `data/eval/eval_dataset.jsonl` 中该 case 的人工标注已补充 `schema_t_budget`，因为“年度预算总金额”本身需要预算事实表或明确的预算主数据口径。

## Iteration 41：单次召回的 recall_context 与选表语义复用

### 背景

Iteration 40 中轻量字段画像仍然依赖 `_ranking_terms()` 的中文 n-gram 词面匹配。它能解决“部门/年度预算”这类字面命中问题，但会产生噪声片段，例如“的年”“门的年”，也不能理解“人员成本”和“工资薪酬”这类同义表达。

系统已经有 `recall_evidence` 节点，并且它位于 `select_tables` 之前：

```text
recall_evidence -> query_enhance -> select_tables
```

因此不应在 `_ranking_terms()` 或 `select_tables` 内重复召回。正确做法是：每轮 query 只执行一次召回，把结构化结果放入 LangGraph state/checkpoint，后续节点全部复用。

### 方案

`recall_evidence` 节点升级为“召回 + 结构化整理”：

```python
{
  "evidence": [...],
  "few_shot_examples": [...],
  "recall_context": {
    "query_key": rewritten_query,
    "business_evidence": [...],
    "few_shot_examples": [...],
    "business_related_tables": ["t_budget", "t_cost_center"],
    "few_shot_related_tables": ["t_budget", "t_cost_center"],
    "matched_terms": ["年度预算", "部门费用"],
    "few_shot_questions": ["查询各部门年度预算总额"]
  }
}
```

约束：

- 每个用户问题只调用一次 `recall_evidence`。
- `query_enhance`、`select_tables`、`sql_generate` 只读 state 中的 `recall_context`，不再触发召回。
- `recall_context.query_key` 必须等于当前 `rewritten_query/query`，否则视为旧 checkpoint 污染并忽略。
- `select_tables` 使用 `business_related_tables` 和 `few_shot_related_tables` 给候选表加权，并把这些表合并进候选结果。
- 字段打分优先使用 `recall_context.matched_terms` 这类业务词，再 fallback 到 `_ranking_terms()`，降低 n-gram 噪声。

### 实施结果

- 召回链路只跑一次，降低重复 IO 和外部检索开销。
- select-table 阶段能使用业务知识和 few-shot 的表证据，不再只靠表注释和字段 n-gram。
- checkpoint/resume 后仍可复用同一份召回结果，同时通过 `query_key` 避免跨轮污染。
- LangSmith/Codzloop 中能看到召回证据如何影响选表，链路更可解释。

本次实现不再兼容旧的“`select_tables` 从原始 `evidence` 字符串里临时解析相关表”的路径。选表阶段的语义来源统一为 `recall_context`：

- `recall_evidence` 是唯一召回节点。
- `select_tables` 只读取 `recall_context`，不再触发召回。
- `recall_context.query_key` 不匹配当前 `rewritten_query/query` 时直接忽略，避免 checkpoint 污染。
- 原始 `evidence` 和 `few_shot_examples` 继续保留给 `query_enhance/sql_generate` prompt 使用，但不作为选表语义加权的兼容入口。

### TDD 验证

1. 新增 `recall_context` 状态字段。
2. 测试 `recall_evidence` 产出结构化 `recall_context`，并从业务知识和 few-shot 中抽取相关表。
3. 测试 `select_tables` 在已有 `recall_context` 时不会调用召回，只复用相关表和 matched terms。
4. 测试 `query_key` 不一致时忽略旧 `recall_context`。
5. 复测第 12 条“查询各个部门的年度预算总金额”的 Top5 召回。

已执行的回归测试：

```bash
.venv/bin/python -m pytest tests/test_sql_react.py -q
# 70 passed

.venv/bin/python -m pytest tests/test_sql_react.py tests/test_seed_semantic_model.py tests/test_eval_dataset_generator.py -q
# 79 passed
```

重点样本复测：

```text
Query: 查询各个部门的年度预算总金额
recall_context:
  business_related_tables: [t_cost_center]
  few_shot_related_tables: [t_budget, t_cost_center]
  matched_terms: [年度预算, 部门]
selected_tables:
  [t_cost_center, t_budget, t_department]
Accuracy@3 = 1.0
Recall@3 = 1.0
Precision@3 = 1.0
Accuracy@5 = 1.0
Recall@5 = 1.0
Precision@5 = 1.0
```

## Iteration 43：recall_context 相关表污染修复

### 问题

用户问题：

```text
查询当前公司去年的（净利润 < 0 时的 ABS(净利润)，即亏损金额），关联 t_journal_item、t_account、t_expense_claim 表
```

`sql.select_tables.llm` 返回：

```text
t_journal_item,t_account,t_expense_claim
```

但后续 `schema.get_table_relationships` 收到：

```text
t_expense_claim,t_budget,t_journal_item,t_journal_entry,t_account,t_fund_transfer,
t_receivable_payable,t_cost_center,t_fixed_asset,t_department
```

这说明问题不在 LLM 选表，而在 LLM 选表后的本地合并/补表阶段。

### 根因

`recall_context` 初版会把召回到的所有业务知识 `关联表` 都写入 `business_related_tables`。RAG 召回可能同时带出预算、资金、应收应付、固定资产等相近财务知识；这些知识虽然被召回，但并不一定命中当前用户问题。

随后 `select_tables` 会把 `business_related_tables` 与 LLM 选表结果合并，再调用 `get_table_relationships(selected)`，导致关系查询入参从 3 张表膨胀到 10 张表。

### 修复

- `_build_recall_context()` 只收录当前 query 命中 `term/synonyms` 的业务知识相关表。
- 未命中的业务知识仍保留在原始 `evidence` 中，供 `query_enhance/sql_generate` 参考，但不参与选表合并。
- few-shot 表证据也只从与当前 query 有足够词面重叠的示例中抽取，避免相似度召回噪声污染选表。
- 不引入版本兼容逻辑；如果线上已有旧 checkpoint/cache 污染，部署后清空 checkpoint/cache。

### TDD 验证

新增两个回归：

1. `recall_evidence` 召回多个业务知识时，只把 query 命中的“净利润/亏损金额”相关表写入 `recall_context.business_related_tables`。
2. 清洗后的 `recall_context` 进入 `select_tables` 后，`get_table_relationships` 只接收 LLM 选出的 3 张表，不再扩散到预算、资金、应收应付、固定资产等表。

已执行：

```bash
.venv/bin/python -m pytest tests/test_sql_react.py tests/test_dispatcher.py tests/test_final_api.py tests/test_eval_pipeline_strategies.py -q
# 108 passed, 16 warnings
```
