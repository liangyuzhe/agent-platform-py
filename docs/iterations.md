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
