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

## 迭代 3：业务知识配置（TODO）

> 参考 DataAgent 的 BusinessKnowledge 模块，支持业务术语、公式定义的存储和检索。

## 迭代 4：SQL 领域智能体知识库（TODO）

> 参考 DataAgent 的 AgentKnowledge 模块，支持 Q&A few-shot 对注入 SQL 生成 prompt。
