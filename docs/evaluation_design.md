# 评测体系设计

本文档描述当前项目的评测闭环，用于量化 NL2SQL/RAG 迭代效果，并在生产环境中定位优化方向。

## 目标

- 用稳定数据集持续评估不同检索/生成策略。
- 量化准确率、召回率、排序质量和延迟。
- 保存每次评测明细，做到可回溯、可对比。
- 前端页面展示关键指标，便于非研发角色理解质量趋势。

## 当前落地范围

第一阶段先覆盖检索评测，因为检索质量直接影响 SQL 生成和 RAG 回答质量。

```text
t_semantic_model -> 生成评测数据集 -> 多策略检索 -> 指标计算 -> JSON 报告 -> 前端 Evaluation 页面
```

Schema 评测数据从 MySQL `t_semantic_model` 生成，不再依赖历史 Milvus `source=mysql_schema` 数据。

## 指标

| 指标 | 含义 | 取值范围/最优值 | 方向 | 为什么重要 |
|------|------|----------------|------|------------|
| `accuracy@K` | Top-K 是否完整命中所有标注相关文档，命中为 1，否则 0 | 0-1，最大 1.0 | 越大越好 | 适合作为用户可理解的“这条 query 是否通过” |
| `precision@K` | Top-K 中相关文档占比 | 0-1，最大 1.0 | 越大越好 | 衡量噪声比例，低 precision 会增加 LLM 干扰 |
| `recall@K` | 标注相关文档中被 Top-K 召回的比例 | 0-1，最大 1.0 | 越大越好 | NL2SQL/RAG 首要指标，漏召回通常直接导致错答 |
| `mrr` | 第一个相关文档排名的倒数 | 0-1，最大 1.0 | 越大越好 | 衡量相关信息是否靠前 |
| `ndcg@K` | 排序质量，越靠前命中收益越高 | 0-1，最大 1.0 | 越大越好 | 衡量排序器/融合策略效果 |
| `avg_latency_ms` | 平均检索耗时 | 最小 0ms，无理论上限 | 越小越好 | 衡量整体体验 |
| `p50_latency_ms` | 中位延迟 | 最小 0ms，无理论上限 | 越小越好 | 衡量典型请求体验 |
| `p95_latency_ms` | 长尾延迟 | 最小 0ms，无理论上限 | 越小越好 | 生产环境更关注长尾稳定性 |
| `first_token_latency_ms` | 首字延迟，当前报告格式已预留 | 最小 0ms，无理论上限 | 越小越好 | 后续接入端到端流式评测，用于衡量用户等待体感 |
| `route_accuracy` | 复杂查询路由模式是否命中标注结果 | 0-1，最大 1.0 | 越大越好 | 衡量超过单 SQL 表数预算时，是进入计划模式还是澄清 |
| `plan_validity_rate` | 复杂计划通过结构校验的比例 | 0-1，最大 1.0 | 越大越好 | 防止不可执行、不可审计的多步骤计划进入执行 |
| `step_success_rate` | 计划执行步骤成功数 / 总步骤数 | 0-1，最大 1.0 | 越大越好 | 衡量多 SQL/本地聚合链路稳定性 |
| `final_answer_correctness` | 复杂计划最终答案是否符合预期 | 0-1，最大 1.0 | 越大越好 | 衡量拆分、合并、展示是否共同回答了用户问题 |

说明：比例类指标的理论最优值都是 `1.0`，页面展示为 `100%`；延迟类指标理论最小值是 `0ms`，没有固定最大值，生产环境应按业务 SLA 设定告警阈值，例如检索 P95、端到端 P95 和 TTFT P95。

复杂查询路由评测与 Schema Recall@K 不同：Recall@K 衡量“有没有把正确表召回到 TopK”，而 `route_accuracy` 衡量“当选表结果已经超过单 SQL 预算时，系统是否选择正确执行模式”。例如超过 8 张表的分析型问题应该进入 `complex_plan`，明细导出或敏感数据问题应该进入 `clarify`。这类评测不要求调用外部 LLM，可用 `route_signal` 标注模拟数据库规则或 LLM 仲裁后的语义信号。

## 报告格式

报告保存为 JSON：

```json
{
  "run_id": "...",
  "created_at": "...",
  "dataset_path": "data/eval/eval_dataset.jsonl",
  "report_type": "retrieval",
  "strategies": [
    {
      "strategy": "hybrid_rerank",
      "description": "...",
      "num_queries": 120,
      "metrics": {
        "accuracy@5": 0.82,
        "precision@5": 0.61,
        "recall@5": 0.9,
        "mrr": 0.73,
        "ndcg@5": 0.84
      },
      "latency": {
        "avg_ms": 120.5,
        "p50_ms": 98.1,
        "p95_ms": 260.4
      },
      "first_token_latency": {
        "avg_ms": null,
        "p50_ms": null,
        "p95_ms": null
      },
      "results": [
        {
          "query": "去年亏损多少",
          "relevant_doc_ids": ["schema_t_journal_entry", "schema_t_journal_item"],
          "retrieved_doc_ids": ["schema_t_journal_entry", "schema_t_account"],
          "metrics": {"accuracy@5": 0.0, "recall@5": 0.5},
          "latency_ms": 88.2
        }
      ]
    }
  ]
}
```

`results` 保留 per-query 明细，用于回溯坏样本和定位是召回、排序还是数据标注问题。

## 使用方式

### 两个命令分别做什么

评测流程分两步：

```text
generate：生成评测数据集
run：使用评测数据集运行评测并生成报告
```

这两个命令的职责不同，不能互相替代。

| 命令 | 做什么 | 输入 | 输出 | 是否计算指标 |
|------|--------|------|------|--------------|
| `generate` | 让 LLM 基于 `t_semantic_model` 生成自然语言问题和标准相关表，并本地补充知识召回标注 | MySQL `t_semantic_model`、`t_business_knowledge`、`t_agent_knowledge` | `eval_dataset.jsonl` | 否 |
| `run` | 用数据集里的 query 执行召回策略，并和标准相关表对比 | `eval_dataset.jsonl` + MySQL `t_semantic_model` | `eval_report.json` | 是 |
| `run-nl2sql` | 对已记录的 NL2SQL SQL/结果样本做离线端到端评测 | `nl2sql_cases.jsonl` | `nl2sql_eval_report.json` | 是 |
| `run-online-nl2sql` | 调用真实 Agent 回放 query，可停在审批中断或自动审批后执行 SQL | `online_nl2sql_cases.jsonl` | `online_nl2sql_eval_report.json` | 是 |

### 1. 生成评测数据集

生成数据集：

```bash
python -m agents.eval.cli generate --num-per-table 3 --output data/eval/eval_dataset.jsonl
```

这一步的原理：

```text
MySQL t_semantic_model
  ↓
渲染为 schema 文本（表名、字段、业务名、同义词、字段描述、主外键）
  ↓
LLM 根据 schema 生成自然语言问题
  ↓
LLM 同时标注回答该问题需要哪些表
  ↓
写入 JSONL 数据集
```

生成的数据格式：

```json
{"query": "去年亏损多少？", "relevant_doc_ids": ["schema_t_journal_entry", "schema_t_journal_item", "schema_t_account"]}
```

字段说明：

| 字段 | 含义 |
|------|------|
| `query` | 模拟用户自然语言问题 |
| `relevant_doc_ids` | 标准答案：回答该问题必须召回的表，格式为 `schema_<table_name>` |
| `relevant_business_doc_ids` | 可选：该问题应召回的业务知识，格式为 `business_<id>` |
| `relevant_agent_doc_ids` | 可选：该问题应召回的 SQL few-shot 示例，格式为 `agent_<id>` |

`--num-per-table 3` 表示每张表让 LLM 生成 3 条问题。假设 `t_semantic_model` 里有 20 张表，理论上约生成 60 条样本；代码会按 query 去重，所以最终数量可能略少。

注意：这是 LLM 生成的 synthetic dataset，适合冷启动和覆盖 schema，但仍建议后续人工审核一批高价值样本，并逐步加入真实用户问题。

默认会排除系统内部表，例如 `domain_summary`、`t_semantic_model`、`t_business_knowledge`、`t_agent_knowledge`。这些表服务于 Agent 运行和知识管理，不是业务人员自然语言查数的目标；如果放进评测集，LLM 可能生成“查询领域摘要”这类内部运维问题，导致业务召回指标失真。

生成后会默认用本地规则补充知识召回标注：

- 业务知识：用 `term` 和 `synonyms` 命中 query 后生成 `business_<id>`。
- Agent 知识：用 query 与 few-shot 的 `question`、`description`、`category` 做词法重叠匹配后生成 `agent_<id>`。

这一步不调用 LLM。如果只想生成 schema 表标注，可使用 `--no-knowledge-labels`。

如需额外排除表，可设置：

```bash
EVAL_EXCLUDE_TABLES=table_a,table_b \
  python -m agents.eval.cli generate --num-per-table 3 --output data/eval/eval_dataset.jsonl
```

### 2. 运行评测

运行评测：

```bash
python -m agents.eval.cli run --dataset data/eval/eval_dataset.jsonl --output data/eval/eval_report.json
```

这一步的原理：

```text
读取 eval_dataset.jsonl
  ↓
对每条 query 运行一个或多个召回策略
  ↓
得到 retrieved_doc_ids
  ↓
和 relevant_doc_ids 对比
  ↓
计算 Accuracy / Precision / Recall / MRR / NDCG / 延迟
  ↓
写入 eval_report.json
```

`run` 会再次读取 MySQL `t_semantic_model` 构建本地 schema 检索器，然后用数据集中的 `query` 和 `relevant_doc_ids` 计算指标。默认策略对齐当前 NL2SQL 架构：

| 策略 | 说明 | 用途 |
|------|------|------|
| `schema_lexical` | 基于表名、字段名、字段注释、业务名、同义词、业务描述做本地词法召回 | 当前 schema metadata 质量基线 |
| `schema_table_name` | 只基于表名做召回 | 对照组，用于判断字段业务描述是否带来收益 |
| `business_knowledge_recall` | 单独评测业务知识召回 | 仅当数据集包含 `relevant_business_doc_ids` 时计算 |
| `agent_knowledge_recall` | 单独评测 SQL few-shot 示例召回 | 仅当数据集包含 `relevant_agent_doc_ids` 时计算 |
| `preselect_pipeline` | 执行线上选表前置链路：`recall_evidence -> recall_context -> query_enhance -> select_tables` | 显式开启后评测真实 NL2SQL 选表路径 |

`business_knowledge_recall` 和 `agent_knowledge_recall` 发生在选表之前，它们不是直接选择 schema 表，而是给 `query_enhance` 和后续 SQL 生成提供业务定义、公式、few-shot 示例。因此这两个策略不会拿 `relevant_doc_ids` 强行计算 schema 召回率，只有数据集显式标注了 `relevant_business_doc_ids` 或 `relevant_agent_doc_ids` 时才参与指标计算；没有对应标注时报告里会显示该策略 `num_queries = 0`。

`preselect_pipeline` 是更贴近线上流程的 schema 召回评测：先由 `recall_evidence` 召回业务知识和 few-shot，并整理为 `recall_context`；后续 `query_enhance` 与 `select_tables` 复用同一份状态，不在选表阶段重复召回；最后把 `select_tables` 产出的表名转换成 `schema_<table_name>` 与 `relevant_doc_ids` 对比。它会走 LLM 选表节点，因此默认不启用，避免普通本地评测消耗 LLM token。需要评测线上预选链路时显式增加参数：

```bash
python -m agents.eval.cli run \
  --dataset data/eval/eval_dataset.jsonl \
  --output data/eval/eval_report.json \
  --include-online-pipeline
```

### 最近一次评测结果

数据集：`data/eval/eval_dataset.jsonl`，45 条 schema 标注样本。时间：2026-05-13。

| 策略 | MRR | Accuracy@5 | Precision@5 | Recall@5 | NDCG@5 | 平均延迟 | P50 延迟 | P95 延迟 |
|------|----:|-----------:|------------:|---------:|-------:|---------:|---------:|---------:|
| `schema_lexical` | 96.67% | 77.78% | 35.56% | 90.63% | 90.03% | 0.1 ms | 0.0 ms | 0.1 ms |
| `preselect_pipeline` | 96.67% | 88.89% | 84.93% | 94.07% | 94.13% | 7545.6 ms | 7426.8 ms | 10541.4 ms |

管理表专项评测（`management_eval_dataset.jsonl`，12 条）中，`preselect_pipeline` 的 `Recall@5 = 100%`、`MRR = 100%`。该结果只说明用户/角色/部门等管理表链路已被修复，不能替代全量业务集指标。

Iteration 41 后，线上预选链路的证据流转方式发生了变化，但指标计算口径不变：

```text
query
  -> recall_evidence: business knowledge + few-shot，只调用一次
  -> recall_context: 相关表、匹配术语、few-shot 问题
  -> query_enhance: 使用 recall_context.evidence
  -> select_tables: 使用轻量 table routing profile + recall_context 加权 + 逻辑外键补表
  -> retrieved_doc_ids: schema_<table_name>
```

因此 `Recall@5` 仍然表示 Top5 是否包含标准相关表；变化只在召回证据如何被线上选表链路复用。

旧版通用文档检索策略（Milvus/ES/RRF）默认不再运行，因为 schema 评测集标注的是 `schema_<table>`，当前线上 schema 权威来源也是 MySQL/Redis。如果需要兼容测试旧链路，可显式开启：

```bash
ENABLE_LEGACY_EVAL_RETRIEVERS=1 \
  python -m agents.eval.cli run --dataset data/eval/eval_dataset.jsonl --output data/eval/eval_report.json
```

### 复杂查询路由专项评测

复杂查询专项样本建议单独维护，不和 schema 召回样本混在一起：

```json
{"query": "收入成本预算回款费用之间的关系", "tables": ["t_1", "t_2", "..."], "route_signal": "analysis", "expected_route": "complex_plan"}
{"query": "员工工资和部门角色权限", "tables": ["t_1", "t_2", "..."], "route_signal": "detail", "expected_route": "clarify"}
```

字段说明：

| 字段 | 含义 |
|------|------|
| `tables` | `select_tables` 和逻辑外键补表后的最终表集合 |
| `route_signal` | 数据库规则或 LLM 仲裁后的语义信号，不在代码里写死关键词 |
| `expected_route` | 标注的目标路由：`single_sql`、`single_sql_with_strict_checks`、`complex_plan`、`clarify` |

当前已提供本地 helper `run_complex_route_eval_case()` 和 `route_accuracy()`，用于单元测试和后续 CLI 接入。下一步可以把它接入评测报告 JSON 与前端 Evaluation 页面，展示 `route_accuracy`、`plan_validity_rate`、`step_success_rate` 和 `final_answer_correctness`。

评测报告会包含两层信息：

| 层级 | 内容 | 用途 |
|------|------|------|
| 策略汇总 | 每个策略的平均指标、P50/P95 延迟 | 判断哪种策略整体更好 |
| Query 明细 | 每条 query 的标准相关表、实际召回表、单条指标、耗时 | 回溯失败样本，定位是漏召回、排序差还是标注问题 |

举例：

```json
{
  "query": "去年亏损多少？",
  "relevant_doc_ids": ["schema_t_journal_entry", "schema_t_journal_item", "schema_t_account"],
  "retrieved_doc_ids": ["schema_t_journal_entry", "schema_t_account", "schema_t_invoice"],
  "metrics": {
    "accuracy@5": 0.0,
    "recall@5": 0.6667
  }
}
```

这个例子表示 Top5 召回了 3 张标准表中的 2 张：

- `Recall@5 = 2/3 = 0.6667`
- `Accuracy@5 = 0`，因为没有完整命中全部标准相关表

### 3. 页面查看

查看页面：

```text
http://localhost:8080
```

进入 `Evaluation` tab。页面会读取 `/api/eval/reports` 获取历史报告列表，并默认展示最新报告。也可以从下拉菜单切换历史报告，回溯某一次评测的策略指标和 query 明细。

页面支持两类报告：

- Retrieval 报告：展示策略对比，并可按策略切换 per-query 明细。
- NL2SQL 端到端报告：展示 SQL 有效率、执行成功率、结果匹配率、延迟和首字延迟。

### 4. NL2SQL 端到端离线评测

离线端到端评测用于生产回放或人工标注 case，不会调用线上 Agent，也不会执行数据库。它评测的是“已经生成/记录下来的 SQL 和执行结果是否符合预期”。

首次使用时先生成 JSONL 模板：

```bash
python -m agents.eval.cli run-nl2sql \
  --dataset data/eval/nl2sql_cases.jsonl \
  --init-template
```

填写真实 `generated_sql`、`actual_result`、`expected_result` 后运行：

```bash
python -m agents.eval.cli run-nl2sql \
  --dataset data/eval/nl2sql_cases.jsonl \
  --output data/eval/nl2sql_eval_report.json
```

样本格式：

```json
{"query":"去年亏损多少","generated_sql":"SELECT ...;","actual_result":[{"loss_amount":"100.00"}],"expected_result":[{"loss_amount":"100.00"}],"latency_ms":1200,"first_token_latency_ms":350}
```

核心指标：

| 指标 | 含义 | 方向 |
|------|------|------|
| `sql_valid` | SQL 是否能通过格式规范化校验 | 0-1，越大越好 |
| `execution_success` | 样本是否没有记录执行错误 | 0-1，越大越好 |
| `result_exact_match` | `actual_result` 与 `expected_result` 规范化后是否一致 | 0-1，越大越好 |
| `latency.p95_ms` | 端到端长尾延迟 | 最小 0ms，越小越好 |
| `first_token_latency.p95_ms` | 首字长尾延迟 | 最小 0ms，越小越好 |

### 5. 在线 NL2SQL 端到端评测

在线评测用于真实回放当前 Agent 链路。它会调用外部 LLM，并复用 LangGraph 审批中断机制。

初始化模板：

```bash
python -m agents.eval.cli run-online-nl2sql \
  --dataset data/eval/online_nl2sql_cases.jsonl \
  --init-template
```

只评测 SQL 生成，到审批中断为止：

```bash
python -m agents.eval.cli run-online-nl2sql \
  --dataset data/eval/online_nl2sql_cases.jsonl \
  --output data/eval/online_nl2sql_eval_report.json
```

自动审批并执行 SQL：

```bash
python -m agents.eval.cli run-online-nl2sql \
  --dataset data/eval/online_nl2sql_cases.jsonl \
  --output data/eval/online_nl2sql_eval_report.json \
  --auto-approve-sql
```

默认会强制 `intent=sql_query`，让指标聚焦 NL2SQL 主链路。如果需要把意图分类也纳入评测，增加 `--full-dispatch`。

在线评测报告沿用 NL2SQL 端到端报告结构，并额外记录：

| 字段 | 含义 |
|------|------|
| `session_id` | 本次回放使用的隔离线程 |
| `sql_rounds` | 每次审批中断生成的 SQL，包含结果反思后的修正 SQL |
| `answer` | Agent 最终返回给前端的用户友好回答 |
| `status` | `completed`、`pending_approval`、`error` 等状态 |
| `first_response_latency_ms` | 首次审批中断或最终响应出现的耗时 |

## API

| Endpoint | 用途 |
|----------|------|
| `GET /api/eval/reports` | 返回可用报告列表和简要摘要 |
| `GET /api/eval/reports/latest` | 返回最近一次完整报告，含 per-query 明细 |
| `GET /api/eval/reports/{name}` | 按已发现的报告文件名返回完整报告 |

## 后续迭代

### P1：真实黄金集沉淀

在线评测 Runner 已补齐，下一步重点是把真实财务 query、期望 SQL/结果、失败修正样本沉淀成稳定黄金集。

### P2：趋势对比

将每次报告落库到 MySQL：

- `t_eval_run`
- `t_eval_strategy_result`
- `t_eval_case_result`

页面支持按时间对比指标曲线，观察迭代是否真实提升。

### P3：LLM 优化建议

低优先级接入 LLM，对失败样本做归因和建议：

- 召回失败：建议补充业务同义词、few-shot 或字段描述。
- 排序失败：建议调整 RRF 权重、reranker 阈值或 top-k。
- SQL 失败：建议补充业务知识或 SQL 示例。

LLM 建议只作为辅助，不作为自动改配置依据。
