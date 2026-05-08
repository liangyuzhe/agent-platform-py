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

说明：比例类指标的理论最优值都是 `1.0`，页面展示为 `100%`；延迟类指标理论最小值是 `0ms`，没有固定最大值，生产环境应按业务 SLA 设定告警阈值，例如检索 P95、端到端 P95 和 TTFT P95。

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
| `generate` | 让 LLM 基于 `t_semantic_model` 生成自然语言问题和标准相关表 | MySQL `t_semantic_model` | `eval_dataset.jsonl` | 否 |
| `run` | 用数据集里的 query 执行召回策略，并和标准相关表对比 | `eval_dataset.jsonl` + MySQL `t_semantic_model` | `eval_report.json` | 是 |

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

`--num-per-table 3` 表示每张表让 LLM 生成 3 条问题。假设 `t_semantic_model` 里有 20 张表，理论上约生成 60 条样本；代码会按 query 去重，所以最终数量可能略少。

注意：这是 LLM 生成的 synthetic dataset，适合冷启动和覆盖 schema，但仍建议后续人工审核一批高价值样本，并逐步加入真实用户问题。

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

旧版通用文档检索策略（Milvus/ES/RRF）默认不再运行，因为 schema 评测集标注的是 `schema_<table>`，当前线上 schema 权威来源也是 MySQL/Redis。如果需要兼容测试旧链路，可显式开启：

```bash
ENABLE_LEGACY_EVAL_RETRIEVERS=1 \
  python -m agents.eval.cli run --dataset data/eval/eval_dataset.jsonl --output data/eval/eval_report.json
```

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

进入 `Evaluation` tab。页面会读取 `/api/eval/reports/latest`，展示最新报告。

## API

| Endpoint | 用途 |
|----------|------|
| `GET /api/eval/reports` | 返回可用报告列表和简要摘要 |
| `GET /api/eval/reports/latest` | 返回最近一次完整报告，含 per-query 明细 |

## 后续迭代

### P1：端到端 NL2SQL 评测

在 retrieval 评测基础上，增加从自然语言到 SQL 执行结果的端到端指标：

- SQL 生成成功率
- SQL 可执行率
- 审批前安全通过率
- 结果正确率
- 异常反思修复率
- 总耗时、首字延迟、SQL 生成耗时、执行耗时

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
