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

| 指标 | 含义 | 为什么重要 |
|------|------|------------|
| `accuracy@K` | Top-K 是否完整命中所有标注相关文档，命中为 1，否则 0 | 适合作为用户可理解的“这条 query 是否通过” |
| `precision@K` | Top-K 中相关文档占比 | 衡量噪声比例，低 precision 会增加 LLM 干扰 |
| `recall@K` | 标注相关文档中被 Top-K 召回的比例 | NL2SQL/RAG 首要指标，漏召回通常直接导致错答 |
| `mrr` | 第一个相关文档排名的倒数 | 衡量相关信息是否靠前 |
| `ndcg@K` | 排序质量，越靠前命中收益越高 | 衡量排序器/融合策略效果 |
| `avg_latency_ms` | 平均检索耗时 | 衡量整体体验 |
| `p50_latency_ms` | 中位延迟 | 衡量典型请求体验 |
| `p95_latency_ms` | 长尾延迟 | 生产环境更关注长尾稳定性 |
| `first_token_latency_ms` | 首字延迟，当前报告格式已预留 | 后续接入端到端流式评测，用于衡量用户等待体感 |

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

生成数据集：

```bash
python -m agents.eval.cli generate --num-per-table 3 --output data/eval/eval_dataset.jsonl
```

运行评测：

```bash
python -m agents.eval.cli run --dataset data/eval/eval_dataset.jsonl --output data/eval/eval_report.json
```

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
