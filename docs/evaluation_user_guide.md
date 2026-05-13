# 评测使用手册

本文档面向日常开发和验收，说明当前已完成的评测能力怎么使用、产出什么、怎么看结果。

## 评测类型

| 类型 | 命令 | 是否调用 Agent | 是否执行 SQL | 主要用途 |
|------|------|----------------|--------------|----------|
| Retrieval 评测 | `generate` + `run` | 默认否 | 否 | 衡量 schema、业务知识、few-shot 召回质量 |
| 线上预选链路评测 | `run --include-online-pipeline` | 是 | 否 | 衡量 `recall_evidence -> query_enhance -> select_tables` 是否选对表 |
| 离线 NL2SQL 评测 | `run-nl2sql` | 否 | 否 | 评测已记录 SQL/结果样本 |
| 在线 NL2SQL 评测 | `run-online-nl2sql` | 是 | 可选 | 真实回放 NL2SQL Agent，记录 SQL、结果和延迟 |

## 1. Retrieval 评测

生成 schema 评测集：

```bash
python -m agents.eval.cli generate \
  --num-per-table 3 \
  --output data/eval/eval_dataset.jsonl
```

这一步会读取 MySQL `t_semantic_model`，让 LLM 基于表、字段、业务名、同义词和字段描述生成自然语言问题，并标注标准相关表 `relevant_doc_ids`。默认还会用本地词法匹配补充：

- `relevant_business_doc_ids`
- `relevant_agent_doc_ids`

运行评测：

```bash
python -m agents.eval.cli run \
  --dataset data/eval/eval_dataset.jsonl \
  --output data/eval/eval_report.json
```

默认策略：

| 策略 | 说明 |
|------|------|
| `schema_lexical` | 基于 schema metadata 的本地词法召回 |
| `schema_table_name` | 只看表名的对照组 |
| `business_knowledge_recall` | 业务知识召回，仅有对应标注时计算 |
| `agent_knowledge_recall` | SQL few-shot 召回，仅有对应标注时计算 |

评测真实线上选表前置链路：

```bash
python -m agents.eval.cli run \
  --dataset data/eval/eval_dataset.jsonl \
  --output data/eval/eval_report.json \
  --include-online-pipeline
```

注意：`--include-online-pipeline` 会调用 `query_enhance` 和 `select_tables` 的 LLM。

最近一次全量线上预选链路评测结果（45 条，2026-05-13）：

| 策略 | MRR | Accuracy@5 | Recall@5 | P50/P95 延迟 |
|------|----:|-----------:|---------:|-------------:|
| `schema_lexical` | 96.67% | 77.78% | 90.63% | 0.0 / 0.1 ms |
| `preselect_pipeline` | 96.67% | 88.89% | 94.07% | 7426.8 / 10541.4 ms |

管理表专项线上评测（12 条）中，`preselect_pipeline Recall@5 = 100%`、`MRR = 100%`。专项数据只用于验证用户/角色/部门等管理表问题，不代表全量业务指标。

## 2. 离线 NL2SQL 评测

初始化模板：

```bash
python -m agents.eval.cli run-nl2sql \
  --dataset data/eval/nl2sql_cases.jsonl \
  --init-template
```

填写真实回放样本：

```json
{"query":"去年亏损多少","generated_sql":"SELECT ...;","actual_result":[{"loss_amount":"100.00"}],"expected_result":[{"loss_amount":"100.00"}],"latency_ms":1200,"first_token_latency_ms":350}
```

运行离线评测：

```bash
python -m agents.eval.cli run-nl2sql \
  --dataset data/eval/nl2sql_cases.jsonl \
  --output data/eval/nl2sql_eval_report.json
```

离线评测不会调用 Agent，也不会执行数据库。它适合评测生产日志、人工标注样本、历史失败 case。

## 3. 在线 NL2SQL 评测

初始化模板：

```bash
python -m agents.eval.cli run-online-nl2sql \
  --dataset data/eval/online_nl2sql_cases.jsonl \
  --init-template
```

样本格式：

```json
{"query":"去年亏损多少","expected_result":[{"loss_amount":"10000.00"}],"tags":["profit_loss"]}
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

默认会强制 `intent=sql_query`，避免把意图分类能力混进 NL2SQL 主链路指标。如果要连同意图分类一起评测：

```bash
python -m agents.eval.cli run-online-nl2sql \
  --dataset data/eval/online_nl2sql_cases.jsonl \
  --output data/eval/online_nl2sql_eval_report.json \
  --auto-approve-sql \
  --full-dispatch
```

安全说明：

- 在线评测会调用外部 LLM。
- 加 `--auto-approve-sql` 后会通过现有 SQL 安全检查，再自动恢复审批中断。
- 只建议在测试库或只读账号上使用自动审批。

## 4. 查看报告

启动服务后打开页面：

```text
http://localhost:8080
```

进入 `Evaluation` tab。页面读取：

| API | 用途 |
|-----|------|
| `GET /api/eval/reports` | 报告列表 |
| `GET /api/eval/reports/latest` | 最新报告 |
| `GET /api/eval/reports/{name}` | 指定历史报告 |

`data/eval/` 已加入 `.gitignore`，本地报告和样本不会进入代码提交。

## 5. 指标解释

| 指标 | 取值范围/最优值 | 方向 | 说明 |
|------|----------------|------|------|
| `accuracy@K` | 0-1，最大 1.0 | 越大越好 | Top-K 是否完整包含标准相关文档 |
| `precision@K` | 0-1，最大 1.0 | 越大越好 | Top-K 中有多少是相关文档 |
| `recall@K` | 0-1，最大 1.0 | 越大越好 | 标准相关文档召回了多少 |
| `mrr` | 0-1，最大 1.0 | 越大越好 | 第一个相关文档出现得越靠前越好 |
| `ndcg@K` | 0-1，最大 1.0 | 越大越好 | 排序质量 |
| `sql_valid` | 0-1，最大 1.0 | 越大越好 | SQL 是否通过格式规范化 |
| `execution_success` | 0-1，最大 1.0 | 越大越好 | 是否没有执行错误 |
| `result_exact_match` | 0-1，最大 1.0 | 越大越好 | 执行结果是否和期望结果一致 |
| `latency.p95_ms` | 最小 0ms | 越小越好 | 端到端长尾耗时 |
| `first_token_latency.p95_ms` | 最小 0ms | 越小越好 | 首次响应/审批中断出现的长尾耗时 |

## 6. 推荐日常流程

开发一个 NL2SQL 优化后：

1. 跑单元测试。
2. 跑小规模 Retrieval 评测，确认召回没有退化。
3. 跑在线 NL2SQL 小样本，不自动审批，确认 SQL 生成质量。
4. 在测试库跑在线 NL2SQL 自动审批，确认执行结果。
5. 把失败 case 回灌到 `nl2sql_cases.jsonl` 或在线评测集。

上线前：

1. 固定一份人工审核过的黄金集。
2. 跑完整 Retrieval + 在线 NL2SQL。
3. 对比上一次报告的指标。
4. 重点查看失败明细，而不是只看平均值。
