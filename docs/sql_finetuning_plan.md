# 财务 NL2SQL 微调方案

本文档用于规划 SQL 生成模型的 SFT/LoRA 微调。结论先行：公开数据只能做泛化补充，真正有效的财务 NL2SQL 样本必须来自本项目的 schema、业务口径、SQL few-shot、线上日志和人工审核。

## 模型选择

推荐基座：

```text
Qwen/Qwen2.5-Coder-7B-Instruct
```

选择原因：

- 7B 级别，推理和 LoRA 微调成本可控。
- Coder 系列对 SQL、结构化输出和工具调用格式更友好。
- Qwen 系列中文能力强，适合中文财务查询。
- Hugging Face 模型卡标注 Apache-2.0，商业使用约束相对清晰。

备选：

| 模型 | 适用场景 | 注意点 |
|------|----------|--------|
| `deepseek-ai/deepseek-coder-6.7b-instruct` | 代码/SQL 能力强 | 中文财务语义可能需要更多领域样本 |
| `codellama/CodeLlama-7b-Instruct-hf` | 英文 SQL/代码基线 | 中文能力弱于 Qwen，许可证和商用条款需单独复核 |
| `bigcode/starcoder2-7b` | 代码生成基线 | instruct 能力和中文财务场景需要额外对齐 |

## 样本来源

### 1. 项目内高价值样本

优先级最高，应该成为主训练集。

| 来源 | 生成方式 | 用途 |
|------|----------|------|
| `t_semantic_model` | 表、字段、字段描述、业务名、同义词、主外键关系 | 让模型学习本项目 schema |
| `t_business_knowledge` | 指标口径、公式、同义词、业务规则 | 让模型学习财务口径 |
| `t_agent_knowledge` | 已沉淀 SQL few-shot | 让模型学习高质量 SQL 模板 |
| 线上成功日志 | 用户 query、最终 SQL、执行结果、人工确认 | 训练真实分布 |
| 线上失败日志 | 错 SQL、错误原因、修正 SQL | 训练纠错和避坑 |
| `result_reflection` 产物 | 异常结果、原 SQL、修正 SQL | 训练结果反思后的 SQL 修正 |
| 人工黄金集 | 财务人员/研发审核 | 作为验证集和测试集 |

推荐样本格式：

```json
{
  "messages": [
    {"role": "system", "content": "你是财务 NL2SQL 专家。只输出 SELECT/WITH SQL。"},
    {"role": "user", "content": "数据库结构和业务口径:\n...\n\n用户问题: 去年亏损多少"},
    {"role": "assistant", "content": "SELECT ...;"}
  ],
  "metadata": {
    "tables": ["t_journal_entry", "t_journal_item", "t_account"],
    "business_terms": ["净利润", "亏损金额"],
    "source": "human_verified"
  }
}
```

### 2. 公开数据补充

公开数据适合做 SQL 泛化，不建议直接作为最终财务 SFT 主数据。

| 数据集 | 价值 | 使用建议 |
|--------|------|----------|
| FINCH | 金融领域 Text-to-SQL，包含金融相关数据库和问题 | 用于金融 SQL 语义热身，注意非商用许可 |
| BIRD | 真实数据库、复杂 Text-to-SQL benchmark | 用于复杂 SQL 泛化 |
| Spider | 经典跨域 Text-to-SQL | 用于基础 SQL 泛化 |
| CHASE / SeSQL | 中文多轮 Text-to-SQL | 用于中文、多轮、省略追问能力 |

使用公开数据时要统一成项目 prompt 格式，并标记 `source=public`。如果许可证不允许商业使用，只能用于研究对比，不能混入生产模型训练。

## 数据飞轮

```text
线上 query
  ↓
Agent 生成 SQL
  ↓
用户审批 / SQL 执行 / 结果反思
  ↓
记录成功、失败、修正样本
  ↓
人工审核高价值 case
  ↓
写入 SFT JSONL
  ↓
LoRA 微调
  ↓
离线评测 + 在线评测
  ↓
灰度上线
```

## 样本分层

| 分层 | 占比建议 | 说明 |
|------|----------|------|
| 本项目人工审核样本 | 50%-70% | 最关键，决定财务口径是否稳定 |
| 本项目合成样本 | 15%-30% | 从 schema 和业务知识生成，补覆盖 |
| 线上失败修正样本 | 10%-20% | 提升鲁棒性 |
| 公开 Text-to-SQL 样本 | 0%-20% | 只补 SQL 泛化，避免稀释项目口径 |

## 合成样本策略

从项目 schema 生成样本时，不要只让 LLM 自由发挥。推荐模板化约束：

1. 每个指标生成多种问法：标准名、同义词、口语表达、追问表达。
2. 每个时间口径生成多种表达：去年、本月、上季度、截至当前期间。
3. 每个样本必须绑定标准表、字段和业务公式。
4. 每条 SQL 必须能通过 `normalize_sql_answer`。
5. 高风险样本必须人工审核，例如损益、余额、往来、凭证状态。

示例：

```json
{
  "query": "去年亏损多少",
  "business_term": "亏损金额",
  "tables": ["t_journal_entry", "t_journal_item", "t_account"],
  "sql": "SELECT ...;",
  "expected_result": [{"loss_amount": "10000.00"}],
  "review_status": "approved"
}
```

## 训练策略

推荐先做 LoRA/QLoRA，不直接全参微调。

| 阶段 | 目标 | 数据 |
|------|------|------|
| S0 | 不训练，建立基线 | 当前外部 LLM + 评测集 |
| S1 | SQL 格式和项目 schema 对齐 | 人工审核 few-shot + schema 合成样本 |
| S2 | 财务口径稳定 | 业务知识样本 + 失败修正样本 |
| S3 | 多轮追问稳定 | 线上追问样本 + 人工构造多轮样本 |

训练建议：

- LoRA rank：16 或 32 起步。
- 学习率：`1e-4` 到 `2e-4` 起步。
- epoch：2-3，避免过拟合。
- max length：根据 schema prompt 长度设置，建议 4096 起步。
- 验证集固定，不参与训练。

## 评测门槛

微调模型上线前必须跑：

```bash
python -m agents.eval.cli run \
  --dataset data/eval/eval_dataset.jsonl \
  --output data/eval/eval_report.json

python -m agents.eval.cli run-online-nl2sql \
  --dataset data/eval/online_nl2sql_cases.jsonl \
  --output data/eval/online_nl2sql_eval_report.json \
  --auto-approve-sql
```

建议最低门槛：

| 指标 | 建议阈值 |
|------|----------|
| `sql_valid` | >= 0.98 |
| `execution_success` | >= 0.95 |
| `result_exact_match` | >= 0.90 |
| `recall@5` | 不低于当前线上模型 |
| `latency.p95_ms` | 不高于当前线上模型 SLA |

## 不建议做的事

- 不要只拿 Spider/BIRD 公开数据微调后直接替换线上模型。
- 不要把未审核的 LLM 合成 SQL 全量加入训练集。
- 不要把执行失败但没修正的 SQL 当正样本。
- 不要把测试集混入训练集。
- 不要忽略许可证，特别是非商用公开数据。
