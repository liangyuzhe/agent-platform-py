# Go-Agent 技术架构全解析与优化方案

## 目录

1. [项目总览](#1-项目总览)
2. [核心架构设计](#2-核心架构设计)
3. [Flow 编排层详解](#3-flow-编排层详解)
4. [RAG 检索增强生成](#4-rag-检索增强生成)
5. [记忆系统（Memory）](#5-记忆系统memory)
6. [模型抽象层](#6-模型抽象层)
7. [工具层（Tool）](#7-工具层tool)
8. [SFT 数据管线](#8-sft-数据管线)
9. [存储与基础设施](#9-存储与基础设施)
10. [当前设计的优势与不足](#10-当前设计的优势与不足)
11. [优化方案：基于最新 Agent 技术的改进路线](#11-优化方案基于最新-agent-技术的改进路线)

---

## 1. 项目总览

Go-Agent 是一个基于 **CloudWeGo Eino** 图编排框架构建的 Go 语言 AI Agent 平台。核心能力包括：

- **RAG 对话**：文档索引 + 向量/关键词双路检索 + LLM 生成
- **SQL 生成与执行**：自然语言 -> SQL -> 人工审批 -> MCP 执行
- **数据分析**：SQL 结果 -> 统计分析 -> 图表生成 -> 文字报告
- **对话记忆**：短期历史 + LLM 摘要压缩的两级记忆
- **SFT 数据采集**：自动收集训练数据，教师模型标注，导出 JSONL

### 1.1 技术栈

| 组件 | 技术选型 | 设计意图 |
|------|---------|---------|
| 图编排 | CloudWeGo Eino | 类 LangGraph 的 DAG 执行引擎，Go 原生，类型安全 |
| HTTP 服务 | Gin | 轻量高性能，适合 API 网关 |
| 向量数据库 | Milvus | 高性能向量检索，支持大规模数据 |
| 关键词检索 | Elasticsearch 8 | 全文检索 + 向量检索双模式 |
| 缓存 | Redis | CheckPoint 持久化、检索缓存、会话存储 |
| MCP 工具 | mcp-server-mysql | 标准化 SQL 执行，通过 MCP 协议隔离 |
| 可观测性 | LangSmith + CozeLoop | 双链路追踪，覆盖调用全链路 |

### 1.2 目录结构

```
go-agent/
├── api/                    # HTTP API 层（Gin 路由 + SSE 流式）
├── flow/                   # 图编排层（RAG Chat、SQL React、Analyst、Final）
├── model/                  # 模型抽象层（Chat Model + Embedding Model 工厂）
├── rag/                    # RAG 管线（索引图、检索图、DB 客户端）
├── tool/                   # 工具层（Memory、Storage、Document、SQL、SFT、Trace）
├── algorithm/              # 算法（BM25、RRF）
├── config/                 # 配置加载（.env -> 结构体）
├── main.go                 # 入口，初始化编排
└── docker-compose.yaml     # 基础设施（Milvus、ES、Redis）
```

---

## 2. 核心架构设计

### 2.1 设计哲学：为什么选择图编排？

项目采用 **DAG（有向无环图）** 作为核心执行模型，而非简单的函数链式调用。这一设计选择背后的思考：

**优势：**
- **可视化与可调试性**：每个节点的输入输出都是显式的，可以单独测试和监控
- **并行执行**：Eino 框架自动识别无依赖的节点并行执行（如 Milvus + ES 双路检索）
- **状态管理**：通过 `compose.ProcessState` 在节点间共享状态，避免参数层层传递
- **中断/恢复**：Eino 原生支持 `compose.Interrupt()` / `compose.ResumeWithData()`，实现 Human-in-the-Loop

**代价：**
- 学习曲线较高，需要理解 Eino 的 Graph/Node/Edge 抽象
- 调试时调用栈较深，不如直接函数调用直观
- 框架耦合度高，迁移到其他框架成本大

### 2.2 四层架构

```
┌─────────────────────────────────────────────┐
│  API 层 (api/)                               │
│  Gin 路由、SSE 流式、请求校验                  │
├─────────────────────────────────────────────┤
│  Flow 编排层 (flow/)                          │
│  DAG 图定义、节点编排、状态流转                  │
├─────────────────────────────────────────────┤
│  能力层 (model/ + rag/ + tool/)               │
│  模型调用、RAG 管线、工具执行                    │
├─────────────────────────────────────────────┤
│  基础设施层 (config/ + storage/ + algorithm/)  │
│  配置、存储、算法                               │
└─────────────────────────────────────────────┘
```

**为什么这样分层？**

API 层只负责 HTTP 协议转换，不包含业务逻辑；Flow 层定义"做什么"和"怎么做"；能力层提供具体能力；基础设施层提供通用支撑。这种分层使得：
- 替换 HTTP 框架（如从 Gin 切换到 Fiber）只影响 API 层
- 新增模型提供商只需在 model/ 下添加文件并注册
- RAG 管线可以独立于 Flow 层被其他系统复用

### 2.3 初始化编排（main.go）

```go
Config -> Redis -> Milvus -> Indexers -> Retrievers -> Parser/Loader/Splitter
-> Tracing(LangSmith+CozeLoop) -> MCP Tools -> IndexingGraph -> RAGChatFlow -> FinalGraph -> HTTP Server
```

**为什么采用启动时全量初始化？**
- 所有 Graph 在启动时编译并缓存（`sync.Once`），避免请求时的编译开销
- MCP 工具通过 `npx -y mcp-server-mysql` 启动子进程，连接建立是昂贵操作
- Milvus 连接池、ES 客户端都需要预热

**潜在问题：** 任何一个组件初始化失败都会导致整个服务不可用。没有"降级启动"机制。

---

## 3. Flow 编排层详解

### 3.1 RAG Chat Flow（RAG 对话流）

```
START -> PreProcess -> Rewrite -> QueryToMsgs -> Retrieve -> ConstructMessages -> Chat -> END
```

**各节点职责：**

| 节点 | 类型 | 职责 |
|------|------|------|
| PreProcess | LambdaNode | 从 Store 加载 Session，存入 State |
| Rewrite | LambdaNode | 利用记忆上下文重写查询，消除指代消解问题 |
| QueryToMsgs | LambdaNode | string -> []*schema.Message 类型转换 |
| Retrieve | GraphNode | 嵌入检索子图（Milvus + ES 并行 -> RRF 融合） |
| ConstructMessages | LambdaNode | 组装最终 Prompt：摘要 + 历史 + 检索文档 + 查询 |
| Chat | ChatModelNode | LLM 生成响应；后处理：更新历史 + 异步压缩 |

**设计亮点：**
- **Query 重写**：在检索前用 LLM 将"它怎么样？"重写为"XX产品的价格怎么样？"，显著提升检索准确率
- **异步压缩**：响应返回后才触发记忆压缩，不阻塞用户请求

**设计缺陷：**
- `Rewrite` 节点在没有历史时直接透传，但没有处理 Rewrite 失败的情况
- `ConstructMessages` 将所有检索文档拼接，没有长度限制，可能超出上下文窗口
- Session 并发写入没有保护（多个请求同时操作同一个 Session 的 History）

### 3.2 SQL React Flow（SQL 生成流）

```
START -> SQL_Retrieve -> ToTplVar -> SQL_Tpl -> SQL_Model -> Format_Response -> Trans_One
    ├── IsSQL=true  -> Approve (Interrupt!) -> [resume] Trans_List -> END
    └── IsSQL=false -> Trans_List -> END
```

**关键设计：Human-in-the-Loop 审批**

```go
// sql_react.go: Approve 节点
func Approve(ctx context.Context, input *schema.Message, state *FinalGraphRequest) ([]*schema.Message, error) {
    sql := input.Content
    interruptInfo := compose.Interrupt(ctx, sql)  // 暂停图执行
    // 用户审批后恢复...
}
```

**为什么需要 SQL 审批？**
- LLM 生成的 SQL 可能包含 `DROP`、`DELETE` 等危险操作
- SQL 语义可能与用户意图不一致
- 审批提供了"刹车"机制，确保安全

**设计亮点：**
- 审批不通过时，用户可以提供修改意见，系统会回到 `SQL_Tpl` 节点重新生成（Refine 循环）
- 使用 `format_response` 工具强制 LLM 输出结构化 JSON（`{answer, is_sql}`），避免解析歧义

**设计缺陷：**
- `state.Condition == "SOL"` 是个拼写错误，应该是 `"SQL"`（`final_graph.go` 中）
- Refine 循环没有最大次数限制，理论上可以无限循环
- MCP 工具执行 SQL 后没有结果校验，直接返回原始结果

### 3.3 Analyst Graph（数据分析流）

```
START -> ParseData -> AnalyzeData -> [GenerateReport, GenerateChart] (并行) -> MergeResult -> END
```

**设计亮点：**
- `GenerateReport` 和 `GenerateChart` 并行执行，利用 Eino 的自动并行能力
- `RecommendChartType` 根据数据特征自动选择图表类型（表格/折线/柱状/饼图）
- `ComputeStatistics` 自动识别数值列，计算均值、中位数、标准差、四分位数

**设计缺陷：**
- 只分析第一个数值列，忽略其他列
- `SimpleTableExtractor` 和 `SimpleImageExtractor` 是空实现（placeholder）
- 图表生成仅支持 ECharts，没有可插拔的图表引擎抽象

### 3.4 Final Graph（主调度图）

```
START -> Intent_Model -> React -> [条件分支]
    ├── Condition="SQL" -> ToToolCall -> MCP -> END
    └── Condition="Chat" -> END
```

**设计意图：** 这是一个"意图路由器"，根据用户输入分类为 SQL 查询或普通对话，分发到不同的子图。

**设计缺陷：**
- `Condition == "SOL"` 拼写错误会导致 SQL 路径永远走不到
- Intent 分类只支持两种意图（SQL/Chat），扩展性差
- Chat 路径直接结束，没有接入 RAG Chat Flow（功能不完整）

---

## 4. RAG 检索增强生成

### 4.1 索引管线（Indexing Graph）

```
START -> Loader -> Parser -> Splitter -> [Milvus, ES] (并行) -> Merge -> END
```

**文档处理链：**

| 阶段 | 实现 | 参数 |
|------|------|------|
| Loader | FileLoader | UseNameAsID=true |
| Parser | ExtParser（按扩展名分发） | 支持 HTML/PDF/TXT/DOCX/MD |
| Splitter | RecursiveTextSplitter | ChunkSize=1000, Overlap=200 |
| Milvus Indexer | 批量写入 | BatchSize=10, 字段: id/vector/content/metadata |
| ES Indexer | 单条写入 | 字段: content/content_vector/metadata |

**设计亮点：**
- Milvus Indexer 启动时自动检测 Embedding 维度，如果现有 Collection 维度不匹配会自动删除重建
- DOCX 解析器通过 XML Tokenizer 直接解析 `<w:t>` 元素，避免依赖重量级的 Word 库
- 分块 ID 使用 `{originalID}_chunk_{index}` 格式，可追溯来源

**设计缺陷：**
- 分块策略固定（1000 字符 + 200 重叠），没有根据文档类型自适应
- PDF 解析的 `SimpleTableExtractor` 和 `SimpleImageExtractor` 是空实现
- 没有文档去重机制，重复上传会产生冗余向量
- 没有增量更新能力，每次都是全量写入

### 4.2 检索管线（Retriever Graph）

```
START -> Trans_String -> [MilvusRetriever, ESRetriever] (并行) -> Reranker(RRF) -> END
```

**双路检索 + RRF 融合：**

```go
// rag_flow/retriever.go: Reranker 节点
// RRF: score += 1/(k+rank+1), k=60
for _, doc := range milvusDocs {
    key := doc.ID
    scores[key] += 1.0 / float64(60+rank+1)
}
// 同样处理 ES 结果，按总分排序，取 TopK
```

**为什么选择 RRF 而不是加权融合？**
- RRF 不需要调参（不需要设定向量检索和关键词检索的权重比例）
- RRF 基于排名而非分数，对不同检索器的分数尺度差异天然免疫
- RRF 在学术界和工业界都被验证有效（Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods, SIGIR 2009）

**设计亮点：**
- 检索结果缓存（`RetrievalCache`）：相同查询的 Embedding 和检索结果都缓存到 Redis，TTL=1h
- 缓存使用 SHA256(query) 作为 key，避免存储原始查询文本

**设计缺陷：**
- ES Retriever 使用的是 `DenseVectorSimilarity` 模式（向量检索），而不是真正的 BM25 关键词检索。这意味着两路检索本质上都是向量检索，只是存储在不同引擎中，失去了"语义 + 关键词"互补的优势
- BM25 算法虽然实现了（`algorithm/bm25.go`），但完全没有被使用
- 没有重排序（Cross-Encoder Reranker）阶段，RRF 只是简单融合排名
- 检索结果没有去重，同一文档的不同 chunk 可能同时被返回

### 4.3 Embedding 维度管理

系统在 Milvus Indexer 初始化时会：
1. 调用 `getEmbeddingDim()` 通过 Embedding "dim" 这个文本获取向量维度
2. 检查现有 Collection 的向量字段维度是否匹配
3. 如果不匹配，自动删除 Collection 并重建

**为什么需要这个机制？** 切换 Embedding 模型时（如从 Qwen 切换到 Ark），维度会变化，旧数据与新模型不兼容。自动重建避免了手动干预。

**风险：** 自动删除 Collection 是破坏性操作，没有确认机制。

---

## 5. 记忆系统（Memory）

### 5.1 两级记忆模型

| 层级 | 存储 | 内容 | 生命周期 |
|------|------|------|---------|
| 短期记忆 | Session.History | 最近 3 轮对话原文 | 压缩触发前 |
| 长期记忆 | Session.Summary | LLM 生成的摘要 | 跨压缩周期累积 |

### 5.2 压缩算法

```go
// compressor.go
func (s *Summarizer) Compress(ctx context.Context, sess *Session) error {
    if len(sess.History) <= s.MaxHistoryLen { return nil }  // 门控
    toCompress := sess.History[:len(sess.History)-3]        // 取旧消息
    sess.History = sess.History[len(sess.History)-3:]       // 保留最近 3 轮
    // LLM 调用：合并 previous_summary + older_messages -> new_summary
    resp, _ := s.Model.Generate(ctx, messages)
    sess.Summary = resp.Content
}
```

**压缩 Prompt（中文）：**
> 你是一个对话摘要助手。请将以下旧对话与已有摘要合并为一段简洁的新摘要。
> 要求：保留核心事实、用户偏好、未解决的问题；去掉寒暄和重复的中间步骤；保持连贯性。

**设计思考：**
- 为什么保留 3 轮而不是更多？ → 平衡上下文连贯性和 Token 消耗
- 为什么用 LLM 摘要而不是简单截断？ → 摘要保留语义信息，截断直接丢失
- 为什么异步执行？ → 压缩是 LLM 调用，耗时长，不能阻塞响应

**设计缺陷：**
- 压缩是**有损**的：原始消息一旦被压缩就永久丢失
- 没有错误恢复：如果 LLM 调用失败，History 已经被截断但 Summary 没有更新
- Summary 没有长度限制，长期对话后可能很大
- 没有并发保护：多个请求同时操作同一个 Session 会竞争
- `UpdatedAt` 字段从未被设置（死代码）

### 5.3 记忆在 RAG 中的三处使用

1. **Query 重写**（检索前）：用 Summary + History 上下文化查询
2. **Prompt 组装**（生成时）：Summary 作为 System Message + History 作为对话历史
3. **压缩**（响应后）：异步压缩旧消息为新 Summary

---

## 6. 模型抽象层

### 6.1 工厂注册模式

```go
// chat_model/chat_model.go
type ChatModelFactory func(ctx context.Context) (model.ToolCallingChatModel, error)
var chatModelRegistry = make(map[string]ChatModelFactory)

func registerChatModel(name string, factory ChatModelFactory) {
    chatModelRegistry[name] = factory
}

func GetChatModel(ctx context.Context, name string) (model.ToolCallingChatModel, error) {
    factory, ok := chatModelRegistry[name]
    if !ok { return nil, fmt.Errorf("chat model %s not found", name) }
    return factory(ctx)
}
```

**为什么用工厂模式而不是直接实例化？**
- **解耦**：Flow 层不需要知道具体用的是 Ark 还是 OpenAI
- **可切换**：通过配置文件切换模型提供商，不改代码
- **可测试**：可以注入 Mock 模型进行单元测试

**当前注册状态：**

| 模型 | Chat Model | Embedding Model | 状态 |
|------|-----------|----------------|------|
| Ark (豆包) | 已注册 | 已注册 | 活跃 |
| OpenAI | 已注册 | 已注册 | 注释掉 |
| DeepSeek | 已注册 | - | 注释掉 |
| Gemini | 已注册 | 已注册 | 注释掉 |
| Qwen (通义) | 已注册 | 已注册 | Embedding 活跃 |

**设计缺陷：** 只有 Ark 的 `init()` 是激活的，其他都被注释掉。切换模型需要改代码重新编译，没有实现真正的配置驱动切换。

### 6.2 Format Tool（结构化输出）

```go
// format_tool/ark_format.go
type FormatOutput struct {
    Answer string `json:"answer" jsonschema_description:"..."`
    IsSQL  bool   `json:"is_sql" jsonschema_description:"..."`
}
```

**为什么需要 Format Tool？** LLM 的自由文本输出难以可靠解析。通过绑定 `format_response` 工具，强制 LLM 输出结构化 JSON，然后在 `Trans_One` 节点解析 `IsSQL` 字段决定后续流程。

这是 **Tool-For-Output** 模式的应用：不是为了"调用工具"，而是为了"约束输出格式"。

---

## 7. 工具层（Tool）

### 7.1 MCP SQL 执行

```go
// sql_tools/exectue.go
func connectMCP(ctx context.Context) ([]tool.BaseTool, error) {
    mcpClient := client.NewClient()
    transport := &client.CommandTransport{
        Command: "npx",
        Args:    []string{"-y", "mcp-server-mysql"},
        Env:     []string{"MYSQL_HOST=...", "MYSQL_PORT=...", ...},
    }
    mcpClient.Connect(ctx, transport)
    tools, _ := mcpClient.ListTools(ctx)
    // tools: mysql_query, list_tables, describe_table
}
```

**为什么用 MCP 而不是直接调用 MySQL？**
- **标准化**：MCP 是 Anthropic 提出的 Model Context Protocol，工具定义标准化
- **隔离**：SQL 执行通过子进程完成，即使崩溃也不影响主进程
- **可扩展**：未来可以接入其他 MCP Server（如 PostgreSQL、MongoDB）

### 7.2 文档处理

**解析器分发逻辑：**

| 扩展名 | 解析器 | 实现 |
|--------|--------|------|
| .html/.htm | HTMLParser | Eino 内置 |
| .pdf | PDFParser | Eino 内置 |
| .txt/.md | TextParser | Eino 内置 |
| .docx/.doc | DocxParser | 自实现 XML Tokenizer |
| 其他 | TextParser | 降级处理 |

**DOCX 解析实现要点：**
- 使用 `docx.ReadDocxFromMemory` 读取 ZIP 包
- 通过 Go 标准库 `encoding/xml` 的 Tokenizer 逐元素解析
- `<w:t>` 提取文本，`<w:p>` 插入换行，`<w:tab>` 插入制表符
- 返回单个 Document，不按段落分块

### 7.3 Redis 存储层

系统有三个独立的 Redis 存储：

| 存储 | 用途 | Key 前缀 | TTL |
|------|------|---------|-----|
| RedisCheckPointStore | 图中断/恢复 | `checkpoint:` | 24h |
| SessionStore | 审批会话上下文 | `session:` | 24h |
| RetrievalCache | 检索结果缓存 | `embedding:` / `retrieval:` | 1h |

**共同的降级模式：** 三个存储都实现了"Redis 不可用时自动降级到内存 Map"的模式。

```go
// checkpoint_store.go
func (r *RedisCheckPointStore) Get(ctx, checkpointID) ([]byte, bool, error) {
    if r.useFallback {
        return r.fallbackGet(checkpointID)
    }
    val, err := r.client.Get(ctx, r.prefix+checkpointID).Bytes()
    if err != nil {
        r.useFallback = true  // 首次 Redis 错误后永久降级
        return r.fallbackGet(checkpointID)
    }
    return val, true, nil
}
```

**设计缺陷：** 降级是永久性的——一旦 Redis 出现一次错误，就永远切换到内存模式，不会尝试恢复。

---

## 8. SFT 数据管线

### 8.1 架构

```
ChatModel 调用 -> SFTHandler.OnStart(记录输入) -> SFTHandler.OnEnd(记录输出)
    -> 异步 SaveSample -> Annotate(教师模型评分) -> 保存标注结果
    -> ExportToJSONL(导出训练数据)
```

### 8.2 关键组件

**SFTHandler（回调中间件）：**
- 实现 Eino 的 `callbacks.Handler` 接口
- `OnStart`：在 ChatModel 调用前记录输入消息
- `OnEnd`：在 ChatModel 调用后异步创建 Sample，调用教师模型标注

**Annotator（教师模型标注）：**
- 使用 DeepSeek 作为教师模型
- 评估维度：Score（0-1）、Reasoning（理由）、Correction（修正答案）
- 输出 JSON 格式，解析后更新 Sample

**SpeculativeEngine（推测解码）：**
```go
func (e *SpeculativeEngine) SpeculativeStream(ctx, msgs) (<-chan string, error) {
    // 1. 小模型先生成（快速 TTFT）
    draftChan := e.DraftModel.Stream(ctx, msgs)
    // 2. 大模型验证
    // 3. 如果大模型说 OK -> 使用小模型输出
    // 4. 如果大模型说 NO -> 发送 [CORRECTION_START] + 修正内容
}
```

**设计意图：** 小模型（Draft）响应快，大模型（Target）质量高。推测解码可以在保证质量的前提下降低首 Token 延迟（TTFT）。

### 8.3 数据导出

```go
func (m *Manager) ExportToJSONL(agentID, outputPath string, opts ExportOptions) (int, error) {
    // 过滤：MinScore >= 阈值 && OnlyLabeled == true
    // 输出格式：{"input": context_messages, "output": teacher_correction}
}
```

训练数据使用教师模型的 **Correction** 作为标签，而非原始 LLM 输出。这意味着训练目标是"让小模型学会大模型的修正能力"。

---

## 9. 存储与基础设施

### 9.1 Docker Compose 部署

```yaml
services:
  etcd:       # Milvus 元数据存储
  minio:      # Milvus 对象存储
  standalone: # Milvus 向量数据库 (v2.5.10)
  attu:       # Milvus Web UI
  es8:        # Elasticsearch 8.12.0 (单节点，关闭安全认证)
```

**设计思考：**
- 为什么 Milvus 用 standalone 模式？ → 开发/测试环境足够，生产环境应切换到分布式模式
- 为什么 ES 关闭安全认证？ → 简化开发环境配置，生产环境必须开启
- 为什么没有 Redis？ → Redis 通过 `storage.InitRedis()` 连接外部实例，不在 Docker Compose 中

### 9.2 配置管理

```go
// config/config.go
func LoadConfig() (*Config, error) {
    godotenv.Load(".env")
    // 逐个读取环境变量，带默认值
}
```

所有配置通过 `.env` 文件加载，使用 `godotenv` 库。配置结构体扁平化，没有分层。

---

## 10. 当前设计的优势与不足

### 10.1 优势

| 优势 | 说明 |
|------|------|
| **图编排引擎** | DAG 模型天然支持并行、中断/恢复、可视化 |
| **双路检索 + RRF** | 向量 + 关键词互补，RRF 无需调参 |
| **Human-in-the-Loop** | SQL 审批机制保障安全，支持 Refine 循环 |
| **SFT 数据管线** | 自动采集 + 教师标注，形成数据飞轮 |
| **优雅降级** | Redis 不可用时自动降级到内存，不影响核心功能 |
| **MCP 工具隔离** | SQL 执行通过 MCP 子进程，故障隔离 |
| **异步压缩** | 记忆压缩不阻塞响应 |

### 10.2 不足

| 不足 | 影响 | 严重程度 |
|------|------|---------|
| ES Retriever 用向量模式而非 BM25 | 双路检索失去语义+关键词互补优势 | 高 |
| 记忆系统只有 2 级，没有结构化记忆 | 长期对话中丢失关键实体和事实 | 高 |
| 没有显式 Token 计数 | 可能超出模型上下文窗口 | 高 |
| `Condition == "SOL"` 拼写错误 | SQL 路径可能无法正常工作 | 中 |
| Session 并发写入无保护 | 高并发下数据竞争 | 中 |
| 检索结果没有 Cross-Encoder 重排序 | 检索精度有提升空间 | 中 |
| 模型切换需要改代码 | 运维不友好 | 中 |
| 降级后不会恢复 Redis | 临时网络抖动导致永久降级 | 低 |
| 分块策略固定 | 不同文档类型需要不同分块策略 | 低 |

---

## 11. 优化方案：基于最新 Agent 技术的改进路线

### 11.1 检索优化：真正的混合检索 + 重排序

**当前问题：** ES Retriever 使用 `DenseVectorSimilarity` 模式，本质上和 Milvus 一样都是向量检索。BM25 算法实现了但没被使用。

**改进方案：**

```
当前：  Query -> [Milvus(向量), ES(向量)] -> RRF
改进后：Query -> [Milvus(向量), ES(BM25关键词)] -> RRF -> CrossEncoder Reranker
```

**具体实现：**

1. **ES 切换为真正的关键词检索：**
```go
// rag/rag_tools/retriever/es.go
// 将 SearchModeDenseVectorSimilarity 改为 SearchModeBM25
// 使用 ES 的 match_query 而非 knn_query
```

2. **添加 Cross-Encoder Reranker：**
```go
// 新增 tool/reranker/reranker.go
type CrossEncoderReranker struct {
    Model model.BaseChatModel  // 或专用的 Cross-Encoder 模型
}

func (r *CrossEncoderReranker) Rerank(ctx context.Context, query string, docs []*schema.Document) ([]*schema.Document, error) {
    // 对每个 (query, doc) 对计算相关性分数
    // 按分数重新排序
}
```

3. **在 Retriever Graph 的 Reranker 节点后添加 Cross-Encoder 阶段：**
```
START -> Trans_String -> [MilvusRetriever, ESRetriever] -> RRF -> CrossEncoderReranker -> END
```

**预期效果：** 检索准确率提升 15-30%（基于学术界 Cross-Encoder Reranker 的典型提升幅度）。

### 11.2 记忆系统升级：三级记忆 + 结构化知识

**当前问题：** 只有短期（History）和长期（Summary）两级，没有结构化记忆。长期对话中关键实体和事实会在摘要中逐渐丢失。

**改进方案：三级记忆架构**

```
┌─────────────────────────────────────────┐
│  L1: 工作记忆 (Working Memory)           │
│  最近 3-5 轮对话原文                      │
│  存储：内存                               │
├─────────────────────────────────────────┤
│  L2: 摘要记忆 (Summary Memory)           │
│  LLM 生成的对话摘要                       │
│  存储：Redis/内存                         │
├─────────────────────────────────────────┤
│  L3: 知识记忆 (Knowledge Memory)          │
│  结构化实体/事实/偏好                      │
│  存储：向量数据库 + KV 存储                │
└─────────────────────────────────────────┘
```

**L3 知识记忆的具体实现：**

```go
// tool/memory/knowledge.go (新增)
type KnowledgeMemory struct {
    Entities  map[string]Entity   // 实体: {名称 -> 属性}
    Facts     []Fact              // 事实列表
    Preferences map[string]string // 用户偏好
}

type Entity struct {
    Name       string
    Type       string            // 人名/地名/产品名/...
    Attributes map[string]string // 属性键值对
    LastUpdate time.Time
}

type Fact struct {
    Content   string
    Source    string            // 来自哪轮对话
    Timestamp time.Time
    Confidence float64          // 置信度
}
```

**知识提取 Prompt：**
```
从以下对话中提取结构化信息：
1. 实体（人名、产品名、地名等）及其属性
2. 事实性陈述
3. 用户偏好

输出 JSON 格式：
{
  "entities": [...],
  "facts": [...],
  "preferences": [...]
}
```

**三处使用方式：**
1. **Query 重写**：注入相关实体信息，提升指代消解准确率
2. **Prompt 组装**：注入相关事实和偏好，生成更个性化的回答
3. **检索增强**：用实体信息扩展检索查询

### 11.3 上下文窗口管理：显式 Token 计数

**当前问题：** 没有 Token 计数，Summary + History + RAG Documents + Query 的总长度可能超出模型上下文窗口。

**改进方案：**

```go
// tool/token_counter/counter.go (新增)
type TokenCounter struct {
    tokenizer tiktoken.Encoding  // 或其他 tokenizer
}

func (t *TokenCounter) Count(text string) int {
    tokens, _ := t.tokenizer.Encode(text, nil)
    return len(tokens)
}

func (t *TokenCounter) FitToBudget(parts []string, maxTokens int) []string {
    // 贪心算法：从最重要部分开始，逐个添加直到达到预算
    total := 0
    var result []string
    for _, part := range parts {
        count := t.Count(part)
        if total+count > maxTokens {
            break
        }
        result = append(result, part)
        total += count
    }
    return result
}
```

**在 ConstructMessages 节点中的应用：**

```go
func ConstructMessages(ctx context.Context, state *GraphState) ([]*schema.Message, error) {
    counter := GetTokenCounter()
    budget := 4096  // 预留给响应的 Token

    parts := []string{}
    if state.Session.Summary != "" {
        parts = append(parts, "背景摘要: "+state.Session.Summary)
    }
    // ... history, docs, query ...

    fitted := counter.FitToBudget(parts, modelContextWindow-budget)
    // 用 fitted 组装最终 Prompt
}
```

### 11.4 Agentic RAG：让 Agent 控制检索

**当前问题：** 检索是固定的管线（检索一次 -> 生成），没有让 Agent 判断"是否需要检索"或"检索结果是否足够"。

**改进方案：引入 Agentic RAG 循环**

```
START -> Agent(判断是否需要检索)
    ├── 不需要检索 -> 直接生成
    └── 需要检索 -> Retrieve -> Agent(评估检索质量)
        ├── 质量足够 -> 生成
        └── 质量不足 -> 改写查询 -> Retrieve -> ... (最多 N 次)
```

**实现思路：**

```go
// flow/agentic_rag.go (新增)
func buildAgenticRAGFlow(ctx context.Context) {
    graph := compose.NewGraph[...]()

    // 判断节点：LLM 决定是否需要检索
    graph.AddLambdaNode("Decide", decideNeedRetrieve)
    // 评估节点：LLM 评估检索结果质量
    graph.AddLambdaNode("Evaluate", evaluateRetrievalQuality)
    // 改写节点：基于评估结果改写查询
    graph.AddLambdaNode("RewriteQuery", rewriteBasedOnFeedback)

    // 边：Decide -> [Retrieve | Generate]
    // 边：Retrieve -> Evaluate -> [Generate | RewriteQuery -> Retrieve]
}
```

**预期效果：** 简单问题（如"你好"）跳过检索，节省延迟和 Token；复杂问题可以多轮检索迭代，提升答案质量。

### 11.5 SQL 安全增强

**当前问题：** SQL 审批只有"批准/拒绝"，没有 SQL 静态分析。

**改进方案：添加 SQL 安全分析层**

```go
// tool/sql_tools/safety.go (新增)
type SQLSafetyChecker struct{}

func (c *SQLSafetyChecker) Check(sql string) (*SafetyReport, error) {
    report := &SafetyReport{}

    // 1. 语法解析
    stmt, err := sqlparser.Parse(sql)

    // 2. 危险操作检测
    if hasDropTable(stmt) { report.Risks = append(report.Risks, "DROP TABLE") }
    if hasDeleteWithoutWhere(stmt) { report.Risks = append(report.Risks, "DELETE without WHERE") }
    if hasTruncate(stmt) { report.Risks = append(report.Risks, "TRUNCATE") }

    // 3. 影响范围估算
    report.EstimatedRows = estimateAffectedRows(stmt)

    // 4. 权限检查
    report.RequiredPermissions = getRequiredPermissions(stmt)

    return report, nil
}
```

**在 Approve 节点中的应用：**
```go
func Approve(ctx context.Context, input *schema.Message, state *FinalGraphRequest) {
    checker := &SQLSafetyChecker{}
    report, _ := checker.Check(sql)

    if len(report.Risks) > 0 {
        // 高风险操作：需要额外确认
        interruptInfo := compose.Interrupt(ctx, fmt.Sprintf(
            "⚠️ SQL 安全分析：\n风险: %v\n预计影响行数: %d\n\n是否继续执行？",
            report.Risks, report.EstimatedRows))
    }
}
```

### 11.6 多模型路由：配置驱动 + 降级链

**当前问题：** 模型切换需要改代码，没有降级机制。

**改进方案：**

```yaml
# config.yaml (新增)
models:
  primary:
    provider: ark
    model: doubao-seed-2-0
  fallback:
    provider: deepseek
    model: deepseek-chat
  intent:
    provider: qwen
    model: qwen-turbo  # 意图分类用小模型，省成本
  rewrite:
    provider: qwen
    model: qwen-turbo  # 查询重写用小模型
  summarizer:
    provider: ark
    model: doubao-seed-2-0
  analyst:
    provider: ark
    model: doubao-seed-2-0

routing:
  strategy: cost_optimized  # cost_optimized | quality_first | latency_first
  fallback_chain: [primary, fallback]
  timeout_ms: 30000
```

```go
// model/chat_model/router.go (新增)
type ModelRouter struct {
    configs  map[string]ModelConfig
    strategy RoutingStrategy
}

func (r *ModelRouter) GetModel(ctx context.Context, purpose string) (model.ToolCallingChatModel, error) {
    config := r.configs[purpose]
    model, err := r.tryCreate(ctx, config.Primary)
    if err != nil {
        for _, fallback := range config.FallbackChain {
            model, err = r.tryCreate(ctx, fallback)
            if err == nil { break }
        }
    }
    return model, err
}
```

### 11.7 并发安全：Session 锁优化

**当前问题：** 多个请求同时操作同一个 Session 的 History 会竞争。

**改进方案：**

```go
// tool/memory/store.go (改进)
type InMemoryStore struct {
    mu   sync.RWMutex
    data map[string]*Session
    locks map[string]*sync.Mutex  // 每个 Session 独立的锁
}

func (s *InMemoryStore) WithLock(sessionID string, fn func(*Session) error) error {
    s.mu.RLock()
    lock, exists := s.locks[sessionID]
    s.mu.RUnlock()

    if !exists {
        s.mu.Lock()
        if s.locks[sessionID] == nil {
            s.locks[sessionID] = &sync.Mutex{}
        }
        lock = s.locks[sessionID]
        s.mu.Unlock()
    }

    lock.Lock()
    defer lock.Unlock()

    sess, _ := s.Get(context.Background(), sessionID)
    return fn(sess)
}
```

### 11.8 反思机制（Reflection/Self-Critique）

**当前问题：** LLM 生成响应后直接返回，没有自我检查。

**改进方案：在 Chat 节点后添加反思节点**

```
... -> Chat -> Reflect -> [Accept | Revise -> Chat] -> END
```

```go
// 反思 Prompt
const ReflectPrompt = `你是一个质量审查员。请评估以下回答：
- 是否准确回答了用户的问题？
- 是否有事实错误？
- 是否有遗漏的重要信息？
- 检索到的文档是否被正确引用？

输出 JSON: {"pass": true/false, "issues": [...], "suggestion": "..."}`
```

### 11.9 改进优先级

| 优先级 | 改进项 | 预期收益 | 实现复杂度 |
|--------|--------|---------|-----------|
| P0 | 修复 `Condition == "SOL"` 拼写错误 | 修复 SQL 路径 | 极低 |
| P0 | ES 切换为真正的 BM25 检索 | 检索质量大幅提升 | 低 |
| P0 | 添加 Token 计数 + 上下文窗口管理 | 防止超出模型限制 | 中 |
| P1 | Session 并发安全 | 防止数据竞争 | 低 |
| P1 | Cross-Encoder Reranker | 检索精度提升 | 中 |
| P1 | 模型配置驱动切换 | 运维友好 | 中 |
| P2 | 三级记忆（知识记忆） | 长期对话质量 | 高 |
| P2 | Agentic RAG 循环 | 复杂问题回答质量 | 高 |
| P2 | SQL 安全分析 | 生产安全 | 中 |
| P3 | 反思机制 | 回答质量 | 中 |
| P3 | Redis 降级恢复 | 系统健壮性 | 低 |

---

## 附录：文件路径速查

| 模块 | 关键文件 |
|------|---------|
| 入口 | `main.go` |
| 配置 | `config/config.go` |
| API 路由 | `api/router.go`, `api/final_graph.go`, `api/rag_chat_stream.go` |
| RAG Chat | `flow/rag_chat.go` |
| SQL React | `flow/sql_react.go` |
| 数据分析 | `flow/analyst_graph.go` |
| 主调度 | `flow/final_graph.go` |
| Chat Model | `model/chat_model/chat_model.go`, `ark.go`, `openai.go`, ... |
| Embedding | `model/embedding_model/embedding_model.go`, `qwen.go`, ... |
| 索引图 | `rag/rag_flow/index.go` |
| 检索图 | `rag/rag_flow/retriever.go` |
| Milvus Indexer | `rag/rag_tools/indexer/milvus.go` |
| ES Retriever | `rag/rag_tools/retriever/es.go` |
| 记忆系统 | `tool/memory/session.go`, `store.go`, `compressor.go` |
| Redis 存储 | `tool/storage/redis.go`, `checkpoint_store.go`, `retrieval_cache.go` |
| 文档处理 | `tool/document/parser.go`, `loader.go`, `splitter.go`, `docx_parser.go` |
| SQL 工具 | `tool/sql_tools/exectue.go`, `generate.go` |
| SFT 管线 | `tool/sft/callback.go`, `annotator.go`, `speculative.go`, `storage.go` |
| 算法 | `algorithm/bm25.go`, `algorithm/rrf.go` |
| 基础设施 | `docker-compose.yaml` |
