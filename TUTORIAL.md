# Agent Platform 新手研读指南

> 本文档面向 Python 初学者和 AI Agent 初学者，从零开始讲解这个项目的架构、技术选型、代码走读。
> 目标：读完后你能理解一个 AI Agent 项目是怎么组织的，一条用户查询是怎么从浏览器跑到 LLM 再返回的。

---

## 目录

1. [这个项目是干什么的](#1-这个项目是干什么的)
2. [技术栈全景：用了什么、为什么用](#2-技术栈全景用了什么为什么用)
3. [项目文件结构：每个文件夹的职责](#3-项目文件结构每个文件夹的职责)
4. [一条 Chat Query 的完整旅程](#4-一条-chat-query-的完整旅程)
5. [核心算法详解](#5-核心算法详解)
6. [Python 编程技巧在这个项目中的体现](#6-python-编程技巧在这个项目中的体现)
7. [AI Agent 核心概念](#7-ai-agent-核心概念)
8. [如何自己动手改这个项目](#8-如何自己动手改这个项目)

---

## 1. 这个项目是干什么的

一句话：**用户用自然语言提问，系统自动查数据库、查知识库，返回答案。**

具体场景：
- 用户问 "各部门经理是谁？" → 系统自动生成 SQL 查数据库 → 返回表格
- 用户问 "公司的报销制度是什么？" → 系统从知识库检索相关文档 → 返回回答
- 用户问 "你好" → 系统直接用 LLM 闲聊

```
用户浏览器
  │
  │  HTTP 请求 (POST /api/rag/chat/stream)
  ▼
FastAPI 服务器
  │
  │  路由到对应的处理函数
  ▼
LangGraph 图 (流水线)
  │
  ├── 1. 加载记忆
  ├── 2. 查询重写
  ├── 3. 文档检索 (Milvus + Elasticsearch)
  ├── 4. 拼装 Prompt
  └── 5. 调用 LLM
  │
  │  SSE 流式响应
  ▼
用户浏览器 (逐字显示)
```

---

## 2. 技术栈全景：用了什么、为什么用

### 2.1 Web 框架：FastAPI

**是什么**：Python 的 Web 框架，用来接收 HTTP 请求、返回响应。

**为什么不用 Flask？** FastAPI 原生支持 `async/await`（异步），能同时处理很多请求而不会互相阻塞。Flask 是同步的，一个请求在等 LLM 回复时，其他请求都得排队。

**类比**：FastAPI 像一个餐厅前台，能同时接待多桌客人点餐。Flask 像一个只有一个服务员的前台，必须等一桌点完才能接下一桌。

```python
# FastAPI 的写法 — 注意 async def
@router.post("/ask")
async def rag_ask(req: RAGChatRequest):
    result = await graph.ainvoke(...)   # await = "去干别的事，等好了叫我"
    return {"answer": result["answer"]}
```

### 2.2 AI 编排：LangChain + LangGraph

**LangChain 是什么**：一个 Python 库，提供统一接口调用各种 LLM（OpenAI、豆包、通义千问等）。

**LangGraph 是什么**：LangChain 生态里的"流水线编排器"。你定义一系列步骤（节点）和步骤之间的连接关系（边），它帮你按顺序执行。

**为什么需要 LangGraph？** 因为一个 AI 应用不是"调一次 LLM 就完了"，而是多个步骤串联：
1. 先查记忆 → 2. 改写问题 → 3. 搜文档 → 4. 拼 Prompt → 5. 调 LLM

LangGraph 让你用"画流程图"的方式写代码：

```python
# 定义流程图
graph = StateGraph(RAGChatState)

# 添加节点（每个节点是一个处理函数）
graph.add_node("preprocess", preprocess)
graph.add_node("rewrite", rewrite)
graph.add_node("retrieve", retrieve)
graph.add_node("chat", chat)

# 添加边（节点之间的连接）
graph.add_edge(START, "preprocess")      # 开始 → 预处理
graph.add_edge("preprocess", "rewrite")  # 预处理 → 查询重写
graph.add_edge("rewrite", "retrieve")    # 查询重写 → 检索
graph.add_edge("retrieve", "chat")       # 检索 → 对话
graph.add_edge("chat", END)              # 对话 → 结束

# 编译成可执行的图
app = graph.compile()
```

画成图就是：
```
START → preprocess → rewrite → retrieve → chat → END
```

**类比**：LangGraph 像工厂的流水线。每个工位（节点）做一件事，产品（数据）沿着传送带（边）流过每个工位。

### 2.3 向量数据库：Milvus

**是什么**：专门存储和搜索"向量"的数据库。

**为什么需要它？** 传统数据库搜索靠关键词匹配（"经理" 只能找到包含 "经理" 两个字的记录）。但用户可能问 "各部门领导是谁"，"领导" 和 "经理" 意思一样，关键词匹配就找不到了。

向量数据库的做法：
1. 把每段文本通过 Embedding 模型转成一个向量（一串数字，比如 `[0.12, -0.34, 0.56, ...]`）
2. 用户查询也转成向量
3. 计算查询向量和所有文档向量的"距离"，距离最近的就是最相关的

```
"各部门经理是谁" → [0.12, -0.34, 0.56, ...]  ─┐
                                                 │ 计算距离
"表名: t_user, 字段: manager_name" → [0.11, -0.33, 0.55, ...]  ─┘ → 距离很近！找到了！
"表名: t_order, 字段: amount" → [0.88, 0.22, -0.11, ...]  ─┘ → 距离远，不相关
```

**类比**：传统数据库像字典，你必须知道确切的词才能查。向量数据库像"语义地图"，意思相近的内容在地图上靠得近。

### 2.4 全文搜索：Elasticsearch

**是什么**：分布式全文搜索引擎。

**为什么和 Milvus 配合使用？** 两种搜索各有擅长：
- **Milvus（向量搜索）**：擅长语义匹配（"领导" ≈ "经理"），但对精确关键词（表名 `t_user`）可能不准
- **Elasticsearch（BM25 关键词搜索）**：擅长精确匹配（搜 "t_user" 就能找到 "t_user"），但不懂同义词

两者配合，取长补短。这就是"混合检索"。

### 2.5 LLM：大语言模型

**是什么**：就是 ChatGPT、豆包这类模型。输入一段文字，它生成一段回复。

**本项目怎么用它**：通过 LangChain 的统一接口调用，不直接写 HTTP 请求：

```python
# 不同 LLM 用同一个接口
model = get_chat_model("ark")      # 豆包
# model = get_chat_model("openai") # OpenAI
# model = get_chat_model("qwen")   # 通义千问

response = await model.ainvoke([HumanMessage(content="你好")])
print(response.content)  # "你好！有什么可以帮你的？"
```

**为什么抽象一层？** 因为业务代码不应该关心底层用的是哪个 LLM。今天用豆包，明天想换成 OpenAI，只需要改配置，不用改业务代码。

### 2.6 缓存：Redis

**是什么**：内存数据库，读写极快。

**用在哪里**：
- 缓存对话 Session（避免每次都从头加载）
- 缓存检索结果（同一个查询搜第二次时直接返回缓存）
- 存储 LangGraph 的 checkpoint（支持对话中断后恢复）

### 2.7 技术栈总结

```
┌─────────────────────────────────────────────────────┐
│                    用户浏览器                          │
│              (index.html + JavaScript)                │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼──────────────────────────────┐
│              FastAPI (Web 服务器)                      │
│         路由、请求校验、SSE 流式响应                     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│           LangGraph (流水线编排)                       │
│     定义节点和边，管理状态，支持中断/恢复                  │
└──────┬───────────────┬──────────────────┬───────────┘
       │               │                  │
┌──────▼──────┐ ┌──────▼──────┐ ┌────────▼────────┐
│  LangChain  │ │  Milvus     │ │ Elasticsearch   │
│  (LLM 调用) │ │  (向量搜索)  │ │  (关键词搜索)    │
└──────┬──────┘ └─────────────┘ └─────────────────┘
       │
┌──────▼──────┐
│  LLM API    │
│ (豆包/OpenAI)│
└─────────────┘
```

---

## 3. 项目文件结构：每个文件夹的职责

```
agents/
│
├── main.py                      # 程序入口（启动服务器）
├── config/
│   └── settings.py              # 所有配置项（从环境变量读取）
│
├── api/                         # 【接待层】接收 HTTP 请求
│   ├── app.py                   # FastAPI 应用初始化
│   ├── sse.py                   # SSE 流式响应工具
│   └── routers/
│       ├── rag.py               # RAG 对话端点 (/api/rag/...)
│       ├── final.py             # 主调度端点 (/api/final/...)
│       └── chat.py              # 测试端点
│
├── flow/                        # 【流程层】LangGraph 图定义
│   ├── state.py                 # 所有图的状态类型定义
│   ├── rag_chat.py              # RAG 对话图（知识问答）
│   ├── sql_react.py             # SQL 生成图（数据库查询）
│   └── final_graph.py           # 主调度图（意图分类 + 路由）
│
├── rag/                         # 【检索层】文档检索相关
│   ├── retriever.py             # 混合检索器（Milvus + ES + RRF）
│   ├── parent_retriever.py      # 父子文档检索器
│   ├── reranker.py              # Cross-Encoder 重排序
│   ├── query_rewrite.py         # 查询重写（消解指代）
│   └── schema_indexer.py        # MySQL 表结构索引
│
├── model/                       # 【模型层】LLM 和 Embedding
│   ├── chat_model.py            # Chat Model 工厂
│   └── providers/               # 各厂商实现（ark, openai, qwen...）
│
├── tool/                        # 【工具层】各种辅助功能
│   ├── memory/                  # 对话记忆系统
│   ├── storage/                 # 存储（Redis, 缓存）
│   ├── sql_tools/               # SQL 执行工具（MCP 协议）
│   └── token_counter.py         # Token 计数器
│
├── algorithm/                   # 【算法层】核心算法
│   └── rrf.py                   # Reciprocal Rank Fusion
│
├── eval/                        # 【评估层】检索质量评估
│   ├── dataset_generator.py     # 评测数据集生成
│   ├── metrics.py               # 评估指标
│   └── runner.py                # 评估运行器
│
└── static/
    └── index.html               # 前端页面
```

**类比**：把项目想象成一家餐厅：
- `api/` = 前台（接待客人、下单）
- `flow/` = 后厨流水线（洗菜 → 切菜 → 炒菜 → 装盘）
- `rag/` = 食材仓库（从不同货架找食材）
- `model/` = 厨师（真正做菜的人，也就是 LLM）
- `tool/` = 厨具（刀、锅、砧板）
- `algorithm/` = 菜谱里的技巧（比如 "食材搭配法则"）
- `config/` = 餐厅规章制度

---

## 4. 一条 Chat Query 的完整旅程

让我们追踪一个真实请求：用户在聊天框输入 "各部门经理是谁"，按下发送。

### 第 1 步：浏览器发送请求

```javascript
// 前端 index.html 中的 JavaScript
const resp = await fetch('/api/rag/chat/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        query: "各部门经理是谁",
        session_id: "user_123"
    }),
});
```

浏览器向服务器发送一个 POST 请求，携带用户的查询和会话 ID。

### 第 2 步：FastAPI 路由

```python
# agents/api/routers/rag.py
@router.post("/chat/stream")
async def rag_chat_stream(req: RAGChatRequest, request: Request):
    graph = build_rag_chat_graph()          # 构建 LangGraph 图
    # ... 启动流式生成 ...
```

FastAPI 根据 URL `/api/rag/chat/stream` 找到对应的处理函数。这个函数做了两件事：
1. 构建 LangGraph 图（流水线）
2. 启动流式生成

### 第 3 步：LangGraph 图开始执行

图的定义在 `agents/flow/rag_chat.py` 的 `build_rag_chat_graph()` 函数中：

```python
# agents/flow/rag_chat.py
def build_rag_chat_graph():
    graph = StateGraph(RAGChatState)
    graph.add_node("preprocess", preprocess)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("construct_messages", construct_messages)
    graph.add_node("chat", chat)
    # ... 连接边 ...
    return graph.compile()
```

图执行时，数据像流水一样流过每个节点。每个节点接收**上一个节点的输出**作为输入，处理后把自己的输出传递给下一个节点。

这个"传递的数据"叫做 **State（状态）**，定义在 `agents/flow/state.py`：

```python
class RAGChatState(TypedDict):
    input: dict               # 原始输入 {"session_id": "...", "query": "..."}
    session_id: str
    query: str                 # 用户查询
    session: dict              # 对话历史
    rewritten_query: str       # 重写后的查询
    docs: list[Document]       # 检索到的文档
    messages: list[BaseMessage] # 发给 LLM 的消息
    answer: str                # LLM 的回答
```

**类比**：State 像一张表格，每个节点在表格上填写自己负责的字段，然后传给下一个节点。

### 第 4 步：preprocess — 加载记忆

```python
# agents/flow/rag_chat.py
async def preprocess(state: RAGChatState) -> dict:
    inp = state["input"]
    session = get_session(inp["session_id"])  # 从 Redis 加载对话历史
    return {
        "session": session.model_dump(),       # 把 Session 对象转成字典
        "query": inp["query"],
        "session_id": inp["session_id"],
    }
```

**做了什么**：根据 `session_id` 从 Redis 加载这个用户之前的对话历史。

**为什么要加载历史？** 因为用户可能说 "它的优点是什么"，这个 "它" 指的是上一轮对话提到的东西。没有历史就不知道 "它" 是什么。

**Python 知识点**：
- `async def` = 异步函数，可以在等待 I/O（比如读 Redis）时让出 CPU 给其他任务
- `session.model_dump()` = Pydantic 对象转字典的方法（类似 Java 的 JSON 序列化）

### 第 5 步：rewrite — 查询重写

```python
# agents/flow/rag_chat.py
async def rewrite(state: RAGChatState) -> dict:
    session = state.get("session", {})
    history = session.get("history", [])
    summary = session.get("summary", "")

    if not history and not summary:
        return {"rewritten_query": state["query"]}  # 没有历史，不需要重写

    rewritten = await rewrite_query(
        summary=summary,
        history=history_dicts,
        query=state["query"],
    )
    return {"rewritten_query": rewritten}
```

**做了什么**：用 LLM 把用户查询改写成独立的、不依赖上下文的查询。

**例子**：
```
对话历史: [用户: "Python 列表推导式是什么？", 助手: "列表推导式是..."]
当前查询: "它的优点是什么？"
重写结果: "Python 列表推导式的优点是什么？"
```

**为什么要重写？** 检索系统（Milvus/ES）不懂上下文。如果直接搜 "它的优点是什么"，搜出来全是废话。必须先把 "它" 替换成具体的东西。

**内部调用链**：

```python
# agents/rag/query_rewrite.py
async def rewrite_query(summary, history, query) -> str:
    llm = get_chat_model(settings.chat_model_type)  # 获取 LLM 实例
    # 拼装 Prompt
    messages = [
        ("system", "你是一个查询重写助手..."),
        ("human", f"摘要: {summary}\n历史: {history}\n查询: {query}"),
    ]
    response = await llm.ainvoke(messages)  # 调用 LLM
    return response.content.strip()
```

这里用到了 LLM 的"指令遵循"能力：给 LLM 一个角色设定（System Prompt）+ 具体任务（Human Prompt），让它输出重写后的查询。

### 第 6 步：retrieve — 文档检索（核心步骤）

```python
# agents/flow/rag_chat.py
async def retrieve(state: RAGChatState, config=None) -> dict:
    query = state.get("rewritten_query", state["query"])

    retriever = get_hybrid_retriever()  # 获取混合检索器

    # 在线程池中运行同步检索（避免阻塞事件循环）
    docs = await asyncio.wait_for(
        asyncio.to_thread(retriever.retrieve, query),
        timeout=15,  # 最多等 15 秒
    )
    return {"docs": docs}
```

**做了什么**：从 Milvus 和 Elasticsearch 中搜索与查询相关的文档。

**为什么要 `asyncio.to_thread()`？** 因为检索器内部是同步代码（不是 async 的），如果直接在 async 函数里调用，会阻塞整个事件循环，其他请求都得等着。`asyncio.to_thread()` 把同步代码扔到线程池里执行，不阻塞。

**检索器内部做了什么？** 这是最核心的部分，见下面的详细分析。

#### 检索器详细流程 (`agents/rag/retriever.py`)

```python
class HybridRetriever:
    def retrieve(self, query: str) -> list[Document]:
        # 第 1 步：并行检索
        with ThreadPoolExecutor(max_workers=2) as pool:
            milvus_future = pool.submit(self._retrieve_milvus, query)  # 向量搜索
            es_future = pool.submit(self._retrieve_es, query)          # 关键词搜索

        milvus_docs = milvus_future.result()  # Milvus 返回的结果
        es_docs = es_future.result()          # ES 返回的结果

        # 第 2 步：RRF 融合
        fused = reciprocal_rank_fusion([milvus_docs, es_docs], k=60)

        # 第 3 步：Cross-Encoder 重排序（可选）
        if self._reranker:
            reranked = self._reranker.rerank(query, fused, top_k=k)
            return reranked

        return fused[:k]
```

画成图：
```
查询: "各部门经理是谁"
         │
    ┌────┴────┐
    ▼         ▼
 Milvus     Elasticsearch
(向量搜索)   (BM25关键词)
    │         │
    ▼         ▼
 [doc1,     [doc2,
  doc3,      doc1,
  doc5]      doc4]
    │         │
    └────┬────┘
         ▼
   RRF 融合排序
         │
         ▼
  [doc1, doc3, doc2, doc5, doc4]
         │
         ▼
  Cross-Encoder 重排序 (可选)
         │
         ▼
  [doc1, doc2, doc3]  ← 最终结果
```

**为什么要并行？** Milvus 搜索需要 100ms，ES 搜索需要 80ms。串行要 180ms，并行只要 100ms（取最长的那个）。

**为什么要融合？** 因为两个搜索引擎的分数不可直接比较（Milvus 的相似度分数和 ES 的 BM25 分数量纲不同）。RRF 算法只看排名不看分数，所以能公平融合。

### 第 7 步：construct_messages — 拼装 Prompt

```python
# agents/flow/rag_chat.py
async def construct_messages(state: RAGChatState) -> dict:
    counter = TokenCounter()
    budget = 32768 - 4096  # 模型上下文窗口 - 预留给回答

    parts = []
    # 1. 摘要记忆
    if summary:
        parts.append(f"背景摘要: {summary}")
    # 2. 历史消息
    for msg in history:
        parts.append(f"[{msg['role']}]: {msg['content']}")
    # 3. 检索到的文档
    doc_texts = [doc.page_content for doc in state.get("docs", [])]
    parts.append(f"参考知识:\n{chr(10).join(doc_texts)}")
    # 4. 当前查询
    parts.append(state["query"])

    # Token 预算裁剪 — 如果内容太长，砍掉后面的部分
    fitted = counter.fit_to_budget(parts, budget)

    system = SystemMessage(content="你是一个智能助手...")
    messages = [system, HumanMessage(content="\n\n".join(fitted))]
    return {"messages": messages}
```

**做了什么**：把所有信息（记忆 + 文档 + 查询）拼成一个 Prompt 发给 LLM。

**为什么要做 Token 裁剪？** LLM 有上下文窗口限制（比如 32768 tokens）。如果历史很长、文档很多，超出限制就会报错。`fit_to_budget()` 从头开始保留内容，直到接近上限。

**类比**：你在写一封信，但信纸大小有限。你先写最重要的（当前问题），再写背景（历史），再补充参考资料。如果写不下，就砍掉最不重要的部分。

### 第 8 步：chat — 调用 LLM

```python
# agents/flow/rag_chat.py
async def chat(state: RAGChatState) -> dict:
    model = get_chat_model(settings.chat_model_type)
    response = await model.ainvoke(state["messages"])

    # 保存对话历史
    session["history"].append({"role": "user", "content": state["query"]})
    session["history"].append({"role": "assistant", "content": response.content})

    # 异步保存（不阻塞响应返回）
    asyncio.create_task(_compress_and_save(state["session_id"], session))

    return {"answer": response.content, "messages": [response]}
```

**做了什么**：
1. 调用 LLM 生成回答
2. 把问答对加入历史
3. 异步保存记忆（压缩 + 存 Redis）

**为什么用 `asyncio.create_task()`？** 保存记忆是耗时操作（可能需要调 LLM 做压缩）。如果同步等待，用户要多等几百毫秒。`create_task()` 把它扔到后台执行，主流程立即返回回答给用户。

### 第 9 步：SSE 流式返回

```python
# agents/api/routers/rag.py
async def generate():
    async for event in graph.astream_events(..., version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield {"event": "message", "data": chunk.content}
    yield {"event": "done", "data": "[DONE]"}

return await sse_response(generate(), request)
```

**做了什么**：LLM 每生成一小段文字（一个 chunk），就立即通过 SSE 推送给浏览器。用户看到的是"逐字出现"的效果，而不是等 LLM 全部生成完再一次性显示。

**SSE 是什么？** Server-Sent Events，一种 HTTP 协议，服务器可以持续向浏览器推送数据。和 WebSocket 的区别是 SSE 只能服务器→浏览器单向推送，WebSocket 是双向的。对于"AI 逐字输出"这个场景，SSE 完全够用。

### 完整调用链总结

```
浏览器 fetch("/api/rag/chat/stream")
  │
  ▼
api/routers/rag.py :: rag_chat_stream()
  │
  ▼
flow/rag_chat.py :: build_rag_chat_graph()  ← 构建 LangGraph 图
  │
  ├──→ preprocess()         ← tool/memory/store.py :: get_session()
  │                             从 Redis 加载对话历史
  │
  ├──→ rewrite()            ← rag/query_rewrite.py :: rewrite_query()
  │                             调用 LLM 改写查询
  │
  ├──→ retrieve()           ← rag/retriever.py :: HybridRetriever.retrieve()
  │    │                      ├─ Milvus 向量搜索 (并行)
  │    │                      ├─ ES BM25 搜索 (并行)
  │    │                      ├─ algorithm/rrf.py :: reciprocal_rank_fusion()
  │    │                      └─ rag/reranker.py :: CrossEncoderReranker.rerank()
  │    │
  │    ▼
  ├──→ construct_messages()  ← tool/token_counter.py :: TokenCounter.fit_to_budget()
  │                             拼装 Prompt + Token 裁剪
  │
  ├──→ chat()               ← model/chat_model.py :: get_chat_model()
  │                             调用 LLM 生成回答
  │
  ▼
SSE 流式推送到浏览器
```

---

## 5. 核心算法详解

### 5.1 RRF（Reciprocal Rank Fusion）— 融合多路搜索结果

**问题**：Milvus 和 ES 各返回一组文档，怎么合并成一个排序？

**直觉**：不看分数（因为两个搜索引擎的分数不可比），只看排名。一个文档在多个搜索结果中排名都靠前，它就很可能是好文档。

**公式**：
```
score(doc) = Σ 1 / (k + rank + 1)

其中：
- k = 60（常数，防止排名差异过大）
- rank = 文档在某个搜索结果中的排名（从 0 开始）
```

**例子**：

```
Milvus 结果:  [docA, docB, docC]    (rank: 0, 1, 2)
ES 结果:      [docB, docD, docA]    (rank: 0, 1, 2)

docA 的 RRF 分数 = 1/(60+0+1) + 1/(60+2+1) = 1/61 + 1/63 = 0.0323
docB 的 RRF 分数 = 1/(60+1+1) + 1/(60+0+1) = 1/62 + 1/61 = 0.0326  ← 最高！
docC 的 RRF 分数 = 1/(60+2+1)                  = 1/63           = 0.0159
docD 的 RRF 分数 =                   1/(60+1+1) = 1/62           = 0.0161

最终排序: [docB, docA, docD, docC]
```

docB 在两个搜索结果中都排第一或第二，所以总分最高。

**代码**（`agents/algorithm/rrf.py`）：

```python
def reciprocal_rank_fusion(doc_lists, k=60):
    scores = {}
    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list):
            key = doc.page_content  # 用文档内容作为唯一标识
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
    # 按分数降序排列
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 5.2 Cross-Encoder 重排序 — 精排

**问题**：RRF 融合后的排序是粗排，可能不够精确。需要一个更精细的排序。

**直觉**：RRF 只看"排名"，不看"查询和文档到底有多相关"。Cross-Encoder 会仔细阅读查询和文档的每一对组合，给出精确的相关性分数。

**和 Embedding 的区别**：
- **Embedding（向量模型）**：分别把查询和文档编码成向量，计算向量距离。快，但不够精确（因为压缩了信息）。
- **Cross-Encoder**：把查询和文档拼在一起，输入模型，直接输出相关性分数。慢，但更精确。

```
Embedding 方式（快，粗略）:
  查询 → [向量A] ─┐
                   ├→ 计算距离 → 0.85
  文档 → [向量B] ─┘

Cross-Encoder 方式（慢，精确）:
  [查询 + 文档] → 模型 → 0.97（相关性分数）
```

**代码**（`agents/rag/reranker.py`）：

```python
class CrossEncoderReranker:
    def rerank(self, query, docs, top_k=None):
        # 构造 (查询, 文档) 对
        pairs = [(query, doc.page_content) for doc in docs]
        # 模型一次性给所有对打分
        scores = self._model.predict(pairs)
        # 按分数排序
        scored = sorted(zip(scores, docs), reverse=True)
        return [doc for _, doc in scored[:top_k]]
```

### 5.3 Token 预算管理 — 防止超出上下文窗口

**问题**：LLM 有上下文窗口限制。如果历史消息 + 检索文档 + 当前查询的总 token 数超过限制，API 会报错。

**解决**：`TokenCounter.fit_to_budget()` 从头开始保留内容，直到接近上限。

```python
def fit_to_budget(self, parts: list[str], max_tokens: int) -> list[str]:
    result = []
    used = 0
    for part in parts:
        tokens = self.count(part)
        if used + tokens > max_tokens:
            break  # 超了，后面的都不加了
        result.append(parts)
        used += tokens
    return result
```

**为什么从头开始保留？** 因为 `parts` 的顺序是：摘要 → 历史 → 文档 → 查询。越前面越重要（摘要和历史提供了上下文），越后面的可以丢弃。

---

## 6. Python 编程技巧在这个项目中的体现

### 6.1 async/await — 异步编程

**为什么需要异步？** 一个 AI 请求需要等待很多 I/O 操作：
- 读 Redis（~5ms）
- 调 LLM API（~2000ms）
- 查 Milvus（~100ms）
- 查 ES（~80ms）

如果是同步代码，一个请求在等 LLM 回复时，整个服务器就卡住了，其他请求都得排队。

异步代码在等待 I/O 时会"让出"CPU，去处理其他请求：

```python
# 同步（阻塞）— 糟糕！
def handle_request():
    result = llm.call(query)     # 等 2 秒，服务器卡住
    return result

# 异步（非阻塞）— 好！
async def handle_request():
    result = await llm.acall(query)  # 等 2 秒，但服务器可以处理其他请求
    return result
```

**类比**：
- 同步 = 你去餐厅点餐后站在柜台等，后面的客人都被你挡住
- 异步 = 你点餐后回座位坐着，服务员叫号你再去取，其他客人可以继续点餐

**什么时候用 `await`？** 当你要调用一个"可能需要等待"的操作时：
- 网络请求（调 LLM API、查数据库）
- 文件读写
- 任何标了 `async def` 的函数

### 6.2 类型注解（Type Hints）

Python 是动态类型语言（变量不需要声明类型），但这个项目大量使用类型注解：

```python
def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
    #              参数类型标注            返回类型标注
```

**为什么用类型注解？**
1. **IDE 自动补全**：写了 `query:` 后面加 `str`，IDE 就知道你可以用 `.split()`、`.strip()` 等字符串方法
2. **提前发现错误**：如果你传了一个 `int` 给 `query`，IDE 会标红警告
3. **代码即文档**：不用猜参数是什么类型

### 6.3 TypedDict — 结构化字典

```python
class RAGChatState(TypedDict):
    query: str
    docs: list[Document]
    answer: str
```

**是什么**：告诉 Python "这个字典必须有这些 key，且 value 是这些类型"。

**为什么不直接用普通字典？** 因为普通字典的 key 是随便写的，容易拼错（`state["querry"]` 不会报错，但 `state["query"]` 才是对的）。TypedDict 让 IDE 能检查拼写。

### 6.4 设计模式：工厂模式

```python
# agents/model/chat_model.py
_chat_model_registry = {}

def register_chat_model(name, factory):
    _chat_model_registry[name] = factory

def get_chat_model(name=None):
    factory = _chat_model_registry[name]
    return factory()  # 调用工厂函数创建实例

# 使用时
model = get_chat_model("ark")  # 返回豆包模型
model = get_chat_model("openai")  # 返回 OpenAI 模型
```

**为什么用工厂模式？** 因为业务代码不应该关心"怎么创建模型"这个细节。如果不用工厂，代码里到处都是：

```python
# 坏的做法 — 如果要换模型，得改每个文件
model = ChatOpenAI(model="doubao", api_key="xxx", base_url="xxx")
```

用了工厂后，业务代码只需要 `get_chat_model("ark")`，换模型只改配置，不改代码。

### 6.5 装饰器（Decorator）

```python
@router.post("/ask")         # 装饰器 1：注册 HTTP 路由
@traceable(name="rerank")    # 装饰器 2：添加链路追踪
def rerank(self, query, docs):
    ...
```

**是什么**：装饰器是一个"包装函数"，它在不修改原函数代码的情况下，给函数添加额外功能。

**`@router.post("/ask")` 做了什么？** 告诉 FastAPI："当收到 POST /ask 请求时，调用下面这个函数。"

**`@traceable` 做了什么？** 在函数执行前后自动记录日志（耗时、参数、返回值），方便排查问题。

### 6.6 依赖注入（Dependency Injection）

```python
# rag_chat.py
async def retrieve(state: RAGChatState, config=None) -> dict:
    retriever = get_hybrid_retriever()  # 不直接 new，而是从工厂获取
```

**是什么**：不自己创建依赖的对象，而是从外部获取。这样在测试时可以替换为 mock 对象：

```python
# 测试代码
def test_retrieve():
    mock_retriever = Mock()
    mock_retriever.retrieve.return_value = [fake_doc]
    # 用 mock 替换真实检索器
```

---

## 7. AI Agent 核心概念

### 7.1 RAG（Retrieval-Augmented Generation）

**问题**：LLM 的知识有截止日期，且不了解你公司的内部数据。

**解决**：先检索相关文档，再把文档内容塞进 Prompt，让 LLM 基于文档回答。

```
传统 LLM:
  用户: "我们公司的报销制度是什么？"
  LLM: "我不了解贵公司的具体制度..."（它不知道）

RAG:
  用户: "我们公司的报销制度是什么？"
  检索: 找到《财务报销管理办法》第3章
  Prompt: "参考以下文档回答：[文档内容]... 问题：报销制度是什么？"
  LLM: "根据《财务报销管理办法》，差旅报销标准为..."（它现在知道了！）
```

### 7.2 Query Rewrite（查询重写）

**问题**：用户的查询可能包含指代（"它"、"这个"），直接搜索搜不准。

**解决**：用 LLM 把查询改写成独立的、不依赖上下文的版本。

```
原始: "它的优点是什么？"
重写: "Python 列表推导式的优点是什么？"
```

### 7.3 Hybrid Search（混合检索）

**问题**：单一搜索方式有局限。

**解决**：向量搜索（懂语义）+ 关键词搜索（懂精确匹配）= 取长补短。

### 7.4 Intent Classification（意图分类）

**问题**：用户的问题类型多样（查数据库、查知识库、闲聊），需要不同的处理流程。

**解决**：先用 LLM 判断用户意图，再路由到对应的处理图。

```
"各部门经理是谁"     → 意图: sql_query     → 走 SQL 生成流程
"报销制度是什么"     → 意图: knowledge     → 走 RAG 检索流程
"你好"              → 意图: chat          → 走闲聊流程
```

### 7.5 Human-in-the-Loop（人工介入）

**问题**：AI 生成的 SQL 可能有错误，直接执行有风险。

**解决**：SQL 生成后先暂停，等人工确认后再执行。LangGraph 的 `interrupt` 机制支持这种"暂停-恢复"模式。

```
生成 SQL: DELETE FROM users WHERE id = 1
  ↓
interrupt("请确认是否执行此 SQL？")
  ↓ (等待用户点击)
用户: 确认 → 执行
用户: 拒绝 → 重新生成
```

### 7.6 Memory（记忆系统）

**问题**：LLM 是无状态的，每次调用都是独立的。但用户期望 AI 记得之前的对话。

**解决**：把对话历史存在外部存储（Redis），每次请求时加载。

三级记忆架构：
```
L1 工作记忆: 最近 5 轮对话原文（精确但占空间）
L2 摘要记忆: 更早对话的 LLM 压缩摘要（节省空间）
L3 知识记忆: 提取的实体和事实（结构化）
```

---

## 8. 如何自己动手改这个项目

### 8.1 环境搭建

```bash
# 克隆项目
git clone <repo-url>
cd financial-copilot-platform

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -e .

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 API Key 等

# 启动
python -m agents.main
```

### 8.2 新手练手任务

**任务 1：加一个新的 LLM Provider**

1. 在 `agents/model/providers/` 下新建文件（比如 `claude.py`）
2. 实现 `init()` 函数，注册 Chat Model
3. 在 `agents/model/chat_model.py` 的 `init_chat_models()` 中导入
4. 在 `settings.py` 中添加配置项
5. 在 `.env` 中配置 API Key

**任务 2：修改 System Prompt**

找到 `agents/flow/rag_chat.py` 的 `construct_messages()` 函数，修改 `SystemMessage` 的内容。试试让 LLM 用英文回答，或者用更简短的方式回答。

**任务 3：添加新的意图类型**

1. 在 `agents/flow/final_graph.py` 的意图分类 prompt 中添加新意图
2. 在 `route_intent()` 函数中添加路由逻辑
3. 实现新的子图处理函数
4. 测试：发送一条触发新意图的查询

**任务 4：运行评估并改进检索**

```bash
# 生成评测数据集
python -m agents.eval.cli generate --num-per-table 3

# 跑评估
python -m agents.eval.cli run --detail

# 看结果，尝试调参（chunk_size、top_k、reranker 阈值等）
```

### 8.3 调试技巧

**看日志**：
```bash
# 设置日志级别
export LOG_LEVEL=DEBUG
python -m agents.main
```

**用 LangSmith 追踪**：项目集成了 LangSmith，每次 LLM 调用都会自动记录。在 LangSmith 控制台可以看到完整的调用链。

**单步调试**：在 IDE 中打断点，直接运行 `tests/` 下的测试文件。

---

## 附录：关键文件速查表

| 你想了解什么 | 看哪个文件 |
|-------------|-----------|
| 服务器怎么启动的 | `agents/api/app.py` |
| 配置项有哪些 | `agents/config/settings.py` |
| RAG 对话的完整流程 | `agents/flow/rag_chat.py` |
| SQL 生成的完整流程 | `agents/flow/sql_react.py` |
| 意图分类和路由 | `agents/flow/final_graph.py` |
| 混合检索怎么实现的 | `agents/rag/retriever.py` |
| RRF 融合算法 | `agents/algorithm/rrf.py` |
| Reranker 怎么用的 | `agents/rag/reranker.py` |
| 对话记忆怎么存的 | `agents/tool/memory/store.py` |
| LLM 怎么调用的 | `agents/model/chat_model.py` |
| 前端怎么展示的 | `agents/static/index.html` |
| 测试怎么写的 | `tests/test_rag_flow.py` |

---

## 9. Agent 设计常见问题与解决方案

### 9.1 Checkpoint 设计：怎么让对话"暂停-恢复"？

**场景**：AI 生成了一条 SQL，但执行前需要人工确认。用户点了"确认"后，系统要从上次暂停的地方继续执行。

**LangGraph 的解决方案**：`interrupt()` + `Command(resume=...)`

```python
# agents/flow/sql_react.py — approve 节点
async def approve(state: SQLReactState) -> dict:
    # 暂停执行，把控制权交还给用户
    user_decision = interrupt({
        "sql": state["sql"],
        "message": "请确认是否执行此 SQL",
    })
    # 用户回复后，从这里继续
    if user_decision["approved"]:
        return {"approved": True}
    else:
        return {"approved": False, "refine_feedback": user_decision["feedback"]}
```

```
时间线:
  T1: 用户发请求 → 执行到 approve → interrupt() → 暂停
  T2: 用户看到 SQL，思考中...
  T3: 用户点"确认" → Command(resume={approved: true}) → 从暂停处继续
  T4: 执行 SQL → 返回结果
```

**Checkpoint 怎么存状态？**

暂停时，LangGraph 把当前 State 序列化存起来。恢复时，反序列化 State，继续执行。

```python
# agents/tool/storage/checkpoint.py
from langgraph.checkpoint.memory import MemorySaver

def get_checkpointer():
    # 内存版 — 进程重启后丢失
    return MemorySaver()

def get_redis_checkpointer():
    # Redis 版 — 持久化，进程重启后仍可恢复
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver
    return AsyncRedisSaver(redis_client=redis_client)
```

**两种 Checkpointer 对比**：

| | MemorySaver | AsyncRedisSaver |
|---|---|---|
| 存储位置 | Python 进程内存 | Redis (RedisJSON) |
| 进程重启后 | 状态丢失 | 状态保留 |
| 适用场景 | 开发/测试 | 生产环境 |
| 依赖 | 无 | redis-stack-server |

### 9.2 用户数据隔离：怎么保证用户 A 看不到用户 B 的数据？

**问题**：多个用户同时使用系统，每个人有自己的对话历史和中断状态。怎么隔离？

**解决方案**：用 `session_id` 作为命名空间。

```python
# agents/api/routers/final.py
def _make_config(session_id: str) -> dict:
    return {
        "configurable": {"thread_id": session_id},  # session_id → thread_id
    }

# 用户 A 的请求
config_a = _make_config("user_alice")
result = await graph.ainvoke({...}, config=config_a)  # 状态存在 "user_alice" 命名空间下

# 用户 B 的请求
config_b = _make_config("user_bob")
result = await graph.ainvoke({...}, config=config_b)  # 状态存在 "user_bob" 命名空间下
```

**两层隔离**：

```
┌─────────────────────────────────────────┐
│  Layer 1: LangGraph Checkpoint          │
│  thread_id = session_id                 │
│  每个用户的 interrupt/resume 状态独立     │
│  key: (thread_id, checkpoint_ns, ...)   │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  Layer 2: Session Memory (Redis)        │
│  key = "session:memory:{session_id}"    │
│  每个用户的对话历史、摘要独立              │
│  TTL = 24 小时自动过期                    │
└─────────────────────────────────────────┘
```

**类比**：Checkpoint 像银行的保险柜，每个客户有自己的柜子（thread_id）。Session Memory 像银行的客户档案，每个客户有自己的档案袋（session_id）。

### 9.3 状态序列化：怎么保证数据不丢？

**问题**：Python 对象（如 `Document`、`BaseMessage`）不能直接存到 Redis。需要序列化成 JSON。

**项目中的两种序列化方式**：

**方式 1：LangGraph 内部处理**（Checkpoint 层）

LangGraph 的 State 定义为 `TypedDict`，LangGraph 内部自动处理序列化。你不需要手动序列化：

```python
class SQLReactState(TypedDict):
    query: str
    docs: list[Document]     # LangGraph 自动处理 Document 的序列化
    messages: Annotated[list[BaseMessage], add_messages]  # 自动处理
```

**方式 2：Pydantic 手动序列化**（Session Memory 层）

Session 记忆用 Pydantic 模型，需要手动调用序列化方法：

```python
# agents/tool/memory/store.py

# 存：Python 对象 → JSON 字符串 → Redis
data = session.model_dump_json()           # Session → JSON
client.set(f"session:memory:{session_id}", data, ex=86400)

# 取：Redis → JSON 字符串 → Python 对象
data = client.get(f"session:memory:{session_id}")
session = Session.model_validate_json(data)  # JSON → Session
```

**防丢策略**：

1. **TTL 过期**：Session 数据设置 24 小时 TTL，过期自动清理，防止 Redis 内存溢出
2. **内存降级**：Redis 不可用时自动切换到内存存储，保证服务不中断
3. **异常恢复**：压缩历史时如果 LLM 调用失败，回滚到压缩前的状态

```python
# agents/tool/memory/compressor.py
async def compress_session(session, llm):
    to_compress = session.history[:-6]
    session.history = session.history[-6:]  # 先裁剪

    try:
        response = await llm.ainvoke([...])
        session.summary = response.content
    except Exception:
        session.history = to_compress + session.history  # 失败则回滚
        raise
```

### 9.4 记忆压缩：怎么防止历史越来越长？

**问题**：用户聊了 100 轮，历史消息占了大量 Token，导致：
1. Prompt 太长，超出 LLM 上下文窗口
2. Token 费用飙升
3. 检索噪音增大

**解决方案**：LLM 摘要压缩

```
压缩前:
  history = [msg1, msg2, ..., msg20, msg21, msg22, msg23, msg24, msg25]
  summary = ""

压缩后:
  history = [msg20, msg21, msg22, msg23, msg24, msg25]  (只保留最近 6 轮)
  summary = "用户询问了 Python 列表推导式的用法，讨论了性能优化..."  (LLM 生成的摘要)
```

```python
# agents/tool/memory/compressor.py
DEFAULT_MAX_HISTORY_LEN = 20  # 超过 20 条消息触发压缩
KEEP_RECENT = 6               # 压缩后保留最近 6 条

async def compress_session(session, llm):
    if len(session.history) <= DEFAULT_MAX_HISTORY_LEN:
        return  # 不够长，不压缩

    to_compress = session.history[:-KEEP_RECENT]  # 要压缩的旧消息
    session.history = session.history[-KEEP_RECENT:]  # 保留的新消息

    # 让 LLM 把旧消息 + 已有摘要 合并成新摘要
    prompt = f"旧摘要: {session.summary}\n旧消息: {to_compress}\n请合并为新摘要"
    session.summary = await llm.ainvoke(prompt)
```

### 9.5 Redis 降级：Redis 挂了怎么办？

**问题**：Redis 是外部依赖，可能宕机。如果代码直接依赖 Redis，Redis 挂了整个服务就挂了。

**解决方案**：每一层 Redis 使用都有内存降级策略。

```python
# agents/tool/memory/store.py — SessionStore
class SessionStore:
    def __init__(self):
        self._fallback = {}        # 内存降级存储
        self._use_fallback = True  # 默认先用降级模式

    def get(self, session_id):
        if self._use_fallback:
            return self._fallback.get(session_id, Session(id=session_id))
        try:
            data = self._redis.get(key)
            return Session.model_validate_json(data)
        except Exception:
            self._use_fallback = True  # Redis 出错，切换到降级模式
            return self.get(session_id)  # 重试（这次走降级分支）
```

**三层降级**：

| 组件 | 正常模式 | 降级模式 | 影响 |
|------|---------|---------|------|
| Session Store | Redis | 内存 dict | 进程重启后历史丢失 |
| Retrieval Cache | Redis | 内存 dict | 缓存不跨进程 |
| LangGraph Checkpoint | Redis | MemorySaver | interrupt/resume 不跨重启 |

---

## 10. 我们踩过的坑：真实 Bug 与修复

### 10.1 SSE 流式输出不显示（最高频 Bug）

**现象**：Chat 对话框显示"思考中..."，但 LLM 的回答始终不出现。后端日志显示数据已发送，浏览器 Network 面板也能看到数据流，但前端就是不显示。

**排查过程**：

1. 先怀疑是前端 `fetch` 的问题，改用 `ReadableStream` 手动读取 — 没解决
2. 添加 `console.log` 调试，发现 `parts` 数组长度始终为 1 — 说明 `split('\n\n')` 没切开
3. 在浏览器控制台手动测试：

```javascript
var b = "event: message\r\ndata: hello\r\n\r\n";
b.split('\n\n');  // 返回 ["event: message\r\ndata: hello\r\n\r\n"]  ← 没切开！
b.split('\r\n\r\n');  // 返回 ["event: message\r\ndata: hello", ""]  ← 切开了！
```

**根因**：SSE 协议规定行分隔符是 `\r\n`（CRLF），块分隔符是 `\r\n\r\n`。但 JavaScript 的 `split('\n\n')` 找不到 `\n\n`，因为两个 `\n` 之间隔了一个 `\r`！

```
\r\n\r\n 的字节:  \r  \n  \r  \n
                    ↑     ↑
                    这两个 \n 之间有 \r，不连续！
```

**修复**：

```javascript
// 之前（错误）
let parts = buffer.split('\n\n');    // 永远切不开

// 之后（正确）
let parts = buffer.split('\r\n\r\n');  // 正确切分 SSE 块
```

**教训**：协议规范很重要！SSE 规范（RFC 8895）明确要求 CRLF 行结尾。浏览器的 `EventSource` API 会自动处理，但手动解析时必须遵守规范。

### 10.2 LLM 把检索到的文档原封不动输出

**现象**：用户问 "各部门经理是谁"，LLM 回答的不是 "张三是研发部经理..."，而是把数据库表结构原样输出：

```
表名: t_user
字段:
  id int PRIMARY KEY NOT NULL
  username varchar(50) UNIQUE
  real_name varchar(50) COMMENT '真实姓名'
  department_id int COMMENT '部门ID'
```

**根因**：Prompt 中没有告诉 LLM "不要复述原文"。LLM 看到参考文档后，以为用户想看文档内容，就直接输出了。

**修复**：添加 SystemMessage 指导 LLM 行为：

```python
# agents/flow/rag_chat.py
system = SystemMessage(content=(
    "你是一个智能助手。根据参考知识回答用户问题。"
    "只使用与问题相关的信息，忽略无关内容。"
    "直接回答问题，不要复述参考知识原文。"         # ← 关键指令
    "如果参考知识中没有相关信息，根据你的知识回答。"
))
```

**教训**：LLM 会"照字面意思理解"。你不告诉它"不要复述"，它就可能复述。Prompt Engineering 的核心是"把你的期望说清楚"。

### 10.3 SQL Agent Tab 流式输出无显示

**现象**：Chat Tab 流式输出正常，但 SQL Agent Tab 完全没有输出。

**排查过程**：

1. SQL Agent 使用 `astream_events(version="v2")` 捕获子图的 LLM 输出
2. 添加日志发现：955 个事件中只有 2 个 LLM 事件（来自 `classify_intent`），SQL 子图的事件为 0

**根因**：`astream_events` 只能捕获**直接子节点**的 LLM 事件。SQL React 子图内部用 `ainvoke` 调用，不产生流式事件。

```
final_graph (astream_events 能捕获)
  ├── classify_intent  → 有 LLM 事件 ✅
  └── sql_react (ainvoke)  → 内部的 LLM 事件丢失 ❌
       └── sql_generate (LLM)  → 外层看不到
```

**修复方案**：采用"先分类再路由"模式：

```javascript
// 1. 先调用非流式分类接口
const classify = await fetch('/api/final/classify', {body: query});
const {intent} = await classify.json();

// 2. 根据意图选择不同端点
if (intent === 'sql_query') {
    // SQL 路径：用非流式（因为无法流式捕获子图事件）
    const resp = await fetch('/api/final/invoke', {body: query});
    // 处理审批流程...
} else {
    // Chat 路径：用流式
    const resp = await fetch('/api/rag/chat/stream', {body: query});
    // 流式显示...
}
```

**教训**：LangGraph 的 `astream_events` 有作用域限制。嵌套子图如果用 `ainvoke` 调用，外层捕获不到其内部事件。设计流式架构时要考虑这个约束。

### 10.4 `request.is_disconnected()` 导致 SSE 中断

**现象**：SSE 测试端点（简单发送 5 条消息）在浏览器中正常，但真实 RAG 端点有时会中途断开。

**根因**：SSE 响应函数中加了 `request.is_disconnected()` 检查：

```python
async def sse_response(generator, request):
    async for event in generator:
        if await request.is_disconnected():  # ← 这行导致问题
            break
        yield event
```

`is_disconnected()` 在使用 `fetch` + `ReadableStream` 时行为不稳定，可能误判为"已断开"。

**修复**：删除手动断连检查，让 `EventSourceResponse` 自己处理：

```python
async def sse_response(generator, request):
    return EventSourceResponse(generator)  # 内部已有 _listen_for_disconnect
```

**教训**：框架已经处理了的事情，不要自己重复处理。`sse-starlette` 的 `EventSourceResponse` 内部已经有断连检测机制。

### 10.5 astream_events 过滤了意图分类输出

**现象**：SQL Agent 流式输出时，先显示了意图分类的结果（"sql_query"），然后才显示真正的 SQL 内容。

**根因**：`classify_intent` 节点的 LLM 输出也通过 `astream_events` 流出来了，前端没有过滤。

**修复**：在流式生成器中过滤掉非目标节点的输出：

```python
async for event in graph.astream_events(..., version="v2"):
    if event["event"] == "on_chat_model_stream":
        # 只处理目标节点的输出
        if event.get("name") == "chat":  # 只要 chat 节点的输出
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield {"event": "message", "data": chunk.content}
```

**教训**：`astream_events` 会捕获图中**所有**节点的事件。必须通过 `event["name"]` 过滤，只处理你关心的节点。

### 10.6 最后一个 SSE 块丢失

**现象**：LLM 回答的最后一两个字总是不显示。

**根因**：SSE 解析逻辑中，`buffer.split('\r\n\r\n')` 的最后一个元素留在 buffer 里，但循环结束后没有处理它：

```javascript
while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});
    let parts = buffer.split('\r\n\r\n');
    buffer = parts.pop();  // 最后一个不完整的块留在 buffer
    for (const part of parts) { ... }
}
// 循环结束后，buffer 里还有最后一个完整的块没处理！
```

**修复**：循环结束后处理剩余 buffer：

```javascript
while (true) { ... }

// 处理最后的 buffer
if (buffer.trim()) {
    const ev = parseSSEBlock(buffer);
    if (ev.event !== 'done' && ev.data) fullContent += ev.data;
}
bubble.textContent = fullContent || '(无响应)';
```

**教训**：流式处理中，"最后一个"数据块是常见的遗漏点。写循环时要考虑：循环结束后还有没有剩余数据？

### 10.7 Embedding API 批量大小限制

**现象**：Schema 索引时，15 张表的文档一次性发给 Embedding API，报错 `batch size is invalid, it should not be larger than 10`。

**根因**：Embedding API（豆包）限制单次最多处理 10 条文本，但代码一次发了 15 条。

**修复**：分批调用：

```python
# 之前（错误）
vectors = embeddings.embed_documents(texts)  # 15 条一次性发

# 之后（正确）
vectors = []
batch_size = 10
for i in range(0, len(texts), batch_size):
    vectors.extend(embeddings.embed_documents(texts[i:i + batch_size]))
```

**教训**：第三方 API 往往有批量限制。调用前要查文档，或者默认做分批处理。

### 10.8 ES 检索返回空 metadata

**现象**：ES BM25 检索返回的文档 `metadata` 为空字典 `{}`，导致评估时 `doc_id` 匹配失败。

**根因**：Schema 文档存入 ES 时直接用 `es.index()` API，字段是扁平的：

```python
# 存入时
es.index(index=..., id=doc_id, document={
    "text": "...",
    "source": "mysql_schema",
    "table_name": "t_user",
    "doc_id": "schema_t_user",
})
```

但 langchain 的 `ElasticsearchStore` 读取时，期望 metadata 在一个嵌套的 `metadata` 字段下。直接存的扁平字段不会被映射到 `Document.metadata`。

**教训**：存储和读取要用同一套协议。如果用 langchain 的 `ElasticsearchStore` 读，就应该用它的 `add_documents()` 方法存，而不是直接调 ES client。

**修复**：将 ES 存入格式改为嵌套 `metadata` 结构，同步修改 `check_docs` fallback 的字段访问路径：

```python
# schema_indexer.py — 修复后
es.index(index=..., id=doc_id, document={
    "text": doc.page_content,
    "metadata": {                          # ← 嵌套 metadata
        "source": _SCHEMA_SOURCE,
        "table_name": doc.metadata.get("table_name", ""),
        "doc_id": doc_id,
    },
})

# sql_react.py check_docs — 修复后
resp = es.search(
    index=settings.es.index,
    query={"term": {"metadata.source": _SCHEMA_SOURCE}},  # ← metadata.source
    size=50,
)
docs = [
    Document(
        page_content=hit["_source"]["text"],
        metadata=hit["_source"].get("metadata", {}),       # ← 读嵌套字段
    )
    for hit in resp["hits"]["hits"]
]
```

---

### 10.9 SystemMessage 导致测试索引偏移

**现象**：6 个 RAG 测试突然全部失败，报 `assert 2 == 1` 或内容在 `messages[0]` 找不到。

**根因**：`construct_messages()` 后来加了 `SystemMessage`（系统提示词），返回从 `[HumanMessage]` 变成了 `[SystemMessage, HumanMessage]`。但测试还在用 `messages[0]` 取内容：

```python
# 测试代码（未更新）
result = await construct_messages(state)
assert len(result["messages"]) == 1                          # ← 实际是 2
assert "hello" in result["messages"][0].content              # ← 现在 [0] 是 SystemMessage
```

**教训**：给函数加返回值字段（如新增一个 message）时，必须同步检查所有调用方和测试。这类"接口变更但测试没跟上"的问题在多人协作中非常常见。

**修复**：更新所有测试的索引和长度断言：

```python
# 修复后
result = await construct_messages(state)
assert len(result["messages"]) == 2                          # SystemMessage + HumanMessage
assert "hello" in result["messages"][1].content              # [1] 才是 HumanMessage
```

影响文件：`tests/test_rag_flow.py`（3 处）、`tests/test_rag_e2e.py`（3 处），共 6 个测试用例。

---

## 附录 A：Bug 排查方法论

这个项目中我们用了"二分法排查"来定位 SSE 问题：

```
第 1 步：最简端点测试
  → 创建 GET /api/rag/test/stream，发送 5 条固定消息
  → 结果：正常显示
  → 结论：SSE 基础设施没问题

第 2 步：对比真实端点
  → test 端点用 GET，真实端点用 POST
  → 改 test 为 POST — 仍然正常
  → 结论：POST 不是问题

第 3 步：对比生成方式
  → test 端点直接 yield，真实端点用 astream_events
  → 创建 test/stream2 用 astream_events
  → 结果：test/stream2 也不显示
  → 结论：astream_events 是问题所在

第 4 步：检查前端解析
  → 添加 console.log 打印 buffer
  → 发现 buffer 有内容但 parts 为空
  → 结论：split('\n\n') 没有切开 buffer

第 5 步：验证根因
  → 浏览器控制台测试: "a\r\n\r\nb".split('\n\n')
  → 返回 ["a\r\n\r\nb"]，确认没切开
  → 改为 split('\r\n\r\n') — 解决！
```

**通用排查思路**：
1. **最小复现**：用最简单的代码复现问题
2. **逐步排除**：每次只改一个变量，确认是否影响结果
3. **查看原始数据**：不要假设数据格式，打印出来看（`console.log`、`print`）
4. **查阅规范**：不要凭记忆猜测协议格式，看 RFC 文档
