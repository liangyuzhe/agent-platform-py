# Go-Agent Memory System - Technical Summary

## 1. Project Overview

Go-Agent is a Go-based AI agent framework built on **CloudWeGo Eino** (`github.com/cloudwego/eino v0.7.21`). The architecture uses a directed graph execution model where flows are compiled into `compose.Graph` structures with typed nodes and edges.

### Directory Structure

| Directory | Purpose |
|-----------|---------|
| `api/` | HTTP API layer (Gin framework), REST endpoints |
| `flow/` | Compiled graph flows: RAG chat, SQL ReAct, analyst, final orchestrator |
| `model/` | Chat model and embedding model factories (Ark, OpenAI, Qwen, DeepSeek, Gemini) |
| `tool/` | Utility packages: memory, storage, document processing, SQL tools, tracing, SFT |
| `rag/` | RAG pipeline: indexing graph, retriever graph, vector/ES retrievers |
| `config/` | Configuration loading from `.env` via godotenv |
| `algorithm/` | BM25 and RRF ranking algorithms |

### Startup Sequence (`main.go:23-125`)

Config -> Redis -> Milvus -> Indexers -> Retrievers -> Parsers -> Loaders -> Splitters -> Tracing -> MCP Tools -> Compile Graph Flows (IndexingGraph, RAGChatFlow, FinalGraph)

---

## 2. Memory Package (`tool/memory`)

The memory system implements a **two-tier conversation memory** with four files.

### 2.1 Session Data Model (`session.go`)

```go
const SummaryMessageExtraKey = "_is_summary_message"

type Session struct {
    ID        string
    History   []*schema.Message  // Raw message list (user + assistant turns)
    Summary   string             // LLM-generated compressed summary
    UpdatedAt time.Time          // Dead field - never set
}
```

- `History` stores raw messages using Eino `schema.Message` type
- `Summary` stores LLM-generated compressed summary of older turns
- `SummaryMessageExtraKey` is defined but never referenced (placeholder)
- `UpdatedAt` is declared but never set (dead field)

### 2.2 Session Store (`store.go`)

```go
type Store interface {
    Get(ctx context.Context, sessionID string) (*Session, error)
    Save(ctx context.Context, sessionID string, session *Session) error
}
```

**InMemoryStore implementation:**
- Uses `sync.RWMutex` for concurrent access
- `Get`: Returns new empty `Session` if ID not found (no error)
- `Save`: Overwrites session in map
- No TTL, no eviction, no disk persistence - purely volatile
- Instantiated once in `main.go:105`: `memStore := memory.NewMemoryStore()`

### 2.3 CheckPoint Store (`memory.go`)

Separate from Session Store - implements Eino's `compose.CheckPointStore` interface with `Get(ctx, key) ([]byte, bool, error)` and `Set(ctx, key, val) error`. **Never called** in the codebase. FinalGraph uses `storage.NewRedisCheckPointStore()` instead. Appears to be an unused fallback/prototype.

### 2.4 Compressor (`compressor.go`)

Core of the memory compression system.

```go
type Summarizer struct {
    Model         model.BaseChatModel
    MaxHistoryLen int
}
```

**Summary Prompt (Chinese):**
Instructs the LLM to:
1. Merge `<previous_summary>` with `<older_messages>` into a new summary
2. Preserve core facts, user preferences, unresolved issues
3. Discard pleasantries and redundant intermediate steps
4. Maintain coherence

**Compress Algorithm:**
1. **Gate check**: If `len(sess.History) <= MaxHistoryLen`, return (no compression)
2. **Split**: Take all messages except last 3 as `toCompress`, keep last 3 in `History`
3. **Format**: Convert messages to `[role]: content\n`
4. **LLM call**: Call `s.Model.Generate()` with summary prompt
5. **Update**: Replace `sess.Summary` with LLM response

**Critical observations:**
- Compression is **lossy** - original messages discarded permanently
- Uses non-streaming `Generate` (blocking operation)
- No error recovery - if LLM fails, history already truncated but summary not updated

---

## 3. Memory in RAG Chat Flow (`flow/rag_chat.go`)

### 3.1 Graph Structure

```
START -> PreProcess -> Rewrite -> QueryToMsgs -> Retrieve -> ConstructMessages -> Chat -> END
```

**State type:**
```go
type GraphState struct {
    Input   RAGChatInput
    Session *memory.Session
    Query   string
    Docs    []*schema.Document
}
```

### 3.2 Memory Integration Points

**PreProcess (lines 80-88):**
- Loads session from store: `store.Get(ctx, in.SessionID)`
- Stores in `state.Session`

**Rewrite (lines 90-110):**
- Uses `RewritePrompt` with `state.Session.Summary` and `state.Session.History`
- Calls `rag_tools.Rewrite()` to transform query into standalone search query
- First message (no history/summary) passes through unchanged

**ConstructMessages (lines 120-139):**
Assembles final prompt:
1. If `Summary != ""`, prepend system message: `"背景摘要: " + Summary`
2. Append all `History` messages
3. Build user message with RAG documents + query

**Chat post-handler (lines 149-161):**
After LLM response:
1. Append user query and assistant response to `History`
2. **Asynchronously** (goroutine) call `sm.Compress(bgCtx, s)` then `store.Save(bgCtx, s.ID, s)`
3. Uses `context.Background()` - not cancelled if request ends

### 3.3 Initialization (`main.go:105-113`)

```go
memStore := memory.NewMemoryStore()
taskModel, err := chat_model.GetChatModel(ctx, config.Cfg.ChatModelType)
err = flow.InitRAGChatFlow(ctx, memStore, taskModel)
```

Summarizer uses same chat model as main task model (`CHAT_MODEL_TYPE` env var).

---

## 4. Memory Layering / Hierarchy

| Tier | Storage | Content | Lifetime |
|------|---------|---------|----------|
| **Short-term** (Working Memory) | `Session.History` | Last 3 message pairs | Until next compression trigger |
| **Long-term** (Summary) | `Session.Summary` | LLM-generated summary | Persisted across compressions, accumulates |

**How tiers interact:**
- New turns added to `History` (short-term)
- When `History` exceeds 3 messages, `Compress` splits: older -> summarized into `Summary`, last 3 remain in `History`
- `Summary` evolves via LLM merge (not simple append) - can lose detail over time

**What is NOT present:**
- No long-term fact extraction or entity memory
- No vector-based semantic memory
- No explicit token counting or budget management
- No structured key-value facts between raw messages and LLM summary

---

## 5. Token Management and Context Window Optimization

**No explicit token counting** - no tokenizer library, no token budget constants.

**Implicit context window management:**
1. **History truncation**: `MaxHistoryLen = 3` limits raw history to 3 turns
2. **Summary compression**: Replaces N old messages with single summary string (no length limit on summary)
3. **RAG document injection**: Concatenates retrieved documents verbatim (only bound: `TopK` config, default 10)

**Potential issues:**
- No guard against summary + history + documents exceeding model's context window
- Summary can grow unbounded over long conversations
- Retrieved documents concatenated without length limit

---

## 6. Memory Interaction with LLM Calls

| Call | Location | Model | Purpose | Critical Path? |
|------|----------|-------|---------|----------------|
| Query Rewriting | `rag_chat.go:90-110` | "rewrite" model | Transform query into standalone search query | Yes |
| Response Generation | `rag_chat.go:146-161` | "ark" model | Generate assistant response with memory context | Yes |
| Memory Compression | `compressor.go:45-52` | "task" model (same as main) | Compress old messages into summary | No (async goroutine) |

---

## 7. Configuration

**No dedicated memory configuration** - all values hardcoded:

| Parameter | Value | Location |
|-----------|-------|----------|
| `MaxHistoryLen` | `3` | `flow/rag_chat.go:67` |
| Summary prompt | Chinese template | `tool/memory/compressor.go:16-25` |
| Recent messages retained | `3` | `tool/memory/compressor.go:35` |
| Store type | In-memory only | `main.go:105` |
| Summary model | Same as `CHAT_MODEL_TYPE` | `main.go:106`, `flow/rag_chat.go:67` |
| Session ID default | `"default_user"` | `api/rag_chat_stream.go:25`, `api/rag_ask.go:17` |

---

## 8. Related Storage Systems (Not Conversation Memory)

**RedisCheckPointStore** (`tool/storage/checkpoint_store.go`):
- Eino `compose.CheckPointStore` using Redis with in-memory fallback
- Used by FinalGraph for graph checkpoint persistence
- TTL: 24h, Key prefix: `checkpoint:`

**SessionStore** (`tool/storage/session.go`):
- Stores `SessionContext` (interrupt ID, checkpoint ID, original query, waiting refine flag)
- Used by FinalGraph's human-in-the-loop approval flow
- Redis with in-memory fallback, TTL: 24h, Key prefix: `session:`

**RetrievalCache** (`tool/storage/retrieval_cache.go`):
- Caches embedding vectors and retrieval results in Redis
- TTL: 1h, Key prefixes: `embedding:`, `retrieval:`

These are separate from the conversation memory system (`tool/memory`).

---

## 9. Key Architectural Patterns

1. **Graph-based flow composition**: All flows are Eino `compose.Graph` instances compiled at startup, cached as package-level singletons using `sync.Once`

2. **Factory pattern for models**: `chat_model.GetChatModel(ctx, name)` uses registry of factory functions

3. **State management via `compose.ProcessState`**: Graph nodes read/write shared state through Eino state mechanism

4. **Dual retrieval with RRF reranking**: Retriever graph fans out to Milvus (vector) + Elasticsearch (keyword) in parallel, merges with Reciprocal Rank Fusion

5. **Graceful degradation**: Redis-dependent stores auto-fallback to in-memory maps when Redis unavailable

---

## 10. File Paths Reference

**Memory system (core):**
- `tool/memory/session.go` - Session struct, SummaryMessageExtraKey constant
- `tool/memory/store.go` - Store interface, InMemoryStore implementation
- `tool/memory/memory.go` - inMemoryStore for Eino CheckPointStore (unused)
- `tool/memory/compressor.go` - Summarizer with LLM-based compression

**Memory integration:**
- `flow/rag_chat.go` - RAG chat graph using memory
- `rag/rag_tools/query.go` - Rewrite function using session summary/history

**API endpoints:**
- `api/rag_chat_stream.go` - SSE streaming RAG chat endpoint
- `api/rag_ask.go` - Non-streaming RAG ask endpoint

**Initialization:**
- `main.go` - Lines 105-113 create InMemoryStore and initialize RAGChatFlow

**Related storage:**
- `tool/storage/checkpoint_store.go` - Redis checkpoint store
- `tool/storage/session.go` - Redis session context store
- `tool/storage/retrieval_cache.go` - Redis retrieval cache
- `tool/storage/redis.go` - Redis client initialization
