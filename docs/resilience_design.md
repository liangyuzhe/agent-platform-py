# 熔断降级与 Fallback 设计方案

## 一、DataAgent 容错模式详解

DataAgent（`/Users/a0000/project/DataAgent`）是 Java/Spring Boot 项目，**无熔断器库**（无 resilience4j/spring-retry），所有容错逻辑手写在图节点和 Dispatcher 中。

### 1. SQL 执行重试（核心模式）

DataAgent 的 SQL 重试是**图循环**而非简单重试：

```
SqlGenerateNode → SemanticConsistencyNode → SqlExecuteNode
       ↑                    ↑                     |
       |     (语义不通过)     |     (执行失败)        |
       └────────────────────┴─────────────────────┘
                     SqlRetryDto 携带错误分类
```

- **最大重试**：`maxSqlRetryCount=10`（可配置，`DataAgentProperties`）
- **错误分类**：`SqlRetryDto` 区分 `semanticFail` 和 `sqlExecuteFail`
- **重试上下文**：`SqlGenerateNode.handleRetryGenerateSql()` 将原始 SQL + 错误信息注入重试 prompt
- **语义一致性**：`SemanticConsistencyNode` 验证 SQL 语义，失败标记 `SqlRetryDto.semantic()`

```java
// SqlGenerateNode.java:93-106
if (sqlRetryDto.sqlExecuteFail()) {
    handleRetryGenerateSql(sqlRetryDto.reason(), ...);  // 带错误上下文重生成
}
if (sqlRetryDto.semanticFail()) {
    handleRetryGenerateSql(sqlRetryDto.reason(), ...);
}
```

### 2. 错误码分类（ErrorCodeEnum）

```java
// ErrorCodeEnum.java - 16 种 SQLState 映射
连接失败: 08001, 08S01, 08002, 08003, 08004, 08006, 08P01
认证错误: 28P01(密码), 28000(认证), 42501(权限)
数据库错误: 3D000, 42000(库不存在), 3D070(schema不存在)
```

用于 `AbstractDBConnectionPool.errorMapping(sqlState)` 分类连接错误。

### 3. DB 连接重试 + 线性退避

```java
// AbstractDBConnectionPool.java:82-135
maxRetries = 3;
retryDelay = 1000;  // ms
for (int attempt = 1; attempt <= maxRetries; attempt++) {
    try { return dataSource.getConnection(); }
    catch (SQLException e) {
        Thread.sleep(retryDelay * attempt);  // 1s, 2s, 3s
    }
}
throw new RuntimeException("Failed after " + maxRetries + " attempts");
```

Druid 连接池配置：`maxWait=10s`, `breakAfterAcquireFailure=true`, `connectionErrorRetryAttempts=2`

### 4. Python 执行降级模式（PYTHON_FALLBACK_MODE）

```
PythonExecuteNode (失败 N 次后)
  ├─ PYTHON_FALLBACK_MODE = true
  ├─ 返回空 JSON "{}"
  ↓
PythonExecutorDispatcher
  ├─ 检测到 FALLBACK_MODE → 路由到 PYTHON_ANALYZE_NODE
  ↓
PythonAnalyzeNode
  ├─ 检测到 FALLBACK_MODE → 返回降级消息 "Python 高级分析功能暂时不可用"
  └─ 不调用 LLM，直接返回
```

- 最大重试：`pythonMaxTriesCount=5`（可配置）
- 降级策略：重试耗尽 → 进入 fallback 模式 → 跳过执行直接分析 → 返回降级提示

### 5. 图表生成降级（catch-and-swallow）

```java
// SqlExecuteNode.java:208-267
try {
    displayStyle = llmService.call(...)
        .block(Duration.ofMillis(properties.getEnrichSqlResultTimeout()));  // 3s 超时
} catch (Exception e) {
    // 不抛异常，允许流程继续
    displayStyle = null;
}
```

图表配置失败 → 返回 null → SQL 结果正常返回但无图表。

### 6. 向量检索降级

```java
// EvidenceRecallNode.java:134-138
try {
    documents = vectorStore.similaritySearch(...);
} catch (Exception e) {
    emitError(displaySink, e);
    return Map.of(EVIDENCE, "");  // 返回空 evidence，流程继续
}
```

### 7. JSON 解析 LLM 自修复

```java
// JsonParseUtil.java:72-105
MAX_RETRY_COUNT = 3;
for (int i = 0; i < MAX_RETRY_COUNT; i++) {
    try { return objectMapper.readValue(json, clazz); }
    catch (JsonProcessingException e) {
        json = callLlmToFix(json, e.getMessage());  // LLM 修复 JSON
    }
}
throw new IllegalArgumentException("无法解析 JSON");
```

### 8. 可配置超时汇总

| 参数 | 默认值 | 配置项 |
|------|--------|--------|
| SQL 最大重试 | 10 | `max-sql-retry-count` |
| Python 最大重试 | 5 | `python-max-tries-count` |
| SQL 执行超时 | 30s | `SqlExecutor.STATEMENT_TIMEOUT` |
| 图表 LLM 超时 | 3s | `enrichSqlResultTimeout` |
| 标题生成 LLM 超时 | 15s | 硬编码 |
| Python 代码超时 | 60s | `codeTimeout` |
| REST 连接超时 | 600s | `rest.connect.timeout` |
| DB 连接池 maxWait | 10s | Druid 配置 |

### 9. 关键设计思想

| 思想 | DataAgent 实现 |
|------|---------------|
| **catch-and-mark** | 向量失败不回滚 MySQL，标记 `EmbeddingStatus.FAILED` |
| **retry-with-context** | SQL 重试时带原始 SQL + 错误信息给 LLM |
| **error classification** | `SqlRetryDto` 区分语义失败 vs 执行失败 |
| **fallback mode** | Python 超限 → `FALLBACK_MODE` → 降级提示 |
| **catch-and-swallow** | 图表失败 → null → 不阻塞主流程 |
| **LLM auto-fix** | JSON 解析失败 → LLM 修复 → 重试 |
| **可配置重试** | 所有重试次数通过配置文件控制 |

---

## 二、与 DataAgent 对比

| 能力 | DataAgent | 我们 | 差距 |
|------|-----------|------|------|
| SQL 重试（LLM 引导） | 最多 10 次，可配置 | 最多 5 次，可配置 | ✅ 已有 |
| SQL 重试上下文 | 带原始 SQL + 错误 | 带原始 SQL + 错误 | ✅ 已有 |
| 错误码分类 | 16 种 SQLState | 16 种 SQLState + `is_retryable()` | ✅ 已有 |
| 可配置重试次数 | 配置文件 | `ResilienceSettings`（环境变量） | ✅ 已有 |
| DB 连接重试 | 3 次 + 线性退避 | MCP 长连接 | ⚠️ 场景不同 |
| SQL 执行超时 | 30s | 15s（可配置） | ✅ 已有 |
| LLM 超时 | 仅图表 3s | 全部 15~60s（可配置） | ✅ 超越 |
| LLM 重试 | ❌ 无 | max_retries=2 | ✅ 超越 |
| 向量检索降级 | catch → 返回空 | catch → 返回空 | ✅ 已有 |
| 知识入库降级 | mark FAILED | log only | **需补齐** |
| Python 降级模式 | FALLBACK_MODE | 不适用 | - |
| 图表降级 | catch-and-swallow | 不适用 | - |
| JSON LLM 自修复 | 3 次 LLM 修复 | 无 | ⚠️ 可选 |
| 熔断器 | ❌ 无 | ❌ 无 | 双方都无 |

---

## 三、需补齐的能力

### P1：错误码分类 + 可配置重试 ✅

新建 `agents/tool/sql_tools/error_codes.py`（已实现）：

```python
SQL_ERROR_CODES = {
    # 连接错误（可重试）
    "08001": ("连接建立失败", True),
    "08S01": ("连接中断", True),
    "08003": ("连接不存在", True),
    "08004": ("服务器拒绝连接", True),
    "08006": ("连接被关闭", True),
    # 认证错误（不可重试）
    "28P01": ("密码错误", False),
    "28000": ("认证失败", False),
    "42501": ("权限不足", False),
    # 数据库错误（不可重试）
    "3D000": ("数据库不存在", False),
    "42000": ("语法错误或数据库不存在", False),
    "3D070": ("Schema 不存在", False),
    # MySQL 特有
    "42S02": ("表不存在", False),
    "42S22": ("列不存在", False),
    "23000": ("约束违反", False),
    "HY000": ("通用错误", False),
}

def is_retryable(error_msg: str) -> bool:
    """判断 SQL 错误是否值得重试。连接类错误重试，语法/权限类不重试。"""
    for code, (desc, retryable) in SQL_ERROR_CODES.items():
        if code in error_msg:
            return retryable
    return False  # 未知错误默认不重试
```

在 `settings.py` 新增配置：

```python
class ResilienceSettings(BaseModel):
    max_sql_retries: int = 5          # SQL 重试次数
    sql_execution_timeout: float = 15  # SQL 执行超时
    milvus_timeout: float = 8          # Milvus 查询超时
    llm_timeout: float = 60            # LLM 调用超时
```

在 `sql_react.py` 的 `route_after_execute` 中使用：

```python
def route_after_execute(state):
    if not state.get("error"):
        return END
    if not is_retryable(state["error"]):
        return END  # 语法/权限错误不重试
    if state.get("retry_count", 0) < settings.resilience.max_sql_retries:
        return "error_analysis"
    return END
```

### P2：知识入库降级（catch-and-mark）

seed 脚本和 admin API 的向量入库失败时，标记状态而非阻塞：

```python
try:
    client.insert(collection_name=..., data=records)
except Exception as e:
    logger.warning("Vector insert failed for %d records: %s", len(records), e)
    # 标记 MySQL 记录为 FAILED，不阻塞流程
    mark_embedding_failed(record_ids, str(e))
```

### P3：DB 连接重试（MCP 场景）

MCP 使用长连接，连接重试在 MCP server 内部处理。我们只需确保 `execute_sql` 超时后正确进入 error_analysis。

---

## 四、实施计划

### Phase 1：超时保护 ✅（迭代 7 已完成）

### Phase 2：错误码分类 + 可配置重试 ✅（迭代 8 已完成）

| 文件 | 改动 |
|------|------|
| 新建 `agents/tool/sql_tools/error_codes.py` | 16 种 SQLState 错误码 + `is_retryable()` |
| `agents/flow/sql_react.py` | `route_after_execute` 使用 `is_retryable` 判断，超时从配置读取 |
| `agents/config/settings.py` | 新增 `ResilienceSettings`（5 个可配置参数，环境变量） |
| `agents/flow/rag_chat.py` | 超时从 `settings.resilience` 读取 |
| `agents/tool/sql_tools/mcp_client.py` | SQL 执行超时从配置读取 |

### Phase 3：知识入库降级

| 文件 | 改动 |
|------|------|
| `scripts/seed_*.py` | 向量入库失败 → 标记 status=FAILED |
| `agents/api/routers/admin.py` | 新增 retry_embedding 端点 |

### Phase 4：熔断器（可选，双方都无）

| 文件 | 改动 |
|------|------|
| 新建 `agents/tool/resilience/circuit_breaker.py` | 熔断器实现 |
| `agents/rag/retriever.py` | Milvus 调用加熔断 |
| `agents/model/chat_model.py` | LLM 调用加熔断 |
