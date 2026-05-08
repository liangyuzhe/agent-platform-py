"""LangGraph 图共享状态定义。"""

from typing import Annotated, Any, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from langchain_core.documents import Document


def keep_existing_query(current: str | None, incoming: str | None) -> str:
    """Keep query stable when parent/child graph state is merged in one step."""
    if current:
        return current
    return incoming or ""


class RAGChatState(TypedDict):
    """RAG Chat 图的状态。"""
    input: dict                                  # {"session_id": str, "query": str, "rag_mode"?: str}
    session_id: str
    query: str
    rag_mode: str                                # "traditional" or "parent"
    session: dict                                # Session 数据
    rewritten_query: str                         # 重写后的查询
    docs: list[Document]                         # 检索到的文档
    messages: Annotated[list[BaseMessage], add_messages]  # 对话消息
    answer: str                                  # 最终回答


class SQLReactState(TypedDict):
    """SQL React 图的状态。"""
    query: Annotated[str, keep_existing_query]   # 当前用户问题（可能是代词化的）
    rewritten_query: str                         # 上下文化后的独立问题
    enhanced_query: str                          # 业务术语增强后的查询
    chat_history: list[dict]                     # 对话历史 [{"role": str, "content": str}]
    table_names: list[str]                       # 所有可用表名（启动时缓存）
    selected_tables: list[str]                   # LLM 选中的相关表名
    table_relationships: list[dict]              # 表关系 [{from_table, from_column, to_table, to_column}]
    evidence: list[str]                          # 业务知识检索结果
    few_shot_examples: list[str]                 # SQL Q&A few-shot 参考
    docs: list[Document]                         # 检索到的表结构（从 t_semantic_model 构建）
    semantic_model: dict                         # {table_name: {column_name: row_dict}}
    sql: str                                     # 生成的 SQL
    is_sql: bool                                 # 是否为 SQL 输出
    answer: str                                  # 非 SQL 回答
    approved: bool                               # 是否已审批
    refine_feedback: str                         # 修改意见（用户拒绝或错误分析生成）
    result: str                                  # SQL 执行结果
    safety_report: dict | None                   # 安全分析报告
    error: str | None                            # SQL 执行错误信息
    retry_count: int                             # 重试次数
    execution_history: list[dict]                # 执行历史 [{sql, result, error}]
    reflection_notice: str                       # 结果异常反思提示


class AnalystState(TypedDict):
    """数据分析图的状态。"""
    sql_result: str
    parsed_data: dict                            # ParsedData
    statistics: dict                             # Statistics
    text_analysis: str
    chart_config: dict
    analysis_result: dict                        # AnalysisResult


class FinalGraphState(TypedDict):
    """主调度图的状态。"""
    query: Annotated[str, keep_existing_query]
    session_id: str
    chat_history: list[dict]                     # 对话历史 [{"role": str, "content": str}]
    intent: str                                  # sql_query | anomaly_detect | reconciliation | report | audit | knowledge | chat
    sql: str
    result: str
    answer: str
    status: str                                  # "pending" | "approved" | "rejected" | "completed"
