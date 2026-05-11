"""RAG Chat 图：文档检索增强对话。"""

import asyncio
import logging

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

from agents.flow.state import RAGChatState
from agents.tool.memory.store import get_session, save_session
from agents.tool.memory.compressor import compress_session
from agents.rag.query_rewrite import rewrite_query
from agents.rag.retriever import get_hybrid_retriever
from agents.model.chat_model import get_chat_model
from agents.tool.token_counter import TokenCounter
from agents.config.settings import settings
from agents.tool.trace.tracing import callbacks_from_config, child_trace_config

logger = logging.getLogger(__name__)


async def preprocess(state: RAGChatState) -> dict:
    """加载 Session。"""
    inp = state["input"]
    session = get_session(inp["session_id"])
    return {
        "session": session.model_dump(),
        "query": inp["query"],
        "rewritten_query": inp.get("rewritten_query", ""),
        "session_id": inp["session_id"],
        "rag_mode": inp.get("rag_mode", settings.rag.mode),
    }


async def rewrite(state: RAGChatState, config=None) -> dict:
    """查询重写：利用记忆上下文化查询。

    如果 rewritten_query 已由外层 classify 提供，直接复用，跳过 LLM 调用。
    """
    # 外层已重写，跳过
    existing = state.get("rewritten_query", "")
    if existing:
        return {"rewritten_query": existing}

    session = state.get("session", {})
    history = session.get("history", [])
    summary = session.get("summary", "")

    if not history and not summary:
        return {"rewritten_query": state["query"]}

    history_dicts = [
        {"role": h["role"], "content": h["content"]} for h in history
    ]

    try:
        rewritten = await asyncio.wait_for(
            rewrite_query(
                summary=summary,
                history=history_dicts,
                query=state["query"],
                config=config,
            ),
            timeout=settings.resilience.llm_rewrite_timeout,
        )
    except Exception as e:
        logger.warning("Query rewrite failed, using original: %s", e)
        rewritten = state["query"]
    return {"rewritten_query": rewritten}


async def retrieve(state: RAGChatState, config=None) -> dict:
    """双路检索 + RRF 融合 + Cross-Encoder 重排序。"""
    query = state.get("rewritten_query", state["query"])
    mode = state.get("rag_mode", settings.rag.mode)
    session_id = state.get("session_id", "")
    callbacks = callbacks_from_config(config)

    try:
        if mode == "parent":
            from agents.rag.parent_retriever import ParentDocumentRetriever
            retriever = ParentDocumentRetriever()
        else:
            # user_document 按 session_id 隔离检索
            retriever = get_hybrid_retriever(
                source_filter="user_document",
                session_id_filter=session_id if session_id else None,
            )

        # Run sync retrieval in thread pool with timeout
        docs = await asyncio.wait_for(
            asyncio.to_thread(retriever.retrieve, query, callbacks=callbacks),
            timeout=settings.resilience.milvus_timeout,
        )
    except Exception as e:
        logger.warning("Retrieval failed, continuing with empty docs: %s", e)
        docs = []

    return {"docs": docs}


async def construct_messages(state: RAGChatState) -> dict:
    """组装最终 Prompt，带 Token 预算管理。"""
    counter = TokenCounter()
    model_context = 32768  # 默认上下文窗口
    budget = model_context - 4096  # 预留给响应

    parts = []
    session = state.get("session", {})

    # 1. 摘要记忆
    summary = session.get("summary", "")
    if summary:
        parts.append(f"背景摘要: {summary}")

    # 2. 工作记忆（历史消息）
    for msg in session.get("history", []):
        parts.append(f"[{msg['role']}]: {msg['content']}")

    # 3. 检索文档
    doc_texts = [doc.page_content for doc in state.get("docs", [])]
    if doc_texts:
        parts.append(f"参考知识:\n{'\\n---\\n'.join(doc_texts)}")

    # 4. 当前查询
    parts.append(state["query"])

    # Token 预算裁剪
    fitted = counter.fit_to_budget(parts, budget)
    context = "\n\n".join(fitted)

    system = SystemMessage(content=(
        "你是一个智能助手。根据参考知识回答用户问题。"
        "只使用与问题相关的信息，忽略无关内容。"
        "直接回答问题，不要复述参考知识原文。"
        "如果参考知识中没有相关信息，根据你的知识回答。"
    ))
    messages = [system, HumanMessage(content=context)]
    return {"messages": messages}


async def chat(state: RAGChatState, config=None) -> dict:
    """LLM 生成响应。"""
    model = get_chat_model(settings.chat_model_type)
    response = await asyncio.wait_for(
        model.ainvoke(
            state["messages"],
            config=child_trace_config(config, "rag_chat.chat.llm", tags=["llm", "rag_chat"]),
        ),
        timeout=settings.resilience.llm_timeout,
    )

    # 更新记忆
    session = state.get("session", {})
    history = session.get("history", [])
    history.append({"role": "user", "content": state["query"]})
    history.append({"role": "assistant", "content": response.content})
    session["history"] = history

    # 异步压缩 + 保存（不阻塞响应）
    asyncio.create_task(_compress_and_save(state["session_id"], session))

    return {"answer": response.content, "messages": [response]}


async def _compress_and_save(session_id: str, session: dict):
    """后台任务：压缩记忆并保存。"""
    from agents.tool.memory.session import Session
    from agents.tool.memory.vector_store import index_long_term_memory
    session_obj = Session(**session)
    if len(session_obj.history) <= settings.memory.summary_max_history_len:
        save_session(session_id, session_obj)
        return

    # 压缩记忆（如果历史过长）
    try:
        model = get_chat_model(settings.chat_model_type)
        archived = await asyncio.wait_for(
            compress_session(
                session_obj,
                llm=model,
                max_history_len=settings.memory.summary_max_history_len,
                keep_recent=settings.memory.summary_keep_recent,
            ),
            timeout=settings.resilience.llm_rewrite_timeout,
        )
        if archived:
            doc_id = await asyncio.to_thread(index_long_term_memory, session_id, archived, session_obj.summary)
            if doc_id:
                session_obj.preferences["_has_long_term_memory"] = "1"
    except Exception as e:
        logger.warning("Memory compression failed: %s", e)

    save_session(session_id, session_obj)


def build_rag_chat_graph():
    """构建 RAG Chat 图。"""
    graph = StateGraph(RAGChatState)

    graph.add_node("preprocess", preprocess)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("construct_messages", construct_messages)
    graph.add_node("chat", chat)

    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "construct_messages")
    graph.add_edge("construct_messages", "chat")
    graph.add_edge("chat", END)

    return graph.compile()
