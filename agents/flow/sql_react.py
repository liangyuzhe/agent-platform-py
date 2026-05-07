"""SQL React 图：自然语言 -> SQL -> 审批 -> 执行，支持自动纠错重试。"""

import asyncio
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from agents.flow.state import SQLReactState
from agents.model.chat_model import get_chat_model
from agents.model.format_tool import create_format_tool
from agents.tool.sql_tools.safety import SQLSafetyChecker
from agents.tool.sql_tools.error_codes import is_retryable
from agents.rag.retriever import recall_business_knowledge, recall_agent_knowledge, get_semantic_model_by_tables, load_full_table_metadata, get_table_relationships
from agents.rag.query_rewrite import rewrite_query
from agents.tool.storage.checkpoint import get_checkpointer
from agents.config.settings import settings

try:
    from elasticsearch import Elasticsearch
except ImportError:
    Elasticsearch = None

logger = logging.getLogger(__name__)


async def contextualize_query(state: SQLReactState) -> dict:
    """利用对话历史将代词化查询重写为独立查询。

    如果 rewritten_query 已由外层 classify 提供，直接复用，跳过 LLM 调用。
    """
    # 外层已重写，跳过
    existing = state.get("rewritten_query", "")
    if existing:
        return {"rewritten_query": existing}

    query = state.get("query", "")
    chat_history = state.get("chat_history", [])

    if not chat_history:
        return {"rewritten_query": query}

    # 提取最近对话摘要和历史
    summary = ""
    history_dicts = []
    for h in chat_history:
        if h.get("role") == "system" and h.get("content", "").startswith("[对话摘要]"):
            summary = h["content"].replace("[对话摘要] ", "")
        else:
            history_dicts.append(h)

    if not history_dicts:
        return {"rewritten_query": query}

    try:
        rewritten = await asyncio.wait_for(
            rewrite_query(
                summary=summary,
                history=history_dicts[-6:],  # 最近 3 轮
                query=query,
            ),
            timeout=settings.resilience.llm_rewrite_timeout,
        )
        logger.info("contextualize_query: '%s' -> '%s'", query, rewritten)
        return {"rewritten_query": rewritten}
    except Exception as e:
        logger.warning("Query contextualization failed, using original: %s", e)
        return {"rewritten_query": query}


async def query_enhance(state: SQLReactState) -> dict:
    """用证据（业务知识）翻译查询中的业务术语，增强向量检索命中率。

    示例：
        Query: "华东区上月GMV是多少"
        Evidence: "GMV = 已支付订单总额", "华东包含上海、江苏、浙江..."
        Enhanced: "华东区（上海、江苏、浙江）上月已支付订单总额是多少"
    """
    query = state.get("rewritten_query") or state.get("query", "")
    evidence = state.get("evidence", [])

    if not evidence:
        return {"enhanced_query": query}

    evidence_text = "\n".join(evidence)

    try:
        model = get_chat_model(settings.chat_model_type)
        response = await asyncio.wait_for(
            model.ainvoke([
                SystemMessage(content=(
                    "你是一个查询增强助手。根据业务知识，将用户查询中的业务术语、缩写、"
                    "隐含条件翻译/展开为数据库字段或通用表达，使查询更适合数据库检索。\n\n"
                    "规则：\n"
                    "1. 只翻译/展开查询中出现的业务术语，不要添加查询中没有的条件\n"
                    "2. 保持查询的原始意图不变\n"
                    "3. 如果知识中有区域/维度的映射（如华东包含哪些省），用括号补充\n"
                    "4. 如果知识中有术语定义（如GMV=已支付订单总额），替换为更明确的表达\n"
                    "5. 只输出增强后的查询，不要解释"
                )),
                HumanMessage(content=f"业务知识:\n{evidence_text}\n\n用户查询: {query}"),
            ]),
            timeout=settings.resilience.llm_rewrite_timeout,
        )
        enhanced = response.content.strip()
        if enhanced:
            logger.info("query_enhance: '%s' -> '%s'", query[:80], enhanced[:80])
            return {"enhanced_query": enhanced}
    except Exception as e:
        logger.warning("query_enhance failed, using original: %s", e)

    return {"enhanced_query": query}


async def select_tables(state: SQLReactState) -> dict:
    """表选择：从 MySQL t_semantic_model 加载表名+描述 → LLM 精选。

    不再依赖 Milvus 向量检索，直接从统一语义模型获取表元数据。
    """
    query = state.get("enhanced_query") or state.get("rewritten_query") or state.get("query", "")

    # Stage 1: 从 MySQL 加载全量表名 + 描述
    metadata_list = []
    try:
        metadata_list = await asyncio.wait_for(
            asyncio.to_thread(load_full_table_metadata),
            timeout=settings.resilience.milvus_timeout,
        )
    except Exception as e:
        logger.warning("Failed to load table metadata: %s", e)

    if not metadata_list:
        return {"selected_tables": []}

    candidate_tables = [m["table_name"] for m in metadata_list]

    # Stage 2: 候选少于等于 3 个，直接使用（省一次 LLM 调用）
    if len(candidate_tables) <= 3:
        selected = candidate_tables
        relationships = []
        try:
            relationships = await asyncio.wait_for(
                asyncio.to_thread(get_table_relationships, selected),
                timeout=settings.resilience.milvus_timeout,
            )
        except Exception as e:
            logger.warning("Failed to load table relationships: %s", e)
        logger.info("select_tables: %d candidates, using directly: %s, %d relationships",
                    len(candidate_tables), selected, len(relationships))
        return {"selected_tables": selected, "table_relationships": relationships}

    # Stage 2: 候选 > 3 个，LLM 精选
    model = get_chat_model(settings.chat_model_type)

    table_metadata = {m["table_name"]: m.get("table_comment", "") for m in metadata_list}
    names_with_desc = []
    for t in candidate_tables:
        desc = table_metadata.get(t, "")
        if desc:
            names_with_desc.append(f"- {t}: {desc}")
        else:
            names_with_desc.append(f"- {t}")
    names_text = "\n".join(names_with_desc)

    response = await model.ainvoke([
        SystemMessage(content=f"""你是一个数据库专家。根据用户的问题，从候选表名中选出需要用到的表。

候选表名:
{names_text}

要求：
1. 只返回需要用到的表名，用逗号分隔
2. 如果需要多表关联，选出所有涉及的表
3. 如果问题与数据库无关（如闲聊），返回空
4. 只返回表名，不要其他内容"""),
        HumanMessage(content=query),
    ])

    raw = response.content.strip()
    if not raw:
        selected = candidate_tables
    else:
        selected = [n.strip() for n in raw.split(",") if n.strip() in candidate_tables]
        if not selected:
            selected = candidate_tables

    relationships = []
    try:
        relationships = await asyncio.wait_for(
            asyncio.to_thread(get_table_relationships, selected),
            timeout=settings.resilience.milvus_timeout,
        )
    except Exception as e:
        logger.warning("Failed to load table relationships: %s", e)

    logger.info("select_tables: LLM selected %d from %d candidates: %s, %d relationships",
                len(selected), len(candidate_tables), selected, len(relationships))
    return {"selected_tables": selected, "table_relationships": relationships}


async def recall_evidence(state: SQLReactState) -> dict:
    """并行检索业务知识 + 智能体知识库，注入 SQL 生成上下文。"""
    query = state.get("rewritten_query") or state.get("query", "")
    if not query:
        return {"evidence": [], "few_shot_examples": []}

    async def _recall_business():
        try:
            docs = await asyncio.wait_for(
                asyncio.to_thread(recall_business_knowledge, query, 5),
                timeout=settings.resilience.milvus_timeout,
            )
            result = [d.page_content for d in docs if d.metadata.get("score", 0) > 0.3]
            logger.info("recall_evidence: %d business knowledge entries", len(result))
            return result
        except Exception as e:
            logger.warning("Business knowledge recall failed: %s", e)
            return []

    async def _recall_agent():
        try:
            # 增加 top_k 以确保过滤后仍有足够的 SQL 示例
            docs = await asyncio.wait_for(
                asyncio.to_thread(recall_agent_knowledge, query, 10),
                timeout=settings.resilience.milvus_timeout,
            )
            result = [d.page_content for d in docs if d.metadata.get("score", 0) > 0.3]
            logger.info("recall_evidence: %d agent knowledge entries", len(result))
            return result
        except Exception as e:
            logger.warning("Agent knowledge recall failed: %s", e)
            return []

    # 并行检索，耗时 = max(单个) 而非 sum
    evidence, few_shot = await asyncio.gather(_recall_business(), _recall_agent())

    return {"evidence": evidence, "few_shot_examples": few_shot}




def _build_schema_docs_from_semantic(semantic_model: dict) -> list[Document]:
    """从语义模型构建 schema 文档（替代 Milvus 向量检索）。"""
    docs = []
    for table_name, columns in semantic_model.items():
        lines = [f"表名: {table_name}"]
        for col_name, meta in columns.items():
            col_type = meta.get("column_type", "")
            col_comment = meta.get("column_comment", "")
            business_name = meta.get("business_name", "")
            synonyms = meta.get("synonyms", "")
            description = meta.get("business_description", "")
            is_pk = meta.get("is_pk", 0)
            is_fk = meta.get("is_fk", 0)
            ref_table = meta.get("ref_table", "")
            ref_column = meta.get("ref_column", "")

            parts = [col_name]
            if col_type:
                parts.append(col_type)
            if is_pk:
                parts.append("PRIMARY KEY")
            if is_fk and ref_table:
                parts.append(f"REFERENCES {ref_table}({ref_column})")
            if col_comment:
                parts.append(f"-- {col_comment}")
            if business_name:
                parts.append(f"[业务名: {business_name}]")
            if synonyms:
                parts.append(f"[同义词: {synonyms}]")
            if description:
                parts.append(f"[描述: {description}]")

            lines.append(" ".join(parts))

        doc = Document(
            page_content="\n".join(lines),
            metadata={"table_name": table_name, "source": "semantic_model"},
        )
        docs.append(doc)
    return docs


async def sql_retrieve(state: SQLReactState, config=None) -> dict:
    """从 MySQL t_semantic_model 按表名加载完整 schema + 业务映射。"""
    selected = state.get("selected_tables", [])
    tables = selected or state.get("table_names", [])
    result = {}

    # 1. 加载语义模型（包含完整 schema + 业务映射）
    if tables:
        try:
            semantic = await asyncio.wait_for(
                asyncio.to_thread(get_semantic_model_by_tables, tables),
                timeout=settings.resilience.milvus_timeout,
            )
            result["semantic_model"] = semantic
        except Exception as e:
            logger.warning("Semantic model load failed: %s", e)
            result["semantic_model"] = {}
    else:
        result["semantic_model"] = {}

    # 2. 从语义模型构建 schema 文档（不做过滤，靠业务知识+语义模型让 LLM 判断字段）
    if result["semantic_model"]:
        result["docs"] = _build_schema_docs_from_semantic(result["semantic_model"])
    else:
        result["docs"] = []

    return result


async def check_docs(state: SQLReactState) -> dict:
    """检查是否检索到相关表结构。"""
    docs = state.get("docs", [])
    if not docs:
        return {
            "answer": "未找到相关的数据库表结构信息，无法生成 SQL。请先上传数据库表结构文档。",
            "is_sql": False,
        }
    return {}


_MAX_TABLE_SEARCH_ROUNDS = 3


async def _retrieve_missing_tables(missing_tables: list[str], existing_docs: list) -> list:
    """Re-retrieve schema docs for missing table names from t_semantic_model."""
    semantic = await asyncio.to_thread(get_semantic_model_by_tables, missing_tables)
    if not semantic:
        return []
    new_docs = _build_schema_docs_from_semantic(semantic)
    existing_names = {d.metadata.get("table_name", "") for d in existing_docs}
    unique_new = [d for d in new_docs if d.metadata.get("table_name", "") not in existing_names]
    return unique_new


def _build_sql_messages(query: str, docs_text: str, refine_context: str, history_context: str, evidence_text: str, few_shot_text: str, relationships_text: str = "") -> list:
    """Build messages for sql_generate LLM call."""
    return [
        SystemMessage(content=f"""你是一个 SQL 专家。根据用户的问题和数据库表结构信息，生成正确的 SQL 查询。

表结构信息:
        {docs_text}{relationships_text}{evidence_text}{few_shot_text}{refine_context}{history_context}

要求：
1. 使用 MySQL 语法
2. 只生成 SELECT 查询（禁止 DROP/DELETE/TRUNCATE 等危险操作）
3. 如果有执行历史和错误信息，请分析错误原因并生成修正后的 SQL
4. 如果现有表结构不足以生成正确的 SQL（例如缺少关联表、缺少字段所在的表），
   请设置 needs_more_tables=true 并在 missing_tables 中列出你需要的表名
5. 参考相似问题的 SQL 示例，但要根据实际表结构调整
6. 表结构中已包含字段的业务名称、同义词和描述，生成 SQL 时优先使用物理字段名，但可参考业务信息理解字段含义
7. 使用表关系信息来确定正确的 JOIN 条件
8. 使用 format_response 工具输出结果"""),
        HumanMessage(content=query),
    ]


async def sql_generate(state: SQLReactState) -> dict:
    """LLM 生成 SQL，支持自动补表（最多重试 3 次）。"""
    model = get_chat_model(settings.chat_model_type)
    model_with_tools = model.bind_tools([create_format_tool()])

    # 如果有修改意见，加入上下文
    refine_context = ""
    if state.get("refine_feedback"):
        refine_context = f"\n修改意见: {state['refine_feedback']}"

    # 如果有执行历史（纠错场景），加入上下文
    history_context = ""
    execution_history = state.get("execution_history", [])
    if execution_history:
        history_lines = []
        for i, h in enumerate(execution_history, 1):
            entry = f"第{i}次尝试: SQL={h['sql']}"
            if h.get("error"):
                entry += f"\n  错误: {h['error']}"
            elif h.get("result"):
                entry += f"\n  结果: {h['result'][:200]}"
            history_lines.append(entry)
        history_context = f"\n执行历史:\n" + "\n".join(history_lines)

    # Business knowledge evidence
    evidence = state.get("evidence", [])
    evidence_text = ""
    if evidence:
        evidence_text = "\n\n业务知识:\n" + "\n".join(evidence)

    # Agent knowledge few-shot examples
    few_shot = state.get("few_shot_examples", [])
    few_shot_text = ""
    if few_shot:
        few_shot_text = "\n\n相似问题参考:\n" + "\n---\n".join(few_shot)

    # Semantic model (字段级业务映射) - 已合并到 docs 中，不再单独构建
    # 保留 semantic_model 用于其他用途（如关键词过滤）

    # Table relationships
    relationships = state.get("table_relationships", [])
    relationships_text = ""
    if relationships:
        lines = ["\n\n表关系（外键关联）:"]
        for rel in relationships:
            lines.append(f"  {rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']}")
        relationships_text = "\n".join(lines)

    # Accumulate docs across re-retrieval rounds
    all_docs = list(state.get("docs", []))

    # 使用上下文化后的查询生成 SQL，但保留原始查询作为参考
    effective_query = state.get("rewritten_query") or state.get("query", "")
    original_query = state.get("query", "")
    query_for_sql = effective_query
    if effective_query != original_query:
        query_for_sql = f"{effective_query}\n（用户原始问题: {original_query}）"

    for round_idx in range(_MAX_TABLE_SEARCH_ROUNDS):
        docs_text = "\n\n".join([d.page_content for d in all_docs])

        messages = _build_sql_messages(query_for_sql, docs_text, refine_context, history_context, evidence_text, few_shot_text, relationships_text)
        response = await model_with_tools.ainvoke(messages)

        if not response.tool_calls:
            return {"answer": response.content, "sql": response.content, "is_sql": False, "error": None}

        tool_call = response.tool_calls[0]
        args = tool_call["args"]
        needs_more = args.get("needs_more_tables", False)
        missing = args.get("missing_tables", [])

        # If LLM says it has enough tables, return the result
        if not needs_more or not missing:
            answer_text = args.get("answer", "")
            is_sql = args.get("is_sql", False)
            logger.info("sql_generate: produced SQL after %d round(s)", round_idx + 1)
            return {
                "answer": answer_text,
                "sql": answer_text if is_sql else answer_text,
                "is_sql": is_sql,
                "error": None,
            }

        # LLM needs more tables — re-retrieve
        logger.info("sql_generate: round %d, LLM needs tables: %s", round_idx + 1, missing)
        new_docs = await _retrieve_missing_tables(missing, all_docs)
        if not new_docs:
            logger.info("sql_generate: no new docs found for %s, using what we have", missing)
            answer_text = args.get("answer", "")
            is_sql = args.get("is_sql", False)
            return {
                "answer": answer_text,
                "sql": answer_text if is_sql else answer_text,
                "is_sql": is_sql,
                "error": None,
            }
        all_docs.extend(new_docs)
        logger.info("sql_generate: added %d new docs, total %d", len(new_docs), len(all_docs))

    # Exhausted retries — return what we have
    answer_text = args.get("answer", "")
    is_sql = args.get("is_sql", False)
    return {
        "answer": answer_text,
        "sql": answer_text if is_sql else answer_text,
        "is_sql": is_sql,
        "error": None,
    }


async def safety_check(state: SQLReactState) -> dict:
    """SQL 安全分析。"""
    if not state.get("is_sql"):
        return {"safety_report": None}

    checker = SQLSafetyChecker()
    report = checker.check(state["sql"])

    if not report.is_safe:
        return {
            "safety_report": {
                "risks": report.risks,
                "estimated_rows": report.estimated_rows,
                "required_permissions": report.required_permissions,
            },
            "answer": f"SQL 安全检查未通过: {', '.join(report.risks)}",
            "is_sql": False,
        }

    return {"safety_report": None}


def approve(state: SQLReactState) -> dict:
    """人工审批 SQL。使用 interrupt 暂停图执行，等待用户确认。"""
    result = interrupt({
        "sql": state["sql"],
        "message": "请确认是否执行以上 SQL？",
    })

    if result.get("approved"):
        return {"approved": True}

    return {
        "approved": False,
        "answer": result.get("feedback", "SQL 已被拒绝。"),
        "is_sql": False,
    }


async def execute_sql(state: SQLReactState) -> dict:
    """通过 MCP 执行 SQL。捕获错误用于自动纠错。"""
    from agents.tool.sql_tools.mcp_client import execute_sql as mcp_execute

    sql = state["sql"]
    execution_history = list(state.get("execution_history", []))

    try:
        result = await mcp_execute(sql)
        execution_history.append({"sql": sql, "result": result, "error": None})
        return {
            "result": result,
            "answer": result,
            "error": None,
            "execution_history": execution_history,
        }
    except Exception as e:
        error_msg = str(e)
        logger.warning("SQL execution failed (retry %d/%d): %s", state.get("retry_count", 0), settings.resilience.max_sql_retries, error_msg)
        execution_history.append({"sql": sql, "result": None, "error": error_msg})
        return {
            "result": f"SQL 执行失败: {error_msg}",
            "answer": f"SQL 执行失败: {error_msg}",
            "error": error_msg,
            "execution_history": execution_history,
        }


async def error_analysis(state: SQLReactState) -> dict:
    """分析 SQL 执行错误，生成修正建议。"""
    model = get_chat_model(settings.chat_model_type)

    docs_text = "\n".join([d.page_content for d in state.get("docs", [])])
    last_error = state.get("error", "")
    last_sql = state.get("sql", "")
    retry_count = state.get("retry_count", 0)

    response = await model.ainvoke([
        SystemMessage(content=f"""你是一个 SQL 调试专家。以下 SQL 执行失败，请分析错误原因并给出修正建议。

表结构信息:
{docs_text}

失败的 SQL:
{last_sql}

错误信息:
{last_error}

请简要分析错误原因（1-2 句话），并给出修正建议。"""),
        HumanMessage(content=f"这是第 {retry_count} 次重试，请分析错误并给出修正建议。"),
    ])

    return {
        "refine_feedback": response.content.strip(),
        "retry_count": retry_count + 1,
    }


def build_sql_react_graph():
    """构建 SQL React 图，支持自动纠错重试。

    流程: contextualize_query → recall_evidence → query_enhance → select_tables → sql_retrieve → check_docs → generate → ...
    """
    graph = StateGraph(SQLReactState)

    graph.add_node("contextualize_query", contextualize_query)
    graph.add_node("recall_evidence", recall_evidence)
    graph.add_node("query_enhance", query_enhance)
    graph.add_node("select_tables", select_tables)
    graph.add_node("sql_retrieve", sql_retrieve)
    graph.add_node("check_docs", check_docs)
    graph.add_node("sql_generate", sql_generate)
    graph.add_node("safety_check", safety_check)
    graph.add_node("approve", approve)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("error_analysis", error_analysis)

    graph.add_edge(START, "contextualize_query")
    graph.add_edge("contextualize_query", "recall_evidence")
    graph.add_edge("recall_evidence", "query_enhance")
    graph.add_edge("query_enhance", "select_tables")
    graph.add_edge("select_tables", "sql_retrieve")
    graph.add_edge("sql_retrieve", "check_docs")

    def route_after_check(state: SQLReactState) -> str:
        if state.get("is_sql") is False and state.get("answer"):
            return END
        return "sql_generate"

    graph.add_conditional_edges("check_docs", route_after_check)
    graph.add_edge("sql_generate", "safety_check")

    def route_after_safety(state: SQLReactState) -> str:
        if state.get("is_sql"):
            return "approve"
        return END

    graph.add_conditional_edges("safety_check", route_after_safety)

    def route_after_approve(state: SQLReactState) -> str:
        if state.get("approved"):
            return "execute_sql"
        return END

    graph.add_conditional_edges("approve", route_after_approve)

    def route_after_execute(state: SQLReactState) -> str:
        # 执行成功
        if not state.get("error"):
            return END
        # 错误不可重试（语法/权限/表不存在），直接结束
        if not is_retryable(state["error"]):
            logger.info("route_after_execute: non-retryable error, ending: %s", state["error"][:200])
            return END
        # 可重试错误：检查次数
        max_retries = settings.resilience.max_sql_retries
        if state.get("retry_count", 0) < max_retries:
            return "error_analysis"
        # 超过最大重试次数
        logger.warning("route_after_execute: max retries (%d) reached", max_retries)
        return END

    graph.add_conditional_edges("execute_sql", route_after_execute)
    graph.add_edge("error_analysis", "sql_generate")

    checkpointer = get_checkpointer()
    return graph.compile(checkpointer=checkpointer)
