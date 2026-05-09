"""Tests for the final dispatcher graph."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage


class _RecordingSqlGraph:
    def __init__(self):
        self.states = []

    async def ainvoke(self, state, config=None):
        self.states.append(dict(state))
        return {
            "sql": "SELECT 1;",
            "result": [{"x": 1}],
            "answer": state.get("rewritten_query", ""),
        }


@pytest.mark.asyncio
async def test_dispatcher_uses_current_query_and_rewrite_with_same_thread():
    """New turns in the same session must not reuse old checkpoint query."""
    from agents.flow.dispatcher import build_final_graph

    fake_sql_graph = _RecordingSqlGraph()

    with (
        patch("agents.flow.dispatcher.get_checkpointer", return_value=MemorySaver()),
        patch("agents.flow.sql_react.build_sql_react_graph", return_value=fake_sql_graph),
    ):
        app = build_final_graph()
        config = {"configurable": {"thread_id": "same-web-session"}}

        await app.ainvoke(
            {
                "query": "我们公司去年亏损",
                "intent": "sql_query",
                "rewritten_query": "我们公司去年亏损",
                "session_id": "same-web-session",
                "chat_history": [],
            },
            config=config,
        )
        await app.ainvoke(
            {
                "query": "第一季度员工工资",
                "intent": "sql_query",
                "rewritten_query": "我们公司第一季度的员工工资情况",
                "session_id": "same-web-session",
                "chat_history": [],
            },
            config=config,
        )

    assert fake_sql_graph.states[0]["query"] == "我们公司去年亏损"
    assert fake_sql_graph.states[1]["query"] == "第一季度员工工资"
    assert fake_sql_graph.states[1]["rewritten_query"] == "我们公司第一季度的员工工资情况"


def test_company_finance_query_guard_forces_sql_intent():
    """Company finance data queries should not be downgraded to chat."""
    from agents.flow.dispatcher import _normalize_intent

    intent = _normalize_intent("chat", "去年亏损", "我们公司去年是否亏损")

    assert intent == "sql_query"


def test_public_company_query_guard_does_not_force_sql_intent():
    """External public-company questions can remain chat/knowledge."""
    from agents.flow.dispatcher import _normalize_intent

    intent = _normalize_intent("chat", "去年贵州茅台的亏损情况", "去年贵州茅台的亏损情况")

    assert intent == "chat"


@pytest.mark.asyncio
@patch("agents.flow.dispatcher.get_domain_summary", return_value="企业财务核算，可回答财务指标计算")
@patch("agents.flow.dispatcher.get_chat_model")
async def test_classify_intent_overrides_llm_chat_for_company_loss(mock_get_model, mock_domain):
    """Regression for LLM returning chat for '去年亏损' due to history bias."""
    from agents.flow.dispatcher import classify_intent

    mock_model = mock_get_model.return_value
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "chat", "rewritten_query": "我们公司去年是否亏损"}'
    ))

    result = await classify_intent({
        "query": "去年亏损",
        "chat_history": [
            {"role": "assistant", "content": "参考知识中未提供您公司去年亏损的相关信息。"},
            {"role": "user", "content": "第一季度员工工资"},
        ],
    })

    assert result["intent"] == "sql_query"
    assert result["rewritten_query"] == "我们公司去年是否亏损"
