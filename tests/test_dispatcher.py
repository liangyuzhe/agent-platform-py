"""Tests for the final dispatcher graph."""

from __future__ import annotations

from types import SimpleNamespace
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


def test_arbitration_uses_llm_intent_when_no_rule_signal():
    from agents.flow.dispatcher import _arbitrate_intent

    intent = _arbitrate_intent("chat", None)

    assert intent == "chat"


def test_arbitration_prefers_database_rule_signal():
    from agents.flow.dispatcher import _arbitrate_intent

    rule = SimpleNamespace(intent="sql_query")
    intent = _arbitrate_intent("chat", rule)

    assert intent == "sql_query"


@pytest.mark.asyncio
@patch("agents.flow.dispatcher.evaluate_intent_rules", new_callable=AsyncMock)
@patch("agents.flow.dispatcher.get_domain_summary", return_value="企业财务核算，可回答财务指标计算")
@patch("agents.flow.dispatcher.get_chat_model")
async def test_classify_intent_keeps_llm_chat_without_rule_override(mock_get_model, mock_domain, mock_rules):
    """Public-company questions should stay chat when LLM classifies them as chat."""
    from agents.flow.dispatcher import classify_intent

    mock_rules.return_value = None
    mock_model = mock_get_model.return_value
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "chat", "rewritten_query": "茅台第一季度盈利"}'
    ))

    result = await classify_intent({"query": "茅台第一季度盈利", "chat_history": []})

    assert result["intent"] == "chat"
    assert result["rewritten_query"] == "茅台第一季度盈利"


@pytest.mark.asyncio
@patch("agents.flow.dispatcher.evaluate_intent_rules", new_callable=AsyncMock)
@patch("agents.flow.dispatcher.get_domain_summary", return_value="企业财务核算，可回答财务指标计算")
@patch("agents.flow.dispatcher.get_chat_model")
async def test_classify_intent_allows_database_rule_to_override_llm(mock_get_model, mock_domain, mock_rules):
    """Rules are data-driven overrides, not keyword lists embedded in dispatcher code."""
    from agents.flow.dispatcher import classify_intent

    mock_rules.return_value = SimpleNamespace(
        intent="sql_query",
        to_dict=lambda: {"intent": "sql_query", "rule_id": 1},
    )
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


@pytest.mark.asyncio
@patch("agents.flow.dispatcher.evaluate_intent_rules", new_callable=AsyncMock)
@patch("agents.flow.dispatcher.get_domain_summary", return_value="企业财务核算，可回答财务指标计算")
@patch("agents.flow.dispatcher.get_chat_model")
async def test_classify_intent_applies_rule_rewrite_template_for_omitted_subject(
    mock_get_model,
    mock_domain,
    mock_rules,
):
    """A DB rule can normalize omitted-subject finance questions without code keywords."""
    from agents.flow.dispatcher import classify_intent

    mock_rules.return_value = SimpleNamespace(
        intent="sql_query",
        rewrite_template="公司{query}",
        to_dict=lambda: {"intent": "sql_query", "rule_id": 2, "rewrite_template": "公司{query}"},
    )
    mock_model = mock_get_model.return_value
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "chat", "rewritten_query": "第一季度毛利率"}'
    ))

    result = await classify_intent({"query": "第一季度毛利率", "chat_history": []})

    assert result["intent"] == "sql_query"
    assert result["rewritten_query"] == "公司第一季度毛利率"


@pytest.mark.asyncio
@patch("agents.flow.dispatcher.evaluate_intent_rules", new_callable=AsyncMock)
@patch("agents.flow.dispatcher.get_domain_summary", return_value="企业财务核算，可回答财务指标计算")
@patch("agents.flow.dispatcher.get_chat_model")
async def test_classify_intent_does_not_skip_when_only_intent_is_provided(
    mock_get_model,
    mock_domain,
    mock_rules,
):
    """Old clients without rewritten_query should still run current classification."""
    from agents.flow.dispatcher import classify_intent

    mock_rules.return_value = None
    mock_model = mock_get_model.return_value
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "chat", "rewritten_query": "茅台第一季度盈利"}'
    ))

    result = await classify_intent({
        "query": "茅台第一季度盈利",
        "intent": "sql_query",
        "chat_history": [],
    })

    assert result["intent"] == "chat"
    assert result["rewritten_query"] == "茅台第一季度盈利"
    mock_model.ainvoke.assert_awaited_once()
