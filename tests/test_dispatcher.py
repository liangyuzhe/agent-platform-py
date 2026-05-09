"""Tests for the final dispatcher graph."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import MemorySaver


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
