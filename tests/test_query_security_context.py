import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
@patch("agents.api.routers.query.get_trace_callbacks", return_value=[])
@patch("agents.api.routers.query.build_final_graph")
async def test_query_invoke_passes_default_security_context(mock_build_graph, mock_callbacks):
    from agents.api.routers.query import QueryRequest, query_invoke

    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value={"answer": "ok", "status": "completed"})
    mock_build_graph.return_value = mock_graph

    await query_invoke(QueryRequest(query="查询用户", session_id="s1"))

    initial_state = mock_graph.ainvoke.call_args[0][0]
    assert initial_state["security_context"]["user_id"] == "s1"
    assert initial_state["security_context"]["username"] == "s1"
    assert initial_state["security_context"]["role_ids"] == []


def test_build_security_context_from_headers():
    from agents.api.routers.query import _build_security_context

    context = _build_security_context(
        session_id="fallback",
        headers={
            "x-user-id": "u-1",
            "x-user-name": "Alice",
            "x-role-ids": "finance,auditor",
            "x-allowed-tables": "t_user,t_role",
            "x-denied-tables": "t_salary",
        },
    )

    assert context["user_id"] == "u-1"
    assert context["username"] == "Alice"
    assert context["role_ids"] == ["finance", "auditor"]
    assert context["allowed_tables"] == ["t_user", "t_role"]
    assert context["denied_tables"] == ["t_salary"]
