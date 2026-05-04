"""Tests for Final Graph API endpoints: invoke, approve, interrupt handling."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.api.routers.final import _extract_interrupt


# ---------------------------------------------------------------------------
# _extract_interrupt helper
# ---------------------------------------------------------------------------

class TestExtractInterrupt:
    """Test interrupt extraction from graph result."""

    def test_extract_from_dict_value(self):
        """Should extract dict from interrupt value."""
        interrupt = MagicMock()
        interrupt.value = {"sql": "SELECT 1", "message": "confirm?"}
        result = _extract_interrupt({"__interrupt__": [interrupt]})

        assert result is not None
        assert result["sql"] == "SELECT 1"
        assert result["message"] == "confirm?"

    def test_extract_from_list_value(self):
        """Should extract first item when value is a list."""
        interrupt = MagicMock()
        interrupt.value = [{"sql": "SELECT 1", "message": "confirm?"}]
        result = _extract_interrupt({"__interrupt__": [interrupt]})

        assert result is not None
        assert result["sql"] == "SELECT 1"

    def test_no_interrupt_returns_none(self):
        """Should return None when no interrupt."""
        result = _extract_interrupt({"answer": "done"})
        assert result is None

    def test_empty_interrupt_returns_none(self):
        """Should return None when interrupt list is empty."""
        result = _extract_interrupt({"__interrupt__": []})
        assert result is None

    def test_non_dict_value_returns_none(self):
        """Should return None when value is not a dict."""
        interrupt = MagicMock()
        interrupt.value = "just a string"
        result = _extract_interrupt({"__interrupt__": [interrupt]})
        assert result is None


# ---------------------------------------------------------------------------
# /api/final/invoke
# ---------------------------------------------------------------------------

class TestFinalInvoke:
    """Test POST /api/final/invoke."""

    @pytest.mark.asyncio
    @patch("agents.api.routers.final.get_trace_callbacks", return_value=[])
    @patch("agents.api.routers.final.build_final_graph")
    async def test_invoke_returns_answer(self, mock_build_graph, mock_callbacks):
        """Normal completion should return answer."""
        from agents.api.routers.final import final_invoke, FinalRequest

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "answer": "42 rows found",
            "status": "completed",
        })
        mock_build_graph.return_value = mock_graph

        req = FinalRequest(query="count users", session_id="s1")
        result = await final_invoke(req)

        assert result.answer == "42 rows found"
        assert result.status == "completed"
        assert result.pending_approval is False

    @pytest.mark.asyncio
    @patch("agents.api.routers.final.get_trace_callbacks", return_value=[])
    @patch("agents.api.routers.final.build_final_graph")
    async def test_invoke_returns_pending_approval(self, mock_build_graph, mock_callbacks):
        """When graph returns interrupt, should return pending_approval."""
        from agents.api.routers.final import final_invoke, FinalRequest

        interrupt_obj = MagicMock()
        interrupt_obj.value = {"sql": "SELECT * FROM users", "message": "请确认?"}

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "__interrupt__": [interrupt_obj],
        })
        mock_build_graph.return_value = mock_graph

        req = FinalRequest(query="查询用户", session_id="s1")
        result = await final_invoke(req)

        assert result.pending_approval is True
        assert result.status == "pending_approval"
        assert result.sql == "SELECT * FROM users"
        assert "确认" in result.answer

    @pytest.mark.asyncio
    @patch("agents.api.routers.final.get_trace_callbacks", return_value=[])
    @patch("agents.api.routers.final.build_final_graph")
    async def test_invoke_passes_thread_id(self, mock_build_graph, mock_callbacks):
        """Should pass thread_id in config."""
        from agents.api.routers.final import final_invoke, FinalRequest

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={"answer": "ok", "status": "completed"})
        mock_build_graph.return_value = mock_graph

        req = FinalRequest(query="test", session_id="my-session")
        await final_invoke(req)

        call_kwargs = mock_graph.ainvoke.call_args
        config = call_kwargs[1].get("config") or call_kwargs[0][1] if len(call_kwargs[0]) > 1 else call_kwargs[1].get("config")
        # Config should have thread_id
        assert config["configurable"]["thread_id"] == "my-session"


# ---------------------------------------------------------------------------
# /api/final/approve
# ---------------------------------------------------------------------------

class TestFinalApprove:
    """Test POST /api/final/approve."""

    @pytest.mark.asyncio
    @patch("agents.api.routers.final.get_trace_callbacks", return_value=[])
    @patch("agents.api.routers.final.build_final_graph")
    async def test_approve_returns_result(self, mock_build_graph, mock_callbacks):
        """Approved SQL should return execution result."""
        from agents.api.routers.final import approve_sql, ApproveRequest

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "answer": '[{"id": 1}]',
            "status": "completed",
        })
        mock_build_graph.return_value = mock_graph

        req = ApproveRequest(session_id="s1", approved=True)
        result = await approve_sql(req)

        assert result.answer == '[{"id": 1}]'
        assert result.status == "completed"

    @pytest.mark.asyncio
    @patch("agents.api.routers.final.get_trace_callbacks", return_value=[])
    @patch("agents.api.routers.final.build_final_graph")
    async def test_approve_sends_command_resume(self, mock_build_graph, mock_callbacks):
        """Should send Command(resume=...) to graph."""
        from agents.api.routers.final import approve_sql, ApproveRequest
        from langgraph.types import Command

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={"answer": "ok", "status": "completed"})
        mock_build_graph.return_value = mock_graph

        req = ApproveRequest(session_id="s1", approved=True, feedback="looks good")
        await approve_sql(req)

        call_args = mock_graph.ainvoke.call_args
        cmd = call_args[0][0]
        assert isinstance(cmd, Command)
        assert cmd.resume["approved"] is True
        assert cmd.resume["feedback"] == "looks good"

    @pytest.mark.asyncio
    @patch("agents.api.routers.final.get_trace_callbacks", return_value=[])
    @patch("agents.api.routers.final.build_final_graph")
    async def test_approve_reject_still_returns(self, mock_build_graph, mock_callbacks):
        """Rejection should still return a result."""
        from agents.api.routers.final import approve_sql, ApproveRequest

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "answer": "SQL 已被拒绝。",
            "status": "completed",
        })
        mock_build_graph.return_value = mock_graph

        req = ApproveRequest(session_id="s1", approved=False, feedback="too dangerous")
        result = await approve_sql(req)

        assert "拒绝" in result.answer


# ---------------------------------------------------------------------------
# _make_config
# ---------------------------------------------------------------------------

class TestMakeConfig:
    """Test config construction."""

    @patch("agents.api.routers.final.get_trace_callbacks", return_value=[])
    def test_config_has_thread_id(self, mock_callbacks):
        from agents.api.routers.final import _make_config

        config = _make_config("session-123")

        assert config["configurable"]["thread_id"] == "session-123"

    @patch("agents.api.routers.final.get_trace_callbacks", return_value=["handler1"])
    def test_config_has_callbacks(self, mock_callbacks):
        from agents.api.routers.final import _make_config

        config = _make_config("s1")

        assert "callbacks" in config
        assert config["callbacks"] == ["handler1"]

    @patch("agents.api.routers.final.get_trace_callbacks", return_value=[])
    def test_config_no_empty_callbacks(self, mock_callbacks):
        from agents.api.routers.final import _make_config

        config = _make_config("s1")

        assert "callbacks" not in config
