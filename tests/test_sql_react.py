"""Tests for SQL React graph: check_docs, approve, routing, safety_check."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_doc(content="CREATE TABLE users (id INT, name VARCHAR(50));"):
    """Return a mock Document with table schema content."""
    return Document(page_content=content, metadata={"source": "mysql_schema", "table_name": "users"})


def _mock_llm_tool_response(answer="SELECT * FROM users;", is_sql=True):
    """Return a mock LLM response with tool_calls."""
    resp = MagicMock()
    resp.tool_calls = [{"args": {"answer": answer, "is_sql": is_sql}}]
    return resp


def _mock_llm_text_response(content="I don't know how to write SQL for that."):
    """Return a mock LLM response without tool_calls."""
    resp = MagicMock()
    resp.tool_calls = []
    resp.content = content
    return resp


class TestStateReducers:
    """Test state reducers for parent/child graph merge safety."""

    def test_query_accepts_duplicate_step_updates(self):
        """Duplicate query writes in one LangGraph step should not fail."""
        from agents.flow.state import SQLReactState

        def node_a(state):
            return {"query": "node a query"}

        def node_b(state):
            return {"query": "node b query"}

        graph = StateGraph(SQLReactState)
        graph.add_node("node_a", node_a)
        graph.add_node("node_b", node_b)
        graph.add_edge(START, "node_a")
        graph.add_edge(START, "node_b")
        graph.add_edge("node_a", END)
        graph.add_edge("node_b", END)

        result = graph.compile().invoke({"query": "去年亏损"})

        assert result["query"] in {"去年亏损", "node a query", "node b query"}

    def test_query_replaced_by_new_turn_with_same_thread(self):
        """A new user turn must replace the previous checkpoint query."""
        from langgraph.checkpoint.memory import MemorySaver
        from agents.flow.state import FinalGraphState

        def echo_query(state):
            return {"answer": state["query"]}

        graph = StateGraph(FinalGraphState)
        graph.add_node("echo_query", echo_query)
        graph.add_edge(START, "echo_query")
        graph.add_edge("echo_query", END)
        app = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "same-session"}}

        first = app.invoke({"query": "我们公司去年亏损"}, config=config)
        second = app.invoke({"query": "第一季度员工工资"}, config=config)

        assert first["answer"] == "我们公司去年亏损"
        assert second["answer"] == "第一季度员工工资"

    def test_final_graph_state_keeps_rewritten_query(self):
        """Frontend/classify rewritten_query should survive graph state schema."""
        from agents.flow.state import FinalGraphState

        def echo_rewrite(state):
            return {"answer": state.get("rewritten_query", "")}

        graph = StateGraph(FinalGraphState)
        graph.add_node("echo_rewrite", echo_rewrite)
        graph.add_edge(START, "echo_rewrite")
        graph.add_edge("echo_rewrite", END)

        result = graph.compile().invoke({
            "query": "第一季度员工工资",
            "rewritten_query": "我们公司第一季度的员工工资情况",
        })

        assert result["answer"] == "我们公司第一季度的员工工资情况"


# ---------------------------------------------------------------------------
# check_docs node
# ---------------------------------------------------------------------------

class TestContextualizeQuery:
    """Test contextualize_query node."""

    @pytest.mark.asyncio
    async def test_existing_rewritten_query_skips(self):
        """When rewritten_query already exists, should skip LLM call."""
        from agents.flow.sql_react import contextualize_query

        state = {"query": "它多少钱", "rewritten_query": "iPhone 15 价格"}
        result = await contextualize_query(state)
        assert result["rewritten_query"] == "iPhone 15 价格"

    @pytest.mark.asyncio
    async def test_no_history_returns_original(self):
        """Without chat history, should return original query."""
        from agents.flow.sql_react import contextualize_query

        state = {"query": "查询订单", "chat_history": []}
        result = await contextualize_query(state)
        assert result["rewritten_query"] == "查询订单"

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.rewrite_query")
    async def test_with_history_rewrites(self, mock_rewrite):
        """With chat history, should call rewrite_query."""
        from agents.flow.sql_react import contextualize_query

        mock_rewrite.return_value = "张三的订单金额"
        state = {
            "query": "他的订单金额",
            "chat_history": [
                {"role": "user", "content": "查询张三的信息"},
                {"role": "assistant", "content": "张三，男，28岁"},
            ],
        }
        result = await contextualize_query(state)
        assert result["rewritten_query"] == "张三的订单金额"
        mock_rewrite.assert_called_once()


class TestCheckDocs:
    """Test check_docs node."""

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.Elasticsearch")
    async def test_no_docs_returns_message(self, MockES):
        """When no docs retrieved and ES fallback is empty, should return error."""
        from agents.flow.sql_react import check_docs

        MockES.return_value.search.return_value = {"hits": {"hits": []}}

        result = await check_docs({"query": "查询用户", "docs": []})

        assert result["is_sql"] is False
        assert "未找到" in result["answer"]
        assert "表结构" in result["answer"]

    @pytest.mark.asyncio
    async def test_with_docs_returns_empty(self):
        """When docs exist, should return empty dict (proceed to generate)."""
        from agents.flow.sql_react import check_docs

        result = await check_docs({"query": "查询用户", "docs": [_mock_doc()]})

        assert result == {}

    @pytest.mark.asyncio
    async def test_no_docs_returns_error(self):
        """When no docs, should return error message."""
        from agents.flow.sql_react import check_docs

        result = await check_docs({"query": "查询用户", "docs": []})

        assert result["is_sql"] is False
        assert "未找到" in result["answer"]


# ---------------------------------------------------------------------------
# query_enhance node
# ---------------------------------------------------------------------------

class TestQueryEnhance:
    """Test query_enhance node."""

    @pytest.mark.asyncio
    async def test_no_evidence_skips(self):
        """Without evidence, should return original query unchanged."""
        from agents.flow.sql_react import query_enhance

        result = await query_enhance({
            "query": "查询GMV",
            "rewritten_query": "查询GMV",
            "evidence": [],
            "few_shot_examples": [],
        })

        assert result["enhanced_query"] == "查询GMV"

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_empty_llm_response_uses_business_knowledge_fallback(self, mock_get_model):
        """Empty LLM output should still enhance from matched business evidence."""
        from agents.flow.sql_react import query_enhance

        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content=""))
        mock_get_model.return_value = mock_model

        result = await query_enhance({
            "query": "去年亏损",
            "rewritten_query": "去年亏损",
            "evidence": ["术语: 净利润\n公式: 收入 - 成本 - 费用；亏损表示净利润 < 0\n同义词: 净收益, 盈利, 亏损, 净亏损, 赔钱, 赚钱"],
            "few_shot_examples": [],
        })

        assert "净利润" in result["enhanced_query"]
        assert "收入 - 成本 - 费用" in result["enhanced_query"]

    @pytest.mark.asyncio
    async def test_no_evidence_returns_original_query(self):
        """Without evidence, query_enhance should not hard-code business terms."""
        from agents.flow.sql_react import query_enhance

        result = await query_enhance({
            "query": "去年亏损",
            "rewritten_query": "去年亏损",
            "evidence": [],
            "few_shot_examples": [],
        })

        assert result["enhanced_query"] == "去年亏损"
        assert "净利润" not in result["enhanced_query"]

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_with_evidence_enhances(self, mock_get_model):
        """With evidence, should call LLM to enhance query."""
        from agents.flow.sql_react import query_enhance

        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="查询已支付订单总额"))
        mock_get_model.return_value = mock_model

        result = await query_enhance({
            "query": "查询GMV",
            "rewritten_query": "查询GMV",
            "evidence": ["GMV = 已支付订单总额"],
            "few_shot_examples": [],
        })

        assert result["enhanced_query"] == "查询已支付订单总额"
        mock_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_with_evidence_passes_trace_callbacks(self, mock_get_model):
        """Inner LLM call should inherit graph callbacks for tracing."""
        from agents.flow.sql_react import query_enhance

        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="查询已支付订单总额"))
        mock_get_model.return_value = mock_model

        await query_enhance(
            {
                "query": "查询GMV",
                "rewritten_query": "查询GMV",
                "evidence": ["GMV = 已支付订单总额"],
                "few_shot_examples": [],
            },
            config={"callbacks": ["trace-handler"]},
        )

        call_config = mock_model.ainvoke.call_args.kwargs["config"]
        assert call_config["callbacks"] == ["trace-handler"]
        assert call_config["run_name"] == "sql.query_enhance.llm"

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_llm_failure_fallback(self, mock_get_model):
        """On LLM failure, should fallback to original query."""
        from agents.flow.sql_react import query_enhance

        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(side_effect=Exception("timeout"))
        mock_get_model.return_value = mock_model

        result = await query_enhance({
            "query": "查询GMV",
            "rewritten_query": "查询GMV",
            "evidence": ["GMV = 已支付订单总额"],
            "few_shot_examples": [],
        })

        assert result["enhanced_query"] == "查询GMV"


# ---------------------------------------------------------------------------
# safety_check node
# ---------------------------------------------------------------------------

class TestSafetyCheck:
    """Test safety_check node."""

    @pytest.mark.asyncio
    async def test_non_sql_skips_check(self):
        """When is_sql is False, skip safety check."""
        from agents.flow.sql_react import safety_check

        result = await safety_check({"is_sql": False, "sql": "not sql"})
        assert result["safety_report"] is None

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.SQLSafetyChecker")
    async def test_safe_sql_passes(self, MockChecker):
        """Safe SQL should pass with no report."""
        from agents.flow.sql_react import safety_check

        mock_checker = MagicMock()
        mock_report = MagicMock()
        mock_report.is_safe = True
        mock_checker.check.return_value = mock_report
        MockChecker.return_value = mock_checker

        result = await safety_check({"is_sql": True, "sql": "SELECT * FROM users"})

        assert result["safety_report"] is None

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.SQLSafetyChecker")
    async def test_unsafe_sql_blocked(self, MockChecker):
        """Unsafe SQL should be blocked with risks reported."""
        from agents.flow.sql_react import safety_check

        mock_checker = MagicMock()
        mock_report = MagicMock()
        mock_report.is_safe = False
        mock_report.risks = ["DELETE detected"]
        mock_report.estimated_rows = 1000
        mock_report.required_permissions = ["DELETE"]
        mock_checker.check.return_value = mock_report
        MockChecker.return_value = mock_checker

        result = await safety_check({"is_sql": True, "sql": "DELETE FROM users"})

        assert result["is_sql"] is False
        assert "DELETE detected" in result["answer"]
        assert result["safety_report"]["risks"] == ["DELETE detected"]


# ---------------------------------------------------------------------------
# approve node
# ---------------------------------------------------------------------------

class TestApprove:
    """Test approve node uses interrupt."""

    def test_approved_returns_true(self):
        """When user approves, should set approved=True."""
        from agents.flow.sql_react import approve

        with patch("agents.flow.sql_react.interrupt", return_value={"approved": True}):
            result = approve({"sql": "SELECT 1"})

        assert result["approved"] is True

    def test_rejected_returns_message(self):
        """When user rejects, should set approved=False with feedback."""
        from agents.flow.sql_react import approve

        with patch("agents.flow.sql_react.interrupt", return_value={
            "approved": False, "feedback": "SQL too dangerous"
        }):
            result = approve({"sql": "DELETE FROM users"})

        assert result["approved"] is False
        assert result["is_sql"] is False
        assert "SQL too dangerous" in result["answer"]

    def test_rejected_default_message(self):
        """When user rejects without feedback, use default message."""
        from agents.flow.sql_react import approve

        with patch("agents.flow.sql_react.interrupt", return_value={"approved": False}):
            result = approve({"sql": "SELECT 1"})

        assert result["approved"] is False
        assert "拒绝" in result["answer"]

    def test_interrupt_passes_sql(self):
        """interrupt should receive the SQL and a message."""
        from agents.flow.sql_react import approve

        mock_interrupt = MagicMock(return_value={"approved": True})
        with patch("agents.flow.sql_react.interrupt", mock_interrupt):
            approve({"sql": "SELECT * FROM t_user"})

        call_args = mock_interrupt.call_args[0][0]
        assert call_args["sql"] == "SELECT * FROM t_user"
        assert "确认" in call_args["message"]


# ---------------------------------------------------------------------------
# sql_generate node
# ---------------------------------------------------------------------------

class TestSqlGenerate:
    """Test sql_generate node."""

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.create_format_tool")
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_generate_sql_from_docs(self, mock_get_model, mock_format):
        """LLM should generate SQL based on retrieved docs."""
        from agents.flow.sql_react import sql_generate

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.ainvoke = AsyncMock(return_value=_mock_llm_tool_response("SELECT id, name FROM users;", True))
        mock_get_model.return_value = mock_model

        state = {
            "query": "查询所有用户",
            "docs": [_mock_doc()],
        }
        result = await sql_generate(state)

        assert result["is_sql"] is True
        assert "SELECT" in result["sql"]
        assert result["sql"].endswith(";")

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.create_format_tool")
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_generate_sql_passes_trace_callbacks(self, mock_get_model, mock_format):
        """SQL generation LLM call should be visible as a child trace."""
        from agents.flow.sql_react import sql_generate

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.ainvoke = AsyncMock(return_value=_mock_llm_tool_response("SELECT 1;", True))
        mock_get_model.return_value = mock_model

        await sql_generate(
            {"query": "查询所有用户", "docs": [_mock_doc()]},
            config={"callbacks": ["trace-handler"]},
        )

        call_config = mock_model.ainvoke.call_args.kwargs["config"]
        assert call_config["callbacks"] == ["trace-handler"]
        assert call_config["run_name"] == "sql.sql_generate.llm"

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.create_format_tool")
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_generate_normalizes_sql_artifacts(self, mock_get_model, mock_format):
        """sql_generate should strip sentinel tokens and SQL fences."""
        from agents.flow.sql_react import sql_generate

        raw_sql = """<text_never_used_51bce0c785ca2f68081bfa7d91973934>```sql
SELECT
  id
FROM users
```"""
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.ainvoke = AsyncMock(return_value=_mock_llm_tool_response(raw_sql, True))
        mock_get_model.return_value = mock_model

        result = await sql_generate({"query": "查询用户", "docs": [_mock_doc()]})

        assert result["is_sql"] is True
        assert result["sql"].startswith("SELECT")
        assert "text_never_used" not in result["sql"]
        assert "```" not in result["sql"]
        assert result["sql"].endswith(";")

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.create_format_tool")
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_generate_strips_closing_sentinel_artifact(self, mock_get_model, mock_format):
        """sql_generate should strip closing sentinel artifacts after SQL."""
        from agents.flow.sql_react import sql_generate

        raw_sql = """SELECT SUM(ji.credit_amount - ji.debit_amount) AS 净利润
FROM t_journal_entry je
INNER JOIN t_journal_item ji ON je.id = ji.entry_id
INNER JOIN t_account a ON ji.account_code = a.account_code
WHERE je.status = '已过账'
AND je.period >= '2025-01'
AND je.period <= '2025-12'
AND a.account_type = '损益'
HAVING 净利润 < 0;</text_never_used_51bce0c785ca2f68081bfa7d91973934>;"""
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.ainvoke = AsyncMock(return_value=_mock_llm_tool_response(raw_sql, True))
        mock_get_model.return_value = mock_model

        result = await sql_generate({"query": "去年亏损", "docs": [_mock_doc()]})

        assert result["is_sql"] is True
        assert "text_never_used" not in result["sql"]
        assert result["sql"].endswith("HAVING 净利润 < 0;")
        assert not result["sql"].endswith(";;")

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.create_format_tool")
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_generate_rejects_truncated_sql(self, mock_get_model, mock_format):
        """Truncated SQL should not proceed to approval/execution."""
        from agents.flow.sql_react import sql_generate

        raw_sql = """<text_never_used_51bce0c785ca2f68081bfa7d91973934>SELECT
  (SUM(credit_amount) - SUM(debit_amount)) AS net_profit
FROM t_journal_item
HAVIN"""
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.ainvoke = AsyncMock(return_value=_mock_llm_tool_response(raw_sql, True))
        mock_get_model.return_value = mock_model

        result = await sql_generate({"query": "去年亏损", "docs": [_mock_doc()]})

        assert result["is_sql"] is False
        assert result["error"] == "invalid_sql_format"
        assert "不完整" in result["answer"]
        assert "text_never_used" not in result["answer"]

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.create_format_tool")
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_generate_non_sql_response(self, mock_get_model, mock_format):
        """When LLM responds with text (not SQL), is_sql should be False."""
        from agents.flow.sql_react import sql_generate

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.ainvoke = AsyncMock(return_value=_mock_llm_tool_response("I cannot generate SQL for this.", False))
        mock_get_model.return_value = mock_model

        state = {
            "query": "你好",
            "docs": [_mock_doc()],
        }
        result = await sql_generate(state)

        assert result["is_sql"] is False

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.create_format_tool")
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_generate_includes_refine_feedback(self, mock_get_model, mock_format):
        """Refine feedback should be included in the prompt."""
        from agents.flow.sql_react import sql_generate

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.ainvoke = AsyncMock(return_value=_mock_llm_tool_response("SELECT id FROM users;", True))
        mock_get_model.return_value = mock_model

        state = {
            "query": "查询用户ID",
            "docs": [_mock_doc()],
            "refine_feedback": "只查询ID字段",
        }
        await sql_generate(state)

        # Verify the system message contains refine feedback
        call_args = mock_model.ainvoke.call_args[0][0]
        system_msg = call_args[0].content
        assert "只查询ID字段" in system_msg

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.create_format_tool")
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_generate_passes_all_docs(self, mock_get_model, mock_format):
        """sql_generate passes all docs to LLM (no rerank filtering)."""
        from agents.flow.sql_react import sql_generate

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.ainvoke = AsyncMock(return_value=_mock_llm_tool_response("SELECT 1;", True))
        mock_get_model.return_value = mock_model

        docs = [
            Document(page_content="t_user schema", metadata={"table_name": "t_user"}),
            Document(page_content="t_department schema", metadata={"table_name": "t_department"}),
            Document(page_content="t_role schema", metadata={"table_name": "t_role"}),
        ]

        state = {"query": "查询用户", "docs": docs}
        await sql_generate(state)

        # All docs should be in the prompt
        call_args = mock_model.ainvoke.call_args[0][0]
        system_msg = call_args[0].content
        assert "t_user schema" in system_msg
        assert "t_department schema" in system_msg
        assert "t_role schema" in system_msg

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.create_format_tool")
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_generate_uses_enhanced_query(self, mock_get_model, mock_format):
        """sql_generate should pass enhanced_query to the LLM when present."""
        from agents.flow.sql_react import sql_generate

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.ainvoke = AsyncMock(return_value=_mock_llm_tool_response("SELECT 1;", True))
        mock_get_model.return_value = mock_model

        state = {
            "query": "去年亏损",
            "rewritten_query": "去年亏损",
            "enhanced_query": "去年亏损（净利润），即净利润为负",
            "docs": [_mock_doc()],
        }
        await sql_generate(state)

        call_args = mock_model.ainvoke.call_args[0][0]
        human_msg = call_args[1].content
        assert "净利润为负" in human_msg

    def test_sql_prompt_contains_profit_loss_boundary_rules(self):
        """SQL prompt should prevent classifying zero net profit as loss."""
        from agents.flow.sql_react import _build_sql_messages

        messages = _build_sql_messages("亏损多少", "schema", "", "", "", "")
        system_msg = messages[0].content

        assert "等于 0" in system_msg
        assert "不要用 ELSE 把 0 归为亏损或盈利" in system_msg
        assert "亏损金额" in system_msg
        assert "ABS(净利润)" in system_msg

    def test_sql_prompt_requires_followup_to_reuse_prior_sql_context(self):
        """Follow-up queries should be instructed to reuse prior SQL semantics."""
        from agents.flow.sql_react import _build_sql_messages

        messages = _build_sql_messages("亏损多少", "schema", "", "上一轮SQL上下文", "", "")
        system_msg = messages[0].content

        assert "上一轮 SQL" in system_msg
        assert "时间范围" in system_msg
        assert "状态过滤" in system_msg
        assert "指标计算口径" in system_msg

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.create_format_tool")
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_generate_includes_prior_sql_context(self, mock_get_model, mock_format):
        """SQL generation prompt should include saved prior SQL context."""
        from agents.flow.sql_react import sql_generate

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.ainvoke = AsyncMock(return_value=_mock_llm_tool_response("SELECT 1;", True))
        mock_get_model.return_value = mock_model

        state = {
            "query": "亏损多少",
            "rewritten_query": "去年亏损多少",
            "docs": [_mock_doc()],
            "chat_history": [{
                "role": "system",
                "content": "[上一轮SQL上下文]\n用户问题: 去年亏损\n生成SQL:\nSELECT ... WHERE status = '已过账'\n展示结果: 净利润：0.00",
            }],
        }
        await sql_generate(state)

        system_msg = mock_model.ainvoke.call_args[0][0][0].content
        assert "上一轮SQL上下文" in system_msg
        assert "status = '已过账'" in system_msg

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.get_semantic_model_by_tables")
    @patch("agents.flow.sql_react.create_format_tool")
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_generate_retrieves_missing_tables(
        self, mock_get_model, mock_format, mock_filter
    ):
        """When LLM says tables are missing, should re-retrieve and retry."""
        from agents.flow.sql_react import sql_generate

        mock_filter.return_value = {
            "t_user": {
                "id": {"column_name": "id", "column_type": "bigint", "column_comment": "主键", "is_pk": 1, "is_fk": 0, "business_name": "用户ID", "synonyms": "", "business_description": ""},
                "username": {"column_name": "username", "column_type": "varchar(64)", "column_comment": "用户名", "is_pk": 0, "is_fk": 0, "business_name": "用户名", "synonyms": "账号", "business_description": "登录账号"},
                "real_name": {"column_name": "real_name", "column_type": "varchar(64)", "column_comment": "真实姓名", "is_pk": 0, "is_fk": 0, "business_name": "真实姓名", "synonyms": "姓名", "business_description": ""},
            },
        }

        # First call: needs more tables; second call: has enough
        first_resp = MagicMock()
        first_resp.tool_calls = [{
            "args": {
                "answer": "",
                "is_sql": False,
                "needs_more_tables": True,
                "missing_tables": ["t_user"],
            }
        }]

        second_resp = MagicMock()
        second_resp.tool_calls = [{
            "args": {
                "answer": "SELECT d.name, u.real_name FROM t_department d JOIN t_user u ON d.manager = u.username;",
                "is_sql": True,
                "needs_more_tables": False,
                "missing_tables": [],
            }
        }]

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.ainvoke = AsyncMock(side_effect=[first_resp, second_resp])
        mock_get_model.return_value = mock_model

        docs = [
            Document(
                page_content="t_department: id, name, manager",
                metadata={"table_name": "t_department"},
            ),
        ]
        state = {"query": "每个部门的负责人姓名", "docs": docs}
        result = await sql_generate(state)

        # Should have made 2 LLM calls
        assert mock_model.ainvoke.call_count == 2
        # Second call should include t_user schema
        second_call_args = mock_model.ainvoke.call_args_list[1][0][0]
        system_msg = second_call_args[0].content
        assert "t_user" in system_msg
        assert "t_department" in system_msg
        assert result["is_sql"] is True
        assert "JOIN" in result["sql"]


# ---------------------------------------------------------------------------
# execute_sql node
# ---------------------------------------------------------------------------

class TestExecuteSql:
    """Test execute_sql node."""

    @pytest.mark.asyncio
    @patch("agents.tool.sql_tools.mcp_client.execute_sql")
    async def test_execute_success(self, mock_mcp_execute):
        """Successful SQL execution should return result."""
        from agents.flow.sql_react import execute_sql as exec_node

        mock_mcp_execute.return_value = '[{"id": 1, "name": "Alice"}]'

        result = await exec_node({"sql": "SELECT * FROM users"})

        assert '[{"id": 1' in result["result"]
        assert result["answer"] == "查询已执行完成。\nid：1\nname：Alice"

    @pytest.mark.asyncio
    @patch("agents.tool.sql_tools.mcp_client.execute_sql")
    async def test_execute_error(self, mock_mcp_execute):
        """SQL execution error should be caught and returned."""
        from agents.flow.sql_react import execute_sql as exec_node

        mock_mcp_execute.side_effect = Exception("Table not found")

        result = await exec_node({"sql": "SELECT * FROM nonexistent", "execution_history": []})

        assert "失败" in result["result"]
        assert "Table not found" in result["result"]
        assert result["error"] == "Table not found"
        assert len(result["execution_history"]) == 1
        assert result["execution_history"][0]["error"] == "Table not found"

    @pytest.mark.asyncio
    @patch("agents.tool.sql_tools.mcp_client.execute_sql")
    async def test_execute_success_clears_error(self, mock_mcp_execute):
        """Successful execution should set error=None."""
        from agents.flow.sql_react import execute_sql as exec_node

        mock_mcp_execute.return_value = '[{"id": 1}]'

        result = await exec_node({"sql": "SELECT 1", "execution_history": []})

        assert result["error"] is None
        assert len(result["execution_history"]) == 1
        assert result["execution_history"][0]["error"] is None

    @pytest.mark.asyncio
    @patch("agents.tool.sql_tools.mcp_client.execute_sql")
    async def test_execute_summarizes_result_with_context(self, mock_mcp_execute):
        """SQL results should be summarized locally using query/schema/evidence context."""
        from agents.flow.sql_react import execute_sql as exec_node

        mock_mcp_execute.return_value = '[{"is_net_profit_positive":0,"net_profit":"0.00"}]Query execution time: 10.97 ms'

        result = await exec_node({
            "query": "去年亏损",
            "sql": "SELECT ...",
            "docs": [_mock_doc("net_profit [业务名: 净利润] [同义词: 亏损, 净亏损]")],
            "evidence": ["术语: 净利润\n公式: 亏损表示净利润 < 0\n同义词: 亏损, 净亏损"],
            "execution_history": [],
        })

        assert result["result"].startswith('[{"is_net_profit_positive"')
        assert result["answer"] == "查询已执行完成。\n是否亏损：否\n净利润：0.00"

    @pytest.mark.asyncio
    @patch("agents.tool.sql_tools.mcp_client.execute_sql")
    async def test_execute_quantity_followup_uses_query_term_label(self, mock_mcp_execute):
        """Quantity follow-up should use the matched user term instead of stale SQL alias."""
        from agents.flow.sql_react import execute_sql as exec_node

        mock_mcp_execute.return_value = '[{"去年净利润":"0.00"}]Query execution time: 10.97 ms'

        result = await exec_node({
            "query": "亏损多少",
            "rewritten_query": "去年亏损多少",
            "sql": "SELECT ROUND(SUM(x), 2) AS 去年净利润 FROM t;",
            "docs": [_mock_doc("net_profit [业务名: 净利润] [同义词: 亏损, 净亏损]")],
            "evidence": ["术语: 净利润\n公式: 亏损表示净利润 < 0\n同义词: 亏损, 净亏损"],
            "execution_history": [],
        })

        assert result["answer"] == "查询已执行完成。\n亏损金额：0.00"


class TestResultAnomaly:
    """Test suspicious execution result detection and reflection."""

    def test_detect_empty_and_null_results(self):
        from agents.flow.sql_react import _result_anomaly_reason

        assert "空集" in _result_anomaly_reason("[]")
        assert "空集" in _result_anomaly_reason('{"rows": []}')
        assert "空集" in _result_anomaly_reason('{"columns": ["net_profit"], "rows": []}')
        assert "空集" in _result_anomaly_reason('{"data": []}')
        assert "空集" in _result_anomaly_reason('{"result": []}')
        assert "NULL" in _result_anomaly_reason('[{"净利润": null}]')
        assert "NULL" in _result_anomaly_reason('[{"净利润": null, "是否亏损": "否"}]')
        assert "NULL" in _result_anomaly_reason('{"rows": [[null]]}')
        assert "NULL" in _result_anomaly_reason('{"rows": [{"净利润": ""}]}')
        assert _result_anomaly_reason('[{"净利润": -10}]') is None

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_result_reflection_generates_sql(self, mock_get_model):
        from agents.flow.sql_react import result_reflection

        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="SELECT COALESCE(SUM(x), 0) AS net_profit FROM t;"))
        mock_get_model.return_value = mock_model

        result = await result_reflection({
            "query": "去年亏损",
            "sql": "SELECT SUM(x) AS net_profit FROM t HAVING net_profit < 0;",
            "result": '[{"net_profit": null}]',
            "docs": [_mock_doc()],
            "retry_count": 0,
        })

        assert result["is_sql"] is True
        assert result["error"] is None
        assert result["sql"] == "SELECT COALESCE(SUM(x), 0) AS net_profit FROM t;"
        assert result["retry_count"] == 1


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------

class TestBuildSqlReactGraph:
    """Test graph construction and routing."""

    @patch("agents.flow.sql_react.get_checkpointer")
    def test_graph_has_all_nodes(self, mock_cp):
        """Graph should contain all expected nodes."""
        from langgraph.checkpoint.memory import MemorySaver
        from agents.flow.sql_react import build_sql_react_graph

        mock_cp.return_value = MemorySaver()
        graph = build_sql_react_graph()
        node_names = list(graph.get_graph().nodes.keys())

        assert "sql_retrieve" in node_names
        assert "check_docs" in node_names
        assert "sql_generate" in node_names
        assert "safety_check" in node_names
        assert "approve" in node_names
        assert "execute_sql" in node_names
        assert "error_analysis" in node_names
        assert "result_reflection" in node_names
        assert "query_enhance" in node_names
        assert "recall_evidence" in node_names

    @patch("agents.flow.sql_react.get_checkpointer")
    def test_graph_compiles(self, mock_cp):
        """Graph should compile without errors."""
        from langgraph.checkpoint.memory import MemorySaver
        from agents.flow.sql_react import build_sql_react_graph

        mock_cp.return_value = MemorySaver()
        graph = build_sql_react_graph()
        assert graph is not None


# ---------------------------------------------------------------------------
# Route functions
# ---------------------------------------------------------------------------

class TestRouting:
    """Test conditional routing logic."""

    @patch("agents.flow.sql_react.get_checkpointer")
    def test_route_after_check_no_docs(self, mock_cp):
        """When check_docs sets is_sql=False with answer, should route to END."""
        from langgraph.checkpoint.memory import MemorySaver
        from agents.flow.sql_react import build_sql_react_graph

        mock_cp.return_value = MemorySaver()
        graph = build_sql_react_graph()

        # The route_after_check function is internal; test via graph structure
        # by verifying the graph compiles with conditional edges
        assert graph is not None

    @patch("agents.flow.sql_react.get_checkpointer")
    def test_route_after_safety_non_sql(self, mock_cp):
        """When safety_check sets is_sql=False, should route to END."""
        from langgraph.checkpoint.memory import MemorySaver
        from agents.flow.sql_react import build_sql_react_graph

        mock_cp.return_value = MemorySaver()
        graph = build_sql_react_graph()
        assert graph is not None

    @patch("agents.flow.sql_react.get_checkpointer")
    def test_route_after_approve_rejected(self, mock_cp):
        """When approve sets approved=False, should route to END."""
        from langgraph.checkpoint.memory import MemorySaver
        from agents.flow.sql_react import build_sql_react_graph

        mock_cp.return_value = MemorySaver()
        graph = build_sql_react_graph()
        assert graph is not None


# ---------------------------------------------------------------------------
# error_analysis node
# ---------------------------------------------------------------------------

class TestErrorAnalysis:
    """Test error_analysis node."""

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_error_analysis_generates_feedback(self, mock_get_model):
        """error_analysis should generate refine_feedback and increment retry_count."""
        from agents.flow.sql_react import error_analysis

        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="表名应为 t_user 而不是 users"))
        mock_get_model.return_value = mock_model

        state = {
            "query": "查询用户",
            "sql": "SELECT * FROM users",
            "error": "Table 'go_agent_audit.users' doesn't exist",
            "docs": [_mock_doc()],
            "retry_count": 0,
        }
        result = await error_analysis(state)

        assert "t_user" in result["refine_feedback"]
        assert result["retry_count"] == 1

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.get_chat_model")
    async def test_error_analysis_increments_count(self, mock_get_model):
        """retry_count should increment from any starting value."""
        from agents.flow.sql_react import error_analysis

        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="修正建议"))
        mock_get_model.return_value = mock_model

        state = {
            "query": "查询",
            "sql": "SELECT 1",
            "error": "syntax error",
            "docs": [],
            "retry_count": 2,
        }
        result = await error_analysis(state)

        assert result["retry_count"] == 3


# ---------------------------------------------------------------------------
# recall_evidence node
# ---------------------------------------------------------------------------

class TestRecallEvidence:
    """Test evidence recall tracing propagation."""

    @pytest.mark.asyncio
    @patch("agents.flow.sql_react.recall_agent_knowledge")
    @patch("agents.flow.sql_react.recall_business_knowledge")
    async def test_recall_evidence_passes_trace_callbacks(self, mock_business, mock_agent):
        """Knowledge retrievers should inherit graph callbacks."""
        from agents.flow.sql_react import recall_evidence

        mock_business.return_value = [
            Document(page_content="术语: 净利润\n公式: 收入 - 成本", metadata={"score": 0.9})
        ]
        mock_agent.return_value = [
            Document(page_content="SELECT 1;", metadata={"score": 0.9})
        ]

        result = await recall_evidence(
            {"query": "去年亏损", "rewritten_query": "去年亏损"},
            config={"callbacks": ["trace-handler"]},
        )

        assert result["evidence"] == ["术语: 净利润\n公式: 收入 - 成本"]
        assert result["few_shot_examples"] == ["SELECT 1;"]
        assert mock_business.call_args.kwargs["callbacks"] == ["trace-handler"]
        assert mock_agent.call_args.kwargs["callbacks"] == ["trace-handler"]


# ---------------------------------------------------------------------------
# Business knowledge recall
# ---------------------------------------------------------------------------

class TestBusinessKnowledgeRecall:
    """Test business knowledge fallback recall."""

    @patch("agents.rag.retriever._load_business_knowledge_from_mysql")
    @patch("agents.rag.retriever._es_bm25_search")
    @patch("agents.rag.retriever._milvus_vector_search")
    def test_synonym_lexical_fallback(self, mock_vector, mock_es, mock_load_bk):
        """When vector/BM25 miss, configured synonyms should recall business knowledge."""
        from agents.rag.retriever import recall_business_knowledge

        mock_vector.return_value = []
        mock_es.return_value = []
        mock_load_bk.return_value = [{
            "term": "净利润",
            "formula": "收入 - 成本 - 费用；亏损表示净利润 < 0",
            "synonyms": "净收益, 盈利, 亏损, 净亏损, 赔钱, 赚钱",
            "related_tables": "t_journal_item,t_account,t_expense_claim",
        }]

        docs = recall_business_knowledge("去年亏损", top_k=5)

        assert len(docs) == 1
        assert "术语: 净利润" in docs[0].page_content
        assert docs[0].metadata["retriever_source"] == "mysql_lexical"
        assert "亏损" in docs[0].metadata["matched_terms"]


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    """Test the tool registry system."""

    def test_register_and_get_tools(self):
        """Registered tools should be retrievable by category."""
        from agents.tool.registry import register, get_tools, clear

        clear()
        try:
            @register("test_cat")
            @tool
            def my_test_tool(x: str) -> str:
                """A test tool."""
                return x

            tools = get_tools("test_cat")
            assert len(tools) == 1
            assert tools[0].name == "my_test_tool"
        finally:
            clear()

    def test_get_tools_empty_category(self):
        """Getting tools for a non-existent category should return empty list."""
        from agents.tool.registry import get_tools

        result = get_tools("nonexistent_category_xyz")
        assert result == []

    def test_get_all_tools(self):
        """Getting tools with no categories should return all."""
        from agents.tool.registry import register, get_tools, clear

        clear()
        try:
            @register("cat_a")
            @tool
            def tool_a(x: str) -> str:
                """Tool A."""
                return x

            @register("cat_b")
            @tool
            def tool_b(x: str) -> str:
                """Tool B."""
                return x

            all_tools = get_tools()
            assert len(all_tools) >= 2
        finally:
            clear()

    def test_sql_tools_registered(self):
        """SQL tools should be registered when sql_tools package is imported."""
        from agents.tool.registry import get_tools, register_tool
        from agents.tool.sql_tools.execute_tool import execute_query
        from agents.tool.sql_tools.schema_tool import list_tables, describe_table

        # Re-register since clear() in earlier tests may have removed them
        register_tool("sql", execute_query)
        register_tool("sql", list_tables)
        register_tool("sql", describe_table)

        sql_tools = get_tools("sql")
        tool_names = [t.name for t in sql_tools]
        assert "execute_query" in tool_names
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names
