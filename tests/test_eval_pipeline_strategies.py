"""Unit tests for online evaluation strategies."""

from __future__ import annotations

import json
from unittest.mock import patch

from langchain_core.documents import Document

from agents.eval.runner import StrategyConfig, evaluate_strategy, run_evaluation
from agents.eval.strategies import run_preselect_pipeline


async def _fake_recall_evidence(state):
    return {"business_knowledge": [{"term": "净利润"}]}


async def _fake_query_enhance(state):
    assert state["business_knowledge"] == [{"term": "净利润"}]
    return {"enhanced_query": f"{state['query']} 净利润口径"}


async def _fake_select_tables(state):
    assert state["enhanced_query"].endswith("净利润口径")
    return {"selected_tables": ["t_journal_entry", "t_journal_item"]}


@patch("agents.flow.sql_react.recall_evidence", side_effect=_fake_recall_evidence)
@patch("agents.flow.sql_react.query_enhance", side_effect=_fake_query_enhance)
@patch("agents.flow.sql_react.select_tables", side_effect=_fake_select_tables)
def test_preselect_pipeline_runs_real_node_order(mock_select, mock_enhance, mock_recall):
    import asyncio

    result = asyncio.run(run_preselect_pipeline("去年亏损多少"))

    assert result.retrieved_doc_ids == ["schema_t_journal_entry", "schema_t_journal_item"]
    assert result.latency_ms >= 0
    assert mock_recall.call_count == 1
    assert mock_enhance.call_count == 1
    assert mock_select.call_count == 1


@patch("agents.rag.retriever.recall_business_knowledge")
def test_business_knowledge_strategy_uses_business_labels_only(mock_recall):
    mock_recall.return_value = [
        Document(page_content="净利润公式", metadata={"doc_id": "business_net_profit"}),
        Document(page_content="毛利率公式", metadata={"doc_id": "business_gross_margin"}),
    ]
    dataset = [
        {
            "query": "去年亏损多少",
            "relevant_doc_ids": ["schema_t_journal_entry"],
            "relevant_business_doc_ids": ["business_net_profit"],
        },
        {
            "query": "查凭证",
            "relevant_doc_ids": ["schema_t_journal_entry"],
        },
    ]

    report = evaluate_strategy(
        StrategyConfig(
            name="business",
            description="business",
            mode="business_knowledge_recall",
            relevant_field="relevant_business_doc_ids",
            reranker_top_k=5,
        ),
        dataset,
        k_values=[1, 5],
    )

    assert len(report.results) == 1
    assert report.results[0].retrieved_doc_ids == ["business_net_profit", "business_gross_margin"]
    assert report.aggregate["accuracy@1"] == 1.0
    mock_recall.assert_called_once_with("去年亏损多少", 5)


@patch("agents.rag.retriever.recall_agent_knowledge")
def test_agent_knowledge_strategy_supports_term_metadata_fallback(mock_recall):
    mock_recall.return_value = [Document(page_content="SQL 示例", metadata={"term": "agent_loss_case"})]

    report = evaluate_strategy(
        StrategyConfig(
            name="agent",
            description="agent",
            mode="agent_knowledge_recall",
            relevant_field="relevant_agent_doc_ids",
            reranker_top_k=3,
        ),
        [{"query": "亏损多少", "relevant_agent_doc_ids": ["agent_loss_case"]}],
        k_values=[1],
    )

    assert report.results[0].retrieved_doc_ids == ["agent_loss_case"]
    assert report.aggregate["recall@1"] == 1.0
    mock_recall.assert_called_once_with("亏损多少", 3)


def test_strategy_without_matching_labels_is_not_applicable():
    report = evaluate_strategy(
        StrategyConfig(
            name="business",
            description="business",
            mode="business_knowledge_recall",
            relevant_field="relevant_business_doc_ids",
        ),
        [{"query": "查凭证", "relevant_doc_ids": ["schema_t_journal_entry"]}],
        k_values=[1],
    )

    assert report.results == []
    assert report.aggregate == {}


@patch("agents.eval.dataset_generator._fetch_all_schema_docs")
@patch("agents.flow.sql_react.recall_evidence", side_effect=_fake_recall_evidence)
@patch("agents.flow.sql_react.query_enhance", side_effect=_fake_query_enhance)
@patch("agents.flow.sql_react.select_tables", side_effect=_fake_select_tables)
def test_default_run_skips_llm_backed_preselect_strategy(
    mock_select,
    mock_enhance,
    mock_recall,
    mock_schema_docs,
    tmp_path,
):
    mock_schema_docs.return_value = [
        {
            "doc_id": "schema_t_journal_entry",
            "table_name": "t_journal_entry",
            "text": "凭证 会计期间",
        }
    ]
    dataset_path = tmp_path / "eval.jsonl"
    output_path = tmp_path / "report.json"
    dataset_path.write_text(
        json.dumps(
            {
                "query": "去年亏损多少",
                "relevant_doc_ids": ["schema_t_journal_entry", "schema_t_journal_item"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    reports = run_evaluation(dataset_path=dataset_path, output_path=output_path, k_values=[1, 5])
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert [r.config.name for r in reports] == [
        "schema_lexical",
        "schema_table_name",
        "business_knowledge_recall",
        "agent_knowledge_recall",
    ]
    assert {entry["strategy"] for entry in payload["strategies"]} == {
        "schema_lexical",
        "schema_table_name",
        "business_knowledge_recall",
        "agent_knowledge_recall",
    }
    assert mock_recall.call_count == 0
    assert mock_enhance.call_count == 0
    assert mock_select.call_count == 0


@patch("agents.eval.dataset_generator._fetch_all_schema_docs")
@patch("agents.flow.sql_react.recall_evidence", side_effect=_fake_recall_evidence)
@patch("agents.flow.sql_react.query_enhance", side_effect=_fake_query_enhance)
@patch("agents.flow.sql_react.select_tables", side_effect=_fake_select_tables)
def test_online_flag_includes_preselect_strategy(
    mock_select,
    mock_enhance,
    mock_recall,
    mock_schema_docs,
    tmp_path,
):
    mock_schema_docs.return_value = [
        {
            "doc_id": "schema_t_journal_entry",
            "table_name": "t_journal_entry",
            "text": "凭证 会计期间",
        }
    ]
    dataset_path = tmp_path / "eval.jsonl"
    output_path = tmp_path / "report.json"
    dataset_path.write_text(
        json.dumps(
            {
                "query": "去年亏损多少",
                "relevant_doc_ids": ["schema_t_journal_entry", "schema_t_journal_item"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    reports = run_evaluation(
        dataset_path=dataset_path,
        output_path=output_path,
        k_values=[1, 5],
        include_online_pipeline=True,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert [r.config.name for r in reports] == [
        "schema_lexical",
        "schema_table_name",
        "preselect_pipeline",
        "business_knowledge_recall",
        "agent_knowledge_recall",
    ]
    assert payload["strategies"][2]["strategy"] == "preselect_pipeline"
    assert payload["strategies"][2]["num_queries"] == 1
    assert payload["strategies"][3]["num_queries"] == 0
    assert payload["strategies"][4]["num_queries"] == 0
