"""Unit tests for schema-metadata evaluation retrievers."""

from unittest.mock import patch

from agents.eval.runner import StrategyConfig, _SchemaMetadataRetriever, _build_retriever


_DOCS = [
    {
        "doc_id": "schema_t_journal_entry",
        "pk": "schema_t_journal_entry",
        "table_name": "t_journal_entry",
        "text": "表名: t_journal_entry\n字段:\n  status [业务名: 凭证状态]\n  period [业务名: 会计期间]",
    },
    {
        "doc_id": "schema_t_account",
        "pk": "schema_t_account",
        "table_name": "t_account",
        "text": "表名: t_account\n字段:\n  account_type [业务名: 科目类型]\n  account_name [业务名: 科目名称]",
    },
]


@patch("agents.eval.dataset_generator._fetch_all_schema_docs", return_value=_DOCS)
def test_schema_metadata_retriever_uses_business_field_text(mock_fetch):
    retriever = _SchemaMetadataRetriever(include_fields=True, top_k=5)

    docs = retriever.retrieve("凭证状态")

    assert docs
    assert docs[0].metadata["doc_id"] == "schema_t_journal_entry"
    assert docs[0].metadata["retriever_source"] == "schema_metadata"


@patch("agents.eval.dataset_generator._fetch_all_schema_docs", return_value=_DOCS)
def test_schema_table_name_retriever_ignores_field_text(mock_fetch):
    retriever = _SchemaMetadataRetriever(include_fields=False, top_k=5)

    docs = retriever.retrieve("凭证状态")

    assert docs == []


@patch("agents.eval.dataset_generator._fetch_all_schema_docs", return_value=_DOCS)
def test_build_retriever_defaults_to_schema_lexical(mock_fetch):
    retriever = _build_retriever(StrategyConfig(name="schema", description="schema"))

    docs = retriever.retrieve("科目类型")

    assert docs[0].metadata["doc_id"] == "schema_t_account"
