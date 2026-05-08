"""Unit tests for evaluation dataset generation helpers."""

from agents.eval.dataset_generator import _parse_eval_items, _schema_rows_to_docs


def test_schema_rows_to_docs_renders_semantic_metadata():
    rows = [
        {
            "table_name": "t_journal_entry",
            "column_name": "id",
            "column_type": "int",
            "column_comment": "主键",
            "is_pk": 1,
            "is_fk": 0,
            "ref_table": None,
            "ref_column": None,
            "business_name": "凭证ID",
            "synonyms": "",
            "business_description": "凭证唯一标识",
        },
        {
            "table_name": "t_journal_item",
            "column_name": "entry_id",
            "column_type": "int",
            "column_comment": "凭证ID",
            "is_pk": 0,
            "is_fk": 1,
            "ref_table": "t_journal_entry",
            "ref_column": "id",
            "business_name": "凭证ID",
            "synonyms": "凭证主表ID",
            "business_description": "关联凭证主表",
        },
    ]

    docs = _schema_rows_to_docs(rows)

    assert [doc["doc_id"] for doc in docs] == ["schema_t_journal_entry", "schema_t_journal_item"]
    assert "id int PRIMARY KEY -- 主键 [业务名: 凭证ID] [描述: 凭证唯一标识]" in docs[0]["text"]
    assert "entry_id int REFERENCES t_journal_entry(id)" in docs[1]["text"]
    assert "[同义词: 凭证主表ID]" in docs[1]["text"]


def test_schema_rows_to_docs_ignores_rows_without_table_name():
    docs = _schema_rows_to_docs([{"table_name": "", "column_name": "id"}])

    assert docs == []


def test_parse_eval_items_accepts_json_lines_and_embedded_json():
    raw = """
    {"query": "去年亏损多少", "relevant_doc_ids": ["schema_t_journal_entry", "schema_t_journal_item"]}
    1. {"query": "查询凭证状态", "relevant_doc_ids": ["schema_t_journal_entry"]}
    not json
    ```
    """

    items = _parse_eval_items(raw)

    assert items == [
        {
            "query": "去年亏损多少",
            "relevant_doc_ids": ["schema_t_journal_entry", "schema_t_journal_item"],
        },
        {
            "query": "查询凭证状态",
            "relevant_doc_ids": ["schema_t_journal_entry"],
        },
    ]


def test_parse_eval_items_requires_query_and_relevant_doc_ids():
    raw = """
    {"query": "缺少 doc ids"}
    {"relevant_doc_ids": ["schema_t_account"]}
    {"query": "完整样本", "relevant_doc_ids": ["schema_t_account"]}
    """

    items = _parse_eval_items(raw)

    assert items == [{"query": "完整样本", "relevant_doc_ids": ["schema_t_account"]}]
