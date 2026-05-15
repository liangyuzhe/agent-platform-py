import json

from agents.tool.security.presentation import (
    build_column_display_map,
    format_result_for_user,
)


SEMANTIC_MODEL = {
    "t_role": {
        "name": {"business_name": "角色名称", "column_comment": "系统角色"},
    },
    "t_department": {
        "name": {"business_name": "部门名称", "column_comment": "组织名称"},
    },
    "t_user": {
        "real_name": {"business_name": "真实姓名", "column_comment": "姓名"},
        "amount": {"business_name": "", "column_comment": "金额"},
    },
}


def test_duplicate_name_maps_using_role_table_order():
    display_map = build_column_display_map(
        ["name"],
        semantic_model=SEMANTIC_MODEL,
        table_names=["t_role"],
    )

    assert display_map == {"name": "角色名称"}


def test_duplicate_name_maps_using_department_table_order():
    display_map = build_column_display_map(
        ["name"],
        semantic_model=SEMANTIC_MODEL,
        table_names=["t_department"],
    )

    assert display_map == {"name": "部门名称"}


def test_chinese_alias_is_preserved():
    display_map = build_column_display_map(
        ["角色名称"],
        semantic_model=SEMANTIC_MODEL,
        table_names=["t_role"],
        sql="select name as 角色名称 from t_role",
    )

    assert display_map == {"角色名称": "角色名称"}


def test_chinese_alias_for_other_column_does_not_override_requested_column():
    display_map = build_column_display_map(
        ["id"],
        sql="select name as 角色名称, id from t_user",
        semantic_model={"t_user": {"id": {"business_name": "用户ID"}}},
        table_names=["t_user"],
    )

    assert display_map == {"id": "用户ID"}


def test_sql_react_raw_payload_with_execution_time_is_parsed():
    text, metadata = format_result_for_user(
        '[{"real_name":"张三"}]Query execution time: 10.97 ms',
        semantic_model=SEMANTIC_MODEL,
        table_names=["t_user"],
    )

    assert text.startswith("查询已执行完成。")
    assert "真实姓名：张三" in text
    assert metadata == {"display_columns": {"real_name": "真实姓名"}, "row_count": 1}


def test_json_list_rows_format_to_business_labels_and_row_count():
    result = json.dumps(
        [
            {"real_name": "张三", "amount": 100},
            {"real_name": "李四", "amount": 200},
        ],
        ensure_ascii=False,
    )

    text, metadata = format_result_for_user(
        result,
        semantic_model=SEMANTIC_MODEL,
        table_names=["t_user"],
    )

    assert text.startswith("查询已执行完成。")
    assert "真实姓名：张三" in text
    assert "金额：100" in text
    assert "真实姓名：李四" in text
    assert metadata == {
        "display_columns": {"real_name": "真实姓名", "amount": "金额"},
        "row_count": 2,
    }


def test_wrapper_row_count_metadata_is_preserved():
    text, metadata = format_result_for_user(
        json.dumps({"rows": [{"real_name": "张三"}], "row_count": 50}, ensure_ascii=False),
        semantic_model=SEMANTIC_MODEL,
        table_names=["t_user"],
    )

    assert text.startswith("查询已执行完成。")
    assert "真实姓名：张三" in text
    assert metadata == {"display_columns": {"real_name": "真实姓名"}, "row_count": 50}


def test_empty_json_list_returns_completed_message_prefix():
    text, metadata = format_result_for_user("[]")

    assert text.startswith("查询已执行完成。")
    assert metadata == {"display_columns": {}, "row_count": 0}


def test_empty_json_list_has_exact_completed_prefix():
    text, _ = format_result_for_user("[]")

    assert text.startswith("查询已执行完成。")
    assert text != "查询已执行完成，未查询到符合条件的数据。"


def test_raw_non_json_result_returns_completed_message_and_unknown_row_count():
    text, metadata = format_result_for_user(
        "Query execution time: 2 ms",
        semantic_model=SEMANTIC_MODEL,
        table_names=["t_user"],
    )

    assert text.startswith("查询已执行完成。")
    assert "Query execution time: 2 ms" in text
    assert metadata == {"display_columns": {}, "row_count": None}
