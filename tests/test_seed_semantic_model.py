"""Tests for semantic seed configuration."""

from scripts.seed_semantic_model import (
    semantic_model_records,
    table_visible_semantics,
)


def test_management_tables_have_visible_table_semantics():
    comments = table_visible_semantics()

    assert "预算金额" in comments["t_budget"]
    assert "成本中心" in comments["t_budget"]
    assert "年度预算" in comments["t_cost_center"]
    assert "真实姓名" in comments["t_user"]
    assert "联系电话" in comments["t_user"]
    assert "角色名称" in comments["t_role"]
    assert "用户角色绑定" in comments["t_user_role"]
    assert "部门负责人" in comments["t_department"]
    assert "用户部门归属" in comments["t_user_department"]


def test_management_fields_have_business_semantics():
    records = {
        (table_name, column_name): (business_name, synonyms, description)
        for table_name, column_name, business_name, synonyms, description in semantic_model_records()
    }

    assert records[("t_user", "real_name")][0] == "真实姓名"
    assert "员工姓名" in records[("t_user", "real_name")][1]
    assert records[("t_user", "phone")][0] == "联系电话"
    assert records[("t_role", "name")][0] == "角色名称"
    assert records[("t_department", "manager")][0] == "部门负责人"
    assert records[("t_user_department", "is_leader")][0] == "是否部门负责人"
