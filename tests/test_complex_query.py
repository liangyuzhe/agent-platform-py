from agents.flow.complex_query import classify_query_complexity, validate_complex_plan


def test_small_schema_routes_to_single_sql():
    result = classify_query_complexity(
        query="去年亏损",
        selected_tables=["t_journal_entry", "t_journal_item", "t_account"],
        relationships=[],
    )

    assert result.route_mode == "single_sql"
    assert result.selected_tables_count == 3


def test_eight_tables_still_single_sql_with_strict_checks():
    result = classify_query_complexity(
        query="查询项目预算和费用明细",
        selected_tables=[f"t_{i}" for i in range(8)],
        relationships=[],
    )

    assert result.route_mode == "single_sql_with_strict_checks"


def test_more_than_eight_analysis_signal_routes_to_complex_plan():
    result = classify_query_complexity(
        query="今年收入成本预算回款费用之间的关系",
        selected_tables=[f"t_{i}" for i in range(9)],
        relationships=[],
        route_signal="analysis",
    )

    assert result.route_mode == "complex_plan"


def test_more_than_eight_detail_signal_routes_to_clarify():
    result = classify_query_complexity(
        query="员工工资和部门角色权限",
        selected_tables=[f"t_{i}" for i in range(9)],
        relationships=[],
        route_signal="detail",
    )

    assert result.route_mode == "clarify"


def test_more_than_eight_without_signal_routes_to_clarify():
    result = classify_query_complexity(
        query="员工工资和部门角色权限",
        selected_tables=[f"t_{i}" for i in range(9)],
        relationships=[],
    )

    assert result.route_mode == "clarify"
    assert result.query_intent_complexity == "ambiguous"


def test_validate_complex_plan_accepts_valid_plan():
    plan = {
        "mode": "complex_plan",
        "steps": [
            {"step": 1, "type": "sql", "goal": "查收入", "tables": ["a", "b"], "depends_on": [], "merge_keys": ["period"]},
            {"step": 2, "type": "sql", "goal": "查预算", "tables": ["c"], "depends_on": [], "merge_keys": ["period"]},
            {"step": 3, "type": "python_merge", "goal": "合并", "tables": [], "depends_on": [1, 2], "merge_keys": ["period"]},
        ],
        "requires_user_confirmation": True,
    }

    ok, error = validate_complex_plan(plan, allowed_tables={"a", "b", "c"})

    assert ok is True
    assert error == ""


def test_validate_complex_plan_rejects_unknown_table():
    plan = {
        "mode": "complex_plan",
        "steps": [
            {"step": 1, "type": "sql", "goal": "查收入", "tables": ["missing"], "depends_on": [], "merge_keys": ["period"]}
        ],
    }

    ok, error = validate_complex_plan(plan, allowed_tables={"a"})

    assert ok is False
    assert "unknown table" in error


def test_validate_complex_plan_rejects_missing_merge_key_for_merge():
    plan = {
        "mode": "complex_plan",
        "steps": [
            {"step": 1, "type": "sql", "goal": "查收入", "tables": ["a"], "depends_on": [], "merge_keys": []},
            {"step": 2, "type": "python_merge", "goal": "合并", "tables": [], "depends_on": [1], "merge_keys": []},
        ],
    }

    ok, error = validate_complex_plan(plan, allowed_tables={"a"})

    assert ok is False
    assert "merge_keys" in error
