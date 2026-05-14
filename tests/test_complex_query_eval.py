from agents.eval.strategies import run_complex_route_eval_case, route_accuracy


def test_route_accuracy_counts_matching_routes():
    score = route_accuracy([
        ("single_sql", "single_sql"),
        ("complex_plan", "complex_plan"),
        ("clarify", "complex_plan"),
    ])

    assert score == 2 / 3


def test_complex_route_eval_case_uses_route_signal_for_broad_schema():
    result = run_complex_route_eval_case({
        "query": "收入成本预算回款费用之间的关系",
        "tables": [f"t_{i}" for i in range(9)],
        "route_signal": "analysis",
        "expected_route": "complex_plan",
    })

    assert result["actual_route"] == "complex_plan"
    assert result["expected_route"] == "complex_plan"
    assert result["passed"] is True
