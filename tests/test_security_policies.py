from agents.tool.security.policies import (
    SecurityContext,
    authorize_tables,
    build_audit_event,
)


def test_authorize_tables_default_allows_without_context():
    result = authorize_tables(["t_orders", "t_customers", "t_orders"], None)

    assert result.allowed is True
    assert result.allowed_tables == ["t_orders", "t_customers"]
    assert result.denied_tables == []
    assert result.display_denied_tables == []
    assert result.message == ""


def test_authorize_tables_allow_list_denial_uses_display_names():
    result = authorize_tables(
        ["t_orders", "t_payroll"],
        {"allowed_tables": ["t_orders"]},
        table_metadata={"t_payroll": "薪资数据"},
    )

    assert result.allowed is False
    assert result.allowed_tables == ["t_orders"]
    assert result.denied_tables == ["t_payroll"]
    assert result.display_denied_tables == ["薪资数据"]
    assert "薪资数据" in result.message
    assert "t_payroll" not in result.message


def test_authorize_tables_denied_tables_override_allow_list():
    context = SecurityContext(
        allowed_tables=["t_orders", "t_payroll"],
        denied_tables=["t_payroll"],
    )

    result = authorize_tables(
        ["t_orders", "t_payroll"],
        context,
        table_metadata={"t_payroll": "薪资数据"},
        stage="sql_generation",
    )

    assert result.allowed is False
    assert result.allowed_tables == ["t_orders"]
    assert result.denied_tables == ["t_payroll"]
    assert result.display_denied_tables == ["薪资数据"]
    assert result.stage == "sql_generation"


def test_build_audit_event_includes_security_and_table_fields():
    event = build_audit_event(
        "table_authorization",
        query="select * from t_orders",
        context={
            "user_id": "u-1",
            "role_ids": ["finance", "auditor"],
        },
        selected_tables=["t_orders", "t_payroll"],
        denied_tables=["t_payroll"],
        display_tables=["薪资数据"],
        status="denied",
        error="missing permission",
        extra={"stage": "selected_tables"},
    )

    assert event["event_type"] == "table_authorization"
    assert event["query"] == "select * from t_orders"
    assert event["user_id"] == "u-1"
    assert event["role_ids"] == ["finance", "auditor"]
    assert event["selected_tables"] == ["t_orders", "t_payroll"]
    assert event["denied_tables"] == ["t_payroll"]
    assert event["display_tables"] == ["薪资数据"]
    assert event["status"] == "denied"
    assert event["error"] == "missing permission"
    assert event["extra"] == {"stage": "selected_tables"}


def test_build_audit_event_preserves_extra_without_overriding_core_fields():
    event = build_audit_event(
        "table_authorization",
        context={"user_id": "u-1"},
        extra={"user_id": "evil", "event_type": "other"},
    )

    assert event["event_type"] == "table_authorization"
    assert event["user_id"] == "u-1"
    assert event["extra"] == {"user_id": "evil", "event_type": "other"}
