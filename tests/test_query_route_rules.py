from agents.tool.storage.query_route_rules import _matches, _normalize_route_signal


def test_query_route_rule_contains_match_uses_database_pattern_value():
    rule = {"pattern": "configured-analysis-pattern", "match_type": "contains"}

    assert _matches("ask configured-analysis-pattern now", rule)


def test_query_route_rule_regex_match_invalid_pattern_is_non_match():
    rule = {"id": 1, "pattern": "(", "match_type": "regex"}

    assert not _matches("anything", rule)


def test_normalize_route_signal_rejects_unknown_signal():
    try:
        _normalize_route_signal("unknown")
    except ValueError as exc:
        assert "Unsupported route signal" in str(exc)
    else:
        raise AssertionError("expected ValueError")
