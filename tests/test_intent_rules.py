"""Tests for database-backed intent rule matching mechanics."""

from agents.tool.storage.intent_rules import _matches


def test_intent_rule_contains_match_uses_database_pattern_value():
    rule = {"pattern": "external-company", "match_type": "contains"}

    assert _matches("ask external-company revenue", rule)


def test_intent_rule_regex_match_invalid_pattern_is_non_match():
    rule = {"id": 1, "pattern": "(", "match_type": "regex"}

    assert not _matches("anything", rule)
