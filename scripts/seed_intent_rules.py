"""Seed configurable intent rules into MySQL.

Usage:
    python -m scripts.seed_intent_rules

Optionally override the seed file:
    INTENT_RULES_SEED_FILE=/path/to/intent_rules.json python -m scripts.seed_intent_rules
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.tool.storage.intent_rules import ensure_intent_rule_table, upsert_intent_rule

_DEFAULT_SEED_FILE = Path(__file__).resolve().parents[1] / "data" / "intent_rules_seed.json"


def load_seed_records() -> list[dict]:
    seed_file = Path(os.getenv("INTENT_RULES_SEED_FILE", str(_DEFAULT_SEED_FILE))).expanduser()
    with seed_file.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Intent rules seed file must contain a list: {seed_file}")
    return records


def main() -> None:
    ensure_intent_rule_table()
    records = load_seed_records()
    for record in records:
        upsert_intent_rule(record)
    print(f"Seeded {len(records)} intent rules")


if __name__ == "__main__":
    main()
