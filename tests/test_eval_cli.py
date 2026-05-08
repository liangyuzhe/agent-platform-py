"""Tests for evaluation CLI behavior."""

from __future__ import annotations

from argparse import Namespace

import pytest

from agents.eval.cli import cmd_run_nl2sql


def test_run_nl2sql_missing_dataset_prints_actionable_error(tmp_path, capsys):
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(SystemExit) as exc:
        cmd_run_nl2sql(Namespace(dataset=str(missing), output=str(tmp_path / "report.json"), init_template=False))

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "NL2SQL dataset not found" in captured.err
    assert "--init-template" in captured.err


def test_run_nl2sql_init_template_writes_dataset(tmp_path, capsys):
    dataset = tmp_path / "cases.jsonl"

    cmd_run_nl2sql(Namespace(dataset=str(dataset), output=str(tmp_path / "report.json"), init_template=True))

    assert dataset.exists()
    assert "Wrote NL2SQL evaluation template" in capsys.readouterr().out
