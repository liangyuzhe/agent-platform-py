"""Tests for evaluation report API endpoints."""

from __future__ import annotations

import json
import os

from fastapi.testclient import TestClient


def _write_report(path, run_id: str, accuracy: float):
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": f"2026-05-08T00:00:0{int(accuracy)}Z",
                "dataset_path": "data/eval/eval_dataset.jsonl",
                "report_type": "retrieval",
                "strategies": [
                    {
                        "strategy": "schema_lexical",
                        "description": "schema baseline",
                        "num_queries": 1,
                        "metrics": {"accuracy@5": accuracy, "recall@5": accuracy},
                        "latency": {"avg_ms": 10.0, "p50_ms": 10.0, "p95_ms": 10.0},
                        "first_token_latency": {"avg_ms": None, "p50_ms": None, "p95_ms": None},
                        "results": [],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_eval_reports_list_and_load_by_name(tmp_path, monkeypatch):
    from agents.api.app import app
    from agents.api.routers import eval as eval_router

    old_report = tmp_path / "eval_report_old.json"
    new_report = tmp_path / "nl2sql_eval_report.json"
    _write_report(old_report, "old", 0.5)
    _write_report(new_report, "new", 1.0)
    os.utime(old_report, (100, 100))
    os.utime(new_report, (200, 200))

    monkeypatch.setattr(eval_router, "_REPORT_DIRS", [tmp_path])
    monkeypatch.setattr(eval_router, "_REPORT_PATTERNS", ["*eval_report*.json"])

    client = TestClient(app, raise_server_exceptions=False)

    list_resp = client.get("/api/eval/reports")
    assert list_resp.status_code == 200
    reports = list_resp.json()["reports"]
    assert [r["name"] for r in reports] == ["nl2sql_eval_report.json", "eval_report_old.json"]

    detail_resp = client.get("/api/eval/reports/nl2sql_eval_report.json")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["run_id"] == "new"
    assert detail_resp.json()["_name"] == "nl2sql_eval_report.json"


def test_eval_report_by_name_rejects_path_traversal(tmp_path, monkeypatch):
    from agents.api.app import app
    from agents.api.routers import eval as eval_router

    _write_report(tmp_path / "eval_report.json", "run-1", 1.0)
    monkeypatch.setattr(eval_router, "_REPORT_DIRS", [tmp_path])
    monkeypatch.setattr(eval_router, "_REPORT_PATTERNS", ["*eval_report*.json"])

    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/eval/reports/%2E%2E%2Fsecret.json")
    assert resp.status_code == 404
