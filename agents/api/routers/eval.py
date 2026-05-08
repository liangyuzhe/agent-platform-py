"""Evaluation report APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from agents.eval.reporting import compact_report

router = APIRouter()

_ROOT = Path(__file__).resolve().parents[3]
_REPORT_DIRS = [_ROOT / "data" / "eval", _ROOT]
_REPORT_PATTERNS = ["eval_report*.json", "*.eval.json"]


def _candidate_report_paths() -> list[Path]:
    paths: list[Path] = []
    for directory in _REPORT_DIRS:
        if not directory.exists():
            continue
        for pattern in _REPORT_PATTERNS:
            paths.extend(directory.glob(pattern))
    unique = {p.resolve(): p for p in paths}
    return sorted(unique.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def _load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        strategies = []
        for item in payload:
            strategy = dict(item)
            avg_latency = strategy.pop("avg_latency_ms", None)
            strategy.setdefault("latency", {
                "avg_ms": avg_latency,
                "p50_ms": avg_latency,
                "p95_ms": avg_latency,
            })
            strategy.setdefault("first_token_latency", {
                "avg_ms": None,
                "p50_ms": None,
                "p95_ms": None,
            })
            strategy.setdefault("results", [])
            strategies.append(strategy)
        payload = {
            "run_id": path.stem,
            "created_at": None,
            "dataset_path": None,
            "report_type": "retrieval",
            "strategies": strategies,
        }
    payload["_path"] = str(path)
    return payload


@router.get("/reports")
async def list_reports():
    """List available evaluation reports."""
    reports = []
    for path in _candidate_report_paths():
        try:
            payload = _load_report(path)
            summary = compact_report(payload)
            summary["path"] = str(path)
            reports.append(summary)
        except Exception:
            continue
    return {"reports": reports}


@router.get("/reports/latest")
async def latest_report():
    """Return the latest evaluation report with per-query trace details."""
    paths = _candidate_report_paths()
    if not paths:
        raise HTTPException(status_code=404, detail="No evaluation report found")
    return _load_report(paths[0])
