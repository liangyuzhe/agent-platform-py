"""SFT sample storage and JSONL export.

Samples are stored in-memory by default with an optional persistent
backend.  The :func:`export_to_jsonl` function writes collected samples
to a JSONL file suitable for SFT training.

This is a reserved extension module. The default application path does not
persist SFT samples unless a caller explicitly binds the SFT callback.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory store keyed by sample id.
# ---------------------------------------------------------------------------
_store: Dict[str, Dict[str, Any]] = {}


async def save_sample(sample: Dict[str, Any]) -> str:
    """Persist a collected sample.

    Parameters
    ----------
    sample:
        A dict representing a single prompt-completion pair (see
        :class:`SFTCallbackHandler` for the expected schema).

    Returns
    -------
    str
        The sample id (assigned automatically if not present).
    """
    sid = sample.get("id") or str(uuid.uuid4())
    sample["id"] = sid
    sample.setdefault("created_at", time.time())
    _store[sid] = sample
    logger.debug("Saved SFT sample %s", sid)
    return sid


def get_sample(sample_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single sample by id."""
    return _store.get(sample_id)


def list_samples(
    agent_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Return samples, optionally filtered by *agent_id*."""
    items = list(_store.values())
    if agent_id is not None:
        items = [s for s in items if s.get("agent_id") == agent_id]
    # Sort newest-first.
    items.sort(key=lambda s: s.get("created_at", 0), reverse=True)
    return items[offset : offset + limit]


def delete_sample(sample_id: str) -> bool:
    """Remove a sample. Returns ``True`` if it existed."""
    return _store.pop(sample_id, None) is not None


def clear_samples(agent_id: Optional[str] = None) -> int:
    """Remove all samples (optionally scoped to *agent_id*). Returns count removed."""
    if agent_id is None:
        count = len(_store)
        _store.clear()
        return count
    to_remove = [sid for sid, s in _store.items() if s.get("agent_id") == agent_id]
    for sid in to_remove:
        del _store[sid]
    return len(to_remove)


def export_to_jsonl(
    agent_id: str,
    output_path: str | Path,
    opts: Optional[Dict[str, Any]] = None,
) -> int:
    """Export samples for *agent_id* to a JSONL file.

    Parameters
    ----------
    agent_id:
        Only samples belonging to this agent are exported.
    output_path:
        Destination file path.  Parent directories are created automatically.
    opts:
        Optional export configuration:

        - ``"min_score"`` (float): skip samples whose annotation score is
          below this threshold.
        - ``"include_raw"`` (bool): include the un-annotated fields in
          addition to the revised completion (default ``False``).
        - ``"fields"`` (list[str]): only include these top-level keys.

    Returns
    -------
    int
        Number of samples written.
    """
    opts = opts or {}
    min_score: float = opts.get("min_score", 0.0)
    include_raw: bool = opts.get("include_raw", False)
    allowed_fields: Optional[set[str]] = (
        set(opts["fields"]) if "fields" in opts else None
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    samples = list_samples(agent_id=agent_id, limit=10_000, offset=0)
    count = 0

    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            annotation = sample.get("annotation") or {}
            score = annotation.get("score", 1.0)
            if score < min_score:
                continue

            record: Dict[str, Any] = {
                "prompt": sample.get("prompt", ""),
                "completion": annotation.get("revised_completion") or sample.get("completion", ""),
            }

            if include_raw:
                record["raw_completion"] = sample.get("completion", "")
                record["annotation"] = annotation

            if allowed_fields is not None:
                record = {k: v for k, v in record.items() if k in allowed_fields}

            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Exported %d SFT samples to %s", count, path)
    return count
