"""Best-effort audit logging helpers.

The audit writer must never break the user query path. Storage failures are
logged and swallowed so permission denials and SQL execution can still return a
deterministic response.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def write_audit_log(event: dict[str, Any]) -> bool:
    """Persist an audit event when audit storage is available.

    V1 intentionally keeps this best-effort and dependency-light. A later
    migration can create ``t_query_audit_log`` and replace this with a concrete
    insert while preserving the no-throw contract.
    """
    try:
        logger.info("query_audit_event=%s", json.dumps(event, ensure_ascii=False, default=str))
        return True
    except Exception as exc:
        logger.warning("Failed to write audit log: %s", exc)
        return False
