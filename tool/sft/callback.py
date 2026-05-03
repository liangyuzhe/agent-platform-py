"""SFT data collection callback handler.

Captures LLM prompts and completions during agent runs so they can be
annotated and exported for supervised fine-tuning.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

from agents.tool.sft.storage import save_sample

logger = logging.getLogger(__name__)


class SFTCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that collects prompt-completion pairs for SFT.

    Usage::

        handler = SFTCallbackHandler(agent_id="my_agent")
        llm.bind(callbacks=[handler]).invoke("Hello")

    Every completed LLM call is persisted asynchronously via
    :func:`agents.tool.sft.storage.save_sample`.
    """

    def __init__(self, agent_id: str = "default") -> None:
        super().__init__()
        self.agent_id = agent_id
        self._pending: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # LLM callbacks
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Record the prompt at the start of an LLM call."""
        rid = str(run_id) if run_id else str(uuid.uuid4())
        self._pending[rid] = {
            "id": rid,
            "agent_id": self.agent_id,
            "prompt": prompts[0] if prompts else "",
            "prompts": prompts,
            "model_name": serialized.get("name", serialized.get("id", [""])[-1] if isinstance(serialized.get("id"), list) else ""),
            "start_time": time.time(),
            "metadata": {k: v for k, v in kwargs.items() if k != "run_id"},
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Capture the completion and schedule an async save."""
        rid = str(run_id) if run_id else None
        if rid is None:
            # Try to find the most recently started run.
            if self._pending:
                rid = max(self._pending, key=lambda k: self._pending[k]["start_time"])
            else:
                return

        sample = self._pending.pop(rid, None)
        if sample is None:
            return

        # Extract completion text.
        generations = response.generations
        if generations and generations[0]:
            sample["completion"] = generations[0][0].text
        else:
            sample["completion"] = ""

        sample["end_time"] = time.time()
        sample["duration_ms"] = round((sample["end_time"] - sample["start_time"]) * 1000, 2)

        # LLM usage metadata.
        if response.llm_output:
            sample["token_usage"] = response.llm_output.get("token_usage", {})
            sample["model_name"] = response.llm_output.get("model_name", sample.get("model_name", ""))

        # Persist asynchronously.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(save_sample(sample))
        except RuntimeError:
            asyncio.run(save_sample(sample))

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Log and discard the pending sample on error."""
        rid = str(run_id) if run_id else None
        if rid and rid in self._pending:
            logger.warning("LLM error for run %s: %s", rid, error)
            self._pending.pop(rid, None)
