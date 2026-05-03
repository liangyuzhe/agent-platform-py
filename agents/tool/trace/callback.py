"""Tracing callback handler for LangChain/LangGraph agent runs.

Records every LLM call, tool invocation, and chain step so that
debugging and performance analysis can be performed after the fact.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class TraceRecord:
    """Lightweight container for a single trace event."""

    __slots__ = (
        "id",
        "parent_id",
        "event_type",
        "name",
        "start_time",
        "end_time",
        "input_data",
        "output_data",
        "metadata",
    )

    def __init__(
        self,
        event_type: str,
        name: str = "",
        parent_id: str | None = None,
    ) -> None:
        self.id: str = str(uuid.uuid4())
        self.parent_id: str | None = parent_id
        self.event_type: str = event_type
        self.name: str = name
        self.start_time: float = time.time()
        self.end_time: float | None = None
        self.input_data: Any = None
        self.output_data: Any = None
        self.metadata: Dict[str, Any] = {}

    def finish(self, output: Any = None) -> None:
        self.end_time = time.time()
        self.output_data = output

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return round((self.end_time - self.start_time) * 1000, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "event_type": self.event_type,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "input": self.input_data,
            "output": self.output_data,
            "metadata": self.metadata,
        }


class TraceCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that builds a full execution trace.

    Usage::

        tracer = TraceCallbackHandler(run_id="my-run")
        result = agent.invoke(input, config={"callbacks": [tracer]})
        records = tracer.get_records()
    """

    def __init__(self, run_id: str | None = None) -> None:
        super().__init__()
        self.run_id: str = run_id or str(uuid.uuid4())
        self._records: List[TraceRecord] = []
        self._stack: List[TraceRecord] = []  # for nesting

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_parent(self) -> str | None:
        return self._stack[-1].id if self._stack else None

    def _push(self, record: TraceRecord) -> None:
        self._records.append(record)
        self._stack.append(record)

    def _pop(self, record: TraceRecord | None = None) -> None:
        if record is not None:
            record.finish()
        if self._stack:
            self._stack.pop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_records(self) -> List[Dict[str, Any]]:
        """Return all collected trace records as dicts."""
        return [r.to_dict() for r in self._records]

    def get_records_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Return records filtered by *event_type*."""
        return [r.to_dict() for r in self._records if r.event_type == event_type]

    # ------------------------------------------------------------------
    # LLM callbacks
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", serialized.get("id", ["llm"])[-1] if isinstance(serialized.get("id"), list) else "llm")
        rec = TraceRecord("llm", name=name, parent_id=self._current_parent())
        rec.input_data = {"prompts": prompts}
        rec.metadata["run_id"] = str(run_id) if run_id else None
        self._push(rec)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if not self._stack:
            return
        rec = self._stack[-1]
        texts = []
        for gen_list in response.generations:
            for gen in gen_list:
                texts.append(gen.text)
        rec.finish(output={"generations": texts, "llm_output": response.llm_output})
        self._stack.pop()

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if self._stack:
            rec = self._stack.pop()
            rec.finish(output={"error": str(error)})

    # ------------------------------------------------------------------
    # Chain callbacks
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", serialized.get("id", ["chain"])[-1] if isinstance(serialized.get("id"), list) else "chain")
        rec = TraceRecord("chain", name=name, parent_id=self._current_parent())
        rec.input_data = inputs
        rec.metadata["run_id"] = str(run_id) if run_id else None
        self._push(rec)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if not self._stack:
            return
        rec = self._stack[-1]
        rec.finish(output=outputs)
        self._stack.pop()

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if self._stack:
            rec = self._stack.pop()
            rec.finish(output={"error": str(error)})

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "tool")
        rec = TraceRecord("tool", name=name, parent_id=self._current_parent())
        rec.input_data = input_str
        rec.metadata["run_id"] = str(run_id) if run_id else None
        self._push(rec)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if not self._stack:
            return
        rec = self._stack[-1]
        rec.finish(output=output)
        self._stack.pop()

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if self._stack:
            rec = self._stack.pop()
            rec.finish(output={"error": str(error)})

    # ------------------------------------------------------------------
    # Agent callbacks
    # ------------------------------------------------------------------

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        rec = TraceRecord("agent_action", name=getattr(action, "tool", "action"), parent_id=self._current_parent())
        rec.input_data = {
            "tool": getattr(action, "tool", ""),
            "tool_input": getattr(action, "tool_input", ""),
            "log": getattr(action, "log", ""),
        }
        self._push(rec)

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if not self._stack:
            return
        rec = self._stack[-1]
        rec.finish(output=getattr(finish, "return_values", {}))
        self._stack.pop()

    # ------------------------------------------------------------------
    # Retriever callbacks
    # ------------------------------------------------------------------

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        rec = TraceRecord("retriever", name=serialized.get("name", "retriever"), parent_id=self._current_parent())
        rec.input_data = query
        self._push(rec)

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if not self._stack:
            return
        rec = self._stack[-1]
        rec.finish(output=[getattr(d, "page_content", str(d))[:200] for d in documents])
        self._stack.pop()

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if self._stack:
            rec = self._stack.pop()
            rec.finish(output={"error": str(error)})
