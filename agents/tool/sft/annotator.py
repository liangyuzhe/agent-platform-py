"""Teacher-model annotation for SFT samples.

Uses DeepSeek as the teacher model to evaluate and enrich raw
prompt-completion pairs before they are exported for fine-tuning.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from agents.config.settings import settings

logger = logging.getLogger(__name__)

# Annotation prompt template.
_ANNOTATION_PROMPT = """\
You are a senior AI trainer. Given the following prompt-completion pair,
produce a JSON object with these keys:

- "score" (float 0-1): quality of the completion given the prompt.
- "strengths" (list[str]): what the completion does well.
- "weaknesses" (list[str]): what could be improved.
- "revised_completion" (str): a better version of the completion, or the
  original if it is already excellent. Keep the same format and style.

Respond with **only** the JSON object, no extra text.

---
PROMPT:
{prompt}

COMPLETION:
{completion}
"""


def _build_teacher_llm() -> ChatOpenAI:
    """Construct a ChatOpenAI client pointing at the DeepSeek API."""
    cfg = settings.deepseek
    return ChatOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.chat_model or "deepseek-chat",
        temperature=0.3,
        max_tokens=4096,
    )


def annotate(sample: dict[str, Any]) -> None:
    """Annotate *sample* in-place with teacher-model feedback.

    The function adds ``"annotation"`` and ``"annotation_model"`` keys
    to *sample*.  On failure the keys are set to ``None`` so the sample
    is still usable.

    Parameters
    ----------
    sample:
        A dict containing at least ``"prompt"`` and ``"completion"``.
        Modified in-place.
    """
    prompt = sample.get("prompt", "")
    completion = sample.get("completion", "")

    if not prompt or not completion:
        logger.warning("Skipping annotation: empty prompt or completion.")
        sample["annotation"] = None
        sample["annotation_model"] = None
        return

    llm = _build_teacher_llm()
    message = _ANNOTATION_PROMPT.format(prompt=prompt, completion=completion)

    try:
        response = llm.invoke(message)
        raw_text = response.content.strip()
        # Try to parse the annotation as JSON.
        annotation = json.loads(raw_text)
        sample["annotation"] = annotation
        sample["annotation_model"] = "deepseek"
    except json.JSONDecodeError:
        # The model may wrap JSON in markdown fences.
        try:
            cleaned = raw_text
            if "```" in cleaned:
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            annotation = json.loads(cleaned.strip())
            sample["annotation"] = annotation
            sample["annotation_model"] = "deepseek"
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse annotation JSON from teacher model.")
            sample["annotation"] = {"raw": raw_text}
            sample["annotation_model"] = "deepseek"
    except Exception:
        logger.exception("Unexpected error during annotation.")
        sample["annotation"] = None
        sample["annotation_model"] = None
