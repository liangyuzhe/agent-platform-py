"""Knowledge extraction from conversation messages."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from agents.tool.memory.session import Entity, Fact, Message, Session

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = """你是一个知识提取助手。请从以下对话消息中提取结构化信息。

要求：
1. 提取提到的实体（人名、组织、产品、概念等）
2. 提取明确陈述的事实
3. 提取用户表达的偏好

请严格以JSON格式返回，格式如下：
{{
  "entities": [
    {{"name": "实体名", "type": "类型", "attributes": {{}}}}
  ],
  "facts": [
    {{"content": "事实内容", "confidence": 0.9}}
  ],
  "preferences": {{
    "偏好类别": "偏好内容"
  }}
}}

如果某类信息没有提取到，返回空数组或空对象。

对话消息：
{messages}

JSON结果："""


def _format_messages(messages: list[Message]) -> str:
    """Format messages into a readable text block."""
    lines = []
    for m in messages:
        lines.append(f"[{m.role}]: {m.content}")
    return "\n".join(lines)


def _parse_extraction(raw: str) -> dict:
    """Parse the LLM JSON response, handling markdown code blocks."""
    text = raw.strip()
    if text.startswith("```"):
        # Strip markdown code fences
        lines = text.split("\n")
        lines = [
            l for l in lines if not l.strip().startswith("```")
        ]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse extraction JSON: %s", text[:200])
        return {}


async def extract_knowledge(
    session: Session,
    new_messages: list[Message],
    llm: BaseChatModel,
) -> None:
    """Extract entities, facts, and preferences from new messages.

    Uses the configured LLM to analyze messages and update the session's
    knowledge store (entities, facts, preferences).

    Args:
        session: The session to update with extracted knowledge.
        new_messages: New messages to analyze.
        llm: A LangChain chat model instance for extraction.
    """
    if not new_messages:
        return

    messages_text = _format_messages(new_messages)
    prompt = EXTRACT_PROMPT.format(messages=messages_text)

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        data = _parse_extraction(response.content)
    except Exception as e:
        logger.error("Knowledge extraction failed for session %s: %s", session.id, e)
        return

    now = datetime.now()

    # Merge extracted entities
    for ent_data in data.get("entities", []):
        name = ent_data.get("name", "").strip()
        if not name:
            continue
        etype = ent_data.get("type", "unknown")
        attrs = ent_data.get("attributes", {})

        if name in session.entities:
            # Update existing entity
            existing = session.entities[name]
            existing.attributes.update(attrs)
            existing.last_update = now
            if etype != "unknown":
                existing.type = etype
        else:
            session.entities[name] = Entity(
                name=name,
                type=etype,
                attributes=attrs,
                last_update=now,
            )

    # Append extracted facts
    for fact_data in data.get("facts", []):
        content = fact_data.get("content", "").strip()
        if not content:
            continue
        confidence = float(fact_data.get("confidence", 0.8))
        session.facts.append(
            Fact(
                content=content,
                source="extraction",
                timestamp=now,
                confidence=min(max(confidence, 0.0), 1.0),
            )
        )

    # Merge extracted preferences
    for key, value in data.get("preferences", {}).items():
        key = key.strip()
        if key and value:
            session.preferences[key] = str(value).strip()

    logger.info(
        "Extracted knowledge for session %s: %d entities, %d facts, %d preferences",
        session.id,
        len(data.get("entities", [])),
        len(data.get("facts", [])),
        len(data.get("preferences", {})),
    )
