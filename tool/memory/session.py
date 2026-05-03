"""Pydantic models for session memory management."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in the conversation history."""

    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class Entity(BaseModel):
    """An entity extracted from conversation context."""

    name: str
    type: str
    attributes: dict = Field(default_factory=dict)
    last_update: datetime = Field(default_factory=datetime.now)


class Fact(BaseModel):
    """A factual piece of information extracted from conversation."""

    content: str
    source: str
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence: float = 1.0


class Session(BaseModel):
    """A conversation session with history, summary, and extracted knowledge."""

    id: str
    history: list[Message] = Field(default_factory=list)
    summary: str = ""
    entities: dict[str, Entity] = Field(default_factory=dict)
    facts: list[Fact] = Field(default_factory=list)
    preferences: dict[str, str] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.now)
