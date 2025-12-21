from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class SourceKind(StrEnum):
    FILE = "file"
    URL = "url"
    YOUTUBE = "youtube"


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    kind: SourceKind
    locator: str
    title: str | None
    description: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class SourceVersion:
    id: str
    source_id: str
    content_hash: str
    extracted_at: datetime
    extractor: str
    status: str
    error: str | None


@dataclass(frozen=True, slots=True)
class Conversation:
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Message:
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    created_at: datetime
    provider: str | None
    model: str | None
    usage: dict[str, Any] | None


