from __future__ import annotations

from typing import Any

from insights.storage.db import Database
from insights.storage.models import MessageRole


def _truncate_title(text: str, *, max_chars: int = 80) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max(0, max_chars - 1)].rstrip() + "…"


def save_one_shot_qa(
    db: Database,
    *,
    source_ids: list[str],
    question: str,
    answer: str,
    provider: str | None,
    model: str | None,
    usage: dict[str, Any] | None,
) -> str:
    """
    Persist a one-off Q&A as a new conversation, so it can be resumed later.

    Returns:
        conversation_id
    """
    conv = db.create_conversation(title=_truncate_title(question))
    conv_id = conv.id

    for sid in source_ids:
        db.bind_source_to_conversation(conversation_id=conv_id, source_id=sid)

    db.add_message(conversation_id=conv_id, role=MessageRole.USER, content=question)
    db.add_message(
        conversation_id=conv_id,
        role=MessageRole.ASSISTANT,
        content=answer,
        provider=provider,
        model=model,
        usage=usage,
    )
    return conv_id


