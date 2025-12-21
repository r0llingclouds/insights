from __future__ import annotations

from pathlib import Path

from insights.chat.save import save_one_shot_qa
from insights.storage.db import Database
from insights.storage.models import MessageRole, SourceKind


def test_save_one_shot_qa_creates_conversation_and_messages(tmp_path: Path) -> None:
    db = Database.open(tmp_path / "insights.db")
    try:
        s = db.upsert_source(kind=SourceKind.URL, locator="https://example.com/a", title="Example")
        conv_id = save_one_shot_qa(
            db,
            source_ids=[s.id],
            question="What are the key points?",
            answer="Here are the key points: ...",
            provider="openai",
            model="gpt-4o-mini",
            usage={"total_tokens": 10},
        )

        conv = db.get_conversation(conv_id)
        assert conv.id == conv_id
        assert conv.title is not None

        bound = db.list_conversation_sources(conv_id)
        assert [b.id for b in bound] == [s.id]

        msgs = db.list_messages(conversation_id=conv_id)
        assert len(msgs) == 2
        assert msgs[0].role == MessageRole.USER
        assert "key points" in msgs[0].content.lower()
        assert msgs[1].role == MessageRole.ASSISTANT
        assert "key points" in msgs[1].content.lower()
        assert msgs[1].provider == "openai"
        assert msgs[1].model == "gpt-4o-mini"
        assert isinstance(msgs[1].usage, dict)
    finally:
        db.close()


