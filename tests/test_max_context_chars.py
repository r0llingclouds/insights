from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from insights.retrieval import build_context
from insights.storage.db import Database
from insights.storage.models import SourceKind
from insights.utils.tokens import estimate_tokens


def _setup_one_doc(tmp_path: Path, *, plain_text: str) -> tuple[Database, str]:
    db = Database.open(tmp_path / "insights.db")
    s = db.upsert_source(kind=SourceKind.FILE, locator=str(tmp_path / "a.txt"), title="a")
    sv = db.create_source_version(
        source_id=s.id,
        content_hash="deadbeef",
        extractor="docling",
        status="ok",
        error=None,
    )
    db.upsert_document(
        source_version_id=sv.id,
        markdown=plain_text,
        plain_text=plain_text,
        token_count=estimate_tokens(plain_text),
    )
    return db, s.id


def test_build_context_full_respects_max_context_chars(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INSIGHTS_MAX_CONTEXT_CHARS", "1000")

    long_text = "A" * 1500
    db, source_id = _setup_one_doc(tmp_path, plain_text=long_text)
    try:
        out = build_context(
            db=db,
            source_ids=[source_id],
            question="what is this",
            max_context_tokens=10**9,
        )
        # Header + newline + trimmed body.
        body = out.context_text.split("\n", 1)[1]
        assert body == ("A" * 1000)
    finally:
        db.close()


def test_build_context_token_trims_to_budget(monkeypatch, tmp_path: Path) -> None:
    # With FULL-only context, the final assembled context should be token-trimmed to max_context_tokens.
    monkeypatch.setenv("INSIGHTS_MAX_CONTEXT_CHARS", "100000")

    # Large repetitive text to guarantee we exceed the budget.
    text = ("hello world " * 10000).strip()
    db, source_id = _setup_one_doc(tmp_path, plain_text=text)
    try:
        out = build_context(
            db=db,
            source_ids=[source_id],
            question="what is this",
            max_context_tokens=200,
        )
        assert estimate_tokens(out.context_text) <= 200
    finally:
        db.close()


