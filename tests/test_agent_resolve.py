from __future__ import annotations

from pathlib import Path

from insights.agent.resolve import resolve_source
from insights.ingest.detect import detect_source
from insights.storage.db import Database
from insights.storage.models import SourceKind


def test_resolve_by_id_and_url_and_basename(tmp_path: Path) -> None:
    db = Database.open(tmp_path / "insights.db")
    try:
        # Two file sources with same basename -> ambiguous
        s1 = db.upsert_source(kind=SourceKind.FILE, locator="/a/b/onepager.pdf", title=None)
        s2 = db.upsert_source(kind=SourceKind.FILE, locator="/c/d/onepager.pdf", title=None)

        # URL source
        detected = detect_source("https://example.com/article", forced_type="auto")
        s3 = db.upsert_source(kind=detected.kind, locator=detected.locator, title="Example Article")

        r_id = resolve_source(db=db, ref=s1.id, suggestions_limit=5)
        assert r_id.found and r_id.source is not None
        assert r_id.source.id == s1.id

        r_url = resolve_source(db=db, ref="https://example.com/article", suggestions_limit=5)
        assert r_url.found and r_url.source is not None
        assert r_url.source.id == s3.id

        r_base = resolve_source(db=db, ref="onepager.pdf", suggestions_limit=5)
        assert not r_base.found
        assert r_base.ambiguous
        assert len(r_base.suggestions) == 2
        assert {s.id for s in r_base.suggestions} == {s1.id, s2.id}
    finally:
        db.close()


def test_resolve_single_basename(tmp_path: Path) -> None:
    db = Database.open(tmp_path / "insights.db")
    try:
        s1 = db.upsert_source(kind=SourceKind.FILE, locator="/a/b/onepager.pdf", title=None)
        r = resolve_source(db=db, ref="onepager.pdf", suggestions_limit=5)
        assert r.found and r.source is not None
        assert r.source.id == s1.id
    finally:
        db.close()


