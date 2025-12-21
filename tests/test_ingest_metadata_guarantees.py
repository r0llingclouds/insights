from __future__ import annotations

from pathlib import Path


def test_ingest_sets_title_description_summary_even_when_llm_fails(monkeypatch, tmp_path: Path) -> None:
    # Patch extraction to avoid network / heavy deps.
    from insights.ingest.pipeline import IngestBackend, ingest
    from insights.storage.db import Database
    from insights.storage.models import SourceKind

    monkeypatch.setattr(
        "insights.ingest.pipeline.extract_markdown_with_docling",
        lambda _url: "# Escaping the Rut\n\nThis post explains a practical system to break productivity ruts.\n",
    )
    # Force LLM-based generators to fail; ensure_* should fall back deterministically.
    monkeypatch.setattr("insights.title.generate_title", lambda **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("insights.describe.generate_description", lambda **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("insights.summarize.generate_summary", lambda **_kw: (_ for _ in ()).throw(RuntimeError("boom")))

    db = Database.open(tmp_path / "insights.db")
    try:
        res = ingest(
            db=db,
            input_value="https://lelouch.dev/blog/escaping-rut",
            cache_dir=tmp_path / "cache",
            forced_type="auto",
            url_backend=IngestBackend.DOCLING,
            refresh=False,
            title=None,
            summary_progress=None,
        )

        # Source should have non-empty title/description persisted.
        src = db.get_source_by_id(res.source.id)
        assert src is not None
        assert src.kind == SourceKind.URL
        assert (src.title or "").strip()
        assert (src.description or "").strip()

        # Version should have a non-empty summary persisted.
        sv = db.get_source_version_by_id(res.source_version.id)
        assert sv is not None
        assert (sv.summary or "").strip()
        # Paragraph summary (no bullet list).
        assert not sv.summary.strip().startswith("- ")
    finally:
        db.close()



