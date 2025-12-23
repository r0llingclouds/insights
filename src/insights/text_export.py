from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from insights.config import Paths
from insights.ingest import IngestBackend, extract_ephemeral, ingest as ingest_source
from insights.storage.db import Database
from insights.storage.models import SourceKind
from insights.utils.progress import ProgressFn


def default_downloads_dir() -> Path:
    return (Path.home() / "Downloads").expanduser().resolve()


def safe_filename_base(value: str) -> str:
    # Keep it portable: letters, numbers, space, dash, underscore, dot.
    collapsed = " ".join((value or "").split()).strip()
    out = []
    for ch in collapsed:
        if ch.isalnum() or ch in {" ", "-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip().replace(" ", "_")
    return s or "source"


@dataclass(frozen=True, slots=True)
class ExportSuggestion:
    id: str
    kind: str
    title: str | None
    locator: str


class AmbiguousSourceRefError(RuntimeError):
    def __init__(self, ref: str, suggestions: list[ExportSuggestion]):
        super().__init__(f"Ambiguous source reference: {ref}")
        self.ref = ref
        self.suggestions = suggestions


def export_source_text(
    *,
    paths: Paths,
    source_ref: str,
    out_dir: Path | None = None,
    out_file: Path | None = None,
    backend: IngestBackend = IngestBackend.DOCLING,
    refresh: bool = False,
    no_store: bool = False,
    name: str | None = None,
    include_plain: bool = False,
    include_markdown: bool = True,
    progress: ProgressFn | None = None,
) -> list[Path]:
    """
    Resolve/ingest a source and write its cached plain_text (.txt) and markdown (.md) to files.

    Args:
        no_store: Don't persist source to DB (reuse cache if exists).

    Returns:
        List of written file paths.
    """
    from insights.agent.resolve import resolve_source as resolve_source_any

    ref = (source_ref or "").strip()
    if not ref:
        raise ValueError("source_ref cannot be empty")

    out_dir_resolved = (out_dir or default_downloads_dir()).expanduser().resolve()
    out_dir_resolved.mkdir(parents=True, exist_ok=True)

    db = Database.open(paths.db_path)
    try:
        resolved = resolve_source_any(db=db, ref=ref, suggestions_limit=5)
        source = resolved.source if (resolved.found and resolved.source) else None

        if resolved.ambiguous:
            raise AmbiguousSourceRefError(
                ref,
                [
                    ExportSuggestion(id=s.id, kind=s.kind.value, title=s.title, locator=s.locator)
                    for s in resolved.suggestions
                ],
            )

        source_version_id: str | None = None
        plain_text: str = ""
        markdown: str = ""
        source_title: str | None = None
        source_locator: str = ref
        source_kind: SourceKind | None = None
        source_id: str | None = None

        # Ingest/extract if missing, or refresh if requested.
        if source is None or refresh:
            if source is None:
                input_value = ref
            else:
                if source.kind == SourceKind.YOUTUBE:
                    input_value = f"https://www.youtube.com/watch?v={source.locator}"
                else:
                    input_value = source.locator

            if no_store and source is None:
                # Ephemeral mode: extract without storing (reuse cache if exists).
                result = extract_ephemeral(
                    db=db,
                    input_value=input_value,
                    cache_dir=paths.cache_dir,
                    forced_type="auto",
                    url_backend=backend,
                    progress=progress,
                )
                if result.from_cache:
                    # Source exists in DB - use it.
                    assert result.source is not None
                    source = result.source
                    source_version_id = result.source_version.id if result.source_version else None
                else:
                    # Fresh extraction - use ephemeral document directly.
                    assert result.document is not None
                    plain_text = result.document.plain_text.strip()
                    markdown = result.document.markdown.strip()
                    source_title = result.document.title
                    source_locator = result.document.locator
                    source_kind = result.document.kind
                    # No source_id for ephemeral docs.
            else:
                # Normal mode: ingest and store.
                result = ingest_source(
                    db=db,
                    input_value=input_value,
                    cache_dir=paths.cache_dir,
                    forced_type="auto",
                    url_backend=backend,
                    refresh=bool(refresh),
                    title=None,
                    summary_progress=progress,
                )
                source = result.source
                source_version_id = result.source_version.id

        # If we have a source (from cache or ingest), get the document from DB.
        if source is not None:
            source_title = source.title
            source_locator = source.locator
            source_kind = source.kind
            source_id = source.id

            # Decide extractor preference.
            if source.kind == SourceKind.YOUTUBE:
                extractor_preference = ["assemblyai"]
            elif source.kind == SourceKind.URL:
                extractor_preference = [backend.value, "firecrawl" if backend.value != "firecrawl" else "docling"]
            else:
                extractor_preference = ["docling"]

            doc = db.get_latest_document_for_source(source_id=source.id, extractor_preference=extractor_preference)
            if not doc and source_version_id:
                plain = db.get_document_plain_text_by_source_version(source_version_id)
                md = db.get_document_markdown_by_source_version(source_version_id)
                doc = {"source_version_id": source_version_id, "plain_text": plain, "markdown": md, "extractor": None}
            if not doc:
                raise RuntimeError("No cached document found for this source. Try re-ingesting with --refresh.")

            plain_text = (doc.get("plain_text") or "").strip()
            markdown = (doc.get("markdown") or "").strip()

        # Build the filename.
        base = name
        if not base:
            if source_title:
                base = source_title
            else:
                if source_kind == SourceKind.YOUTUBE:
                    base = f"youtube_{source_locator}"
                elif source_kind == SourceKind.URL:
                    base = "url_" + re.sub(r"^https?://", "", source_locator).replace("/", "_")
                else:
                    base = Path(source_locator).name

        # For ephemeral docs, use "ephemeral" as the id suffix.
        stem = f"{safe_filename_base(base)}__{source_id or 'ephemeral'}"

        written: list[Path] = []
        if include_markdown:
            md_path = (out_file.expanduser().resolve() if out_file else (out_dir_resolved / f"{stem}.md"))
            # Always write a markdown file; if markdown is empty, fall back to plain text/transcript.
            md_content = markdown or plain_text
            if progress is not None:
                progress(f"writing export: {md_path}")
            md_path.write_text(md_content + ("\n" if md_content and not md_content.endswith("\n") else ""), encoding="utf-8")
            written.append(md_path)

        if include_plain:
            txt_path = out_dir_resolved / f"{stem}.txt"
            if progress is not None:
                progress(f"writing export: {txt_path}")
            txt_path.write_text(
                plain_text + ("\n" if plain_text and not plain_text.endswith("\n") else ""),
                encoding="utf-8",
            )
            written.append(txt_path)

        return written
    finally:
        db.close()


