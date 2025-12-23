from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import re
from typing import Any

from insights.config import Paths
from insights.ingest import IngestBackend, ingest as ingest_source
from insights.storage.db import Database
from insights.storage.models import SourceKind
from insights.utils.progress import ProgressFn


class ExportFormat(str, Enum):
    MARKDOWN = "md"
    PLAIN = "txt"
    JSON = "json"
    HTML = "html"


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
    name: str | None = None,
    include_plain: bool = False,
    include_markdown: bool = True,
    progress: ProgressFn | None = None,
) -> list[Path]:
    """
    Resolve/ingest a source and write its cached plain_text (.txt) and markdown (.md) to files.

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
        # Ingest if missing, or refresh if requested.
        if source is None or refresh:
            if source is None:
                input_value = ref
            else:
                if source.kind == SourceKind.YOUTUBE:
                    input_value = f"https://www.youtube.com/watch?v={source.locator}"
                else:
                    input_value = source.locator
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

        base = name
        if not base:
            if source.title:
                base = source.title
            else:
                if source.kind == SourceKind.YOUTUBE:
                    base = f"youtube_{source.locator}"
                elif source.kind == SourceKind.URL:
                    base = "url_" + re.sub(r"^https?://", "", source.locator).replace("/", "_")
                else:
                    base = Path(source.locator).name

        stem = f"{safe_filename_base(base)}__{source.id}"

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


def _build_export_metadata(
    *,
    source: Any,
    doc: dict[str, Any],
    plain_text: str,
    markdown: str,
) -> dict[str, Any]:
    """Build metadata dict for JSON/HTML export."""
    return {
        "id": source.id,
        "kind": source.kind.value,
        "locator": source.locator,
        "title": source.title,
        "description": source.description,
        "created_at": source.created_at.isoformat() if source.created_at else None,
        "updated_at": source.updated_at.isoformat() if source.updated_at else None,
        "extractor": doc.get("extractor"),
        "char_count": len(plain_text),
        "token_count": doc.get("token_count"),
        "exported_at": datetime.now().isoformat(),
    }


def export_source_json(
    *,
    paths: Paths,
    source_ref: str,
    out_dir: Path | None = None,
    out_file: Path | None = None,
    backend: IngestBackend = IngestBackend.DOCLING,
    refresh: bool = False,
    name: str | None = None,
    progress: ProgressFn | None = None,
) -> Path:
    """Export a source as JSON with full metadata."""
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
        if source is None or refresh:
            if source is None:
                input_value = ref
            else:
                if source.kind == SourceKind.YOUTUBE:
                    input_value = f"https://www.youtube.com/watch?v={source.locator}"
                else:
                    input_value = source.locator
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

        base = name
        if not base:
            if source.title:
                base = source.title
            else:
                if source.kind == SourceKind.YOUTUBE:
                    base = f"youtube_{source.locator}"
                elif source.kind == SourceKind.URL:
                    base = "url_" + re.sub(r"^https?://", "", source.locator).replace("/", "_")
                else:
                    base = Path(source.locator).name

        stem = f"{safe_filename_base(base)}__{source.id}"

        metadata = _build_export_metadata(source=source, doc=doc, plain_text=plain_text, markdown=markdown)
        export_data = {
            "metadata": metadata,
            "content": {
                "plain_text": plain_text,
                "markdown": markdown,
            },
        }

        json_path = out_file.expanduser().resolve() if out_file else (out_dir_resolved / f"{stem}.json")
        if progress is not None:
            progress(f"writing export: {json_path}")
        json_path.write_text(json.dumps(export_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return json_path
    finally:
        db.close()


def export_source_html(
    *,
    paths: Paths,
    source_ref: str,
    out_dir: Path | None = None,
    out_file: Path | None = None,
    backend: IngestBackend = IngestBackend.DOCLING,
    refresh: bool = False,
    name: str | None = None,
    progress: ProgressFn | None = None,
) -> Path:
    """Export a source as styled HTML."""
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
        if source is None or refresh:
            if source is None:
                input_value = ref
            else:
                if source.kind == SourceKind.YOUTUBE:
                    input_value = f"https://www.youtube.com/watch?v={source.locator}"
                else:
                    input_value = source.locator
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

        base = name
        if not base:
            if source.title:
                base = source.title
            else:
                if source.kind == SourceKind.YOUTUBE:
                    base = f"youtube_{source.locator}"
                elif source.kind == SourceKind.URL:
                    base = "url_" + re.sub(r"^https?://", "", source.locator).replace("/", "_")
                else:
                    base = Path(source.locator).name

        stem = f"{safe_filename_base(base)}__{source.id}"
        title = source.title or base

        # Use markdown content if available, otherwise plain text
        content = markdown or plain_text
        # Escape HTML in content
        content_escaped = (
            content.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>\n")
        )

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            color: #333;
        }}
        .metadata {{
            background: #f5f5f5;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            font-size: 0.9rem;
        }}
        .metadata dt {{
            font-weight: bold;
            color: #666;
        }}
        .metadata dd {{
            margin: 0 0 0.5rem 0;
        }}
        .content {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        h1 {{
            border-bottom: 2px solid #eee;
            padding-bottom: 0.5rem;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="metadata">
        <dl>
            <dt>Source ID</dt><dd>{source.id}</dd>
            <dt>Type</dt><dd>{source.kind.value}</dd>
            <dt>Locator</dt><dd>{source.locator}</dd>
            <dt>Characters</dt><dd>{len(plain_text):,}</dd>
            <dt>Exported</dt><dd>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</dd>
        </dl>
    </div>
    <div class="content">{content_escaped}</div>
</body>
</html>
"""

        html_path = out_file.expanduser().resolve() if out_file else (out_dir_resolved / f"{stem}.html")
        if progress is not None:
            progress(f"writing export: {html_path}")
        html_path.write_text(html_content, encoding="utf-8")
        return html_path
    finally:
        db.close()


