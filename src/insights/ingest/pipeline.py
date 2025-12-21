from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable

from insights.ingest.detect import DetectedSource, detect_source
from insights.ingest.docling_extractor import extract_markdown_with_docling
from insights.ingest.firecrawl_extractor import extract_markdown_with_firecrawl
from insights.ingest.youtube_extractor import download_youtube_audio, transcribe_with_assemblyai
from insights.storage.db import Database
from insights.storage.models import Source, SourceKind, SourceVersion
from insights.utils.hashing import content_hash_for_file, sha256_file, sha256_text
from insights.utils.text import markdown_to_text
from insights.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)


def _require_description(*, db: Database, source_id: str, source_version_id: str | None, force: bool) -> None:
    """
    Ensure sources.description is populated as part of ingestion (critical-path).
    """
    from insights.describe import ensure_source_description

    desc = ensure_source_description(
        db=db,
        source_id=source_id,
        source_version_id=source_version_id,
        force=force,
    )
    if not desc:
        raise RuntimeError("Description generation failed (sources.description is empty)")


class IngestBackend(StrEnum):
    DOCLING = "docling"
    FIRECRAWL = "firecrawl"  # implemented in a later milestone/todo


@dataclass(frozen=True, slots=True)
class IngestResult:
    source: Source
    source_version: SourceVersion
    document_id: str
    reused_cache: bool


def ingest(
    *,
    db: Database,
    input_value: str,
    cache_dir: Path | None = None,
    forced_type: str = "auto",
    url_backend: IngestBackend = IngestBackend.DOCLING,
    refresh: bool = False,
    title: str | None = None,
    summary_progress: Callable[[str], None] | None = None,
) -> IngestResult:
    detected = detect_source(input_value, forced_type=forced_type)
    if detected.kind == SourceKind.FILE:
        return _ingest_file(
            db=db,
            detected=detected,
            refresh=refresh,
            title=title or detected.display_title,
            summary_progress=summary_progress,
        )
    if detected.kind == SourceKind.URL:
        if url_backend == IngestBackend.DOCLING:
            return _ingest_url_docling(
                db=db,
                detected=detected,
                refresh=refresh,
                title=title,
                summary_progress=summary_progress,
            )
        if url_backend == IngestBackend.FIRECRAWL:
            return _ingest_url_firecrawl(
                db=db,
                detected=detected,
                refresh=refresh,
                title=title,
                summary_progress=summary_progress,
            )
        raise ValueError(f"Unsupported URL backend: {url_backend}")
    if detected.kind == SourceKind.YOUTUBE:
        if cache_dir is None:
            raise ValueError("cache_dir is required for YouTube ingestion")
        return _ingest_youtube_assemblyai(
            db=db,
            detected=detected,
            refresh=refresh,
            title=title,
            cache_dir=cache_dir,
            summary_progress=summary_progress,
        )
    raise ValueError(f"Unsupported source kind: {detected.kind.value}")


def _ingest_file(
    *,
    db: Database,
    detected: DetectedSource,
    refresh: bool,
    title: str | None,
    summary_progress: Callable[[str], None] | None,
) -> IngestResult:
    path = Path(detected.locator)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    source = db.upsert_source(kind=SourceKind.FILE, locator=str(path), title=title)
    extractor = "docling"
    content_hash = content_hash_for_file(path)

    if not refresh:
        try:
            existing = db.get_source_version(
                source_id=source.id,
                content_hash=content_hash,
                extractor=extractor,
            )
        except KeyError:
            existing = None
        if existing and existing.status == "ok":
            doc_id = db.get_document_id_by_source_version(existing.id)
            if doc_id:
                _require_description(db=db, source_id=source.id, source_version_id=existing.id, force=False)
                return IngestResult(
                    source=source,
                    source_version=existing,
                    document_id=doc_id,
                    reused_cache=True,
                )

    try:
        markdown = extract_markdown_with_docling(str(path))
        plain = markdown_to_text(markdown)
        token_est = estimate_tokens(plain)
        version = db.create_source_version(
            source_id=source.id,
            content_hash=content_hash,
            extractor=extractor,
            status="ok",
            error=None,
        )
        doc_id = db.upsert_document(
            source_version_id=version.id,
            markdown=markdown,
            plain_text=plain,
            token_estimate=token_est,
        )
        # Best-effort per-version summary for faster recall/semantic matching.
        try:
            if not version.summary:
                from insights.summarize import generate_summary

                summary = generate_summary(content=plain, progress=summary_progress)
                if summary:
                    db.set_source_version_summary(source_version_id=version.id, summary=summary)
        except Exception:
            pass
        _require_description(db=db, source_id=source.id, source_version_id=version.id, force=bool(refresh))
        return IngestResult(source=source, source_version=version, document_id=doc_id, reused_cache=False)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.exception("Failed to ingest file via docling")
        version = db.create_source_version(
            source_id=source.id,
            content_hash=content_hash,
            extractor=extractor,
            status="error",
            error=msg,
        )
        raise RuntimeError(msg) from e


def _ingest_url_docling(
    *,
    db: Database,
    detected: DetectedSource,
    refresh: bool,
    title: str | None,
    summary_progress: Callable[[str], None] | None,
) -> IngestResult:
    url = detected.locator
    source = db.upsert_source(kind=SourceKind.URL, locator=url, title=title)
    extractor = "docling"

    if not refresh:
        cached = _reuse_any_cached_document(
            db=db,
            source=source,
            extractor_preference=[extractor, "firecrawl"],
        )
        if cached is not None:
            _require_description(db=db, source_id=cached.source.id, source_version_id=cached.source_version.id, force=False)
            return cached

    try:
        markdown = extract_markdown_with_docling(url)
        plain = markdown_to_text(markdown)
        content_hash = sha256_text(markdown)
        token_est = estimate_tokens(plain)
        version = db.create_source_version(
            source_id=source.id,
            content_hash=content_hash,
            extractor=extractor,
            status="ok",
            error=None,
        )
        doc_id = db.upsert_document(
            source_version_id=version.id,
            markdown=markdown,
            plain_text=plain,
            token_estimate=token_est,
        )
        try:
            if not version.summary:
                from insights.summarize import generate_summary

                summary = generate_summary(content=plain, progress=summary_progress)
                if summary:
                    db.set_source_version_summary(source_version_id=version.id, summary=summary)
        except Exception:
            pass
        _require_description(db=db, source_id=source.id, source_version_id=version.id, force=bool(refresh))
        return IngestResult(source=source, source_version=version, document_id=doc_id, reused_cache=False)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.exception("Failed to ingest URL via docling")
        version = db.create_source_version(
            source_id=source.id,
            content_hash=sha256_text(url),
            extractor=extractor,
            status="error",
            error=msg,
        )
        raise RuntimeError(msg) from e


def _ingest_url_firecrawl(
    *,
    db: Database,
    detected: DetectedSource,
    refresh: bool,
    title: str | None,
    summary_progress: Callable[[str], None] | None,
) -> IngestResult:
    url = detected.locator
    source = db.upsert_source(kind=SourceKind.URL, locator=url, title=title)
    extractor = "firecrawl"

    if not refresh:
        cached = _reuse_any_cached_document(
            db=db,
            source=source,
            extractor_preference=[extractor, "docling"],
        )
        if cached is not None:
            _require_description(db=db, source_id=cached.source.id, source_version_id=cached.source_version.id, force=False)
            return cached

    try:
        markdown = extract_markdown_with_firecrawl(url)
        plain = markdown_to_text(markdown)
        content_hash = sha256_text(markdown)
        token_est = estimate_tokens(plain)
        version = db.create_source_version(
            source_id=source.id,
            content_hash=content_hash,
            extractor=extractor,
            status="ok",
            error=None,
        )
        doc_id = db.upsert_document(
            source_version_id=version.id,
            markdown=markdown,
            plain_text=plain,
            token_estimate=token_est,
        )
        try:
            if not version.summary:
                from insights.summarize import generate_summary

                summary = generate_summary(content=plain, progress=summary_progress)
                if summary:
                    db.set_source_version_summary(source_version_id=version.id, summary=summary)
        except Exception:
            pass
        _require_description(db=db, source_id=source.id, source_version_id=version.id, force=bool(refresh))
        return IngestResult(source=source, source_version=version, document_id=doc_id, reused_cache=False)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.exception("Failed to ingest URL via firecrawl")
        db.create_source_version(
            source_id=source.id,
            content_hash=sha256_text(url),
            extractor=extractor,
            status="error",
            error=msg,
        )
        raise RuntimeError(msg) from e


def _reuse_any_cached_document(
    *,
    db: Database,
    source: Source,
    extractor_preference: list[str],
) -> IngestResult | None:
    docs = db.get_documents_for_sources_latest(
        source_ids=[source.id],
        extractor_preference=extractor_preference,
    )
    if not docs:
        return None
    d = docs[0]
    version = db.get_source_version_by_id(str(d["source_version_id"]))
    if not version:
        return None
    return IngestResult(
        source=source,
        source_version=version,
        document_id=str(d["document_id"]),
        reused_cache=True,
    )


def _ingest_youtube_assemblyai(
    *,
    db: Database,
    detected: DetectedSource,
    refresh: bool,
    title: str | None,
    cache_dir: Path,
    summary_progress: Callable[[str], None] | None,
) -> IngestResult:
    video_id = detected.locator
    source = db.upsert_source(kind=SourceKind.YOUTUBE, locator=video_id, title=title)
    extractor = "assemblyai"

    if not refresh:
        existing = db.get_latest_source_version(source_id=source.id, extractor=extractor)
        if existing and existing.status == "ok":
            doc_id = db.get_document_id_by_source_version(existing.id)
            if doc_id:
                _require_description(db=db, source_id=source.id, source_version_id=existing.id, force=False)
                return IngestResult(
                    source=source,
                    source_version=existing,
                    document_id=doc_id,
                    reused_cache=True,
                )

    try:
        audio_path, yt_title = download_youtube_audio(video_id=video_id, cache_dir=cache_dir, refresh=refresh)
        if not source.title and yt_title:
            source = db.upsert_source(kind=SourceKind.YOUTUBE, locator=video_id, title=yt_title)

        audio_hash = sha256_file(audio_path)
        text = transcribe_with_assemblyai(audio_path=audio_path)
        markdown = text.strip()
        plain = markdown
        token_est = estimate_tokens(plain)

        version = db.create_source_version(
            source_id=source.id,
            content_hash=audio_hash,
            extractor=extractor,
            status="ok",
            error=None,
        )
        doc_id = db.upsert_document(
            source_version_id=version.id,
            markdown=markdown,
            plain_text=plain,
            token_estimate=token_est,
        )
        try:
            if not version.summary:
                from insights.summarize import generate_summary

                summary = generate_summary(content=plain, progress=summary_progress)
                if summary:
                    db.set_source_version_summary(source_version_id=version.id, summary=summary)
        except Exception:
            pass
        _require_description(db=db, source_id=source.id, source_version_id=version.id, force=bool(refresh))
        return IngestResult(source=source, source_version=version, document_id=doc_id, reused_cache=False)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.exception("Failed to ingest YouTube via assemblyai")
        db.create_source_version(
            source_id=source.id,
            content_hash=sha256_text(video_id),
            extractor=extractor,
            status="error",
            error=msg,
        )
        raise RuntimeError(msg) from e


