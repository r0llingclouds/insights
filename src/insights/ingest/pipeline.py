from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

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
) -> IngestResult:
    detected = detect_source(input_value, forced_type=forced_type)
    if detected.kind == SourceKind.FILE:
        return _ingest_file(
            db=db,
            detected=detected,
            refresh=refresh,
            title=title or detected.display_title,
        )
    if detected.kind == SourceKind.URL:
        if url_backend == IngestBackend.DOCLING:
            return _ingest_url_docling(
                db=db,
                detected=detected,
                refresh=refresh,
                title=title,
            )
        if url_backend == IngestBackend.FIRECRAWL:
            return _ingest_url_firecrawl(
                db=db,
                detected=detected,
                refresh=refresh,
                title=title,
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
        )
    raise ValueError(f"Unsupported source kind: {detected.kind.value}")


def _ingest_file(*, db: Database, detected: DetectedSource, refresh: bool, title: str | None) -> IngestResult:
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
) -> IngestResult:
    url = detected.locator
    source = db.upsert_source(kind=SourceKind.URL, locator=url, title=title)
    extractor = "docling"

    if not refresh:
        existing = db.get_latest_source_version(source_id=source.id, extractor=extractor)
        if existing and existing.status == "ok":
            doc_id = db.get_document_id_by_source_version(existing.id)
            if doc_id:
                return IngestResult(
                    source=source,
                    source_version=existing,
                    document_id=doc_id,
                    reused_cache=True,
                )

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
) -> IngestResult:
    url = detected.locator
    source = db.upsert_source(kind=SourceKind.URL, locator=url, title=title)
    extractor = "firecrawl"

    if not refresh:
        existing = db.get_latest_source_version(source_id=source.id, extractor=extractor)
        if existing and existing.status == "ok":
            doc_id = db.get_document_id_by_source_version(existing.id)
            if doc_id:
                return IngestResult(
                    source=source,
                    source_version=existing,
                    document_id=doc_id,
                    reused_cache=True,
                )

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


def _ingest_youtube_assemblyai(
    *,
    db: Database,
    detected: DetectedSource,
    refresh: bool,
    title: str | None,
    cache_dir: Path,
) -> IngestResult:
    video_id = detected.locator
    source = db.upsert_source(kind=SourceKind.YOUTUBE, locator=video_id, title=title)
    extractor = "assemblyai"

    if not refresh:
        existing = db.get_latest_source_version(source_id=source.id, extractor=extractor)
        if existing and existing.status == "ok":
            doc_id = db.get_document_id_by_source_version(existing.id)
            if doc_id:
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


