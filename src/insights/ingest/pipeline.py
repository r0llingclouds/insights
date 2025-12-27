from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable

from insights.ingest.detect import DetectedSource, detect_source
from insights.ingest.docling_extractor import extract_markdown_with_docling
from insights.ingest.firecrawl_extractor import extract_markdown_with_firecrawl
from insights.ingest.tweet_extractor import fetch_tweet
from insights.ingest.youtube_extractor import download_youtube_audio, transcribe_with_assemblyai
from insights.storage.db import Database
from insights.storage.models import Source, SourceKind, SourceVersion
from insights.utils.hashing import content_hash_for_file, sha256_file, sha256_text
from insights.utils.text import markdown_to_text
from insights.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)


def _p(progress: Callable[[str], None] | None, msg: str) -> None:
    if progress is None:
        return
    try:
        progress(msg)
    except Exception:
        return


def _require_description(
    *,
    db: Database,
    source_id: str,
    source_version_id: str | None,
    force: bool,
    progress: Callable[[str], None] | None,
) -> None:
    """
    Ensure sources.description is populated as part of ingestion (critical-path).
    """
    from insights.describe import ensure_source_description

    _p(progress, "describing: start")
    desc = ensure_source_description(
        db=db,
        source_id=source_id,
        source_version_id=source_version_id,
        force=force,
    )
    _p(progress, "describing: done")
    if not desc:
        raise RuntimeError("Description generation failed (sources.description is empty)")


def _require_metadata(
    *,
    db: Database,
    source_id: str,
    source_version_id: str,
    plain_text: str,
    force: bool,
    progress: Callable[[str], None] | None,
) -> None:
    """
    Ensure title/description/summary are populated as part of ingestion (critical-path).
    Uses deterministic fallbacks when LLM generation fails.
    """
    from insights.describe import ensure_source_description
    from insights.summarize import ensure_source_version_summary
    from insights.title import ensure_source_title

    _p(progress, "titling: start")
    title = ensure_source_title(db=db, source_id=source_id, source_version_id=source_version_id, force=force)
    _p(progress, "titling: done")
    if not (title or "").strip():
        raise RuntimeError("Title generation failed (sources.title is empty)")

    _p(progress, "describing: start")
    desc = ensure_source_description(db=db, source_id=source_id, source_version_id=source_version_id, force=force)
    _p(progress, "describing: done")
    if not (desc or "").strip():
        raise RuntimeError("Description generation failed (sources.description is empty)")

    _p(progress, "summarizing: start")
    summary = ensure_source_version_summary(
        db=db,
        source_version_id=source_version_id,
        content=plain_text or "",
        force=force,
        progress=progress,
    )
    _p(progress, "summarizing: done")
    if not (summary or "").strip():
        raise RuntimeError("Summary generation failed (source_versions.summary is empty)")


class IngestBackend(StrEnum):
    DOCLING = "docling"
    FIRECRAWL = "firecrawl"  # implemented in a later milestone/todo


@dataclass(frozen=True, slots=True)
class IngestResult:
    source: Source
    source_version: SourceVersion
    document_id: str
    reused_cache: bool


@dataclass(frozen=True, slots=True)
class EphemeralDocument:
    """In-memory document content, not persisted to DB."""

    locator: str
    kind: SourceKind
    title: str | None
    markdown: str
    plain_text: str
    token_count: int


@dataclass(frozen=True, slots=True)
class EphemeralResult:
    """Result of ephemeral extraction (no DB storage for new sources)."""

    document: EphemeralDocument | None  # Set if fresh extraction (not from cache)
    source: Source | None  # Set if from cache
    source_version: SourceVersion | None  # Set if from cache
    from_cache: bool


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
    if detected.kind == SourceKind.TWEET:
        return _ingest_tweet(
            db=db,
            detected=detected,
            refresh=refresh,
            title=title,
            summary_progress=summary_progress,
        )
    if detected.kind == SourceKind.LINKEDIN:
        # LinkedIn uses same extraction as URLs (docling/firecrawl)
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
    if detected.kind == SourceKind.GITHUB:
        # GitHub uses same extraction as URLs (docling/firecrawl)
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
                plain = db.get_document_plain_text_by_source_version(existing.id) or ""
                _require_metadata(
                    db=db,
                    source_id=source.id,
                    source_version_id=existing.id,
                    plain_text=plain,
                    force=False,
                    progress=summary_progress,
                )
                return IngestResult(
                    source=source,
                    source_version=existing,
                    document_id=doc_id,
                    reused_cache=True,
                )

    try:
        _p(summary_progress, f"extracting (docling): {path}")
        markdown = extract_markdown_with_docling(str(path))
        _p(summary_progress, "extracting (docling): done")
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
            token_count=token_est,
        )
        _require_metadata(
            db=db,
            source_id=source.id,
            source_version_id=version.id,
            plain_text=plain,
            force=bool(refresh),
            progress=summary_progress,
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
    summary_progress: Callable[[str], None] | None,
) -> IngestResult:
    url = detected.locator
    source = db.upsert_source(kind=detected.kind, locator=url, title=title)
    extractor = "docling"

    if not refresh:
        cached = _reuse_any_cached_document(
            db=db,
            source=source,
            extractor_preference=[extractor, "firecrawl"],
        )
        if cached is not None:
            plain = db.get_document_plain_text_by_source_version(cached.source_version.id) or ""
            _require_metadata(
                db=db,
                source_id=cached.source.id,
                source_version_id=cached.source_version.id,
                plain_text=plain,
                force=False,
                progress=summary_progress,
            )
            return cached

    try:
        _p(summary_progress, f"extracting (docling): {url}")
        markdown = extract_markdown_with_docling(url)
        _p(summary_progress, "extracting (docling): done")
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
            token_count=token_est,
        )
        _require_metadata(
            db=db,
            source_id=source.id,
            source_version_id=version.id,
            plain_text=plain,
            force=bool(refresh),
            progress=summary_progress,
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
    summary_progress: Callable[[str], None] | None,
) -> IngestResult:
    url = detected.locator
    source = db.upsert_source(kind=detected.kind, locator=url, title=title)
    extractor = "firecrawl"

    if not refresh:
        cached = _reuse_any_cached_document(
            db=db,
            source=source,
            extractor_preference=[extractor, "docling"],
        )
        if cached is not None:
            plain = db.get_document_plain_text_by_source_version(cached.source_version.id) or ""
            _require_metadata(
                db=db,
                source_id=cached.source.id,
                source_version_id=cached.source_version.id,
                plain_text=plain,
                force=False,
                progress=summary_progress,
            )
            return cached

    try:
        _p(summary_progress, f"extracting (firecrawl): {url}")
        markdown = extract_markdown_with_firecrawl(url)
        _p(summary_progress, "extracting (firecrawl): done")
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
            token_count=token_est,
        )
        _require_metadata(
            db=db,
            source_id=source.id,
            source_version_id=version.id,
            plain_text=plain,
            force=bool(refresh),
            progress=summary_progress,
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
                plain = db.get_document_plain_text_by_source_version(existing.id) or ""
                _require_metadata(
                    db=db,
                    source_id=source.id,
                    source_version_id=existing.id,
                    plain_text=plain,
                    force=False,
                    progress=summary_progress,
                )
                return IngestResult(
                    source=source,
                    source_version=existing,
                    document_id=doc_id,
                    reused_cache=True,
                )

    try:
        _p(summary_progress, f"downloading (yt-dlp): {video_id}")
        audio_path, yt_title = download_youtube_audio(video_id=video_id, cache_dir=cache_dir, refresh=refresh)
        _p(summary_progress, "downloading (yt-dlp): done")
        if not source.title and yt_title:
            source = db.upsert_source(kind=SourceKind.YOUTUBE, locator=video_id, title=yt_title)

        audio_hash = sha256_file(audio_path)
        _p(summary_progress, f"transcribing (assemblyai): {audio_path.name}")
        text = transcribe_with_assemblyai(audio_path=audio_path)
        _p(summary_progress, "transcribing (assemblyai): done")
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
            token_count=token_est,
        )
        _require_metadata(
            db=db,
            source_id=source.id,
            source_version_id=version.id,
            plain_text=plain,
            force=bool(refresh),
            progress=summary_progress,
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


def _ingest_tweet(
    *,
    db: Database,
    detected: DetectedSource,
    refresh: bool,
    title: str | None,
    summary_progress: Callable[[str], None] | None,
) -> IngestResult:
    """Ingest a tweet from Twitter/X."""
    tweet_url = detected.locator
    source = db.upsert_source(kind=SourceKind.TWEET, locator=tweet_url, title=title)
    extractor = "twitterapi"

    if not refresh:
        existing = db.get_latest_source_version(source_id=source.id, extractor=extractor)
        if existing and existing.status == "ok":
            doc_id = db.get_document_id_by_source_version(existing.id)
            if doc_id:
                plain = db.get_document_plain_text_by_source_version(existing.id) or ""
                _require_metadata(
                    db=db,
                    source_id=source.id,
                    source_version_id=existing.id,
                    plain_text=plain,
                    force=False,
                    progress=summary_progress,
                )
                return IngestResult(
                    source=source,
                    source_version=existing,
                    document_id=doc_id,
                    reused_cache=True,
                )

    try:
        _p(summary_progress, f"fetching tweet: {tweet_url}")
        tweet_data = fetch_tweet(tweet_url)
        _p(summary_progress, "fetching tweet: done")

        # Build markdown content with metadata
        markdown_parts = [f"# Tweet by @{tweet_data.author_username}"]
        if tweet_data.author_name:
            markdown_parts.append(f"**{tweet_data.author_name}**")
        markdown_parts.append("")
        markdown_parts.append(tweet_data.text)
        markdown_parts.append("")
        markdown_parts.append("---")
        markdown_parts.append(f"Source: [{tweet_url}]({tweet_url})")
        if tweet_data.created_at:
            markdown_parts.append(f"Posted: {tweet_data.created_at.isoformat()}")
        if tweet_data.likes is not None:
            markdown_parts.append(f"Likes: {tweet_data.likes}")
        if tweet_data.retweets is not None:
            markdown_parts.append(f"Retweets: {tweet_data.retweets}")

        markdown = "\n".join(markdown_parts)
        plain = tweet_data.text  # Use raw tweet text for title/summary generation
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
            token_count=token_est,
        )
        _require_metadata(
            db=db,
            source_id=source.id,
            source_version_id=version.id,
            plain_text=plain,
            force=bool(refresh),
            progress=summary_progress,
        )
        return IngestResult(source=source, source_version=version, document_id=doc_id, reused_cache=False)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.exception("Failed to ingest tweet")
        db.create_source_version(
            source_id=source.id,
            content_hash=sha256_text(tweet_url),
            extractor=extractor,
            status="error",
            error=msg,
        )
        raise RuntimeError(msg) from e


def extract_ephemeral(
    *,
    db: Database,
    input_value: str,
    cache_dir: Path | None = None,
    forced_type: str = "auto",
    url_backend: IngestBackend = IngestBackend.DOCLING,
    progress: Callable[[str], None] | None = None,
) -> EphemeralResult:
    """
    Extract content without storing to DB (unless already cached).

    If the source already exists in the DB, returns the cached content.
    Otherwise, extracts fresh content and returns it in-memory without DB writes.
    """
    detected = detect_source(input_value, forced_type=forced_type)

    # Check if source already exists in DB.
    try:
        existing_source = db.get_source_by_kind_locator(kind=detected.kind, locator=detected.locator)
    except KeyError:
        existing_source = None

    if existing_source:
        # Source exists - return cached content.
        docs = db.get_documents_for_sources_latest(
            source_ids=[existing_source.id],
            extractor_preference=["docling", "firecrawl", "assemblyai", "twitterapi"],
        )
        if docs:
            d = docs[0]
            version = db.get_source_version_by_id(str(d["source_version_id"]))
            return EphemeralResult(
                document=None,
                source=existing_source,
                source_version=version,
                from_cache=True,
            )

    # Source not in DB - extract ephemerally.
    if detected.kind == SourceKind.FILE:
        return _extract_file_ephemeral(detected=detected, progress=progress)
    if detected.kind == SourceKind.URL:
        return _extract_url_ephemeral(
            detected=detected,
            url_backend=url_backend,
            progress=progress,
        )
    if detected.kind == SourceKind.YOUTUBE:
        if cache_dir is None:
            raise ValueError("cache_dir is required for YouTube extraction")
        return _extract_youtube_ephemeral(
            detected=detected,
            cache_dir=cache_dir,
            progress=progress,
        )
    if detected.kind == SourceKind.TWEET:
        return _extract_tweet_ephemeral(detected=detected, progress=progress)
    if detected.kind == SourceKind.LINKEDIN:
        # LinkedIn uses same extraction as URLs (docling/firecrawl)
        return _extract_url_ephemeral(
            detected=detected,
            url_backend=url_backend,
            progress=progress,
        )
    if detected.kind == SourceKind.GITHUB:
        # GitHub uses same extraction as URLs (docling/firecrawl)
        return _extract_url_ephemeral(
            detected=detected,
            url_backend=url_backend,
            progress=progress,
        )
    raise ValueError(f"Unsupported source kind: {detected.kind.value}")


def _extract_file_ephemeral(
    *,
    detected: DetectedSource,
    progress: Callable[[str], None] | None,
) -> EphemeralResult:
    """Extract file content in-memory without DB storage."""
    path = Path(detected.locator)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    _p(progress, f"extracting (docling): {path}")
    markdown = extract_markdown_with_docling(str(path))
    _p(progress, "extracting (docling): done")
    plain = markdown_to_text(markdown)
    token_est = estimate_tokens(plain)

    return EphemeralResult(
        document=EphemeralDocument(
            locator=str(path),
            kind=SourceKind.FILE,
            title=detected.display_title,
            markdown=markdown,
            plain_text=plain,
            token_count=token_est,
        ),
        source=None,
        source_version=None,
        from_cache=False,
    )


def _extract_url_ephemeral(
    *,
    detected: DetectedSource,
    url_backend: IngestBackend,
    progress: Callable[[str], None] | None,
) -> EphemeralResult:
    """Extract URL content in-memory without DB storage."""
    url = detected.locator

    if url_backend == IngestBackend.DOCLING:
        _p(progress, f"extracting (docling): {url}")
        markdown = extract_markdown_with_docling(url)
        _p(progress, "extracting (docling): done")
    elif url_backend == IngestBackend.FIRECRAWL:
        _p(progress, f"extracting (firecrawl): {url}")
        markdown = extract_markdown_with_firecrawl(url)
        _p(progress, "extracting (firecrawl): done")
    else:
        raise ValueError(f"Unsupported URL backend: {url_backend}")

    plain = markdown_to_text(markdown)
    token_est = estimate_tokens(plain)

    return EphemeralResult(
        document=EphemeralDocument(
            locator=url,
            kind=detected.kind,
            title=detected.display_title,
            markdown=markdown,
            plain_text=plain,
            token_count=token_est,
        ),
        source=None,
        source_version=None,
        from_cache=False,
    )


def _extract_youtube_ephemeral(
    *,
    detected: DetectedSource,
    cache_dir: Path,
    progress: Callable[[str], None] | None,
) -> EphemeralResult:
    """Extract YouTube transcript in-memory without DB storage."""
    video_id = detected.locator

    _p(progress, f"downloading (yt-dlp): {video_id}")
    audio_path, yt_title = download_youtube_audio(video_id=video_id, cache_dir=cache_dir, refresh=False)
    _p(progress, "downloading (yt-dlp): done")

    _p(progress, f"transcribing (assemblyai): {audio_path.name}")
    text = transcribe_with_assemblyai(audio_path=audio_path)
    _p(progress, "transcribing (assemblyai): done")

    markdown = text.strip()
    plain = markdown
    token_est = estimate_tokens(plain)

    return EphemeralResult(
        document=EphemeralDocument(
            locator=video_id,
            kind=SourceKind.YOUTUBE,
            title=yt_title or detected.display_title,
            markdown=markdown,
            plain_text=plain,
            token_count=token_est,
        ),
        source=None,
        source_version=None,
        from_cache=False,
    )


def _extract_tweet_ephemeral(
    *,
    detected: DetectedSource,
    progress: Callable[[str], None] | None,
) -> EphemeralResult:
    """Extract tweet content in-memory without DB storage."""
    tweet_url = detected.locator

    _p(progress, f"fetching tweet: {tweet_url}")
    tweet_data = fetch_tweet(tweet_url)
    _p(progress, "fetching tweet: done")

    # Build markdown content
    markdown_parts = [f"# Tweet by @{tweet_data.author_username}"]
    if tweet_data.author_name:
        markdown_parts.append(f"**{tweet_data.author_name}**")
    markdown_parts.append("")
    markdown_parts.append(tweet_data.text)
    markdown = "\n".join(markdown_parts)

    plain = tweet_data.text
    token_est = estimate_tokens(plain)

    return EphemeralResult(
        document=EphemeralDocument(
            locator=tweet_url,
            kind=SourceKind.TWEET,
            title=f"Tweet by @{tweet_data.author_username}",
            markdown=markdown,
            plain_text=plain,
            token_count=token_est,
        ),
        source=None,
        source_version=None,
        from_cache=False,
    )

