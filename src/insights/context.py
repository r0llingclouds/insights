from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from insights.storage.db import Database
from insights.storage.models import Source
from insights.utils import estimate_tokens, truncate_to_tokens

ProgressFn = Callable[[str], None]


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return int(default)
    try:
        n = int(v)
    except ValueError:
        return int(default)
    return int(n) if int(n) > 0 else int(default)


def _trim_head(text: str, *, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return text, False
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    context_text: str
    token_estimate: int
    sources: list[Source]


def build_context(
    *,
    db: Database,
    source_ids: list[str],
    question: str,
    extractor_preference: list[str] | None = None,
    max_context_tokens: int = 12000,
    progress: ProgressFn | None = None,
) -> ContextBuildResult:
    """
    Build an LLM-ready FULL context for a set of sources.

    - Per-document head trim: INSIGHTS_MAX_CONTEXT_CHARS (default 400,000).
    - Total token trim: --max-context-tokens (tiktoken).
    """
    _ = question  # reserved for future prompt-shaping; kept for call-site compatibility
    _ = progress

    extractor_preference = extractor_preference or ["docling", "firecrawl", "assemblyai"]
    max_context_chars = _env_int("INSIGHTS_MAX_CONTEXT_CHARS", 400_000)

    sources: list[Source] = []
    for sid in source_ids:
        s = db.get_source_by_id(sid)
        if not s:
            raise KeyError(f"Unknown source id: {sid}")
        sources.append(s)

    docs = db.get_documents_for_sources_latest(source_ids=source_ids, extractor_preference=extractor_preference)
    raw_by_source: dict[str, dict[str, Any]] = {d["source_id"]: d for d in docs}

    # Apply per-document trim (head-only) for FULL context building.
    doc_by_source: dict[str, dict[str, Any]] = {}
    for sid in source_ids:
        d = raw_by_source.get(sid)
        if not d:
            continue
        plain = str(d.get("plain_text") or "")
        trimmed, _trimmed = _trim_head(plain, max_chars=max_context_chars)
        doc_by_source[sid] = {
            **d,
            "plain_text": trimmed,
            # Recompute token count on the trimmed text (DB token_count is for full doc).
            "token_count": int(estimate_tokens(trimmed)),
        }

    total_tokens = 0
    for sid in source_ids:
        d = doc_by_source.get(sid)
        if not d:
            raise RuntimeError(f"No extracted document found for source {sid}. Run `insights ingest` first.")
        total_tokens += int(d["token_count"])

    parts: list[str] = []
    for s in sources:
        d = doc_by_source[s.id]
        title = s.title or s.locator
        parts.append(f"[Source: {title} | id={s.id} | locator={s.locator}]\n{d['plain_text']}".strip())
    context_text = "\n\n".join(parts).strip()

    # Enforce overall token budget for FULL context.
    if int(max_context_tokens) > 0 and estimate_tokens(context_text) > int(max_context_tokens):
        context_text = truncate_to_tokens(context_text, int(max_context_tokens)).strip()

    return ContextBuildResult(
        context_text=context_text,
        token_estimate=estimate_tokens(context_text),
        sources=sources,
    )


def build_context_with_retrieval(
    *,
    db: Database,
    source_ids: list[str] | None,
    query: str,
    cache_dir: Path,
    n_chunks: int = 10,
    max_context_tokens: int = 12000,
    progress: ProgressFn | None = None,
) -> ContextBuildResult:
    """
    Build LLM context using semantic retrieval (RAG).

    Retrieves relevant chunks from the vector store instead of sending full documents.
    If source_ids is None or empty, searches ALL indexed sources.
    Falls back to full-text context if sources aren't indexed.
    """
    from insights.retrieval.search import semantic_search
    from insights.retrieval.store import VectorStore

    vector_store = VectorStore(persist_dir=cache_dir / "vectors")
    indexed_ids = vector_store.get_indexed_sources()

    # If no sources specified, search all indexed sources
    if not source_ids:
        if not indexed_ids:
            raise RuntimeError("No indexed sources found. Run `insights index --all` first.")
        source_ids = list(indexed_ids)
        if progress:
            progress(f"Searching across {len(source_ids)} indexed source(s)...")

    # Load sources
    sources: list[Source] = []
    for sid in source_ids:
        s = db.get_source_by_id(sid)
        if not s:
            raise KeyError(f"Unknown source id: {sid}")
        sources.append(s)

    # Check which sources are indexed
    indexed_source_ids = [sid for sid in source_ids if sid in indexed_ids]

    if not indexed_source_ids:
        # No indexed sources - fall back to full-text
        if progress:
            progress("No indexed sources found, falling back to full-text context")
        return build_context(
            db=db,
            source_ids=source_ids,
            question=query,
            max_context_tokens=max_context_tokens,
            progress=progress,
        )

    # Some sources not indexed - warn
    not_indexed = [sid for sid in source_ids if sid not in indexed_ids]
    if not_indexed and progress:
        progress(f"Warning: {len(not_indexed)} source(s) not indexed, using full-text for them")

    # Perform semantic search on indexed sources
    if progress:
        progress(f"Searching {len(indexed_source_ids)} indexed source(s)...")

    results = semantic_search(
        query,
        store=vector_store,
        n_results=n_chunks,
        source_ids=indexed_source_ids,
    )

    if progress:
        progress(f"Found {len(results)} relevant chunks")

    # Build context from retrieved chunks
    parts: list[str] = []

    # Add retrieved chunks with citations
    if results:
        chunk_parts: list[str] = []
        for r in results:
            source = db.get_source_by_id(r.source_id)
            title = source.title if source else r.source_id
            chunk_parts.append(
                f"[Source: {title} | chunk {r.chunk_index} | score={r.score:.3f}]\n{r.content}"
            )
        parts.append("## Retrieved chunks (most relevant)\n\n" + "\n\n---\n\n".join(chunk_parts))

    # Add full-text for non-indexed sources
    if not_indexed:
        extractor_preference = ["docling", "firecrawl", "assemblyai"]
        docs = db.get_documents_for_sources_latest(
            source_ids=not_indexed,
            extractor_preference=extractor_preference,
        )
        for d in docs:
            source = db.get_source_by_id(d["source_id"])
            title = source.title if source else d["source_id"]
            plain = str(d.get("plain_text") or "")
            if plain:
                parts.append(f"[Source: {title} | full text]\n{plain}")

    context_text = "\n\n".join(parts).strip()

    # Enforce token budget
    if max_context_tokens > 0 and estimate_tokens(context_text) > max_context_tokens:
        context_text = truncate_to_tokens(context_text, max_context_tokens).strip()

    return ContextBuildResult(
        context_text=context_text,
        token_estimate=estimate_tokens(context_text),
        sources=sources,
    )


