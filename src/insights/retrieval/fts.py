from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from insights.storage.db import Database
from insights.storage.models import Source
from insights.utils.tokens import estimate_tokens


class ContextMode(StrEnum):
    FULL = "full"
    RETRIEVAL = "retrieval"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    source: Source
    document_id: str
    chunk_index: int
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    mode: ContextMode
    context_text: str
    token_estimate: int
    sources: list[Source]
    retrieved_chunks: list[RetrievedChunk]


_RE_TERM = re.compile(r"[A-Za-z0-9_]{3,}")


def _fts_query_from_question(question: str, *, max_terms: int = 12) -> str:
    # FTS5 MATCH syntax is picky; build a conservative OR query from words.
    terms = [t.lower() for t in _RE_TERM.findall(question)]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
        if len(uniq) >= max_terms:
            break
    if not uniq:
        # Fall back to a single token; this may still fail on some punctuation,
        # but it's better than refusing.
        return question.strip() or "insights"
    return " OR ".join(uniq)


def chunk_text(
    text: str,
    *,
    max_chunk_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[str]:
    """
    Chunk text into overlapping windows based on token estimates.
    """
    text = (text or "").strip()
    if not text:
        return []

    max_chars = max(200, max_chunk_tokens * 4)
    overlap_chars = max(0, overlap_tokens * 4)

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)

        # Prefer breaking on paragraph boundaries near the end.
        boundary = text.rfind("\n\n", start, end)
        if boundary != -1 and boundary > start + (max_chars // 2):
            end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        next_start = end - overlap_chars
        # Ensure progress even if overlap is large.
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _ensure_indexed(
    db: Database,
    *,
    document_id: str,
    plain_text: str,
    max_chunk_tokens: int,
    overlap_tokens: int,
) -> None:
    if db.chunk_count(document_id=document_id) > 0:
        return
    chunks = chunk_text(plain_text, max_chunk_tokens=max_chunk_tokens, overlap_tokens=overlap_tokens)
    db.replace_chunks(document_id=document_id, chunks=chunks)


def build_context(
    *,
    db: Database,
    source_ids: list[str],
    question: str,
    extractor_preference: list[str] | None = None,
    max_context_tokens: int = 12000,
    retrieval_top_k: int = 10,
    max_chunk_tokens: int = 800,
    overlap_tokens: int = 100,
) -> ContextBuildResult:
    """
    Build an LLM-ready context for a set of sources.

    - If combined doc token estimates fit under max_context_tokens → include full text.
    - Else → retrieve top-k chunks via SQLite FTS5.
    """
    extractor_preference = extractor_preference or ["docling", "firecrawl", "assemblyai"]

    sources: list[Source] = []
    for sid in source_ids:
        s = db.get_source_by_id(sid)
        if not s:
            raise KeyError(f"Unknown source id: {sid}")
        sources.append(s)

    docs = db.get_documents_for_sources_latest(source_ids=source_ids, extractor_preference=extractor_preference)
    doc_by_source: dict[str, dict[str, Any]] = {d["source_id"]: d for d in docs}

    total_tokens = 0
    for sid in source_ids:
        d = doc_by_source.get(sid)
        if not d:
            raise RuntimeError(f"No extracted document found for source {sid}. Run `insights ingest` first.")
        total_tokens += int(d["token_estimate"])

    if total_tokens <= max_context_tokens:
        parts: list[str] = []
        for s in sources:
            d = doc_by_source[s.id]
            title = s.title or s.locator
            parts.append(f"[Source: {title} | id={s.id} | locator={s.locator}]\n{d['plain_text']}".strip())
        context_text = "\n\n".join(parts).strip()
        return ContextBuildResult(
            mode=ContextMode.FULL,
            context_text=context_text,
            token_estimate=estimate_tokens(context_text),
            sources=sources,
            retrieved_chunks=[],
        )

    # Retrieval mode: ensure chunk index exists, then search.
    document_ids: list[str] = []
    for sid in source_ids:
        d = doc_by_source[sid]
        document_id = str(d["document_id"])
        document_ids.append(document_id)
        _ensure_indexed(
            db,
            document_id=document_id,
            plain_text=str(d["plain_text"]),
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
        )

    fts_query = _fts_query_from_question(question)
    matches = db.search_chunks_fts(query=fts_query, document_ids=document_ids, limit=retrieval_top_k)
    chunk_rows = db.get_chunks_by_ids([m.chunk_id for m in matches])
    chunk_by_id = {str(r["id"]): r for r in chunk_rows}
    doc_to_source: dict[str, Source] = {}
    for s in sources:
        d = doc_by_source[s.id]
        doc_to_source[str(d["document_id"])] = s

    retrieved: list[RetrievedChunk] = []
    for m in matches:
        row = chunk_by_id.get(m.chunk_id)
        if not row:
            continue
        doc_id = str(row["document_id"])
        source = doc_to_source.get(doc_id)
        if not source:
            continue
        retrieved.append(
            RetrievedChunk(
                source=source,
                document_id=doc_id,
                chunk_index=int(row["chunk_index"]),
                text=str(row["text"]),
                score=float(m.score),
            )
        )

    # Assemble context
    parts: list[str] = []
    for rc in retrieved:
        title = rc.source.title or rc.source.locator
        parts.append(
            f"[Source: {title} | id={rc.source.id} | locator={rc.source.locator} | chunk={rc.chunk_index}]\n{rc.text}".strip()
        )
    context_text = "\n\n".join(parts).strip()
    return ContextBuildResult(
        mode=ContextMode.RETRIEVAL,
        context_text=context_text,
        token_estimate=estimate_tokens(context_text),
        sources=sources,
        retrieved_chunks=retrieved,
    )


