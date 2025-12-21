from __future__ import annotations

import os
from dataclasses import dataclass
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


