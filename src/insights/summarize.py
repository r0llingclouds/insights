from __future__ import annotations

import os
from dataclasses import dataclass
import re
from typing import Callable

from insights.llm import AnthropicClient, ChatMessage, OpenAIClient
from insights.llm.routing import get_large_content_cutoff, is_large_content, pick_anthropic_model
from insights.storage.db import Database


SUMMARY_SYSTEM = """\
You write a compact summary of content for quick recall.

Requirements:
- Output as ONE short paragraph (no bullets, no headings)
- About 3–6 sentences
- Prefer concrete facts / claims / key takeaways
- Avoid fluff
- Keep it readable and information-dense
"""

MAP_SYSTEM = """\
You summarize ONE chunk of a larger document.

Requirements:
- Output as ONE short paragraph (no bullets, no headings)
- About 2–3 sentences
- Prefer concrete facts / requirements / key takeaways in this chunk
"""

REDUCE_SYSTEM = """\
You are combining summaries of multiple chunks of a document into a final summary.

Requirements:
- Output as ONE short paragraph (no bullets, no headings)
- About 3–6 sentences
- Remove redundancy; keep the most important points across the whole document
"""


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        n = int(v)
    except ValueError:
        return default
    return n if n > 0 else default


def chunk_text(
    text: str,
    *,
    chunk_chars: int,
    overlap_chars: int,
) -> list[str]:
    """
    Chunk the entire text with overlap. Prefer paragraph-aware boundaries.

    Implementation note: avoid splitting the whole doc into a huge list of paragraphs
    (which can explode memory for extracted PDFs). Instead, pick breakpoints near
    the chunk boundary using rfind on common separators.
    """
    t = ((text or "").replace("\r\n", "\n").replace("\r", "\n")).strip()
    if not t:
        return []
    if chunk_chars <= 0:
        return [t]
    if len(t) <= chunk_chars:
        return [t]

    out: list[str] = []
    n = len(t)
    start = 0
    # If we can't find a reasonable boundary, don't shrink below this fraction,
    # otherwise we risk many tiny chunks and slow progress.
    min_end = max(1, chunk_chars // 3)

    while start < n:
        target_end = min(n, start + chunk_chars)
        end = target_end

        if end < n:
            # Prefer breaking at paragraph boundaries, then line breaks, then spaces.
            para = t.rfind("\n\n", start, end)
            if para != -1 and (para - start) >= min_end:
                end = para
            else:
                nl = t.rfind("\n", start, end)
                if nl != -1 and (nl - start) >= min_end:
                    end = nl
                else:
                    sp = t.rfind(" ", start, end)
                    if sp != -1 and (sp - start) >= min_end:
                        end = sp

        chunk = t[start:end].strip()
        if chunk:
            out.append(chunk)

        if end >= n:
            break

        prev_start = start
        start = max(0, end - max(0, overlap_chars))
        # Ensure progress even in pathological whitespace / separator patterns.
        if start <= prev_start:
            start = end
        while start < n and t[start].isspace():
            start += 1

    return out


_RE_LEADING_BULLET = re.compile(r"^\s*([-*•]\s+)+")


def _clean_paragraph(text: str) -> str:
    """
    Normalize model output into a single paragraph.
    - Collapse newlines/whitespace
    - Strip quotes
    - If the model accidentally outputs bullets, remove bullet prefixes and join into one paragraph
    """
    raw = (text or "").strip().strip("\"'")
    if not raw:
        return ""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    cleaned_lines: list[str] = []
    for ln in lines:
        ln = _RE_LEADING_BULLET.sub("", ln).strip()
        if ln:
            cleaned_lines.append(ln)
    return " ".join(" ".join(cleaned_lines).split()).strip()


_RE_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _fallback_summary(content: str) -> str:
    """
    Deterministic fallback summary (single paragraph) when the LLM path fails.
    """
    t = " ".join((content or "").replace("\r\n", "\n").replace("\r", "\n").split()).strip()
    if not t:
        return "(no content cached)"
    head = t[:4000]
    sentences = [s.strip() for s in _RE_SENT_SPLIT.split(head) if s.strip()]
    if not sentences:
        return _clean_paragraph(head)
    # Take 3–6 sentences.
    chosen = sentences[:6]
    if len(chosen) < 3:
        chosen = sentences[:3]
    return _clean_paragraph(" ".join(chosen))


@dataclass(frozen=True, slots=True)
class _LLM:
    provider: str
    model: str
    generate: Callable[[list[ChatMessage], str, int], str]


def _build_llm(*, provider: str, model: str | None, content_len: int) -> _LLM:
    p = provider.strip().lower()
    if p == "openai":
        client = OpenAIClient()
        used_model = model or "gpt-4o-mini"

        def _gen(messages: list[ChatMessage], used_model: str, max_tokens: int) -> str:
            return client.generate(messages=messages, model=used_model, temperature=0.2, max_tokens=max_tokens).text

        return _LLM(provider="openai", model=used_model, generate=_gen)

    if p == "anthropic":
        client = AnthropicClient()
        if model:
            used_model = model
        else:
            used_model = pick_anthropic_model(
                env_model=os.getenv("INSIGHTS_SUMMARY_MODEL"),
                default_model="claude-sonnet-4-5-20250929",
                content_len=content_len,
                large_model="claude-haiku-4-5-20251001",
            )

        def _gen(messages: list[ChatMessage], used_model: str, max_tokens: int) -> str:
            return client.generate(messages=messages, model=used_model, temperature=0.2, max_tokens=max_tokens).text

        return _LLM(provider="anthropic", model=used_model, generate=_gen)

    raise ValueError("provider must be 'openai' or 'anthropic'")


def map_reduce_summary(
    *,
    content: str,
    llm: _LLM,
    chunk_chars: int,
    overlap_chars: int,
    reduce_batch_size: int,
    progress: Callable[[str], None] | None = None,
    progress_every_chunks: int = 5,
) -> str:
    chunks = chunk_text(content, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    if not chunks:
        return ""
    if len(chunks) == 1:
        # Single-pass summary over full doc.
        messages = [
            ChatMessage(role="system", content=SUMMARY_SYSTEM),
            ChatMessage(role="user", content=f"Summarize this content as a single short paragraph:\n\n---\n{chunks[0]}\n---\n\nSummary:"),
        ]
        raw = llm.generate(messages, llm.model, 400)
        return _clean_paragraph(raw)

    if progress is not None:
        content_len = len(content or "")
        progress(
            "Large content detected: generating whole-doc summary via map-reduce "
            f"(chars={content_len}, cutoff={get_large_content_cutoff()}, model={llm.model}, "
            f"chunk_chars={chunk_chars}, overlap_chars={overlap_chars}, chunks={len(chunks)})"
        )

    # Map: summarize each chunk.
    mapped: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        if progress is not None:
            every = max(1, int(progress_every_chunks))
            if i == 1 or i == len(chunks) or (i % every) == 0:
                progress(f"summary map: chunk {i}/{len(chunks)}")
        user = f"Chunk {i}/{len(chunks)}:\n\n---\n{chunk}\n---\n\nSummary paragraph:"
        messages = [ChatMessage(role="system", content=MAP_SYSTEM), ChatMessage(role="user", content=user)]
        raw = llm.generate(messages, llm.model, 250)
        mapped.append(_clean_paragraph(raw))

    # Reduce: hierarchical batch reduction until one summary remains.
    batch = max(2, int(reduce_batch_size))
    current = mapped
    round_idx = 0
    while len(current) > 1:
        round_idx += 1
        if progress is not None:
            groups = (len(current) + batch - 1) // batch
            progress(f"summary reduce: round {round_idx} (inputs={len(current)}, batch_size={batch}, groups={groups})")
        next_round: list[str] = []
        for start in range(0, len(current), batch):
            group = current[start : start + batch]
            combined = "\n".join(group).strip()
            messages = [
                ChatMessage(role="system", content=REDUCE_SYSTEM),
                ChatMessage(
                    role="user",
                    content=f"Combine these chunk summaries into one short paragraph summary:\n\n---\n{combined}\n---\n\nSummary:",
                ),
            ]
            raw = llm.generate(messages, llm.model, 450)
            next_round.append(_clean_paragraph(raw))
        current = next_round

    return _clean_paragraph(current[0])


def generate_summary(
    *,
    content: str,
    provider: str = "anthropic",
    model: str | None = None,
    max_content_chars: int = 12000,
    progress: Callable[[str], None] | None = None,
) -> str:
    content_len = len(content or "")
    # Back-compat: previously this parameter controlled how much content we fed the model.
    # Now we always summarize the whole doc; this value acts as the default chunk size unless
    # overridden by env.
    chunk_chars = _env_int("INSIGHTS_SUMMARY_CHUNK_CHARS", int(max_content_chars))
    overlap_chars = _env_int("INSIGHTS_SUMMARY_OVERLAP_CHARS", 400)
    reduce_batch = _env_int("INSIGHTS_SUMMARY_REDUCE_BATCH_SIZE", 10)
    progress_every = _env_int("INSIGHTS_SUMMARY_PROGRESS_EVERY_CHUNKS", 5)

    llm = _build_llm(provider=provider, model=model, content_len=content_len)
    # Always use whole-doc map-reduce; never truncate the source text.
    # Only emit progress for \"large\" content (unless caller wants to force it by passing
    # a progress callback and setting INSIGHTS_LARGE_CONTENT_CUTOFF_CHARS very low).
    use_progress = progress if (progress is not None and is_large_content(content_len=content_len)) else None
    return map_reduce_summary(
        content=content,
        llm=llm,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
        reduce_batch_size=reduce_batch,
        progress=use_progress,
        progress_every_chunks=progress_every,
    )


def ensure_source_version_summary(
    *,
    db: Database,
    source_version_id: str,
    content: str,
    force: bool = False,
    provider: str = "anthropic",
    model: str | None = None,
    max_content_chars: int = 12000,
    progress: Callable[[str], None] | None = None,
) -> str | None:
    """
    Ensure `source_versions.summary` is populated for a given version.

    If LLM summarization fails or returns empty, persists a deterministic fallback summary.
    """
    sv = db.get_source_version_by_id(source_version_id)
    if sv is None:
        return None
    if (sv.summary or "").strip() and not force:
        return sv.summary

    summary = ""
    try:
        summary = generate_summary(
            content=content or "",
            provider=provider,
            model=model,
            max_content_chars=max_content_chars,
            progress=progress,
        )
    except Exception:
        summary = ""
    summary = (summary or "").strip()
    if not summary:
        summary = _fallback_summary(content or "")

    db.set_source_version_summary(source_version_id=source_version_id, summary=summary)
    return summary


