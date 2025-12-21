from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from insights.llm import AnthropicClient, ChatMessage, OpenAIClient
from insights.llm.routing import pick_anthropic_model


SUMMARY_SYSTEM = """\
You write a compact summary of content for quick recall.

Requirements:
- 3–7 bullet points max
- Prefer concrete facts / claims / key takeaways
- Avoid fluff
- Output as Markdown bullets only (lines starting with '- ')
- Each bullet must be a SINGLE LINE (no wrapping/newlines)
- Keep bullets short (<= 200 chars each)
- If the document includes examples/templates, summarize the rules/requirements first
"""

MAP_SYSTEM = """\
You summarize ONE chunk of a larger document.

Requirements:
- 3–5 bullet points max
- Output as Markdown bullets only (lines starting with '- ')
- Each bullet must be a SINGLE LINE (no wrapping/newlines)
- Keep bullets short (<= 200 chars each)
- Prefer concrete facts / requirements / key takeaways in this chunk
"""

REDUCE_SYSTEM = """\
You are combining summaries of multiple chunks of a document into a final summary.

Requirements:
- 3–7 bullet points max
- Output as Markdown bullets only (lines starting with '- ')
- Each bullet must be a SINGLE LINE (no wrapping/newlines)
- Keep bullets short (<= 200 chars each)
- Remove redundancy, keep the most important points across the whole document
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


def _parse_bullets(text: str, *, max_bullets: int) -> list[str]:
    """
    Parse markdown bullets from model output; join wrapped lines into previous bullet.
    Returns bullet strings WITHOUT the leading '- '.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    items: list[str] = []
    cur: str | None = None
    for ln in lines:
        if ln.startswith("- "):
            if cur:
                items.append(cur)
            cur = ln[2:].strip()
        else:
            if cur is None:
                continue
            cur = (cur + " " + ln.lstrip("-").strip()).strip()
    if cur:
        items.append(cur)

    cleaned: list[str] = []
    for it in items:
        it = it.replace("**", "")
        it = " ".join(it.split())
        if len(it) > 200:
            it = it[:199].rstrip() + "…"
        if it:
            cleaned.append(it)
    return cleaned[:max_bullets]


def _format_bullets(items: list[str]) -> str:
    return "\n".join(f"- {it}" for it in items if it).strip()


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
                default_model="claude-sonnet-4-20250514",
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
) -> str:
    chunks = chunk_text(content, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    if not chunks:
        return ""
    if len(chunks) == 1:
        # Single-pass summary over full doc.
        messages = [
            ChatMessage(role="system", content=SUMMARY_SYSTEM),
            ChatMessage(role="user", content=f"Summarize this content as Markdown bullets:\n\n---\n{chunks[0]}\n---\n\nBullets:"),
        ]
        raw = llm.generate(messages, llm.model, 500)
        bullets = _parse_bullets(raw, max_bullets=7)
        return _format_bullets(bullets)

    # Map: summarize each chunk.
    mapped: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        user = f"Chunk {i}/{len(chunks)}:\n\n---\n{chunk}\n---\n\nBullets:"
        messages = [ChatMessage(role="system", content=MAP_SYSTEM), ChatMessage(role="user", content=user)]
        raw = llm.generate(messages, llm.model, 300)
        bullets = _parse_bullets(raw, max_bullets=5)
        mapped.append(_format_bullets(bullets))

    # Reduce: hierarchical batch reduction until one summary remains.
    batch = max(2, int(reduce_batch_size))
    current = mapped
    while len(current) > 1:
        next_round: list[str] = []
        for start in range(0, len(current), batch):
            group = current[start : start + batch]
            combined = "\n".join(group).strip()
            messages = [
                ChatMessage(role="system", content=REDUCE_SYSTEM),
                ChatMessage(
                    role="user",
                    content=f"Combine these chunk summaries into a final Markdown bullet summary:\n\n---\n{combined}\n---\n\nBullets:",
                ),
            ]
            raw = llm.generate(messages, llm.model, 500)
            bullets = _parse_bullets(raw, max_bullets=7)
            next_round.append(_format_bullets(bullets))
        current = next_round

    return current[0].strip()


def generate_summary(
    *,
    content: str,
    provider: str = "anthropic",
    model: str | None = None,
    max_content_chars: int = 12000,
) -> str:
    content_len = len(content or "")
    # Back-compat: previously this parameter controlled how much content we fed the model.
    # Now we always summarize the whole doc; this value acts as the default chunk size unless
    # overridden by env.
    chunk_chars = _env_int("INSIGHTS_SUMMARY_CHUNK_CHARS", int(max_content_chars))
    overlap_chars = _env_int("INSIGHTS_SUMMARY_OVERLAP_CHARS", 400)
    reduce_batch = _env_int("INSIGHTS_SUMMARY_REDUCE_BATCH_SIZE", 10)

    llm = _build_llm(provider=provider, model=model, content_len=content_len)
    # Always use whole-doc map-reduce; never truncate the source text.
    return map_reduce_summary(
        content=content,
        llm=llm,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
        reduce_batch_size=reduce_batch,
    )


