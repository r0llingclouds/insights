from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlparse
import re

from insights.llm import AnthropicClient, ChatMessage, OpenAIClient
from insights.llm.routing import pick_anthropic_model
from insights.storage.db import Database
from insights.storage.models import SourceKind


TITLE_SYSTEM = """\
You generate concise, searchable titles for content.

Requirements:
- 6–12 words (roughly)
- <= 80 characters (hard limit; if longer, it will be truncated)
- Plain text (no quotes, no prefixes like "Title:")
- Use specific terms someone might search for
- Avoid clickbait and filler words

Respond with ONLY the title."""


def _truncate_content(content: str, *, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    half = max_chars // 2
    return content[:half] + "\n\n[...content truncated...]\n\n" + content[-half:]


def generate_title(
    *,
    content: str,
    provider: str = "anthropic",
    model: str | None = None,
    max_content_chars: int = 8000,
) -> str:
    content_len = len(content or "")
    p = provider.strip().lower()
    if p == "openai":
        client = OpenAIClient()
        used_model = model or "gpt-4o-mini"
    elif p == "anthropic":
        client = AnthropicClient()
        if model:
            used_model = model
        else:
            used_model = pick_anthropic_model(
                env_model=os.getenv("INSIGHTS_TITLE_MODEL"),
                default_model="claude-sonnet-4-5-20250929",
                content_len=content_len,
                large_model="claude-haiku-4-5-20251001",
            )
    else:
        raise ValueError("provider must be 'openai' or 'anthropic'")

    truncated = _truncate_content(content, max_chars=max(1000, int(max_content_chars)))

    user = (
        "Generate a short title for this content:\n\n---\n"
        f"{truncated}\n"
        "---\n\nTitle:"
    )
    messages = [
        ChatMessage(role="system", content=TITLE_SYSTEM),
        ChatMessage(role="user", content=user),
    ]
    resp = client.generate(messages=messages, model=used_model, temperature=0.2, max_tokens=32)

    title = (resp.text or "").strip().strip("\"'")
    title = " ".join(title.split())
    if len(title) > 80:
        title = title[:79].rstrip() + "…"
    return title


DEFAULT_EXTRACTOR_PREFERENCE: list[str] = ["firecrawl", "docling", "assemblyai"]


_RE_SLUG_SPLIT = re.compile(r"[-_]+")


def _clean_title(text: str, *, max_chars: int = 80) -> str:
    t = " ".join((text or "").split()).strip().strip("\"'")
    if not t:
        return ""
    if len(t) > max_chars:
        t = t[: max_chars - 1].rstrip() + "…"
    return t


def _fallback_title(*, kind: SourceKind, locator: str) -> str:
    loc = (locator or "").strip()
    if kind == SourceKind.FILE:
        name = Path(loc).name if loc else ""
        return _clean_title(name or loc or "Untitled")
    if kind == SourceKind.URL:
        try:
            p = urlparse(loc)
        except Exception:
            return _clean_title(loc or "Untitled")
        host = (p.hostname or "").lower().strip(".")
        seg = ""
        if p.path:
            parts = [x for x in p.path.split("/") if x]
            if parts:
                seg = parts[-1]
        seg = unquote(seg or "").strip()
        seg_words = [w for w in _RE_SLUG_SPLIT.split(seg) if w]
        seg_title = " ".join(seg_words).strip()
        if seg_title:
            seg_title = seg_title[:1].upper() + seg_title[1:]
        if host and seg_title:
            return _clean_title(f"{host} — {seg_title}")
        if host:
            return _clean_title(host)
        return _clean_title(loc or "Untitled")
    if kind == SourceKind.YOUTUBE:
        return _clean_title(f"YouTube {loc}" if loc else "YouTube")
    return _clean_title(loc or "Untitled")


def ensure_source_title(
    *,
    db: Database,
    source_id: str,
    source_version_id: str | None = None,
    force: bool = False,
    provider: str = "anthropic",
    model: str | None = None,
    max_content_chars: int = 8000,
    extractor_preference: list[str] | None = None,
) -> str | None:
    """
    Ensure `sources.title` is populated for a source.

    - If a title already exists and `force` is False, it's returned unchanged.
    - If missing, we generate from cached plain text and persist it.
    - If generation fails/empty, we persist a deterministic fallback (never leaves title NULL).
    """
    src = db.get_source_by_id(source_id)
    if src is None:
        return None
    if (src.title or "").strip() and not force:
        return src.title

    plain: str | None = None
    if source_version_id:
        plain = db.get_document_plain_text_by_source_version(source_version_id)
    if not plain:
        plain = db.get_latest_plain_text_for_source(
            source_id=source_id,
            extractor_preference=extractor_preference or DEFAULT_EXTRACTOR_PREFERENCE,
        )

    title = ""
    if plain and plain.strip():
        try:
            title = generate_title(
                content=plain,
                provider=provider,
                model=model,
                max_content_chars=max_content_chars,
            )
        except Exception:
            title = ""

    title = _clean_title(title) or _fallback_title(kind=src.kind, locator=src.locator)
    db.set_source_title(source_id=source_id, title=title)
    return title


