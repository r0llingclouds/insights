from __future__ import annotations

import os

from insights.llm import AnthropicClient, ChatMessage, OpenAIClient
from insights.storage.db import Database


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
    p = provider.strip().lower()
    if p == "openai":
        client = OpenAIClient()
        used_model = model or "gpt-4o-mini"
    elif p == "anthropic":
        client = AnthropicClient()
        used_model = model or os.getenv("INSIGHTS_TITLE_MODEL") or "claude-sonnet-4-20250514"
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
    - This is best-effort; callers should catch exceptions and proceed.
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
    if not plain or not plain.strip():
        return None

    title = generate_title(
        content=plain,
        provider=provider,
        model=model,
        max_content_chars=max_content_chars,
    )
    if not title:
        return None
    db.set_source_title(source_id=source_id, title=title)
    return title


