from __future__ import annotations

import os
import re

from insights.llm import AnthropicClient, ChatMessage, OpenAIClient
from insights.llm.routing import pick_anthropic_model
from insights.storage.db import Database


DESCRIBE_SYSTEM = """\
You generate concise, searchable one-liner descriptions of content.

Requirements:
- Under 150 characters (hard limit 200; if longer, it will be truncated)
- Capture the main topic, thesis, or key takeaway
- Use specific terms someone might search for
- Avoid generic phrases like \"discusses various topics\"

Respond with ONLY the description, no quotes or prefixes."""


def _truncate_content(content: str, *, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    half = max_chars // 2
    return content[:half] + "\n\n[...content truncated...]\n\n" + content[-half:]


def generate_description(
    *,
    content: str,
    provider: str = "anthropic",
    model: str | None = None,
    max_content_chars: int = 8000,
) -> str:
    """
    Generate a short, searchable one-liner description for content.
    """
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
                env_model=os.getenv("INSIGHTS_DESCRIBE_MODEL"),
                default_model="claude-sonnet-4-20250514",
                content_len=content_len,
                large_model="claude-haiku-4-5-20251001",
            )
    else:
        raise ValueError("provider must be 'openai' or 'anthropic'")

    truncated = _truncate_content(content, max_chars=max(1000, int(max_content_chars)))

    user = (
        "Generate a one-liner description for this content:\n\n---\n"
        f"{truncated}\n"
        "---\n\nOne-liner:"
    )
    messages = [
        ChatMessage(role="system", content=DESCRIBE_SYSTEM),
        ChatMessage(role="user", content=user),
    ]
    resp = client.generate(messages=messages, model=used_model, temperature=0.2, max_tokens=100)

    desc = (resp.text or "").strip().strip("\"'")
    # Collapse whitespace/newlines to a single line.
    desc = " ".join(desc.split())
    if len(desc) > 200:
        desc = desc[:197].rstrip() + "..."
    return desc


DEFAULT_EXTRACTOR_PREFERENCE: list[str] = ["firecrawl", "docling", "assemblyai"]


_RE_SENT_END = re.compile(r"([.!?])\s+")


def _clean_one_liner(text: str, *, max_chars: int = 200) -> str:
    s = " ".join((text or "").split()).strip().strip("\"'")
    if not s:
        return ""
    if len(s) > max_chars:
        s = s[: max_chars - 3].rstrip() + "..."
    return s


def _fallback_description(*, plain_text: str | None, title: str | None, locator: str) -> str:
    txt = (plain_text or "").strip()
    if txt:
        # Prefer first paragraph; fall back to first sentence.
        para = ""
        for p in txt.split("\n\n"):
            p = p.strip()
            if p:
                para = p
                break
        if not para:
            para = txt
        # If paragraph is huge, cut early.
        para = para[:1000]
        # Take first sentence if it exists.
        m = _RE_SENT_END.search(para)
        if m:
            para = para[: m.end()].strip()
        return _clean_one_liner(para) or _clean_one_liner(title or "") or _clean_one_liner(locator)
    return _clean_one_liner(title or "") or _clean_one_liner(locator) or "Source"


def ensure_source_description(
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
    Ensure `sources.description` is populated for a source.

    - If a description already exists and `force` is False, it's returned unchanged.
    - If missing, we generate from the cached plain text and persist it.
    - If generation fails/empty, we persist a deterministic fallback (never leaves description NULL).
    """
    src = db.get_source_by_id(source_id)
    if src is None:
        return None
    if src.description and not force:
        return src.description

    plain: str | None = None
    if source_version_id:
        plain = db.get_document_plain_text_by_source_version(source_version_id)
    if not plain:
        plain = db.get_latest_plain_text_for_source(
            source_id=source_id,
            extractor_preference=extractor_preference or DEFAULT_EXTRACTOR_PREFERENCE,
        )

    desc = ""
    if plain and plain.strip():
        try:
            desc = generate_description(
                content=plain,
                provider=provider,
                model=model,
                max_content_chars=max_content_chars,
            )
        except Exception:
            desc = ""

    desc = _clean_one_liner(desc) or _fallback_description(plain_text=plain, title=src.title, locator=src.locator)
    db.set_source_description(source_id=source_id, description=desc)
    return desc


