from __future__ import annotations

from insights.llm import AnthropicClient, ChatMessage, OpenAIClient
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
    p = provider.strip().lower()
    if p == "openai":
        client = OpenAIClient()
        used_model = model or "gpt-4o-mini"
    elif p == "anthropic":
        client = AnthropicClient()
        used_model = model or "claude-3-5-haiku-latest"
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
    - This should never be treated as critical-path; callers are expected to catch errors.
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
    if not plain or not plain.strip():
        return None

    desc = generate_description(
        content=plain,
        provider=provider,
        model=model,
        max_content_chars=max_content_chars,
    )
    if not desc:
        return None
    db.set_source_description(source_id=source_id, description=desc)
    return desc


