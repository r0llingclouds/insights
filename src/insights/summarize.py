from __future__ import annotations

import os

from insights.llm import AnthropicClient, ChatMessage, OpenAIClient
from insights.llm.routing import pick_anthropic_model


SUMMARY_SYSTEM = """\
You write a compact summary of content for quick recall.

Requirements:
- 3–7 bullet points max
- Prefer concrete facts / claims / key takeaways
- Avoid fluff
- Output as Markdown bullets only (lines starting with '- ')
"""


def _truncate_content(content: str, *, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    half = max_chars // 2
    return content[:half] + "\n\n[...content truncated...]\n\n" + content[-half:]


def generate_summary(
    *,
    content: str,
    provider: str = "anthropic",
    model: str | None = None,
    max_content_chars: int = 12000,
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
                env_model=os.getenv("INSIGHTS_SUMMARY_MODEL"),
                default_model="claude-sonnet-4-20250514",
                content_len=content_len,
                large_model="claude-haiku-4-5-20251001",
            )
    else:
        raise ValueError("provider must be 'openai' or 'anthropic'")

    truncated = _truncate_content(content, max_chars=max(2000, int(max_content_chars)))
    user = (
        "Summarize this content as Markdown bullets:\n\n---\n"
        f"{truncated}\n"
        "---\n\nBullets:"
    )
    messages = [
        ChatMessage(role="system", content=SUMMARY_SYSTEM),
        ChatMessage(role="user", content=user),
    ]
    resp = client.generate(messages=messages, model=used_model, temperature=0.2, max_tokens=200)
    text = (resp.text or "").strip()
    # Normalize to bullets.
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    bullets = []
    for ln in lines:
        if ln.startswith("- "):
            bullets.append(ln)
        else:
            bullets.append(f"- {ln.lstrip('-').strip()}")
    # Hard cap output size.
    out = "\n".join(bullets[:7]).strip()
    return out


