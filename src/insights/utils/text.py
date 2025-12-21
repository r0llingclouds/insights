from __future__ import annotations

import re

_RE_CODE_FENCE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_RE_INLINE_CODE = re.compile(r"`([^`]+)`")
_RE_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_RE_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_RE_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_RE_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_RE_LIST_PREFIX = re.compile(r"^\s*([-*+]|\d+\.)\s+", re.MULTILINE)
_RE_HRULE = re.compile(r"^\s{0,3}(-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)


def markdown_to_text(md: str) -> str:
    """
    Best-effort Markdown → plain text normalization for LLM context.
    Keeps code content by stripping fences but not dropping inner text.
    """
    # Replace images with alt text.
    md = _RE_MD_IMAGE.sub(lambda m: m.group(1) or "", md)
    # Replace links with visible text.
    md = _RE_MD_LINK.sub(lambda m: m.group(1), md)
    # Remove heading markers.
    md = _RE_HEADING.sub("", md)
    # Remove horizontal rules.
    md = _RE_HRULE.sub("", md)
    # Strip blockquote markers.
    md = _RE_BLOCKQUOTE.sub("", md)

    # Replace fenced code blocks with their inner content (remove ```lang ... ```).
    def _strip_fence(match: re.Match[str]) -> str:
        block = match.group(0)
        lines = block.splitlines()
        if len(lines) >= 2:
            inner = "\n".join(lines[1:-1])
            return inner
        return ""

    md = _RE_CODE_FENCE.sub(_strip_fence, md)
    # Inline code → content.
    md = _RE_INLINE_CODE.sub(lambda m: m.group(1), md)
    # List prefixes removed (keep content).
    md = _RE_LIST_PREFIX.sub("", md)

    # Collapse excessive blank lines.
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


