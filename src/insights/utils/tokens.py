from __future__ import annotations

import os
from functools import lru_cache


def estimate_tokens(text: str) -> int:
    """
    Exact token counter using tiktoken.

    Notes:
    - The DB stores this value in `documents.token_count`.
    - Encoding selection:
        1) INSIGHTS_TOKEN_ENCODING if set (e.g. 'o200k_base', 'cl100k_base')
        2) Default: 'o200k_base'
        3) Fallback: 'cl100k_base' (if 'o200k_base' isn't available in the installed tiktoken build)
    """
    if not text:
        return 0
    enc = _get_encoding()
    return len(enc.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to at most max_tokens tokens using the configured tiktoken encoding.

    This is used to enforce a hard context budget for FULL-mode prompting.
    """
    if not text:
        return ""
    mt = int(max_tokens)
    if mt <= 0:
        return ""
    enc = _get_encoding()
    toks = enc.encode(text)
    if len(toks) <= mt:
        return text
    return enc.decode(toks[:mt])


@lru_cache(maxsize=1)
def _get_encoding():
    try:
        import tiktoken  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "tiktoken is required for exact token counting but is not available. "
            "Install dependencies (e.g. `uv sync`) and try again."
        ) from e

    name = (os.getenv("INSIGHTS_TOKEN_ENCODING") or "").strip() or "o200k_base"
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        # Conservative fallback for environments where o200k_base isn't shipped.
        return tiktoken.get_encoding("cl100k_base")


