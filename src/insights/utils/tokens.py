from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """
    Fast, dependency-free token estimator.

    Rule of thumb: ~4 characters per token for English-ish text.
    This is not exact, but it's good enough for deciding full-context vs retrieval.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


