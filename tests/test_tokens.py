from __future__ import annotations


def test_estimate_tokens_matches_tiktoken_default(monkeypatch) -> None:
    # Ensure default path uses o200k_base when available, otherwise cl100k_base.
    # We compare against tiktoken directly to confirm we're not using a heuristic.
    monkeypatch.delenv("INSIGHTS_TOKEN_ENCODING", raising=False)

    import tiktoken

    from insights.utils.tokens import estimate_tokens

    try:
        enc = tiktoken.get_encoding("o200k_base")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    texts = [
        "hello world",
        "Café naïve résumé — déjà vu.",
        "https://example.com/a/b?c=1&d=two#frag",
        "x" * 257,
        "Line1\nLine2\n\nLine4",
    ]

    # Must match exact tokenizer count for all samples.
    for t in texts:
        assert estimate_tokens(t) == len(enc.encode(t))

    # And must differ from the old heuristic for at least one sample (sanity check).
    def heuristic(s: str) -> int:
        return 0 if not s else max(1, (len(s) + 3) // 4)

    assert any(len(enc.encode(t)) != heuristic(t) for t in texts)


def test_estimate_tokens_respects_env_encoding(monkeypatch) -> None:
    import tiktoken

    from insights.utils.tokens import estimate_tokens

    monkeypatch.setenv("INSIGHTS_TOKEN_ENCODING", "cl100k_base")
    enc = tiktoken.get_encoding("cl100k_base")
    text = "This is a test string with some punctuation: (a), [b], {c}."
    assert estimate_tokens(text) == len(enc.encode(text))


