from __future__ import annotations

from insights.llm.routing import pick_anthropic_model


def test_env_wins_over_auto_switch(monkeypatch) -> None:
    monkeypatch.setenv("INSIGHTS_LARGE_CONTENT_CUTOFF_CHARS", "10")
    model = pick_anthropic_model(
        env_model="my-custom-model",
        default_model="sonnet",
        content_len=10_000_000,
        large_model="haiku",
    )
    assert model == "my-custom-model"


def test_large_content_uses_haiku(monkeypatch) -> None:
    monkeypatch.delenv("INSIGHTS_LARGE_CONTENT_CUTOFF_CHARS", raising=False)
    model = pick_anthropic_model(
        env_model=None,
        default_model="sonnet",
        content_len=10_001,
        cutoff=10_000,
        large_model="haiku",
    )
    assert model == "haiku"


def test_small_content_uses_default(monkeypatch) -> None:
    monkeypatch.delenv("INSIGHTS_LARGE_CONTENT_CUTOFF_CHARS", raising=False)
    model = pick_anthropic_model(
        env_model=None,
        default_model="sonnet",
        content_len=10_000,
        cutoff=10_000,
        large_model="haiku",
    )
    assert model == "sonnet"


def test_cutoff_env_overrides_default(monkeypatch) -> None:
    monkeypatch.setenv("INSIGHTS_LARGE_CONTENT_CUTOFF_CHARS", "5")
    model = pick_anthropic_model(
        env_model=None,
        default_model="sonnet",
        content_len=6,
        cutoff=10_000,
        large_model="haiku",
    )
    assert model == "haiku"


