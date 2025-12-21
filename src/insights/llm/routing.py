from __future__ import annotations

import os


def _env_int(name: str) -> int | None:
    v = os.getenv(name)
    if not v:
        return None
    try:
        n = int(v)
    except ValueError:
        return None
    if n <= 0:
        return None
    return n


def pick_anthropic_model(
    *,
    env_model: str | None,
    default_model: str,
    content_len: int,
    cutoff: int = 10_000,
    large_model: str = "claude-haiku-4-5",
) -> str:
    """
    Pick an Anthropic model with a large-content optimization.

    Rules:
    1) If env_model is a non-empty string -> return it (env wins; disables auto-switch)
    2) Else if content_len > cutoff -> return large_model
    3) Else -> return default_model

    Optional env override:
    - INSIGHTS_LARGE_CONTENT_CUTOFF_CHARS (positive int)
    """
    if env_model and env_model.strip():
        return env_model.strip()

    cutoff_env = _env_int("INSIGHTS_LARGE_CONTENT_CUTOFF_CHARS")
    effective_cutoff = cutoff_env if cutoff_env is not None else cutoff

    if int(content_len) > int(effective_cutoff):
        return large_model
    return default_model


