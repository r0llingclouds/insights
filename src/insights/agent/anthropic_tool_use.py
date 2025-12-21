from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from insights.config import require_env


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    try:
        payload = resp.json()
        detail = json.dumps(payload, ensure_ascii=False)
    except Exception:
        detail = resp.text
    raise RuntimeError(f"Anthropic API error {resp.status_code}: {detail}")


StopReason = Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]


@dataclass(frozen=True, slots=True)
class AnthropicMessageResponse:
    stop_reason: StopReason
    content: list[dict[str, Any]]
    usage: dict[str, Any] | None


class AnthropicToolUseClient:
    """
    Minimal Anthropic Messages API client that supports tool-use.

    We intentionally use httpx directly (no anthropic SDK dependency).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        anthropic_version: str = "2023-06-01",
        timeout_s: float = 120.0,
    ) -> None:
        self._api_key = api_key or require_env("ANTHROPIC_API_KEY")
        self._base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self._anthropic_version = anthropic_version
        self._timeout_s = timeout_s

    def create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.2,
    ) -> AnthropicMessageResponse:
        url = f"{self._base_url}/v1/messages"
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": messages,
            "tools": tools,
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self._anthropic_version,
            "content-type": "application/json",
        }
        with httpx.Client(timeout=self._timeout_s) as client:
            resp = client.post(url, headers=headers, json=payload)
        _raise_for_status(resp)
        data = resp.json()

        stop_reason_raw = data.get("stop_reason")
        if not isinstance(stop_reason_raw, str) or not stop_reason_raw:
            raise RuntimeError("Anthropic response missing stop_reason")

        content = data.get("content") or []
        if not isinstance(content, list):
            raise RuntimeError("Anthropic response content is not a list")

        usage = data.get("usage")
        if usage is not None and not isinstance(usage, dict):
            usage = None

        # Preserve unknown stop reasons as strings, but type-narrow known ones.
        stop_reason: StopReason
        if stop_reason_raw in {"end_turn", "tool_use", "max_tokens", "stop_sequence"}:
            stop_reason = stop_reason_raw  # type: ignore[assignment]
        else:
            # Treat unknown stop reasons as max_tokens-like (loop will stop).
            stop_reason = "max_tokens"

        # Normalize blocks to dicts.
        norm_blocks: list[dict[str, Any]] = []
        for b in content:
            if isinstance(b, dict):
                norm_blocks.append(b)

        return AnthropicMessageResponse(stop_reason=stop_reason, content=norm_blocks, usage=usage)


