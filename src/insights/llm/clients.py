from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from insights.config import require_env
from insights.llm.types import ChatMessage


@dataclass(frozen=True, slots=True)
class LLMResponse:
    provider: str
    model: str
    text: str
    usage: dict[str, Any] | None


class LLMClient(Protocol):
    provider: str

    def generate(
        self,
        *,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        timeout_s: float = 120.0,
    ) -> LLMResponse: ...


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    try:
        payload = resp.json()
        detail = json.dumps(payload, ensure_ascii=False)
    except Exception:
        detail = resp.text
    raise RuntimeError(f"LLM API error {resp.status_code}: {detail}")


class OpenAIClient:
    provider = "openai"

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key or require_env("OPENAI_API_KEY")
        self._base_url = (base_url or "https://api.openai.com").rstrip("/")

    def generate(
        self,
        *,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        timeout_s: float = 120.0,
    ) -> LLMResponse:
        url = f"{self._base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(url, headers=headers, json=payload)
        _raise_for_status(resp)
        data = resp.json()
        text = (data["choices"][0]["message"].get("content") or "").strip()
        usage = data.get("usage")
        if not text:
            raise RuntimeError("OpenAI returned an empty response")
        return LLMResponse(provider=self.provider, model=model, text=text, usage=usage)


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key or require_env("ANTHROPIC_API_KEY")
        self._base_url = (base_url or "https://api.anthropic.com").rstrip("/")

    def generate(
        self,
        *,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        timeout_s: float = 120.0,
    ) -> LLMResponse:
        url = f"{self._base_url}/v1/messages"

        system_parts: list[str] = []
        anthro_messages: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                anthro_messages.append({"role": m.role, "content": m.content})

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthro_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(url, headers=headers, json=payload)
        _raise_for_status(resp)
        data = resp.json()

        content_blocks = data.get("content") or []
        texts: list[str] = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    texts.append(t)
        text = "\n".join(texts).strip()
        usage = data.get("usage")
        if not text:
            raise RuntimeError("Anthropic returned an empty response")
        return LLMResponse(provider=self.provider, model=model, text=text, usage=usage)


