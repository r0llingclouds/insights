from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

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

    def generate_stream(
        self,
        *,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        timeout_s: float = 120.0,
    ) -> Iterator[str]: ...

    def close(self) -> None: ...


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
        self._client: httpx.Client | None = None

    def _get_client(self, timeout_s: float) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=timeout_s)
        return self._client

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

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
        client = self._get_client(timeout_s)
        resp = client.post(url, headers=self._headers(), json=payload)
        _raise_for_status(resp)
        data = resp.json()
        text = (data["choices"][0]["message"].get("content") or "").strip()
        usage = data.get("usage")
        if not text:
            raise RuntimeError("OpenAI returned an empty response")
        return LLMResponse(provider=self.provider, model=model, text=text, usage=usage)

    def generate_stream(
        self,
        *,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        timeout_s: float = 120.0,
    ) -> Iterator[str]:
        url = f"{self._base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        client = self._get_client(timeout_s)
        with client.stream("POST", url, headers=self._headers(), json=payload) as resp:
            _raise_for_status(resp)
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]  # Remove "data: " prefix
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key or require_env("ANTHROPIC_API_KEY")
        self._base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self._client: httpx.Client | None = None

    def _get_client(self, timeout_s: float) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=timeout_s)
        return self._client

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _prepare_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[list[dict[str, Any]], str | None]:
        system_parts: list[str] = []
        anthro_messages: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                anthro_messages.append({"role": m.role, "content": m.content})
        system = "\n\n".join(system_parts) if system_parts else None
        return anthro_messages, system

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

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
        anthro_messages, system = self._prepare_messages(messages)

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthro_messages,
        }
        if system:
            payload["system"] = system

        client = self._get_client(timeout_s)
        resp = client.post(url, headers=self._headers(), json=payload)
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

    def generate_stream(
        self,
        *,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        timeout_s: float = 120.0,
    ) -> Iterator[str]:
        url = f"{self._base_url}/v1/messages"
        anthro_messages, system = self._prepare_messages(messages)

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthro_messages,
            "stream": True,
        }
        if system:
            payload["system"] = system

        client = self._get_client(timeout_s)
        with client.stream("POST", url, headers=self._headers(), json=payload) as resp:
            _raise_for_status(resp)
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]  # Remove "data: " prefix
                try:
                    data = json.loads(data_str)
                    event_type = data.get("type")
                    if event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text")
                            if text:
                                yield text
                    elif event_type == "message_stop":
                        break
                except json.JSONDecodeError:
                    continue


