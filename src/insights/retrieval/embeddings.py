"""Embedding generation using OpenAI API."""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from insights.config import require_env


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Result of embedding generation."""
    embeddings: list[list[float]]
    model: str
    total_tokens: int


def embed_texts(
    texts: list[str],
    *,
    model: str = "text-embedding-3-small",
    api_key: str | None = None,
    batch_size: int = 100,
) -> EmbeddingResult:
    """
    Generate embeddings for a list of texts using OpenAI API.

    Args:
        texts: List of texts to embed
        model: OpenAI embedding model to use
        api_key: Optional API key (defaults to OPENAI_API_KEY env var)
        batch_size: Number of texts to embed per API call

    Returns:
        EmbeddingResult with embeddings and metadata
    """
    if not texts:
        return EmbeddingResult(embeddings=[], model=model, total_tokens=0)

    key = api_key or require_env("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")

    all_embeddings: list[list[float]] = []
    total_tokens = 0

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{base_url}/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": batch,
                },
            )

        if resp.status_code >= 400:
            raise RuntimeError(f"OpenAI embeddings API error {resp.status_code}: {resp.text}")

        data = resp.json()
        total_tokens += data.get("usage", {}).get("total_tokens", 0)

        # Extract embeddings in order
        embeddings_data = sorted(data["data"], key=lambda x: x["index"])
        batch_embeddings = [item["embedding"] for item in embeddings_data]
        all_embeddings.extend(batch_embeddings)

    return EmbeddingResult(
        embeddings=all_embeddings,
        model=model,
        total_tokens=total_tokens,
    )


def embed_single(
    text: str,
    *,
    model: str = "text-embedding-3-small",
    api_key: str | None = None,
) -> list[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Text to embed
        model: OpenAI embedding model to use
        api_key: Optional API key

    Returns:
        Embedding vector as list of floats
    """
    result = embed_texts([text], model=model, api_key=api_key)
    return result.embeddings[0] if result.embeddings else []
