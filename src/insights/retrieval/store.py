"""Vector store using numpy and JSON for semantic search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class StoredChunk:
    """A chunk stored in the vector store."""
    id: str
    source_id: str
    source_version_id: str
    chunk_index: int
    content: str
    char_offset: int
    char_length: int
    distance: float | None = None


class VectorStore:
    """
    Simple vector store for semantic search using numpy.

    Stores embeddings as numpy arrays and metadata as JSON.
    Uses cosine similarity for search.
    """

    def __init__(self, persist_dir: Path) -> None:
        """
        Initialize the vector store.

        Args:
            persist_dir: Directory for persistent storage
        """
        self._persist_dir = persist_dir
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._embeddings_file = persist_dir / "embeddings.npy"
        self._metadata_file = persist_dir / "metadata.json"

        self._embeddings: np.ndarray | None = None
        self._metadata: list[dict[str, Any]] = []

        self._load()

    def _load(self) -> None:
        """Load existing data from disk."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
            except Exception:
                self._metadata = []

        if self._embeddings_file.exists() and self._metadata:
            try:
                self._embeddings = np.load(str(self._embeddings_file))
            except Exception:
                self._embeddings = None

    def _save(self) -> None:
        """Save data to disk."""
        with open(self._metadata_file, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False)

        if self._embeddings is not None and len(self._embeddings) > 0:
            np.save(str(self._embeddings_file), self._embeddings)

    def add_chunks(
        self,
        *,
        source_id: str,
        source_version_id: str,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        """
        Add chunks with their embeddings to the store.

        Args:
            source_id: ID of the source document
            source_version_id: ID of the source version
            chunks: List of chunk dicts with content and metadata
            embeddings: Corresponding embeddings for each chunk

        Returns:
            Number of chunks added
        """
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        # First, delete any existing chunks for this source version
        self.delete_source_version(source_version_id)

        # Add new chunks
        new_metadata = []
        for chunk in chunks:
            chunk_id = f"{source_version_id}_{chunk['index']}"
            new_metadata.append({
                "id": chunk_id,
                "source_id": source_id,
                "source_version_id": source_version_id,
                "chunk_index": chunk["index"],
                "content": chunk["content"],
                "char_offset": chunk["char_offset"],
                "char_length": chunk["char_length"],
            })

        # Convert embeddings to numpy array
        new_embeddings = np.array(embeddings, dtype=np.float32)

        # Append to existing data
        if self._embeddings is not None and len(self._embeddings) > 0:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])
        else:
            self._embeddings = new_embeddings

        self._metadata.extend(new_metadata)
        self._save()

        return len(chunks)

    def query(
        self,
        query_embedding: list[float],
        *,
        n_results: int = 10,
        source_ids: list[str] | None = None,
    ) -> list[StoredChunk]:
        """
        Query for similar chunks using cosine similarity.

        Args:
            query_embedding: Embedding of the query
            n_results: Maximum number of results to return
            source_ids: Optional filter to specific sources

        Returns:
            List of StoredChunk objects ordered by similarity
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        # Convert query to numpy array
        query_vec = np.array(query_embedding, dtype=np.float32)

        # Normalize vectors for cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        embeddings_norm = self._embeddings / (np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10)

        # Compute cosine similarities
        similarities = np.dot(embeddings_norm, query_norm)

        # Filter by source IDs if provided
        if source_ids:
            source_set = set(source_ids)
            mask = np.array([m.get("source_id") in source_set for m in self._metadata])
            similarities = np.where(mask, similarities, -1.0)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:n_results]

        chunks: list[StoredChunk] = []
        for idx in top_indices:
            if similarities[idx] < 0:
                continue  # Filtered out

            metadata = self._metadata[idx]
            distance = 1.0 - similarities[idx]  # Convert similarity to distance

            chunks.append(StoredChunk(
                id=metadata.get("id", ""),
                source_id=metadata.get("source_id", ""),
                source_version_id=metadata.get("source_version_id", ""),
                chunk_index=metadata.get("chunk_index", 0),
                content=metadata.get("content", ""),
                char_offset=metadata.get("char_offset", 0),
                char_length=metadata.get("char_length", 0),
                distance=float(distance),
            ))

        return chunks

    def delete_source_version(self, source_version_id: str) -> None:
        """Delete all chunks for a source version."""
        if not self._metadata:
            return

        # Find indices to keep
        keep_indices = [
            i for i, m in enumerate(self._metadata)
            if m.get("source_version_id") != source_version_id
        ]

        if len(keep_indices) == len(self._metadata):
            return  # Nothing to delete

        # Filter metadata
        self._metadata = [self._metadata[i] for i in keep_indices]

        # Filter embeddings
        if self._embeddings is not None and len(keep_indices) > 0:
            self._embeddings = self._embeddings[keep_indices]
        else:
            self._embeddings = None

        self._save()

    def delete_source(self, source_id: str) -> None:
        """Delete all chunks for a source."""
        if not self._metadata:
            return

        # Find indices to keep
        keep_indices = [
            i for i, m in enumerate(self._metadata)
            if m.get("source_id") != source_id
        ]

        if len(keep_indices) == len(self._metadata):
            return

        self._metadata = [self._metadata[i] for i in keep_indices]

        if self._embeddings is not None and len(keep_indices) > 0:
            self._embeddings = self._embeddings[keep_indices]
        else:
            self._embeddings = None

        self._save()

    def count(self) -> int:
        """Return the total number of chunks in the store."""
        return len(self._metadata)

    def get_indexed_sources(self) -> set[str]:
        """Get set of source IDs that have indexed chunks."""
        return {m.get("source_id") for m in self._metadata if m.get("source_id")}
