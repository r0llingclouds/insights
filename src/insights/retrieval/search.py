"""Semantic search functionality."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from insights.retrieval.chunking import Chunk, chunk_text
from insights.retrieval.embeddings import embed_single, embed_texts
from insights.retrieval.store import StoredChunk, VectorStore


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A semantic search result."""
    source_id: str
    source_version_id: str
    chunk_index: int
    content: str
    char_offset: int
    char_length: int
    score: float  # 0-1, higher is better (1 - distance for cosine)


def semantic_search(
    query: str,
    *,
    store: VectorStore,
    n_results: int = 10,
    source_ids: list[str] | None = None,
    embedding_model: str = "text-embedding-3-small",
) -> list[SearchResult]:
    """
    Perform semantic search over indexed documents.

    Args:
        query: The search query
        store: Vector store to search
        n_results: Maximum number of results
        source_ids: Optional filter to specific sources
        embedding_model: Model to use for query embedding

    Returns:
        List of SearchResult objects ordered by relevance
    """
    if not query.strip():
        return []

    # Embed the query
    query_embedding = embed_single(query, model=embedding_model)

    # Search the store
    chunks = store.query(
        query_embedding=query_embedding,
        n_results=n_results,
        source_ids=source_ids,
    )

    # Convert to search results
    results: list[SearchResult] = []
    for chunk in chunks:
        # Convert cosine distance to similarity score (0-1, higher is better)
        score = 1.0 - (chunk.distance or 0.0) if chunk.distance is not None else 1.0
        results.append(SearchResult(
            source_id=chunk.source_id,
            source_version_id=chunk.source_version_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            char_offset=chunk.char_offset,
            char_length=chunk.char_length,
            score=score,
        ))

    return results


def index_source(
    *,
    source_id: str,
    source_version_id: str,
    plain_text: str,
    store: VectorStore,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    embedding_model: str = "text-embedding-3-small",
    progress: Callable[[str], None] | None = None,
) -> int:
    """
    Index a source document for semantic search.

    Args:
        source_id: ID of the source
        source_version_id: ID of the source version
        plain_text: The document text to index
        store: Vector store to add chunks to
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
        embedding_model: Model to use for embeddings
        progress: Optional progress callback

    Returns:
        Number of chunks indexed
    """
    if not plain_text.strip():
        return 0

    # Chunk the text
    if progress:
        progress(f"Chunking text ({len(plain_text):,} chars)...")
    chunks = chunk_text(
        plain_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        preserve_sentences=True,
    )

    if not chunks:
        return 0

    if progress:
        progress(f"Generated {len(chunks)} chunks")

    # Generate embeddings
    if progress:
        progress(f"Generating embeddings for {len(chunks)} chunks...")

    chunk_texts = [c.content for c in chunks]
    embedding_result = embed_texts(chunk_texts, model=embedding_model)

    if progress:
        progress(f"Embeddings generated ({embedding_result.total_tokens} tokens)")

    # Store in vector store
    chunk_dicts = [
        {
            "index": c.index,
            "content": c.content,
            "char_offset": c.char_offset,
            "char_length": c.char_length,
        }
        for c in chunks
    ]

    count = store.add_chunks(
        source_id=source_id,
        source_version_id=source_version_id,
        chunks=chunk_dicts,
        embeddings=embedding_result.embeddings,
    )

    if progress:
        progress(f"Indexed {count} chunks")

    return count
