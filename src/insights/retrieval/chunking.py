"""Text chunking for RAG with sentence boundary preservation."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """A chunk of text with metadata."""
    index: int
    content: str
    char_offset: int
    char_length: int


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving boundaries."""
    # Simple sentence splitting on common terminators
    # Handles: . ! ? followed by space or end of string
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    preserve_sentences: bool = True,
) -> list[Chunk]:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to chunk
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        preserve_sentences: If True, try to break at sentence boundaries

    Returns:
        List of Chunk objects with content and position metadata
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    if len(text) <= chunk_size:
        return [Chunk(index=0, content=text, char_offset=0, char_length=len(text))]

    chunks: list[Chunk] = []

    if preserve_sentences:
        sentences = _split_sentences(text)
        if not sentences:
            # Fallback to character-based chunking
            return _chunk_by_chars(text, chunk_size, chunk_overlap)

        current_chunk: list[str] = []
        current_length = 0
        chunk_start_offset = 0
        text_offset = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence would exceed chunk_size
            if current_length + sentence_len > chunk_size and current_chunk:
                # Save current chunk
                chunk_content = " ".join(current_chunk)
                chunks.append(Chunk(
                    index=len(chunks),
                    content=chunk_content,
                    char_offset=chunk_start_offset,
                    char_length=len(chunk_content),
                ))

                # Start new chunk with overlap
                # Keep sentences from the end to provide overlap
                overlap_sentences: list[str] = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s) + 1  # +1 for space
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1 if current_chunk else 0
                chunk_start_offset = text_offset - current_length if current_length > 0 else text_offset

            current_chunk.append(sentence)
            current_length += sentence_len + (1 if current_chunk else 0)  # +1 for space
            text_offset += sentence_len + 1  # +1 for space between sentences

        # Don't forget the last chunk
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            chunks.append(Chunk(
                index=len(chunks),
                content=chunk_content,
                char_offset=chunk_start_offset,
                char_length=len(chunk_content),
            ))
    else:
        chunks = _chunk_by_chars(text, chunk_size, chunk_overlap)

    return chunks


def _chunk_by_chars(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Simple character-based chunking with overlap."""
    chunks: list[Chunk] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_content = text[start:end]

        chunks.append(Chunk(
            index=len(chunks),
            content=chunk_content,
            char_offset=start,
            char_length=len(chunk_content),
        ))

        # Move start forward, accounting for overlap
        start = end - chunk_overlap
        if start >= len(text) - chunk_overlap:
            break

    return chunks
