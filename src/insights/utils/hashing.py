from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def content_hash_for_file(path: Path) -> str:
    """
    Stable-ish cache key that changes when file content or metadata changes.

    The underlying content hash is enough, but including size/mtime makes it
    easier to detect changes even if a caller wants to avoid hashing large files
    in the future.
    """
    stat = path.stat()
    content = sha256_file(path)
    meta = f"{stat.st_size}:{int(stat.st_mtime)}:{content}"
    return sha256_text(meta)


