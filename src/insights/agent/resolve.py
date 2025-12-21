from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from insights.ingest.detect import detect_source
from insights.storage.db import Database
from insights.storage.models import Source, SourceKind


_HEX_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")
_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,16}$")


def _is_hex_id(value: str) -> bool:
    return bool(_HEX_ID_RE.fullmatch(value))


def _looks_like_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _looks_like_path(value: str) -> bool:
    # Treat any explicit path separators or common path prefixes as a path.
    return (
        ("/" in value)
        or ("\\" in value)
        or value.startswith(("~", "./", "../"))
        or value.startswith(os.sep)
    )


@dataclass(frozen=True, slots=True)
class ResolveResult:
    found: bool
    source: Source | None
    ambiguous: bool
    suggestions: list[Source]


def resolve_source_strict(db: Database, ref: str) -> Source | None:
    r = (ref or "").strip()
    if not r:
        return None

    if _is_hex_id(r):
        return db.get_source_by_id(r)

    if _looks_like_url(r):
        detected = detect_source(r, forced_type="auto")
        try:
            return db.get_source_by_kind_locator(kind=detected.kind, locator=detected.locator)
        except KeyError:
            return None

    # Raw YouTube id
    if _YOUTUBE_ID_RE.fullmatch(r):
        try:
            return db.get_source_by_kind_locator(kind=SourceKind.YOUTUBE, locator=r)
        except KeyError:
            pass

    # Explicit file path (not a basename)
    if _looks_like_path(r):
        try:
            detected = detect_source(r, forced_type="file")
        except Exception:
            return None
        try:
            return db.get_source_by_kind_locator(kind=detected.kind, locator=detected.locator)
        except KeyError:
            return None

    return None


def resolve_source_candidates(*, db_path: Any, ref: str, limit: int = 8) -> list[dict[str, Any]]:
    """
    Return candidate source rows (dict) for a fuzzy reference.

    We query via sqlite directly so we can:
    - do LIKE across locator/title/description
    - do basename matching for file sources: locator LIKE '%/<ref>'
    """
    q = (ref or "").strip().lower()
    if not q:
        return []
    lim = max(1, min(int(limit), 50))

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(sources);").fetchall()]
        has_description = "description" in cols

        like = f"%{q}%"
        params: list[Any] = []

        clauses: list[str] = [
            "lower(locator) LIKE ?",
            "lower(coalesce(title,'')) LIKE ?",
        ]
        params.extend([like, like])
        if has_description:
            clauses.append("lower(coalesce(description,'')) LIKE ?")
            params.append(like)

        # Basename match for file sources (handles user passing 'onepager.pdf')
        clauses.append("(kind = 'file' AND (lower(locator) LIKE ? OR lower(locator) LIKE ?))")
        params.append(f"%/{q}")
        params.append(f"%\\\\{q}")

        sql = f"""
        SELECT id, kind, locator, title, updated_at{", description" if has_description else ""}
        FROM sources
        WHERE ({' OR '.join(clauses)})
        ORDER BY updated_at DESC
        LIMIT ?;
        """
        params.append(lim)
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()

    # Score candidates so we return the most likely matches first.
    scored: list[tuple[int, dict[str, Any]]] = []
    for r in rows:
        kind = str(r.get("kind") or "")
        locator = str(r.get("locator") or "")
        title = str(r.get("title") or "")
        desc = str(r.get("description") or "")

        locator_l = locator.lower()
        title_l = title.lower()
        desc_l = desc.lower()

        score = 0
        if q == locator_l:
            score += 120
        if q and kind == "file":
            if locator_l.endswith("/" + q) or locator_l.endswith("\\" + q):
                score += 110
        if q == title_l and q:
            score += 100
        if q in title_l and q:
            score += 60
        if q in desc_l and q:
            score += 30
        if q in locator_l and q:
            score += 20
        scored.append((score, r))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in scored]


def resolve_source(*, db: Database, ref: str, suggestions_limit: int = 5) -> ResolveResult:
    """
    Resolve a source ref by id/url/path/title/basename.
    """
    strict = resolve_source_strict(db, ref)
    if strict is not None:
        return ResolveResult(found=True, source=strict, ambiguous=False, suggestions=[])

    candidates = resolve_source_candidates(db_path=db.path, ref=ref, limit=max(2, suggestions_limit))
    if not candidates:
        return ResolveResult(found=False, source=None, ambiguous=False, suggestions=[])

    # Convert candidates into Source objects via DB lookups (cheap: small N).
    sources: list[Source] = []
    for c in candidates[: suggestions_limit]:
        sid = str(c.get("id") or "")
        if not sid:
            continue
        s = db.get_source_by_id(sid)
        if s:
            sources.append(s)

    if len(sources) == 1:
        return ResolveResult(found=True, source=sources[0], ambiguous=False, suggestions=[])
    return ResolveResult(found=False, source=None, ambiguous=True, suggestions=sources)


