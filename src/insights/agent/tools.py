from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from insights.text_export import AmbiguousSourceRefError, default_downloads_dir, export_source_text
from insights.config import Paths
from insights.ingest import IngestBackend, ingest as ingest_source
from insights.llm import AnthropicClient, ChatMessage, OpenAIClient
from insights.context import build_context
from insights.storage.db import Database
from insights.storage.models import Source, SourceKind
from insights.agent.resolve import resolve_source as resolve_source_any
from insights.chat.session import ChatRunConfig, run_chat
from insights.chat.save import save_one_shot_qa
from insights.utils.progress import make_progress_printer


Provider = Literal["openai", "anthropic"]


def _is_hex_id(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{32}", value))


def _display_locator(kind: SourceKind, locator: str) -> str:
    if kind == SourceKind.YOUTUBE:
        return f"https://www.youtube.com/watch?v={locator}"
    return locator


def _source_to_dict(s: Source) -> dict[str, Any]:
    # `description` lands in a later migration; keep compatible while rolling out.
    desc = getattr(s, "description", None)
    return {
        "id": s.id,
        "kind": s.kind.value,
        "locator": s.locator,
        "display_locator": _display_locator(s.kind, s.locator),
        "title": s.title,
        "description": desc,
        "updated_at": s.updated_at.isoformat(),
    }


def _pick_llm(provider: Provider, model: str | None):
    if provider == "openai":
        return OpenAIClient(), (model or "gpt-4o-mini")
    if provider == "anthropic":
        return AnthropicClient(), (model or "claude-3-5-sonnet-latest")
    raise ValueError("provider must be 'openai' or 'anthropic'")


def _looks_like_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _looks_like_path(value: str) -> bool:
    return ("/" in value) or ("\\" in value) or value.startswith(("~", "./", "../"))


def _proposed_ingest_command(*, app_dir: Path, source: str, backend: str | None, refresh: bool, title: str | None) -> str:
    cmd: list[str] = ["uv", "run", "insights", "--app-dir", str(app_dir), "ingest", source]
    if backend:
        cmd.extend(["--backend", backend])
    if refresh:
        cmd.append("--refresh")
    if title:
        cmd.extend(["--title", title])
    return " ".join(shlex.quote(c) for c in cmd)


@dataclass(frozen=True, slots=True)
class ToolContext:
    paths: Paths
    allow_side_effects: bool


class ToolRunner:
    def __init__(self, *, ctx: ToolContext) -> None:
        self._ctx = ctx

    # -----------------
    # Tools
    # -----------------
    def list_sources(self, *, kind: str | None = None, limit: int = 50) -> dict[str, Any]:
        k = kind.strip().lower() if kind else None
        kind_enum: SourceKind | None = None
        if k:
            kind_enum = SourceKind(k)

        db = Database.open(self._ctx.paths.db_path)
        try:
            sources = db.list_sources(limit=max(1, min(int(limit), 500)))
        finally:
            db.close()

        if kind_enum is not None:
            sources = [s for s in sources if s.kind == kind_enum]

        return {"success": True, "sources": [_source_to_dict(s) for s in sources], "count": len(sources)}

    def find_sources(self, *, text: str, kind: str | None = None, limit: int = 20) -> dict[str, Any]:
        """
        Basic substring search across locator/title/(description when present).
        This is intentionally lightweight (no vectors).
        """
        q = (text or "").strip()
        if not q:
            return {"success": False, "error": "text is required"}

        k = kind.strip().lower() if kind else None
        kind_enum: SourceKind | None = None
        if k:
            kind_enum = SourceKind(k)

        # Use sqlite directly so we can query the optional description column safely.
        import sqlite3

        conn = sqlite3.connect(str(self._ctx.paths.db_path))
        conn.row_factory = sqlite3.Row
        try:
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(sources);").fetchall()]
            has_description = "description" in cols

            where = []
            params: list[Any] = []
            if kind_enum is not None:
                where.append("kind = ?")
                params.append(kind_enum.value)

            like = f"%{q.lower()}%"
            like_clauses = [
                "lower(locator) LIKE ?",
                "lower(coalesce(title,'')) LIKE ?",
            ]
            params.extend([like, like])
            if has_description:
                like_clauses.append("lower(coalesce(description,'')) LIKE ?")
                params.append(like)
            # Basename match for file sources (handles queries like 'onepager.pdf')
            like_clauses.append("(kind = 'file' AND (lower(locator) LIKE ? OR lower(locator) LIKE ?))")
            params.append(f"%/{q.lower()}")
            params.append(f"%\\\\{q.lower()}")

            where.append("(" + " OR ".join(like_clauses) + ")")

            sql = f"""
            SELECT id, kind, locator, title, updated_at{", description" if has_description else ""}
            FROM sources
            WHERE {' AND '.join(where)}
            ORDER BY updated_at DESC
            LIMIT ?;
            """
            params.append(max(1, min(int(limit), 200)))

            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
            # Add display locator
            for r in rows:
                r["display_locator"] = _display_locator(SourceKind(r["kind"]), str(r["locator"]))
            return {"success": True, "query": q, "matches": rows, "match_count": len(rows)}
        finally:
            conn.close()

    def semantic_search_sources(self, *, query: str, kind: str | None = None, limit: int = 200) -> dict[str, Any]:
        """
        Return sources with descriptions so the model can semantically match the query.

        This intentionally does NOT do vector search; it surfaces (title, description) pairs.
        """
        q = (query or "").strip()
        if not q:
            return {"success": False, "error": "query is required"}

        k = kind.strip().lower() if kind else None
        kind_enum: SourceKind | None = None
        if k:
            kind_enum = SourceKind(k)

        lim = max(1, min(int(limit), 500))

        db = Database.open(self._ctx.paths.db_path)
        try:
            sources = db.list_sources(limit=lim)
        finally:
            db.close()

        if kind_enum is not None:
            sources = [s for s in sources if s.kind == kind_enum]

        # Prefer described sources; if descriptions are missing, tell the model.
        described = [s for s in sources if (s.description or "").strip()]

        return {
            "success": True,
            "query": q,
            "sources": [_source_to_dict(s) for s in described],
            "count": len(described),
            "note": "Match the user's query against title+description to find the best source. If few/no descriptions exist, suggest running `insights describe backfill`.",
        }

    def resolve_source(self, *, ref: str) -> dict[str, Any]:
        db = Database.open(self._ctx.paths.db_path)
        try:
            resolved = resolve_source_any(db=db, ref=ref, suggestions_limit=5)
        finally:
            db.close()

        if resolved.found and resolved.source:
            return {"success": True, "found": True, "source": _source_to_dict(resolved.source)}
        if resolved.ambiguous:
            return {
                "success": True,
                "found": False,
                "ambiguous": True,
                "ref": ref,
                "suggestions": [_source_to_dict(s) for s in resolved.suggestions],
            }
        # Not found; provide fuzzy suggestions (small limit) to help the model.
        suggestions = self.find_sources(text=ref, limit=5).get("matches", [])
        return {"success": True, "found": False, "ref": ref, "suggestions": suggestions}

    def list_conversations(self, *, source_ref: str | None = None, limit: int = 10) -> dict[str, Any]:
        lim = max(1, min(int(limit), 50))

        source_id: str | None = None
        if source_ref:
            db = Database.open(self._ctx.paths.db_path)
            try:
                resolved = resolve_source_any(db=db, ref=source_ref, suggestions_limit=5)
            finally:
                db.close()
            if resolved.ambiguous:
                return {
                    "success": True,
                    "ambiguous": True,
                    "suggestions": [_source_to_dict(s) for s in resolved.suggestions],
                }
            if not (resolved.found and resolved.source):
                return {"success": True, "conversations": [], "count": 0, "note": "source not found"}
            source_id = resolved.source.id

        import sqlite3

        conn = sqlite3.connect(str(self._ctx.paths.db_path))
        conn.row_factory = sqlite3.Row
        try:
            params: list[Any] = []
            where = []
            if source_id is not None:
                where.append(
                    "c.id IN (SELECT conversation_id FROM conversation_sources WHERE source_id = ?)"
                )
                params.append(source_id)

            sql = f"""
            SELECT
              c.id,
              c.title,
              c.updated_at,
              (SELECT COUNT(1) FROM messages m WHERE m.conversation_id = c.id) AS message_count,
              (SELECT COUNT(1) FROM conversation_sources cs WHERE cs.conversation_id = c.id) AS source_count,
              (SELECT content
                 FROM messages m2
                WHERE m2.conversation_id = c.id AND m2.role = 'user'
                ORDER BY m2.created_at ASC
                LIMIT 1) AS first_user_message
            FROM conversations c
            {("WHERE " + " AND ".join(where)) if where else ""}
            ORDER BY c.updated_at DESC
            LIMIT ?;
            """
            params.append(lim)
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
            convos: list[dict[str, Any]] = []
            for r in rows:
                excerpt = (r.get("first_user_message") or "").strip()
                excerpt = " ".join(excerpt.split())
                if len(excerpt) > 160:
                    excerpt = excerpt[:159].rstrip() + "…"
                convos.append(
                    {
                        "id": str(r.get("id")),
                        "title": r.get("title"),
                        "updated_at": r.get("updated_at"),
                        "message_count": r.get("message_count"),
                        "source_count": r.get("source_count"),
                        "excerpt": excerpt,
                    }
                )
            return {"success": True, "conversations": convos, "count": len(convos)}
        finally:
            conn.close()

    def search_conversations(self, *, query: str, source_ref: str | None = None, limit: int = 10) -> dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {"success": False, "error": "query is required"}
        lim = max(1, min(int(limit), 50))

        import sqlite3

        conn = sqlite3.connect(str(self._ctx.paths.db_path))
        conn.row_factory = sqlite3.Row
        try:
            params: list[Any] = []
            where = []

            if source_ref:
                # Resolve source_ref to source_id.
                db = Database.open(self._ctx.paths.db_path)
                try:
                    resolved = resolve_source_any(db=db, ref=source_ref, suggestions_limit=5)
                finally:
                    db.close()
                if resolved.ambiguous:
                    return {
                        "success": True,
                        "ambiguous": True,
                        "suggestions": [_source_to_dict(s) for s in resolved.suggestions],
                    }
                if not (resolved.found and resolved.source):
                    return {"success": True, "matches": [], "match_count": 0, "note": "source not found"}
                where.append(
                    "c.id IN (SELECT conversation_id FROM conversation_sources WHERE source_id = ?)"
                )
                params.append(resolved.source.id)

            like = f"%{q.lower()}%"
            where.append(
                "("
                "lower(coalesce(c.title,'')) LIKE ? OR "
                "c.id IN ("
                "  SELECT conversation_id FROM messages WHERE lower(content) LIKE ?"
                ")"
                ")"
            )
            params.extend([like, like])

            sql = f"""
            SELECT
              c.id,
              c.title,
              c.updated_at,
              (SELECT COUNT(1) FROM messages m WHERE m.conversation_id = c.id) AS message_count,
              (SELECT content
                 FROM messages m2
                WHERE m2.conversation_id = c.id AND m2.role = 'user'
                ORDER BY m2.created_at ASC
                LIMIT 1) AS first_user_message
            FROM conversations c
            WHERE {' AND '.join(where)}
            ORDER BY c.updated_at DESC
            LIMIT ?;
            """
            params.append(lim)
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
            for r in rows:
                excerpt = (r.get("first_user_message") or "").strip()
                excerpt = " ".join(excerpt.split())
                if len(excerpt) > 160:
                    excerpt = excerpt[:159].rstrip() + "…"
                r["excerpt"] = excerpt
                r.pop("first_user_message", None)
            return {"success": True, "query": q, "matches": rows, "match_count": len(rows)}
        finally:
            conn.close()

    def find_conversations(
        self,
        *,
        topic: str,
        source_ref: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Find conversations about a topic by searching conversation titles + message contents.
        Intended for \"resume conversation ... about ...\".
        """
        q = (topic or "").strip()
        if not q:
            return {"success": False, "error": "topic is required"}
        lim = max(1, min(int(limit), 50))

        import sqlite3

        conn = sqlite3.connect(str(self._ctx.paths.db_path))
        conn.row_factory = sqlite3.Row
        try:
            params: list[Any] = []
            where = []

            if source_ref:
                db = Database.open(self._ctx.paths.db_path)
                try:
                    resolved = resolve_source_any(db=db, ref=source_ref, suggestions_limit=5)
                finally:
                    db.close()
                if resolved.ambiguous:
                    return {
                        "success": True,
                        "ambiguous": True,
                        "suggestions": [_source_to_dict(s) for s in resolved.suggestions],
                    }
                if not (resolved.found and resolved.source):
                    return {"success": True, "matches": [], "match_count": 0, "note": "source not found"}
                where.append(
                    "c.id IN (SELECT conversation_id FROM conversation_sources WHERE source_id = ?)"
                )
                params.append(resolved.source.id)

            like = f"%{q.lower()}%"
            # Get best-matching message per conversation (simple heuristic: first matching message).
            where.append(
                "("
                "lower(coalesce(c.title,'')) LIKE ? OR "
                "c.id IN (SELECT conversation_id FROM messages WHERE lower(content) LIKE ?)"
                ")"
            )
            params.extend([like, like])

            sql = f"""
            SELECT
              c.id,
              c.title,
              c.updated_at,
              (SELECT COUNT(1) FROM messages m WHERE m.conversation_id = c.id) AS message_count,
              (SELECT content
                 FROM messages m2
                WHERE m2.conversation_id = c.id AND lower(m2.content) LIKE ?
                ORDER BY m2.created_at ASC
                LIMIT 1) AS match_message
            FROM conversations c
            WHERE {' AND '.join(where)}
            ORDER BY c.updated_at DESC
            LIMIT ?;
            """
            params.append(like)
            params.append(lim)
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
            matches: list[dict[str, Any]] = []
            for r in rows:
                excerpt = (r.get("match_message") or "").strip()
                excerpt = " ".join(excerpt.split())
                if len(excerpt) > 180:
                    excerpt = excerpt[:179].rstrip() + "…"
                matches.append(
                    {
                        "id": str(r.get("id")),
                        "title": r.get("title"),
                        "updated_at": r.get("updated_at"),
                        "message_count": r.get("message_count"),
                        "match_excerpt": excerpt,
                        "match_reason": "title_or_message",
                    }
                )
            return {"success": True, "topic": q, "matches": matches, "match_count": len(matches)}
        finally:
            conn.close()

    def get_conversation_info(self, *, conversation_id: str) -> dict[str, Any]:
        cid = (conversation_id or "").strip()
        if not cid:
            return {"success": False, "error": "conversation_id is required"}
        db = Database.open(self._ctx.paths.db_path)
        try:
            conv = db.get_conversation(cid)
            sources = db.list_conversation_sources(cid)
        except KeyError:
            return {"success": True, "found": False, "conversation_id": cid}
        finally:
            db.close()
        return {
            "success": True,
            "found": True,
            "conversation": {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
            },
            "sources": [_source_to_dict(s) for s in sources],
        }

    def start_chat(
        self,
        *,
        source_ref: str | None = None,
        conversation_id: str | None = None,
        provider: Provider = "openai",
        model: str | None = None,
        backend: str | None = None,
        refresh_sources: bool = False,
        max_context_tokens: int = 12000,
        max_output_tokens: int = 800,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """
        Start or resume an interactive chat session.

        - If conversation_id is provided: resumes that conversation.
        - Else: resolves source_ref (id/url/path/title/basename) and starts a new chat on that source.
        - If source_ref isn't found and looks like a URL/path, this may ingest depending on allow_side_effects.
        """
        conv_id = (conversation_id or "").strip() or None
        ref = (source_ref or "").strip() or None
        if not conv_id and not ref:
            return {"success": False, "error": "Provide conversation_id or source_ref"}

        backend_enum = IngestBackend.DOCLING
        if backend:
            backend_enum = IngestBackend(backend.strip().lower())

        db = Database.open(self._ctx.paths.db_path)
        try:
            if conv_id:
                run_chat(
                    db=db,
                    cache_dir=self._ctx.paths.cache_dir,
                    conversation_id=conv_id,
                    sources=[],
                    config=ChatRunConfig(
                        provider=provider,
                        model=model,
                        backend=backend_enum,
                        refresh_sources=bool(refresh_sources),
                        max_context_tokens=int(max_context_tokens),
                        max_output_tokens=int(max_output_tokens),
                        temperature=float(temperature),
                    ),
                )
                return {"success": True, "started": True, "conversation_id": conv_id}

            # Resolve a source ref first; this supports id/url/path/title/basename.
            resolved = resolve_source_any(db=db, ref=ref or "", suggestions_limit=5)
            if resolved.ambiguous:
                return {
                    "success": True,
                    "ambiguous": True,
                    "ref": ref,
                    "suggestions": [_source_to_dict(s) for s in resolved.suggestions],
                }
            if resolved.found and resolved.source:
                # Prefer using internal id so chat doesn't depend on filesystem paths.
                run_chat(
                    db=db,
                    cache_dir=self._ctx.paths.cache_dir,
                    conversation_id=None,
                    sources=[resolved.source.id],
                    config=ChatRunConfig(
                        provider=provider,
                        model=model,
                        backend=backend_enum,
                        refresh_sources=bool(refresh_sources),
                        max_context_tokens=int(max_context_tokens),
                        max_output_tokens=int(max_output_tokens),
                        temperature=float(temperature),
                    ),
                )
                return {"success": True, "started": True, "source": _source_to_dict(resolved.source)}

            # Not found. If it looks like a URL/path, we can ingest by passing it to chat,
            # but that is a side effect.
            if ref and (_looks_like_url(ref) or _looks_like_path(ref)):
                if not self._ctx.allow_side_effects:
                    proposed = _proposed_ingest_command(
                        app_dir=self._ctx.paths.app_dir,
                        source=ref,
                        backend=backend_enum.value,
                        refresh=bool(refresh_sources),
                        title=None,
                    )
                    return {
                        "success": False,
                        "blocked": True,
                        "reason": "safe_mode",
                        "note": "Source not found locally. Ingest is required before starting chat.",
                        "proposed_command": proposed,
                    }
                run_chat(
                    db=db,
                    cache_dir=self._ctx.paths.cache_dir,
                    conversation_id=None,
                    sources=[ref],
                    config=ChatRunConfig(
                        provider=provider,
                        model=model,
                        backend=backend_enum,
                        refresh_sources=bool(refresh_sources),
                        max_context_tokens=int(max_context_tokens),
                        max_output_tokens=int(max_output_tokens),
                        temperature=float(temperature),
                    ),
                )
                return {"success": True, "started": True, "source_ref": ref}

            # Otherwise: ask for clarification by returning suggestions (if any).
            suggestions = self.find_sources(text=ref or "", limit=5).get("matches", [])
            return {"success": True, "found": False, "ref": ref, "suggestions": suggestions}
        finally:
            db.close()

    def export_text(
        self,
        *,
        source_ref: str,
        out_dir: str | None = None,
        out_file: str | None = None,
        backend: str | None = None,
        refresh: bool = False,
        name: str | None = None,
        include_plain: bool = False,
        include_markdown: bool = True,
    ) -> dict[str, Any]:
        """
        Export a source's plain text/transcript (and markdown when available) to files.

        This writes to disk and may ingest the source if it's not cached, so it is treated as a side effect:
        - requires allow_side_effects
        """
        ref = (source_ref or "").strip()
        if not ref:
            return {"success": False, "error": "source_ref is required"}

        backend_enum = IngestBackend.DOCLING
        if backend:
            backend_enum = IngestBackend(backend.strip().lower())

        out_path = Path(out_dir).expanduser().resolve() if out_dir else default_downloads_dir()
        out_file_path = Path(out_file).expanduser().resolve() if out_file else None

        if out_file_path is not None and include_plain:
            return {"success": False, "error": "--out-file writes a single .md; use include_plain=false (default)."}

        if not self._ctx.allow_side_effects:
            cmd: list[str] = [
                "uv",
                "run",
                "insights",
                "--app-dir",
                str(self._ctx.paths.app_dir),
                "text",
                ref,
                "--backend",
                backend_enum.value,
            ]
            if out_file_path is not None:
                cmd.extend(["--out-file", str(out_file_path)])
            else:
                cmd.extend(["--out-dir", str(out_path)])
            if refresh:
                cmd.append("--refresh")
            if name:
                cmd.extend(["--name", name])
            if not include_plain:
                cmd.append("--no-plain")
            if not include_markdown:
                cmd.append("--no-markdown")
            return {
                "success": False,
                "blocked": True,
                "reason": "safe_mode",
                "note": "Export writes files to disk and may ingest sources; run with --yes or use the proposed command.",
                "proposed_command": " ".join(shlex.quote(c) for c in cmd),
            }

        try:
            progress = make_progress_printer(prefix="insights")
            written = export_source_text(
                paths=self._ctx.paths,
                source_ref=ref,
                out_dir=out_path,
                out_file=out_file_path,
                backend=backend_enum,
                refresh=bool(refresh),
                name=name,
                include_plain=bool(include_plain),
                include_markdown=bool(include_markdown),
                progress=progress,
            )
        except AmbiguousSourceRefError as e:
            return {
                "success": True,
                "ambiguous": True,
                "ref": e.ref,
                "suggestions": [
                    {
                        "id": s.id,
                        "kind": s.kind,
                        "title": s.title,
                        "locator": s.locator,
                    }
                    for s in e.suggestions
                ],
            }
        return {"success": True, "written_files": [str(p) for p in written]}

    def ingest_source(
        self,
        *,
        source: str,
        backend: str | None = None,
        refresh: bool = False,
        title: str | None = None,
    ) -> dict[str, Any]:
        if not self._ctx.allow_side_effects:
            proposed = _proposed_ingest_command(
                app_dir=self._ctx.paths.app_dir,
                source=source,
                backend=backend,
                refresh=refresh,
                title=title,
            )
            return {
                "success": False,
                "blocked": True,
                "reason": "safe_mode",
                "proposed_command": proposed,
            }

        backend_enum = IngestBackend.DOCLING
        if backend:
            b = backend.strip().lower()
            backend_enum = IngestBackend(b)

        db = Database.open(self._ctx.paths.db_path)
        try:
            result = ingest_source(
                db=db,
                input_value=source,
                cache_dir=self._ctx.paths.cache_dir,
                forced_type="auto",
                url_backend=backend_enum,
                refresh=refresh,
                title=title,
            )
            try:
                from insights.describe import ensure_source_description

                ensure_source_description(
                    db=db,
                    source_id=result.source.id,
                    source_version_id=result.source_version.id,
                    force=bool(refresh),
                )
            except Exception:
                # Best-effort; never fail ingestion due to description generation.
                pass
            try:
                from insights.title import ensure_source_title

                ensure_source_title(
                    db=db,
                    source_id=result.source.id,
                    source_version_id=result.source_version.id,
                    force=False,
                )
            except Exception:
                pass
        finally:
            db.close()
        return {
            "success": True,
            "source": _source_to_dict(result.source),
            "reused_cache": result.reused_cache,
            "extractor": result.source_version.extractor,
            "document_id": result.document_id,
        }

    def ask_source(
        self,
        *,
        source_ref: str,
        question: str,
        provider: Provider = "anthropic",
        model: str | None = None,
        max_context_tokens: int = 12000,
        max_output_tokens: int = 800,
        temperature: float = 0.2,
        backend: str | None = None,
        refresh_source: bool = False,
    ) -> dict[str, Any]:
        """
        One-off Q&A over a single source.

        Safe mode: will NOT auto-ingest. If missing, returns a blocked response with a proposed ingest command.
        """
        q = (question or "").strip()
        if not q:
            return {"success": False, "error": "question is required"}

        b = None
        if backend:
            b = backend.strip().lower()
            IngestBackend(b)  # validate

        db = Database.open(self._ctx.paths.db_path)
        try:
            progress = make_progress_printer(prefix="insights")
            resolved = resolve_source_any(db=db, ref=source_ref, suggestions_limit=5)
            if resolved.ambiguous:
                return {
                    "success": True,
                    "ambiguous": True,
                    "suggestions": [_source_to_dict(s) for s in resolved.suggestions],
                }

            src = resolved.source if (resolved.found and resolved.source) else None
            if src is None:
                if not self._ctx.allow_side_effects:
                    proposed = _proposed_ingest_command(
                        app_dir=self._ctx.paths.app_dir,
                        source=source_ref,
                        backend=b,
                        refresh=refresh_source,
                        title=None,
                    )
                    return {
                        "success": False,
                        "blocked": True,
                        "reason": "safe_mode",
                        "note": "Source not found locally. Ingest is required before asking.",
                        "proposed_command": proposed,
                    }
                backend_enum = IngestBackend(b or IngestBackend.DOCLING.value)
                ingested = ingest_source(
                    db=db,
                    input_value=source_ref,
                    cache_dir=self._ctx.paths.cache_dir,
                    forced_type="auto",
                    url_backend=backend_enum,
                    refresh=refresh_source,
                    title=None,
                )
                src = ingested.source
                try:
                    from insights.describe import ensure_source_description

                    ensure_source_description(
                        db=db,
                        source_id=ingested.source.id,
                        source_version_id=ingested.source_version.id,
                        force=bool(refresh_source),
                    )
                except Exception:
                    pass
                try:
                    from insights.title import ensure_source_title

                    ensure_source_title(
                        db=db,
                        source_id=ingested.source.id,
                        source_version_id=ingested.source_version.id,
                        force=False,
                    )
                except Exception:
                    pass

            context = build_context(
                db=db,
                source_ids=[src.id],
                question=q,
                max_context_tokens=max(500, int(max_context_tokens)),
                progress=progress,
            )
        finally:
            db.close()

        system = (
            "You are a precise assistant. Answer using ONLY the provided sources. "
            "If the sources do not contain the answer, say what is missing. "
            "Cite sources by name when relevant."
        )
        user_payload = f"Sources context:\n\n{context.context_text}\n\nQuestion:\n{q}".strip()
        messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user_payload)]

        client, used_model = _pick_llm(provider, model)
        resp = client.generate(
            messages=messages,
            model=used_model,
            temperature=float(temperature),
            max_tokens=max(64, int(max_output_tokens)),
        )

        # Persist as a resumable conversation.
        db = Database.open(self._ctx.paths.db_path)
        try:
            conv_id = save_one_shot_qa(
                db,
                source_ids=[src.id],
                question=q,
                answer=resp.text,
                provider=resp.provider,
                model=resp.model,
                usage=resp.usage,
            )
        finally:
            db.close()

        return {
            "success": True,
            "answer": resp.text,
            "provider": resp.provider,
            "model": resp.model,
            "usage": resp.usage,
            "conversation_id": conv_id,
            "source": {
                "id": src.id,
                "kind": src.kind.value,
                "display_locator": _display_locator(src.kind, src.locator),
                "title": src.title,
            },
        }

    # -----------------
    # Dispatcher
    # -----------------
    def execute(self, *, name: str, input_args: dict[str, Any]) -> Any:
        fn = {
            "list_sources": self.list_sources,
            "find_sources": self.find_sources,
            "semantic_search_sources": self.semantic_search_sources,
            "resolve_source": self.resolve_source,
            "list_conversations": self.list_conversations,
            "search_conversations": self.search_conversations,
            "find_conversations": self.find_conversations,
            "get_conversation_info": self.get_conversation_info,
            "start_chat": self.start_chat,
            "export_text": self.export_text,
            "ingest_source": self.ingest_source,
            "ask_source": self.ask_source,
        }.get(name)
        if fn is None:
            return {"success": False, "error": f"Unknown tool: {name}"}

        try:
            return fn(**input_args)
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_sources",
        "description": "List sources ingested into the Insights DB. Use kind+limit to keep output small.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["file", "url", "youtube"], "description": "Optional filter"},
                "limit": {"type": "integer", "description": "Max rows (default 50)"},
            },
        },
    },
    {
        "name": "find_sources",
        "description": "Substring search across sources by locator/title/(description). Use this when the user provides a fuzzy reference.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Search string"},
                "kind": {"type": "string", "enum": ["file", "url", "youtube"], "description": "Optional filter"},
                "limit": {"type": "integer", "description": "Max matches (default 20)"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "semantic_search_sources",
        "description": "Search for sources by topic using natural language by returning their titles+descriptions for semantic matching. Use this when looking for what something is ABOUT rather than a substring match.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language description of what you're looking for"},
                "kind": {"type": "string", "enum": ["file", "url", "youtube"], "description": "Optional filter"},
                "limit": {"type": "integer", "description": "Max sources to return (default 200)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "resolve_source",
        "description": "Resolve a source reference (id, URL, file path, or YouTube id) to a canonical source record.",
        "input_schema": {
            "type": "object",
            "properties": {"ref": {"type": "string", "description": "Source ref"}},
            "required": ["ref"],
        },
    },
    {
        "name": "list_conversations",
        "description": "List conversations, optionally filtered to a source ref. Returns conversation IDs suitable for `insights chat --conversation <id>`.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_ref": {"type": "string", "description": "Optional source ref (id/url/path)"},
                "limit": {"type": "integer", "description": "Max conversations (default 10)"},
            },
        },
    },
    {
        "name": "search_conversations",
        "description": "Search conversations by topic/keywords across conversation titles and message content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "source_ref": {"type": "string", "description": "Optional: limit to a source ref"},
                "limit": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_conversations",
        "description": "Find conversations about a topic by searching conversation titles + message contents. Use this to resume a conversation about a topic, optionally scoped to a source.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic/keywords to search for"},
                "source_ref": {"type": "string", "description": "Optional: limit to a source ref (id/url/path/title/basename)"},
                "limit": {"type": "integer", "description": "Max matches (default 10)"},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "get_conversation_info",
        "description": "Get details about a specific conversation and its bound sources.",
        "input_schema": {
            "type": "object",
            "properties": {"conversation_id": {"type": "string", "description": "Conversation id"}},
            "required": ["conversation_id"],
        },
    },
    {
        "name": "start_chat",
        "description": "Start or resume an interactive chat session. Use source_ref to start a new chat on a source, or conversation_id to resume.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_ref": {"type": "string", "description": "Source reference (id/url/path/title/basename)"},
                "conversation_id": {"type": "string", "description": "Conversation id to resume"},
                "provider": {"type": "string", "enum": ["openai", "anthropic"], "description": "LLM provider for chat"},
                "model": {"type": "string", "description": "Model override"},
                "backend": {"type": "string", "enum": ["docling", "firecrawl"], "description": "URL backend when ingesting"},
                "refresh_sources": {"type": "boolean", "description": "Force re-ingestion for provided sources"},
                "max_context_tokens": {"type": "integer", "description": "Context budget"},
                "max_output_tokens": {"type": "integer", "description": "Max output tokens"},
                "temperature": {"type": "number", "description": "Sampling temperature"},
            },
        },
    },
    {
        "name": "export_text",
        "description": "Export a source's plain text/transcript (and markdown when available) to files (default ~/Downloads). In safe-mode it returns a proposed CLI command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_ref": {"type": "string", "description": "Source id/url/path/title/basename"},
                "out_dir": {"type": "string", "description": "Output directory path (default ~/Downloads)"},
                "out_file": {"type": "string", "description": "Exact output markdown file path (writes one .md)"},
                "backend": {"type": "string", "enum": ["docling", "firecrawl"], "description": "URL backend when ingesting"},
                "refresh": {"type": "boolean", "description": "Force re-ingest"},
                "name": {"type": "string", "description": "Optional base filename override"},
                "include_plain": {"type": "boolean", "description": "Write .txt (default true)"},
                "include_markdown": {"type": "boolean", "description": "Write .md when available (default true)"},
            },
            "required": ["source_ref"],
        },
    },
    {
        "name": "ingest_source",
        "description": "Ingest a file/URL/YouTube URL into the DB. In safe-mode this tool returns a blocked response with a proposed command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Path or URL"},
                "backend": {"type": "string", "enum": ["docling", "firecrawl"], "description": "URL backend"},
                "refresh": {"type": "boolean", "description": "Force re-ingestion"},
                "title": {"type": "string", "description": "Optional title override"},
            },
            "required": ["source"],
        },
    },
    {
        "name": "ask_source",
        "description": "Ask a one-off question about a source using cached text. In safe-mode it will not auto-ingest missing sources.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_ref": {"type": "string", "description": "Source id/url/path"},
                "question": {"type": "string", "description": "Question"},
                "provider": {"type": "string", "enum": ["openai", "anthropic"], "description": "LLM provider"},
                "model": {"type": "string", "description": "Model override"},
                "max_context_tokens": {"type": "integer", "description": "Context budget"},
                "max_output_tokens": {"type": "integer", "description": "Max output tokens"},
                "temperature": {"type": "number", "description": "Sampling temperature"},
                "backend": {"type": "string", "enum": ["docling", "firecrawl"], "description": "URL backend when ingesting"},
                "refresh_source": {"type": "boolean", "description": "Force re-ingestion when ingesting"},
            },
            "required": ["source_ref", "question"],
        },
    },
]


