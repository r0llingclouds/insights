from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from insights.config import Paths
from insights.ingest import IngestBackend, ingest as ingest_source
from insights.ingest.detect import detect_source
from insights.llm import AnthropicClient, ChatMessage, OpenAIClient
from insights.retrieval import build_context
from insights.storage.db import Database
from insights.storage.models import Source, SourceKind


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


def _resolve_source_ref(db: Database, ref: str) -> Source | None:
    ref = ref.strip()
    if not ref:
        return None

    if _is_hex_id(ref):
        return db.get_source_by_id(ref)

    # Try URL / YouTube URL
    if ref.startswith(("http://", "https://")):
        detected = detect_source(ref, forced_type="auto")
        try:
            return db.get_source_by_kind_locator(kind=detected.kind, locator=detected.locator)
        except KeyError:
            return None

    # Try a raw YouTube video id (common convenience)
    if re.fullmatch(r"[A-Za-z0-9_-]{6,16}", ref):
        try:
            return db.get_source_by_kind_locator(kind=SourceKind.YOUTUBE, locator=ref)
        except KeyError:
            pass

    # Fall back to file path
    try:
        detected = detect_source(ref, forced_type="file")
        try:
            return db.get_source_by_kind_locator(kind=detected.kind, locator=detected.locator)
        except KeyError:
            return None
    except Exception:
        return None


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
            s = _resolve_source_ref(db, ref)
        finally:
            db.close()

        if s is None:
            # Provide fuzzy suggestions (small limit) to help the model.
            suggestions = self.find_sources(text=ref, limit=5).get("matches", [])
            return {"success": True, "found": False, "ref": ref, "suggestions": suggestions}
        return {"success": True, "found": True, "source": _source_to_dict(s)}

    def list_conversations(self, *, source_ref: str | None = None, limit: int = 10) -> dict[str, Any]:
        lim = max(1, min(int(limit), 50))

        source_id: str | None = None
        if source_ref:
            db = Database.open(self._ctx.paths.db_path)
            try:
                s = _resolve_source_ref(db, source_ref)
            finally:
                db.close()
            if not s:
                return {"success": True, "conversations": [], "count": 0, "note": "source not found"}
            source_id = s.id

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
                    s = _resolve_source_ref(db, source_ref)
                finally:
                    db.close()
                if not s:
                    return {"success": True, "matches": [], "match_count": 0, "note": "source not found"}
                where.append(
                    "c.id IN (SELECT conversation_id FROM conversation_sources WHERE source_id = ?)"
                )
                params.append(s.id)

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
            src = _resolve_source_ref(db, source_ref)
            if not src:
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

            context = build_context(
                db=db,
                source_ids=[src.id],
                question=q,
                max_context_tokens=max(500, int(max_context_tokens)),
            )
        finally:
            db.close()

        system = (
            "You are a precise assistant. Answer using ONLY the provided sources. "
            "If the sources do not contain the answer, say what is missing. "
            "When sources are chunked, cite them as Source + chunk number."
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

        return {
            "success": True,
            "answer": resp.text,
            "provider": resp.provider,
            "model": resp.model,
            "usage": resp.usage,
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
            "get_conversation_info": self.get_conversation_info,
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
        "name": "get_conversation_info",
        "description": "Get details about a specific conversation and its bound sources.",
        "input_schema": {
            "type": "object",
            "properties": {"conversation_id": {"type": "string", "description": "Conversation id"}},
            "required": ["conversation_id"],
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


