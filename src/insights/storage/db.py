from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from insights.storage.migrations import MIGRATIONS
from insights.storage.models import Conversation, Message, MessageRole, Source, SourceKind, SourceVersion


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _dt_to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _iso_to_dt(value: str) -> datetime:
    # Python 3.11+ supports fromisoformat for 'YYYY-MM-DDTHH:MM:SS+00:00'
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class FtsMatch:
    chunk_id: str
    document_id: str
    score: float


class Database:
    """
    Thin, typed layer over sqlite3.

    - Owns migrations
    - Provides CRUD for sources, versions, documents/chunks, conversations/messages
    """

    def __init__(self, conn: sqlite3.Connection, path: Path) -> None:
        self._conn = conn
        self.path = path

    @classmethod
    def open(cls, db_path: Path) -> "Database":
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        db = cls(conn=conn, path=db_path)
        db.apply_migrations()
        return db

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self._conn.execute("BEGIN;")
        try:
            yield
            self._conn.execute("COMMIT;")
        except Exception:
            self._conn.execute("ROLLBACK;")
            raise

    def apply_migrations(self) -> None:
        self._conn.execute("PRAGMA foreign_keys = ON;")
        # Make sure schema_migrations exists before reading from it.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
              version INTEGER PRIMARY KEY,
              name TEXT NOT NULL,
              applied_at TEXT NOT NULL
            );
            """
        )
        rows = self._conn.execute("SELECT version FROM schema_migrations;").fetchall()
        seen = {int(r["version"]) for r in rows}

        for mig in MIGRATIONS:
            if mig.version in seen:
                continue
            with self.transaction():
                for stmt in mig.statements:
                    self._conn.execute(stmt)
                self._conn.execute(
                    "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?, ?, ?);",
                    (mig.version, mig.name, _dt_to_iso(_utcnow())),
                )

        # Validate FTS5 exists early so later retrieval errors are clearer.
        self._assert_fts5_available()

    def _assert_fts5_available(self) -> None:
        try:
            self._conn.execute("SELECT 1 FROM chunks_fts LIMIT 1;").fetchone()
        except sqlite3.OperationalError as e:
            raise RuntimeError(
                "SQLite FTS5 is required but not available in this SQLite build. "
                "Install a Python/SQLite build with FTS5 enabled."
            ) from e

    # -----------------
    # Sources & versions
    # -----------------
    def upsert_source(self, *, kind: SourceKind, locator: str, title: str | None) -> Source:
        now = _utcnow()
        source_id = _new_id()
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO sources(id, kind, locator, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(kind, locator) DO UPDATE SET
                  title = COALESCE(excluded.title, sources.title),
                  updated_at = excluded.updated_at
                ;
                """,
                (source_id, kind.value, locator, title, _dt_to_iso(now), _dt_to_iso(now)),
            )
        return self.get_source_by_kind_locator(kind=kind, locator=locator)

    def get_source_by_id(self, source_id: str) -> Source | None:
        row = self._conn.execute("SELECT * FROM sources WHERE id = ?;", (source_id,)).fetchone()
        return _row_to_source(row) if row else None

    def get_source_by_kind_locator(self, *, kind: SourceKind, locator: str) -> Source:
        row = self._conn.execute(
            "SELECT * FROM sources WHERE kind = ? AND locator = ?;",
            (kind.value, locator),
        ).fetchone()
        if not row:
            raise KeyError(f"Source not found: {kind.value}:{locator}")
        return _row_to_source(row)

    def list_sources(self, *, limit: int = 100) -> list[Source]:
        rows = self._conn.execute(
            "SELECT * FROM sources ORDER BY updated_at DESC LIMIT ?;",
            (limit,),
        ).fetchall()
        return [_row_to_source(r) for r in rows]

    def set_source_description(self, *, source_id: str, description: str) -> None:
        with self.transaction():
            self._conn.execute(
                "UPDATE sources SET description = ?, updated_at = ? WHERE id = ?;",
                (description, _dt_to_iso(_utcnow()), source_id),
            )

    def list_sources_missing_description(
        self,
        *,
        limit: int = 100,
        kind: SourceKind | None = None,
    ) -> list[Source]:
        """
        List sources that have no description yet (NULL or empty).
        """
        lim = max(1, min(int(limit), 1000))
        params: list[Any] = []
        where = ["(description IS NULL OR trim(description) = '')"]
        if kind is not None:
            where.append("kind = ?")
            params.append(kind.value)
        rows = self._conn.execute(
            f"SELECT * FROM sources WHERE {' AND '.join(where)} ORDER BY updated_at DESC LIMIT ?;",
            [*params, lim],
        ).fetchall()
        return [_row_to_source(r) for r in rows]

    def get_latest_plain_text_for_source(
        self,
        *,
        source_id: str,
        extractor_preference: Sequence[str],
    ) -> str | None:
        docs = self.get_documents_for_sources_latest(
            source_ids=[source_id],
            extractor_preference=extractor_preference,
        )
        if not docs:
            return None
        return str(docs[0].get("plain_text") or "") or None

    def create_source_version(
        self,
        *,
        source_id: str,
        content_hash: str,
        extractor: str,
        status: str,
        error: str | None,
    ) -> SourceVersion:
        version_id = _new_id()
        extracted_at = _utcnow()
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO source_versions(id, source_id, content_hash, extracted_at, extractor, status, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, content_hash, extractor) DO UPDATE SET
                  extracted_at = excluded.extracted_at,
                  status = excluded.status,
                  error = excluded.error
                ;
                """,
                (
                    version_id,
                    source_id,
                    content_hash,
                    _dt_to_iso(extracted_at),
                    extractor,
                    status,
                    error,
                ),
            )
        return self.get_source_version(source_id=source_id, content_hash=content_hash, extractor=extractor)

    def get_latest_source_version(self, *, source_id: str, extractor: str) -> SourceVersion | None:
        row = self._conn.execute(
            """
            SELECT *
            FROM source_versions
            WHERE source_id = ? AND extractor = ?
            ORDER BY extracted_at DESC
            LIMIT 1;
            """,
            (source_id, extractor),
        ).fetchone()
        return _row_to_source_version(row) if row else None

    def get_source_version_by_id(self, version_id: str) -> SourceVersion | None:
        row = self._conn.execute(
            "SELECT * FROM source_versions WHERE id = ?;",
            (version_id,),
        ).fetchone()
        return _row_to_source_version(row) if row else None

    def get_source_version(self, *, source_id: str, content_hash: str, extractor: str) -> SourceVersion:
        row = self._conn.execute(
            """
            SELECT *
            FROM source_versions
            WHERE source_id = ? AND content_hash = ? AND extractor = ?;
            """,
            (source_id, content_hash, extractor),
        ).fetchone()
        if not row:
            raise KeyError("Source version not found")
        return _row_to_source_version(row)

    # -----------------
    # Documents & chunks
    # -----------------
    def upsert_document(
        self,
        *,
        source_version_id: str,
        markdown: str,
        plain_text: str,
        token_estimate: int,
    ) -> str:
        doc_id = _new_id()
        char_count = len(plain_text)
        with self.transaction():
            # Ensure at most one document per source_version (replace-on-write).
            existing = self._conn.execute(
                "SELECT id FROM documents WHERE source_version_id = ?;",
                (source_version_id,),
            ).fetchone()
            if existing:
                doc_id = str(existing["id"])
                self._conn.execute(
                    """
                    UPDATE documents
                    SET markdown = ?, plain_text = ?, char_count = ?, token_estimate = ?
                    WHERE id = ?;
                    """,
                    (markdown, plain_text, char_count, token_estimate, doc_id),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO documents(id, source_version_id, markdown, plain_text, char_count, token_estimate)
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (doc_id, source_version_id, markdown, plain_text, char_count, token_estimate),
                )
        return doc_id

    def get_document_plain_text_by_source_version(self, source_version_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT plain_text FROM documents WHERE source_version_id = ?;",
            (source_version_id,),
        ).fetchone()
        return str(row["plain_text"]) if row else None

    def get_document_id_by_source_version(self, source_version_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT id FROM documents WHERE source_version_id = ?;",
            (source_version_id,),
        ).fetchone()
        return str(row["id"]) if row else None

    def get_document_by_id(self, document_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM documents WHERE id = ?;", (document_id,)).fetchone()
        return dict(row) if row else None

    def replace_chunks(self, *, document_id: str, chunks: Sequence[str]) -> None:
        with self.transaction():
            # Contentless FTS table isn't linked to `chunks`, so delete from both.
            self._conn.execute("DELETE FROM chunks_fts WHERE document_id = ?;", (document_id,))
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?;", (document_id,))

            rows_to_insert: list[tuple[str, str, int, str]] = []
            fts_rows: list[tuple[str, str, str]] = []
            for idx, text in enumerate(chunks):
                chunk_id = _new_id()
                rows_to_insert.append((chunk_id, document_id, idx, text))
                fts_rows.append((text, chunk_id, document_id))

            self._conn.executemany(
                "INSERT INTO chunks(id, document_id, chunk_index, text) VALUES (?, ?, ?, ?);",
                rows_to_insert,
            )
            self._conn.executemany(
                "INSERT INTO chunks_fts(text, chunk_id, document_id) VALUES (?, ?, ?);",
                fts_rows,
            )

    def chunk_count(self, *, document_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(1) AS c FROM chunks WHERE document_id = ?;",
            (document_id,),
        ).fetchone()
        return int(row["c"]) if row else 0

    def search_chunks_fts(self, *, query: str, document_ids: Sequence[str], limit: int) -> list[FtsMatch]:
        if not document_ids:
            return []
        placeholders = ",".join("?" for _ in document_ids)
        sql = f"""
        SELECT chunk_id, document_id, bm25(chunks_fts) AS score
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
          AND document_id IN ({placeholders})
        ORDER BY score
        LIMIT ?;
        """
        params: list[Any] = [query, *document_ids, limit]
        rows = self._conn.execute(sql, params).fetchall()
        return [FtsMatch(chunk_id=str(r["chunk_id"]), document_id=str(r["document_id"]), score=float(r["score"])) for r in rows]

    def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders});",
            list(chunk_ids),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_documents_for_sources_latest(
        self,
        *,
        source_ids: Sequence[str],
        extractor_preference: Sequence[str],
    ) -> list[dict[str, Any]]:
        """
        Returns documents (id, plain_text, token_estimate, source_version_id, source_id, extractor) for each source.
        If multiple extractors exist, first matching extractor_preference wins per source.
        """
        if not source_ids:
            return []
        placeholders = ",".join("?" for _ in source_ids)
        rows = self._conn.execute(
            f"""
            SELECT sv.source_id,
                   sv.id AS source_version_id,
                   sv.extractor,
                   sv.extracted_at,
                   d.id AS document_id,
                   d.plain_text,
                   d.token_estimate
            FROM source_versions sv
            JOIN documents d ON d.source_version_id = sv.id
            WHERE sv.source_id IN ({placeholders})
              AND sv.status = 'ok'
            ORDER BY sv.extracted_at DESC;
            """,
            list(source_ids),
        ).fetchall()

        pref_rank = {name: i for i, name in enumerate(extractor_preference)}
        best: dict[str, dict[str, Any]] = {}
        for r in rows:
            source_id = str(r["source_id"])
            extractor = str(r["extractor"])
            candidate = {
                "source_id": source_id,
                "source_version_id": str(r["source_version_id"]),
                "extractor": extractor,
                "document_id": str(r["document_id"]),
                "plain_text": str(r["plain_text"]),
                "token_estimate": int(r["token_estimate"]),
                "extracted_at": str(r["extracted_at"]),
            }
            cand_rank = pref_rank.get(extractor, 999)
            prev = best.get(source_id)
            if prev is None:
                candidate["_pref_rank"] = cand_rank
                best[source_id] = candidate
                continue
            prev_rank = int(prev.get("_pref_rank", 999))
            # If extractor is preferred, take it. If same rank, keep latest extracted_at (rows are already desc).
            if cand_rank < prev_rank:
                candidate["_pref_rank"] = cand_rank
                best[source_id] = candidate

        out: list[dict[str, Any]] = []
        for source_id in source_ids:
            v = best.get(source_id)
            if v:
                v.pop("_pref_rank", None)
                out.append(v)
        return out

    # -----------------
    # Conversations & messages
    # -----------------
    def create_conversation(self, *, title: str | None) -> Conversation:
        conv_id = _new_id()
        now = _utcnow()
        with self.transaction():
            self._conn.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at) VALUES (?, ?, ?, ?);",
                (conv_id, title, _dt_to_iso(now), _dt_to_iso(now)),
            )
        return self.get_conversation(conv_id)

    def get_conversation(self, conversation_id: str) -> Conversation:
        row = self._conn.execute("SELECT * FROM conversations WHERE id = ?;", (conversation_id,)).fetchone()
        if not row:
            raise KeyError(f"Conversation not found: {conversation_id}")
        return _row_to_conversation(row)

    def list_conversations(self, *, limit: int = 50) -> list[Conversation]:
        rows = self._conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?;",
            (limit,),
        ).fetchall()
        return [_row_to_conversation(r) for r in rows]

    def list_conversation_summaries(
        self, *, limit: int = 50, source_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        List conversations with message/source counts. Optionally filter to those
        that include a given source_id.
        """
        params: list[Any] = []
        where = ""
        if source_id is not None:
            where = "WHERE c.id IN (SELECT conversation_id FROM conversation_sources WHERE source_id = ?)"
            params.append(source_id)

        rows = self._conn.execute(
            f"""
            SELECT
              c.id,
              c.title,
              c.created_at,
              c.updated_at,
              (SELECT COUNT(1) FROM conversation_sources cs WHERE cs.conversation_id = c.id) AS source_count,
              (SELECT COUNT(1) FROM messages m WHERE m.conversation_id = c.id) AS message_count
            FROM conversations c
            {where}
            ORDER BY c.updated_at DESC
            LIMIT ?;
            """,
            [*params, limit],
        ).fetchall()
        return [dict(r) for r in rows]

    def list_conversations_for_source(
        self,
        *,
        source_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        List conversations bound to a given source, including the first user message.
        """
        rows = self._conn.execute(
            """
            SELECT
              c.id AS conversation_id,
              c.title AS conversation_title,
              c.created_at AS conversation_created_at,
              c.updated_at AS conversation_updated_at,
              (SELECT m.content
                 FROM messages m
                WHERE m.conversation_id = c.id AND m.role = 'user'
                ORDER BY m.created_at ASC
                LIMIT 1) AS first_user_message,
              (SELECT m.created_at
                 FROM messages m
                WHERE m.conversation_id = c.id AND m.role = 'user'
                ORDER BY m.created_at ASC
                LIMIT 1) AS first_user_created_at,
              (SELECT COUNT(1) FROM messages m WHERE m.conversation_id = c.id) AS message_count
            FROM conversation_sources cs
            JOIN conversations c ON c.id = cs.conversation_id
            WHERE cs.source_id = ?
            ORDER BY c.updated_at DESC
            LIMIT ?;
            """,
            (source_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def set_conversation_title(self, conversation_id: str, title: str) -> None:
        with self.transaction():
            self._conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?;",
                (title, _dt_to_iso(_utcnow()), conversation_id),
            )

    def bind_source_to_conversation(self, *, conversation_id: str, source_id: str) -> None:
        with self.transaction():
            self._conn.execute(
                "INSERT OR IGNORE INTO conversation_sources(conversation_id, source_id) VALUES (?, ?);",
                (conversation_id, source_id),
            )
            self._conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?;",
                (_dt_to_iso(_utcnow()), conversation_id),
            )

    def list_conversation_sources(self, conversation_id: str) -> list[Source]:
        rows = self._conn.execute(
            """
            SELECT s.*
            FROM conversation_sources cs
            JOIN sources s ON s.id = cs.source_id
            WHERE cs.conversation_id = ?
            ORDER BY s.updated_at DESC;
            """,
            (conversation_id,),
        ).fetchall()
        return [_row_to_source(r) for r in rows]

    def add_message(
        self,
        *,
        conversation_id: str,
        role: MessageRole,
        content: str,
        provider: str | None = None,
        model: str | None = None,
        usage: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> Message:
        msg_id = _new_id()
        ts = created_at or _utcnow()
        usage_json = json.dumps(usage, ensure_ascii=False) if usage is not None else None
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO messages(id, conversation_id, role, content, created_at, provider, model, usage_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    msg_id,
                    conversation_id,
                    role.value,
                    content,
                    _dt_to_iso(ts),
                    provider,
                    model,
                    usage_json,
                ),
            )
            self._conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?;",
                (_dt_to_iso(_utcnow()), conversation_id),
            )
        return self.get_message(msg_id)

    def get_message(self, message_id: str) -> Message:
        row = self._conn.execute("SELECT * FROM messages WHERE id = ?;", (message_id,)).fetchone()
        if not row:
            raise KeyError("Message not found")
        return _row_to_message(row)

    def list_messages(self, *, conversation_id: str, limit: int = 1000) -> list[Message]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            LIMIT ?;
            """,
            (conversation_id, limit),
        ).fetchall()
        return [_row_to_message(r) for r in rows]


def _row_to_source(row: sqlite3.Row) -> Source:
    return Source(
        id=str(row["id"]),
        kind=SourceKind(str(row["kind"])),
        locator=str(row["locator"]),
        title=str(row["title"]) if row["title"] is not None else None,
        description=str(row["description"]) if row["description"] is not None else None,
        created_at=_iso_to_dt(str(row["created_at"])),
        updated_at=_iso_to_dt(str(row["updated_at"])),
    )


def _row_to_source_version(row: sqlite3.Row) -> SourceVersion:
    return SourceVersion(
        id=str(row["id"]),
        source_id=str(row["source_id"]),
        content_hash=str(row["content_hash"]),
        extracted_at=_iso_to_dt(str(row["extracted_at"])),
        extractor=str(row["extractor"]),
        status=str(row["status"]),
        error=str(row["error"]) if row["error"] is not None else None,
    )


def _row_to_conversation(row: sqlite3.Row) -> Conversation:
    return Conversation(
        id=str(row["id"]),
        title=str(row["title"]) if row["title"] is not None else None,
        created_at=_iso_to_dt(str(row["created_at"])),
        updated_at=_iso_to_dt(str(row["updated_at"])),
    )


def _row_to_message(row: sqlite3.Row) -> Message:
    usage_json = row["usage_json"]
    usage: dict[str, Any] | None
    if usage_json is None:
        usage = None
    else:
        usage = json.loads(str(usage_json))
    return Message(
        id=str(row["id"]),
        conversation_id=str(row["conversation_id"]),
        role=MessageRole(str(row["role"])),
        content=str(row["content"]),
        created_at=_iso_to_dt(str(row["created_at"])),
        provider=str(row["provider"]) if row["provider"] is not None else None,
        model=str(row["model"]) if row["model"] is not None else None,
        usage=usage,
    )


