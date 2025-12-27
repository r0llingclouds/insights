from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


MIGRATIONS: list[Migration] = [
    Migration(
        version=1,
        name="init",
        statements=(
            "PRAGMA foreign_keys = ON;",
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
              version INTEGER PRIMARY KEY,
              name TEXT NOT NULL,
              applied_at TEXT NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS sources (
              id TEXT PRIMARY KEY,
              kind TEXT NOT NULL CHECK (kind IN ('file', 'url', 'youtube')),
              locator TEXT NOT NULL,
              title TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(kind, locator)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS source_versions (
              id TEXT PRIMARY KEY,
              source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
              content_hash TEXT NOT NULL,
              extracted_at TEXT NOT NULL,
              extractor TEXT NOT NULL,
              status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
              error TEXT,
              UNIQUE(source_id, content_hash, extractor)
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_source_versions_source_id
              ON source_versions(source_id);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_source_versions_extracted_at
              ON source_versions(extracted_at);
            """,
            """
            CREATE TABLE IF NOT EXISTS documents (
              id TEXT PRIMARY KEY,
              source_version_id TEXT NOT NULL REFERENCES source_versions(id) ON DELETE CASCADE,
              markdown TEXT NOT NULL,
              plain_text TEXT NOT NULL,
              char_count INTEGER NOT NULL,
              token_estimate INTEGER NOT NULL
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_documents_source_version_id
              ON documents(source_version_id);
            """,
            """
            CREATE TABLE IF NOT EXISTS conversations (
              id TEXT PRIMARY KEY,
              title TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS conversation_sources (
              conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
              source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
              PRIMARY KEY (conversation_id, source_id)
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_sources_source_id
              ON conversation_sources(source_id);
            """,
            """
            CREATE TABLE IF NOT EXISTS messages (
              id TEXT PRIMARY KEY,
              conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
              role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
              content TEXT NOT NULL,
              created_at TEXT NOT NULL,
              provider TEXT,
              model TEXT,
              usage_json TEXT
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id_created_at
              ON messages(conversation_id, created_at);
            """,
        ),
    )
    ,
    Migration(
        version=2,
        name="add_source_description",
        statements=(
            # Lightweight semantic search support (LLM-generated one-liner per source).
            "ALTER TABLE sources ADD COLUMN description TEXT;",
            "CREATE INDEX IF NOT EXISTS idx_sources_kind_updated_at ON sources(kind, updated_at);",
        ),
    ),
    Migration(
        version=3,
        name="add_source_version_summary",
        statements=(
            "ALTER TABLE source_versions ADD COLUMN summary TEXT;",
            "CREATE INDEX IF NOT EXISTS idx_source_versions_extracted_at_summary ON source_versions(extracted_at);",
        ),
    ),
    Migration(
        version=4,
        name="rename_documents_token_estimate_to_token_count",
        statements=(
            # `token_estimate` is now computed exactly (tiktoken); rename for clarity.
            "ALTER TABLE documents RENAME COLUMN token_estimate TO token_count;",
        ),
    ),
    Migration(
        version=6,
        name="drop_fts_tables",
        statements=(
            # FTS/retrieval mode removed; drop chunk + FTS artifacts.
            "DROP TABLE IF EXISTS chunks_fts;",
            "DROP TABLE IF EXISTS chunks;",
            "DROP TABLE IF EXISTS document_chunking_meta;",
        ),
    ),
    Migration(
        version=7,
        name="add_tweet_source_kind",
        statements=(
            # Add 'tweet' to the source kind CHECK constraint.
            # SQLite doesn't support ALTER TABLE to modify CHECK constraints,
            # so we recreate the table with the updated constraint.
            "PRAGMA foreign_keys = OFF;",
            """
            CREATE TABLE sources_new (
              id TEXT PRIMARY KEY,
              kind TEXT NOT NULL CHECK (kind IN ('file', 'url', 'youtube', 'tweet')),
              locator TEXT NOT NULL,
              title TEXT,
              description TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(kind, locator)
            );
            """,
            """
            INSERT INTO sources_new (id, kind, locator, title, description, created_at, updated_at)
            SELECT id, kind, locator, title, description, created_at, updated_at FROM sources;
            """,
            "DROP TABLE sources;",
            "ALTER TABLE sources_new RENAME TO sources;",
            "CREATE INDEX IF NOT EXISTS idx_sources_kind_updated_at ON sources(kind, updated_at);",
            "PRAGMA foreign_keys = ON;",
        ),
    ),
    Migration(
        version=8,
        name="add_linkedin_source_kind",
        statements=(
            # Add 'linkedin' to the source kind CHECK constraint.
            "PRAGMA foreign_keys = OFF;",
            """
            CREATE TABLE sources_new (
              id TEXT PRIMARY KEY,
              kind TEXT NOT NULL CHECK (kind IN ('file', 'url', 'youtube', 'tweet', 'linkedin')),
              locator TEXT NOT NULL,
              title TEXT,
              description TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(kind, locator)
            );
            """,
            """
            INSERT INTO sources_new (id, kind, locator, title, description, created_at, updated_at)
            SELECT id, kind, locator, title, description, created_at, updated_at FROM sources;
            """,
            "DROP TABLE sources;",
            "ALTER TABLE sources_new RENAME TO sources;",
            "CREATE INDEX IF NOT EXISTS idx_sources_kind_updated_at ON sources(kind, updated_at);",
            "PRAGMA foreign_keys = ON;",
        ),
    ),
]


