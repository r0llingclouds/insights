from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Annotated
from typing import Any

import typer
from typer.core import TyperGroup
import click
from rich.console import Console
from rich.table import Table

from insights.config import load_config, resolve_paths
from insights.chat.session import ChatRunConfig, run_chat
from insights.ingest import EphemeralDocument, IngestBackend, extract_ephemeral, ingest as ingest_source
from insights.ingest.detect import detect_source
from insights.llm import AnthropicClient, ChatMessage, OpenAIClient
from insights.context import build_context, build_context_with_retrieval
from insights.storage.db import Database
from insights.storage.models import SourceKind
from insights.utils.progress import make_progress_printer


class _InsightsDefaultGroup(TyperGroup):
    """
    Allow `insights \"<natural language query>\"` by routing unknown commands to `insights do`.

    Click normally interprets the first non-option token as a subcommand name; when the query is quoted,
    that token contains spaces and won't match any subcommand. We treat that as the natural-language query.
    """

    def resolve_command(self, ctx: click.Context, args: list[str]):  # type: ignore[override]
        """
        If the first non-option token doesn't match a subcommand, route to `do`
        and treat the original token(s) as `do`'s arguments.
        """
        if args:
            first = args[0]
            if super().get_command(ctx, first) is None:
                do_cmd = super().get_command(ctx, "do")
                if do_cmd is not None:
                    # Return args unchanged so `do` receives the original token(s) as its QUERY.
                    return "do", do_cmd, args
        return super().resolve_command(ctx, args)


app = typer.Typer(
    add_completion=False,
    help="Ingest sources into cached text, then chat/Q&A over them.",
    cls=_InsightsDefaultGroup,
)

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)
logger = logging.getLogger(__name__)

describe_app = typer.Typer(add_completion=False, help="Generate/backfill source descriptions for semantic matching.")
app.add_typer(describe_app, name="describe")

title_app = typer.Typer(add_completion=False, help="Generate/backfill source titles.")
app.add_typer(title_app, name="title")

summary_app = typer.Typer(add_completion=False, help="Generate/backfill source version summaries.")
app.add_typer(summary_app, name="summary")

tokens_app = typer.Typer(add_completion=False, help="Token utilities (backfill exact token counts).")
app.add_typer(tokens_app, name="tokens")


@app.callback()
def _global_options(
    ctx: typer.Context,
    db: Annotated[
        Path | None,
        typer.Option(
            "--db",
            help="Path to SQLite DB file (default: ~/Documents/insights/insights.db).",
            dir_okay=False,
        ),
    ] = None,
    app_dir: Annotated[
        Path | None,
        typer.Option("--app-dir", help="App directory (default: ~/Documents/insights)."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Enable debug logging.")] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            help="Allow side effects for natural-language agent queries (e.g. ingest). Without this, the agent proposes commands instead.",
        ),
    ] = False,
    agent_model: Annotated[
        str,
        typer.Option(
            "--agent-model",
            help="Anthropic model for the natural-language agent (default: claude-sonnet-4-5-20250929).",
        ),
    ] = "claude-sonnet-4-5-20250929",
    agent_max_steps: Annotated[
        int,
        typer.Option("--agent-max-steps", help="Max tool-use steps for the agent (default 10)."),
    ] = 10,
    agent_verbose: Annotated[
        bool,
        typer.Option("--agent-verbose", help="Print agent tool calls and step debug output."),
    ] = False,
) -> None:
    paths = resolve_paths(db=db, app_dir=app_dir)
    config = load_config(app_dir=paths.app_dir)
    ctx.obj = {
        "paths": paths,
        "config": config,
        "agent": {
            "allow_side_effects": bool(yes),
            "model": agent_model or config.agent.model,
            "max_steps": int(agent_max_steps) if agent_max_steps != 10 else config.agent.max_steps,
            "verbose": bool(agent_verbose),
        },
    }
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # httpx is very chatty at INFO ("HTTP Request: ..."). Only show it when user enabled --verbose.
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


@app.command()
def do(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="Natural-language query.")],
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            help="Allow side effects for this agent run (e.g. ingest/export). Equivalent to the global --yes.",
        ),
    ] = False,
) -> None:
    """
    Run a natural-language query through the agent (tool-use).

    You can also run the agent implicitly by passing a quoted query to the root command:
      insights \"show conversations for https://...\"\n
    """
    paths = ctx.obj["paths"]
    cfg = ctx.obj.get("agent") or {}
    allow_side_effects = bool(yes) or bool(cfg.get("allow_side_effects") or False)

    from insights.agent.loop import run_agent

    text = (query or "").strip()
    if not text:
        raise typer.BadParameter("query cannot be empty")

    out = run_agent(
        query=text,
        paths=paths,
        model=str(cfg.get("model") or "claude-sonnet-4-5-20250929"),
        max_steps=int(cfg.get("max_steps") or 10),
        verbose=bool(cfg.get("verbose") or False),
        allow_side_effects=allow_side_effects,
    )
    console.print(out, markup=False, highlight=False, soft_wrap=True)


@app.command()
def agent(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="Natural-language query.")],
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            help="Allow side effects for this agent run (e.g. ingest/export). Equivalent to the global --yes.",
        ),
    ] = False,
) -> None:
    """Alias for `insights do`."""
    do(ctx=ctx, query=query, yes=yes)


@describe_app.command("backfill")
def describe_backfill(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", help="Max sources to process (default 100).")] = 100,
    force: Annotated[
        bool,
        typer.Option("--force", help="Regenerate descriptions even if already present."),
    ] = False,
    provider: Annotated[str, typer.Option("--provider", help="openai|anthropic")] = "anthropic",
    model: Annotated[str | None, typer.Option("--model", help="Model override.")] = None,
    max_content_chars: Annotated[
        int,
        typer.Option("--max-content-chars", help="Max characters from source text to send to the LLM."),
    ] = 8000,
) -> None:
    """
    Backfill missing (or all, with --force) source descriptions.

    Descriptions are stored on `sources.description` and used for lightweight semantic matching.
    """
    from insights.describe import generate_description

    paths = ctx.obj["paths"]
    db = Database.open(paths.db_path)
    try:
        lim = max(1, min(int(limit), 5000))
        if force:
            sources = db.list_sources(limit=lim)
        else:
            sources = db.list_sources_missing_description(limit=lim)

        if not sources:
            console.print("No sources to process.", markup=False, highlight=False)
            return

        extractor_preference = ["firecrawl", "docling", "assemblyai"]
        processed = 0
        skipped = 0
        errors = 0

        for idx, s in enumerate(sources, 1):
            display_locator = s.locator
            if s.kind == SourceKind.YOUTUBE:
                display_locator = f"https://www.youtube.com/watch?v={s.locator}"
            display = s.title or display_locator
            err_console.print(f"[{idx}/{len(sources)}] {display}", markup=False, highlight=False)

            plain = db.get_latest_plain_text_for_source(
                source_id=s.id,
                extractor_preference=extractor_preference,
            )
            if not plain:
                skipped += 1
                err_console.print("  (skip) no cached content found", markup=False, highlight=False)
                continue

            try:
                desc = generate_description(
                    content=plain,
                    provider=provider,
                    model=model,
                    max_content_chars=max_content_chars,
                )
                if not desc:
                    skipped += 1
                    err_console.print("  (skip) empty description", markup=False, highlight=False)
                    continue
                db.set_source_description(source_id=s.id, description=desc)
                processed += 1
                console.print(f"- {s.id} → {desc}", markup=False, highlight=False)
            except Exception as e:
                errors += 1
                err_console.print(f"  (error) {type(e).__name__}: {e}", markup=False, highlight=False)

        err_console.print(
            f"Done. processed={processed} skipped={skipped} errors={errors}",
            markup=False,
            highlight=False,
        )
    finally:
        db.close()


@title_app.command("backfill")
def title_backfill(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", help="Max sources to process (default 100).")] = 100,
    force: Annotated[
        bool,
        typer.Option("--force", help="Regenerate titles even if already present."),
    ] = False,
    provider: Annotated[str, typer.Option("--provider", help="openai|anthropic")] = "anthropic",
    model: Annotated[str | None, typer.Option("--model", help="Model override.")] = None,
    max_content_chars: Annotated[
        int,
        typer.Option("--max-content-chars", help="Max characters from source text to send to the LLM."),
    ] = 8000,
) -> None:
    """
    Backfill missing (or all, with --force) source titles.
    """
    from insights.title import ensure_source_title

    paths = ctx.obj["paths"]
    db = Database.open(paths.db_path)
    try:
        lim = max(1, min(int(limit), 5000))
        if force:
            sources = db.list_sources(limit=lim)
        else:
            sources = db.list_sources_missing_title(limit=lim)

        if not sources:
            console.print("No sources to process.", markup=False, highlight=False)
            return

        extractor_preference = ["firecrawl", "docling", "assemblyai"]
        processed = 0
        skipped = 0
        errors = 0

        for idx, s in enumerate(sources, 1):
            display_locator = s.locator
            if s.kind == SourceKind.YOUTUBE:
                display_locator = f"https://www.youtube.com/watch?v={s.locator}"
            display = s.title or display_locator
            err_console.print(f"[{idx}/{len(sources)}] {display}", markup=False, highlight=False)

            plain = db.get_latest_plain_text_for_source(
                source_id=s.id,
                extractor_preference=extractor_preference,
            )
            if not plain:
                skipped += 1
                err_console.print("  (skip) no cached content found", markup=False, highlight=False)
                continue

            try:
                title = ensure_source_title(
                    db=db,
                    source_id=s.id,
                    source_version_id=None,
                    force=bool(force),
                    provider=provider,
                    model=model,
                    max_content_chars=max_content_chars,
                    extractor_preference=extractor_preference,
                )
                if not title:
                    skipped += 1
                    err_console.print("  (skip) empty title", markup=False, highlight=False)
                    continue
                processed += 1
                console.print(f"- {s.id} → {title}", markup=False, highlight=False)
            except Exception as e:
                errors += 1
                err_console.print(f"  (error) {type(e).__name__}: {e}", markup=False, highlight=False)

        err_console.print(
            f"Done. processed={processed} skipped={skipped} errors={errors}",
            markup=False,
            highlight=False,
        )
    finally:
        db.close()


@summary_app.command("backfill")
def summary_backfill(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", help="Max source versions to process (default 100).")] = 100,
    force: Annotated[bool, typer.Option("--force", help="Regenerate summaries even if already present.")] = False,
    provider: Annotated[str, typer.Option("--provider", help="openai|anthropic")] = "anthropic",
    model: Annotated[str | None, typer.Option("--model", help="Model override.")] = None,
    max_content_chars: Annotated[
        int,
        typer.Option(
            "--max-content-chars",
            help="Chunk size (chars) used for whole-document map-reduce summarization.",
        ),
    ] = 12000,
) -> None:
    """
    Backfill missing (or all, with --force) summaries for source versions.
    """
    from insights.summarize import generate_summary

    paths = ctx.obj["paths"]
    db = Database.open(paths.db_path)
    try:
        lim = max(1, min(int(limit), 5000))

        import sqlite3

        conn = sqlite3.connect(str(paths.db_path))
        conn.row_factory = sqlite3.Row
        try:
            if force:
                rows = conn.execute(
                    """
                    SELECT sv.id AS source_version_id
                    FROM source_versions sv
                    WHERE sv.status = 'ok'
                    ORDER BY sv.extracted_at DESC
                    LIMIT ?;
                    """,
                    (lim,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT sv.id AS source_version_id
                    FROM source_versions sv
                    WHERE sv.status = 'ok' AND (sv.summary IS NULL OR trim(sv.summary) = '')
                    ORDER BY sv.extracted_at DESC
                    LIMIT ?;
                    """,
                    (lim,),
                ).fetchall()
        finally:
            conn.close()

        if not rows:
            console.print("No source versions to process.", markup=False, highlight=False)
            return

        processed = 0
        skipped = 0
        errors = 0

        for idx, r in enumerate(rows, 1):
            version_id = str(r["source_version_id"])
            sv = db.get_source_version_by_id(version_id)
            if not sv:
                skipped += 1
                continue
            plain = db.get_document_plain_text_by_source_version(version_id)
            if not plain:
                skipped += 1
                continue
            try:
                def _summary_progress(
                    msg: str,
                    *,
                    _vid: str = version_id,
                    _i: int = idx,
                    _n: int = len(rows),
                ) -> None:
                    err_console.print(f"[{_i}/{_n}] {_vid} {msg}", markup=False, highlight=False)

                summary = generate_summary(
                    content=plain,
                    provider=provider,
                    model=model,
                    max_content_chars=max_content_chars,
                    progress=_summary_progress,
                )
                if not summary:
                    skipped += 1
                    continue
                db.set_source_version_summary(source_version_id=version_id, summary=summary)
                processed += 1
                console.print(f"- {version_id} → (summary saved)", markup=False, highlight=False)
            except Exception as e:
                errors += 1
                err_console.print(f"(error) {version_id}: {type(e).__name__}: {e}", markup=False, highlight=False)

        err_console.print(
            f"Done. processed={processed} skipped={skipped} errors={errors}",
            markup=False,
            highlight=False,
        )
    finally:
        db.close()


@tokens_app.command("backfill")
def tokens_backfill(
    ctx: typer.Context,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            help="Max documents to process (default 0 = no limit).",
        ),
    ] = 0,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Batch size for DB updates (default 100).",
        ),
    ] = 100,
) -> None:
    """
    Backfill exact token counts for cached documents.

    Notes:
    - Tokens are stored in `documents.token_count`.
    - The count is computed by `insights.utils.tokens.estimate_tokens()` which now uses tiktoken.
    """
    from insights.utils.tokens import estimate_tokens

    paths = ctx.obj["paths"]

    import sqlite3

    # Ensure DB migrations (including token_count rename) are applied.
    db = Database.open(paths.db_path)
    db.close()

    lim = max(0, int(limit))
    bs = max(1, min(int(batch_size), 5000))

    conn = sqlite3.connect(str(paths.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        # Determine total for progress printing (best-effort; fast COUNT).
        total_sql = "SELECT COUNT(1) AS c FROM documents"
        params: tuple[int, ...] = ()
        if lim > 0:
            total = min(lim, int(conn.execute(total_sql + ";").fetchone()["c"]))
        else:
            total = int(conn.execute(total_sql + ";").fetchone()["c"])

        if total <= 0:
            console.print("No documents to process.", markup=False, highlight=False)
            return

        # Stream rows to avoid holding all documents in memory.
        sql = "SELECT id, plain_text FROM documents ORDER BY id"
        if lim > 0:
            sql += " LIMIT ?"
            params = (lim,)
        sql += ";"

        processed = 0
        errors = 0
        pending: list[tuple[int, str]] = []

        # Wrap updates in explicit transactions for speed.
        conn.execute("BEGIN;")
        try:
            for row in conn.execute(sql, params):
                doc_id = str(row["id"])
                plain = str(row["plain_text"] or "")
                try:
                    tok = int(estimate_tokens(plain))
                except Exception as e:
                    errors += 1
                    err_console.print(f"(error) {doc_id}: {type(e).__name__}: {e}", markup=False, highlight=False)
                    continue

                pending.append((tok, doc_id))
                processed += 1
                if processed == 1 or processed == total or (processed % max(1, bs)) == 0:
                    err_console.print(f"[{processed}/{total}] tokens backfill", markup=False, highlight=False)

                if len(pending) >= bs:
                    conn.executemany("UPDATE documents SET token_count = ? WHERE id = ?;", pending)
                    pending.clear()

            if pending:
                conn.executemany("UPDATE documents SET token_count = ? WHERE id = ?;", pending)
                pending.clear()
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

        err_console.print(
            f"Done. processed={processed} errors={errors}",
            markup=False,
            highlight=False,
        )
    finally:
        conn.close()


@app.command()
def version() -> None:
    """Print version info."""
    from insights import __version__

    console.print(__version__)


@app.command()
def ingest(
    ctx: typer.Context,
    input_value: Annotated[str, typer.Argument(help="Path to file or URL.")],
    type: Annotated[
        str,
        typer.Option("--type", help="auto|file|url|youtube|tweet|linkedin"),
    ] = "auto",
    backend: Annotated[
        IngestBackend,
        typer.Option("--backend", help="Backend for URL ingestion."),
    ] = IngestBackend.DOCLING,
    refresh: Annotated[bool, typer.Option("--refresh", help="Force re-ingestion.")] = False,
    title: Annotated[str | None, typer.Option("--title", help="Optional title override.")] = None,
) -> None:
    """Ingest a source and cache its extracted text."""
    paths = ctx.obj["paths"]
    db = Database.open(paths.db_path)
    try:
        def _summary_progress(msg: str) -> None:
            err_console.print(msg, markup=False, highlight=False)

        result = ingest_source(
            db=db,
            input_value=input_value,
            cache_dir=paths.cache_dir,
            forced_type=type,
            url_backend=backend,
            refresh=refresh,
            title=title,
            summary_progress=_summary_progress,
        )
        try:
            from insights.describe import ensure_source_description

            ensure_source_description(
                db=db,
                source_id=result.source.id,
                source_version_id=result.source_version.id,
                force=bool(refresh),
            )
        except Exception as e:
            # Never fail ingest due to description generation.
            logger.debug("Description generation failed for source_id=%s: %s", result.source.id, e)
        try:
            from insights.title import ensure_source_title

            ensure_source_title(
                db=db,
                source_id=result.source.id,
                source_version_id=result.source_version.id,
                force=False,
            )
        except Exception as e:
            # Never fail ingest due to title generation.
            logger.debug("Title generation failed for source_id=%s: %s", result.source.id, e)
    finally:
        db.close()

    if result.reused_cache:
        err_console.print(
            f"(cache) {result.source.kind.value} {result.source.locator} (extractor={result.source_version.extractor})",
            markup=False,
            highlight=False,
        )

    console.print(
        {
            "source_id": result.source.id,
            "kind": result.source.kind.value,
            "locator": result.source.locator,
            "extractor": result.source_version.extractor,
            "document_id": result.document_id,
            "reused_cache": result.reused_cache,
        }
    )


@app.command("sources")
def list_sources(
    ctx: typer.Context,
    kind: Annotated[
        str | None,
        typer.Option("--kind", help="Filter by kind: file|url|youtube|tweet|linkedin"),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max rows to show.")] = 100,
    show_description: Annotated[
        bool,
        typer.Option("--show-description", help="Include a description column (truncated) in table output."),
    ] = False,
    show_summary: Annotated[
        bool,
        typer.Option("--show-summary", help="Include latest summary (truncated) from the most recent source version."),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """List ingested sources stored in the database."""
    kind_norm = kind.strip().lower() if kind else None
    kind_enum: SourceKind | None = None
    if kind_norm:
        try:
            kind_enum = SourceKind(kind_norm)
        except Exception as e:
            raise typer.BadParameter("kind must be one of: file, url, youtube") from e

    paths = ctx.obj["paths"]
    db = Database.open(paths.db_path)
    try:
        sources = db.list_sources(limit=limit)
        extractor_preference = ["firecrawl", "docling", "assemblyai"]
        stats_by_source_id: dict[str, dict] = {}
        if sources and (as_json or show_summary):
            stats = db.get_document_stats_for_sources_latest(
                source_ids=[s.id for s in sources],
                extractor_preference=extractor_preference,
            )
            stats_by_source_id = {str(d["source_id"]): dict(d) for d in stats}
    finally:
        db.close()

    if kind_enum is not None:
        sources = [s for s in sources if s.kind == kind_enum]

    if as_json:
        console.print(
            [
                {
                    "id": s.id,
                    "kind": s.kind.value,
                    "title": s.title,
                    "locator": s.locator,
                    "description": s.description,
                    # Latest cached document stats (preferred extractor). Null if no cached doc.
                    "token_count": (stats_by_source_id.get(s.id) or {}).get("token_count"),
                    "char_count": (stats_by_source_id.get(s.id) or {}).get("char_count"),
                    **({"summary": (stats_by_source_id.get(s.id) or {}).get("summary")} if show_summary else {}),
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                }
                for s in sources
            ],
            markup=False,
            highlight=False,
        )
        return

    def _excerpt(text: str | None, n: int = 120) -> str:
        if not text:
            return ""
        collapsed = " ".join(str(text).split())
        if len(collapsed) <= n:
            return collapsed
        return collapsed[: max(0, n - 1)].rstrip() + "…"

    table = Table(title=f"Sources ({len(sources)})", show_lines=False)
    table.add_column("id", no_wrap=True)
    table.add_column("kind", no_wrap=True)
    table.add_column("title")
    table.add_column("locator")
    if show_description:
        table.add_column("description")
    if show_summary:
        table.add_column("summary")
    table.add_column("updated_at", no_wrap=True)
    for s in sources:
        table.add_row(
            s.id,
            s.kind.value,
            s.title or "",
            s.locator,
            _excerpt(s.description) if show_description else None,
            _excerpt((stats_by_source_id.get(s.id) or {}).get("summary"), n=160) if show_summary else None,
            s.updated_at.isoformat(),
        )
    console.print(table)


@app.command("conversations")
def list_conversations(
    ctx: typer.Context,
    source: Annotated[
        str | None,
        typer.Option(
            "--source",
            "-s",
            help="Filter to conversations that include this source (source id, path, or URL).",
        ),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max rows to show.")] = 50,
    sources_limit: Annotated[
        int,
        typer.Option("--sources-limit", help="Max sources to show (when not filtering by --source)."),
    ] = 100,
    excerpt_chars: Annotated[
        int,
        typer.Option("--excerpt-chars", help="Max characters to show from the first user message."),
    ] = 96,
    flat: Annotated[bool, typer.Option("--flat", help="Show a flat conversation summary table.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """List conversations (optionally filtered by source)."""
    source_ref = source.strip() if source else None

    paths = ctx.obj["paths"]
    db = Database.open(paths.db_path)
    try:
        source_id: str | None = None
        source_obj = None
        if source_ref:
            s = db.get_source_by_id(source_ref)
            if s:
                source_id = s.id
                source_obj = s
            else:
                # If it looks like an id but doesn't exist, fail fast.
                if re.fullmatch(r"[0-9a-fA-F]{32}", source_ref):
                    raise typer.BadParameter(f"Unknown source id: {source_ref}")
                detected = detect_source(source_ref, forced_type="auto")
                try:
                    s = db.get_source_by_kind_locator(kind=detected.kind, locator=detected.locator)
                except KeyError as e:
                    raise typer.BadParameter(f"Unknown source: {source_ref}") from e
                source_id = s.id
                source_obj = s

        if flat:
            rows = db.list_conversation_summaries(limit=limit, source_id=source_id)
        else:
            if source_obj is not None:
                sources = [source_obj]
            else:
                sources = db.list_sources(limit=sources_limit)
    finally:
        db.close()

    def _excerpt(text: str | None) -> str:
        if not text:
            return ""
        collapsed = " ".join(str(text).split())
        if len(collapsed) <= excerpt_chars:
            return collapsed
        return collapsed[: max(0, excerpt_chars - 1)].rstrip() + "…"

    if flat:
        if as_json:
            console.print(
                [
                    {
                        "id": r["id"],
                        "title": r["title"],
                        "created_at": r["created_at"],
                        "updated_at": r["updated_at"],
                        "source_count": r["source_count"],
                        "message_count": r["message_count"],
                    }
                    for r in rows
                ],
                markup=False,
                highlight=False,
            )
            return

        table = Table(title=f"Conversations ({len(rows)})", show_lines=False)
        table.add_column("id", no_wrap=True)
        table.add_column("title")
        table.add_column("messages", justify="right", no_wrap=True)
        table.add_column("sources", justify="right", no_wrap=True)
        table.add_column("updated_at", no_wrap=True)
        for r in rows:
            table.add_row(
                str(r["id"]),
                str(r["title"] or ""),
                str(r["message_count"]),
                str(r["source_count"]),
                str(r["updated_at"]),
            )
        console.print(table)
        return

    # Grouped by source (default)
    paths = ctx.obj["paths"]
    db = Database.open(paths.db_path)
    try:
        out_json = []
        for s in sources:
            convs = db.list_conversations_for_source(source_id=s.id, limit=limit)
            display_locator = s.locator
            if s.kind == SourceKind.YOUTUBE:
                display_locator = f"https://www.youtube.com/watch?v={s.locator}"

            header = f"{s.kind.value}  {s.title or ''}".rstrip()
            console.print(f"\nSource: {header}", markup=False, highlight=False)
            console.print(f"Locator: {display_locator}", markup=False, highlight=False)

            if as_json:
                out_json.append(
                    {
                        "source": {
                            "id": s.id,
                            "kind": s.kind.value,
                            "title": s.title,
                            "locator": s.locator,
                            "display_locator": display_locator,
                        },
                        "conversations": [
                            {
                                "id": c["conversation_id"],
                                "title": c["conversation_title"],
                                "updated_at": c["conversation_updated_at"],
                                "message_count": c["message_count"],
                                "first_user_message_excerpt": _excerpt(c.get("first_user_message")),
                            }
                            for c in convs
                        ],
                    }
                )
                continue

            if not convs:
                console.print("Conversations: (none)", markup=False, highlight=False)
                continue

            table = Table(show_lines=False)
            table.add_column("conversation_id", no_wrap=True)
            table.add_column("title")
            table.add_column("first_user_message")
            table.add_column("messages", justify="right", no_wrap=True)
            table.add_column("updated_at", no_wrap=True)
            for c in convs:
                table.add_row(
                    str(c["conversation_id"]),
                    str(c["conversation_title"] or ""),
                    _excerpt(c.get("first_user_message")),
                    str(c["message_count"]),
                    str(c["conversation_updated_at"]),
                )
            console.print(table)
        if as_json:
            console.print(out_json, markup=False, highlight=False)
    finally:
        db.close()


@app.command()
def ask(
    ctx: typer.Context,
    question: Annotated[str, typer.Argument(help="Question to ask.")],
    source: Annotated[
        list[str],
        typer.Option(
            "--source",
            "-s",
            help="Source id or a path/URL (will ingest if not already cached). Repeatable. Optional with --retrieval.",
        ),
    ] = [],
    provider: Annotated[str, typer.Option("--provider", help="openai|anthropic")] = "openai",
    model: Annotated[str | None, typer.Option("--model", help="Model name override.")] = None,
    backend: Annotated[
        IngestBackend,
        typer.Option("--backend", help="Backend for URL ingestion when auto-ingesting sources."),
    ] = IngestBackend.DOCLING,
    refresh_sources: Annotated[
        bool, typer.Option("--refresh-sources", help="Force re-ingestion for provided sources.")
    ] = False,
    no_store: Annotated[
        bool, typer.Option("--no-store", help="Don't persist source or conversation to DB (reuse cache if exists).")
    ] = False,
    max_context_tokens: Annotated[
        int, typer.Option("--max-context-tokens", help="Context budget (token estimate).")
    ] = 12000,
    max_output_tokens: Annotated[
        int, typer.Option("--max-output-tokens", help="Max output tokens for the LLM.")
    ] = 800,
    temperature: Annotated[float, typer.Option("--temperature", help="Sampling temperature.")] = 0.2,
    retrieval: Annotated[
        bool, typer.Option("--retrieval", "-r", help="Use semantic search (RAG) instead of full documents. Searches all indexed sources if no --source given.")
    ] = False,
) -> None:
    """Ask a one-off question over one or more sources."""
    if not source and not retrieval:
        raise typer.BadParameter("At least one --source is required (or use --retrieval to search all indexed sources).")

    paths = ctx.obj["paths"]
    db = Database.open(paths.db_path)
    try:
        source_ids: list[str] = []
        ephemeral_docs: list[EphemeralDocument] = []

        for ref in source:
            # Check if ref is an existing source ID first.
            existing = db.get_source_by_id(ref)
            if existing:
                err_console.print(
                    f"(cache) {existing.kind.value} {existing.locator}",
                    markup=False,
                    highlight=False,
                )
                source_ids.append(existing.id)
                continue

            if no_store:
                # Ephemeral mode: extract without storing (reuse cache if exists).
                progress_fn = make_progress_printer(prefix="insights")
                result = extract_ephemeral(
                    db=db,
                    input_value=ref,
                    cache_dir=paths.cache_dir,
                    forced_type="auto",
                    url_backend=backend,
                    progress=progress_fn,
                )
                if result.from_cache:
                    # Source exists in DB - use it.
                    assert result.source is not None
                    err_console.print(
                        f"(cache) {result.source.kind.value} {result.source.locator}",
                        markup=False,
                        highlight=False,
                    )
                    source_ids.append(result.source.id)
                else:
                    # Fresh extraction - don't store.
                    assert result.document is not None
                    err_console.print(
                        f"(ephemeral) {result.document.kind.value} {result.document.locator}",
                        markup=False,
                        highlight=False,
                    )
                    ephemeral_docs.append(result.document)
            else:
                # Normal mode: ingest and store.
                ingested = ingest_source(
                    db=db,
                    input_value=ref,
                    cache_dir=paths.cache_dir,
                    forced_type="auto",
                    url_backend=backend,
                    refresh=refresh_sources,
                    title=None,
                )
                if ingested.reused_cache:
                    err_console.print(
                        f"(cache) {ingested.source.kind.value} {ingested.source.locator} (extractor={ingested.source_version.extractor})",
                        markup=False,
                        highlight=False,
                    )
                try:
                    from insights.describe import ensure_source_description

                    ensure_source_description(
                        db=db,
                        source_id=ingested.source.id,
                        source_version_id=ingested.source_version.id,
                        force=bool(refresh_sources),
                    )
                except Exception as e:
                    # Never fail ask due to description generation.
                    logger.debug("Description generation failed for source_id=%s: %s", ingested.source.id, e)
                try:
                    from insights.title import ensure_source_title

                    ensure_source_title(
                        db=db,
                        source_id=ingested.source.id,
                        source_version_id=ingested.source_version.id,
                        force=False,
                    )
                except Exception as e:
                    logger.debug("Title generation failed for source_id=%s: %s", ingested.source.id, e)
                source_ids.append(ingested.source.id)

        progress = make_progress_printer(prefix="insights")
        if retrieval:
            # RAG mode: semantic search over indexed chunks
            # Note: ephemeral docs not supported with retrieval (they need to be indexed first)
            if ephemeral_docs:
                err_console.print(
                    "Warning: --retrieval ignores ephemeral sources (use --no-store without --retrieval for ephemeral mode)",
                    markup=False,
                    highlight=False,
                )
            context = build_context_with_retrieval(
                db=db,
                source_ids=source_ids if source_ids else None,
                query=question,
                cache_dir=paths.cache_dir,
                max_context_tokens=max_context_tokens,
                progress=progress,
            )
        else:
            # Full context mode: supports both stored and ephemeral sources
            context = build_context(
                db=db,
                source_ids=source_ids if source_ids else None,
                ephemeral_docs=ephemeral_docs if ephemeral_docs else None,
                question=question,
                max_context_tokens=max_context_tokens,
                progress=progress,
            )
    finally:
        db.close()

    system = (
        "You are a precise assistant. Answer using ONLY the provided sources. "
        "If the sources do not contain the answer, say what is missing. "
        "Cite sources by name when relevant."
    )
    user = f"Sources context:\n\n{context.context_text}\n\nQuestion:\n{question}".strip()
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]

    provider_norm = provider.strip().lower()
    client: Any
    if provider_norm == "openai":
        client = OpenAIClient()
        used_model = model or "gpt-4o-mini"
    elif provider_norm == "anthropic":
        client = AnthropicClient()
        used_model = model or "claude-sonnet-4-5-20250929"
    else:
        raise typer.BadParameter("provider must be 'openai' or 'anthropic'")

    resp = client.generate(
        messages=messages,
        model=used_model,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )

    console.print(resp.text, markup=False, highlight=False, soft_wrap=True)

    # Print source citations
    if context.sources:
        console.print("\n---\nSources:", markup=False, highlight=False)
        for source in context.sources:
            kind_label = source.kind.value.capitalize()
            title = source.title or source.locator
            score = context.source_scores.get(source.id) if context.source_scores else None
            if score is not None:
                console.print(f"  [{kind_label}] {title} (score: {score:.3f})", markup=False, highlight=False)
            else:
                console.print(f"  [{kind_label}] {title}", markup=False, highlight=False)

    if no_store:
        # Ephemeral mode: don't save conversation.
        err_console.print("\n(no-store mode: conversation not saved)", markup=False, highlight=False)
    else:
        # Persist this one-off Q&A as a resumable conversation.
        # Use sources from context result (handles retrieval without explicit sources)
        used_source_ids = [s.id for s in context.sources] if context.sources else source_ids
        db = Database.open(paths.db_path)
        try:
            from insights.chat.save import save_one_shot_qa

            conv_id = save_one_shot_qa(
                db,
                source_ids=used_source_ids,
                question=question,
                answer=resp.text,
                provider=resp.provider,
                model=resp.model,
                usage=resp.usage,
            )
        finally:
            db.close()

        console.print(f"\nConversation: {conv_id}", markup=False, highlight=False)
        console.print(
            f'Resume: uv run insights --app-dir "{paths.app_dir}" chat --conversation {conv_id}',
            markup=False,
            highlight=False,
        )


@app.command()
def chat(
    ctx: typer.Context,
    source: Annotated[
        list[str],
        typer.Option(
            "--source",
            "-s",
            help="Source id or a path/URL. Repeatable. Required when starting a new conversation.",
        ),
    ] = [],
    conversation: Annotated[
        str | None,
        typer.Option("--conversation", help="Conversation id to resume."),
    ] = None,
    provider: Annotated[str | None, typer.Option("--provider", help="openai|anthropic (default from config)")] = None,
    model: Annotated[str | None, typer.Option("--model", help="Model name override.")] = None,
    backend: Annotated[
        IngestBackend,
        typer.Option("--backend", help="Backend for URL ingestion when auto-ingesting sources."),
    ] = IngestBackend.DOCLING,
    refresh_sources: Annotated[
        bool, typer.Option("--refresh-sources", help="Force re-ingestion for provided sources.")
    ] = False,
    max_context_tokens: Annotated[
        int | None, typer.Option("--max-context-tokens", help="Context budget (default from config).")
    ] = None,
    max_output_tokens: Annotated[
        int | None, typer.Option("--max-output-tokens", help="Max output tokens (default from config).")
    ] = None,
    temperature: Annotated[float | None, typer.Option("--temperature", help="Sampling temperature (default from config).")] = None,
    no_stream: Annotated[
        bool, typer.Option("--no-stream", help="Disable streaming (wait for full response).")
    ] = False,
    retrieval: Annotated[
        bool, typer.Option("--retrieval", "-r", help="Use semantic search (RAG) for context. Searches all indexed sources if no --source given.")
    ] = False,
) -> None:
    """Interactive chat with persistent conversations."""
    paths = ctx.obj["paths"]
    config = ctx.obj["config"]
    db = Database.open(paths.db_path)
    try:
        run_chat(
            db=db,
            cache_dir=paths.cache_dir,
            conversation_id=conversation,
            sources=source,
            config=ChatRunConfig(
                provider=provider or config.defaults.provider,
                model=model or config.defaults.model,
                backend=backend,
                refresh_sources=refresh_sources,
                max_context_tokens=max_context_tokens if max_context_tokens is not None else config.defaults.max_context_tokens,
                max_output_tokens=max_output_tokens if max_output_tokens is not None else config.defaults.max_output_tokens,
                temperature=temperature if temperature is not None else config.defaults.temperature,
                stream=config.defaults.stream if not no_stream else False,
                retrieval=retrieval,
            ),
        )
    finally:
        db.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()


@app.command("text")
def export_text(
    ctx: typer.Context,
    source_ref: Annotated[str, typer.Argument(help="Source id/url/path/title/basename to export.")],
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", help="Output directory (default: ~/Downloads)."),
    ] = None,  # resolved at runtime (default: ~/Downloads)
    out_file: Annotated[
        Path | None,
        typer.Option("--out-file", help="Write a single file to this exact path (deterministic output)."),
    ] = None,
    format: Annotated[
        str | None,
        typer.Option("--format", "-f", help="Export format: md, txt, json, html (default: md)."),
    ] = None,
    backend: Annotated[
        IngestBackend,
        typer.Option("--backend", help="Backend for URL ingestion when ingesting."),
    ] = IngestBackend.DOCLING,
    refresh: Annotated[bool, typer.Option("--refresh", help="Force re-ingestion.")] = False,
    no_store: Annotated[
        bool, typer.Option("--no-store", help="Don't persist source to DB (reuse cache if exists).")
    ] = False,
    name: Annotated[str | None, typer.Option("--name", help="Optional base filename override.")] = None,
    include_plain: Annotated[
        bool,
        typer.Option("--include-plain/--no-plain", help="Write plain text (.txt). Ignored if --format is set."),
    ] = False,
    include_markdown: Annotated[
        bool,
        typer.Option("--include-markdown/--no-markdown", help="Write markdown (.md). Ignored if --format is set."),
    ] = True,
) -> None:
    """
    Export a source's cached content to files.

    Formats:
    - md: Markdown (default)
    - txt: Plain text
    - json: JSON with full metadata
    - html: Styled HTML page

    - Resolves by source id / URL / title / file basename
    - Auto-ingests if not cached
    - Default output dir: ~/Downloads
    """
    ref = (source_ref or "").strip()
    if not ref:
        raise typer.BadParameter("source_ref cannot be empty")

    paths = ctx.obj["paths"]
    from insights.text_export import (
        AmbiguousSourceRefError,
        ExportFormat,
        default_downloads_dir,
        export_source_html,
        export_source_json,
        export_source_text,
    )

    progress = make_progress_printer(prefix="insights")
    written: list[Path] = []

    try:
        # If --format is specified, use the new export functions
        if format:
            fmt = format.lower().strip()
            if fmt in ("json",):
                if no_store:
                    err_console.print("Warning: --no-store not yet supported for JSON export", markup=False, highlight=False)
                path = export_source_json(
                    paths=paths,
                    source_ref=ref,
                    out_dir=out_dir or default_downloads_dir(),
                    out_file=out_file,
                    backend=backend,
                    refresh=refresh,
                    name=name,
                    progress=progress,
                )
                written.append(path)
            elif fmt in ("html",):
                if no_store:
                    err_console.print("Warning: --no-store not yet supported for HTML export", markup=False, highlight=False)
                path = export_source_html(
                    paths=paths,
                    source_ref=ref,
                    out_dir=out_dir or default_downloads_dir(),
                    out_file=out_file,
                    backend=backend,
                    refresh=refresh,
                    name=name,
                    progress=progress,
                )
                written.append(path)
            elif fmt in ("md", "markdown"):
                written = export_source_text(
                    paths=paths,
                    source_ref=ref,
                    out_dir=out_dir or default_downloads_dir(),
                    out_file=out_file,
                    backend=backend,
                    refresh=refresh,
                    no_store=no_store,
                    name=name,
                    include_plain=False,
                    include_markdown=True,
                    progress=progress,
                )
            elif fmt in ("txt", "text", "plain"):
                written = export_source_text(
                    paths=paths,
                    source_ref=ref,
                    out_dir=out_dir or default_downloads_dir(),
                    out_file=None,  # txt doesn't support out_file
                    backend=backend,
                    refresh=refresh,
                    no_store=no_store,
                    name=name,
                    include_plain=True,
                    include_markdown=False,
                    progress=progress,
                )
            else:
                raise typer.BadParameter(f"Unknown format: {fmt}. Use: md, txt, json, html")
        else:
            # Legacy behavior with --include-plain / --include-markdown
            if out_file is not None and include_plain:
                raise typer.BadParameter("--out-file writes a single .md; combine with --no-plain (default) or omit --include-plain.")

            written = export_source_text(
                paths=paths,
                source_ref=ref,
                out_dir=out_dir or default_downloads_dir(),
                out_file=out_file,
                backend=backend,
                refresh=refresh,
                no_store=no_store,
                name=name,
                include_plain=include_plain,
                include_markdown=include_markdown,
                progress=progress,
            )
    except AmbiguousSourceRefError as e:
        err_console.print("Ambiguous source reference. Did you mean:", markup=False, highlight=False)
        for s in e.suggestions:
            err_console.print(f"- {s.id} {s.kind} {s.title or ''} {s.locator}", markup=False, highlight=False)
        raise typer.Exit(code=2) from e

    for p in written:
        console.print(str(p), markup=False, highlight=False)


@app.command("index")
def index_source_cmd(
    ctx: typer.Context,
    source_ref: Annotated[
        str | None,
        typer.Argument(help="Source id/url/path to index. If omitted with --all, indexes all sources."),
    ] = None,
    all_sources: Annotated[
        bool,
        typer.Option("--all", help="Index all sources that aren't already indexed."),
    ] = False,
    reindex: Annotated[
        bool,
        typer.Option("--reindex", help="Re-index even if already indexed."),
    ] = False,
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", help="Chunk size in characters."),
    ] = 1000,
    chunk_overlap: Annotated[
        int,
        typer.Option("--chunk-overlap", help="Overlap between chunks in characters."),
    ] = 100,
) -> None:
    """
    Index sources for semantic search (RAG).

    Creates embeddings and stores them in a vector database for fast similarity search.
    """
    if not source_ref and not all_sources:
        raise typer.BadParameter("Provide a source reference or use --all")

    paths = ctx.obj["paths"]
    from insights.retrieval.search import index_source
    from insights.retrieval.store import VectorStore

    db = Database.open(paths.db_path)
    vector_store = VectorStore(persist_dir=paths.cache_dir / "vectors")

    def progress(msg: str) -> None:
        err_console.print(f"index: {msg}", markup=False, highlight=False)

    try:
        if all_sources:
            # Index all sources
            sources = db.list_sources(limit=5000)
            indexed_ids = vector_store.get_indexed_sources() if not reindex else set()
            to_index = [s for s in sources if s.id not in indexed_ids]

            if not to_index:
                console.print("All sources already indexed.")
                return

            console.print(f"Indexing {len(to_index)} sources...")

            for i, source in enumerate(to_index, 1):
                progress(f"[{i}/{len(to_index)}] {source.title or source.locator}")

                # Get the latest document
                if source.kind == SourceKind.YOUTUBE:
                    extractor_preference = ["assemblyai"]
                elif source.kind == SourceKind.URL:
                    extractor_preference = ["docling", "firecrawl"]
                else:
                    extractor_preference = ["docling"]

                doc = db.get_latest_document_for_source(
                    source_id=source.id,
                    extractor_preference=extractor_preference,
                )
                if not doc:
                    progress(f"  Skipping (no document found)")
                    continue

                plain_text = doc.get("plain_text", "")
                source_version_id = doc.get("source_version_id", "")

                if not plain_text or not source_version_id:
                    progress(f"  Skipping (no content)")
                    continue

                try:
                    count = index_source(
                        source_id=source.id,
                        source_version_id=source_version_id,
                        plain_text=plain_text,
                        store=vector_store,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        progress=progress,
                    )
                    progress(f"  Indexed {count} chunks")
                except Exception as e:
                    progress(f"  Error: {e}")

            console.print(f"Total chunks in index: {vector_store.count()}")
        else:
            # Index single source
            from insights.agent.resolve import resolve_source

            resolved = resolve_source(db=db, ref=source_ref, suggestions_limit=5)
            if not resolved.found or not resolved.source:
                raise typer.BadParameter(f"Source not found: {source_ref}")

            source = resolved.source

            # Check if already indexed
            indexed_ids = vector_store.get_indexed_sources()
            if source.id in indexed_ids and not reindex:
                console.print(f"Source already indexed. Use --reindex to re-index.")
                return

            # Get the latest document
            if source.kind == SourceKind.YOUTUBE:
                extractor_preference = ["assemblyai"]
            elif source.kind == SourceKind.URL:
                extractor_preference = ["docling", "firecrawl"]
            else:
                extractor_preference = ["docling"]

            doc = db.get_latest_document_for_source(
                source_id=source.id,
                extractor_preference=extractor_preference,
            )
            if not doc:
                raise typer.BadParameter("No document found for this source. Try ingesting first.")

            plain_text = doc.get("plain_text", "")
            source_version_id = doc.get("source_version_id", "")

            if not plain_text or not source_version_id:
                raise typer.BadParameter("Source has no content to index.")

            progress(f"Indexing: {source.title or source.locator}")

            count = index_source(
                source_id=source.id,
                source_version_id=source_version_id,
                plain_text=plain_text,
                store=vector_store,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                progress=progress,
            )

            console.print(f"Indexed {count} chunks for source {source.id}")
            console.print(f"Total chunks in index: {vector_store.count()}")
    finally:
        db.close()


@app.command("search")
def search_cmd(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="Search query.")],
    source: Annotated[
        list[str],
        typer.Option("--source", "-s", help="Filter to specific sources. Repeatable."),
    ] = [],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximum results.")] = 10,
    as_json: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """
    Semantic search over indexed sources.

    Finds relevant chunks based on meaning, not just keywords.
    """
    if not query.strip():
        raise typer.BadParameter("Query cannot be empty")

    paths = ctx.obj["paths"]
    from insights.retrieval.search import semantic_search
    from insights.retrieval.store import VectorStore

    db = Database.open(paths.db_path)
    vector_store = VectorStore(persist_dir=paths.cache_dir / "vectors")

    try:
        # Resolve source IDs if provided
        source_ids: list[str] | None = None
        if source:
            from insights.agent.resolve import resolve_source
            source_ids = []
            for ref in source:
                resolved = resolve_source(db=db, ref=ref, suggestions_limit=1)
                if resolved.found and resolved.source:
                    source_ids.append(resolved.source.id)

        results = semantic_search(
            query,
            store=vector_store,
            n_results=limit,
            source_ids=source_ids,
        )

        if not results:
            console.print("No results found.")
            return

        if as_json:
            import json
            output = [
                {
                    "source_id": r.source_id,
                    "chunk_index": r.chunk_index,
                    "score": round(r.score, 4),
                    "content": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                }
                for r in results
            ]
            console.print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            for i, r in enumerate(results, 1):
                source_obj = db.get_source_by_id(r.source_id)
                title = source_obj.title if source_obj else r.source_id
                preview = r.content[:200].replace("\n", " ")
                if len(r.content) > 200:
                    preview += "..."

                console.print(f"\n[{i}] {title} (score: {r.score:.3f})")
                console.print(f"    {preview}", markup=False, highlight=False)
    finally:
        db.close()


