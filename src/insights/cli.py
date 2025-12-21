from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Annotated

import typer
from typer.core import TyperGroup
import click
from rich.console import Console
from rich.table import Table

from insights.config import resolve_paths
from insights.chat.session import ChatRunConfig, run_chat
from insights.ingest import IngestBackend, ingest as ingest_source
from insights.ingest.detect import detect_source
from insights.llm import AnthropicClient, ChatMessage, OpenAIClient
from insights.retrieval import ContextMode, build_context
from insights.storage.db import Database
from insights.storage.models import SourceKind


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


@app.callback()
def _global_options(
    ctx: typer.Context,
    db: Annotated[
        Path | None,
        typer.Option(
            "--db",
            help="Path to SQLite DB file (default: ~/.insights/insights.db).",
            dir_okay=False,
        ),
    ] = None,
    app_dir: Annotated[
        Path | None,
        typer.Option("--app-dir", help="App directory (default: ~/.insights)."),
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
            help="Anthropic model for the natural-language agent (default: claude-sonnet-4-20250514).",
        ),
    ] = "claude-sonnet-4-20250514",
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
    ctx.obj = {
        "paths": paths,
        "agent": {
            "allow_side_effects": bool(yes),
            "model": agent_model,
            "max_steps": int(agent_max_steps),
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
        model=str(cfg.get("model") or "claude-sonnet-4-20250514"),
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
        typer.Option("--type", help="auto|file|url|youtube"),
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
        typer.Option("--kind", help="Filter by kind: file|url|youtube"),
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
        summary_by_source_id: dict[str, str | None] = {}
        if show_summary:
            docs = db.get_documents_for_sources_latest(
                source_ids=[s.id for s in sources],
                extractor_preference=["firecrawl", "docling", "assemblyai"],
            )
            summary_by_source_id = {str(d["source_id"]): d.get("summary") for d in docs}
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
                    **({"summary": summary_by_source_id.get(s.id)} if show_summary else {}),
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
            _excerpt(summary_by_source_id.get(s.id), n=160) if show_summary else None,
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
            help="Source id or a path/URL (will ingest if not already cached). Repeatable.",
        ),
    ],
    provider: Annotated[str, typer.Option("--provider", help="openai|anthropic")] = "openai",
    model: Annotated[str | None, typer.Option("--model", help="Model name override.")] = None,
    backend: Annotated[
        IngestBackend,
        typer.Option("--backend", help="Backend for URL ingestion when auto-ingesting sources."),
    ] = IngestBackend.DOCLING,
    refresh_sources: Annotated[
        bool, typer.Option("--refresh-sources", help="Force re-ingestion for provided sources.")
    ] = False,
    max_context_tokens: Annotated[
        int, typer.Option("--max-context-tokens", help="Context budget (token estimate).")
    ] = 12000,
    max_output_tokens: Annotated[
        int, typer.Option("--max-output-tokens", help="Max output tokens for the LLM.")
    ] = 800,
    temperature: Annotated[float, typer.Option("--temperature", help="Sampling temperature.")] = 0.2,
) -> None:
    """Ask a one-off question over one or more sources."""
    if not source:
        raise typer.BadParameter("At least one --source is required.")

    paths = ctx.obj["paths"]
    db = Database.open(paths.db_path)
    try:
        source_ids: list[str] = []
        for ref in source:
            existing = db.get_source_by_id(ref)
            if existing:
                err_console.print(
                    f"(cache) {existing.kind.value} {existing.locator}",
                    markup=False,
                    highlight=False,
                )
                source_ids.append(existing.id)
                continue
            # Auto-ingest when not found by id.
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

        context = build_context(
            db=db,
            source_ids=source_ids,
            question=question,
            max_context_tokens=max_context_tokens,
        )
    finally:
        db.close()

    system = (
        "You are a precise assistant. Answer using ONLY the provided sources. "
        "If the sources do not contain the answer, say what is missing. "
        "When sources are chunked, cite them as Source + chunk number."
    )
    user = f"Sources context:\n\n{context.context_text}\n\nQuestion:\n{question}".strip()
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]

    provider_norm = provider.strip().lower()
    if provider_norm == "openai":
        client = OpenAIClient()
        used_model = model or "gpt-4o-mini"
    elif provider_norm == "anthropic":
        client = AnthropicClient()
        used_model = model or "claude-3-5-sonnet-latest"
    else:
        raise typer.BadParameter("provider must be 'openai' or 'anthropic'")

    resp = client.generate(
        messages=messages,
        model=used_model,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )

    console.print(resp.text, markup=False, highlight=False, soft_wrap=True)
    # Persist this one-off Q&A as a resumable conversation.
    db = Database.open(paths.db_path)
    try:
        from insights.chat.save import save_one_shot_qa

        conv_id = save_one_shot_qa(
            db,
            source_ids=source_ids,
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
    if context.mode == ContextMode.RETRIEVAL and context.retrieved_chunks:
        console.print("\nSources:", markup=False, highlight=False)
        for rc in context.retrieved_chunks:
            title = rc.source.title or rc.source.locator
            console.print(
                f"- {title} ({rc.source.locator}) chunk={rc.chunk_index}",
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
    provider: Annotated[str, typer.Option("--provider", help="openai|anthropic")] = "openai",
    model: Annotated[str | None, typer.Option("--model", help="Model name override.")] = None,
    backend: Annotated[
        IngestBackend,
        typer.Option("--backend", help="Backend for URL ingestion when auto-ingesting sources."),
    ] = IngestBackend.DOCLING,
    refresh_sources: Annotated[
        bool, typer.Option("--refresh-sources", help="Force re-ingestion for provided sources.")
    ] = False,
    max_context_tokens: Annotated[
        int, typer.Option("--max-context-tokens", help="Context budget (token estimate).")
    ] = 12000,
    max_output_tokens: Annotated[
        int, typer.Option("--max-output-tokens", help="Max output tokens for the LLM.")
    ] = 800,
    temperature: Annotated[float, typer.Option("--temperature", help="Sampling temperature.")] = 0.2,
) -> None:
    """Interactive chat with persistent conversations."""
    paths = ctx.obj["paths"]
    db = Database.open(paths.db_path)
    try:
        run_chat(
            db=db,
            cache_dir=paths.cache_dir,
            conversation_id=conversation,
            sources=source,
            config=ChatRunConfig(
                provider=provider,
                model=model,
                backend=backend,
                refresh_sources=refresh_sources,
                max_context_tokens=max_context_tokens,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
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
        typer.Option("--out-file", help="Write a single markdown file to this exact path (deterministic output)."),
    ] = None,
    backend: Annotated[
        IngestBackend,
        typer.Option("--backend", help="Backend for URL ingestion when ingesting."),
    ] = IngestBackend.DOCLING,
    refresh: Annotated[bool, typer.Option("--refresh", help="Force re-ingestion.")] = False,
    name: Annotated[str | None, typer.Option("--name", help="Optional base filename override.")] = None,
    include_plain: Annotated[
        bool,
        typer.Option("--include-plain/--no-plain", help="Write plain text (.txt)."),
    ] = False,
    include_markdown: Annotated[
        bool,
        typer.Option("--include-markdown/--no-markdown", help="Write markdown (.md). If markdown isn't available, falls back to plain text/transcript."),
    ] = True,
) -> None:
    """
    Export a source's cached plain text/transcript (and markdown when available) to files.

    - Resolves by source id / URL / title / file basename
    - Auto-ingests if not cached
    - Default output dir: ~/Downloads
    """
    ref = (source_ref or "").strip()
    if not ref:
        raise typer.BadParameter("source_ref cannot be empty")

    paths = ctx.obj["paths"]
    from insights.text_export import AmbiguousSourceRefError, default_downloads_dir, export_source_text

    if out_file is not None and include_plain:
        raise typer.BadParameter("--out-file writes a single .md; combine with --no-plain (default) or omit --include-plain.")

    try:
        written = export_source_text(
            paths=paths,
            source_ref=ref,
            out_dir=out_dir or default_downloads_dir(),
            out_file=out_file,
            backend=backend,
            refresh=refresh,
            name=name,
            include_plain=include_plain,
            include_markdown=include_markdown,
        )
    except AmbiguousSourceRefError as e:
        err_console.print("Ambiguous source reference. Did you mean:", markup=False, highlight=False)
        for s in e.suggestions:
            err_console.print(f"- {s.id} {s.kind} {s.title or ''} {s.locator}", markup=False, highlight=False)
        raise typer.Exit(code=2) from e

    for p in written:
        console.print(str(p), markup=False, highlight=False)


