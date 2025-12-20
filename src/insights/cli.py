from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Annotated

import typer
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

app = typer.Typer(
    add_completion=False,
    help="Ingest sources into cached text, then chat/Q&A over them.",
)

console = Console(highlight=False)


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
) -> None:
    paths = resolve_paths(db=db, app_dir=app_dir)
    ctx.obj = {"paths": paths}
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


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
        result = ingest_source(
            db=db,
            input_value=input_value,
            cache_dir=paths.cache_dir,
            forced_type=type,
            url_backend=backend,
            refresh=refresh,
            title=title,
        )
    finally:
        db.close()

    console.print(
        {
            "source_id": result.source.id,
            "kind": result.source.kind.value,
            "locator": result.source.locator,
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
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                }
                for s in sources
            ],
            markup=False,
            highlight=False,
        )
        return

    table = Table(title=f"Sources ({len(sources)})", show_lines=False)
    table.add_column("id", no_wrap=True)
    table.add_column("kind", no_wrap=True)
    table.add_column("title")
    table.add_column("locator")
    table.add_column("updated_at", no_wrap=True)
    for s in sources:
        table.add_row(
            s.id,
            s.kind.value,
            s.title or "",
            s.locator,
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
    as_json: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """List conversations (optionally filtered by source)."""
    source_ref = source.strip() if source else None

    paths = ctx.obj["paths"]
    db = Database.open(paths.db_path)
    try:
        source_id: str | None = None
        if source_ref:
            s = db.get_source_by_id(source_ref)
            if s:
                source_id = s.id
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

        rows = db.list_conversation_summaries(limit=limit, source_id=source_id)
    finally:
        db.close()

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


