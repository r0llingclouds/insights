from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from insights.config import resolve_paths
from insights.chat.session import ChatRunConfig, run_chat
from insights.ingest import IngestBackend, ingest as ingest_source
from insights.llm import AnthropicClient, ChatMessage, OpenAIClient
from insights.retrieval import ContextMode, build_context
from insights.storage.db import Database

app = typer.Typer(
    add_completion=False,
    help="Ingest sources into cached text, then chat/Q&A over them.",
)

console = Console()


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

    console.print(resp.text)
    if context.mode == ContextMode.RETRIEVAL and context.retrieved_chunks:
        console.print("\nSources:")
        for rc in context.retrieved_chunks:
            title = rc.source.title or rc.source.locator
            console.print(f"- {title} ({rc.source.locator}) chunk={rc.chunk_index}")


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


