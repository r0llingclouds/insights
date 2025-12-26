from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console

from insights.ingest import IngestBackend, ingest as ingest_source
from insights.llm import AnthropicClient, ChatMessage, OpenAIClient
from insights.context import build_context, build_context_with_retrieval
from insights.storage.db import Database
from insights.storage.models import MessageRole, Source
from insights.utils.progress import make_progress_printer

console = Console(highlight=False)


@dataclass(frozen=True, slots=True)
class ChatRunConfig:
    provider: str
    model: str | None
    backend: IngestBackend
    refresh_sources: bool
    max_context_tokens: int
    max_output_tokens: int
    temperature: float
    stream: bool = True
    retrieval: bool = False


def _pick_llm(provider: str, model: str | None):
    p = provider.strip().lower()
    if p == "openai":
        return OpenAIClient(), (model or "gpt-4o-mini")
    if p == "anthropic":
        return AnthropicClient(), (model or "claude-3-5-sonnet-latest")
    raise ValueError("provider must be 'openai' or 'anthropic'")


def _resolve_or_ingest_source(
    *,
    db: Database,
    cache_dir: Path,
    ref: str,
    backend: IngestBackend,
    refresh: bool,
) -> Source:
    existing = db.get_source_by_id(ref)
    if existing:
        console.print(
            f"(cache) {existing.kind.value} {existing.locator}",
            markup=False,
            highlight=False,
        )
        return existing
    result = ingest_source(
        db=db,
        input_value=ref,
        cache_dir=cache_dir,
        forced_type="auto",
        url_backend=backend,
        refresh=refresh,
        title=None,
    )
    if result.reused_cache:
        console.print(
            f"(cache) {result.source.kind.value} {result.source.locator} (extractor={result.source_version.extractor})",
            markup=False,
            highlight=False,
        )
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
    try:
        from insights.describe import ensure_source_description

        ensure_source_description(
            db=db,
            source_id=result.source.id,
            source_version_id=result.source_version.id,
            force=bool(refresh),
        )
    except Exception:
        # Description generation is best-effort; never fail chat due to it.
        pass
    return result.source


def _export_conversation(db: Database, *, conversation_id: str, path: Path) -> None:
    msgs = db.list_messages(conversation_id=conversation_id)
    lines: list[str] = []
    for m in msgs:
        ts = m.created_at.isoformat()
        lines.append(f"[{ts}] {m.role.value.upper()}: {m.content}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_chat(
    *,
    db: Database,
    cache_dir: Path,
    conversation_id: str | None,
    sources: list[str],
    config: ChatRunConfig,
) -> None:
    progress = make_progress_printer(prefix="insights")
    if conversation_id:
        conv = db.get_conversation(conversation_id)
        conv_id = conv.id
        console.print(f"Resuming conversation: {conv_id}")
    else:
        conv = db.create_conversation(title=None)
        conv_id = conv.id
        console.print(f"New conversation: {conv_id}")
        if not sources and not config.retrieval:
            raise ValueError(
                "Provide at least one source when starting a new conversation (or use --retrieval)."
            )

    # Bind initial sources if provided.
    for ref in sources:
        s = _resolve_or_ingest_source(
            db=db,
            cache_dir=cache_dir,
            ref=ref,
            backend=config.backend,
            refresh=config.refresh_sources,
        )
        db.bind_source_to_conversation(conversation_id=conv_id, source_id=s.id)

    system = (
        "You are a precise assistant. Answer using ONLY the provided sources. "
        "If the sources do not contain the answer, say what is missing. "
        "Cite sources by name when relevant."
    )

    client, used_model = _pick_llm(config.provider, config.model)

    # Track retrieval mode (can be toggled during chat)
    use_retrieval = config.retrieval

    session = PromptSession()
    mode_str = " (RAG mode)" if use_retrieval else ""
    console.print(f"Type /help for commands.{mode_str}")

    with patch_stdout():
        while True:
            try:
                text = session.prompt("> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nExiting.")
                break

            if not text:
                continue

            if text.startswith("/"):
                parts = text.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in {"/exit", "/quit"}:
                    break
                if cmd == "/help":
                    console.print(
                        "\n".join(
                            [
                                "/sources                List bound sources",
                                "/add <source>           Bind another source (id/path/url)",
                                "/new                    Start a new conversation (keeps current sources)",
                                "/save <title>           Set conversation title",
                                "/export <path>          Export conversation transcript",
                                "/export-md [dir]        Export bound sources to markdown files (default: ~/Downloads)",
                                "/retrieval [on|off]     Toggle RAG mode (semantic search)",
                                "/exit                   Exit chat",
                            ]
                        )
                    )
                    continue
                if cmd == "/sources":
                    bound = db.list_conversation_sources(conv_id)
                    if not bound:
                        console.print("No sources bound.")
                    else:
                        for s in bound:
                            console.print(
                                f"- {s.id} {s.kind.value} {s.title or ''} {s.locator}"
                            )
                    continue
                if cmd == "/add":
                    if not arg:
                        console.print("Usage: /add <source>")
                        continue
                    s = _resolve_or_ingest_source(
                        db=db,
                        cache_dir=cache_dir,
                        ref=arg,
                        backend=config.backend,
                        refresh=config.refresh_sources,
                    )
                    db.bind_source_to_conversation(
                        conversation_id=conv_id, source_id=s.id
                    )
                    console.print(f"Added source: {s.id}")
                    continue
                if cmd == "/new":
                    # Carry forward currently bound sources.
                    bound = db.list_conversation_sources(conv_id)
                    conv = db.create_conversation(title=None)
                    conv_id = conv.id
                    for s in bound:
                        db.bind_source_to_conversation(
                            conversation_id=conv_id, source_id=s.id
                        )
                    console.print(f"New conversation: {conv_id}")
                    continue
                if cmd == "/save":
                    if not arg:
                        console.print("Usage: /save <title>")
                        continue
                    db.set_conversation_title(conv_id, arg)
                    console.print("Saved.")
                    continue
                if cmd == "/export":
                    if not arg:
                        console.print("Usage: /export <path>")
                        continue
                    path = Path(arg).expanduser().resolve()
                    _export_conversation(db, conversation_id=conv_id, path=path)
                    console.print(f"Exported: {path}")
                    continue
                if cmd == "/export-md":
                    out_dir = (
                        Path(arg).expanduser().resolve()
                        if arg
                        else (Path.home() / "Downloads").expanduser().resolve()
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    bound = db.list_conversation_sources(conv_id)
                    if not bound:
                        console.print(
                            "No sources bound.", markup=False, highlight=False
                        )
                        continue
                    from insights.config import Paths
                    from insights.text_export import export_source_text

                    # Build minimal Paths for exporter.
                    paths = Paths(
                        app_dir=db.path.parent, db_path=db.path, cache_dir=cache_dir
                    )
                    progress = make_progress_printer(prefix="insights")
                    for s in bound:
                        written = export_source_text(
                            paths=paths,
                            source_ref=s.id,
                            out_dir=out_dir,
                            include_markdown=True,
                            include_plain=False,
                            refresh=False,
                            progress=progress,
                        )
                        for p in written:
                            console.print(str(p), markup=False, highlight=False)
                    continue
                if cmd == "/retrieval":
                    arg_lower = arg.strip().lower()
                    if arg_lower == "on":
                        use_retrieval = True
                        console.print("RAG mode enabled.")
                    elif arg_lower == "off":
                        use_retrieval = False
                        console.print("RAG mode disabled.")
                    else:
                        # Toggle if no argument
                        use_retrieval = not use_retrieval
                        status = "enabled" if use_retrieval else "disabled"
                        console.print(f"RAG mode {status}.")
                    continue

                console.print("Unknown command. Type /help.")
                continue

            # Normal user message
            bound_sources = db.list_conversation_sources(conv_id)
            if not bound_sources and not use_retrieval:
                console.print("No sources bound. Use /add <source> or /retrieval on.")
                continue

            db.add_message(conversation_id=conv_id, role=MessageRole.USER, content=text)

            if use_retrieval:
                # Search all indexed sources if none bound, otherwise filter to bound sources
                source_ids = [s.id for s in bound_sources] if bound_sources else None
                context = build_context_with_retrieval(
                    db=db,
                    source_ids=source_ids,
                    query=text,
                    cache_dir=cache_dir,
                    max_context_tokens=config.max_context_tokens,
                    progress=progress,
                )
            else:
                context = build_context(
                    db=db,
                    source_ids=[s.id for s in bound_sources],
                    question=text,
                    max_context_tokens=config.max_context_tokens,
                    progress=progress,
                )

            history = db.list_messages(conversation_id=conv_id)
            # Keep last ~12 non-system messages for context.
            non_system = [m for m in history if m.role != MessageRole.SYSTEM]
            tail = non_system[-12:]

            messages: list[ChatMessage] = [ChatMessage(role="system", content=system)]
            for m in tail[:-1]:
                # Exclude the current user message (last), we will re-send it with source context.
                messages.append(ChatMessage(role=m.role.value, content=m.content))

            user_payload = f"Sources context:\n\n{context.context_text}\n\nUser question:\n{text}".strip()
            messages.append(ChatMessage(role="user", content=user_payload))

            # BUG: Crashes when using streaming and chat
            if config.stream:
                # Stream response tokens as they arrive
                full_text_parts: list[str] = []
                for token in client.generate_stream(
                    messages=messages,
                    model=used_model,
                    temperature=config.temperature,
                    max_tokens=config.max_output_tokens,
                ):
                    console.print(token, end="", markup=False, highlight=False)
                    full_text_parts.append(token)
                console.print()  # Newline after streaming
                response_text = "".join(full_text_parts).strip()
                # Note: streaming doesn't return usage stats
                db.add_message(
                    conversation_id=conv_id,
                    role=MessageRole.ASSISTANT,
                    content=response_text,
                    provider=client.provider,
                    model=used_model,
                    usage=None,
                )
            else:
                resp = client.generate(
                    messages=messages,
                    model=used_model,
                    temperature=config.temperature,
                    max_tokens=config.max_output_tokens,
                )
                db.add_message(
                    conversation_id=conv_id,
                    role=MessageRole.ASSISTANT,
                    content=resp.text,
                    provider=resp.provider,
                    model=resp.model,
                    usage=resp.usage,
                )
                console.print(resp.text, markup=False, highlight=False, soft_wrap=True)
