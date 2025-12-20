from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console

from insights.ingest import IngestBackend, ingest as ingest_source
from insights.llm import AnthropicClient, ChatMessage, OpenAIClient
from insights.retrieval import ContextMode, build_context
from insights.storage.db import Database
from insights.storage.models import MessageRole, Source

console = Console()


@dataclass(frozen=True, slots=True)
class ChatRunConfig:
    provider: str
    model: str | None
    backend: IngestBackend
    refresh_sources: bool
    max_context_tokens: int
    max_output_tokens: int
    temperature: float


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
    if conversation_id:
        conv = db.get_conversation(conversation_id)
        conv_id = conv.id
        console.print(f"Resuming conversation: {conv_id}")
    else:
        conv = db.create_conversation(title=None)
        conv_id = conv.id
        console.print(f"New conversation: {conv_id}")
        if not sources:
            raise ValueError("Provide at least one source when starting a new conversation.")

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
        "When sources are chunked, cite them as Source + chunk number."
    )

    client, used_model = _pick_llm(config.provider, config.model)

    session = PromptSession()
    console.print("Type /help for commands.")

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
                            console.print(f"- {s.id} {s.kind.value} {s.title or ''} {s.locator}")
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
                    db.bind_source_to_conversation(conversation_id=conv_id, source_id=s.id)
                    console.print(f"Added source: {s.id}")
                    continue
                if cmd == "/new":
                    # Carry forward currently bound sources.
                    bound = db.list_conversation_sources(conv_id)
                    conv = db.create_conversation(title=None)
                    conv_id = conv.id
                    for s in bound:
                        db.bind_source_to_conversation(conversation_id=conv_id, source_id=s.id)
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

                console.print("Unknown command. Type /help.")
                continue

            # Normal user message
            bound_sources = db.list_conversation_sources(conv_id)
            if not bound_sources:
                console.print("No sources bound. Use /add <source>.")
                continue

            db.add_message(conversation_id=conv_id, role=MessageRole.USER, content=text)

            context = build_context(
                db=db,
                source_ids=[s.id for s in bound_sources],
                question=text,
                max_context_tokens=config.max_context_tokens,
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

            console.print(resp.text)
            if context.mode == ContextMode.RETRIEVAL and context.retrieved_chunks:
                console.print("\nSources:")
                for rc in context.retrieved_chunks:
                    title = rc.source.title or rc.source.locator
                    console.print(f"- {title} ({rc.source.locator}) chunk={rc.chunk_index}")


