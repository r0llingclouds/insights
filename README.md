### Insights (terminal app)

Ingest **local documents**, **web pages**, and **YouTube videos** into cached plain text, then run **Q&A / chat** over the sources using **OpenAI** or **Anthropic**.

### Install (uv)

```bash
cd /Users/tirso.lopez/Documents/insights
uv sync
```

Run commands using `uv run` (recommended):

```bash
uv run insights version
```

### Storage / caching

- **Default app dir**: `~/.insights/`
  - DB: `~/.insights/insights.db`
  - Cache: `~/.insights/cache/`
- For testing / isolation, point the app at a custom directory:

```bash
export INSIGHTS_APP_DIR="/tmp/insights-test"
```

Then use `--app-dir "$INSIGHTS_APP_DIR"` in all commands below.

### Environment variables

Set the provider keys you plan to use:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export ASSEMBLYAI_API_KEY="..."
export FIRECRAWL_API_KEY="..."
```

Optional: you can also store them in a `.env` file in your **current directory** or inside your **app dir** (e.g. `/tmp/insights-test/.env`).

### Commands

#### Ingest sources

Ingest a **local PDF/DOCX/PPTX/etc** (Docling):

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" ingest /path/to/file.pdf
```

Ingest a **web page** (Docling):

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" ingest "https://example.com/article"
```

Ingest a **web page** (Firecrawl):

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" ingest "https://example.com/article" --backend firecrawl
```

Ingest a **YouTube video** (yt-dlp → AssemblyAI transcript):

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" ingest "https://www.youtube.com/watch?v=VIDEO_ID"
```

Force re-ingestion:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" ingest /path/to/file.pdf --refresh
```

#### List sources stored in the DB

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" sources
```

Filter by kind:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" sources --kind file
uv run insights --app-dir "$INSIGHTS_APP_DIR" sources --kind url
uv run insights --app-dir "$INSIGHTS_APP_DIR" sources --kind youtube
```

JSON output:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" sources --json
```

#### One-off Q&A (`ask`)

Ask over a local file (auto-ingests if needed):

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" ask -s /path/to/file.pdf "Summarize the main thesis."
```

Ask over a URL (auto-ingests if needed):

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" ask -s "https://example.com/article" "What are the key claims?"
```

Use Anthropic instead of OpenAI:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" ask -s /path/to/file.pdf "What is this about?" --provider anthropic
```

Override the model:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" ask -s /path/to/file.pdf "Summarize." --provider openai --model gpt-4o-mini
uv run insights --app-dir "$INSIGHTS_APP_DIR" ask -s /path/to/file.pdf "Summarize." --provider anthropic --model claude-3-5-sonnet-latest
```

Force retrieval mode (FTS5) for testing by making the context budget tiny:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" ask -s /path/to/file.pdf "What does it say about pricing?" --max-context-tokens 50
```

#### Interactive chat (persistent)

Start a new chat with a source:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" chat -s /path/to/file.pdf
```

Chat on multiple sources (PDF + YouTube source id):

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" chat -s /path/to/file.pdf -s 830eb7dfaaac428a87fb2dae2e80a2a5
```

Inside chat, useful commands:
- `/sources` list bound sources
- `/add <source_id_or_url_or_path>` add another source
- `/save <title>` set a title
- `/export <path>` export transcript
- `/new` start a new conversation (keeps current sources)
- `/exit` quit

Resume a conversation (use the id printed as `New conversation: ...`):

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" chat --conversation <conversation_id>
```

#### List conversations (grouped by source)

Default view (for each source: show locator and conversations, including first user message excerpt):

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" conversations
```

Filter to one source (id, path, or URL):

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" conversations --source 830eb7dfaaac428a87fb2dae2e80a2a5
uv run insights --app-dir "$INSIGHTS_APP_DIR" conversations --source "/Users/tirso.lopez/Desktop/onepager.pdf"
```

Tune output:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" conversations --excerpt-chars 140
uv run insights --app-dir "$INSIGHTS_APP_DIR" conversations --sources-limit 20
uv run insights --app-dir "$INSIGHTS_APP_DIR" conversations --limit 25
```

Old flat summary table:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" conversations --flat
```

### Inspecting the DB directly (sqlite3)

List sources:

```bash
sqlite3 "$INSIGHTS_APP_DIR/insights.db" \
  "select id, kind, coalesce(title,''), locator, updated_at from sources order by updated_at desc;"
```

Map sources ↔ conversations:

```bash
sqlite3 "$INSIGHTS_APP_DIR/insights.db" "
select
  s.kind,
  s.locator,
  coalesce(s.title,'') as title,
  c.id as conversation_id,
  coalesce(c.title,'') as conversation_title,
  c.updated_at
from conversation_sources cs
join sources s on s.id = cs.source_id
join conversations c on c.id = cs.conversation_id
order by s.updated_at desc, c.updated_at desc;
"
```

