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

### Global options (apply to all commands)

- `--app-dir PATH`: app directory (DB + cache). Default: `~/.insights/`
- `--db PATH`: explicit DB file path (overrides app dir DB)
- `--verbose`: enable debug logging

Agent-related global options:
- `--yes`: allow side effects for natural-language agent queries (ingest + export-to-file)
- `--agent-model MODEL` (default `claude-sonnet-4-20250514`)
- `--agent-max-steps N` (default 10)
- `--agent-verbose`

Important: `--yes` is a **global** flag, so it must appear **before** the quoted query (or before `do`).

### Source references (how to refer to a source)

Many commands accept a “source ref”, which can be:
- **source id**: `830eb7dfaaac428a87fb2dae2e80a2a5`
- **URL**: `https://example.com/article`
- **YouTube URL**: `https://www.youtube.com/watch?v=VIDEO_ID` (internally stored as video id)
- **local file path**: `/Users/tirso.lopez/Desktop/onepager.pdf`
- **basename / title fragment**: `onepager.pdf` (if ambiguous, you’ll be prompted to pick one)

### Natural-language agent (Anthropic tool-use)

You can run complex requests as a single quoted query:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" "show conversations for https://www.youtube.com/watch?v=zPMPqzjM0Fw"
uv run insights --app-dir "$INSIGHTS_APP_DIR" "resume last conversation on https://www.youtube.com/watch?v=zPMPqzjM0Fw"
uv run insights --app-dir "$INSIGHTS_APP_DIR" "check if I have a youtube video about someone who stopped using IDEs"
```

Equivalent explicit form:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" do "resume last conversation on https://www.youtube.com/watch?v=zPMPqzjM0Fw"
```

Notes:
- The agent requires `ANTHROPIC_API_KEY` (it orchestrates via Anthropic tool-use, even if Q&A uses another provider).
- By default the agent runs in **safe mode** (no ingestion / network writes). If ingestion is needed, it prints the exact command to run.
- To allow the agent to ingest and export-to-file automatically, add `--yes` (and remember it must appear **before** the quoted query):

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" --yes "ingest https://www.youtube.com/watch?v=zPMPqzjM0Fw"
uv run insights --app-dir "$INSIGHTS_APP_DIR" --yes "ask https://www.youtube.com/watch?v=zPMPqzjM0Fw what is this about?"
uv run insights --app-dir "$INSIGHTS_APP_DIR" --yes "chat on 830eb7dfaaac428a87fb2dae2e80a2a5"
uv run insights --app-dir "$INSIGHTS_APP_DIR" --yes "export transcript for https://www.youtube.com/watch?v=zPMPqzjM0Fw"
uv run insights --app-dir "$INSIGHTS_APP_DIR" --yes "export text for https://karpathy.bearblog.dev/year-in-review-2025/"
```

Agent tuning:
- `--agent-model claude-sonnet-4-20250514`
- `--agent-max-steps 10`
- `--agent-verbose`
- You can also use `agent` as an alias of `do`:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" agent "ask https://example.com/article summarize it"
uv run insights --app-dir "$INSIGHTS_APP_DIR" agent --yes "export text for https://example.com/article"
```

### Source descriptions (for semantic matching)

Insights stores an LLM-generated one-liner description for each source (`sources.description`) to support lightweight “semantic search” without vectors.

- Descriptions are generated on ingest (best-effort; ingestion never fails if description generation fails).
- Optional: override the default description model:

```bash
export INSIGHTS_DESCRIBE_MODEL="claude-sonnet-4-20250514"
```

- To backfill missing descriptions for existing sources:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" describe backfill
```

Backfill options:
- `--limit N`
- `--force`
- `--provider openai|anthropic`
- `--model MODEL`
- `--max-content-chars N`

### Source titles (auto-generate when missing)

If a source has no title, Insights will generate a short plain title (best-effort) from the cached text and store it in `sources.title`.

- Set a specific model for title generation:

```bash
export INSIGHTS_TITLE_MODEL="claude-sonnet-4-20250514"
```

- Backfill missing titles for existing sources:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" title backfill
```

Backfill options:
- `--limit N`
- `--force`
- `--provider openai|anthropic`
- `--model MODEL`
- `--max-content-chars N`

### Summaries (per source version)

Each extraction (`source_versions`) stores a bullet summary (`source_versions.summary`) to improve recall.

Important: summaries are generated from the **entire document/transcript**.

- If the content is small: single-pass summary over the whole text.
- If the content is large: **chunked map-reduce** (summarize all chunks, then reduce).

Optional: override the default summary model:

```bash
export INSIGHTS_SUMMARY_MODEL="claude-sonnet-4-20250514"
```

Optional tuning (whole-doc map-reduce):
- `INSIGHTS_SUMMARY_CHUNK_CHARS` (default: `12000` unless overridden by `--max-content-chars`)
- `INSIGHTS_SUMMARY_OVERLAP_CHARS` (default: `400`)
- `INSIGHTS_SUMMARY_REDUCE_BATCH_SIZE` (default: `10`)
- `INSIGHTS_SUMMARY_PROGRESS_EVERY_CHUNKS` (default: `5`) — print progress every N chunks when processing large content

Progress / warnings for large docs:
- When summarizing **large** content, Insights prints a warning and map/reduce progress **to stderr** (so stdout stays JSON-safe).
- If you redirect stdout (e.g. `> out.json`), you’ll still see progress in the terminal.
- To capture or silence progress:
  - capture: `2> progress.log`
  - silence: `2>/dev/null`

Backfill missing summaries:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" summary backfill
```

Backfill options:
- `--limit N`
- `--force`
- `--provider openai|anthropic`
- `--model MODEL`
- `--max-content-chars N` (chunk size; the whole doc is still covered)

### Large document optimization (auto-switch to Haiku)

When generating **summaries**, **descriptions**, or **titles**, if the source content exceeds **10,000 characters**, Insights automatically uses **`claude-haiku-4-5-20251001`** (cheaper) for that generation.

Overrides:
- If you set one of the model env vars (`INSIGHTS_SUMMARY_MODEL`, `INSIGHTS_DESCRIBE_MODEL`, `INSIGHTS_TITLE_MODEL`), that model is always used (no auto-switch).
- You can change the cutoff:

```bash
export INSIGHTS_LARGE_CONTENT_CUTOFF_CHARS=30000
```

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

Ingest options:
- `--type auto|file|url|youtube`
- `--backend docling|firecrawl` (URLs)
- `--refresh`
- `--title "..."` (optional title override)

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

Sources options:
- `--kind file|url|youtube`
- `--limit N`
- `--json` (includes `description`)
- `--show-description` (table view; truncated)
- `--show-summary` (table/JSON; latest `source_versions.summary` for the preferred extractor)

Show latest summary per source:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" sources --show-summary
uv run insights --app-dir "$INSIGHTS_APP_DIR" sources --json --show-summary
```

#### One-off Q&A (`ask`)

Ask over a local file (auto-ingests if needed):

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" ask -s /path/to/file.pdf "Summarize the main thesis."
```

After answering, `ask` automatically creates a **new conversation** containing the question + answer, and prints a conversation id you can resume:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" chat --conversation <conversation_id>
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

Ask options:
- `-s/--source` (repeatable)
- `--provider openai|anthropic`
- `--model MODEL`
- `--backend docling|firecrawl` (when auto-ingesting URLs)
- `--refresh-sources`
- `--max-context-tokens N`
- `--max-output-tokens N`
- `--temperature FLOAT`

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
- `/export-md [dir]` export bound sources to markdown files (default: ~/Downloads)
- `/new` start a new conversation (keeps current sources)
- `/exit` quit

Resume a conversation (use the id printed as `New conversation: ...`):

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" chat --conversation <conversation_id>
```

Chat options:
- `-s/--source` (repeatable)
- `--conversation <conversation_id>`
- `--provider openai|anthropic`
- `--model MODEL`
- `--backend docling|firecrawl` (when auto-ingesting URLs)
- `--refresh-sources`
- `--max-context-tokens N`
- `--max-output-tokens N`
- `--temperature FLOAT`

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

Conversations options:
- `--source <source_ref>`
- `--limit N`
- `--sources-limit N`
- `--excerpt-chars N`
- `--flat`
- `--json`

#### Export plain text / transcript (and markdown) to files

Export the cached **markdown** to a `.md` file under `~/Downloads` by default (falls back to plain text/transcript if markdown is not available):

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" text 830eb7dfaaac428a87fb2dae2e80a2a5
```

You can also reference sources by URL or basename/title:

```bash
uv run insights --app-dir "$INSIGHTS_APP_DIR" text "https://www.youtube.com/watch?v=VIDEO_ID"
uv run insights --app-dir "$INSIGHTS_APP_DIR" text "https://example.com/article"
uv run insights --app-dir "$INSIGHTS_APP_DIR" text "onepager.pdf"
```

Options:
- `--out-dir PATH` (default `~/Downloads`)
- `--out-file PATH` (write a single deterministic `.md` path; incompatible with `--include-plain`)
- `--backend docling|firecrawl` (when ingesting URLs)
- `--refresh` (force re-ingest)
- `--name NAME` (override the output filename base)
- `--include-plain` (also write a `.txt` file)
- `--no-markdown` (disable markdown export; combine with `--include-plain` if you still want output)

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


