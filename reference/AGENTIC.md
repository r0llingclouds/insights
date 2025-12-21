# Insights Agent

Natural language interface for the Insights app using Anthropic's native tool use.

## Setup

```bash
# From your insights project root, copy the agent folder
cp -r /path/to/insights-agent/agent ./agent
cp /path/to/insights-agent/describe.py ./insights/describe.py

# Or add as a subdirectory and install dependencies
uv add anthropic
```

### Database Migration

Before using the agent, add the description column to your database:

```bash
# Run the migration
python migrations/001_add_description.py

# Backfill descriptions for existing sources
python migrations/002_backfill_descriptions.py

# Or dry-run first to see what will happen
python migrations/002_backfill_descriptions.py --dry-run
```

### Integrate Description Generation

Add description generation to your ingest workflow. See `examples/integrate_describe.py` for patterns, but the key is:

```python
from insights.describe import generate_description

# After extracting content during ingest:
description = generate_description(content)

# Save to database:
db.execute("UPDATE sources SET description = ? WHERE id = ?", (description, source_id))
```

## Usage

### As a standalone module

```bash
# Set your API key
export ANTHROPIC_API_KEY="..."

# Set app directory (optional, defaults to ~/.insights)
export INSIGHTS_APP_DIR="/tmp/insights-test"

# Single query
uv run python -m agent "ask https://youtube.com/watch?v=xyz what is this about?"

# Interactive mode
uv run python -m agent --interactive

# Verbose mode (shows tool calls)
uv run python -m agent -v "check if I have a youtube video about coding"
```

### Integrate into your existing CLI

Add a new command to your `insights` CLI:

```python
# In your main CLI file (e.g., insights/cli.py)
import click

@click.command()
@click.argument("query")
@click.option("-v", "--verbose", is_flag=True, help="Show debug info")
@click.option("--model", default="claude-sonnet-4-20250514", help="Model to use")
def do(query: str, verbose: bool, model: str):
    """Run a natural language query through the agent."""
    from agent import run_agent
    
    response = run_agent(query, model=model, verbose=verbose)
    click.echo(response)

# Register the command
cli.add_command(do)
```

Then use it:

```bash
uv run insights do "resume conversation about best IDE for https://youtube.com/..."
```

## Example Queries

### Easy (single tool)
```
"ask https://www.youtube.com/watch?v=zPMPqzjM0Fw what is this about?"
"ingest https://www.youtube.com/watch?v=zPMPqzjM0Fw"
"list all my youtube sources"
```

### Medium (2-3 tools)
```
"show conversations for https://www.youtube.com/watch?v=zPMPqzjM0Fw"
"check if I have this source https://www.youtube.com/watch?v=zPMPqzjM0Fw"
```

### Hard (multi-step reasoning)
```
"resume conversation about best IDE for https://www.youtube.com/watch?v=zPMPqzjM0Fw"
"resume last conversation on https://www.youtube.com/watch?v=zPMPqzjM0Fw"
"check if I have a youtube video about a user that can't stand using IDEs anymore"
"check if I have a pdf doc about a project of agentic AI with ViveBioTech"
```

## Architecture

```
agent/
├── __init__.py      # Package exports
├── __main__.py      # CLI entry point
├── loop.py          # Agent loop (the core)
└── tools.py         # Tool definitions + implementations
```

### How it works

1. User query comes in
2. Agent sends query + tool definitions to Claude
3. Claude decides which tool(s) to call
4. Agent executes tools, returns results to Claude
5. Claude either calls more tools or returns final answer
6. Loop continues until `end_turn` or max steps

### Tools available

| Tool | Description |
|------|-------------|
| `ingest_source` | Ingest file/URL/YouTube into the database |
| `list_sources` | List ingested sources, filter by type |
| `check_source_exists` | Check if a source exists (partial match on URL/title/description) |
| `semantic_search_sources` | Find sources by topic using natural language (uses descriptions) |
| `list_conversations` | List conversations for a source |
| `search_conversations` | Search conversations by topic/content |
| `ask_source` | One-off Q&A on a source |
| `get_conversation_info` | Get details about a conversation |

## How Semantic Search Works

Instead of using a vector database, we generate a one-liner description for each source during ingest:

```
Source: youtube video "zPMPqzjM0Fw"
Title: "Why I stopped using IDEs"
Description: "Developer explains mass exodus from traditional IDEs to Vim/terminal workflows"
```

When you ask "find a video about someone who hates IDEs", the agent:

1. Calls `semantic_search_sources(query="video about someone who hates IDEs", kind="youtube")`
2. Gets back all YouTube sources with their titles and descriptions
3. Claude matches "hates IDEs" → "stopped using IDEs" / "exodus from traditional IDEs"
4. Returns the matching source

This gives you ~80% of vector search benefits with zero infrastructure.

## Extending

### Add a new tool

1. Add the implementation function in `tools.py`:

```python
def my_new_tool(arg1: str, arg2: int = 10) -> dict:
    """Do something useful."""
    # Your logic here
    return {"success": True, "result": "..."}
```

2. Add the schema to `TOOL_SCHEMAS`:

```python
{
    "name": "my_new_tool",
    "description": "Does something useful",
    "input_schema": {
        "type": "object",
        "properties": {
            "arg1": {"type": "string", "description": "First argument"},
            "arg2": {"type": "integer", "description": "Second argument"},
        },
        "required": ["arg1"]
    }
}
```

3. Register it in `TOOL_FUNCTIONS`:

```python
TOOL_FUNCTIONS = {
    # ...existing tools...
    "my_new_tool": my_new_tool,
}
```

### Customize the system prompt

Edit `SYSTEM_PROMPT` in `loop.py` to change the agent's behavior.

## Debugging

Use `-v` / `--verbose` to see what the agent is doing:

```bash
uv run python -m agent -v "find my conversations about quantum computing"
```

Output:
```
[agent] Step 1/10
[agent] Executing tool: search_conversations({"query": "quantum computing"})
[agent] Tool result: {"success": true, "matches": [...]}...
[agent] Stop reason: end_turn

I found 3 conversations about quantum computing:
1. abc123 - "Quantum basics discussion" (updated 2024-01-15)
...
```

## Notes

- The agent calls your existing CLI commands via subprocess, so it works alongside your current setup
- Tool results are returned as JSON for Claude to interpret
- Max steps defaults to 10 to prevent infinite loops
- Uses `claude-sonnet-4-20250514` by default (good balance of speed/capability)
