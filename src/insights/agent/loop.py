from __future__ import annotations

import json
from typing import Any

from insights.agent.anthropic_tool_use import AnthropicToolUseClient
from insights.agent.tools import TOOL_SCHEMAS, ToolContext, ToolRunner
from insights.config import Paths


SYSTEM_PROMPT = """\
You are an assistant for the Insights app, a tool for ingesting documents, web pages, and YouTube videos,
then running Q&A and chat over them.

You can call tools to:
- Ingest new sources
- List/search sources
- List/search conversations
- Ask questions about a source
- Start an interactive chat session (terminal)

Safety / side effects:
- If a tool returns {"blocked": true, "reason": "safe_mode", ...}, you MUST stop and tell the user
  the proposed command to run, plus how to re-run the agent with --yes.
- If a tool returns {"ambiguous": true, ...}, you MUST ask a clarifying question and show the suggestions.

When handling requests:
1) Think step-by-step about what tools you need
2) Resolve sources/conversations before using them
3) Be concise and operational (give exact commands/ids)
4) For ambiguous requests, ask a clarifying question rather than guessing

Natural-language patterns (examples):
- "chat on <source_ref>": resolve the source, then call start_chat(source_ref=...). Do NOT require existing conversations.
- "resume last conversation on <source_ref>": resolve the source, list_conversations(source_ref=...), pick latest, then start_chat(conversation_id=...).
- "ask <source_ref> <question>": resolve the source and call ask_source; after it runs, show the returned conversation_id and a resume command.
- "ingest <url_or_path>": call ingest_source (blocked in safe_mode).

If a task requires resuming a chat, provide the conversation ID and the exact command:
  uv run insights --app-dir "<APP_DIR>" chat --conversation <id>
"""


class InsightsAgent:
    """
    Agent that uses Anthropic's native tool-use (Messages API tools) to handle complex queries.
    """

    def __init__(
        self,
        *,
        paths: Paths,
        model: str = "claude-sonnet-4-20250514",
        max_steps: int = 10,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        verbose: bool = False,
        allow_side_effects: bool = False,
    ) -> None:
        self._paths = paths
        self._model = model
        self._max_steps = max_steps
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._verbose = verbose

        self._client = AnthropicToolUseClient()
        self._tools = ToolRunner(ctx=ToolContext(paths=paths, allow_side_effects=allow_side_effects))

    def _log(self, message: str) -> None:
        if self._verbose:
            print(f"[agent] {message}")

    def _execute_tool(self, name: str, input_args: dict[str, Any]) -> Any:
        self._log(f"Executing tool: {name}({json.dumps(input_args, ensure_ascii=False)})")
        result = self._tools.execute(name=name, input_args=input_args)
        self._log(f"Tool result: {json.dumps(result, ensure_ascii=False)[:400]}...")
        return result

    def run(self, user_query: str) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_query}]

        for step in range(self._max_steps):
            self._log(f"Step {step + 1}/{self._max_steps}")

            resp = self._client.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=SYSTEM_PROMPT.replace("<APP_DIR>", str(self._paths.app_dir)),
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            self._log(f"Stop reason: {resp.stop_reason}")

            if resp.stop_reason == "end_turn":
                texts = [b.get("text", "") for b in resp.content if b.get("type") == "text"]
                return "\n".join(t for t in texts if isinstance(t, str) and t).strip()

            if resp.stop_reason == "tool_use":
                # Add assistant tool_use blocks.
                messages.append({"role": "assistant", "content": resp.content})

                tool_results: list[dict[str, Any]] = []
                for block in resp.content:
                    if block.get("type") != "tool_use":
                        continue
                    name = block.get("name")
                    tool_id = block.get("id")
                    tool_input = block.get("input")
                    if not isinstance(name, str) or not isinstance(tool_id, str) or not isinstance(tool_input, dict):
                        continue
                    # Interactive chat is a terminal action: start it and exit cleanly after it ends.
                    if name == "start_chat":
                        self._execute_tool(name, tool_input)
                        return ""
                    result = self._execute_tool(name, tool_input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

                messages.append({"role": "user", "content": tool_results})
                continue

            # Any other stop reason: return what we have (best-effort).
            texts = [b.get("text", "") for b in resp.content if b.get("type") == "text"]
            joined = "\n".join(t for t in texts if isinstance(t, str) and t).strip()
            if joined:
                return joined
            return "I wasn't able to complete this task. Try rephrasing or splitting it into smaller steps."

        return "I wasn't able to complete this task within the step limit. Please try a simpler query or break it into parts."


def run_agent(
    *,
    query: str,
    paths: Paths,
    model: str = "claude-sonnet-4-20250514",
    max_steps: int = 10,
    verbose: bool = False,
    allow_side_effects: bool = False,
) -> str:
    agent = InsightsAgent(
        paths=paths,
        model=model,
        max_steps=max_steps,
        verbose=verbose,
        allow_side_effects=allow_side_effects,
    )
    return agent.run(query)


